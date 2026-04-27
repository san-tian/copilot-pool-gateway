package instance

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"copilot-go/config"
	"copilot-go/store"
)

type ProxyInstance struct {
	Account  store.Account
	State    *config.State
	Status   string // "running", "stopped", "error"
	Error    string
	stopChan chan struct{}
}

type CopilotUser struct {
	Login     string `json:"login"`
	AvatarURL string `json:"avatar_url"`
}

var (
	githubCopilotURL = config.GithubCopilotURL
	copilotBaseURL   = config.CopilotBaseURL
)

func StopInstance(accountID string) {
	mu.Lock()
	inst, ok := instances[accountID]
	if !ok {
		mu.Unlock()
		return
	}
	if inst.Status == "running" && inst.stopChan != nil {
		close(inst.stopChan)
		inst.stopChan = nil
	}
	inst.Status = "stopped"
	mu.Unlock()
	log.Printf("Instance stopped for account: %s", inst.Account.Name)
}

func GetInstanceStatus(accountID string) string {
	mu.RLock()
	defer mu.RUnlock()
	if inst, ok := instances[accountID]; ok {
		return inst.Status
	}
	return "stopped"
}

func GetInstanceError(accountID string) string {
	mu.RLock()
	defer mu.RUnlock()
	if inst, ok := instances[accountID]; ok {
		return inst.Error
	}
	return ""
}

func GetInstanceState(accountID string) *config.State {
	mu.RLock()
	defer mu.RUnlock()
	if inst, ok := instances[accountID]; ok {
		return inst.State
	}
	return nil
}

// ForceRefreshToken forces a Copilot token refresh for the given account. Intended for
// request-path callers that just observed a 401 from upstream, letting the next retry on
// the same account use a freshly-minted token instead of the one that may have just expired.
func ForceRefreshToken(accountID string) error {
	state := GetInstanceState(accountID)
	if state == nil {
		return fmt.Errorf("instance not found for account %s", accountID)
	}
	return refreshCopilotToken(state)
}

// GetAllCachedModels collects and deduplicates model entries from all running instances.
func GetAllCachedModels() []config.ModelEntry {
	mu.RLock()
	defer mu.RUnlock()

	seen := make(map[string]bool)
	var result []config.ModelEntry
	for _, inst := range instances {
		if inst.Status != "running" {
			continue
		}
		inst.State.RLock()
		models := inst.State.Models
		inst.State.RUnlock()
		if models == nil {
			continue
		}
		for _, m := range models.Data {
			if !seen[m.ID] {
				seen[m.ID] = true
				result = append(result, m)
			}
		}
	}
	return result
}

func GetUser(accountID string) (*CopilotUser, error) {
	state := GetInstanceState(accountID)
	if state == nil {
		return nil, fmt.Errorf("instance not found")
	}

	req, err := http.NewRequest("GET", config.GithubUserURL, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range config.GithubHeaders(state) {
		req.Header[k] = v
	}

	resp, err := getDefaultClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var user CopilotUser
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, err
	}
	return &user, nil
}

func tokenRefreshLoop(inst *ProxyInstance) {
	const fallbackInterval = 25 * time.Minute
	const minInterval = 30 * time.Second

	for {
		// Calculate next refresh time based on token expiry.
		sleepDur := fallbackInterval
		inst.State.RLock()
		expiresAt := inst.State.TokenExpiresAt
		inst.State.RUnlock()

		if expiresAt > 0 {
			remaining := time.Until(time.Unix(expiresAt, 0))
			if remaining > 0 {
				// Refresh at 80% of token lifetime (i.e., when 20% remains).
				sleepDur = time.Duration(float64(remaining) * 0.8)
				if sleepDur < minInterval {
					sleepDur = minInterval
				}
			} else {
				// Token already expired, refresh immediately.
				sleepDur = 0
			}
		}

		if sleepDur > 0 {
			timer := time.NewTimer(sleepDur)
			select {
			case <-inst.stopChan:
				timer.Stop()
				return
			case <-timer.C:
			}
		}

		// Check stop again before doing work.
		select {
		case <-inst.stopChan:
			return
		default:
		}

		if err := refreshCopilotTokenWithRetry(inst.State, 3); err != nil {
			log.Printf("Token refresh failed for %s: %v", inst.Account.Name, err)
			mu.Lock()
			inst.Status = "error"
			inst.Error = err.Error()
			mu.Unlock()
			continue
		}

	}
}

func refreshCopilotToken(state *config.State) error {
	req, err := http.NewRequest("GET", githubCopilotURL, nil)
	if err != nil {
		return err
	}
	for k, v := range config.GithubHeaders(state) {
		req.Header[k] = v
	}

	resp, err := getDefaultClient().Do(req)
	if err != nil {
		return fmt.Errorf("failed to get copilot token: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("copilot token request failed (status %d): %s", resp.StatusCode, string(body))
	}

	var tokenResp config.CopilotTokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return fmt.Errorf("failed to decode copilot token: %w", err)
	}

	state.Lock()
	state.CopilotToken = tokenResp.Token
	state.TokenExpiresAt = tokenResp.ExpiresAt
	state.Unlock()

	if tokenResp.ExpiresAt > 0 {
		expiresIn := time.Until(time.Unix(tokenResp.ExpiresAt, 0))
		log.Printf("Copilot token refreshed, expires in %v", expiresIn.Round(time.Second))
	}
	return nil
}

// refreshCopilotTokenWithRetry attempts token refresh with exponential backoff.
// Retries up to maxRetries times on failure before giving up.
func refreshCopilotTokenWithRetry(state *config.State, maxRetries int) error {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 2s, 8s, 18s (2 * attempt^2)
			backoff := time.Duration(2*math.Pow(float64(attempt), 2)) * time.Second
			log.Printf("Token refresh retry %d/%d after %v", attempt, maxRetries, backoff)
			time.Sleep(backoff)
		}
		lastErr = refreshCopilotToken(state)
		if lastErr == nil {
			return nil
		}
		log.Printf("Token refresh attempt %d failed: %v", attempt+1, lastErr)
	}
	return fmt.Errorf("token refresh failed after %d attempts: %w", maxRetries+1, lastErr)
}

func fetchModels(state *config.State) error {
	state.RLock()
	baseURL := copilotBaseURL(state.AccountType)
	state.RUnlock()

	req, err := http.NewRequest("GET", baseURL+"/models", nil)
	if err != nil {
		return err
	}
	for k, v := range config.CopilotHeaders(state, false) {
		req.Header[k] = v
	}

	resp, err := getDefaultClient().Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("models request failed (status %d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var models config.ModelsResponse
	if err := json.Unmarshal(body, &models); err != nil {
		// Try parsing as array
		var modelList []config.ModelEntry
		if err2 := json.Unmarshal(body, &modelList); err2 != nil {
			return fmt.Errorf("failed to parse models response: %s", strings.TrimSpace(string(body)))
		}
		models = config.ModelsResponse{
			Object: "list",
			Data:   modelList,
		}
	}

	if models.Object == "" {
		models.Object = "list"
	}

	state.Lock()
	state.Models = &models
	state.Unlock()
	return nil
}

var (
	clientMu            sync.RWMutex
	streamingHTTPClient *http.Client
	defaultHTTPClient   *http.Client
)

func init() {
	rebuildHTTPClients()
}

func buildTransport(streaming bool, proxyRawURL string) *http.Transport {
	t := &http.Transport{
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	if streaming {
		t.MaxIdleConns = 100
		t.MaxIdleConnsPerHost = 20
		t.ResponseHeaderTimeout = 2 * time.Minute
	} else {
		t.MaxIdleConns = 50
		t.MaxIdleConnsPerHost = 10
	}
	if proxyRawURL != "" {
		if parsed, err := url.Parse(proxyRawURL); err == nil {
			t.Proxy = http.ProxyURL(parsed)
		}
	}
	return t
}

func rebuildHTTPClients() {
	pURL := config.GetProxyURL()
	streaming := &http.Client{
		// No Timeout set — streaming responses can last indefinitely.
		Transport: buildTransport(true, pURL),
	}
	nonStreaming := &http.Client{
		Timeout:   15 * time.Second,
		Transport: buildTransport(false, pURL),
	}
	clientMu.Lock()
	streamingHTTPClient = streaming
	defaultHTTPClient = nonStreaming
	clientMu.Unlock()
}

// RebuildHTTPClients rebuilds the shared HTTP clients using the current proxy setting.
func RebuildHTTPClients() {
	rebuildHTTPClients()
}

func getStreamingClient() *http.Client {
	clientMu.RLock()
	c := streamingHTTPClient
	clientMu.RUnlock()
	return c
}

func getDefaultClient() *http.Client {
	clientMu.RLock()
	c := defaultHTTPClient
	clientMu.RUnlock()
	return c
}

// GetDefaultHTTPClient returns the current default HTTP client for use by other packages.
func GetDefaultHTTPClient() *http.Client {
	return getDefaultClient()
}

func ProxyRequestWithBytes(state *config.State, method, path string, bodyBytes []byte, extraHeaders http.Header, hasVision bool) (*http.Response, error) {
	return ProxyRequestWithBytesCtx(context.Background(), state, method, path, bodyBytes, extraHeaders, hasVision)
}

var (
	workerClientOnce sync.Once
	workerHTTPClient *http.Client
)

// getWorkerClient returns a shared HTTP client for per-account sidecar workers on loopback.
// Connect timeout is 2s — if the worker unit is down or the port is wrong, fail fast so the
// retry loop can hop to the next account. No overall request timeout: SSE streams last as
// long as the upstream stream does.
func getWorkerClient() *http.Client {
	workerClientOnce.Do(func() {
		t := &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   2 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   20,
			ResponseHeaderTimeout: 2 * time.Minute,
		}
		workerHTTPClient = &http.Client{Transport: t}
	})
	return workerHTTPClient
}

// copilotHeaderBlocklist lists request headers that must NOT be forwarded to a worker.
// Transport/auth headers always stay local to the worker hop. Copilot turn headers
// such as X-Interaction-* and X-Initiator are intentionally allowed so the gateway
// can preserve already-resolved continuation context across the worker boundary.
// Keys are lowercased.
var copilotHeaderBlocklist = map[string]struct{}{
	"authorization":                       {},
	"editor-version":                      {},
	"editor-plugin-version":               {},
	"openai-intent":                       {},
	"x-request-id":                        {},
	"x-github-api-version":                {},
	"x-vscode-user-agent-library-version": {},
	"host":                                {},
	"content-length":                      {},
	"connection":                          {},
	"transfer-encoding":                   {},
}

// copyNonCopilotHeaders forwards client headers to dst, stripping anything Copilot-specific
// (the worker re-applies its own). Catches both the explicit blocklist and any Copilot-*
// prefix.
func copyNonCopilotHeaders(dst, src http.Header) {
	for k, vs := range src {
		lk := strings.ToLower(k)
		if _, blocked := copilotHeaderBlocklist[lk]; blocked {
			continue
		}
		if strings.HasPrefix(lk, "copilot-") {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

// ProxyRequestViaWorker forwards a request to a per-account sidecar worker (caozhiyuan/copilot-api)
// bound to loopback. The worker owns payload translation (compact, tool rewrites, stream-id
// sync, web-search filter). When the gateway has already resolved Copilot turn context for a
// request, it can also forward that context via X-Interaction-* / X-Initiator headers.
func ProxyRequestViaWorker(ctx context.Context, workerURL, method, path string, bodyBytes []byte, clientHeaders http.Header, traceID string) (*http.Response, error) {
	target := strings.TrimRight(workerURL, "/") + path
	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	copyNonCopilotHeaders(req.Header, clientHeaders)
	if strings.TrimSpace(traceID) != "" {
		req.Header.Set("X-Trace-Id", traceID)
	}
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if req.Header.Get("Accept") == "" {
		req.Header.Set("Accept", "text/event-stream, application/json")
	}
	start := time.Now()
	log.Printf("[worker-hop trace=%s] dispatch method=%s path=%s target=%s body_bytes=%d", traceID, method, path, target, len(bodyBytes))
	resp, err := getWorkerClient().Do(req)
	statusCode := 0
	if resp != nil {
		statusCode = resp.StatusCode
	}
	log.Printf("[worker-hop trace=%s] complete method=%s path=%s target=%s status=%d elapsed_ms=%d err=%v",
		traceID, method, path, target, statusCode, time.Since(start).Milliseconds(), err)
	return resp, err
}

func ProxyRequestWithBytesCtx(ctx context.Context, state *config.State, method, path string, bodyBytes []byte, extraHeaders http.Header, hasVision bool) (*http.Response, error) {
	state.RLock()
	baseURL := copilotBaseURL(state.AccountType)
	state.RUnlock()

	url := baseURL + path

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}

	for k, v := range config.CopilotHeaders(state, hasVision) {
		req.Header[k] = v
	}
	for k, v := range extraHeaders {
		req.Header[k] = v
	}

	return getStreamingClient().Do(req)
}
