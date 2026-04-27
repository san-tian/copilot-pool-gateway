package config

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	CopilotVersion   = "0.26.7"
	GithubClientID   = "Iv1.b507a08c87ecfe98"
	GithubAPIVersion = "2025-04-01"

	CopilotIndividualChatURL = "https://api.githubcopilot.com"

	GithubCopilotURL = "https://api.github.com/copilot_internal/v2/token"
	GithubDeviceURL  = "https://github.com/login/device/code"
	GithubTokenURL   = "https://github.com/login/oauth/access_token"
	GithubUserURL    = "https://api.github.com/user"
)

// proxyURL is the global outbound HTTP proxy. Protected by proxyMu.
var (
	proxyMu  sync.RWMutex
	proxyURL string
)

func SetProxyURL(u string) {
	proxyMu.Lock()
	proxyURL = u
	proxyMu.Unlock()
}

func GetProxyURL() string {
	proxyMu.RLock()
	u := proxyURL
	proxyMu.RUnlock()
	return u
}

// NewHTTPClient creates an HTTP client with the current proxy setting and given timeout.
func NewHTTPClient(timeout time.Duration) *http.Client {
	t := &http.Transport{
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	if pURL := GetProxyURL(); pURL != "" {
		if parsed, err := url.Parse(pURL); err == nil {
			t.Proxy = http.ProxyURL(parsed)
		}
	}
	return &http.Client{Timeout: timeout, Transport: t}
}

type ModelsResponse struct {
	Object string       `json:"object"`
	Data   []ModelEntry `json:"data"`
}

type ModelCapabilities struct {
	Family string      `json:"family,omitempty"`
	Limits ModelLimits `json:"limits,omitempty"`
	Type   string      `json:"type,omitempty"`
}

type ModelLimits struct {
	MaxOutputTokens  int `json:"max_output_tokens,omitempty"`
	MaxPromptTokens  int `json:"max_prompt_tokens,omitempty"`
	MaxContextWindow int `json:"max_context_window,omitempty"`
}

type ModelEntry struct {
	ID           string             `json:"id"`
	Object       string             `json:"object"`
	Created      int64              `json:"created"`
	OwnedBy      string             `json:"owned_by"`
	Name         string             `json:"name,omitempty"`
	Version      string             `json:"version,omitempty"`
	Vendor       string             `json:"vendor,omitempty"`
	Capabilities *ModelCapabilities `json:"capabilities,omitempty"`
}

type CopilotTokenResponse struct {
	Token     string `json:"token"`
	ExpiresAt int64  `json:"expires_at"`
}

type State struct {
	mu             sync.RWMutex
	GithubToken    string
	CopilotToken   string
	TokenExpiresAt int64 // Unix timestamp when the Copilot token expires
	AccountType    string
	Models         *ModelsResponse
	VSCodeVersion  string
}

func NewState() *State {
	return &State{}
}

func (s *State) Lock()    { s.mu.Lock() }
func (s *State) Unlock()  { s.mu.Unlock() }
func (s *State) RLock()   { s.mu.RLock() }
func (s *State) RUnlock() { s.mu.RUnlock() }

// WorkerPoolMode returns the per-account worker routing mode for /v1/responses.
// "auto"  (default) — route to account.WorkerURL when set, otherwise direct.
// "off"           — force direct mode globally regardless of account.WorkerURL.
func WorkerPoolMode() string {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("USE_WORKER_POOL")))
	if v == "off" || v == "disabled" || v == "0" || v == "false" {
		return "off"
	}
	return "auto"
}

// WorkerAutoAdopt controls whether the WorkerSupervisor auto-spawns a worker
// child for an account when device-flow completes, and auto-cleans it on
// account delete. Default on. Set to off/disabled/0/false to keep the classic
// behavior (admin must fill WorkerURL manually / external worker lifecycle).
func WorkerAutoAdopt() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("COPILOT_WORKER_AUTO_ADOPT")))
	if v == "off" || v == "disabled" || v == "0" || v == "false" {
		return false
	}
	return true
}

// OrphanPassthrough controls what happens when a /v1/responses request arrives
// with function_call_output items whose call_ids the router's sticky cache
// has never seen (the cross-relay migration scenario).
//
// "auto" (default) — when a worker-enabled account is available, skip the
//   410 session_expired check, clear sticky state, and pool-route the request
//   as a fresh session. Worker translates to stateless chat/completions so
//   tool_call_ids don't have to match any server-side session. Only kicks in
//   for fc_id orphans; previous_response_id-only orphans (truncated input
//   depending on server-side history) still 410 because input is unrecoverable.
// "off" — preserve historical behavior: 410 on any orphan, unless the client
//   set X-Copilot-Continuation-Degrade: orphan.
func OrphanPassthrough() string {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("ORPHAN_PASSTHROUGH")))
	if v == "off" || v == "disabled" || v == "0" || v == "false" {
		return "off"
	}
	return "auto"
}

// ResponsesOrphanTranslate controls whether orphan /v1/responses requests are
// translated in-process to chat/completions and routed to the worker's
// /v1/chat/completions endpoint, with the chat SSE stream wrapped back into
// Responses SSE events on the way out. This keeps the stateful /v1/responses
// upstream out of the cross-relay migration path entirely — Copilot's
// chat/completions endpoint does not session-validate tool_call_ids, so
// orphan fc_ids pass through cleanly.
//
// "on" — translate orphan /v1/responses → chat/completions in Go before
//        the worker call; wrap the chat SSE reply back into Responses SSE.
//        Only effective when the selected account has a WorkerURL.
// "off" (default) — orphan passthrough forwards the raw Responses payload
//                   to the worker's /v1/responses, which proxies to Copilot
//                   and can 401 on unknown fc_ids.
func ResponsesOrphanTranslate() string {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("RESPONSES_ORPHAN_TRANSLATE")))
	if v == "on" || v == "enabled" || v == "1" || v == "true" {
		return "on"
	}
	return "off"
}

func CopilotBaseURL(accountType string) string {
	if accountType == "" || accountType == "individual" {
		return CopilotIndividualChatURL
	}
	return fmt.Sprintf("https://api.%s.githubcopilot.com", accountType)
}

func CopilotHeaders(state *State, vision bool) http.Header {
	state.RLock()
	defer state.RUnlock()

	h := make(http.Header)
	h.Set("Authorization", "Bearer "+state.CopilotToken)
	h.Set("Content-Type", "application/json")
	h.Set("Accept", "application/json")
	h.Set("Copilot-Integration-Id", "vscode-chat")
	h.Set("Editor-Version", "vscode/"+state.VSCodeVersion)
	h.Set("Editor-Plugin-Version", "copilot-chat/"+CopilotVersion)
	h.Set("User-Agent", fmt.Sprintf("GitHubCopilotChat/%s", CopilotVersion))
	h.Set("Openai-Intent", "conversation-panel")
	h.Set("X-GitHub-API-Version", GithubAPIVersion)
	h.Set("X-Request-Id", uuid.NewString())
	h.Set("X-Vscode-User-Agent-Library-Version", "electron-fetch")
	if vision {
		h.Set("Copilot-Vision-Request", "true")
	}
	return h
}

func GithubHeaders(state *State) http.Header {
	state.RLock()
	defer state.RUnlock()

	h := make(http.Header)
	h.Set("Authorization", "token "+state.GithubToken)
	h.Set("Content-Type", "application/json")
	h.Set("Accept", "application/json")
	h.Set("Editor-Version", "vscode/"+state.VSCodeVersion)
	h.Set("Editor-Plugin-Version", "copilot-chat/"+CopilotVersion)
	h.Set("User-Agent", fmt.Sprintf("GitHubCopilotChat/%s", CopilotVersion))
	h.Set("X-GitHub-API-Version", GithubAPIVersion)
	h.Set("X-Vscode-User-Agent-Library-Version", "electron-fetch")
	return h
}
