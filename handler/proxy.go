package handler

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"copilot-go/anthropic"
	"copilot-go/config"
	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

func newRequestID() string {
	var b [6]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "000000000000"
	}
	return hex.EncodeToString(b[:])
}

func truncateForLog(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// peekAndRestoreBody reads resp.Body in full for logging, then rewinds it via
// NopCloser so downstream code can still read it. Safe to call when the caller
// will close or forward the body next. Returns "" for nil / streaming / read-err.
func peekAndRestoreBody(resp *http.Response, maxBytes int) string {
	if resp == nil || resp.Body == nil {
		return ""
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		return ""
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))
	return truncateForLog(string(body), maxBytes)
}

// degradedBodyTopLevelKeys parses a JSON body and returns a sorted comma-joined
// list of its top-level keys. Used to quickly see which fields survive the
// orphan-continuation degrade pass — we want to know whether fields like
// "conversation", "previous_response_id", or "store" are leaking through.
func degradedBodyTopLevelKeys(body []byte) string {
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "<parse-err>"
	}
	keys := make([]string, 0, len(payload))
	for k := range payload {
		keys = append(keys, k)
	}
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if keys[j] < keys[i] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	return strings.Join(keys, ",")
}

// degradedInputShape summarizes the degraded input[] array: counts per item
// type, and how many items carry an "encrypted_content" field or a role
// message.content[].type. Used to pinpoint which surviving item structure is
// tripping upstream's "input item does not belong to this connection" check.
func degradedInputShape(body []byte) string {
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "<parse-err>"
	}
	input, ok := payload["input"].([]interface{})
	if !ok {
		return "<no-input>"
	}
	typeCounts := map[string]int{}
	encryptedContent := 0
	hasID := 0
	hasCallID := 0
	for _, raw := range input {
		m, ok := raw.(map[string]interface{})
		if !ok {
			typeCounts["<non-object>"]++
			continue
		}
		itemType, _ := m["type"].(string)
		typeCounts[itemType]++
		if _, ok := m["encrypted_content"]; ok {
			encryptedContent++
		}
		if id, _ := m["id"].(string); id != "" {
			hasID++
		}
		if callID, _ := m["call_id"].(string); callID != "" {
			hasCallID++
		}
	}
	parts := make([]string, 0, len(typeCounts)+3)
	typeKeys := make([]string, 0, len(typeCounts))
	for k := range typeCounts {
		typeKeys = append(typeKeys, k)
	}
	for i := 0; i < len(typeKeys); i++ {
		for j := i + 1; j < len(typeKeys); j++ {
			if typeKeys[j] < typeKeys[i] {
				typeKeys[i], typeKeys[j] = typeKeys[j], typeKeys[i]
			}
		}
	}
	for _, k := range typeKeys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, typeCounts[k]))
	}
	parts = append(parts, fmt.Sprintf("encrypted_content=%d", encryptedContent))
	parts = append(parts, fmt.Sprintf("id=%d", hasID))
	parts = append(parts, fmt.Sprintf("call_id=%d", hasCallID))
	return strings.Join(parts, " ")
}

// RegisterProxy sets up the proxy server routes.
func RegisterProxy(r *gin.Engine) {
	// Initialize rate limiter from environment.
	instance.InitRateLimiter()

	// Load per-account rate limit from pool config.
	if poolCfg, err := store.GetPoolConfig(); err == nil && poolCfg != nil {
		instance.SetPerAccountRPM(poolCfg.RateLimitRPM)
	}

	r.Use(proxyAuth())

	// OpenAI compatible endpoints
	r.POST("/chat/completions", proxyCompletions)
	r.POST("/v1/chat/completions", proxyCompletions)
	r.GET("/models", proxyModels)
	r.GET("/v1/models", proxyModels)
	r.POST("/embeddings", proxyEmbeddings)
	r.POST("/v1/embeddings", proxyEmbeddings)

	// Anthropic compatible endpoints
	r.POST("/v1/messages", proxyMessages)
	r.POST("/v1/messages/count_tokens", proxyCountTokens)

	// OpenAI Responses API endpoint
	r.POST("/responses/compact", proxyResponsesCompact)
	r.POST("/v1/responses/compact", proxyResponsesCompact)
	r.POST("/v1/responses", proxyResponses)
}

func proxyAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			// Also check x-api-key for Anthropic-style auth
			apiKey := c.GetHeader("x-api-key")
			if apiKey != "" {
				authHeader = "Bearer " + apiKey
			}
		}

		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "missing authorization"})
			return
		}

		token := strings.TrimPrefix(authHeader, "Bearer ")

		// Check pool API key first
		poolCfg, _ := store.GetPoolConfig()
		if poolCfg != nil && poolCfg.Enabled && poolCfg.ApiKey == token {
			c.Set("isPool", true)
			c.Set("poolStrategy", poolCfg.Strategy)
			c.Next()
			return
		}

		// Check individual account API key
		account, err := store.GetAccountByApiKey(token)
		if err != nil || account == nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "invalid API key"})
			return
		}

		c.Set("accountID", account.ID)
		c.Set("isPool", false)
		c.Next()
	}
}

// resolvedAccount holds the resolved state and account ID.
type resolvedAccount struct {
	State     *config.State
	AccountID string
}

func extractRequestedModel(bodyBytes []byte) string {
	if len(bodyBytes) == 0 {
		return ""
	}
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return ""
	}
	return strings.TrimSpace(payload.Model)
}

func extractPreviousResponseID(bodyBytes []byte) string {
	if len(bodyBytes) == 0 {
		return ""
	}
	var payload struct {
		PreviousResponseID string `json:"previous_response_id"`
	}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return ""
	}
	return strings.TrimSpace(payload.PreviousResponseID)
}

func extractMessageToolResultIDs(bodyBytes []byte) []string {
	if len(bodyBytes) == 0 {
		return nil
	}
	var payload anthropic.AnthropicMessagesPayload
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return nil
	}
	for idx := len(payload.Messages) - 1; idx >= 0; idx-- {
		message := payload.Messages[idx]
		if message.Role != "user" {
			continue
		}
		blocks := anthropic.ParseContentBlocksPublic(message.Content)
		ids := make([]string, 0)
		for _, block := range blocks {
			if block.Type != "tool_result" {
				continue
			}
			if toolUseID := strings.TrimSpace(block.ToolUseID); toolUseID != "" {
				ids = append(ids, toolUseID)
			}
		}
		return uniqueTrimmedStrings(ids)
	}
	return nil
}

func extractResponseFunctionCallOutputIDs(bodyBytes []byte) []string {
	if len(bodyBytes) == 0 {
		return nil
	}
	var payload struct {
		Input interface{} `json:"input"`
	}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return nil
	}

	collect := func(item interface{}, ids *[]string) {
		entry, ok := item.(map[string]interface{})
		if !ok {
			return
		}
		itemType, _ := entry["type"].(string)
		if strings.TrimSpace(strings.ToLower(itemType)) != "function_call_output" {
			return
		}
		if callID, _ := entry["call_id"].(string); strings.TrimSpace(callID) != "" {
			*ids = append(*ids, callID)
		}
	}

	ids := make([]string, 0)
	switch typed := payload.Input.(type) {
	case []interface{}:
		for _, item := range typed {
			collect(item, &ids)
		}
	case map[string]interface{}:
		collect(typed, &ids)
	}
	return uniqueTrimmedStrings(ids)
}

func uniqueTrimmedStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		result = append(result, trimmed)
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func readReplayInvalidResponse(resp *http.Response) (bool, string) {
	if resp == nil || (resp.StatusCode != http.StatusBadRequest && resp.StatusCode != http.StatusUnauthorized) {
		return false, ""
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, ""
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))
	message := strings.ToLower(string(body))
	if strings.Contains(message, "input item does not belong to this connection") ||
		strings.Contains(message, "input item id does not belong to this connection") ||
		strings.Contains(message, "no tool call found for function call output") ||
		strings.Contains(message, "previous_response_id is not supported") {
		return true, string(body)
	}
	return false, string(body)
}

func resolveStateForResponseReplay(c *gin.Context, previousResponseID string, requestedModel string) *resolvedAccount {
	accountID, ok := instance.LookupResponsesReplayAccount(previousResponseID)
	if !ok {
		accountID, ok = instance.LookupResponseTurnAccount(previousResponseID)
	}
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("previous_response_id %q was not found or has expired", previousResponseID)})
		return nil
	}
	if store.IsModelBlocked(accountID, requestedModel) {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": fmt.Sprintf("account is paused for model %s", requestedModel)})
		return nil
	}
	state := instance.GetInstanceState(accountID)
	if state == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "account instance not running"})
		return nil
	}
	return &resolvedAccount{State: state, AccountID: accountID}
}

func resolveStateForMessageReplay(toolResultIDs []string, requestedModel string) *resolvedAccount {
	accountID, ok := instance.LookupMessageToolCallAccount(toolResultIDs)
	if !ok {
		return nil
	}
	if store.IsModelBlocked(accountID, requestedModel) {
		return nil
	}
	state := instance.GetInstanceState(accountID)
	if state == nil {
		return nil
	}
	return &resolvedAccount{State: state, AccountID: accountID}
}

func resolveStateForResponseFunctionCallReplay(callIDs []string, requestedModel string) *resolvedAccount {
	accountID, ok := instance.LookupResponseFunctionCallAccount(callIDs)
	if !ok {
		return nil
	}
	if store.IsModelBlocked(accountID, requestedModel) {
		return nil
	}
	state := instance.GetInstanceState(accountID)
	if state == nil {
		return nil
	}
	return &resolvedAccount{State: state, AccountID: accountID}
}

func resolveState(c *gin.Context, exclude map[string]bool, requestedModel string) *resolvedAccount {
	isPool, _ := c.Get("isPool")
	if isPool == true {
		strategy := ""
		if s, ok := c.Get("poolStrategy"); ok {
			strategy = s.(string)
		}
		combinedExclude := map[string]bool{}
		for accountID, skipped := range exclude {
			combinedExclude[accountID] = skipped
		}
		for accountID := range store.GetBlockedAccountIDs(requestedModel) {
			combinedExclude[accountID] = true
		}
		account, err := instance.SelectAccount(strategy, combinedExclude)
		if err != nil || account == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "no available accounts in pool"})
			return nil
		}
		state := instance.GetInstanceState(account.ID)
		if state == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "selected account instance not running"})
			return nil
		}
		return &resolvedAccount{State: state, AccountID: account.ID}
	}

	accountID, exists := c.Get("accountID")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "no account context"})
		return nil
	}
	aid := accountID.(string)
	if store.IsModelBlocked(aid, requestedModel) {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": fmt.Sprintf("account is paused for model %s", requestedModel)})
		return nil
	}
	state := instance.GetInstanceState(aid)
	if state == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "account instance not running"})
		return nil
	}
	return &resolvedAccount{State: state, AccountID: aid}
}

// isRetryableStatus returns true for HTTP status codes that warrant a retry with a different account.
// 401 is included because a single account can transiently return 401 during its Copilot-token
// refresh window; rolling over to another account lets the client succeed while the failing
// account's instance manager refreshes its token in the background.
func isRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode == http.StatusUnauthorized || (statusCode >= 500 && statusCode <= 599)
}

type upstreamErrorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

func disableOnFatalUpstream(resp *http.Response, accountID string, requestedModel string) (bool, string) {
	if resp == nil || (resp.StatusCode != http.StatusBadRequest && resp.StatusCode != http.StatusPaymentRequired) {
		return false, ""
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, ""
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))

	var payload upstreamErrorEnvelope
	if err := json.Unmarshal(body, &payload); err != nil {
		return false, ""
	}

	switch payload.Error.Code {
	case "model_not_supported":
		reason := fmt.Sprintf("%s: %s", payload.Error.Code, payload.Error.Message)
		if err := store.BlockModelForAccount(accountID, requestedModel); err != nil {
			log.Printf("Failed to block model %s for account %s: %v", requestedModel, accountID, err)
		}
		return true, reason
	case "quota_exceeded":
		reason := fmt.Sprintf("%s: %s", payload.Error.Code, payload.Error.Message)
		if err := instance.DisableAccount(accountID, reason); err != nil {
			log.Printf("Failed to auto-disable account %s: %v", accountID, err)
		}
		return true, reason
	default:
		return false, ""
	}
}

// checkRateLimit checks the rate limit for the account and writes a 429 response if exceeded.
// Returns true if the request is allowed, false if rate limited.
func checkRateLimit(c *gin.Context, accountID string) bool {
	allowed, retryAfter := instance.CheckRateLimit(accountID)
	if !allowed {
		c.Header("Retry-After", fmt.Sprintf("%.0f", retryAfter))
		c.JSON(http.StatusTooManyRequests, gin.H{
			"error": gin.H{
				"message": "rate limit exceeded",
				"type":    "rate_limit_error",
			},
		})
		return false
	}
	return true
}

// proxyCompletions handles completions with pool-mode retry support.
func proxyCompletions(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	// Read body once for potential retries.
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	requestedModel := extractRequestedModel(bodyBytes)
	previousResponseID := extractPreviousResponseID(bodyBytes)
	exclude := make(map[string]bool)
	for attempt := 0; attempt < maxAttempts; attempt++ {
		var resolved *resolvedAccount
		if previousResponseID != "" {
			resolved = resolveStateForResponseReplay(c, previousResponseID, requestedModel)
		} else {
			resolved = resolveState(c, exclude, requestedModel)
		}
		if resolved == nil {
			return // resolveState already wrote the error response
		}

		// Check rate limit.
		if !checkRateLimit(c, resolved.AccountID) {
			return
		}

		// Record the request.
		instance.RecordRequest(resolved.AccountID, false, false)

		resp, proxyErr := instance.DoCompletionsProxy(c, resolved.State, bodyBytes)
		if proxyErr != nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			instance.RecordRequest(resolved.AccountID, true, false)
			var rewriteErr *instance.MessagesRewriteError
			if errors.As(proxyErr, &rewriteErr) {
				c.JSON(http.StatusBadRequest, gin.H{"error": rewriteErr.Error()})
				return
			}
			if attempt < maxAttempts-1 {
				exclude[resolved.AccountID] = true
				log.Printf("Completions proxy error for account %s, retrying: %v", resolved.AccountID, proxyErr)
				continue
			}
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		// Check if retryable.
		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < maxAttempts-1 {
				_ = resp.Body.Close()
				exclude[resolved.AccountID] = true
				log.Printf("Disabled unhealthy account %s after upstream error (%s), retrying", resolved.AccountID, reason)
				continue
			}
		}

		if isRetryableStatus(resp.StatusCode) && attempt < maxAttempts-1 {
			is429 := resp.StatusCode == http.StatusTooManyRequests
			instance.RecordRequest(resolved.AccountID, true, is429)
			_ = resp.Body.Close()
			exclude[resolved.AccountID] = true
			log.Printf("Upstream returned %d for account %s, retrying with different account", resp.StatusCode, resolved.AccountID)
			continue
		}

		// Forward the response.
		c.Set("respReqID", newRequestID())
		c.Set("respAccountID", resolved.AccountID)
		instance.ForwardCompletionsResponse(c, resp)
		return
	}
}

func proxyModels(c *gin.Context) {
	resolved := resolveState(c, nil, "")
	if resolved == nil {
		return
	}
	instance.ModelsHandler(c, resolved.State)
}

func proxyEmbeddings(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	requestedModel := extractRequestedModel(bodyBytes)
	previousResponseID := extractPreviousResponseID(bodyBytes)
	exclude := make(map[string]bool)
	for attempt := 0; attempt < maxAttempts; attempt++ {
		var resolved *resolvedAccount
		if previousResponseID != "" {
			resolved = resolveStateForResponseReplay(c, previousResponseID, requestedModel)
		} else {
			resolved = resolveState(c, exclude, requestedModel)
		}
		if resolved == nil {
			return
		}

		if !checkRateLimit(c, resolved.AccountID) {
			return
		}

		instance.RecordRequest(resolved.AccountID, false, false)

		resp, proxyErr := instance.DoEmbeddingsProxy(resolved.State, bodyBytes)
		if proxyErr != nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < maxAttempts-1 {
				exclude[resolved.AccountID] = true
				log.Printf("Embeddings proxy error for account %s, retrying: %v", resolved.AccountID, proxyErr)
				continue
			}
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < maxAttempts-1 {
				_ = resp.Body.Close()
				exclude[resolved.AccountID] = true
				log.Printf("Disabled unhealthy account %s after upstream error (%s), retrying", resolved.AccountID, reason)
				continue
			}
		}

		if isRetryableStatus(resp.StatusCode) && attempt < maxAttempts-1 {
			is429 := resp.StatusCode == http.StatusTooManyRequests
			instance.RecordRequest(resolved.AccountID, true, is429)
			_ = resp.Body.Close()
			exclude[resolved.AccountID] = true
			log.Printf("Upstream returned %d for account %s, retrying with different account", resp.StatusCode, resolved.AccountID)
			continue
		}

		instance.ForwardEmbeddingsResponse(c, resp)
		return
	}
}

func proxyMessages(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	requestedModel := extractRequestedModel(bodyBytes)
	toolResultIDs := extractMessageToolResultIDs(bodyBytes)
	continuationRequested := len(toolResultIDs) > 0
	exclude := make(map[string]bool)
	crossAccountRollover := false
	for attempt := 0; attempt < maxAttempts; attempt++ {
		var resolved *resolvedAccount
		if continuationRequested && !crossAccountRollover {
			resolved = resolveStateForMessageReplay(toolResultIDs, requestedModel)
		}
		if resolved == nil {
			resolved = resolveState(c, exclude, requestedModel)
		}
		if resolved == nil {
			return
		}

		if !checkRateLimit(c, resolved.AccountID) {
			return
		}

		instance.RecordRequest(resolved.AccountID, false, false)

		invoke := instance.DoMessagesProxy
		detachedMode := continuationRequested
		if detachedMode {
			invoke = instance.DoDetachedMessagesProxy
		}
		resp, turnRequest, proxyErr := invoke(c, resolved.AccountID, resolved.State, bodyBytes)
		if proxyErr != nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			instance.RecordRequest(resolved.AccountID, true, false)
			var rewriteErr *instance.MessagesRewriteError
			if errors.As(proxyErr, &rewriteErr) {
				c.JSON(http.StatusBadRequest, gin.H{"error": rewriteErr.Error()})
				return
			}
			if attempt < maxAttempts-1 {
				if continuationRequested && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("Messages detached continuation failed for account %s, retrying different account: %v", resolved.AccountID, proxyErr)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					log.Printf("Messages proxy error for account %s, retrying: %v", resolved.AccountID, proxyErr)
					continue
				}
			}
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		if continuationRequested {
			if replayInvalid, detail := readReplayInvalidResponse(resp); replayInvalid {
				instance.RecordRequest(resolved.AccountID, true, false)
				_ = resp.Body.Close()
				if attempt < maxAttempts-1 && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("Messages detached continuation unexpectedly replay-invalid for account %s, retrying different account: %s", resolved.AccountID, detail)
					continue
				}
				c.JSON(http.StatusBadGateway, gin.H{"error": "message continuation recovery failed"})
				return
			}
		}

		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < maxAttempts-1 {
				_ = resp.Body.Close()
				if continuationRequested && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("Disabled unhealthy account %s after detached message continuation error (%s), retrying different account", resolved.AccountID, reason)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					log.Printf("Disabled unhealthy account %s after upstream error (%s), retrying", resolved.AccountID, reason)
					continue
				}
			}
		}

		if isRetryableStatus(resp.StatusCode) && attempt < maxAttempts-1 {
			is429 := resp.StatusCode == http.StatusTooManyRequests
			instance.RecordRequest(resolved.AccountID, true, is429)
			_ = resp.Body.Close()
			if continuationRequested && isPool == true {
				exclude[resolved.AccountID] = true
				crossAccountRollover = true
				log.Printf("Detached message continuation returned %d for account %s, retrying with different account", resp.StatusCode, resolved.AccountID)
				continue
			}
			if !continuationRequested {
				exclude[resolved.AccountID] = true
				log.Printf("Upstream returned %d for account %s, retrying with different account", resp.StatusCode, resolved.AccountID)
				continue
			}
		}

		instance.ForwardMessagesResponse(c, resolved.AccountID, turnRequest, resp, bodyBytes)
		return
	}
}

func proxyCountTokens(c *gin.Context) {
	resolved := resolveState(c, nil, "")
	if resolved == nil {
		return
	}
	instance.CountTokensHandler(c, resolved.State)
}

func proxyResponsesCompact(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	requestedModel := extractRequestedModel(bodyBytes)
	exclude := make(map[string]bool)
	for attempt := 0; attempt < maxAttempts; attempt++ {
		resolved := resolveState(c, exclude, requestedModel)
		if resolved == nil {
			return
		}

		if !checkRateLimit(c, resolved.AccountID) {
			return
		}

		instance.RecordRequest(resolved.AccountID, false, false)

		resp, compactBody, proxyErr := instance.DoResponsesCompactProxy(resolved.State, bodyBytes)
		if proxyErr != nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			instance.RecordRequest(resolved.AccountID, true, false)
			var rewriteErr *instance.ResponsesRewriteError
			if errors.As(proxyErr, &rewriteErr) {
				c.JSON(http.StatusBadRequest, gin.H{"error": rewriteErr.Error()})
				return
			}
			if attempt < maxAttempts-1 {
				exclude[resolved.AccountID] = true
				log.Printf("Responses compact proxy error for account %s, retrying: %v", resolved.AccountID, proxyErr)
				continue
			}
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < maxAttempts-1 {
				_ = resp.Body.Close()
				exclude[resolved.AccountID] = true
				log.Printf("Disabled unhealthy account %s after upstream compact error (%s), retrying", resolved.AccountID, reason)
				continue
			}
		}

		if isRetryableStatus(resp.StatusCode) && attempt < maxAttempts-1 {
			is429 := resp.StatusCode == http.StatusTooManyRequests
			instance.RecordRequest(resolved.AccountID, true, is429)
			_ = resp.Body.Close()
			exclude[resolved.AccountID] = true
			log.Printf("Upstream compact returned %d for account %s, retrying with different account", resp.StatusCode, resolved.AccountID)
			continue
		}

		instance.ForwardResponsesCompactResponse(c, resp, compactBody)
		return
	}
}

func proxyResponses(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	requestedModel := extractRequestedModel(bodyBytes)
	previousResponseID := extractPreviousResponseID(bodyBytes)
	functionCallOutputIDs := extractResponseFunctionCallOutputIDs(bodyBytes)
	continuationRequested := previousResponseID != "" || len(functionCallOutputIDs) > 0
	exclude := make(map[string]bool)
	refreshedOnAccount := make(map[string]bool)
	crossAccountRollover := false
	orphanDegraded := false
	// pinnedAccount, when non-nil, forces the next iteration to reuse the same account
	// without consulting sticky caches or the pool. Used after a successful force-refresh
	// so the refreshed token is actually exercised on the account that minted it, rather
	// than re-running resolveState (which, on a fallback_round_robin path, would pick a
	// different account and waste the refresh).
	var pinnedAccount *resolvedAccount
	// attemptLimit starts at maxAttempts. Each successful force-refresh that pins a retry
	// grants one bonus slot so the pinned retry does not cannibalize the cross-account
	// rollover budget. Without this, a pool where A orphan-degrades and B persistently
	// 401s would exhaust the 3-attempt budget on A+B without ever trying C.
	attemptLimit := maxAttempts

	reqID := newRequestID()
	poolStrategy := ""
	if s, ok := c.Get("poolStrategy"); ok {
		poolStrategy = s.(string)
	}
	prevStickyReplay, prevStickyReplayOK := "", false
	prevStickyTurn, prevStickyTurnOK := "", false
	if previousResponseID != "" {
		prevStickyReplay, prevStickyReplayOK = instance.LookupResponsesReplayAccount(previousResponseID)
		prevStickyTurn, prevStickyTurnOK = instance.LookupResponseTurnAccount(previousResponseID)
	}
	fcSticky, fcStickyOK := "", false
	if len(functionCallOutputIDs) > 0 {
		fcSticky, fcStickyOK = instance.LookupResponseFunctionCallAccount(functionCallOutputIDs)
	}
	log.Printf("[responses rid=%s] recv model=%q prev_id=%q fc_ids=%v pool=%v strategy=%q continuation=%v prev_sticky_replay=%q(%v) prev_sticky_turn=%q(%v) fc_sticky=%q(%v)",
		reqID, requestedModel, previousResponseID, functionCallOutputIDs, isPool == true, poolStrategy, continuationRequested,
		prevStickyReplay, prevStickyReplayOK, prevStickyTurn, prevStickyTurnOK, fcSticky, fcStickyOK)

	for attempt := 0; attempt < attemptLimit; attempt++ {
		var resolved *resolvedAccount
		stickyKind := "none"
		if pinnedAccount != nil {
			resolved = pinnedAccount
			stickyKind = "pinned_refresh_retry"
			pinnedAccount = nil
		} else if !crossAccountRollover {
			switch {
			case previousResponseID != "":
				resolved = resolveStateForResponseReplay(c, previousResponseID, requestedModel)
				if resolved != nil {
					stickyKind = "prev_response_id"
				}
			case len(functionCallOutputIDs) > 0:
				resolved = resolveStateForResponseFunctionCallReplay(functionCallOutputIDs, requestedModel)
				if resolved != nil {
					stickyKind = "function_call_output"
				}
			}
		}
		fellBackToPool := false
		if resolved == nil {
			resolved = resolveState(c, exclude, requestedModel)
			if resolved != nil {
				fellBackToPool = true
				stickyKind = "fallback_round_robin"
			}
		}
		if resolved == nil {
			log.Printf("[responses rid=%s attempt=%d] resolve failed; error already written", reqID, attempt)
			return
		}

		log.Printf("[responses rid=%s attempt=%d] resolved account=%s sticky_kind=%s fell_back=%v cross_rollover=%v exclude=%v",
			reqID, attempt, resolved.AccountID, stickyKind, fellBackToPool, crossAccountRollover, exclude)

		if !checkRateLimit(c, resolved.AccountID) {
			log.Printf("[responses rid=%s attempt=%d] rate limited on account=%s", reqID, attempt, resolved.AccountID)
			return
		}

		instance.RecordRequest(resolved.AccountID, false, false)

		continuationCanDetach := continuationRequested && instance.CanReplayResponsesContinuation(resolved.AccountID, previousResponseID)
		invoke := instance.DoResponsesProxy
		modeName := "direct"
		if continuationRequested {
			switch {
			case crossAccountRollover:
				invoke = instance.DoDetachedResponsesProxyCrossAccount
				modeName = "detached_cross"
			case continuationCanDetach:
				invoke = instance.DoDetachedResponsesProxy
				modeName = "detached_same"
			}
		}
		log.Printf("[responses rid=%s attempt=%d] mode=%s can_detach=%v account=%s",
			reqID, attempt, modeName, continuationCanDetach, resolved.AccountID)

		resp, forwardedBody, turnRequest, proxyErr := invoke(resolved.AccountID, resolved.State, bodyBytes)
		log.Printf("[responses rid=%s attempt=%d] turn_ctx=%s interaction_type=%s",
			reqID, attempt, turnRequest.CacheSource, turnRequest.InteractionType)
		if proxyErr != nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			instance.RecordRequest(resolved.AccountID, true, false)
			var rewriteErr *instance.ResponsesRewriteError
			if errors.As(proxyErr, &rewriteErr) {
				log.Printf("[responses rid=%s attempt=%d] rewrite error on account=%s mode=%s: %v", reqID, attempt, resolved.AccountID, modeName, rewriteErr)
				c.JSON(http.StatusBadRequest, gin.H{"error": rewriteErr.Error()})
				return
			}
			if attempt < attemptLimit-1 {
				if continuationRequested && continuationCanDetach && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("[responses rid=%s attempt=%d] detached continuation failed on account=%s mode=%s, rolling over cross-account: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					log.Printf("[responses rid=%s attempt=%d] proxy error on account=%s mode=%s, retrying: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
					continue
				}
			}
			log.Printf("[responses rid=%s attempt=%d] giving up with 502 after proxy error on account=%s mode=%s: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		if continuationRequested {
			if replayInvalid, detail := readReplayInvalidResponse(resp); replayInvalid {
				instance.RecordRequest(resolved.AccountID, true, false)
				_ = resp.Body.Close()
				if continuationCanDetach && attempt < attemptLimit-1 && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("[responses rid=%s attempt=%d] replay-invalid on account=%s mode=%s, rolling over cross-account; detail=%q",
						reqID, attempt, resolved.AccountID, modeName, truncateForLog(detail, 240))
					continue
				}
				if !orphanDegraded && attempt < attemptLimit-1 {
					degradedBody, changed, degErr := instance.DegradeOrphanContinuationPayload(bodyBytes)
					if degErr == nil && changed > 0 {
						bodyBytes = degradedBody
						orphanDegraded = true
						exclude[resolved.AccountID] = true
						previousResponseID = ""
						functionCallOutputIDs = nil
						continuationRequested = false
						crossAccountRollover = false
						log.Printf("[responses rid=%s attempt=%d] orphan continuation on account=%s mode=%s, degraded %d fc items to text, retrying fresh; detail=%q",
							reqID, attempt, resolved.AccountID, modeName, changed, truncateForLog(detail, 240))
						log.Printf("[responses rid=%s attempt=%d] degraded body top-level keys=%s input_shape=[%s]",
							reqID, attempt, degradedBodyTopLevelKeys(degradedBody), degradedInputShape(degradedBody))
						continue
					}
					log.Printf("[responses rid=%s attempt=%d] orphan degrade skipped (changed=%d err=%v)", reqID, attempt, changed, degErr)
				}
				log.Printf("[responses rid=%s attempt=%d] replay-invalid on account=%s mode=%s, emitting 502 (can_detach=%v attempt_budget_left=%v is_pool=%v orphan_degraded=%v); detail=%q",
					reqID, attempt, resolved.AccountID, modeName, continuationCanDetach, attempt < attemptLimit-1, isPool == true, orphanDegraded, truncateForLog(detail, 240))
				c.JSON(http.StatusBadGateway, gin.H{"error": "response continuation recovery failed"})
				return
			}
		}

		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < attemptLimit-1 {
				_ = resp.Body.Close()
				if continuationRequested && continuationCanDetach && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					log.Printf("[responses rid=%s attempt=%d] disabled account=%s after detached continuation error (%s), rolling over cross-account", reqID, attempt, resolved.AccountID, reason)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					log.Printf("[responses rid=%s attempt=%d] disabled account=%s after upstream error (%s), retrying", reqID, attempt, resolved.AccountID, reason)
					continue
				}
			}
		}

		// 401 from a specific account is often a transient token-refresh-window race:
		// the scheduled refresh hasn't completed yet, so the cached copilot token upstream
		// rejects as expired. Force a refresh on this account and pin the next iteration
		// to the same account so the refreshed token is actually exercised — otherwise the
		// next iteration's resolveState may pick a different pool account and waste the
		// refresh. If the pinned retry also 401s, refreshedOnAccount guards this block
		// and we fall through to isRetryableStatus for cross-account rollover.
		if resp.StatusCode == http.StatusUnauthorized && !refreshedOnAccount[resolved.AccountID] && attempt < attemptLimit-1 {
			instance.RecordRequest(resolved.AccountID, true, false)
			if orphanDegraded {
				log.Printf("[responses rid=%s attempt=%d] post-degrade 401 body on account=%s: %q",
					reqID, attempt, resolved.AccountID, peekAndRestoreBody(resp, 400))
			}
			_ = resp.Body.Close()
			refreshedOnAccount[resolved.AccountID] = true
			if refreshErr := instance.ForceRefreshToken(resolved.AccountID); refreshErr != nil {
				log.Printf("[responses rid=%s attempt=%d] 401 on account=%s: token refresh failed (%v), falling back to cross-account retry", reqID, attempt, resolved.AccountID, refreshErr)
				exclude[resolved.AccountID] = true
				if continuationRequested && continuationCanDetach && isPool == true {
					crossAccountRollover = true
				}
				continue
			}
			pinnedAccount = resolved
			attemptLimit++
			log.Printf("[responses rid=%s attempt=%d] 401 on account=%s: forced token refresh ok, pinning retry to same account (attemptLimit=%d)", reqID, attempt, resolved.AccountID, attemptLimit)
			continue
		}

		if isRetryableStatus(resp.StatusCode) && attempt < attemptLimit-1 {
			is429 := resp.StatusCode == http.StatusTooManyRequests
			instance.RecordRequest(resolved.AccountID, true, is429)
			if orphanDegraded {
				log.Printf("[responses rid=%s attempt=%d] post-degrade %d body on account=%s: %q",
					reqID, attempt, resp.StatusCode, resolved.AccountID, peekAndRestoreBody(resp, 400))
			}
			_ = resp.Body.Close()
			if continuationRequested && continuationCanDetach && isPool == true {
				exclude[resolved.AccountID] = true
				crossAccountRollover = true
				log.Printf("[responses rid=%s attempt=%d] upstream %d on account=%s mode=%s, rolling over cross-account", reqID, attempt, resp.StatusCode, resolved.AccountID, modeName)
				continue
			}
			if !continuationRequested {
				exclude[resolved.AccountID] = true
				log.Printf("[responses rid=%s attempt=%d] upstream %d on account=%s, retrying with different account", reqID, attempt, resp.StatusCode, resolved.AccountID)
				continue
			}
		}

		if orphanDegraded && resp.StatusCode >= 400 {
			log.Printf("[responses rid=%s attempt=%d] post-degrade %d body on account=%s (final): %q",
				reqID, attempt, resp.StatusCode, resolved.AccountID, peekAndRestoreBody(resp, 400))
		}
		log.Printf("[responses rid=%s attempt=%d] forwarding upstream_status=%d account=%s mode=%s",
			reqID, attempt, resp.StatusCode, resolved.AccountID, modeName)
		c.Set("respReqID", reqID)
		instance.ForwardResponsesResponse(c, resolved.AccountID, turnRequest, forwardedBody, resp)
		return
	}
}
