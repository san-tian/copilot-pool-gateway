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
	"time"

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

// RegisterProxy sets up the proxy server's authenticated API routes.
//
// proxyAuth is scoped to a sub-group (not the engine) so that public routes
// mounted on the same engine by RegisterProxyPublic — the SPA aliases and the
// small /api subset needed by the login / supplier-auth pages — are not forced
// through Bearer-token auth meant for LLM API callers.
func RegisterProxy(r *gin.Engine) {
	// Initialize rate limiter from environment.
	instance.InitRateLimiter()

	// Load per-account rate limit from pool config.
	if poolCfg, err := store.GetPoolConfig(); err == nil && poolCfg != nil {
		instance.SetPerAccountRPM(poolCfg.RateLimitRPM)
	}

	api := r.Group("")
	api.Use(proxyAuth())

	// OpenAI compatible endpoints
	api.POST("/chat/completions", proxyCompletions)
	api.POST("/v1/chat/completions", proxyCompletions)
	api.GET("/models", proxyModels)
	api.GET("/v1/models", proxyModels)
	api.POST("/embeddings", proxyEmbeddings)
	api.POST("/v1/embeddings", proxyEmbeddings)

	// Anthropic compatible endpoints
	api.POST("/v1/messages", proxyMessages)
	api.POST("/v1/messages/count_tokens", proxyCountTokens)

	// OpenAI Responses API endpoint
	api.POST("/responses/compact", proxyResponsesCompact)
	api.POST("/v1/responses/compact", proxyResponsesCompact)
	api.POST("/v1/responses", proxyResponses)
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

func choosePinnedResponsesAccount(pinnedAccount, continuationPinned, sameTurnPinned *resolvedAccount, pinnedKind string) (*resolvedAccount, string, bool) {
	if pinnedAccount != nil {
		if pinnedKind == "" {
			pinnedKind = "pinned_retry"
		}
		return pinnedAccount, pinnedKind, true
	}
	if continuationPinned != nil {
		return continuationPinned, "session_binding_canonical", false
	}
	if sameTurnPinned != nil {
		return sameTurnPinned, "same_turn_pinned", false
	}
	return nil, "none", false
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

// canonicalizeCopilotCallID reverses a downstream agent's "mangled" rewriting
// of a Copilot function-call id. Some agents (observed with Pi's routing
// layer) take the upstream-minted `call_<24body>` id, drop the underscore,
// and append a 12-char routing tag, yielding a 40-char `call<24body><12tag>`
// form when they later submit the corresponding function_call_output.
//
// Leaving the mangled id in place breaks us two ways:
//  1. Our sticky cache keys on the canonical form the stream capture stored,
//     so the tag-suffixed id misses and we round-robin to a random account.
//  2. Upstream Copilot validates every input item's id against its own
//     function_call history on the chosen connection — the mangled form
//     never matches, so upstream rejects the continuation with
//     "input item ID does not belong to this connection" and we fall back
//     to the lossy orphan-degrade path.
//
// Canonicalizing at ingestion makes both problems vanish with a single
// transform: strip the 12-char tail and reinsert the underscore.
func canonicalizeCopilotCallID(id string) string {
	if strings.HasPrefix(id, "call_") {
		return id
	}
	if len(id) != 40 || !strings.HasPrefix(id, "call") {
		return id
	}
	return "call_" + id[4:28]
}

// rewriteFunctionCallOutputCallIDs parses a /v1/responses request body and
// canonicalizes every downstream-mangled call_id on both function_call /
// function_call_output AND custom_tool_call / custom_tool_call_output items.
// Covering only one half of a pair breaks the pair validation upstream
// performs at continuation — a canonical output id with a still-mangled call
// id yields "input item ID does not belong to this connection" (or the older
// "No tool output found for function call <mangled-id>") — so the rewrite
// must span every call_id-bearing item type. custom_tool_call items come from
// Codex-style custom tools and exhibit the same 40-char mangling from the
// downstream passthrough path; skipping them triggered the same orphan-
// continuation → degrade → retry loop that motivated the original fix.
// Returns the original bytes untouched if the body doesn't parse, carries no
// rewritable ids, or wasn't actually rewritten.
func rewriteFunctionCallOutputCallIDs(bodyBytes []byte) []byte {
	if len(bodyBytes) == 0 {
		return bodyBytes
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return bodyBytes
	}
	rewrite := func(item interface{}) bool {
		entry, ok := item.(map[string]interface{})
		if !ok {
			return false
		}
		itemType, _ := entry["type"].(string)
		normalized := strings.TrimSpace(strings.ToLower(itemType))
		switch normalized {
		case "function_call", "function_call_output",
			"custom_tool_call", "custom_tool_call_output":
		default:
			return false
		}
		callID, _ := entry["call_id"].(string)
		canonical := canonicalizeCopilotCallID(strings.TrimSpace(callID))
		if canonical == callID {
			return false
		}
		entry["call_id"] = canonical
		return true
	}
	changed := false
	switch typed := payload["input"].(type) {
	case []interface{}:
		for _, item := range typed {
			if rewrite(item) {
				changed = true
			}
		}
	case map[string]interface{}:
		if rewrite(typed) {
			changed = true
		}
	}
	if !changed {
		return bodyBytes
	}
	out, err := json.Marshal(payload)
	if err != nil {
		return bodyBytes
	}
	return out
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
		if affinityKey, ok := c.Get("responsesSessionAffinityKey"); ok {
			if key, _ := affinityKey.(string); key != "" {
				if resolved := lookupResponsesSessionAffinity(key, requestedModel, combinedExclude); resolved != nil {
					c.Set("responsesSessionAffinityStatus", "hit")
					return resolved
				}
			}
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
		if affinityKey, ok := c.Get("responsesSessionAffinityKey"); ok {
			if key, _ := affinityKey.(string); key != "" {
				rememberResponsesSessionAffinity(key, account.ID)
				c.Set("responsesSessionAffinityStatus", "bind")
			}
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

// canOrphanPassthrough reports whether an fc-id orphan can be safely routed
// as a fresh request. When RESPONSES_ORPHAN_TRANSLATE=on, direct-mode accounts
// are also recoverable because the gateway can translate orphan Responses
// turns in-process and proxy the translated request upstream without relying
// on a per-account sidecar worker. When orphan translation is off, we still
// require a worker-capable target because raw /v1/responses passthrough would
// session-reject the foreign call_ids.
func canOrphanPassthrough(c *gin.Context, requestedModel string, exclude map[string]bool) bool {
	requireWorker := config.ResponsesOrphanTranslate() != "on"
	isPool, _ := c.Get("isPool")
	if isPool == true {
		accounts, err := store.GetEnabledAccounts()
		if err != nil {
			return false
		}
		blocked := store.GetBlockedAccountIDs(requestedModel)
		for _, a := range accounts {
			if requireWorker && strings.TrimSpace(a.WorkerURL) == "" {
				continue
			}
			if exclude[a.ID] || blocked[a.ID] {
				continue
			}
			if instance.GetInstanceState(a.ID) == nil {
				continue
			}
			return true
		}
		return false
	}
	accountID, exists := c.Get("accountID")
	if !exists {
		return false
	}
	aid, _ := accountID.(string)
	if aid == "" {
		return false
	}
	acct, err := store.GetAccount(aid)
	if err != nil || acct == nil {
		return false
	}
	if requireWorker && strings.TrimSpace(acct.WorkerURL) == "" {
		return false
	}
	if store.IsModelBlocked(aid, requestedModel) {
		return false
	}
	if instance.GetInstanceState(aid) == nil {
		return false
	}
	return true
}

// workerDisabledAccountIDs returns enabled account IDs that have no WorkerURL.
// Used to exclude direct-mode-only accounts from pool routing in the orphan
// passthrough branch, so the pool only picks worker-capable targets.
func workerDisabledAccountIDs(requestedModel string) []string {
	accounts, err := store.GetEnabledAccounts()
	if err != nil {
		return nil
	}
	out := make([]string, 0)
	for _, a := range accounts {
		if strings.TrimSpace(a.WorkerURL) == "" {
			out = append(out, a.ID)
		}
	}
	return out
}

// orphanTranslateCompatibleModel reports whether the requested model can be
// served via Copilot's /v1/chat/completions endpoint. The gpt-5 family and
// Anthropic models are rejected by Copilot with
//
//	{"error":{"message":"Please use `/v1/responses` or `/v1/messages` API"}}
//
// so we must NOT translate their orphan requests to chat/completions — a
// separate /v1/messages translator handles them. Everything else is attempted
// via chat/completions; if a model we haven't seen before 400s, the body-snippet
// log in DoOrphanTranslateResponsesProxy surfaces it for blocklist expansion.
func orphanTranslateCompatibleModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return false
	}
	for _, prefix := range []string{"gpt-5", "claude-"} {
		if strings.HasPrefix(m, prefix) {
			return false
		}
	}
	return true
}

// orphanTranslateMessagesModel reports whether the requested model should be
// served via Copilot's /v1/messages endpoint (Anthropic-compat, stateless).
// This covers exactly the gpt-5 family and Claude models — the families that
// /v1/chat/completions rejects with the "Please use /v1/responses or /v1/messages"
// error. Everything else stays on the chat translator path.
func orphanTranslateMessagesModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return false
	}
	for _, prefix := range []string{"gpt-5", "claude-"} {
		if strings.HasPrefix(m, prefix) {
			return true
		}
	}
	return false
}

type orphanTranslateRoute string

const (
	orphanTranslateRouteNone     orphanTranslateRoute = ""
	orphanTranslateRouteChat     orphanTranslateRoute = "chat"
	orphanTranslateRouteMessages orphanTranslateRoute = "messages"
)

func (r orphanTranslateRoute) logName() string {
	if r == "" {
		return "none"
	}
	return string(r)
}

func (r orphanTranslateRoute) modeName() string {
	switch r {
	case orphanTranslateRouteChat:
		return "orphan_translate"
	case orphanTranslateRouteMessages:
		return "orphan_translate_messages"
	default:
		return "direct"
	}
}

func orphanTranslateRouteForModel(model string) orphanTranslateRoute {
	switch {
	case orphanTranslateMessagesModel(model):
		return orphanTranslateRouteMessages
	case orphanTranslateCompatibleModel(model):
		return orphanTranslateRouteChat
	default:
		return orphanTranslateRouteNone
	}
}

type continuationRecoveryState struct {
	Route       orphanTranslateRoute
	Reason      string
	FromAccount string
}

func (s continuationRecoveryState) armed() bool {
	return s.Route != orphanTranslateRouteNone
}

func resolveOrphanRecoveryState(c *gin.Context, requestedModel string, exclude map[string]bool, reason string, fromAccount string) continuationRecoveryState {
	if config.OrphanPassthrough() == "off" || config.ResponsesOrphanTranslate() != "on" {
		return continuationRecoveryState{}
	}
	if !canOrphanPassthrough(c, requestedModel, exclude) {
		return continuationRecoveryState{}
	}
	route := orphanTranslateRouteForModel(requestedModel)
	if route == orphanTranslateRouteNone {
		return continuationRecoveryState{}
	}
	return continuationRecoveryState{
		Route:       route,
		Reason:      reason,
		FromAccount: fromAccount,
	}
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

// continuationBindingKind classifies how a /v1/responses continuation request
// maps onto the sticky cache before we pick an upstream account. The four
// outcomes are disjoint and drive deterministic routing: canonical → single
// pinned account, anything else → typed HTTP error (unless the caller opts
// in to lossy orphan degrade).
type continuationBindingKind int

const (
	// continuationBindingCanonical — every cache hit agrees on one account
	// and that account is currently usable for the requested model. The loop
	// pins this account for all attempts; no cross-account rollover.
	continuationBindingCanonical continuationBindingKind = iota
	// continuationBindingAccountUnavailable — cache points at a single
	// account but it's currently model-blocked or its instance is not
	// running. 503: the user must wait or restart; we will not silently
	// rotate to a different account (which would orphan on upstream).
	continuationBindingAccountUnavailable
	// continuationBindingSplit — two or more accounts each own a subset of
	// this request's function_call_output call_ids. Upstream validation is
	// all-or-nothing per account, so no dispatch is safe. 409: the user
	// must restart.
	continuationBindingSplit
	// continuationBindingOrphan — no cache entry at all (new session, TTL
	// expiry, or eviction on account removal). 410 by default; if the
	// caller sets X-Copilot-Continuation-Degrade: orphan, we textify the
	// fc history and fall through to first-turn routing.
	continuationBindingOrphan
)

type continuationBindingResult struct {
	Kind          continuationBindingKind
	Resolved      *resolvedAccount // set when Kind == Canonical
	AccountID     string           // set when Kind == Canonical or AccountUnavailable
	SplitAccounts []string         // set when Kind == Split
	Reason        string           // human-readable diagnostic for logs and 4xx/5xx bodies
	HitCount      int              // diagnostic: fc cache hits
	MissCount     int              // diagnostic: fc cache misses
}

// resolveContinuationBinding classifies a continuation request (prev_id or
// function_call_output ids present) against the sticky cache without
// touching any HTTP context. Pure function: the caller decides how to turn
// the result into a response and does the HTTP work. `previousResponseID`
// takes precedence over the fc_ids path — if both are set, we bind by
// prev_id because it's a single-key lookup with stronger semantics.
func resolveContinuationBinding(previousResponseID string, functionCallOutputIDs []string, requestedModel string) continuationBindingResult {
	previousResponseID = strings.TrimSpace(previousResponseID)
	if previousResponseID != "" {
		accountID, ok := instance.LookupResponsesReplayAccount(previousResponseID)
		if !ok {
			accountID, ok = instance.LookupResponseTurnAccount(previousResponseID)
		}
		if !ok {
			return continuationBindingResult{
				Kind:   continuationBindingOrphan,
				Reason: fmt.Sprintf("previous_response_id %q was not found or has expired", previousResponseID),
			}
		}
		return canonicalAccountBinding(accountID, requestedModel)
	}
	if len(functionCallOutputIDs) == 0 {
		return continuationBindingResult{Kind: continuationBindingOrphan, Reason: "no previous_response_id and no function_call_output ids"}
	}
	session := instance.ResolveResponseFunctionCallSession(functionCallOutputIDs)
	switch session.Kind {
	case instance.SessionCanonical:
		binding := canonicalAccountBinding(session.AccountID, requestedModel)
		binding.HitCount = session.HitCount
		binding.MissCount = session.MissCount
		return binding
	case instance.SessionSplit:
		return continuationBindingResult{
			Kind:          continuationBindingSplit,
			SplitAccounts: session.SplitAccounts,
			Reason:        fmt.Sprintf("function_call_output history spans %d accounts: %s", len(session.SplitAccounts), strings.Join(session.SplitAccounts, ", ")),
			HitCount:      session.HitCount,
			MissCount:     session.MissCount,
		}
	default: // SessionOrphan
		return continuationBindingResult{
			Kind:      continuationBindingOrphan,
			Reason:    fmt.Sprintf("no function_call_output call_id matches any known session (hits=0 misses=%d)", session.MissCount),
			HitCount:  0,
			MissCount: session.MissCount,
		}
	}
}

// canonicalAccountBinding turns a canonical accountID into either a
// fully-resolved Canonical binding or an AccountUnavailable binding with
// the specific reason. Called from both the prev_id and fc_ids paths in
// resolveContinuationBinding.
func canonicalAccountBinding(accountID, requestedModel string) continuationBindingResult {
	if store.IsModelBlocked(accountID, requestedModel) {
		return continuationBindingResult{
			Kind:      continuationBindingAccountUnavailable,
			AccountID: accountID,
			Reason:    fmt.Sprintf("account %s is paused for model %s", accountID, requestedModel),
		}
	}
	state := instance.GetInstanceState(accountID)
	if state == nil {
		return continuationBindingResult{
			Kind:      continuationBindingAccountUnavailable,
			AccountID: accountID,
			Reason:    fmt.Sprintf("account %s instance is not running", accountID),
		}
	}
	return continuationBindingResult{
		Kind:      continuationBindingCanonical,
		AccountID: accountID,
		Resolved:  &resolvedAccount{State: state, AccountID: accountID},
	}
}

// writeSessionBindingError writes a typed 4xx/5xx response for a continuation
// binding that can't be safely dispatched. Clients see a structured error
// body with a `type` field they can key off to decide whether to retry,
// restart the session, or wait and try again.
func writeSessionBindingError(c *gin.Context, reqID string, binding continuationBindingResult) {
	switch binding.Kind {
	case continuationBindingAccountUnavailable:
		log.Printf("[responses rid=%s] session binding unavailable: account=%s reason=%q", reqID, binding.AccountID, binding.Reason)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": gin.H{
				"type":       "session_bound_account_unavailable",
				"message":    binding.Reason,
				"account_id": binding.AccountID,
			},
		})
	case continuationBindingSplit:
		log.Printf("[responses rid=%s] session split history: accounts=%v reason=%q", reqID, binding.SplitAccounts, binding.Reason)
		c.JSON(http.StatusConflict, gin.H{
			"error": gin.H{
				"type":     "session_split_history",
				"message":  binding.Reason,
				"accounts": binding.SplitAccounts,
			},
		})
	default: // continuationBindingOrphan
		log.Printf("[responses rid=%s] session expired: reason=%q", reqID, binding.Reason)
		c.JSON(http.StatusGone, gin.H{
			"error": gin.H{
				"type":    "session_expired",
				"message": binding.Reason,
			},
		})
	}
}

// continuationDegradeOptIn reports whether the caller explicitly opted into
// lossy orphan degrade for this request. The header surface is kept small
// on purpose — one value for now; future modes would extend the enum rather
// than proliferating headers. `sub2api` sets this when its cross-provider
// switching path wants best-effort continuation; native clients do not.
func continuationDegradeOptIn(c *gin.Context) bool {
	return strings.EqualFold(strings.TrimSpace(c.GetHeader("X-Copilot-Continuation-Degrade")), "orphan")
}

func proxyResponses(c *gin.Context) {
	isPool, _ := c.Get("isPool")
	maxAttempts := 1
	if isPool == true {
		maxAttempts = 3
	}

	bodyReadStart := time.Now()
	bodyBytes, err := io.ReadAll(c.Request.Body)
	bodyReadMs := time.Since(bodyReadStart).Milliseconds()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}
	bodyBytes = rewriteFunctionCallOutputCallIDs(bodyBytes)

	requestedModel := extractRequestedModel(bodyBytes)
	setResponsesSessionAffinityContext(c, isPool, bodyBytes)
	previousResponseID := extractPreviousResponseID(bodyBytes)
	functionCallOutputIDs := extractResponseFunctionCallOutputIDs(bodyBytes)
	continuationRequested := previousResponseID != "" || len(functionCallOutputIDs) > 0
	exclude := make(map[string]bool)
	refreshedOnAccount := make(map[string]bool)
	crossAccountRollover := false
	orphanDegraded := false
	// recovery records the explicit broken-continuation recovery route. Normal
	// requests use ordinary pool routing; valid continuations use canonical
	// sticky routing; only split/orphan/upstream-rejected continuations arm
	// this state and switch to the stateless transcript recovery route.
	recovery := continuationRecoveryState{}
	// pinnedAccount, when non-nil, forces the next iteration to reuse a known
	// account without consulting sticky caches or the pool. Successful token
	// refreshes and canonical replay-invalid recovery both use this, with
	// pinnedAccountKind keeping the log semantics explicit.
	var pinnedAccount *resolvedAccount
	pinnedAccountKind := ""
	// sameTurnPinned makes the first resolved account immutable for the
	// lifetime of this HTTP request. /v1/responses carries account-owned
	// interaction and tool-call state; trying another pool account inside the
	// same logical turn can double-spend the turn and still lose history.
	var sameTurnPinned *resolvedAccount
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
	affinitySource := ""
	if s, ok := c.Get("responsesSessionAffinitySource"); ok {
		affinitySource, _ = s.(string)
	}
	affinityKey := ""
	if s, ok := c.Get("responsesSessionAffinityKey"); ok {
		affinityKey, _ = s.(string)
	}
	pendingSwitchFrom := ""
	pendingSwitchReason := ""
	pendingSwitchStatus := 0
	recordSwitchTrigger := func(attempt int, fromAccount string, reason string, status int, mode string) {
		pendingSwitchFrom = fromAccount
		pendingSwitchReason = reason
		pendingSwitchStatus = status
		recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
			Kind:          "account_switch_trigger",
			RequestID:     reqID,
			SessionKey:    affinityKey,
			SessionSource: affinitySource,
			Model:         requestedModel,
			AccountID:     fromAccount,
			FromAccount:   fromAccount,
			Attempt:       attempt,
			StatusCode:    status,
			Reason:        reason,
			RecoveryRoute: recovery.Route.logName(),
			Mode:          mode,
			Continuation:  continuationRequested,
		})
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
	log.Printf("[responses rid=%s] recv model=%q prev_id=%q fc_ids=%v pool=%v strategy=%q affinity_source=%q continuation=%v body_bytes=%d body_read_ms=%d prev_sticky_replay=%q(%v) prev_sticky_turn=%q(%v) fc_sticky=%q(%v)",
		reqID, requestedModel, previousResponseID, functionCallOutputIDs, isPool == true, poolStrategy, affinitySource, continuationRequested, len(bodyBytes), bodyReadMs,
		prevStickyReplay, prevStickyReplayOK, prevStickyTurn, prevStickyTurnOK, fcSticky, fcStickyOK)

	// Strict per-session binding for continuation requests.
	//
	// A continuation is any request that carries previous_response_id OR
	// function_call_output ids from a prior turn. Upstream Copilot validates
	// those ids against the specific account+interaction that minted them:
	// if we dispatch to a different account than the one that owns the
	// history, upstream returns `input item does not belong to this
	// connection`. So we classify the
	// binding here, once, before the retry loop:
	//   - Canonical (every hit agrees on one available account) → pin it for
	//     every attempt. Cross-account rollover is disabled downstream by
	//     forcing continuationCanDetach=false. 401 force-refresh still
	//     retries the same account via pinnedAccount.
	//   - AccountUnavailable (canonical account blocked/stopped) → 503. The
	//     user must wait or restart; we won't silently rotate.
	//   - Split (two or more accounts claim different call_ids) → 409. The
	//     history is unrecoverable on any single account.
	//   - Orphan (no hits) → 410, unless the caller set
	//     X-Copilot-Continuation-Degrade: orphan, in which case we degrade
	//     the fc history into text and fall through to first-turn routing.
	var continuationPinned *resolvedAccount
	// degradeOptIn is computed once and consulted in two places: the pre-loop
	// orphan-from-cache branch (directly below) and the in-loop
	// replay-invalid-from-upstream branch further down. Both must agree so
	// that codex / pi clients (which never set the header) always see typed
	// errors and sub2api (which sets it) keeps its best-effort degrade path.
	degradeOptIn := continuationRequested && continuationDegradeOptIn(c)
	if continuationRequested {
		binding := resolveContinuationBinding(previousResponseID, functionCallOutputIDs, requestedModel)
		log.Printf("[responses rid=%s] continuation_binding kind=%d account=%s split=%v hits=%d misses=%d reason=%q degrade_opt_in=%v",
			reqID, binding.Kind, binding.AccountID, binding.SplitAccounts, binding.HitCount, binding.MissCount, binding.Reason, degradeOptIn)
		switch binding.Kind {
		case continuationBindingCanonical:
			continuationPinned = binding.Resolved
		case continuationBindingAccountUnavailable, continuationBindingSplit:
			// Non-canonical but recoverable via orphan_translate transcript recovery:
			// Split = history spans multiple accounts (common after a client-
			// side compact that exposes cross-account call_ids); Unavailable
			// = sole owner account is blocked/stopped. The recovery path
			// removes account-owned protocol ids and wraps protocol history
			// in a labeled context block, so neither case depends on any
			// specific account owning the ids. Falls back to the original
			// typed error (409/503) when recovery isn't armable.
			recovery = resolveOrphanRecoveryState(c, requestedModel, exclude, binding.Reason, binding.AccountID)
			if recovery.armed() && len(functionCallOutputIDs) > 0 {
				recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
					Kind:          "recovery_route_armed",
					RequestID:     reqID,
					SessionKey:    affinityKey,
					SessionSource: affinitySource,
					Model:         requestedModel,
					FromAccount:   binding.AccountID,
					Reason:        binding.Reason,
					RecoveryRoute: recovery.Route.logName(),
					Continuation:  true,
				})
				log.Printf("[responses rid=%s] recovery armed binding_kind=%d recovery_route=%s recovery_from_account=%q hits=%d misses=%d fc_ids=%d accounts=%v reason=%q",
					reqID, binding.Kind, recovery.Route.logName(), recovery.FromAccount, binding.HitCount, binding.MissCount, len(functionCallOutputIDs), binding.SplitAccounts, binding.Reason)
				previousResponseID = ""
				functionCallOutputIDs = nil
				continuationRequested = false
				break
			}
			writeSessionBindingError(c, reqID, binding)
			return
		case continuationBindingOrphan:
			// Default-on non-lossy passthrough for cross-relay migration:
			// when the orphan is a function_call_output orphan (input is
			// self-contained, history is present in full), worker mode is
			// available, and no degrade opt-in was requested, clear sticky
			// state and route fresh. With RESPONSES_ORPHAN_TRANSLATE=on, the
			// gateway uses the stateless recovery route; otherwise only
			// worker-enabled accounts are eligible for legacy passthrough.
			//
			// Skipped for prev_response_id-only orphans because that case's
			// input is truncated and depends on server-side history — we
			// can't recover it without re-sending full history, so 410 is
			// still the honest answer.
			if !degradeOptIn && config.OrphanPassthrough() != "off" &&
				len(functionCallOutputIDs) > 0 && canOrphanPassthrough(c, requestedModel, exclude) {
				// Queue non-worker accounts into the exclude set so the
				// pool-routing path below only picks worker-enabled ones.
				// Direct-mode accounts would hit the Responses endpoint
				// which IS session-stateful and would reject the orphan.
				if isPool == true && config.ResponsesOrphanTranslate() != "on" {
					for _, id := range workerDisabledAccountIDs(requestedModel) {
						exclude[id] = true
					}
				}
				log.Printf("[responses rid=%s] orphan_passthrough pre-loop: hits=%d misses=%d fc_ids=%d — clearing continuation and routing fresh",
					reqID, binding.HitCount, binding.MissCount, len(functionCallOutputIDs))
				previousResponseID = ""
				functionCallOutputIDs = nil
				continuationRequested = false
				// When orphan-translate mode is armed, the invoke selection
				// below picks a stateless recovery route instead of the
				// stateful /v1/responses upstream.
				recovery = resolveOrphanRecoveryState(c, requestedModel, exclude, binding.Reason, "")
				if recovery.armed() {
					recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
						Kind:          "recovery_route_armed",
						RequestID:     reqID,
						SessionKey:    affinityKey,
						SessionSource: affinitySource,
						Model:         requestedModel,
						Reason:        binding.Reason,
						RecoveryRoute: recovery.Route.logName(),
						Continuation:  true,
					})
					log.Printf("[responses rid=%s] recovery armed binding_kind=%d recovery_route=%s recovery_from_account=%q model=%q reason=%q",
						reqID, binding.Kind, recovery.Route.logName(), recovery.FromAccount, requestedModel, recovery.Reason)
				} else if config.ResponsesOrphanTranslate() == "on" {
					log.Printf("[responses rid=%s] orphan_translate skipped for model=%q; no compatible recovery route, falling through to orphan_passthrough", reqID, requestedModel)
				}
				// bodyBytes unchanged: input carries the full self-contained
				// history including structured tool-use items.
				break
			}
			if !degradeOptIn {
				writeSessionBindingError(c, reqID, binding)
				return
			}
			// Opt-in degrade: textify the fc items, clear continuation state,
			// and fall through to the normal first-turn pool routing below.
			degradedBody, changed, degErr := instance.DegradeOrphanContinuationPayload(bodyBytes)
			if degErr != nil || changed == 0 {
				log.Printf("[responses rid=%s] orphan degrade opt-in requested but could not apply (changed=%d err=%v); emitting 410", reqID, changed, degErr)
				writeSessionBindingError(c, reqID, binding)
				return
			}
			bodyBytes = degradedBody
			orphanDegraded = true
			previousResponseID = ""
			functionCallOutputIDs = nil
			continuationRequested = false
			log.Printf("[responses rid=%s] orphan degrade opt-in applied pre-loop: degraded %d fc items to text", reqID, changed)
		}
	}

	for attempt := 0; attempt < attemptLimit; attempt++ {
		var resolved *resolvedAccount
		stickyKind := "none"
		if chosen, kind, usedOneShot := choosePinnedResponsesAccount(pinnedAccount, continuationPinned, sameTurnPinned, pinnedAccountKind); chosen != nil {
			resolved = chosen
			stickyKind = kind
			if usedOneShot {
				pinnedAccount = nil
				pinnedAccountKind = ""
			}
		}
		if resolved != nil && sameTurnPinned != nil && resolved.AccountID == sameTurnPinned.AccountID {
			crossAccountRollover = false
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
		if sameTurnPinned == nil {
			sameTurnPinned = resolved
		}
		if pendingSwitchFrom != "" {
			kind := "account_switch_avoided"
			if pendingSwitchFrom != resolved.AccountID {
				kind = "account_switched"
			}
			recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
				Kind:          kind,
				RequestID:     reqID,
				SessionKey:    affinityKey,
				SessionSource: affinitySource,
				Model:         requestedModel,
				AccountID:     resolved.AccountID,
				FromAccount:   pendingSwitchFrom,
				ToAccount:     resolved.AccountID,
				Attempt:       attempt,
				StatusCode:    pendingSwitchStatus,
				Reason:        pendingSwitchReason,
				RecoveryRoute: recovery.Route.logName(),
				Continuation:  continuationRequested,
			})
			pendingSwitchFrom = ""
			pendingSwitchReason = ""
			pendingSwitchStatus = 0
		}

		affinityStatus := ""
		if s, ok := c.Get("responsesSessionAffinityStatus"); ok {
			affinityStatus, _ = s.(string)
		}
		if affinityStatus == "hit" && stickyKind == "fallback_round_robin" {
			stickyKind = "session_affinity"
		}
		if (affinityStatus == "bind" || affinityStatus == "hit") && stickyKind != "same_turn_pinned" {
			recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
				Kind:          "session_affinity_" + affinityStatus,
				RequestID:     reqID,
				SessionKey:    affinityKey,
				SessionSource: affinitySource,
				Model:         requestedModel,
				AccountID:     resolved.AccountID,
				Attempt:       attempt,
				StickyKind:    stickyKind,
				RecoveryRoute: recovery.Route.logName(),
				Continuation:  continuationRequested,
			})
		}
		log.Printf("[responses rid=%s attempt=%d] resolved account=%s sticky_kind=%s fell_back=%v cross_rollover=%v session_affinity=%q exclude=%v recovery_route=%s recovery_from_account=%q recovery_to_account=%q recovery_reason=%q",
			reqID, attempt, resolved.AccountID, stickyKind, fellBackToPool, crossAccountRollover, affinityStatus, exclude,
			recovery.Route.logName(), recovery.FromAccount, resolved.AccountID, recovery.Reason)

		if !checkRateLimit(c, resolved.AccountID) {
			log.Printf("[responses rid=%s attempt=%d] rate limited on account=%s", reqID, attempt, resolved.AccountID)
			return
		}

		instance.RecordRequest(resolved.AccountID, false, false)

		continuationCanDetach := continuationRequested && instance.CanReplayResponsesContinuation(resolved.AccountID, previousResponseID)
		// When the session is canonical-bound, disable the detach/rollover
		// machinery outright: the one retry budget we care about is 401
		// force-refresh on the same account (handled below via pinnedAccount),
		// never cross-account. Leaving continuationCanDetach true here would
		// let the retry branches below flip crossAccountRollover=true and
		// route a future attempt to a different account, guaranteeing an
		// orphan reject — which is exactly the bug this refactor closes.
		if continuationPinned != nil {
			continuationCanDetach = false
		}
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
		// Recovery routing overrides direct/detached mode only after the
		// continuation state has been cleared. That keeps normal account
		// binding separate from the stateless transcript recovery retry.
		if recovery.Route == orphanTranslateRouteChat {
			invoke = instance.DoOrphanTranslateResponsesProxy
			modeName = recovery.Route.modeName()
		}
		if recovery.Route == orphanTranslateRouteMessages {
			invoke = instance.DoOrphanTranslateMessagesProxy
			modeName = recovery.Route.modeName()
		}
		log.Printf("[responses rid=%s attempt=%d] mode=%s can_detach=%v account=%s",
			reqID, attempt, modeName, continuationCanDetach, resolved.AccountID)

		invokeStart := time.Now()
		resp, forwardedBody, turnRequest, proxyErr := invoke(resolved.AccountID, resolved.State, bodyBytes)
		invokeMs := time.Since(invokeStart).Milliseconds()
		log.Printf("[responses rid=%s attempt=%d] turn_ctx=%s interaction_type=%s invoke_ms=%d",
			reqID, attempt, turnRequest.CacheSource, turnRequest.InteractionType, invokeMs)
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
					recordSwitchTrigger(attempt, resolved.AccountID, "proxy_error_detached_continuation", 0, modeName)
					log.Printf("[responses rid=%s attempt=%d] detached continuation failed on account=%s mode=%s, retrying same turn account: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					recordSwitchTrigger(attempt, resolved.AccountID, "proxy_error_non_continuation", 0, modeName)
					log.Printf("[responses rid=%s attempt=%d] proxy error on account=%s mode=%s, retrying: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
					continue
				}
			}
			log.Printf("[responses rid=%s attempt=%d] giving up with 502 after proxy error on account=%s mode=%s: %v", reqID, attempt, resolved.AccountID, modeName, proxyErr)
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("proxy request failed: %v", proxyErr)})
			return
		}

		if replayInvalid, detail := readReplayInvalidResponse(resp); replayInvalid {
			instance.RecordRequest(resolved.AccountID, true, false)
			_ = resp.Body.Close()
			if continuationRequested && continuationCanDetach && attempt < attemptLimit-1 && isPool == true {
				exclude[resolved.AccountID] = true
				crossAccountRollover = true
				recordSwitchTrigger(attempt, resolved.AccountID, "replay_invalid_detached_continuation", resp.StatusCode, modeName)
				log.Printf("[responses rid=%s attempt=%d] replay-invalid on account=%s mode=%s, retrying same turn account; detail=%q",
					reqID, attempt, resolved.AccountID, modeName, truncateForLog(detail, 240))
				continue
			}
			// The in-loop orphan-degrade branch text-ifies the tool
			// history when upstream reports `input item does not belong to
			// this connection`. It is the real driver of the codex
			// write_stdin death loop: once degrade fires, the model sees
			// a summarized history every subsequent turn and never
			// regains tool-loop structure. Gate it on the same opt-in
			// header the pre-loop orphan branch uses so native clients
			// (codex, pi) get a typed 502 they can react to, and sub2api
			// continues to get best-effort degrade when it asks.
			if degradeOptIn && !orphanDegraded && attempt < attemptLimit-1 {
				degradedBody, changed, degErr := instance.DegradeOrphanContinuationPayload(bodyBytes)
				if degErr == nil && changed > 0 {
					bodyBytes = degradedBody
					orphanDegraded = true
					exclude[resolved.AccountID] = true
					recordSwitchTrigger(attempt, resolved.AccountID, "orphan_degrade_retry", resp.StatusCode, modeName)
					previousResponseID = ""
					functionCallOutputIDs = nil
					continuationRequested = false
					continuationPinned = nil
					crossAccountRollover = false
					log.Printf("[responses rid=%s attempt=%d] orphan continuation on account=%s mode=%s, degraded %d fc items to text, retrying fresh; detail=%q",
						reqID, attempt, resolved.AccountID, modeName, changed, truncateForLog(detail, 240))
					log.Printf("[responses rid=%s attempt=%d] degraded body top-level keys=%s input_shape=[%s]",
						reqID, attempt, degradedBodyTopLevelKeys(degradedBody), degradedInputShape(degradedBody))
					continue
				}
				log.Printf("[responses rid=%s attempt=%d] orphan degrade skipped (changed=%d err=%v)", reqID, attempt, changed, degErr)
			}
			// Canonical binding said "this account owns the history" but
			// upstream disagreed (session state was evicted / rotated).
			// Treat this as a late orphan and fall through to the orphan
			// translate path, which wraps protocol history in a labeled
			// context block and dispatches to the worker's stateless endpoint.
			// Unlike the opt-in
			// degrade above, this does NOT require the degrade header —
			// the translator is the canonical transcript recovery mechanism
			// and already runs on orphan-bound traffic today. Guarded by
			// orphanDegraded and recovery.armed() so a single request does
			// not keep re-arming the same recovery route.
			if !orphanDegraded && !recovery.armed() {
				recovery = resolveOrphanRecoveryState(c, requestedModel, exclude, "upstream rejected continuation", resolved.AccountID)
				if recovery.armed() {
					recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
						Kind:          "recovery_route_armed",
						RequestID:     reqID,
						SessionKey:    affinityKey,
						SessionSource: affinitySource,
						Model:         requestedModel,
						AccountID:     resolved.AccountID,
						FromAccount:   resolved.AccountID,
						ToAccount:     resolved.AccountID,
						Attempt:       attempt,
						StatusCode:    resp.StatusCode,
						Reason:        "upstream_replay_invalid",
						RecoveryRoute: recovery.Route.logName(),
						Mode:          modeName,
						Continuation:  true,
					})
					orphanDegraded = true
					if attempt >= attemptLimit-1 {
						attemptLimit++
					}
					previousResponseID = ""
					functionCallOutputIDs = nil
					continuationRequested = false
					continuationPinned = nil
					crossAccountRollover = false
					pinnedAccount = resolved
					pinnedAccountKind = "recovery_same_account"
					log.Printf("[responses rid=%s attempt=%d] recovery armed after replay-invalid recovery_from_account=%q recovery_to_account=%q recovery_route=%s previous_mode=%s detail=%q",
						reqID, attempt, resolved.AccountID, resolved.AccountID, recovery.Route.logName(), modeName, truncateForLog(detail, 240))
					continue
				}
			}
			if recovery.armed() && isPool == true && attempt < attemptLimit-1 {
				exclude[resolved.AccountID] = true
				recordSwitchTrigger(attempt, resolved.AccountID, "recovery_replay_invalid", resp.StatusCode, modeName)
				log.Printf("[responses rid=%s attempt=%d] replay-invalid during recovery route=%s account=%s, retrying same turn account; detail=%q",
					reqID, attempt, recovery.Route.logName(), resolved.AccountID, truncateForLog(detail, 240))
				continue
			}
			log.Printf("[responses rid=%s attempt=%d] replay-invalid on account=%s mode=%s, emitting typed 502 (can_detach=%v attempt_budget_left=%v is_pool=%v recovery_route=%s orphan_degraded=%v degrade_opt_in=%v); detail=%q",
				reqID, attempt, resolved.AccountID, modeName, continuationCanDetach, attempt < attemptLimit-1, isPool == true, recovery.Route.logName(), orphanDegraded, degradeOptIn, truncateForLog(detail, 240))
			c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{
				"type":    "session_upstream_orphan",
				"message": "upstream rejected the continuation: the tool-call history cannot be replayed on the bound account. Restart the conversation.",
			}})
			return
		}

		if disabled, reason := disableOnFatalUpstream(resp, resolved.AccountID, requestedModel); disabled {
			instance.RecordRequest(resolved.AccountID, true, false)
			if attempt < attemptLimit-1 {
				_ = resp.Body.Close()
				if continuationRequested && continuationCanDetach && isPool == true {
					exclude[resolved.AccountID] = true
					crossAccountRollover = true
					recordSwitchTrigger(attempt, resolved.AccountID, "account_disabled_detached_continuation", resp.StatusCode, modeName)
					log.Printf("[responses rid=%s attempt=%d] disabled account=%s after detached continuation error (%s), retrying same turn account", reqID, attempt, resolved.AccountID, reason)
					continue
				}
				if !continuationRequested {
					exclude[resolved.AccountID] = true
					recordSwitchTrigger(attempt, resolved.AccountID, "account_disabled_non_continuation", resp.StatusCode, modeName)
					log.Printf("[responses rid=%s attempt=%d] disabled account=%s after upstream error (%s), retrying", reqID, attempt, resolved.AccountID, reason)
					continue
				}
			}
		}

		// 401 from a specific account is often a transient token-refresh-window race:
		// the scheduled refresh hasn't completed yet, so the cached copilot token upstream
		// rejects as expired. Force a refresh on this account and pin the next iteration
		// to the same account so the refreshed token is actually exercised. Responses
		// same-turn routing must not move this request to a different pool account.
		if resp.StatusCode == http.StatusUnauthorized && !refreshedOnAccount[resolved.AccountID] && attempt < attemptLimit-1 {
			instance.RecordRequest(resolved.AccountID, true, false)
			if orphanDegraded {
				log.Printf("[responses rid=%s attempt=%d] post-degrade 401 body on account=%s: %q",
					reqID, attempt, resolved.AccountID, peekAndRestoreBody(resp, 400))
			}
			_ = resp.Body.Close()
			refreshedOnAccount[resolved.AccountID] = true
			if refreshErr := instance.ForceRefreshToken(resolved.AccountID); refreshErr != nil {
				log.Printf("[responses rid=%s attempt=%d] 401 on account=%s: token refresh failed (%v), retrying same turn account", reqID, attempt, resolved.AccountID, refreshErr)
				exclude[resolved.AccountID] = true
				recordSwitchTrigger(attempt, resolved.AccountID, "token_refresh_failed", resp.StatusCode, modeName)
				if continuationRequested && continuationCanDetach && isPool == true {
					crossAccountRollover = true
				}
				continue
			}
			pinnedAccount = resolved
			pinnedAccountKind = "pinned_refresh_retry"
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
				recordSwitchTrigger(attempt, resolved.AccountID, "retryable_status_detached_continuation", resp.StatusCode, modeName)
				log.Printf("[responses rid=%s attempt=%d] upstream %d on account=%s mode=%s, retrying same turn account", reqID, attempt, resp.StatusCode, resolved.AccountID, modeName)
				continue
			}
			if !continuationRequested {
				exclude[resolved.AccountID] = true
				recordSwitchTrigger(attempt, resolved.AccountID, "retryable_status_non_continuation", resp.StatusCode, modeName)
				log.Printf("[responses rid=%s attempt=%d] upstream %d on account=%s, retrying same turn account", reqID, attempt, resp.StatusCode, resolved.AccountID)
				continue
			}
		}

		if orphanDegraded && resp.StatusCode >= 400 {
			log.Printf("[responses rid=%s attempt=%d] post-degrade %d body on account=%s (final): %q",
				reqID, attempt, resp.StatusCode, resolved.AccountID, peekAndRestoreBody(resp, 400))
		}
		log.Printf("[responses rid=%s attempt=%d] forwarding upstream_status=%d account=%s mode=%s invoke_ms=%d",
			reqID, attempt, resp.StatusCode, resolved.AccountID, modeName, invokeMs)
		c.Set("respReqID", reqID)
		instance.ForwardResponsesResponse(c, resolved.AccountID, turnRequest, forwardedBody, resp)
		return
	}
}
