package instance

import (
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"copilot-go/anthropic"

	"github.com/google/uuid"
)

const (
	copilotTurnCacheTTL        = 30 * time.Minute
	copilotTurnCacheMaxEntries = 1024

	copilotInteractionTypeUser  = "conversation-user"
	copilotInteractionTypeAgent = "conversation-agent"
)

var copilotClientMachineID = uuid.NewString()

type copilotTurnContext struct {
	InteractionID   string
	ClientSessionID string
	AgentTaskID     string
}

type copilotTurnRequest struct {
	Context         copilotTurnContext
	InteractionType string
	Initiator       string
	CacheSource     string
}

type CopilotTurnRequest = copilotTurnRequest

type copilotTurnCacheEntry struct {
	AccountID  string
	Context    copilotTurnContext
	CreatedAt  time.Time
	AccessedAt time.Time
}

var responseTurnCache = struct {
	mu      sync.Mutex
	entries map[string]*copilotTurnCacheEntry
}{
	entries: map[string]*copilotTurnCacheEntry{},
}

var responseFunctionCallTurnCache = struct {
	mu      sync.Mutex
	entries map[string]*copilotTurnCacheEntry
}{
	entries: map[string]*copilotTurnCacheEntry{},
}

var messageToolCallTurnCache = struct {
	mu      sync.Mutex
	entries map[string]*copilotTurnCacheEntry
}{
	entries: map[string]*copilotTurnCacheEntry{},
}

func newCopilotTurnContext() copilotTurnContext {
	return copilotTurnContext{
		InteractionID:   uuid.NewString(),
		ClientSessionID: uuid.NewString(),
		AgentTaskID:     uuid.NewString(),
	}
}

func newCopilotTurnRequest(interactionType string) copilotTurnRequest {
	initiator := "user"
	if interactionType == copilotInteractionTypeAgent {
		initiator = "agent"
	}
	return copilotTurnRequest{
		Context:         newCopilotTurnContext(),
		InteractionType: interactionType,
		Initiator:       initiator,
	}
}

func reusableCopilotAgentTurnRequest(req copilotTurnRequest) bool {
	return req.InteractionType == copilotInteractionTypeAgent &&
		req.Context.InteractionID != "" &&
		req.Context.ClientSessionID != "" &&
		req.Context.AgentTaskID != ""
}

func recoveryCopilotTurnRequest(base copilotTurnRequest, freshSource string, reuseSource string) copilotTurnRequest {
	if reusableCopilotAgentTurnRequest(base) {
		base.CacheSource = reuseSource
		return base
	}
	req := newCopilotTurnRequest(copilotInteractionTypeUser)
	req.CacheSource = freshSource
	return req
}

func (r copilotTurnRequest) Headers() http.Header {
	h := make(http.Header)
	h.Set("X-Initiator", r.Initiator)
	h.Set("X-Interaction-Type", r.InteractionType)
	h.Set("X-Interaction-Id", r.Context.InteractionID)
	h.Set("X-Client-Session-Id", r.Context.ClientSessionID)
	h.Set("X-Agent-Task-Id", r.Context.AgentTaskID)
	h.Set("X-Client-Machine-Id", copilotClientMachineID)
	return h
}

func buildResponsesTurnRequest(accountID string, previousResponseID string, rawInput interface{}) copilotTurnRequest {
	previousResponseID = strings.TrimSpace(previousResponseID)
	if previousResponseID != "" {
		if ctx, ok := loadResponseTurnContext(accountID, previousResponseID); ok {
			return copilotTurnRequest{
				Context:         ctx,
				InteractionType: copilotInteractionTypeAgent,
				Initiator:       "agent",
				CacheSource:     "prev_response_id_hit",
			}
		}
		req := newCopilotTurnRequest(copilotInteractionTypeUser)
		req.CacheSource = "prev_response_id_miss"
		return req
	}
	input, _ := normalizeResponsesInput(rawInput)
	fcIDs := collectResponsesFunctionCallOutputIDs(input)
	if ctx, ok := loadResponseFunctionCallTurnContext(accountID, fcIDs); ok {
		return copilotTurnRequest{
			Context:         ctx,
			InteractionType: copilotInteractionTypeAgent,
			Initiator:       "agent",
			CacheSource:     "fc_output_id_hit",
		}
	}
	req := newCopilotTurnRequest(copilotInteractionTypeUser)
	if len(fcIDs) > 0 {
		req.CacheSource = "fc_output_id_miss"
	} else {
		req.CacheSource = "fresh_no_prev"
	}
	return req
}

func buildMessagesTurnRequest(accountID string, payload anthropic.AnthropicMessagesPayload) copilotTurnRequest {
	toolResultIDs := latestAnthropicToolResultIDs(payload.Messages)
	if len(toolResultIDs) == 0 {
		return newCopilotTurnRequest(copilotInteractionTypeUser)
	}
	if ctx, ok := loadMessageToolCallTurnContext(accountID, toolResultIDs); ok {
		return copilotTurnRequest{
			Context:         ctx,
			InteractionType: copilotInteractionTypeAgent,
			Initiator:       "agent",
		}
	}
	return newCopilotTurnRequest(copilotInteractionTypeUser)
}

func latestAnthropicToolResultIDs(messages []anthropic.AnthropicMessage) []string {
	for idx := len(messages) - 1; idx >= 0; idx-- {
		message := messages[idx]
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
		if len(ids) > 0 {
			return uniqueTrimmed(ids)
		}
		break
	}
	return nil
}

func currentCompletionsInitiator(rawMessages interface{}) string {
	messages, ok := rawMessages.([]interface{})
	if !ok || len(messages) == 0 {
		return "user"
	}
	for idx := len(messages) - 1; idx >= 0; idx-- {
		message, ok := messages[idx].(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		role = strings.TrimSpace(role)
		if role == "" || role == "system" || role == "developer" {
			continue
		}
		if role == "tool" {
			return "agent"
		}
		return "user"
	}
	return "user"
}

func collectToolCallIDsFromChatCompletion(resp anthropic.ChatCompletionResponse) []string {
	ids := make([]string, 0)
	for _, choice := range resp.Choices {
		if choice.Message == nil {
			continue
		}
		for _, toolCall := range choice.Message.ToolCalls {
			if toolCallID := strings.TrimSpace(toolCall.ID); toolCallID != "" {
				ids = append(ids, toolCallID)
			}
		}
	}
	return uniqueTrimmed(ids)
}

func collectToolCallIDsFromStreamState(state *anthropic.AnthropicStreamState) []string {
	if state == nil {
		return nil
	}
	ids := make([]string, 0, len(state.ToolCalls))
	for _, toolCall := range state.ToolCalls {
		if toolCall == nil {
			continue
		}
		if toolCallID := strings.TrimSpace(toolCall.ID); toolCallID != "" {
			ids = append(ids, toolCallID)
		}
	}
	return uniqueTrimmed(ids)
}

func storeResponseTurnContext(accountID string, responseID string, ctx copilotTurnContext) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return
	}
	ensureDurableContinuationStateLoaded()
	storeCopilotTurnCacheEntry(&responseTurnCache.mu, responseTurnCache.entries, responseID, accountID, ctx)
	persistDurableContinuationState()
}

func loadResponseTurnContext(accountID string, responseID string) (copilotTurnContext, bool) {
	ensureDurableContinuationStateLoaded()
	return loadCopilotTurnCacheEntry(&responseTurnCache.mu, responseTurnCache.entries, responseID, accountID)
}

func storeResponseFunctionCallTurnContext(accountID string, callIDs []string, ctx copilotTurnContext) {
	callIDs = uniqueTrimmed(callIDs)
	if len(callIDs) == 0 {
		return
	}
	ensureDurableContinuationStateLoaded()
	for _, callID := range callIDs {
		storeCopilotTurnCacheEntry(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, callID, accountID, ctx)
	}
	persistDurableContinuationState()
}

func loadResponseFunctionCallTurnContext(accountID string, callIDs []string) (copilotTurnContext, bool) {
	ensureDurableContinuationStateLoaded()
	for _, callID := range uniqueTrimmed(callIDs) {
		if ctx, ok := loadCopilotTurnCacheEntry(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, callID, accountID); ok {
			return ctx, true
		}
	}
	return copilotTurnContext{}, false
}

func storeMessageToolCallTurnContext(accountID string, toolCallIDs []string, ctx copilotTurnContext) {
	toolCallIDs = uniqueTrimmed(toolCallIDs)
	if len(toolCallIDs) == 0 {
		return
	}
	ensureDurableContinuationStateLoaded()
	for _, toolCallID := range toolCallIDs {
		storeCopilotTurnCacheEntry(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, toolCallID, accountID, ctx)
	}
	persistDurableContinuationState()
}

func loadMessageToolCallTurnContext(accountID string, toolCallIDs []string) (copilotTurnContext, bool) {
	ensureDurableContinuationStateLoaded()
	for _, toolCallID := range uniqueTrimmed(toolCallIDs) {
		if ctx, ok := loadCopilotTurnCacheEntry(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, toolCallID, accountID); ok {
			return ctx, true
		}
	}
	return copilotTurnContext{}, false
}

func LookupMessageToolCallAccount(toolCallIDs []string) (string, bool) {
	ensureDurableContinuationStateLoaded()
	for _, toolCallID := range uniqueTrimmed(toolCallIDs) {
		if accountID, ok := lookupCopilotTurnCacheAccount(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, toolCallID); ok {
			return accountID, true
		}
	}
	return "", false
}

func LookupResponseTurnAccount(responseID string) (string, bool) {
	ensureDurableContinuationStateLoaded()
	return lookupCopilotTurnCacheAccount(&responseTurnCache.mu, responseTurnCache.entries, responseID)
}

func LookupResponseFunctionCallAccount(callIDs []string) (string, bool) {
	ensureDurableContinuationStateLoaded()
	for _, callID := range uniqueTrimmed(callIDs) {
		if accountID, ok := lookupCopilotTurnCacheAccount(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, callID); ok {
			return accountID, true
		}
	}
	return "", false
}

// SessionResolutionKind classifies how a set of function_call_output call_ids
// maps onto the sticky cache. The three outcomes drive routing for
// continuation requests on /v1/responses — see ResolveResponseFunctionCallSession.
type SessionResolutionKind int

const (
	// SessionCanonical — every hit in the cache agrees on a single account.
	// Partial misses (TTL-expired entries) are OK and counted in MissCount.
	SessionCanonical SessionResolutionKind = iota
	// SessionSplit — two or more accounts each own a subset of the call_ids.
	// The history is cross-account contaminated and cannot be safely continued
	// on any single account; upstream will orphan-reject no matter which we
	// pick.
	SessionSplit
	// SessionOrphan — no call_id hits the cache. Either the session is new
	// (first turn, fc_ids empty) or every entry expired out of the cache
	// (30min TTL / 1024 LRU) or was evicted when its account was removed.
	SessionOrphan
)

// SessionResolution is the result of classifying a set of function_call_output
// call_ids against the sticky cache.
type SessionResolution struct {
	Kind          SessionResolutionKind
	AccountID     string   // set when Kind == SessionCanonical
	SplitAccounts []string // set when Kind == SessionSplit (sorted for stable diagnostics)
	HitCount      int      // number of call_ids that hit the cache
	MissCount     int      // number of call_ids that missed
}

// ResolveResponseFunctionCallSession classifies the supplied function_call_output
// call_ids against the sticky cache. Intended as the single decision point for
// continuation routing on /v1/responses: the caller dispatches on Kind instead
// of using "any-match" LookupResponseFunctionCallAccount which silently picks
// one account from a split history.
func ResolveResponseFunctionCallSession(callIDs []string) SessionResolution {
	ensureDurableContinuationStateLoaded()
	trimmed := uniqueTrimmed(callIDs)
	if len(trimmed) == 0 {
		return SessionResolution{Kind: SessionOrphan}
	}
	accountSeen := make(map[string]bool)
	hits := 0
	for _, id := range trimmed {
		if accountID, ok := lookupCopilotTurnCacheAccount(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, id); ok {
			accountSeen[accountID] = true
			hits++
		}
	}
	miss := len(trimmed) - hits
	switch len(accountSeen) {
	case 0:
		return SessionResolution{Kind: SessionOrphan, HitCount: 0, MissCount: miss}
	case 1:
		var accountID string
		for id := range accountSeen {
			accountID = id
		}
		return SessionResolution{Kind: SessionCanonical, AccountID: accountID, HitCount: hits, MissCount: miss}
	default:
		accounts := make([]string, 0, len(accountSeen))
		for id := range accountSeen {
			accounts = append(accounts, id)
		}
		sort.Strings(accounts)
		return SessionResolution{Kind: SessionSplit, SplitAccounts: accounts, HitCount: hits, MissCount: miss}
	}
}

// stashResponseFunctionCallTurnContextInMemory writes an account-bound context
// for a set of call_ids into the in-memory function-call cache without
// triggering the gzip-and-rename durable persist. Used by the stream-capture
// path to record call_ids incrementally on each function_call item so that if
// the upstream connection drops mid-stream the already-seen ids are still
// bound to the account that minted them. Caller is responsible for a single
// persistDurableContinuationState() call at stream terminal or termination.
func stashResponseFunctionCallTurnContextInMemory(accountID string, callIDs []string, ctx copilotTurnContext) {
	callIDs = uniqueTrimmed(callIDs)
	if len(callIDs) == 0 {
		return
	}
	ensureDurableContinuationStateLoaded()
	for _, callID := range callIDs {
		storeCopilotTurnCacheEntry(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, callID, accountID, ctx)
	}
}

// EvictAccountContinuationCaches removes every sticky-cache entry bound to
// the given accountID across the response-turn, function-call-turn,
// message-tool-call-turn, and responses-replay caches, then persists the
// updated snapshot. Called when an account transitions to disabled / deleted
// / blocked-for-all-models so future continuation requests classify as
// SessionOrphan cleanly instead of SessionCanonical-but-unavailable with a
// 503 the user can do nothing about.
func EvictAccountContinuationCaches(accountID string) {
	accountID = strings.TrimSpace(accountID)
	if accountID == "" {
		return
	}
	ensureDurableContinuationStateLoaded()
	evicted := 0
	evicted += evictCopilotTurnCacheForAccountLocked(&responseTurnCache.mu, responseTurnCache.entries, accountID)
	evicted += evictCopilotTurnCacheForAccountLocked(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, accountID)
	evicted += evictCopilotTurnCacheForAccountLocked(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, accountID)
	evicted += evictResponsesReplayForAccountLocked(accountID)
	if evicted > 0 {
		persistDurableContinuationState()
	}
}

func evictCopilotTurnCacheForAccountLocked(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, accountID string) int {
	mu.Lock()
	defer mu.Unlock()
	removed := 0
	for key, entry := range entries {
		if entry.AccountID == accountID {
			delete(entries, key)
			removed++
		}
	}
	return removed
}

func evictResponsesReplayForAccountLocked(accountID string) int {
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()
	removed := 0
	for key, entry := range responsesReplayCache.entries {
		if entry.AccountID == accountID {
			delete(responsesReplayCache.entries, key)
			removed++
		}
	}
	return removed
}

func storeCopilotTurnCacheEntry(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, key string, accountID string, ctx copilotTurnContext) {
	key = strings.TrimSpace(key)
	accountID = strings.TrimSpace(accountID)
	if key == "" || accountID == "" {
		return
	}
	now := time.Now()
	mu.Lock()
	defer mu.Unlock()
	pruneCopilotTurnCacheLocked(entries, now)
	entries[key] = &copilotTurnCacheEntry{
		AccountID:  accountID,
		Context:    ctx,
		CreatedAt:  now,
		AccessedAt: now,
	}
	pruneCopilotTurnCacheLocked(entries, now)
}

func loadCopilotTurnCacheEntry(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, key string, accountID string) (copilotTurnContext, bool) {
	key = strings.TrimSpace(key)
	accountID = strings.TrimSpace(accountID)
	if key == "" || accountID == "" {
		return copilotTurnContext{}, false
	}
	now := time.Now()
	mu.Lock()
	defer mu.Unlock()
	pruneCopilotTurnCacheLocked(entries, now)
	entry, ok := entries[key]
	if !ok || entry.AccountID != accountID {
		return copilotTurnContext{}, false
	}
	entry.AccessedAt = now
	return entry.Context, true
}

func lookupCopilotTurnCacheAccount(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, key string) (string, bool) {
	key = strings.TrimSpace(key)
	if key == "" {
		return "", false
	}
	now := time.Now()
	mu.Lock()
	defer mu.Unlock()
	pruneCopilotTurnCacheLocked(entries, now)
	entry, ok := entries[key]
	if !ok {
		return "", false
	}
	entry.AccessedAt = now
	return entry.AccountID, true
}

func pruneCopilotTurnCacheLocked(entries map[string]*copilotTurnCacheEntry, now time.Time) {
	for key, entry := range entries {
		if now.Sub(entry.CreatedAt) > copilotTurnCacheTTL {
			delete(entries, key)
		}
	}
	if len(entries) <= copilotTurnCacheMaxEntries {
		return
	}
	type cacheKey struct {
		Key        string
		AccessedAt time.Time
	}
	keys := make([]cacheKey, 0, len(entries))
	for key, entry := range entries {
		keys = append(keys, cacheKey{Key: key, AccessedAt: entry.AccessedAt})
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].AccessedAt.Before(keys[j].AccessedAt)
	})
	for len(entries) > copilotTurnCacheMaxEntries && len(keys) > 0 {
		delete(entries, keys[0].Key)
		keys = keys[1:]
	}
}

func uniqueTrimmed(values []string) []string {
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
