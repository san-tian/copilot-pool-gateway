package instance

import (
	"encoding/json"
	"sync"
	"testing"

	"copilot-go/anthropic"
	"copilot-go/store"
)

func init() {
	durableContinuationPersistenceDisabled = true
}

func clearCopilotContinuationCachesOnly() {
	responseTurnCache.mu.Lock()
	responseTurnCache.entries = map[string]*copilotTurnCacheEntry{}
	responseTurnCache.mu.Unlock()

	responseFunctionCallTurnCache.mu.Lock()
	responseFunctionCallTurnCache.entries = map[string]*copilotTurnCacheEntry{}
	responseFunctionCallTurnCache.mu.Unlock()

	messageToolCallTurnCache.mu.Lock()
	messageToolCallTurnCache.entries = map[string]*copilotTurnCacheEntry{}
	messageToolCallTurnCache.mu.Unlock()

	responsesReplayCache.mu.Lock()
	responsesReplayCache.entries = map[string]*responsesReplayEntry{}
	responsesReplayCache.mu.Unlock()

	continuationStateLoadOnce = sync.Once{}
}

func TestDurableContinuationStateReloadsFromDisk(t *testing.T) {
	oldAppDir := store.AppDir
	oldDisabled := durableContinuationPersistenceDisabled
	oldMachineID := copilotClientMachineID

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = false
	copilotClientMachineID = newCopilotTurnContext().InteractionID

	t.Cleanup(func() {
		store.AppDir = oldAppDir
		durableContinuationPersistenceDisabled = oldDisabled
		copilotClientMachineID = oldMachineID
		clearCopilotContinuationCachesOnly()
	})

	ctx := newCopilotTurnContext()
	storeResponseTurnContext("acct-1", "resp_prev", ctx)
	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"}},
	)
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)
	storeMessageToolCallTurnContext("acct-1", []string{"call_1"}, ctx)

	// Persists are now asynchronous and debounced; force a sync write so the
	// on-disk file is guaranteed to reflect the mutations above.
	flushDurableContinuationStateForTests()

	statePath := continuationStateFile()
	body, err := readGzipFile(statePath)
	if err != nil {
		t.Fatalf("readGzipFile(%s): %v", statePath, err)
	}
	var snapshot map[string]interface{}
	if err := json.Unmarshal(body, &snapshot); err != nil {
		t.Fatalf("expected valid JSON snapshot, got error: %v", err)
	}

	persistedMachineID := copilotClientMachineID
	copilotClientMachineID = "machine-reset-for-test"
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = false

	turnRequest := buildResponsesTurnRequest("acct-1", "resp_prev", nil)
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected persisted response turn to reload as agent continuation, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected persisted response context %+v, got %+v", ctx, turnRequest.Context)
	}
	if copilotClientMachineID != persistedMachineID {
		t.Fatalf("expected persisted machine ID %q, got %q", persistedMachineID, copilotClientMachineID)
	}

	if accountID, ok := LookupResponsesReplayAccount("resp_prev"); !ok || accountID != "acct-1" {
		t.Fatalf("expected persisted replay account acct-1, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_1"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected persisted function call account acct-1, got %q ok=%v", accountID, ok)
	}

	payload := map[string]interface{}{
		"previous_response_id": "resp_prev",
		"input":                []interface{}{map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "ok"}},
	}
	if err := rewritePreviousResponseContinuation("acct-1", payload); err != nil {
		t.Fatalf("expected persisted replay rewrite to reload, got error: %v", err)
	}
	if _, hasPrev := payload["previous_response_id"]; hasPrev {
		t.Fatalf("expected previous_response_id to be removed after persisted rewrite")
	}

	messagePayload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "call_1", Content2: "ok"}}},
		},
	}
	messageTurn := buildMessagesTurnRequest("acct-1", messagePayload)
	if messageTurn.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected persisted tool turn to reload as agent continuation, got %q", messageTurn.InteractionType)
	}
	if messageTurn.Context != ctx {
		t.Fatalf("expected persisted tool context %+v, got %+v", ctx, messageTurn.Context)
	}
}

func TestEvictAccountContinuationCachesRemovesOnlyTargetAccount(t *testing.T) {
	clearCopilotContinuationCachesOnly()
	t.Cleanup(clearCopilotContinuationCachesOnly)

	ctx1 := newCopilotTurnContext()
	ctx2 := newCopilotTurnContext()

	// acct-1 has entries in all four caches.
	storeResponseTurnContext("acct-1", "resp_1", ctx1)
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_a1", "call_a2"}, ctx1)
	storeMessageToolCallTurnContext("acct-1", []string{"tool_a1"}, ctx1)
	storeResponsesReplay(
		"acct-1",
		"resp_1",
		[]interface{}{map[string]interface{}{"role": "user", "content": "a"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_a1", "name": "shell", "arguments": "{}"}},
	)

	// acct-2 has entries in all four caches — must be left untouched.
	storeResponseTurnContext("acct-2", "resp_2", ctx2)
	storeResponseFunctionCallTurnContext("acct-2", []string{"call_b1"}, ctx2)
	storeMessageToolCallTurnContext("acct-2", []string{"tool_b1"}, ctx2)
	storeResponsesReplay(
		"acct-2",
		"resp_2",
		[]interface{}{map[string]interface{}{"role": "user", "content": "b"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_b1", "name": "shell", "arguments": "{}"}},
	)

	EvictAccountContinuationCaches("acct-1")

	if accountID, ok := LookupResponseTurnAccount("resp_1"); ok {
		t.Fatalf("expected resp_1 to be evicted, still mapped to %q", accountID)
	}
	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_a1"}); ok {
		t.Fatalf("expected call_a1 to be evicted, still mapped to %q", accountID)
	}
	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_a2"}); ok {
		t.Fatalf("expected call_a2 to be evicted, still mapped to %q", accountID)
	}
	if accountID, ok := LookupMessageToolCallAccount([]string{"tool_a1"}); ok {
		t.Fatalf("expected tool_a1 to be evicted, still mapped to %q", accountID)
	}
	if accountID, ok := LookupResponsesReplayAccount("resp_1"); ok {
		t.Fatalf("expected resp_1 replay to be evicted, still mapped to %q", accountID)
	}

	// acct-2 entries remain intact.
	if accountID, ok := LookupResponseTurnAccount("resp_2"); !ok || accountID != "acct-2" {
		t.Fatalf("expected acct-2 resp_2 preserved, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_b1"}); !ok || accountID != "acct-2" {
		t.Fatalf("expected acct-2 call_b1 preserved, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupMessageToolCallAccount([]string{"tool_b1"}); !ok || accountID != "acct-2" {
		t.Fatalf("expected acct-2 tool_b1 preserved, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupResponsesReplayAccount("resp_2"); !ok || accountID != "acct-2" {
		t.Fatalf("expected acct-2 replay preserved, got %q ok=%v", accountID, ok)
	}
}

func TestEvictAccountContinuationCachesEmptyAccountIsNoop(t *testing.T) {
	clearCopilotContinuationCachesOnly()
	t.Cleanup(clearCopilotContinuationCachesOnly)

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)

	EvictAccountContinuationCaches("")
	EvictAccountContinuationCaches("   ")

	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_1"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected entry preserved on empty accountID evict, got %q ok=%v", accountID, ok)
	}
}
