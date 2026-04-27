package instance

import (
	"encoding/json"
	"os"
	"sync"
	"testing"
	"time"

	"copilot-go/anthropic"
	"copilot-go/store"
)

func init() {
	durableContinuationPersistenceDisabled = true
	continuationMetadataPersistenceDisabled = true
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
	resetContinuationMetadataStoreForTests()
}

func TestDurableContinuationStateWritesDisabled(t *testing.T) {
	oldAppDir := store.AppDir
	oldDisabled := durableContinuationPersistenceDisabled
	oldMetadataDisabled := continuationMetadataPersistenceDisabled

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = false
	continuationMetadataPersistenceDisabled = true

	t.Cleanup(func() {
		store.AppDir = oldAppDir
		durableContinuationPersistenceDisabled = oldDisabled
		continuationMetadataPersistenceDisabled = oldMetadataDisabled
		clearCopilotContinuationCachesOnly()
	})

	ctx := newCopilotTurnContext()
	storeResponseTurnContext("acct-1", "resp_prev", ctx)
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)
	storeMessageToolCallTurnContext("acct-1", []string{"tool_1"}, ctx)
	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"}},
	)
	flushDurableContinuationStateForTests()

	if _, err := os.Stat(continuationStateFile()); !os.IsNotExist(err) {
		t.Fatalf("expected no continuation snapshot write, stat err=%v", err)
	}
}

func TestContinuationMetadataSQLiteReloadsBindings(t *testing.T) {
	oldAppDir := store.AppDir
	oldDisabled := durableContinuationPersistenceDisabled
	oldMetadataDisabled := continuationMetadataPersistenceDisabled
	oldMachineID := copilotClientMachineID

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = true
	continuationMetadataPersistenceDisabled = false
	copilotClientMachineID = newCopilotTurnContext().InteractionID

	t.Cleanup(func() {
		store.AppDir = oldAppDir
		durableContinuationPersistenceDisabled = oldDisabled
		continuationMetadataPersistenceDisabled = oldMetadataDisabled
		copilotClientMachineID = oldMachineID
		clearCopilotContinuationCachesOnly()
	})

	ctx := newCopilotTurnContext()
	persistedMachineID := copilotClientMachineID
	storeResponseTurnContext("acct-1", "resp_prev", ctx)
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)
	storeMessageToolCallTurnContext("acct-1", []string{"tool_1"}, ctx)

	clearCopilotContinuationCachesOnly()
	copilotClientMachineID = "machine-reset-for-test"
	resetContinuationMetadataStoreForTests()

	turnRequest := buildResponsesTurnRequest("acct-1", "resp_prev", nil)
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected sqlite response turn to reload as agent continuation, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected sqlite response context %+v, got %+v", ctx, turnRequest.Context)
	}
	if copilotClientMachineID != persistedMachineID {
		t.Fatalf("expected sqlite machine ID %q, got %q", persistedMachineID, copilotClientMachineID)
	}

	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_1"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected sqlite function call account acct-1, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupMessageToolCallAccount([]string{"tool_1"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected sqlite tool account acct-1, got %q ok=%v", accountID, ok)
	}

	messagePayload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "tool_1", Content2: "ok"}}}},
	}
	messageTurn := buildMessagesTurnRequest("acct-1", messagePayload)
	if messageTurn.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected sqlite tool turn to reload as agent continuation, got %q", messageTurn.InteractionType)
	}
	if messageTurn.Context != ctx {
		t.Fatalf("expected sqlite tool context %+v, got %+v", ctx, messageTurn.Context)
	}
}

func TestContinuationMetadataSQLiteImportsLegacySnapshot(t *testing.T) {
	oldAppDir := store.AppDir
	oldDisabled := durableContinuationPersistenceDisabled
	oldMetadataDisabled := continuationMetadataPersistenceDisabled
	oldMachineID := copilotClientMachineID

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = false
	continuationMetadataPersistenceDisabled = true
	copilotClientMachineID = newCopilotTurnContext().InteractionID

	t.Cleanup(func() {
		store.AppDir = oldAppDir
		durableContinuationPersistenceDisabled = oldDisabled
		continuationMetadataPersistenceDisabled = oldMetadataDisabled
		copilotClientMachineID = oldMachineID
		clearCopilotContinuationCachesOnly()
	})

	ctx := newCopilotTurnContext()
	persistedMachineID := copilotClientMachineID
	now := time.Now()
	snapshot := durableContinuationState{
		Version:         durableContinuationStateVersion,
		SavedAt:         now,
		ClientMachineID: persistedMachineID,
		ResponseTurns: map[string]durableCopilotTurnCacheEntry{
			"resp_legacy": {AccountID: "acct-1", Context: ctx, CreatedAt: now, AccessedAt: now},
		},
		ResponseFunctionCallTurns: map[string]durableCopilotTurnCacheEntry{
			"call_legacy": {AccountID: "acct-1", Context: ctx, CreatedAt: now, AccessedAt: now},
		},
		MessageToolTurns: map[string]durableCopilotTurnCacheEntry{
			"tool_legacy": {AccountID: "acct-1", Context: ctx, CreatedAt: now, AccessedAt: now},
		},
		ResponsesReplay: map[string]durableResponsesReplayEntry{
			"resp_legacy": {AccountID: "acct-1", Input: []interface{}{map[string]interface{}{"role": "user", "content": "hi"}}, ReplayItems: []interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_legacy", "name": "shell", "arguments": "{}"}}, CreatedAt: now, AccessedAt: now},
		},
	}
	body, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("json.Marshal(snapshot): %v", err)
	}
	if err := writeGzipFile(continuationStateFile(), body); err != nil {
		t.Fatalf("writeGzipFile: %v", err)
	}

	clearCopilotContinuationCachesOnly()
	resetContinuationMetadataStoreForTests()
	copilotClientMachineID = "machine-reset-for-test"
	durableContinuationPersistenceDisabled = true
	continuationMetadataPersistenceDisabled = false

	if err := InitContinuationMetadataStore(); err != nil {
		t.Fatalf("InitContinuationMetadataStore: %v", err)
	}
	resetContinuationMetadataStoreForTests()

	if accountID, ok := LookupResponseTurnAccount("resp_legacy"); !ok || accountID != "acct-1" {
		t.Fatalf("expected imported response turn acct-1, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupResponseFunctionCallAccount([]string{"call_legacy"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected imported function call acct-1, got %q ok=%v", accountID, ok)
	}
	if accountID, ok := LookupMessageToolCallAccount([]string{"tool_legacy"}); !ok || accountID != "acct-1" {
		t.Fatalf("expected imported tool acct-1, got %q ok=%v", accountID, ok)
	}
	if copilotClientMachineID != persistedMachineID {
		t.Fatalf("expected imported machine ID %q, got %q", persistedMachineID, copilotClientMachineID)
	}
}

func TestResponsesReplayDuckDBRecoversAfterHotCacheMiss(t *testing.T) {
	oldAppDir := store.AppDir
	oldDisabled := durableContinuationPersistenceDisabled
	oldMetadataDisabled := continuationMetadataPersistenceDisabled
	oldArchiveDisabled := responsesReplayArchiveDisabled

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	clearCopilotContinuationCachesOnly()
	durableContinuationPersistenceDisabled = true
	continuationMetadataPersistenceDisabled = false
	responsesReplayArchiveDisabled = false

	t.Cleanup(func() {
		store.AppDir = oldAppDir
		durableContinuationPersistenceDisabled = oldDisabled
		continuationMetadataPersistenceDisabled = oldMetadataDisabled
		responsesReplayArchiveDisabled = oldArchiveDisabled
		clearCopilotContinuationCachesOnly()
		resetResponsesReplayArchiveStoreForTests()
	})

	if err := InitContinuationMetadataStore(); err != nil {
		t.Fatalf("InitContinuationMetadataStore: %v", err)
	}
	if err := InitResponsesReplayArchiveStore(); err != nil {
		t.Fatalf("InitResponsesReplayArchiveStore: %v", err)
	}
	if err := PingResponsesReplayArchiveStore(); err != nil {
		t.Fatalf("PingResponsesReplayArchiveStore: %v", err)
	}

	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"}},
	)

	responsesReplayCache.mu.Lock()
	responsesReplayCache.entries = map[string]*responsesReplayEntry{}
	responsesReplayCache.mu.Unlock()

	if accountID, ok := LookupResponsesReplayAccount("resp_prev"); !ok || accountID != "acct-1" {
		t.Fatalf("expected sqlite replay metadata acct-1, got %q ok=%v", accountID, ok)
	}
	if !CanReplayResponsesContinuation("acct-1", "resp_prev") {
		t.Fatal("expected replay continuation to be recoverable after hot-cache miss")
	}

	payload := map[string]interface{}{
		"previous_response_id": "resp_prev",
		"input":                []interface{}{map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "ok"}},
	}
	if err := rewritePreviousResponseContinuation("acct-1", payload); err != nil {
		t.Fatalf("expected duckdb replay rewrite to succeed, got %v", err)
	}
	if _, hasPrev := payload["previous_response_id"]; hasPrev {
		t.Fatal("expected previous_response_id removed after duckdb-backed rewrite")
	}
	input, ok := payload["input"].([]interface{})
	if !ok || len(input) != 3 {
		t.Fatalf("expected rebuilt input of len 3, got %#v", payload["input"])
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
