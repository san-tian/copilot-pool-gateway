package instance

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"copilot-go/anthropic"
	"copilot-go/config"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func resetCopilotTurnCaches() {
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
}

func withStreamingClient(t *testing.T, client *http.Client) {
	t.Helper()
	clientMu.Lock()
	prevStreaming := streamingHTTPClient
	prevDefault := defaultHTTPClient
	streamingHTTPClient = client
	defaultHTTPClient = client
	clientMu.Unlock()
	t.Cleanup(func() {
		clientMu.Lock()
		streamingHTTPClient = prevStreaming
		defaultHTTPClient = prevDefault
		clientMu.Unlock()
	})
}

func withWorkerClient(t *testing.T, client *http.Client) {
	t.Helper()
	prevOnce := workerClientOnce
	prevClient := workerHTTPClient
	workerClientOnce = sync.Once{}
	workerHTTPClient = client
	workerClientOnce.Do(func() {})
	t.Cleanup(func() {
		workerClientOnce = prevOnce
		workerHTTPClient = prevClient
	})
}

func setupWorkerStoreEnv(t *testing.T) {
	t.Helper()
	oldAppDir := store.AppDir
	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	t.Cleanup(func() {
		store.AppDir = oldAppDir
	})
}

func testState() *config.State {
	return &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
	}
}

func TestBuildResponsesTurnRequestReusesStoredContext(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseTurnContext("acct-1", "resp_1", ctx)

	turnRequest := buildResponsesTurnRequest("acct-1", "resp_1", nil)
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected agent continuation, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Initiator != "agent" {
		t.Fatalf("expected agent initiator, got %q", turnRequest.Initiator)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}

	headers := turnRequest.Headers()
	assertHeaderValue(t, headers, "X-Interaction-Type", copilotInteractionTypeAgent)
	assertHeaderValue(t, headers, "X-Interaction-Id", ctx.InteractionID)
	assertHeaderValue(t, headers, "X-Client-Session-Id", ctx.ClientSessionID)
	assertHeaderValue(t, headers, "X-Agent-Task-Id", ctx.AgentTaskID)
}

func TestBuildResponsesTurnRequestUsesFunctionCallOutputContextWithoutPreviousResponseID(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1", "call_2"}, ctx)

	turnRequest := buildResponsesTurnRequest("acct-1", "", []interface{}{
		map[string]interface{}{"type": "function_call_output", "call_id": "call_2", "output": "ok"},
	})
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected agent continuation, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Initiator != "agent" {
		t.Fatalf("expected agent initiator, got %q", turnRequest.Initiator)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
}

func TestDoResponsesProxyEmitsContinuationHeaders(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseTurnContext("acct-1", "resp_prev", ctx)
	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "ok"}},
	)

	var gotReq *http.Request
	withStreamingClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotReq = req.Clone(req.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_next","output":[]}`)),
		}, nil
	})})

	body := []byte(`{"model":"gpt-4o","input":[{"role":"user","content":"next"}],"previous_response_id":"resp_prev"}`)
	resp, forwardedBody, turnRequest, err := DoResponsesProxy("acct-1", testState(), body)
	if err != nil {
		t.Fatalf("DoResponsesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if gotReq == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if gotReq.URL.Path != "/responses" {
		t.Fatalf("expected /responses path, got %s", gotReq.URL.Path)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
	assertHeaderValue(t, gotReq.Header, "X-Initiator", "agent")
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Id", ctx.InteractionID)
	assertHeaderValue(t, gotReq.Header, "X-Client-Session-Id", ctx.ClientSessionID)
	assertHeaderValue(t, gotReq.Header, "X-Agent-Task-Id", ctx.AgentTaskID)
	assertHeaderValue(t, gotReq.Header, "X-Client-Machine-Id", copilotClientMachineID)
	if bytes.Contains(forwardedBody, []byte(`"previous_response_id"`)) {
		t.Fatalf("expected forwarded body to drop previous_response_id after rewrite, got %s", string(forwardedBody))
	}
	var forwardedPayload map[string]interface{}
	if err := json.Unmarshal(forwardedBody, &forwardedPayload); err != nil {
		t.Fatalf("failed to parse forwarded body: %v", err)
	}
	input, ok := forwardedPayload["input"].([]interface{})
	if !ok || len(input) != 3 {
		t.Fatalf("expected rewritten input with 3 items, got %#v", forwardedPayload["input"])
	}
}

func TestDoResponsesProxyUsesFunctionCallOutputContextWithoutPreviousResponseID(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)

	var gotReq *http.Request
	withStreamingClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotReq = req.Clone(req.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_next","output":[]}`)),
		}, nil
	})})

	body := []byte(`{"model":"gpt-4o","input":[{"type":"function_call_output","call_id":"call_1","output":"ok"}]}`)
	resp, _, turnRequest, err := DoResponsesProxy("acct-1", testState(), body)
	if err != nil {
		t.Fatalf("DoResponsesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if gotReq == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
	assertHeaderValue(t, gotReq.Header, "X-Initiator", "agent")
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Id", ctx.InteractionID)
}

func TestDoResponsesProxyViaWorkerForwardsContinuationHeaders(t *testing.T) {
	resetCopilotTurnCaches()
	setupWorkerStoreEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	if _, err := store.UpdateAccount(account.ID, map[string]interface{}{"workerUrl": "http://worker.local"}); err != nil {
		t.Fatalf("UpdateAccount workerUrl: %v", err)
	}

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext(account.ID, []string{"call_1"}, ctx)

	var gotReq *http.Request
	withWorkerClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotReq = req.Clone(req.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_next","output":[]}`)),
		}, nil
	})})

	resp, _, turnRequest, err := DoResponsesProxy(account.ID, testState(), []byte(`{"model":"gpt-5.4","input":[{"type":"function_call_output","call_id":"call_1","output":"ok"}]}`))
	if err != nil {
		t.Fatalf("DoResponsesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if gotReq == nil {
		t.Fatal("expected worker request to be captured")
	}
	if gotReq.URL.Path != "/v1/responses" {
		t.Fatalf("expected /v1/responses path, got %s", gotReq.URL.Path)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
	assertHeaderValue(t, gotReq.Header, "X-Initiator", "agent")
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Id", ctx.InteractionID)
	assertHeaderValue(t, gotReq.Header, "X-Client-Session-Id", ctx.ClientSessionID)
	assertHeaderValue(t, gotReq.Header, "X-Agent-Task-Id", ctx.AgentTaskID)
	assertHeaderValue(t, gotReq.Header, "X-Client-Machine-Id", copilotClientMachineID)
}

func TestDoResponsesProxyPreservesParallelToolCalls(t *testing.T) {
	resetCopilotTurnCaches()

	var gotReq *http.Request
	withStreamingClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotReq = req.Clone(req.Context())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_parallel","output":[]}`)),
		}, nil
	})})

	body := []byte(`{"model":"gpt-4o","parallel_tool_calls":true,"input":"use both tools","tools":[{"type":"function","name":"weather","parameters":{"type":"object"}},{"type":"function","name":"time","parameters":{"type":"object"}}]}`)
	resp, forwardedBody, _, err := DoResponsesProxy("acct-1", testState(), body)
	if err != nil {
		t.Fatalf("DoResponsesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if gotReq == nil {
		t.Fatal("expected upstream request to be captured")
	}
	var forwardedPayload map[string]interface{}
	if err := json.Unmarshal(forwardedBody, &forwardedPayload); err != nil {
		t.Fatalf("failed to parse forwarded body: %v", err)
	}
	if got, ok := forwardedPayload["parallel_tool_calls"].(bool); !ok || !got {
		t.Fatalf("expected forwarded payload to preserve parallel_tool_calls=true, got %#v", forwardedPayload["parallel_tool_calls"])
	}
}

func TestBuildMessagesTurnRequestUsesToolResultContext(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeMessageToolCallTurnContext("acct-1", []string{"call_1", "call_2"}, ctx)

	payload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{
			{Role: "assistant", Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "call_1", Name: "shell"}}},
			{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "call_1", Content2: "ok"}}},
		},
	}

	turnRequest := buildMessagesTurnRequest("acct-1", payload)
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("expected agent continuation, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
}

func TestDoMessagesProxyEmitsToolContinuationHeaders(t *testing.T) {
	resetCopilotTurnCaches()
	gin.SetMode(gin.TestMode)

	ctx := newCopilotTurnContext()
	storeMessageToolCallTurnContext("acct-1", []string{"call_1"}, ctx)

	var gotReq *http.Request
	var gotBody []byte
	withStreamingClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotReq = req.Clone(req.Context())
		gotBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"done"}}]}`)),
		}, nil
	})})

	payload := anthropic.AnthropicMessagesPayload{
		Model:     "claude-sonnet-4-5",
		MaxTokens: 64,
		Messages: []anthropic.AnthropicMessage{
			{Role: "assistant", Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "call_1", Name: "shell"}}},
			{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "call_1", Content2: "ok"}}},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))

	resp, turnRequest, err := DoMessagesProxy(c, "acct-1", testState(), body)
	if err != nil {
		t.Fatalf("DoMessagesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if gotReq == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if gotReq.URL.Path != "/chat/completions" {
		t.Fatalf("expected /chat/completions path, got %s", gotReq.URL.Path)
	}
	if turnRequest.Context != ctx {
		t.Fatalf("expected stored context %+v, got %+v", ctx, turnRequest.Context)
	}
	assertHeaderValue(t, gotReq.Header, "X-Initiator", "agent")
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
	assertHeaderValue(t, gotReq.Header, "X-Interaction-Id", ctx.InteractionID)
	assertHeaderValue(t, gotReq.Header, "X-Client-Session-Id", ctx.ClientSessionID)
	assertHeaderValue(t, gotReq.Header, "X-Agent-Task-Id", ctx.AgentTaskID)
	if !bytes.Contains(gotBody, []byte(`"tool_call_id":"call_1"`)) {
		t.Fatalf("expected translated OpenAI payload to include tool call linkage, got %s", string(gotBody))
	}
}

func TestDoMessagesProxyUsesMaxCompletionTokensForGpt5(t *testing.T) {
	resetCopilotTurnCaches()
	gin.SetMode(gin.TestMode)

	var gotBody []byte
	withStreamingClient(t, &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"done"}}]}`)),
		}, nil
	})})

	payload := anthropic.AnthropicMessagesPayload{
		Model:     "gpt-5.4",
		MaxTokens: 64,
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "hello"},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))

	resp, _, err := DoMessagesProxy(c, "acct-1", testState(), body)
	if err != nil {
		t.Fatalf("DoMessagesProxy returned error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var forwarded map[string]interface{}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatalf("failed to unmarshal forwarded body: %v", err)
	}
	if _, ok := forwarded["max_tokens"]; ok {
		t.Fatalf("expected max_tokens to be removed for gpt-5 messages proxy, got %s", string(gotBody))
	}
	if got, ok := forwarded["max_completion_tokens"].(float64); !ok || got != 64 {
		t.Fatalf("expected max_completion_tokens=64, got %#v from %s", forwarded["max_completion_tokens"], string(gotBody))
	}
}

func TestCurrentCompletionsInitiatorTracksTrailingMessage(t *testing.T) {
	tests := []struct {
		name     string
		messages []interface{}
		want     string
	}{
		{
			name: "new user turn with history",
			messages: []interface{}{
				map[string]interface{}{"role": "user", "content": "first"},
				map[string]interface{}{"role": "assistant", "content": "reply"},
				map[string]interface{}{"role": "user", "content": "next question"},
			},
			want: "user",
		},
		{
			name: "tool continuation",
			messages: []interface{}{
				map[string]interface{}{"role": "user", "content": "first"},
				map[string]interface{}{"role": "assistant", "tool_calls": []interface{}{map[string]interface{}{"id": "call_1"}}},
				map[string]interface{}{"role": "tool", "tool_call_id": "call_1", "content": "ok"},
			},
			want: "agent",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := currentCompletionsInitiator(tc.messages); got != tc.want {
				t.Fatalf("expected %q, got %q", tc.want, got)
			}
		})
	}
}

func TestCollectToolCallIDsFromChatCompletion(t *testing.T) {
	resp := anthropic.ChatCompletionResponse{
		Choices: []anthropic.Choice{{
			Message: &anthropic.ChoiceMsg{
				ToolCalls: []anthropic.ToolCall{{ID: "call_1"}, {ID: "call_1"}, {ID: "call_2"}},
			},
		}},
	}

	got := collectToolCallIDsFromChatCompletion(resp)
	want := []string{"call_1", "call_2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func assertHeaderValue(t *testing.T, headers http.Header, key string, want string) {
	t.Helper()
	if got := headers.Get(key); got != want {
		t.Fatalf("expected %s=%q, got %q", key, want, got)
	}
}

func TestBuildResponsesTurnRequestFallsBackToFreshUserTurnWhenContextMissing(t *testing.T) {
	resetCopilotTurnCaches()

	turnRequest := buildResponsesTurnRequest("acct-1", "resp_missing", nil)
	if turnRequest.InteractionType != copilotInteractionTypeUser {
		t.Fatalf("expected fresh user turn, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Initiator != "user" {
		t.Fatalf("expected user initiator, got %q", turnRequest.Initiator)
	}
}

func TestBuildMessagesTurnRequestFallsBackToFreshUserTurnWhenContextMissing(t *testing.T) {
	resetCopilotTurnCaches()

	payload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{
			{Role: "assistant", Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "call_missing", Name: "shell"}}},
			{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "call_missing", Content2: "ok"}}},
		},
	}

	turnRequest := buildMessagesTurnRequest("acct-1", payload)
	if turnRequest.InteractionType != copilotInteractionTypeUser {
		t.Fatalf("expected fresh user turn, got %q", turnRequest.InteractionType)
	}
	if turnRequest.Initiator != "user" {
		t.Fatalf("expected user initiator, got %q", turnRequest.Initiator)
	}
}

func TestRewritePreviousResponseContinuationDropsItemReferences(t *testing.T) {
	resetCopilotTurnCaches()

	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"}},
	)

	payload := map[string]interface{}{
		"previous_response_id": "resp_prev",
		"input": []interface{}{
			map[string]interface{}{"type": "item_reference", "id": "msg_deadbeef"},
			map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "ok", "id": "out_123"},
		},
	}

	if err := rewritePreviousResponseContinuation("acct-1", payload); err != nil {
		t.Fatalf("rewritePreviousResponseContinuation returned error: %v", err)
	}
	input, ok := payload["input"].([]interface{})
	if !ok {
		t.Fatalf("expected rewritten input slice, got %#v", payload["input"])
	}
	if len(input) != 3 {
		t.Fatalf("expected 3 replay items after dropping references, got %#v", input)
	}
	last, ok := input[len(input)-1].(map[string]interface{})
	if !ok {
		t.Fatalf("expected last replay item to be map, got %#v", input[len(input)-1])
	}
	if _, hasID := last["id"]; hasID {
		t.Fatalf("expected continuation item id to be scrubbed, got %#v", last)
	}
}

func TestLookupResponseFunctionCallAccountReturnsStoredAccount(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)

	accountID, ok := LookupResponseFunctionCallAccount([]string{"call_1"})
	if !ok {
		t.Fatal("expected response function call account lookup to succeed")
	}
	if accountID != "acct-1" {
		t.Fatalf("expected account acct-1, got %q", accountID)
	}
}

func TestResolveResponseFunctionCallSessionCanonical(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1", "call_2", "call_3"}, ctx)

	result := ResolveResponseFunctionCallSession([]string{"call_1", "call_2", "call_3"})
	if result.Kind != SessionCanonical {
		t.Fatalf("expected SessionCanonical, got %v", result.Kind)
	}
	if result.AccountID != "acct-1" {
		t.Fatalf("expected AccountID acct-1, got %q", result.AccountID)
	}
	if result.HitCount != 3 || result.MissCount != 0 {
		t.Fatalf("expected hit=3 miss=0, got hit=%d miss=%d", result.HitCount, result.MissCount)
	}
}

func TestResolveResponseFunctionCallSessionCanonicalPartialMiss(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1"}, ctx)

	// call_2 and call_3 were evicted / never stored — canonical still wins
	// because every hit agrees on acct-1.
	result := ResolveResponseFunctionCallSession([]string{"call_1", "call_2", "call_3"})
	if result.Kind != SessionCanonical {
		t.Fatalf("expected SessionCanonical on partial miss, got %v", result.Kind)
	}
	if result.AccountID != "acct-1" {
		t.Fatalf("expected AccountID acct-1, got %q", result.AccountID)
	}
	if result.HitCount != 1 || result.MissCount != 2 {
		t.Fatalf("expected hit=1 miss=2, got hit=%d miss=%d", result.HitCount, result.MissCount)
	}
}

func TestResolveResponseFunctionCallSessionSplit(t *testing.T) {
	resetCopilotTurnCaches()

	ctx1 := newCopilotTurnContext()
	ctx2 := newCopilotTurnContext()
	storeResponseFunctionCallTurnContext("acct-1", []string{"call_1", "call_2"}, ctx1)
	storeResponseFunctionCallTurnContext("acct-2", []string{"call_3"}, ctx2)

	result := ResolveResponseFunctionCallSession([]string{"call_1", "call_2", "call_3"})
	if result.Kind != SessionSplit {
		t.Fatalf("expected SessionSplit, got %v", result.Kind)
	}
	if result.AccountID != "" {
		t.Fatalf("expected AccountID empty for split, got %q", result.AccountID)
	}
	if len(result.SplitAccounts) != 2 {
		t.Fatalf("expected 2 split accounts, got %v", result.SplitAccounts)
	}
	// Split accounts must be sorted for stable diagnostics.
	if result.SplitAccounts[0] != "acct-1" || result.SplitAccounts[1] != "acct-2" {
		t.Fatalf("expected split accounts sorted [acct-1 acct-2], got %v", result.SplitAccounts)
	}
	if result.HitCount != 3 || result.MissCount != 0 {
		t.Fatalf("expected hit=3 miss=0, got hit=%d miss=%d", result.HitCount, result.MissCount)
	}
}

func TestResolveResponseFunctionCallSessionOrphan(t *testing.T) {
	resetCopilotTurnCaches()

	result := ResolveResponseFunctionCallSession([]string{"call_unknown_1", "call_unknown_2"})
	if result.Kind != SessionOrphan {
		t.Fatalf("expected SessionOrphan, got %v", result.Kind)
	}
	if result.AccountID != "" {
		t.Fatalf("expected AccountID empty for orphan, got %q", result.AccountID)
	}
	if result.HitCount != 0 || result.MissCount != 2 {
		t.Fatalf("expected hit=0 miss=2, got hit=%d miss=%d", result.HitCount, result.MissCount)
	}
}

func TestResolveResponseFunctionCallSessionEmptyInput(t *testing.T) {
	resetCopilotTurnCaches()

	result := ResolveResponseFunctionCallSession(nil)
	if result.Kind != SessionOrphan {
		t.Fatalf("expected SessionOrphan on nil input, got %v", result.Kind)
	}

	result = ResolveResponseFunctionCallSession([]string{"", "  "})
	if result.Kind != SessionOrphan {
		t.Fatalf("expected SessionOrphan on whitespace-only input, got %v", result.Kind)
	}
}

func TestStashResponseFunctionCallTurnContextInMemoryBinds(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	stashResponseFunctionCallTurnContextInMemory("acct-1", []string{"call_1", "call_2"}, ctx)

	result := ResolveResponseFunctionCallSession([]string{"call_1", "call_2"})
	if result.Kind != SessionCanonical || result.AccountID != "acct-1" {
		t.Fatalf("expected canonical acct-1 after stash, got kind=%v account=%q", result.Kind, result.AccountID)
	}
	// Confirm the stored context is recoverable (same as the full store path).
	recovered, ok := loadResponseFunctionCallTurnContext("acct-1", []string{"call_1"})
	if !ok {
		t.Fatal("expected stashed context to be loadable")
	}
	if recovered != ctx {
		t.Fatalf("expected recovered ctx %+v, got %+v", ctx, recovered)
	}
}

func TestLookupResponseTurnAccountReturnsStoredAccount(t *testing.T) {
	resetCopilotTurnCaches()

	ctx := newCopilotTurnContext()
	storeResponseTurnContext("acct-1", "resp_1", ctx)

	accountID, ok := LookupResponseTurnAccount("resp_1")
	if !ok {
		t.Fatal("expected response turn account lookup to succeed")
	}
	if accountID != "acct-1" {
		t.Fatalf("expected account acct-1, got %q", accountID)
	}
}

func TestCanReplayResponsesContinuationRequiresMatchingAccount(t *testing.T) {
	resetCopilotTurnCaches()

	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"}},
	)

	if !CanReplayResponsesContinuation("acct-1", "resp_prev") {
		t.Fatal("expected replay continuation to be available for matching account")
	}
	if CanReplayResponsesContinuation("acct-2", "resp_prev") {
		t.Fatal("expected replay continuation to be unavailable for different account")
	}
}
