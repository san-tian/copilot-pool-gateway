package instance

import (
	"strings"
	"testing"

	"copilot-go/anthropic"
)

func TestValidateAnthropicToolContinuationRejectsMissingToolResults(t *testing.T) {
	payload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "check weather"},
			{Role: "assistant", Content: []anthropic.ContentBlock{
				{Type: "tool_use", ID: "call_1", Name: "weather", Input: map[string]interface{}{"city": "Paris"}},
				{Type: "tool_use", ID: "call_2", Name: "time", Input: map[string]interface{}{"city": "Paris"}},
			}},
			{Role: "user", Content: []anthropic.ContentBlock{{Type: "tool_result", ToolUseID: "call_1", Content2: "sunny"}}},
		},
	}

	err := validateAnthropicToolContinuation(payload)
	if err == nil {
		t.Fatal("expected missing tool_result validation error")
	}
	if !strings.Contains(err.Error(), "call_2") {
		t.Fatalf("expected error to mention missing call_2, got %v", err)
	}
}

func TestValidateAnthropicToolContinuationAcceptsMatchingToolResults(t *testing.T) {
	payload := anthropic.AnthropicMessagesPayload{
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "check weather"},
			{Role: "assistant", Content: []anthropic.ContentBlock{
				{Type: "tool_use", ID: "call_1", Name: "weather", Input: map[string]interface{}{"city": "Paris"}},
				{Type: "tool_use", ID: "call_2", Name: "time", Input: map[string]interface{}{"city": "Paris"}},
			}},
			{Role: "user", Content: []anthropic.ContentBlock{
				{Type: "tool_result", ToolUseID: "call_1", Content2: "sunny"},
				{Type: "tool_result", ToolUseID: "call_2", Content2: "10:00"},
			}},
		},
	}

	if err := validateAnthropicToolContinuation(payload); err != nil {
		t.Fatalf("expected matching tool_result continuation to pass, got %v", err)
	}
}

func TestRewritePreviousResponseContinuationRejectsMissingToolOutputs(t *testing.T) {
	resetCopilotTurnCaches()

	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{
			map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{}"},
			map[string]interface{}{"type": "function_call", "call_id": "call_2", "name": "time", "arguments": "{}"},
		},
	)

	payload := map[string]interface{}{
		"previous_response_id": "resp_prev",
		"input": []interface{}{
			map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
		},
	}

	err := rewritePreviousResponseContinuation("acct-1", payload)
	if err == nil {
		t.Fatal("expected missing function_call_output validation error")
	}
	if !strings.Contains(err.Error(), "call_2") {
		t.Fatalf("expected error to mention missing call_2, got %v", err)
	}
}

func TestRewritePreviousResponseContinuationAcceptsMatchingToolOutputs(t *testing.T) {
	resetCopilotTurnCaches()

	storeResponsesReplay(
		"acct-1",
		"resp_prev",
		[]interface{}{map[string]interface{}{"role": "user", "content": "first"}},
		[]interface{}{
			map[string]interface{}{"type": "reasoning", "summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "thinking"}}},
			map[string]interface{}{"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{}"},
			map[string]interface{}{"type": "function_call", "call_id": "call_2", "name": "time", "arguments": "{}"},
		},
	)

	payload := map[string]interface{}{
		"previous_response_id": "resp_prev",
		"input": []interface{}{
			map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "sunny", "id": "out_1"},
			map[string]interface{}{"type": "function_call_output", "call_id": "call_2", "output": "10:00", "id": "out_2"},
		},
	}

	if err := rewritePreviousResponseContinuation("acct-1", payload); err != nil {
		t.Fatalf("expected matching function_call_output continuation to pass, got %v", err)
	}
	if _, hasPrev := payload["previous_response_id"]; hasPrev {
		t.Fatal("expected previous_response_id to be removed after rewrite")
	}
	input, ok := payload["input"].([]interface{})
	if !ok {
		t.Fatalf("expected rewritten input slice, got %#v", payload["input"])
	}
	if len(input) != 6 {
		t.Fatalf("expected 6 replay items, got %#v", input)
	}
	last, ok := input[len(input)-1].(map[string]interface{})
	if !ok {
		t.Fatalf("expected last replay item to be map, got %#v", input[len(input)-1])
	}
	if last["call_id"] != "call_2" {
		t.Fatalf("expected final function_call_output to keep call_2, got %#v", last)
	}
	if _, hasID := last["id"]; hasID {
		t.Fatalf("expected scrubbed continuation item without id, got %#v", last)
	}
}
