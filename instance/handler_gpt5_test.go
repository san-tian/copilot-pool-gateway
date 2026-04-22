package instance

import (
	"encoding/json"
	"strings"
	"testing"

	"copilot-go/config"
)

func TestNormalizeCompletionsPayloadUsesMaxCompletionTokensForGpt5(t *testing.T) {
	state := config.NewState()
	body := []byte(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hi"}],"max_tokens":1}`)

	normalized, _, _, err := normalizeCompletionsPayload(state, body)
	if err != nil {
		t.Fatalf("normalizeCompletionsPayload returned error: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatalf("unmarshal normalized payload: %v", err)
	}
	if _, ok := payload["max_tokens"]; ok {
		t.Fatalf("expected max_tokens to be removed for GPT-5 payload, got %#v", payload)
	}
	if got, ok := payload["max_completion_tokens"].(float64); !ok || got != 1 {
		t.Fatalf("expected max_completion_tokens=1, got %#v", payload)
	}
}

func TestNormalizeCompletionsPayloadRejectsExplicitParallelToolCalls(t *testing.T) {
	state := config.NewState()
	body := []byte(`{"model":"gpt-4o","parallel_tool_calls":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]}`)

	_, _, _, err := normalizeCompletionsPayload(state, body)
	if err == nil {
		t.Fatal("expected explicit parallel_tool_calls=true to be rejected")
	}
	if !strings.Contains(err.Error(), "parallel_tool_calls") {
		t.Fatalf("expected parallel_tool_calls error, got %v", err)
	}
}

func TestNormalizeCompletionsPayloadDefaultsToolsToSerial(t *testing.T) {
	state := config.NewState()
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]}`)

	normalized, _, _, err := normalizeCompletionsPayload(state, body)
	if err != nil {
		t.Fatalf("normalizeCompletionsPayload returned error: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatalf("unmarshal normalized payload: %v", err)
	}
	if got, ok := payload["parallel_tool_calls"].(bool); !ok || got {
		t.Fatalf("expected parallel_tool_calls=false by default, got %#v", payload["parallel_tool_calls"])
	}
}
