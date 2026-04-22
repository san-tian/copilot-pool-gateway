package anthropic

import "testing"

func TestTranslateToOpenAIDisablesParallelToolCallsWhenToolsPresent(t *testing.T) {
	payload := AnthropicMessagesPayload{
		Model:    "claude-sonnet-4-5",
		Messages: []AnthropicMessage{{Role: "user", Content: "check weather"}},
		Tools: []AnthropicTool{{
			Name:        "weather",
			Description: "Return weather",
			InputSchema: map[string]interface{}{"type": "object"},
		}},
	}

	translated := TranslateToOpenAI(payload)
	if translated.ParallelToolCalls == nil {
		t.Fatal("expected ParallelToolCalls to be set")
	}
	if *translated.ParallelToolCalls {
		t.Fatal("expected ParallelToolCalls to be false")
	}
}
