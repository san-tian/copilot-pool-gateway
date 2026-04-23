package orphan_translate

import (
	"encoding/json"
	"strings"
	"testing"
)

func mustTranslateMessages(t *testing.T, body map[string]interface{}) (map[string]interface{}, TranslateStats) {
	t.Helper()
	out, stats, err := ResponsesToMessages(mustMarshal(t, body))
	if err != nil {
		t.Fatalf("ResponsesToMessages: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal messages body: %v", err)
	}
	return parsed, stats
}

func TestResponsesToMessages_InstructionsBecomeTopLevelSystem(t *testing.T) {
	body := map[string]interface{}{
		"model":        "gpt-5",
		"instructions": "You are a helpful assistant.",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	if msg["system"] != "You are a helpful assistant." {
		t.Fatalf("system should be top-level string, got %+v", msg["system"])
	}
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d", len(msgs))
	}
	only := msgs[0].(map[string]interface{})
	if only["role"] != "user" || only["content"] != "hi" {
		t.Fatalf("user message wrong: %+v", only)
	}
}

func TestResponsesToMessages_SystemRoleFoldsIntoUser(t *testing.T) {
	// Anthropic has no per-turn system role — items with role=system should
	// be folded into user turns so their content isn't lost.
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "system", "content": "extra guidance"},
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages, got %d: %+v", len(msgs), msgs)
	}
	first := msgs[0].(map[string]interface{})
	if first["role"] != "user" || first["content"] != "extra guidance" {
		t.Fatalf("system folded to user wrong: %+v", first)
	}
}

func TestResponsesToMessages_FunctionCallFlattenedAfterRealAssistant(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "list files"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": "Sure."},
			map[string]interface{}{"type": "function_call", "call_id": "fc_abc", "name": "Read", "arguments": `{"p":"a"}`},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_abc", "output": "content"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d: %+v", len(msgs), msgs)
	}

	realAsst := msgs[1].(map[string]interface{})
	if realAsst["role"] != "assistant" || realAsst["content"] != "Sure." {
		t.Fatalf("real assistant should pass through verbatim, got %+v", realAsst)
	}

	synAsst := msgs[2].(map[string]interface{})
	if synAsst["role"] != "assistant" || synAsst["content"] != `[Tool call: Read({"p":"a"})]` {
		t.Fatalf("synthetic assistant wrong: %+v", synAsst)
	}
	for _, k := range []string{"tool_use", "tool_calls"} {
		if _, leaked := synAsst[k]; leaked {
			t.Fatalf("flattened path must not emit %s, got %+v", k, synAsst)
		}
	}

	toolUser := msgs[3].(map[string]interface{})
	if toolUser["role"] != "user" {
		t.Fatalf("tail should be user (tool-result + continue), got %+v", toolUser)
	}
	tc := toolUser["content"].(string)
	if !strings.Contains(tc, "[Tool `Read` returned: content]") {
		t.Fatalf("tool-result marker missing: %q", tc)
	}
	if !strings.HasSuffix(tc, "continue") {
		t.Fatalf("trailing function_call_output should trigger continue, got: %q", tc)
	}
}

func TestResponsesToMessages_MultipleFunctionCallsStackIntoOneAssistantTurn(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "do two things"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "A", "arguments": "{}"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_2", "name": "B", "arguments": "{}"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_1", "output": "a"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_2", "output": "b"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("want 3 messages, got %d: %+v", len(msgs), msgs)
	}
	synAsst := msgs[1].(map[string]interface{})
	if synAsst["role"] != "assistant" {
		t.Fatalf("want merged synthetic assistant at idx 1, got %+v", synAsst)
	}
	sc := synAsst["content"].(string)
	if !strings.Contains(sc, "[Tool call: A({})]") || !strings.Contains(sc, "[Tool call: B({})]") {
		t.Fatalf("adjacent tool_calls should merge: %q", sc)
	}
	tail := msgs[2].(map[string]interface{})
	tc := tail["content"].(string)
	if !strings.Contains(tc, "[Tool `A` returned: a]") || !strings.Contains(tc, "[Tool `B` returned: b]") {
		t.Fatalf("adjacent tool_results should merge: %q", tc)
	}
	if !strings.HasSuffix(tc, "continue") {
		t.Fatalf("tail should end with continue: %q", tc)
	}
}

func TestResponsesToMessages_TailFunctionCallOutputSynthesizesContinue(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_1", "output": "done"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	tail := msgs[len(msgs)-1].(map[string]interface{})
	if tail["role"] != "user" {
		t.Fatalf("tail must be user, got %+v", tail)
	}
	content := tail["content"].(string)
	if !strings.HasSuffix(content, "continue") {
		t.Fatalf("continue suffix missing, got %q", content)
	}
	if !strings.Contains(content, "[Tool `N` returned: done]") {
		t.Fatalf("continue should merge into the existing tool-result turn: %q", content)
	}
}

func TestResponsesToMessages_TailRealUserMessageUnchanged(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "just say hi"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d: %+v", len(msgs), msgs)
	}
	only := msgs[0].(map[string]interface{})
	if only["role"] != "user" || only["content"] != "just say hi" {
		t.Fatalf("real user passes through verbatim: %+v", only)
	}
}

func TestResponsesToMessages_DanglingFunctionCallAppendsUserTurn(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("want 3 messages (user, synth assistant, continue user), got %d: %+v", len(msgs), msgs)
	}
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("tail should be fresh user 'continue', got %+v", tail)
	}
}

func TestResponsesToMessages_ReasoningDropped(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
			map[string]interface{}{"type": "reasoning", "encrypted_content": "abcdef"},
			map[string]interface{}{"type": "reasoning", "summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "thought"}}},
			map[string]interface{}{"type": "message", "role": "user", "content": "again"},
		},
	}
	_, stats := mustTranslateMessages(t, body)
	if stats.DroppedReasoning != 2 {
		t.Fatalf("dropped reasoning want 2 got %d", stats.DroppedReasoning)
	}
}

func TestResponsesToMessages_ToolsTranslatedToAnthropicShape(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
		"tools": []interface{}{
			map[string]interface{}{
				"type":        "function",
				"name":        "Read",
				"description": "Read a file",
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"path": map[string]interface{}{"type": "string"}},
					"required":   []interface{}{"path"},
				},
			},
			map[string]interface{}{"type": "web_search"},
			map[string]interface{}{"type": "image_generation"},
		},
	}
	msg, stats := mustTranslateMessages(t, body)
	if stats.ToolsIn != 3 || stats.ToolsOut != 1 || stats.DroppedTools != 2 {
		t.Fatalf("tool stats wrong: %+v", stats)
	}
	tools := msg["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("want 1 anthropic tool, got %d", len(tools))
	}
	first := tools[0].(map[string]interface{})
	if first["name"] != "Read" || first["description"] != "Read a file" {
		t.Fatalf("tool fields wrong: %+v", first)
	}
	if first["input_schema"] == nil {
		t.Fatalf("input_schema missing: %+v", first)
	}
}

func TestResponsesToMessages_ToolChoiceFunction(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
		"tool_choice": map[string]interface{}{"type": "function", "name": "Read"},
	}
	msg, _ := mustTranslateMessages(t, body)
	tc := msg["tool_choice"].(map[string]interface{})
	if tc["type"] != "tool" || tc["name"] != "Read" {
		t.Fatalf("tool_choice should translate to {type:tool,name:Read}, got %+v", tc)
	}
}

func TestResponsesToMessages_ToolChoiceRequiredBecomesAny(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
		"tool_choice": "required",
	}
	msg, _ := mustTranslateMessages(t, body)
	tc := msg["tool_choice"].(map[string]interface{})
	if tc["type"] != "any" {
		t.Fatalf("required should become any, got %+v", tc)
	}
}

func TestResponsesToMessages_MaxOutputTokensMapsToMaxTokens(t *testing.T) {
	body := map[string]interface{}{
		"model":             "gpt-5",
		"max_output_tokens": 4096,
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	if msg["max_tokens"].(float64) != 4096 {
		t.Fatalf("max_tokens want 4096 got %v", msg["max_tokens"])
	}
}

func TestResponsesToMessages_MaxTokensDefaultsWhenUnset(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	// Anthropic requires max_tokens; default must be populated.
	if msg["max_tokens"].(float64) != 8192 {
		t.Fatalf("default max_tokens want 8192 got %v", msg["max_tokens"])
	}
}

func TestResponsesToMessages_ImageContentPreserved(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": "describe"},
					map[string]interface{}{"type": "input_image", "image_url": "https://example.com/a.png"},
				},
			},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	user := msgs[0].(map[string]interface{})
	parts, ok := user["content"].([]interface{})
	if !ok {
		t.Fatalf("image-bearing content should be parts array, got %T", user["content"])
	}
	if len(parts) != 2 {
		t.Fatalf("want 2 parts, got %d: %+v", len(parts), parts)
	}
	img := parts[1].(map[string]interface{})
	if img["type"] != "image" {
		t.Fatalf("image block type wrong: %+v", img)
	}
	src := img["source"].(map[string]interface{})
	if src["type"] != "url" || src["url"] != "https://example.com/a.png" {
		t.Fatalf("image source wrong: %+v", src)
	}
}

func TestResponsesToMessages_FunctionCallOutputNonString(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "fc_1",
				"output": map[string]interface{}{
					"result": "ok",
					"code":   0,
				},
			},
		},
	}
	msg, _ := mustTranslateMessages(t, body)
	msgs := msg["messages"].([]interface{})
	tail := msgs[len(msgs)-1].(map[string]interface{})
	content := tail["content"].(string)
	if !strings.Contains(content, `"result"`) || !strings.Contains(content, `"ok"`) {
		t.Fatalf("non-string output should JSON-stringify into content: %q", content)
	}
}
