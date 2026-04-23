package orphan_translate

import (
	"encoding/json"
	"strings"
	"testing"
)

func mustMarshal(t *testing.T, v interface{}) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func mustTranslate(t *testing.T, body map[string]interface{}) (map[string]interface{}, TranslateStats) {
	t.Helper()
	out, stats, err := ResponsesToChat(mustMarshal(t, body))
	if err != nil {
		t.Fatalf("ResponsesToChat: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal chat body: %v", err)
	}
	return parsed, stats
}

func TestResponsesToChat_InstructionsBecomeSystem(t *testing.T) {
	body := map[string]interface{}{
		"model":        "gpt-5",
		"instructions": "You are a helpful assistant.",
		"input": []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": "hi"},
				},
			},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages, got %d", len(msgs))
	}
	first := msgs[0].(map[string]interface{})
	if first["role"] != "system" || first["content"] != "You are a helpful assistant." {
		t.Fatalf("system message wrong: %+v", first)
	}
	second := msgs[1].(map[string]interface{})
	if second["role"] != "user" || second["content"] != "hi" {
		t.Fatalf("user message wrong: %+v", second)
	}
}

func TestResponsesToChat_FunctionCallFlattenedAfterRealAssistant(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "list files"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": "Sure."},
			map[string]interface{}{"type": "function_call", "call_id": "fc_abc", "name": "Read", "arguments": `{"p":"a"}`},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_abc", "output": "content"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	// Real assistant content is never mutated; flattened tool history opens a
	// new synthetic assistant turn, then a user tool-result turn, and the
	// trailing function_call_output triggers "continue" merge.
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d: %+v", len(msgs), msgs)
	}

	realAsst := msgs[1].(map[string]interface{})
	if realAsst["role"] != "assistant" || realAsst["content"] != "Sure." {
		t.Fatalf("real assistant should pass through verbatim, got %+v", realAsst)
	}
	if _, leaked := realAsst["tool_calls"]; leaked {
		t.Fatalf("real assistant must not carry tool_calls, got %+v", realAsst)
	}

	synAsst := msgs[2].(map[string]interface{})
	if synAsst["role"] != "assistant" {
		t.Fatalf("third msg should be synthetic assistant, got %+v", synAsst)
	}
	if synAsst["content"] != `[Tool call: Read({"p":"a"})]` {
		t.Fatalf("synthetic assistant content wrong: %q", synAsst["content"])
	}
	if _, leaked := synAsst["tool_calls"]; leaked {
		t.Fatalf("flattened path must not emit tool_calls, got %+v", synAsst)
	}

	toolUser := msgs[3].(map[string]interface{})
	if toolUser["role"] != "user" {
		t.Fatalf("last msg should be user (tool-result + continue), got %+v", toolUser)
	}
	got := toolUser["content"].(string)
	if !strings.Contains(got, "[Tool `Read` returned: content]") {
		t.Fatalf("user content missing tool-result marker: %q", got)
	}
	if !strings.HasSuffix(got, "continue") {
		t.Fatalf("tail function_call_output should trigger continue suffix, got %q", got)
	}
}

func TestResponsesToChat_FunctionCallWithoutPriorAssistantOpensSyntheticOne(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_1", "output": "ok"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("want 3 messages, got %d: %+v", len(msgs), msgs)
	}
	asst := msgs[1].(map[string]interface{})
	if asst["role"] != "assistant" || asst["content"] != "[Tool call: N({})]" {
		t.Fatalf("synthetic assistant wrong: %+v", asst)
	}
	if _, leaked := asst["tool_calls"]; leaked {
		t.Fatalf("flattened path must not emit tool_calls: %+v", asst)
	}
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" {
		t.Fatalf("tail should be user, got %+v", tail)
	}
	tc := tail["content"].(string)
	if !strings.Contains(tc, "[Tool `N` returned: ok]") || !strings.HasSuffix(tc, "continue") {
		t.Fatalf("tail content wrong: %q", tc)
	}
}

func TestResponsesToChat_MultipleFunctionCallsStackIntoOneAssistantTurn(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "do two things"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": "ok"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "A", "arguments": "{}"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_2", "name": "B", "arguments": "{}"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_1", "output": "a"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_2", "output": "b"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	// user, real assistant, synthetic assistant (A+B merged), user (a+b merged + continue)
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d: %+v", len(msgs), msgs)
	}
	synAsst := msgs[2].(map[string]interface{})
	if synAsst["role"] != "assistant" {
		t.Fatalf("third msg should be synthetic assistant, got %+v", synAsst)
	}
	sc := synAsst["content"].(string)
	if !strings.Contains(sc, "[Tool call: A({})]") || !strings.Contains(sc, "[Tool call: B({})]") {
		t.Fatalf("adjacent tool_calls should merge into one assistant turn: %q", sc)
	}
	tail := msgs[3].(map[string]interface{})
	tc := tail["content"].(string)
	if !strings.Contains(tc, "[Tool `A` returned: a]") || !strings.Contains(tc, "[Tool `B` returned: b]") {
		t.Fatalf("adjacent tool_results should merge into one user turn: %q", tc)
	}
	if !strings.HasSuffix(tc, "continue") {
		t.Fatalf("tail function_call_output should trigger continue suffix: %q", tc)
	}
}

func TestResponsesToChat_ReasoningDropped(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
			map[string]interface{}{"type": "reasoning", "encrypted_content": "abcdef"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": "hello"},
			map[string]interface{}{"type": "reasoning", "summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "thought"}}},
			map[string]interface{}{"type": "message", "role": "user", "content": "and again"},
		},
	}
	chat, stats := mustTranslate(t, body)
	if stats.DroppedReasoning != 2 {
		t.Fatalf("dropped reasoning want 2 got %d", stats.DroppedReasoning)
	}
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("reasoning should be stripped, got %d messages: %+v", len(msgs), msgs)
	}
}

func TestResponsesToChat_ToolsTranslated(t *testing.T) {
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
				"strict": true,
			},
			map[string]interface{}{"type": "web_search"},
			map[string]interface{}{"type": "image_generation"},
			map[string]interface{}{"type": "custom", "name": "apply_patch", "description": "d", "parameters": map[string]interface{}{"type": "object"}},
		},
	}
	chat, stats := mustTranslate(t, body)
	if stats.ToolsIn != 4 || stats.ToolsOut != 2 || stats.DroppedTools != 2 {
		t.Fatalf("tool stats wrong: %+v", stats)
	}
	tools := chat["tools"].([]interface{})
	if len(tools) != 2 {
		t.Fatalf("expected 2 chat tools, got %d", len(tools))
	}
	first := tools[0].(map[string]interface{})
	if first["type"] != "function" {
		t.Fatalf("first tool type wrong: %+v", first)
	}
	fn := first["function"].(map[string]interface{})
	if fn["name"] != "Read" || fn["description"] != "Read a file" || fn["strict"] != true {
		t.Fatalf("first tool function wrong: %+v", fn)
	}
	if fn["parameters"] == nil {
		t.Fatalf("parameters missing")
	}
}

func TestResponsesToChat_ContentPartsImageMixed(t *testing.T) {
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
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	user := msgs[0].(map[string]interface{})
	parts, ok := user["content"].([]interface{})
	if !ok {
		t.Fatalf("content should be parts array when image present, got %T", user["content"])
	}
	if len(parts) != 2 {
		t.Fatalf("expect 2 parts, got %d: %+v", len(parts), parts)
	}
	first := parts[0].(map[string]interface{})
	if first["type"] != "text" || first["text"] != "describe" {
		t.Fatalf("first part wrong: %+v", first)
	}
	second := parts[1].(map[string]interface{})
	if second["type"] != "image_url" {
		t.Fatalf("second part should be image_url, got %+v", second)
	}
	iu := second["image_url"].(map[string]interface{})
	if iu["url"] != "https://example.com/a.png" {
		t.Fatalf("image_url wrong: %+v", iu)
	}
}

func TestResponsesToChat_ContentPartsAllTextCollapseToString(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": "a"},
					map[string]interface{}{"type": "input_text", "text": "b"},
				},
			},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	user := msgs[0].(map[string]interface{})
	s, ok := user["content"].(string)
	if !ok {
		t.Fatalf("all-text content should collapse to string, got %T", user["content"])
	}
	if !strings.Contains(s, "a") || !strings.Contains(s, "b") {
		t.Fatalf("content missing fragments: %q", s)
	}
}

func TestResponsesToChat_MaxOutputTokensMapsToMaxTokens(t *testing.T) {
	body := map[string]interface{}{
		"model":             "gpt-5",
		"max_output_tokens": 4096,
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	chat, _ := mustTranslate(t, body)
	if chat["max_tokens"].(float64) != 4096 {
		t.Fatalf("max_tokens want 4096 got %v", chat["max_tokens"])
	}
	if _, exists := chat["max_output_tokens"]; exists {
		t.Fatalf("max_output_tokens should be stripped")
	}
}

func TestResponsesToChat_PreviousResponseIDStripped(t *testing.T) {
	body := map[string]interface{}{
		"model":                "gpt-5",
		"previous_response_id": "resp_abc",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
		},
	}
	chat, _ := mustTranslate(t, body)
	if _, exists := chat["previous_response_id"]; exists {
		t.Fatalf("previous_response_id should not leak into chat body")
	}
}

func TestResponsesToChat_FunctionCallOutputNonString(t *testing.T) {
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
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	tail := msgs[len(msgs)-1].(map[string]interface{})
	if tail["role"] != "user" {
		t.Fatalf("tail should be user (flattened tool-result), got %+v", tail)
	}
	content := tail["content"].(string)
	if !strings.Contains(content, `"result"`) || !strings.Contains(content, `"ok"`) {
		t.Fatalf("non-string output should JSON-stringify into user content, got: %q", content)
	}
	if !strings.Contains(content, "[Tool `N` returned:") {
		t.Fatalf("content should carry tool-result marker with tool name, got: %q", content)
	}
}

func TestResponsesToChat_TailFunctionCallOutputSynthesizesContinue(t *testing.T) {
	// Last input item is a function_call_output — agent loop continuation.
	// Model needs a user "continue" signal or it has nothing to respond to.
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
			map[string]interface{}{"type": "function_call_output", "call_id": "fc_1", "output": "done"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	tail := msgs[len(msgs)-1].(map[string]interface{})
	if tail["role"] != "user" {
		t.Fatalf("tail must be user role, got %+v", tail)
	}
	content := tail["content"].(string)
	if !strings.HasSuffix(content, "continue") {
		t.Fatalf("tail content should end with continue, got: %q", content)
	}
	// Continue should merge into the existing tool-result turn (no separate
	// turn appended).
	if !strings.Contains(content, "[Tool `N` returned: done]") {
		t.Fatalf("continue should merge into existing tool-result turn, got: %q", content)
	}
}

func TestResponsesToChat_TailRealUserMessageUnchanged(t *testing.T) {
	// Real user utterance at tail → no continue synthesis, content passes through.
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "just say hi"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d: %+v", len(msgs), msgs)
	}
	only := msgs[0].(map[string]interface{})
	if only["role"] != "user" || only["content"] != "just say hi" {
		t.Fatalf("real user turn should pass through verbatim, got %+v", only)
	}
	if strings.HasSuffix(only["content"].(string), "continue") {
		t.Fatalf("real user tail must not get continue suffix, got %+v", only)
	}
}

func TestResponsesToChat_DanglingFunctionCallAppendsUserTurn(t *testing.T) {
	// function_call with no following function_call_output — unusual but
	// possible mid-flight. Tail would be a synthetic assistant turn; we need
	// to append a user "continue" turn since Anthropic requires last=user and
	// OpenAI benefits from an explicit prompt.
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "x"},
			map[string]interface{}{"type": "function_call", "call_id": "fc_1", "name": "N", "arguments": "{}"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("want 3 messages (user, synth assistant, continue user), got %d: %+v", len(msgs), msgs)
	}
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("tail should be fresh user 'continue' turn, got %+v", tail)
	}
}

func TestResponsesToChat_StringInputNormalizedToUser(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": "hello",
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d", len(msgs))
	}
	msg := msgs[0].(map[string]interface{})
	if msg["role"] != "user" || msg["content"] != "hello" {
		t.Fatalf("bad message: %+v", msg)
	}
}

func TestResponsesToChat_StreamFlagPreserved(t *testing.T) {
	body := map[string]interface{}{
		"model":  "gpt-5",
		"stream": true,
		"input":  "hi",
	}
	chat, _ := mustTranslate(t, body)
	if chat["stream"] != true {
		t.Fatalf("stream flag lost: %+v", chat)
	}
}
