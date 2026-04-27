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

func assertHistoryBlock(t *testing.T, msg interface{}, wants ...string) string {
	t.Helper()
	mapped := msg.(map[string]interface{})
	if mapped["role"] != "user" {
		t.Fatalf("history block must be user context, got %+v", mapped)
	}
	content, ok := mapped["content"].(string)
	if !ok {
		t.Fatalf("history block content should be string, got %T", mapped["content"])
	}
	for _, want := range append([]string{"<ToolCallHistory>", "<Description>", "</Description>"}, wants...) {
		if !strings.Contains(content, want) {
			t.Fatalf("history block missing %q: %s", want, content)
		}
	}
	if strings.Contains(content, "[Tool call") || strings.Contains(content, "[Previous tool call") {
		t.Fatalf("history block must use structured tags instead of legacy prose markers: %s", content)
	}
	return content
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

func TestResponsesToChat_ToolHistoryDroppedAfterRealAssistant(t *testing.T) {
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
	if len(msgs) != 4 {
		t.Fatalf("want ordinary transcript plus continue, got %d: %+v", len(msgs), msgs)
	}

	user := msgs[0].(map[string]interface{})
	if user["role"] != "user" || user["content"] != "list files" {
		t.Fatalf("should keep prior real user message, got %+v", user)
	}
	assistant := msgs[1].(map[string]interface{})
	if assistant["role"] != "assistant" || assistant["content"] != "Sure." {
		t.Fatalf("should keep ordinary assistant message, got %+v", assistant)
	}
	assertHistoryBlock(t, msgs[2], "<Tool>", "<Name>Read</Name>", "<Arguments>{&#34;p&#34;:&#34;a&#34;}</Arguments>", "<Result>content</Result>")
	tail := msgs[3].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("should append continue after assistant tail, got %+v", tail)
	}
	for _, msg := range msgs[:2] {
		content, _ := msg.(map[string]interface{})["content"].(string)
		if strings.Contains(content, "Tool call") || strings.Contains(content, "returned") || strings.Contains(content, "content") {
			t.Fatalf("tool history must not leak into message content: %+v", msg)
		}
	}
}

func TestResponsesToChat_FunctionCallWithoutPriorAssistantDropped(t *testing.T) {
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
		t.Fatalf("want real user message only, got %d: %+v", len(msgs), msgs)
	}
	user := msgs[0].(map[string]interface{})
	if user["role"] != "user" || user["content"] != "x" {
		t.Fatalf("should keep real user message, got %+v", user)
	}
	assertHistoryBlock(t, msgs[1], "<Name>N</Name>", "<Arguments>{}</Arguments>", "<Result>ok</Result>")
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("should append continue after history block, got %+v", tail)
	}
}

func TestResponsesToChat_PreservesOrdinaryTranscript(t *testing.T) {
	body := map[string]interface{}{
		"model": "gpt-5",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "remember alpha"},
			map[string]interface{}{"type": "message", "role": "assistant", "content": "alpha remembered"},
			map[string]interface{}{"type": "message", "role": "user", "content": "what did I ask you to remember?"},
		},
	}
	chat, _ := mustTranslate(t, body)
	msgs := chat["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("want full ordinary transcript, got %d: %+v", len(msgs), msgs)
	}
	for i, want := range []string{"remember alpha", "alpha remembered", "what did I ask you to remember?"} {
		got := msgs[i].(map[string]interface{})["content"]
		if got != want {
			t.Fatalf("message %d content want %q got %+v", i, want, got)
		}
	}
}

func TestResponsesToChat_MultipleFunctionCallsDropped(t *testing.T) {
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
	if len(msgs) != 4 {
		t.Fatalf("want ordinary transcript plus continue, got %d: %+v", len(msgs), msgs)
	}
	user := msgs[0].(map[string]interface{})
	if user["role"] != "user" || user["content"] != "do two things" {
		t.Fatalf("should keep real user message, got %+v", user)
	}
	assistant := msgs[1].(map[string]interface{})
	if assistant["role"] != "assistant" || assistant["content"] != "ok" {
		t.Fatalf("should keep ordinary assistant message, got %+v", assistant)
	}
	assertHistoryBlock(t, msgs[2], "<CallID>fc_1</CallID>", "<CallID>fc_2</CallID>", "<Result>a</Result>", "<Result>b</Result>")
	tail := msgs[3].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("should append continue after dropped tool tail, got %+v", tail)
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
	if len(msgs) != 5 {
		t.Fatalf("recovery should keep ordinary messages, got %d messages: %+v", len(msgs), msgs)
	}
	if msgs[0].(map[string]interface{})["content"] != "hi" {
		t.Fatalf("first user message wrong: %+v", msgs[0])
	}
	if msgs[1].(map[string]interface{})["role"] != "assistant" || msgs[1].(map[string]interface{})["content"] != "hello" {
		t.Fatalf("assistant message wrong: %+v", msgs[1])
	}
	if msgs[2].(map[string]interface{})["content"] != "and again" {
		t.Fatalf("tail user message wrong: %+v", msgs[2])
	}
	assertHistoryBlock(t, msgs[3], "<Reasoning>", "Encrypted reasoning was present", "<Summary>thought</Summary>")
	if msgs[4].(map[string]interface{})["content"] != "continue" {
		t.Fatalf("history block should be followed by continue, got %+v", msgs[4])
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
	if len(msgs) != 3 {
		t.Fatalf("want user, history, continue messages, got %d: %+v", len(msgs), msgs)
	}
	user := msgs[0].(map[string]interface{})
	if user["role"] != "user" || user["content"] != "x" {
		t.Fatalf("should keep real user message, got %+v", user)
	}
	history := assertHistoryBlock(t, msgs[1], "<Result>{&#34;code&#34;:0,&#34;result&#34;:&#34;ok&#34;}</Result>")
	if strings.Contains(history, "[Previous tool call") {
		t.Fatalf("tool output leaked as legacy prose: %q", history)
	}
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("should append continue after history block, got %+v", tail)
	}
}

func TestResponsesToChat_TailFunctionCallOutputDroppedWhenUserExists(t *testing.T) {
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
	if len(msgs) != 3 {
		t.Fatalf("want user, history, continue messages, got %d: %+v", len(msgs), msgs)
	}
	assertHistoryBlock(t, msgs[1], "<Name>N</Name>", "<Arguments>{}</Arguments>", "<Result>done</Result>")
	tail := msgs[2].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "continue" {
		t.Fatalf("tail must be continue user role, got %+v", tail)
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
		t.Fatalf("want real user message only, got %d: %+v", len(msgs), msgs)
	}
	tail := msgs[0].(map[string]interface{})
	if tail["role"] != "user" || tail["content"] != "x" {
		t.Fatalf("tail should be original user turn, got %+v", tail)
	}
	assertHistoryBlock(t, msgs[1], "<Name>N</Name>", "<Arguments>{}</Arguments>")
	if msgs[2].(map[string]interface{})["content"] != "continue" {
		t.Fatalf("history block should be followed by continue, got %+v", msgs[2])
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
