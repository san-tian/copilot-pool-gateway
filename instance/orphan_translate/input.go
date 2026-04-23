// Package orphan_translate converts an OpenAI Responses API request body into
// a chat/completions request body (and wraps a chat/completions SSE response
// back into a Responses SSE event stream).
//
// Purpose: when a /v1/responses request lands on our router as an "orphan"
// (no known sticky binding — typically a cross-relay migration), Copilot's
// stateful /v1/responses endpoint will reject the request because the fc_ids
// and/or previous_response_id don't belong to the new connection's session.
// Chat/completions is stateless with respect to tool_call_ids, so we route
// orphan traffic through /v1/chat/completions via the worker and translate
// back to Responses SSE so the client is unaware.
//
// Only used in the orphan_passthrough branch; normal Responses requests are
// still direct-forwarded to /v1/responses.
package orphan_translate

import (
	"encoding/json"
	"fmt"
	"strings"
)

// TranslateStats summarizes what happened during payload translation.
type TranslateStats struct {
	InputItems       int
	Messages         int
	DroppedReasoning int
	ToolsIn          int
	ToolsOut         int
	DroppedTools     int
}

// ResponsesToChat translates a Responses API request body into a
// chat/completions request body.
//
// Reasoning items are dropped (chat/completions has no equivalent; orphan
// migration can't preserve encrypted_content anyway because it was minted
// by another relay's Copilot session).
//
// Non-function tools (web_search, image_generation) are dropped.
//
// Prior tool history is FLATTENED into conversation prose: function_call
// items become "[Tool call: name(args)]" text inside an assistant turn, and
// function_call_output items become "[Tool `name` returned: output]" text
// inside a user turn. The current-turn `tools` array still passes through
// so the model can issue fresh tool calls. When the input tail is a
// function_call_output (client-side agent loop continuation), synthesize a
// trailing user "continue" signal so the model knows to keep going.
func ResponsesToChat(bodyBytes []byte) ([]byte, TranslateStats, error) {
	var src map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &src); err != nil {
		return nil, TranslateStats{}, fmt.Errorf("parse responses body: %w", err)
	}

	stats := TranslateStats{}
	dst := make(map[string]interface{}, len(src))

	for _, k := range []string{
		"model", "stream", "temperature", "top_p", "user", "metadata",
		"parallel_tool_calls", "tool_choice", "seed", "n", "stop",
		"presence_penalty", "frequency_penalty",
	} {
		if v, ok := src[k]; ok {
			dst[k] = v
		}
	}
	if v, ok := src["max_output_tokens"]; ok {
		dst["max_tokens"] = v
	}
	if v, ok := src["max_completion_tokens"]; ok {
		dst["max_completion_tokens"] = v
	}

	flat := newFlatBuilder()

	if instr, ok := src["instructions"].(string); ok && strings.TrimSpace(instr) != "" {
		flat.appendRealMessage(map[string]interface{}{
			"role":    "system",
			"content": instr,
		})
	}

	input := normalizeInput(src["input"])
	stats.InputItems = len(input)

	for _, item := range input {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(asString(itemMap["type"])))
		role := strings.TrimSpace(strings.ToLower(asString(itemMap["role"])))

		switch {
		case itemType == "reasoning":
			stats.DroppedReasoning++
			continue

		case itemType == "function_call":
			name := asString(itemMap["name"])
			args := asString(itemMap["arguments"])
			if callID := asString(itemMap["call_id"]); callID != "" && name != "" {
				flat.rememberToolName(callID, name)
			}
			flat.appendToolCallText(formatToolCallMarker(name, args))

		case itemType == "function_call_output":
			callID := asString(itemMap["call_id"])
			output := extractFunctionCallOutput(itemMap["output"])
			name := flat.toolName(callID)
			flat.appendToolResultText(formatToolResultMarker(name, output))

		case itemType == "message" || (itemType == "" && role != ""):
			if msg := translateMessage(itemMap); msg != nil {
				flat.appendRealMessage(msg)
			}

		default:
			if role != "" {
				if msg := translateMessage(itemMap); msg != nil {
					flat.appendRealMessage(msg)
				}
			}
		}
	}

	messages := flat.finalize()
	stats.Messages = len(messages)
	dst["messages"] = messages

	if rawTools, ok := src["tools"]; ok {
		toolsArr, _ := rawTools.([]interface{})
		stats.ToolsIn = len(toolsArr)
		out := make([]interface{}, 0, len(toolsArr))
		for _, t := range toolsArr {
			tm, ok := t.(map[string]interface{})
			if !ok {
				continue
			}
			converted := translateTool(tm)
			if converted == nil {
				stats.DroppedTools++
				continue
			}
			out = append(out, converted)
		}
		stats.ToolsOut = len(out)
		if len(out) > 0 {
			dst["tools"] = out
		}
	}

	out, err := json.Marshal(dst)
	if err != nil {
		return nil, stats, fmt.Errorf("marshal chat body: %w", err)
	}
	return out, stats, nil
}

func normalizeInput(raw interface{}) []interface{} {
	switch typed := raw.(type) {
	case nil:
		return nil
	case []interface{}:
		return typed
	case map[string]interface{}:
		return []interface{}{typed}
	case string:
		return []interface{}{map[string]interface{}{"role": "user", "content": typed}}
	default:
		return nil
	}
}

func translateMessage(itemMap map[string]interface{}) map[string]interface{} {
	role := strings.TrimSpace(strings.ToLower(asString(itemMap["role"])))
	if role == "" {
		return nil
	}
	msg := map[string]interface{}{"role": role}
	switch c := itemMap["content"].(type) {
	case string:
		msg["content"] = c
	case []interface{}:
		msg["content"] = translateContentParts(c)
	case nil:
		msg["content"] = ""
	default:
		if b, err := json.Marshal(c); err == nil {
			msg["content"] = string(b)
		} else {
			msg["content"] = ""
		}
	}
	return msg
}

// translateContentParts converts a Responses content array into a
// chat/completions content value. If the parts are all text, returns a
// single concatenated string (cleaner payload). If any image is present,
// returns a parts array.
func translateContentParts(content []interface{}) interface{} {
	var plainText strings.Builder
	parts := make([]interface{}, 0, len(content))
	hasImage := false

	flushText := func() {
		if plainText.Len() > 0 {
			parts = append(parts, map[string]interface{}{"type": "text", "text": plainText.String()})
			plainText.Reset()
		}
	}

	for _, entry := range content {
		em, ok := entry.(map[string]interface{})
		if !ok {
			continue
		}
		et := strings.TrimSpace(strings.ToLower(asString(em["type"])))
		switch et {
		case "input_text", "output_text", "text":
			txt := asString(em["text"])
			if plainText.Len() > 0 {
				plainText.WriteByte('\n')
			}
			plainText.WriteString(txt)
		case "input_image":
			hasImage = true
			flushText()
			if url := extractImageURL(em); url != "" {
				parts = append(parts, map[string]interface{}{
					"type":      "image_url",
					"image_url": map[string]interface{}{"url": url},
				})
			}
		case "input_file":
			// chat/completions has no file-input concept for orphan mode; drop.
		default:
			if txt := asString(em["text"]); txt != "" {
				if plainText.Len() > 0 {
					plainText.WriteByte('\n')
				}
				plainText.WriteString(txt)
			}
		}
	}

	if !hasImage {
		return plainText.String()
	}
	flushText()
	return parts
}

func extractImageURL(em map[string]interface{}) string {
	if u := asString(em["image_url"]); u != "" {
		return u
	}
	if m, ok := em["image_url"].(map[string]interface{}); ok {
		if u := asString(m["url"]); u != "" {
			return u
		}
	}
	return ""
}

func extractFunctionCallOutput(output interface{}) string {
	switch v := output.(type) {
	case string:
		return v
	case nil:
		return ""
	case map[string]interface{}, []interface{}:
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
	}
	b, _ := json.Marshal(output)
	return string(b)
}

func translateTool(t map[string]interface{}) map[string]interface{} {
	toolType := strings.TrimSpace(strings.ToLower(asString(t["type"])))
	switch toolType {
	case "function", "custom":
		name := asString(t["name"])
		if strings.TrimSpace(name) == "" {
			return nil
		}
		fn := map[string]interface{}{"name": name}
		if desc := asString(t["description"]); desc != "" {
			fn["description"] = desc
		}
		if params, ok := t["parameters"]; ok {
			fn["parameters"] = params
		} else {
			fn["parameters"] = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		}
		if strict, ok := t["strict"].(bool); ok {
			fn["strict"] = strict
		}
		return map[string]interface{}{"type": "function", "function": fn}
	default:
		return nil
	}
}

func asString(v interface{}) string {
	s, _ := v.(string)
	return s
}

// formatToolCallMarker renders a function_call history entry as prose. Example:
//
//	[Tool call: Read({"p":"a"})]
//
// Empty arguments render as the name alone with parens. Both translators use
// this so the flattened format stays consistent across chat and messages
// routes.
func formatToolCallMarker(name, args string) string {
	trimmedArgs := strings.TrimSpace(args)
	if trimmedArgs == "" {
		return "[Tool call: " + name + "()]"
	}
	return "[Tool call: " + name + "(" + trimmedArgs + ")]"
}

// formatToolResultMarker renders a function_call_output history entry as
// prose. When the tool name is known (tracked by call_id while walking
// input[]), uses "[Tool `name` returned: output]"; otherwise falls back to
// "[Tool result: output]".
func formatToolResultMarker(name, output string) string {
	if strings.TrimSpace(name) != "" {
		return "[Tool `" + name + "` returned: " + output + "]"
	}
	return "[Tool result: " + output + "]"
}

// flatTurn is one accumulated turn inside flatBuilder. `origin` tracks where
// the content came from so finalize* can decide how to append the "continue"
// signal: merged into a tool-result user turn vs. a fresh user turn after an
// assistant-terminated input.
type flatTurn struct {
	role   string
	text   string
	origin flatOrigin
	// realMsg is set when this turn came from a real `message` item; the
	// turn is emitted verbatim (preserving parts arrays, images, etc.)
	// without going through text accumulation.
	realMsg map[string]interface{}
}

type flatOrigin int

const (
	originReal flatOrigin = iota
	originToolCall
	originToolResult
)

// flatBuilder is a small helper both translators use to flatten prior tool
// history into conversation text. Adjacency merging matches the existing
// batching rules: successive function_call items stack into one assistant
// turn, successive function_call_output items stack into one user turn.
type flatBuilder struct {
	turns         []flatTurn
	toolNames     map[string]string // call_id → tool name, populated on function_call
}

func newFlatBuilder() *flatBuilder {
	return &flatBuilder{
		turns:     make([]flatTurn, 0, 16),
		toolNames: make(map[string]string, 4),
	}
}

func (b *flatBuilder) rememberToolName(callID, name string) {
	b.toolNames[callID] = name
}

func (b *flatBuilder) toolName(callID string) string {
	return b.toolNames[callID]
}

// appendToolCallText appends flattened function_call text to the trailing
// assistant turn, opening one if the last turn isn't a suitable assistant
// carrier.
func (b *flatBuilder) appendToolCallText(text string) {
	b.appendGeneratedText("assistant", originToolCall, text)
}

// appendToolResultText appends flattened function_call_output text to the
// trailing user turn, opening one if needed.
func (b *flatBuilder) appendToolResultText(text string) {
	b.appendGeneratedText("user", originToolResult, text)
}

func (b *flatBuilder) appendGeneratedText(role string, origin flatOrigin, text string) {
	// Merge into trailing turn only if it's same role AND synthetic. Real
	// messages hold their own content verbatim; don't mutate them.
	if n := len(b.turns); n > 0 {
		last := &b.turns[n-1]
		if last.role == role && last.origin == origin && last.realMsg == nil {
			last.text = last.text + "\n" + text
			return
		}
	}
	b.turns = append(b.turns, flatTurn{role: role, origin: origin, text: text})
}

func (b *flatBuilder) appendRealMessage(msg map[string]interface{}) {
	role, _ := msg["role"].(string)
	b.turns = append(b.turns, flatTurn{role: role, origin: originReal, realMsg: msg})
}

// finalize renders the accumulated turns into a messages[] array and
// appends a trailing "continue" user signal if the tail isn't a genuine
// user utterance. Format-agnostic: the chat and messages routes both
// consume a {role, content} shape; image-bearing real messages are
// preserved verbatim with whatever parts array the caller pre-translated.
func (b *flatBuilder) finalize() []interface{} {
	b.appendContinueIfNeeded()
	out := make([]interface{}, 0, len(b.turns))
	for _, t := range b.turns {
		if t.realMsg != nil {
			out = append(out, t.realMsg)
			continue
		}
		out = append(out, map[string]interface{}{
			"role":    t.role,
			"content": t.text,
		})
	}
	return out
}

// appendContinueIfNeeded ensures the final turn is a user turn that looks
// like a real prompt. Rules:
//   - Last turn is user + real message: leave alone (real user said something).
//   - Last turn is user + tool-result origin: merge "\n\ncontinue" into it so
//     strict alternation stays valid and the model sees a continuation signal.
//   - Last turn is assistant (real or tool-call origin) or anything else:
//     append a fresh {role:"user", content:"continue"} turn.
//   - Empty turns (no input at all): append the continue turn so callers get
//     a valid payload; higher-level code also has an empty-messages guard.
func (b *flatBuilder) appendContinueIfNeeded() {
	if n := len(b.turns); n > 0 {
		last := &b.turns[n-1]
		if last.role == "user" && last.origin == originReal {
			return
		}
		if last.role == "user" && last.origin == originToolResult {
			last.text = last.text + "\n\ncontinue"
			return
		}
	}
	b.turns = append(b.turns, flatTurn{role: "user", origin: originToolResult, text: "continue"})
}
