package orphan_translate

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesToMessages translates a Responses API request body into an
// Anthropic Messages API request body.
//
// Rationale: Copilot's /v1/chat/completions endpoint rejects the gpt-5 family
// and Anthropic models with
//
//	{"error":{"message":"Please use `/v1/responses` or `/v1/messages` API"}}
//
// The /v1/messages endpoint, however, is stateless (no session-bound
// tool_use_ids) and accepts BOTH gpt-5.4 and claude-* via the Anthropic shape.
// Routing gpt-5.4 orphan traffic through /v1/messages therefore lets us keep
// the "translate in Go, worker passes through" architecture that the chat path
// already validated for gpt-4o.
//
// Reasoning items are dropped — encrypted_content was minted by another relay
// and Anthropic has no equivalent inline reasoning carrier.
// Non-function tools (web_search, image_generation, custom types) are dropped.
//
// Reasoning and tool protocol history is converted into an explicit
// ToolCallHistory block instead of being flattened as ordinary prose. If the
// original Responses structure cannot be replayed on the bound connection,
// unlabeled prose markers such as "[Tool call: ...]" pollute the next model
// turn and can make the model emit tool/thinking protocol as user visible
// text. Automatic recovery therefore preserves ordinary role-bearing message
// transcript plus a clearly-labeled historical context block. The explicit
// X-Copilot-Continuation-Degrade opt-in still owns legacy lossy prose
// degradation.
func ResponsesToMessages(bodyBytes []byte) ([]byte, TranslateStats, error) {
	var src map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &src); err != nil {
		return nil, TranslateStats{}, fmt.Errorf("parse responses body: %w", err)
	}

	stats := TranslateStats{}
	dst := make(map[string]interface{}, len(src))

	for _, k := range []string{"model", "stream", "temperature", "top_p", "metadata", "stop_sequences"} {
		if v, ok := src[k]; ok {
			dst[k] = v
		}
	}

	// Anthropic requires max_tokens. If not supplied, default to a conservative
	// ceiling; worker enforces per-model limits, so oversetting is safe (capped
	// to the model's own max).
	if v, ok := src["max_output_tokens"]; ok {
		dst["max_tokens"] = v
	} else if v, ok := src["max_tokens"]; ok {
		dst["max_tokens"] = v
	} else {
		dst["max_tokens"] = 8192
	}

	if instr, ok := src["instructions"].(string); ok && strings.TrimSpace(instr) != "" {
		dst["system"] = instr
	}

	messages := translateResponsesInputToAnthropic(src["input"], &stats)
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
			converted := translateToolForAnthropic(tm)
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

	if tc, ok := src["tool_choice"]; ok {
		if converted := translateToolChoiceForAnthropic(tc); converted != nil {
			dst["tool_choice"] = converted
		}
	}

	out, err := json.Marshal(dst)
	if err != nil {
		return nil, stats, fmt.Errorf("marshal messages body: %w", err)
	}
	return out, stats, nil
}

// translateResponsesInputToAnthropic walks Responses input[] and emits an
// Anthropic messages[] array for automatic recovery. It keeps ordinary
// role-bearing messages and converts reasoning/function-call history into a
// labeled context block so protocol artifacts do not look like live protocol.
func translateResponsesInputToAnthropic(raw interface{}, stats *TranslateStats) []interface{} {
	input := normalizeInput(raw)
	stats.InputItems = len(input)

	flat := newFlatBuilder()
	history := newProtocolHistoryBuilder()

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
			history.addReasoning(itemMap)
			continue

		case itemType == "function_call":
			history.addToolCall(itemMap)
			continue

		case itemType == "function_call_output":
			history.addToolResult(itemMap)
			continue

		case itemType == "message" || (itemType == "" && role != ""):
			if msg := translateMessageForAnthropic(itemMap); msg != nil {
				flat.appendRealMessage(msg)
			}

		default:
			if role != "" {
				if msg := translateMessageForAnthropic(itemMap); msg != nil {
					flat.appendRealMessage(msg)
				}
			}
		}
	}
	flat.appendRecoveryHistory(history.render())

	return flat.finalize()
}

// translateMessageForAnthropic converts one Responses message item into one
// Anthropic message. user/assistant/system roles pass through; system turns
// are dropped here (system is a top-level field in Anthropic) unless we have
// no other place to put them — caller handles `instructions` directly.
func translateMessageForAnthropic(itemMap map[string]interface{}) map[string]interface{} {
	role := strings.TrimSpace(strings.ToLower(asString(itemMap["role"])))
	if role == "" {
		return nil
	}
	if role == "system" || role == "developer" {
		// Anthropic has no per-turn system role; fold into a user turn as plain
		// text so the content isn't silently lost. (instructions-level system
		// handled by the caller via dst["system"].)
		role = "user"
	}
	msg := map[string]interface{}{"role": role}
	switch c := itemMap["content"].(type) {
	case string:
		msg["content"] = c
	case []interface{}:
		msg["content"] = translateContentPartsForAnthropic(c)
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

// translateContentPartsForAnthropic converts a Responses content array into an
// Anthropic content value. All-text collapses to a single string; any image
// forces the full parts array.
func translateContentPartsForAnthropic(content []interface{}) interface{} {
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
			if block := buildAnthropicImageBlock(em); block != nil {
				parts = append(parts, block)
			}
		case "input_file":
			// Anthropic /v1/messages has no file-input concept in our orphan
			// scenario; drop.
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

// buildAnthropicImageBlock converts a Responses input_image entry to an
// Anthropic image content block. Supports both data: URLs (base64 source) and
// https: URLs (url source, Claude 3.5+).
func buildAnthropicImageBlock(em map[string]interface{}) map[string]interface{} {
	url := extractImageURL(em)
	if url == "" {
		return nil
	}
	if strings.HasPrefix(url, "data:") {
		mediaType, data := splitDataURL(url)
		if mediaType == "" || data == "" {
			return nil
		}
		return map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": mediaType,
				"data":       data,
			},
		}
	}
	return map[string]interface{}{
		"type": "image",
		"source": map[string]interface{}{
			"type": "url",
			"url":  url,
		},
	}
}

// splitDataURL parses "data:image/png;base64,ABCD..." into ("image/png","ABCD...").
func splitDataURL(u string) (string, string) {
	rest := strings.TrimPrefix(u, "data:")
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", ""
	}
	meta := rest[:comma]
	body := rest[comma+1:]
	semi := strings.IndexByte(meta, ';')
	if semi < 0 {
		return meta, body
	}
	return meta[:semi], body
}

// translateToolForAnthropic converts an OpenAI function-tool entry into an
// Anthropic tool descriptor. Non-function tools return nil so the caller can
// count them as dropped.
func translateToolForAnthropic(t map[string]interface{}) map[string]interface{} {
	toolType := strings.TrimSpace(strings.ToLower(asString(t["type"])))
	switch toolType {
	case "function", "custom":
		name := asString(t["name"])
		if strings.TrimSpace(name) == "" {
			return nil
		}
		tool := map[string]interface{}{"name": name}
		if desc := asString(t["description"]); desc != "" {
			tool["description"] = desc
		}
		if params, ok := t["parameters"]; ok {
			tool["input_schema"] = params
		} else {
			tool["input_schema"] = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}
		return tool
	default:
		return nil
	}
}

// translateToolChoiceForAnthropic converts an OpenAI Responses tool_choice to
// the Anthropic tool_choice shape. Falls back to {"type":"auto"} rather than
// nil so the worker's validator sees a well-formed field.
func translateToolChoiceForAnthropic(tc interface{}) map[string]interface{} {
	switch v := tc.(type) {
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "none":
			return map[string]interface{}{"type": "none"}
		case "required":
			return map[string]interface{}{"type": "any"}
		case "", "auto":
			return map[string]interface{}{"type": "auto"}
		default:
			return map[string]interface{}{"type": "auto"}
		}
	case map[string]interface{}:
		// Responses shape: {"type":"function","function":{"name":"x"}} or
		//                  {"type":"function","name":"x"}
		if ft := strings.ToLower(strings.TrimSpace(asString(v["type"]))); ft == "function" || ft == "tool" {
			name := asString(v["name"])
			if fn, ok := v["function"].(map[string]interface{}); ok && name == "" {
				name = asString(fn["name"])
			}
			if name != "" {
				return map[string]interface{}{"type": "tool", "name": name}
			}
		}
		return map[string]interface{}{"type": "auto"}
	default:
		return nil
	}
}
