package instance

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/config"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

const responsesCompactSummaryTokenLimit = 2048
const responsesCompactLogValueLimit = 1200

// toolResultMaxChars caps the body of each function_call_output in the
// compact transcript. Huge tool results (file dumps, log pastes) are the
// dominant size driver and can push the compact request itself past the
// model's context window; see Option C in the 2026-04-23 compact-overflow
// discussion. Oversized bodies are replaced with head+tail samples.
const toolResultMaxChars = 2048

// truncateToolResultForCompact returns output unchanged if within the cap.
// Otherwise it keeps the first 3/4 and last 1/4 of the budget with an
// elision marker between, so the model still sees what the tool was
// returning without the full dump.
func truncateToolResultForCompact(output string) string {
	if len(output) <= toolResultMaxChars {
		return output
	}
	headLen := toolResultMaxChars * 3 / 4
	tailLen := toolResultMaxChars - headLen
	elided := len(output) - headLen - tailLen
	return fmt.Sprintf("%s\n... [%d chars elided] ...\n%s", output[:headLen], elided, output[len(output)-tailLen:])
}

// responsesCompactInstructions is the developer/system role prompt, adapted
// from pi-coding-agent's SUMMARIZATION_SYSTEM_PROMPT. It frames the call as a
// summarization task on transcript-as-data (not live instructions) so the
// model doesn't try to continue the conversation.
var responsesCompactInstructions = strings.Join([]string{
	"You are a context summarization assistant.",
	"Your task is to read a conversation between a user and an AI coding assistant,",
	"then produce a structured summary following the exact format specified.",
	"Do NOT continue the conversation. Do NOT respond to any questions in the conversation.",
	"ONLY output the structured summary.",
}, " ")

// compactInitialUserPrompt is attached to the user turn when no prior
// compaction_summary is present in input[]. Adapted from pi's
// SUMMARIZATION_PROMPT: requests the fixed ## Goal / ## Progress / ... layout
// so downstream retrieval logic can parse it consistently.
const compactInitialUserPrompt = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`

// compactUpdateUserPrompt is used when input[] contains one or more prior
// compaction_summary items (re-compaction after further turns). Adapted from
// pi's UPDATE_SUMMARIZATION_PROMPT: PRESERVE/ADD/UPDATE rules so the checkpoint
// stays cumulative instead of dropping earlier context each round. The caller
// wraps previous text in <previous-summary>…</previous-summary> tags.
const compactUpdateUserPrompt = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`

// DoResponsesCompactProxy emulates Codex /responses/compact for Copilot-backed accounts.
func DoResponsesCompactProxy(state *config.State, bodyBytes []byte) (*http.Response, []byte, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return nil, nil, &ResponsesRewriteError{Message: fmt.Sprintf("invalid compact request: %v", err)}
	}

	modelID := ""
	if model, ok := payload["model"].(string); ok {
		modelID = store.ToCopilotID(model)
	}
	if strings.TrimSpace(modelID) == "" {
		return nil, nil, &ResponsesRewriteError{Message: "compact request requires model"}
	}

	input, err := normalizeResponsesInput(payload["input"])
	if err != nil {
		return nil, nil, &ResponsesRewriteError{Message: fmt.Sprintf("invalid compact input: %v", err)}
	}
	transcript, previousSummary := buildResponsesCompactTranscript(input)
	if strings.TrimSpace(transcript) == "" && strings.TrimSpace(previousSummary) == "" {
		return nil, nil, &ResponsesRewriteError{Message: "compact request requires non-empty input"}
	}

	userContent := buildCompactUserContent(transcript, previousSummary)

	summaryRequest := map[string]interface{}{
		"model":  modelID,
		"stream": false,
		"store":  false,
		"input": []interface{}{
			map[string]interface{}{"role": "developer", "content": responsesCompactInstructions},
			map[string]interface{}{"role": "user", "content": userContent},
		},
	}
	if usesResponsesMaxCompletionTokens(modelID) {
		summaryRequest["max_completion_tokens"] = compactSummaryTokenLimit(state, modelID)
	}

	normalizedBody, err := json.Marshal(summaryRequest)
	if err != nil {
		return nil, nil, err
	}

	turnRequest := newCopilotTurnRequest(copilotInteractionTypeUser)

	resp, err := ProxyRequestWithBytes(state, http.MethodPost, "/responses", normalizedBody, turnRequest.Headers(), false)
	if err != nil {
		log.Printf("Responses compact upstream transport error: model=%s items=%d item_types=%s transcript_chars=%d prev_summary_chars=%d request_bytes=%d err=%v",
			modelID, len(input), compactInputTypeSummary(input), len(transcript), len(previousSummary), len(normalizedBody), err)
		return nil, nil, err
	}
	if resp == nil {
		log.Printf("Responses compact upstream returned nil response: model=%s items=%d item_types=%s transcript_chars=%d prev_summary_chars=%d request_bytes=%d",
			modelID, len(input), compactInputTypeSummary(input), len(transcript), len(previousSummary), len(normalizedBody))
		return nil, nil, fmt.Errorf("compact upstream returned nil response")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		upstreamBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			log.Printf("Responses compact upstream failure: status=%d model=%s items=%d item_types=%s transcript_chars=%d prev_summary_chars=%d request_bytes=%d body_read_error=%v",
				resp.StatusCode, modelID, len(input), compactInputTypeSummary(input), len(transcript), len(previousSummary), len(normalizedBody), readErr)
			upstreamBody = []byte{}
		} else {
			log.Printf("Responses compact upstream failure: status=%d model=%s items=%d item_types=%s transcript_chars=%d prev_summary_chars=%d request_bytes=%d upstream_body=%s",
				resp.StatusCode, modelID, len(input), compactInputTypeSummary(input), len(transcript), len(previousSummary), len(normalizedBody), truncateCompactLogValue(string(upstreamBody), responsesCompactLogValueLimit))
		}
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(upstreamBody))
		resp.ContentLength = int64(len(upstreamBody))
		return resp, upstreamBody, nil
	}

	upstreamBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, nil, err
	}
	_ = resp.Body.Close()

	summaryText, usage, err := extractResponsesSummaryText(upstreamBody)
	if err != nil {
		return nil, nil, &ResponsesRewriteError{Message: err.Error()}
	}

	compactPayload := map[string]interface{}{
		"id":         fmt.Sprintf("cmp_%d", time.Now().UnixNano()),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"model":      modelID,
		"status":     "completed",
		"output": []interface{}{
			map[string]interface{}{
				"type": "compaction_summary",
				"encrypted_content": summaryText,
				"summary": []interface{}{
					map[string]interface{}{"type": "summary_text", "text": summaryText},
				},
			},
		},
	}
	if usage != nil {
		compactPayload["usage"] = usage
	}

	compactBody, err := json.Marshal(compactPayload)
	if err != nil {
		return nil, nil, err
	}

	resp.StatusCode = http.StatusOK
	resp.Status = http.StatusText(http.StatusOK)
	resp.Header.Set("Content-Type", "application/json")
	resp.Header.Del("Content-Encoding")
	resp.ContentLength = int64(len(compactBody))
	resp.Body = io.NopCloser(bytes.NewReader(compactBody))
	return resp, compactBody, nil
}

func ForwardResponsesCompactResponse(c *gin.Context, resp *http.Response, body []byte) {
	defer func() {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
	}()
	if resp == nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "missing compact response"})
		return
	}
	if requestID := strings.TrimSpace(resp.Header.Get("X-Request-Id")); requestID != "" {
		c.Header("X-Request-Id", requestID)
	}
	if requestID := strings.TrimSpace(resp.Header.Get("x-request-id")); requestID != "" {
		c.Header("x-request-id", requestID)
	}
	c.Data(resp.StatusCode, "application/json", body)
}

func rewriteCompactionInput(payload map[string]interface{}) error {
	items, err := normalizeResponsesInput(payload["input"])
	if err != nil || len(items) == 0 {
		return nil
	}

	rewritten := make([]interface{}, 0, len(items))
	modified := false
	for _, item := range items {
		mapped, ok := item.(map[string]interface{})
		if !ok {
			rewritten = append(rewritten, item)
			continue
		}
		itemType, _ := mapped["type"].(string)
		if itemType != "compaction" && itemType != "compaction_summary" {
			rewritten = append(rewritten, item)
			continue
		}

		summaryText := strings.TrimSpace(extractCompactSummaryText(mapped))
		if summaryText == "" {
			return &ResponsesRewriteError{Message: "compaction item did not contain readable summary text"}
		}

		rewritten = append(rewritten, map[string]interface{}{
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []interface{}{
				map[string]interface{}{
					"type":        "output_text",
					"text":        summaryText,
					"annotations": []interface{}{},
					"logprobs":    []interface{}{},
				},
			},
		})
		modified = true
	}

	if modified {
		payload["input"] = rewritten
	}
	return nil
}

func compactSummaryTokenLimit(state *config.State, modelID string) int {
	if limit := lookupMaxOutputTokens(state, modelID); limit > 0 && limit < responsesCompactSummaryTokenLimit {
		return limit
	}
	return responsesCompactSummaryTokenLimit
}

func compactInputTypeSummary(items []interface{}) string {
	if len(items) == 0 {
		return "none"
	}
	seen := map[string]bool{}
	types := make([]string, 0, len(items))
	for _, item := range items {
		mapped, ok := item.(map[string]interface{})
		itemType := "unknown"
		if ok {
			if role, _ := mapped["role"].(string); strings.TrimSpace(role) != "" {
				itemType = "role:" + strings.TrimSpace(role)
			} else if value, _ := mapped["type"].(string); strings.TrimSpace(value) != "" {
				itemType = strings.TrimSpace(value)
			}
		}
		if seen[itemType] {
			continue
		}
		seen[itemType] = true
		types = append(types, itemType)
	}
	return strings.Join(types, ",")
}

func truncateCompactLogValue(value string, limit int) string {
	value = strings.TrimSpace(value)
	if limit <= 0 || len(value) <= limit {
		return value
	}
	return value[:limit] + "...(truncated)"
}

// buildResponsesCompactTranscript walks input[] and partitions it into the
// conversation body and any prior compaction summary text. Prior summaries
// are pulled out so the outer prompt can wrap them in <previous-summary>
// tags for the update-variant of the prompt; if multiple are present (e.g.
// several re-compactions in a row) they are joined with a blank line.
// reasoning items still go into the conversation body because they encode
// new material, not a replacement for the conversation.
func buildResponsesCompactTranscript(items []interface{}) (string, string) {
	var lines []string
	var prevSummaries []string
	for _, item := range items {
		mapped, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		if role, _ := mapped["role"].(string); role != "" {
			text := extractCompactContentText(mapped["content"])
			if strings.TrimSpace(text) == "" {
				continue
			}
			lines = append(lines, fmt.Sprintf("[%s]\n%s", strings.ToUpper(role), text))
			continue
		}

		itemType, _ := mapped["type"].(string)
		switch itemType {
		case "message":
			role, _ := mapped["role"].(string)
			text := extractCompactContentText(mapped["content"])
			if strings.TrimSpace(text) == "" {
				continue
			}
			if role == "" {
				role = "assistant"
			}
			lines = append(lines, fmt.Sprintf("[%s]\n%s", strings.ToUpper(role), text))
		case "function_call":
			name, _ := mapped["name"].(string)
			args, _ := mapped["arguments"].(string)
			lines = append(lines, fmt.Sprintf("[ASSISTANT TOOL CALL] %s(%s)", name, args))
		case "function_call_output":
			callID, _ := mapped["call_id"].(string)
			output := truncateToolResultForCompact(stringifyCompactValue(mapped["output"]))
			lines = append(lines, fmt.Sprintf("[TOOL RESULT %s]\n%s", callID, output))
		case "reasoning":
			text := extractCompactSummaryText(mapped)
			if strings.TrimSpace(text) == "" {
				continue
			}
			lines = append(lines, fmt.Sprintf("[%s]\n%s", strings.ToUpper(itemType), text))
		case "compaction", "compaction_summary":
			text := extractCompactSummaryText(mapped)
			if strings.TrimSpace(text) == "" {
				continue
			}
			prevSummaries = append(prevSummaries, strings.TrimSpace(text))
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n\n")), strings.TrimSpace(strings.Join(prevSummaries, "\n\n"))
}

// buildCompactUserContent renders the user-role body sent to /responses. When
// a prior compaction summary exists, wrap conversation + summary in the tags
// the update prompt expects (<conversation>…</conversation>,
// <previous-summary>…</previous-summary>). Otherwise fall back to the initial
// prompt with conversation only.
func buildCompactUserContent(conversation, previousSummary string) string {
	conversation = strings.TrimSpace(conversation)
	previousSummary = strings.TrimSpace(previousSummary)

	var buf strings.Builder
	if conversation != "" {
		buf.WriteString("<conversation>\n")
		buf.WriteString(conversation)
		buf.WriteString("\n</conversation>\n\n")
	}
	if previousSummary != "" {
		buf.WriteString("<previous-summary>\n")
		buf.WriteString(previousSummary)
		buf.WriteString("\n</previous-summary>\n\n")
		buf.WriteString(compactUpdateUserPrompt)
	} else {
		buf.WriteString(compactInitialUserPrompt)
	}
	return buf.String()
}

func extractResponsesSummaryText(body []byte) (string, map[string]interface{}, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", nil, fmt.Errorf("failed to parse compact summary response: %w", err)
	}

	if outputText, ok := payload["output_text"].(string); ok && strings.TrimSpace(outputText) != "" {
		return strings.TrimSpace(outputText), mapValue(payload["usage"]), nil
	}
	if output, ok := payload["output"].([]interface{}); ok {
		for _, item := range output {
			mapped, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			itemType, _ := mapped["type"].(string)
			if itemType != "message" {
				continue
			}
			text := extractCompactContentText(mapped["content"])
			if strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text), mapValue(payload["usage"]), nil
			}
		}
	}
	return "", nil, fmt.Errorf("compact summary response did not contain text output")
}

func extractCompactSummaryText(value interface{}) string {
	parts := collectCompactSummaryParts(value)
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n")
}

func collectCompactSummaryParts(value interface{}) []string {
	switch typed := value.(type) {
	case string:
		text := strings.TrimSpace(typed)
		if text == "" {
			return nil
		}
		return []string{text}
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			parts = append(parts, collectCompactSummaryParts(item)...)
		}
		return parts
	case map[string]interface{}:
		parts := make([]string, 0, 4)
		for _, key := range []string{"summary_text", "text", "content", "title", "encrypted_content"} {
			parts = append(parts, collectCompactSummaryParts(typed[key])...)
		}
		if summary, ok := typed["summary"].([]interface{}); ok {
			parts = append(parts, collectCompactSummaryParts(summary)...)
		}
		if textValue, ok := typed["text"].(map[string]interface{}); ok {
			parts = append(parts, collectCompactSummaryParts(textValue["value"])...)
		}
		return parts
	default:
		return nil
	}
}

func extractCompactContentText(value interface{}) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			parts = append(parts, extractCompactContentText(item))
		}
		return strings.TrimSpace(strings.Join(filterNonEmpty(parts), "\n"))
	case map[string]interface{}:
		blockType, _ := typed["type"].(string)
		switch blockType {
		case "input_text", "output_text", "text", "summary_text":
			if text, _ := typed["text"].(string); strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
		for _, key := range []string{"text", "content", "summary"} {
			if text := extractCompactContentText(typed[key]); text != "" {
				return text
			}
		}
		return ""
	default:
		return ""
	}
}

func stringifyCompactValue(value interface{}) string {
	if text := extractCompactSummaryText(value); text != "" {
		return text
	}
	body, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(body)
}

func mapValue(value interface{}) map[string]interface{} {
	mapped, _ := value.(map[string]interface{})
	if mapped == nil {
		return nil
	}
	return mapped
}

func filterNonEmpty(items []string) []string {
	filtered := make([]string, 0, len(items))
	for _, item := range items {
		if strings.TrimSpace(item) != "" {
			filtered = append(filtered, strings.TrimSpace(item))
		}
	}
	return filtered
}
