package instance

import (
	"fmt"
	"strings"

	"copilot-go/anthropic"
)

type MessagesRewriteError struct {
	Message string
}

func (e *MessagesRewriteError) Error() string {
	return e.Message
}

func validateAnthropicToolContinuation(payload anthropic.AnthropicMessagesPayload) error {
	provided := latestAnthropicToolResultIDs(payload.Messages)
	if len(provided) == 0 {
		return nil
	}

	pending := latestAnthropicPendingToolUseIDs(payload.Messages)
	if len(pending) == 0 {
		return &MessagesRewriteError{Message: "tool_result continuation does not match a preceding assistant tool_use block"}
	}

	missing := diffExpectedStrings(pending, provided)
	unexpected := diffExpectedStrings(provided, pending)
	if len(missing) == 0 && len(unexpected) == 0 {
		return nil
	}

	message := fmt.Sprintf("tool continuation mismatch: expected results for %s", formatIDList(pending))
	if len(missing) > 0 {
		message += fmt.Sprintf("; missing %s", formatIDList(missing))
	}
	if len(unexpected) > 0 {
		message += fmt.Sprintf("; unexpected %s", formatIDList(unexpected))
	}
	return &MessagesRewriteError{Message: message}
}

func latestAnthropicPendingToolUseIDs(messages []anthropic.AnthropicMessage) []string {
	seenToolResults := false
	for idx := len(messages) - 1; idx >= 0; idx-- {
		message := messages[idx]
		blocks := anthropic.ParseContentBlocksPublic(message.Content)
		switch message.Role {
		case "user":
			if len(collectAnthropicToolResultIDs(blocks)) > 0 {
				seenToolResults = true
			}
		case "assistant":
			if !seenToolResults {
				continue
			}
			return collectAnthropicToolUseIDs(blocks)
		}
	}
	return nil
}

func collectAnthropicToolUseIDs(blocks []anthropic.ContentBlock) []string {
	ids := make([]string, 0)
	for _, block := range blocks {
		if block.Type != "tool_use" {
			continue
		}
		if toolUseID := strings.TrimSpace(block.ID); toolUseID != "" {
			ids = append(ids, toolUseID)
		}
	}
	return uniqueTrimmed(ids)
}

func collectAnthropicToolResultIDs(blocks []anthropic.ContentBlock) []string {
	ids := make([]string, 0)
	for _, block := range blocks {
		if block.Type != "tool_result" {
			continue
		}
		if toolUseID := strings.TrimSpace(block.ToolUseID); toolUseID != "" {
			ids = append(ids, toolUseID)
		}
	}
	return uniqueTrimmed(ids)
}

func normalizeParallelToolCallsSetting(payload map[string]interface{}, endpoint string) error {
	tools, ok := payload["tools"].([]interface{})
	if !ok || len(tools) == 0 {
		return nil
	}
	if enabled, ok := payload["parallel_tool_calls"].(bool); ok && enabled {
		return fmt.Errorf("parallel_tool_calls is not supported yet for %s; submit tool calls serially", endpoint)
	}
	payload["parallel_tool_calls"] = false
	return nil
}

func diffExpectedStrings(expected []string, provided []string) []string {
	expected = uniqueTrimmed(expected)
	provided = uniqueTrimmed(provided)
	if len(expected) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(provided))
	for _, value := range provided {
		seen[value] = true
	}
	missing := make([]string, 0)
	for _, value := range expected {
		if !seen[value] {
			missing = append(missing, value)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return missing
}

func formatIDList(values []string) string {
	values = uniqueTrimmed(values)
	if len(values) == 0 {
		return "[]"
	}
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, fmt.Sprintf("%q", value))
	}
	if len(quoted) == 1 {
		return quoted[0]
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}
