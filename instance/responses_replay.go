package instance

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	responsesReplayTTL        = 30 * time.Minute
	responsesReplayMaxEntries = 256
)

type ResponsesRewriteError struct {
	Message string
}

func (e *ResponsesRewriteError) Error() string {
	return e.Message
}

type responsesReplayEntry struct {
	AccountID   string
	Input       []interface{}
	ReplayItems []interface{}
	CreatedAt   time.Time
	AccessedAt  time.Time
}

var responsesReplayCache = struct {
	mu      sync.Mutex
	entries map[string]*responsesReplayEntry
}{
	entries: map[string]*responsesReplayEntry{},
}

type responsesStreamCapture struct {
	ResponseID  string
	ReplayItems []interface{}
}

func rewritePreviousResponseContinuation(accountID string, payload map[string]interface{}) error {
	return rewritePreviousResponseContinuationWithMode(accountID, payload, false)
}

func rewritePreviousResponseContinuationAnyAccount(payload map[string]interface{}) error {
	return rewritePreviousResponseContinuationWithMode("", payload, true)
}

func rewritePreviousResponseContinuationWithMode(accountID string, payload map[string]interface{}, allowAnyAccount bool) error {
	previousResponseID, _ := payload["previous_response_id"].(string)
	previousResponseID = strings.TrimSpace(previousResponseID)
	if previousResponseID == "" {
		return nil
	}

	var (
		entry *responsesReplayEntry
		ok    bool
	)
	if allowAnyAccount {
		entry, ok = loadResponsesReplayAny(previousResponseID)
	} else {
		entry, ok = loadResponsesReplay(accountID, previousResponseID)
	}
	if !ok {
		return &ResponsesRewriteError{Message: fmt.Sprintf("previous_response_id %q was not found or has expired", previousResponseID)}
	}
	if len(entry.ReplayItems) == 0 {
		return &ResponsesRewriteError{Message: fmt.Sprintf("previous_response_id %q cannot be replayed yet; only tool continuation responses are supported", previousResponseID)}
	}

	currentInput, err := normalizeResponsesContinuationInput(payload["input"])
	if err != nil {
		return &ResponsesRewriteError{Message: fmt.Sprintf("invalid continuation input: %v", err)}
	}
	if err := validateResponsesContinuationToolOutputs(entry, currentInput); err != nil {
		return err
	}

	rebuiltInput := make([]interface{}, 0, len(entry.Input)+len(entry.ReplayItems)+len(currentInput))
	rebuiltInput = append(rebuiltInput, cloneJSONArray(entry.Input)...)
	rebuiltInput = append(rebuiltInput, cloneJSONArray(entry.ReplayItems)...)
	rebuiltInput = append(rebuiltInput, currentInput...)
	payload["input"] = rebuiltInput
	delete(payload, "previous_response_id")
	return nil
}

func storeResponsesReplayFromBody(accountID string, requestBody []byte, responseBody []byte) {
	input, err := extractReplayRequestInput(requestBody)
	if err != nil || len(input) == 0 {
		return
	}

	var payload struct {
		ID     string        `json:"id"`
		Output []interface{} `json:"output"`
	}
	if err := json.Unmarshal(responseBody, &payload); err != nil || strings.TrimSpace(payload.ID) == "" {
		return
	}

	replayItems := extractReplayItems(payload.Output)
	if len(replayItems) == 0 {
		return
	}
	storeResponsesReplay(accountID, payload.ID, input, replayItems)
}

func storeResponsesReplayFromStream(accountID string, requestBody []byte, capture responsesStreamCapture) {
	if strings.TrimSpace(capture.ResponseID) == "" || len(capture.ReplayItems) == 0 {
		return
	}
	input, err := extractReplayRequestInput(requestBody)
	if err != nil || len(input) == 0 {
		return
	}
	storeResponsesReplay(accountID, capture.ResponseID, input, capture.ReplayItems)
}

func (capture *responsesStreamCapture) absorbLine(line []byte) {
	trimmed := bytes.TrimSpace(line)
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return
	}
	payloadBytes := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
	if len(payloadBytes) == 0 || bytes.Equal(payloadBytes, []byte("[DONE]")) {
		return
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return
	}

	if responseID := extractResponseID(payload); strings.TrimSpace(responseID) != "" {
		capture.ResponseID = responseID
	}
	if responseReplayItems := extractStreamResponseReplayItems(payload); len(responseReplayItems) > 0 {
		capture.ReplayItems = responseReplayItems
		return
	}
	if item := extractStreamOutputItem(payload); item != nil && isReplayableResponseItem(item) {
		capture.ReplayItems = append(capture.ReplayItems, cloneJSONValue(item))
	}
}

func extractReplayRequestInput(body []byte) ([]interface{}, error) {
	var payload struct {
		Input interface{} `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	return normalizeResponsesInput(payload.Input)
}

func normalizeResponsesInput(input interface{}) ([]interface{}, error) {
	switch typed := input.(type) {
	case nil:
		return nil, nil
	case []interface{}:
		return cloneJSONArray(typed), nil
	case map[string]interface{}:
		return []interface{}{cloneJSONValue(typed)}, nil
	case string:
		return []interface{}{map[string]interface{}{"role": "user", "content": typed}}, nil
	default:
		return nil, fmt.Errorf("unsupported input type %T", input)
	}
}

func normalizeResponsesContinuationInput(input interface{}) ([]interface{}, error) {
	items, err := normalizeResponsesInput(input)
	if err != nil {
		return nil, err
	}
	return scrubResponsesContinuationItems(items), nil
}

func scrubResponsesContinuationItems(items []interface{}) []interface{} {
	if len(items) == 0 {
		return nil
	}
	cleaned := make([]interface{}, 0, len(items))
	for _, item := range items {
		cleanedItem, keep := scrubResponsesContinuationItem(item)
		if !keep {
			continue
		}
		cleaned = append(cleaned, cleanedItem)
	}
	if len(cleaned) == 0 {
		return nil
	}
	return cleaned
}

func scrubResponsesContinuationItem(item interface{}) (interface{}, bool) {
	cloned := cloneJSONValue(item)
	payload, ok := cloned.(map[string]interface{})
	if !ok {
		return cloned, true
	}
	itemType, _ := payload["type"].(string)
	itemType = strings.TrimSpace(strings.ToLower(itemType))
	if itemType == "item_reference" || strings.HasSuffix(itemType, "_reference") {
		return nil, false
	}
	delete(payload, "id")
	if content, ok := payload["content"].([]interface{}); ok {
		filtered := make([]interface{}, 0, len(content))
		for _, entry := range content {
			contentItem, keep := scrubResponsesContinuationItem(entry)
			if !keep {
				continue
			}
			filtered = append(filtered, contentItem)
		}
		payload["content"] = filtered
	}
	return payload, true
}

func validateResponsesContinuationToolOutputs(entry *responsesReplayEntry, currentInput []interface{}) error {
	pending := collectResponsesFunctionCallIDs(entry.ReplayItems)
	if len(pending) == 0 {
		return nil
	}
	provided := collectResponsesFunctionCallOutputIDs(currentInput)
	missing := diffExpectedStrings(pending, provided)
	unexpected := diffExpectedStrings(provided, pending)
	if len(missing) == 0 && len(unexpected) == 0 {
		return nil
	}

	message := fmt.Sprintf("tool continuation mismatch: expected function call outputs for %s", formatIDList(pending))
	if len(missing) > 0 {
		message += fmt.Sprintf("; missing %s", formatIDList(missing))
	}
	if len(unexpected) > 0 {
		message += fmt.Sprintf("; unexpected %s", formatIDList(unexpected))
	}
	return &ResponsesRewriteError{Message: message}
}

func collectResponsesFunctionCallIDs(items []interface{}) []string {
	ids := make([]string, 0)
	for _, item := range items {
		payload, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType, _ := payload["type"].(string)
		if !isResponsesFunctionCallType(itemType) {
			continue
		}
		if callID, _ := payload["call_id"].(string); strings.TrimSpace(callID) != "" {
			ids = append(ids, callID)
		}
	}
	return uniqueTrimmed(ids)
}

func collectResponsesFunctionCallOutputIDs(items []interface{}) []string {
	ids := make([]string, 0)
	for _, item := range items {
		payload, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType, _ := payload["type"].(string)
		if !isResponsesFunctionCallOutputType(itemType) {
			continue
		}
		if callID, _ := payload["call_id"].(string); strings.TrimSpace(callID) != "" {
			ids = append(ids, callID)
		}
	}
	return uniqueTrimmed(ids)
}

func extractReplayItems(output []interface{}) []interface{} {
	replayItems := make([]interface{}, 0, len(output))
	for _, item := range output {
		if isReplayableResponseItem(item) {
			replayItems = append(replayItems, cloneJSONValue(item))
		}
	}
	return replayItems
}

func isReplayableResponseItem(item interface{}) bool {
	payload, ok := item.(map[string]interface{})
	if !ok {
		return false
	}
	itemType, _ := payload["type"].(string)
	switch itemType {
	case "reasoning", "function_call", "custom_tool_call":
		return true
	default:
		return false
	}
}

func isResponsesFunctionCallType(itemType string) bool {
	switch strings.TrimSpace(strings.ToLower(itemType)) {
	case "function_call", "custom_tool_call":
		return true
	default:
		return false
	}
}

func isResponsesFunctionCallOutputType(itemType string) bool {
	switch strings.TrimSpace(strings.ToLower(itemType)) {
	case "function_call_output", "custom_tool_call_output":
		return true
	default:
		return false
	}
}

func extractStreamOutputItem(payload map[string]interface{}) interface{} {
	item, ok := payload["item"].(map[string]interface{})
	if !ok {
		return nil
	}
	return item
}

func extractStreamResponseReplayItems(payload map[string]interface{}) []interface{} {
	response, ok := payload["response"].(map[string]interface{})
	if !ok {
		return nil
	}
	output, ok := response["output"].([]interface{})
	if !ok {
		return nil
	}
	return extractReplayItems(output)
}

func extractResponseID(payload map[string]interface{}) string {
	if responseID, _ := payload["response_id"].(string); strings.TrimSpace(responseID) != "" {
		return responseID
	}
	if id, _ := payload["id"].(string); strings.TrimSpace(id) != "" {
		return id
	}
	response, ok := payload["response"].(map[string]interface{})
	if !ok {
		return ""
	}
	id, _ := response["id"].(string)
	return strings.TrimSpace(id)
}

func storeResponsesReplay(accountID string, responseID string, input []interface{}, replayItems []interface{}) {
	responseID = strings.TrimSpace(responseID)
	accountID = strings.TrimSpace(accountID)
	if responseID == "" || accountID == "" {
		return
	}
	ensureDurableContinuationStateLoaded()

	now := time.Now()
	responsesReplayCache.mu.Lock()
	responsesReplayCache.entries[responseID] = &responsesReplayEntry{
		AccountID:   accountID,
		Input:       cloneJSONArray(input),
		ReplayItems: cloneJSONArray(replayItems),
		CreatedAt:   now,
		AccessedAt:  now,
	}
	pruneResponsesReplayLocked(now)
	responsesReplayCache.mu.Unlock()

	persistDurableContinuationState()
}

func loadResponsesReplay(accountID string, responseID string) (*responsesReplayEntry, bool) {
	ensureDurableContinuationStateLoaded()

	now := time.Now()
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()

	pruneResponsesReplayLocked(now)
	entry, ok := responsesReplayCache.entries[responseID]
	if !ok || entry.AccountID != accountID {
		return nil, false
	}
	entry.AccessedAt = now
	return &responsesReplayEntry{
		AccountID:   entry.AccountID,
		Input:       cloneJSONArray(entry.Input),
		ReplayItems: cloneJSONArray(entry.ReplayItems),
		CreatedAt:   entry.CreatedAt,
		AccessedAt:  entry.AccessedAt,
	}, true
}

func loadResponsesReplayAny(responseID string) (*responsesReplayEntry, bool) {
	ensureDurableContinuationStateLoaded()

	now := time.Now()
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()

	pruneResponsesReplayLocked(now)
	entry, ok := responsesReplayCache.entries[responseID]
	if !ok {
		return nil, false
	}
	entry.AccessedAt = now
	return &responsesReplayEntry{
		AccountID:   entry.AccountID,
		Input:       cloneJSONArray(entry.Input),
		ReplayItems: cloneJSONArray(entry.ReplayItems),
		CreatedAt:   entry.CreatedAt,
		AccessedAt:  entry.AccessedAt,
	}, true
}

func LookupResponsesReplayAccount(responseID string) (string, bool) {
	ensureDurableContinuationStateLoaded()

	now := time.Now()
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()

	pruneResponsesReplayLocked(now)
	entry, ok := responsesReplayCache.entries[responseID]
	if !ok {
		return "", false
	}
	entry.AccessedAt = now
	return entry.AccountID, true
}

func CanReplayResponsesContinuation(accountID string, responseID string) bool {
	entry, ok := loadResponsesReplay(accountID, responseID)
	return ok && len(entry.ReplayItems) > 0
}

func pruneResponsesReplayLocked(now time.Time) {
	for responseID, entry := range responsesReplayCache.entries {
		if now.Sub(entry.CreatedAt) > responsesReplayTTL {
			delete(responsesReplayCache.entries, responseID)
		}
	}
	if len(responsesReplayCache.entries) <= responsesReplayMaxEntries {
		return
	}

	type replayKey struct {
		ResponseID string
		AccessedAt time.Time
	}
	keys := make([]replayKey, 0, len(responsesReplayCache.entries))
	for responseID, entry := range responsesReplayCache.entries {
		keys = append(keys, replayKey{ResponseID: responseID, AccessedAt: entry.AccessedAt})
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].AccessedAt.Before(keys[j].AccessedAt)
	})
	for len(responsesReplayCache.entries) > responsesReplayMaxEntries && len(keys) > 0 {
		delete(responsesReplayCache.entries, keys[0].ResponseID)
		keys = keys[1:]
	}
}

// orphanDegradeMaxToolItems bounds how many function_call / function_call_output
// items we serialize into text during orphan-continuation degrade. A single long
// agentic session can accumulate hundreds of tool turns; serializing all of them
// verbatim has produced prompts in the multi-million-token range, blowing past
// the model's context window. Keeping only the tail preserves recency (what the
// assistant most recently did / saw) while insurance-capping prompt size.
const orphanDegradeMaxToolItems = 100

// DegradeOrphanContinuationPayload rewrites function_call / function_call_output
// items in the request payload into plain assistant/user text messages so GitHub
// Copilot accepts the request statelessly. Used as a last-resort fallback when
// upstream rejects the continuation because the call_ids were minted by a
// different backend (e.g. a different relay the user just switched from).
//
// When the tool-item count exceeds orphanDegradeMaxToolItems, the older items
// are dropped and replaced by a single placeholder message, so the resulting
// prompt fits within the model's context window.
//
// Returns the re-marshaled body, the number of items that were degraded, and
// any error encountered during JSON processing.
func DegradeOrphanContinuationPayload(body []byte) ([]byte, int, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, 0, err
	}

	items, err := normalizeResponsesInput(payload["input"])
	if err != nil || len(items) == 0 {
		return body, 0, err
	}

	// First pass: count tool items so we know how many of the earliest
	// ones to drop in order to stay at or below the cap.
	totalToolItems := 0
	for _, item := range items {
		mapped, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(fmt.Sprint(mapped["type"])))
		if itemType == "function_call" || itemType == "function_call_output" {
			totalToolItems++
		}
	}
	dropTool := 0
	if totalToolItems > orphanDegradeMaxToolItems {
		dropTool = totalToolItems - orphanDegradeMaxToolItems
	}

	rewritten := make([]interface{}, 0, len(items))
	changed := 0
	toolsDropped := 0
	markerEmitted := false
	for _, item := range items {
		mapped, ok := item.(map[string]interface{})
		if !ok {
			rewritten = append(rewritten, item)
			continue
		}
		itemType := strings.TrimSpace(strings.ToLower(fmt.Sprint(mapped["type"])))
		if itemType == "reasoning" {
			// Drop reasoning items: their encrypted_content is ciphertext
			// bound to the original response, and upstream rejects the payload
			// with "input item does not belong to this connection" if any
			// reasoning item survives the degrade.
			changed++
			continue
		}
		isTool := itemType == "function_call" || itemType == "function_call_output"
		if isTool && toolsDropped < dropTool {
			toolsDropped++
			continue
		}
		if isTool && dropTool > 0 && !markerEmitted {
			rewritten = append(rewritten, map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": fmt.Sprintf("[%d earlier tool calls and outputs omitted to fit context]", dropTool),
					},
				},
			})
			markerEmitted = true
		}
		switch itemType {
		case "function_call":
			name := strings.TrimSpace(fmt.Sprint(mapped["name"]))
			if name == "" {
				name = "tool"
			}
			arguments := strings.TrimSpace(fmt.Sprint(mapped["arguments"]))
			text := fmt.Sprintf("[Previous tool call: %s] arguments:\n%s", name, arguments)
			rewritten = append(rewritten, map[string]interface{}{
				"type":   "message",
				"role":   "assistant",
				"status": "completed",
				"content": []interface{}{
					map[string]interface{}{
						"type":        "output_text",
						"text":        text,
						"annotations": []interface{}{},
						"logprobs":    []interface{}{},
					},
				},
			})
			changed++
		case "function_call_output":
			callID := strings.TrimSpace(fmt.Sprint(mapped["call_id"]))
			output := strings.TrimSpace(fmt.Sprint(mapped["output"]))
			label := "previous tool"
			if callID != "" {
				label = fmt.Sprintf("previous tool (call_id=%s)", callID)
			}
			text := fmt.Sprintf("[Output from %s]\n%s", label, output)
			rewritten = append(rewritten, map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": text,
					},
				},
			})
			changed++
		default:
			rewritten = append(rewritten, item)
		}
	}

	if changed == 0 && dropTool == 0 {
		return body, 0, nil
	}
	payload["input"] = scrubResponsesContinuationItems(rewritten)
	delete(payload, "previous_response_id")

	out, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, err
	}
	return out, changed, nil
}

func cloneJSONArray(items []interface{}) []interface{} {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]interface{}, 0, len(items))
	for _, item := range items {
		cloned = append(cloned, cloneJSONValue(item))
	}
	return cloned
}

func cloneJSONValue(value interface{}) interface{} {
	body, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var cloned interface{}
	if err := json.Unmarshal(body, &cloned); err != nil {
		return value
	}
	return cloned
}
