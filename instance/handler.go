package instance

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	"copilot-go/anthropic"
	"copilot-go/config"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

// DoCompletionsProxy performs the upstream request for completions and returns the raw response.
// The caller is responsible for closing resp.Body.
func DoCompletionsProxy(_ *gin.Context, state *config.State, bodyBytes []byte) (*http.Response, error) {
	bodyBytes, extraHeaders, hasVision, err := normalizeCompletionsPayload(state, bodyBytes)
	if err != nil {
		return nil, err
	}
	return ProxyRequestWithBytes(state, "POST", "/chat/completions", bodyBytes, extraHeaders, hasVision)
}

// ForwardCompletionsResponse writes the upstream response to the client.
func ForwardCompletionsResponse(c *gin.Context, resp *http.Response) {
	defer func() { _ = resp.Body.Close() }()

	contentType := resp.Header.Get("Content-Type")
	isStream := strings.Contains(contentType, "text/event-stream")

	if isStream {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")
		c.Status(resp.StatusCode)

		reqID, _ := c.Get("respReqID")
		accountID, _ := c.Get("respAccountID")
		reqIDStr, _ := reqID.(string)
		accountIDStr, _ := accountID.(string)

		reader := bufio.NewReaderSize(resp.Body, 10*1024*1024)
		var probe streamTailProbe
		startedAt := time.Now()
		writeErrOccurred := false
		var exitErr error
		c.Stream(func(w io.Writer) bool {
			line, err := reader.ReadBytes('\n')
			if len(line) > 0 {
				probe.observe(line)
				if _, writeErr := w.Write(line); writeErr != nil {
					writeErrOccurred = true
					exitErr = writeErr
					return false
				}
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			if err != nil {
				if err != io.EOF {
					log.Printf("Stream read error: %v", err)
				}
				exitErr = err
				return false
			}
			return true
		})
		probe.log("completions", reqIDStr, accountIDStr, time.Since(startedAt), exitErr, writeErrOccurred)
	} else {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read response"})
			return
		}
		c.Data(resp.StatusCode, "application/json", body)
	}
}

// ModelsHandler returns cached models with display ID mapping.
func ModelsHandler(c *gin.Context, state *config.State) {
	state.RLock()
	models := state.Models
	state.RUnlock()

	if models == nil {
		c.JSON(http.StatusOK, config.ModelsResponse{
			Object: "list",
			Data:   []config.ModelEntry{},
		})
		return
	}

	mapped := config.ModelsResponse{
		Object: models.Object,
		Data:   make([]config.ModelEntry, len(models.Data)),
	}
	for i, m := range models.Data {
		mapped.Data[i] = config.ModelEntry{
			ID:           store.ToDisplayID(m.ID),
			Object:       m.Object,
			Created:      m.Created,
			OwnedBy:      m.OwnedBy,
			Name:         m.Name,
			Version:      m.Version,
			Vendor:       m.Vendor,
			Capabilities: m.Capabilities,
		}
	}

	c.JSON(http.StatusOK, mapped)
}

// DoEmbeddingsProxy performs the upstream request for embeddings.
func DoEmbeddingsProxy(state *config.State, bodyBytes []byte) (*http.Response, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err == nil {
		if model, ok := payload["model"].(string); ok {
			payload["model"] = store.ToCopilotID(model)
			bodyBytes, _ = json.Marshal(payload)
		}
	}

	return ProxyRequestWithBytes(state, "POST", "/embeddings", bodyBytes, nil, false)
}

// ForwardEmbeddingsResponse writes the upstream embeddings response to the client.
func ForwardEmbeddingsResponse(c *gin.Context, resp *http.Response) {
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read response"})
		return
	}
	c.Data(resp.StatusCode, "application/json", body)
}

// DoMessagesProxy performs the upstream request for Anthropic messages.
// Returns the raw response. bodyBytes is the original Anthropic payload.
func DoMessagesProxy(c *gin.Context, accountID string, state *config.State, bodyBytes []byte) (*http.Response, copilotTurnRequest, error) {
	return doMessagesProxy(c, accountID, state, bodyBytes, false)
}

func DoDetachedMessagesProxy(c *gin.Context, accountID string, state *config.State, bodyBytes []byte) (*http.Response, copilotTurnRequest, error) {
	return doMessagesProxy(c, accountID, state, bodyBytes, true)
}

func doMessagesProxy(c *gin.Context, accountID string, state *config.State, bodyBytes []byte, detached bool) (*http.Response, copilotTurnRequest, error) {
	var anthropicPayload anthropic.AnthropicMessagesPayload
	if err := json.Unmarshal(bodyBytes, &anthropicPayload); err != nil {
		return nil, copilotTurnRequest{}, fmt.Errorf("invalid request: %v", err)
	}

	if err := validateAnthropicToolContinuation(anthropicPayload); err != nil {
		return nil, copilotTurnRequest{}, err
	}

	// Auto-fill max_tokens from model capabilities if not provided
	if anthropicPayload.MaxTokens == 0 {
		copilotModelID := anthropic.NormalizeAnthropicModel(store.ToCopilotID(anthropicPayload.Model))
		if limit := lookupMaxOutputTokens(state, copilotModelID); limit > 0 {
			anthropicPayload.MaxTokens = limit
		}
	}

	hasVision := checkVisionContent(anthropicPayload)
	openaiPayload := anthropic.TranslateToOpenAI(anthropicPayload)

	openaiBytes, err := json.Marshal(openaiPayload)
	if err != nil {
		return nil, copilotTurnRequest{}, fmt.Errorf("failed to marshal request: %v", err)
	}

	turnRequest := buildMessagesTurnRequest(accountID, anthropicPayload)
	if detached {
		turnRequest = newCopilotTurnRequest(copilotInteractionTypeUser)
	}

	resp, proxyErr := ProxyRequestWithBytesCtx(c.Request.Context(), state, "POST", "/chat/completions", openaiBytes, turnRequest.Headers(), hasVision)
	return resp, turnRequest, proxyErr
}

// ForwardMessagesResponse writes the upstream response to the client in Anthropic format.
// originalBody is the original Anthropic request (used to determine stream mode).
func ForwardMessagesResponse(c *gin.Context, accountID string, turnRequest copilotTurnRequest, resp *http.Response, originalBody []byte) {
	defer func() { _ = resp.Body.Close() }()

	var anthropicPayload anthropic.AnthropicMessagesPayload
	if err := json.Unmarshal(originalBody, &anthropicPayload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}

	if anthropicPayload.Stream {
		handleAnthropicStream(c, accountID, turnRequest, resp)
	} else {
		handleAnthropicNonStream(c, accountID, turnRequest, resp)
	}
}

func handleAnthropicNonStream(c *gin.Context, accountID string, turnRequest copilotTurnRequest, resp *http.Response) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read response"})
		return
	}

	if resp.StatusCode != 200 {
		c.Data(resp.StatusCode, "application/json", body)
		return
	}

	var openaiResp anthropic.ChatCompletionResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to parse upstream response"})
		return
	}

	storeMessageToolCallTurnContext(accountID, collectToolCallIDsFromChatCompletion(openaiResp), turnRequest.Context)

	anthropicResp := anthropic.TranslateToAnthropic(openaiResp)
	c.JSON(http.StatusOK, anthropicResp)
}

func handleAnthropicStream(c *gin.Context, accountID string, turnRequest copilotTurnRequest, resp *http.Response) {
	// If upstream returned an error, translate it properly instead of trying to SSE-parse
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("[Stream] Upstream returned status %d: %s", resp.StatusCode, string(body))
		c.Data(resp.StatusCode, "application/json", body)
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Header("Transfer-Encoding", "chunked")
	c.Status(http.StatusOK)

	w := c.Writer
	flusher, hasFlusher := w.(http.Flusher)
	clientGone := c.Request.Context().Done()

	state := anthropic.NewStreamState()
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 10*1024*1024), 10*1024*1024)

	for scanner.Scan() {
		select {
		case <-clientGone:
			log.Printf("[Stream] Client disconnected, stopping stream")
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			storeMessageToolCallTurnContext(accountID, collectToolCallIDsFromStreamState(state), turnRequest.Context)
			if err := writeSSE(w, "message_stop", map[string]string{"type": "message_stop"}); err != nil {
				log.Printf("[Stream] Write error on message_stop: %v", err)
				return
			}
			if hasFlusher {
				flusher.Flush()
			}
			return
		}

		var chunk anthropic.ChatCompletionResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("[Stream] Failed to parse SSE chunk: %v", err)
			continue
		}

		events := anthropic.TranslateChunkToAnthropicEvents(chunk, state)
		for _, event := range events {
			if err := writeSSE(w, event.Event, event.Data); err != nil {
				log.Printf("[Stream] Write error: %v", err)
				return
			}
		}
		if hasFlusher {
			flusher.Flush()
		}
	}

	storeMessageToolCallTurnContext(accountID, collectToolCallIDsFromStreamState(state), turnRequest.Context)

	if err := scanner.Err(); err != nil {
		log.Printf("[Stream] Scanner error: %v", err)
		_ = writeSSE(w, "error", map[string]interface{}{
			"type": "error",
			"error": map[string]string{
				"type":    "stream_error",
				"message": fmt.Sprintf("upstream stream error: %v", err),
			},
		})
	} else {
		log.Printf("[Stream] Upstream closed without [DONE], sending message_stop")
		_ = writeSSE(w, "message_stop", map[string]string{"type": "message_stop"})
	}
	if hasFlusher {
		flusher.Flush()
	}
}

func normalizeCompletionsPayload(state *config.State, bodyBytes []byte) ([]byte, http.Header, bool, error) {
	extraHeaders := make(http.Header)
	extraHeaders.Set("X-Initiator", "user")

	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return bodyBytes, extraHeaders, false, nil
	}

	if model, ok := payload["model"].(string); ok {
		payload["model"] = store.ToCopilotID(model)
	}
	if err := normalizeParallelToolCallsSetting(payload, "/v1/chat/completions"); err != nil {
		return nil, nil, false, &MessagesRewriteError{Message: err.Error()}
	}

	// Normalize token limit field for GPT-5 family, which requires max_completion_tokens.
	modelID, _ := payload["model"].(string)
	if usesResponsesMaxCompletionTokens(modelID) {
		if _, hasMaxComp := payload["max_completion_tokens"]; !hasMaxComp {
			if maxTokens, ok := payload["max_tokens"]; ok {
				payload["max_completion_tokens"] = maxTokens
				delete(payload, "max_tokens")
			} else if limit := lookupMaxOutputTokens(state, modelID); limit > 0 {
				payload["max_completion_tokens"] = limit
			}
		} else {
			delete(payload, "max_tokens")
		}
	} else if _, hasMax := payload["max_tokens"]; !hasMax {
		if _, hasMaxComp := payload["max_completion_tokens"]; !hasMaxComp {
			if limit := lookupMaxOutputTokens(state, modelID); limit > 0 {
				payload["max_tokens"] = limit
			}
		}
	}

	extraHeaders.Set("X-Initiator", currentCompletionsInitiator(payload["messages"]))

	hasVision := checkCompletionsVision(payload["messages"])

	normalized, err := json.Marshal(payload)
	if err != nil {
		return bodyBytes, extraHeaders, hasVision, nil
	}
	return normalized, extraHeaders, hasVision, nil
}

// lookupMaxOutputTokens finds the max_output_tokens for a model from cached capabilities.
func lookupMaxOutputTokens(state *config.State, modelID string) int {
	if state == nil || modelID == "" {
		return 0
	}
	state.RLock()
	models := state.Models
	state.RUnlock()
	if models == nil {
		return 0
	}
	for _, m := range models.Data {
		if m.ID == modelID && m.Capabilities != nil && m.Capabilities.Limits.MaxOutputTokens > 0 {
			return m.Capabilities.Limits.MaxOutputTokens
		}
	}
	return 0
}

// checkCompletionsVision checks OpenAI-format messages for image_url content.
func checkCompletionsVision(rawMessages interface{}) bool {
	messages, ok := rawMessages.([]interface{})
	if !ok {
		return false
	}
	for _, rawMsg := range messages {
		msg, ok := rawMsg.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := msg["content"]
		if !ok {
			continue
		}
		parts, ok := content.([]interface{})
		if !ok {
			continue
		}
		for _, rawPart := range parts {
			part, ok := rawPart.(map[string]interface{})
			if !ok {
				continue
			}
			if partType, _ := part["type"].(string); partType == "image_url" {
				return true
			}
		}
	}
	return false
}

func writeSSE(w io.Writer, event string, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonData))
	return err
}

// CountTokensHandler provides a simplified token count estimation.
func CountTokensHandler(c *gin.Context, _ *config.State) {
	anthropicBeta := c.GetHeader("anthropic-beta")

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}

	var payload anthropic.AnthropicMessagesPayload
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}

	openaiPayload := anthropic.TranslateToOpenAI(payload)
	inputTokens, outputTokens := estimateOpenAITokens(openaiPayload)

	if len(payload.Tools) > 0 && !hasClaudeCodeMCPTools(anthropicBeta, payload.Tools) {
		switch {
		case strings.HasPrefix(payload.Model, "claude"):
			inputTokens += 346
		case strings.HasPrefix(payload.Model, "grok"):
			inputTokens += 480
		}
	}

	finalTokenCount := inputTokens + outputTokens
	switch {
	case strings.HasPrefix(payload.Model, "claude"):
		finalTokenCount = int(math.Round(float64(finalTokenCount) * 1.15))
	case strings.HasPrefix(payload.Model, "grok"):
		finalTokenCount = int(math.Round(float64(finalTokenCount) * 1.03))
	}
	finalTokenCount = maxTokenCount(finalTokenCount, 1)

	c.JSON(http.StatusOK, gin.H{
		"input_tokens": finalTokenCount,
	})
}

func estimateOpenAITokens(payload anthropic.ChatCompletionsPayload) (int, int) {
	inputTokens := 0
	outputTokens := 0

	for _, msg := range payload.Messages {
		tokens := estimateJSONTokens(msg) + 3
		if msg.Role == "assistant" {
			outputTokens += tokens
		} else {
			inputTokens += tokens
		}
	}

	if len(payload.Tools) > 0 {
		inputTokens += estimateJSONTokens(payload.Tools)
	}

	return inputTokens, outputTokens
}

func estimateJSONTokens(v interface{}) int {
	data, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	tokens := len(data) / 4
	if tokens < 1 {
		return 1
	}
	return tokens
}

func maxTokenCount(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func hasClaudeCodeMCPTools(anthropicBeta string, tools []anthropic.AnthropicTool) bool {
	if !strings.HasPrefix(anthropicBeta, "claude-code") {
		return false
	}
	for _, tool := range tools {
		if strings.HasPrefix(tool.Name, "mcp__") {
			return true
		}
	}
	return false
}

func checkVisionContent(payload anthropic.AnthropicMessagesPayload) bool {
	for _, msg := range payload.Messages {
		blocks := anthropic.ParseContentBlocksPublic(msg.Content)
		for _, b := range blocks {
			if b.Type == "image" {
				return true
			}
		}
	}
	return false
}
