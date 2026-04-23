package instance

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/config"
	"copilot-go/instance/orphan_translate"
	"copilot-go/store"
)

// DoOrphanTranslateMessagesProxy handles an orphan /v1/responses request by
// translating the payload to Anthropic Messages in-process, forwarding it to
// the resolved account's sidecar worker on /v1/messages, and wrapping the
// Messages SSE reply back into Responses-SSE events on the way out.
//
// Rationale: Copilot's /v1/chat/completions endpoint rejects the gpt-5 family
// and Anthropic models with
//   {"error":{"message":"Please use `/v1/responses` or `/v1/messages` API"}}
// — the DoOrphanTranslateResponsesProxy chat path therefore only works for
// gpt-4o-style models. /v1/messages is stateless (like /v1/chat/completions)
// but accepts both gpt-5.4 AND claude-*, which is the actual traffic mix we
// need to rescue from the cross-relay orphan case.
//
// The wrapped response body is a Responses-SSE stream indistinguishable from
// Copilot's native /v1/responses output (modulo encrypted_content, which
// orphan migration can't preserve regardless of route).
// forwardResponsesStream and the absorbLine sticky-cache tee keep working
// unchanged downstream.
//
// Only effective when config.ResponsesOrphanTranslate()=="on" AND the
// resolved account has a WorkerURL. handler/proxy.go guards both and routes
// by model family (gpt-5*/claude-* → messages path, else chat path).
func DoOrphanTranslateMessagesProxy(accountID string, state *config.State, bodyBytes []byte) (*http.Response, []byte, copilotTurnRequest, error) {
	turnRequest := newCopilotTurnRequest(copilotInteractionTypeUser)
	turnRequest.CacheSource = "orphan_translate_messages_fresh"

	workerURL := ""
	if acct, _ := store.GetAccount(accountID); acct != nil {
		workerURL = strings.TrimSpace(acct.WorkerURL)
	}
	if workerURL == "" {
		return nil, bodyBytes, turnRequest, fmt.Errorf("orphan_translate_messages requires account.WorkerURL, account=%s has none", accountID)
	}

	// Force stream=true — the output translator only handles SSE. Capture the
	// requested model for the synthesized response.completed payload.
	model := ""
	if len(bodyBytes) > 0 {
		var src map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &src); err == nil {
			src["stream"] = true
			if s, ok := src["model"].(string); ok {
				model = s
			}
			if bb, err := json.Marshal(src); err == nil {
				bodyBytes = bb
			}
		}
	}

	translateStart := time.Now()
	messagesBody, stats, err := orphan_translate.ResponsesToMessages(bodyBytes)
	translateMs := time.Since(translateStart).Milliseconds()
	if err != nil {
		log.Printf("[responses account=%s] orphan_translate_messages request-translation failed translate_ms=%d: %v",
			accountID, translateMs, err)
		return nil, bodyBytes, turnRequest, err
	}
	log.Printf("[responses account=%s] orphan_translate=messages worker=%s input_items=%d messages=%d dropped_reasoning=%d tools_in=%d tools_out=%d dropped_tools=%d translate_ms=%d",
		accountID, workerURL,
		stats.InputItems, stats.Messages, stats.DroppedReasoning,
		stats.ToolsIn, stats.ToolsOut, stats.DroppedTools,
		translateMs)

	start := time.Now()
	resp, err := ProxyRequestViaWorker(context.Background(), workerURL, "POST", "/v1/messages", messagesBody, nil)
	workerMs := time.Since(start).Milliseconds()
	if err != nil {
		log.Printf("[responses account=%s] orphan_translate_messages worker call failed worker=%s worker_ms=%d: %v",
			accountID, workerURL, workerMs, err)
		return resp, bodyBytes, turnRequest, err
	}
	log.Printf("[responses account=%s] orphan_translate_messages worker=%s worker_ms=%d status=%d ct=%q",
		accountID, workerURL, workerMs, resp.StatusCode, resp.Header.Get("Content-Type"))

	// On non-2xx, let the caller see the raw error body. isRetryableStatus +
	// disableOnFatalUpstream inspect StatusCode and sometimes sniff the body,
	// and the response-translator would synthesize a fake `response.failed`
	// that hides the real 401 / 429 / 5xx from those branches.
	//
	// We drain the body once so we can log a snippet, then re-wrap it as an
	// io.NopCloser around the captured bytes so the retry loop still sees the
	// original payload verbatim.
	if resp.StatusCode >= 300 {
		errBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		snippet := string(errBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "…(truncated)"
		}
		log.Printf("[responses account=%s] orphan_translate_messages worker non-2xx status=%d read_err=%v body=%s",
			accountID, resp.StatusCode, readErr, snippet)
		resp.Body = io.NopCloser(bytes.NewReader(errBody))
		return resp, bodyBytes, turnRequest, nil
	}

	resp.Body = orphan_translate.NewResponsesReaderFromMessages(resp.Body, model)
	resp.Header.Set("Content-Type", "text/event-stream")
	resp.Header.Del("Content-Length")

	return resp, bodyBytes, turnRequest, nil
}
