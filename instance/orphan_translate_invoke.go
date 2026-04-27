package instance

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/config"
	"copilot-go/instance/orphan_translate"
	"copilot-go/store"
)

func mergeHeaderSets(base, extra http.Header) http.Header {
	out := make(http.Header)
	for k, vs := range base {
		for _, v := range vs {
			out.Add(k, v)
		}
	}
	for k, vs := range extra {
		out.Del(k)
		for _, v := range vs {
			out.Add(k, v)
		}
	}
	return out
}

// DoOrphanTranslateResponsesProxy handles an orphan /v1/responses request by
// translating the payload to chat/completions in-process, forwarding it to
// the resolved account's sidecar worker on /v1/chat/completions, and
// wrapping the chat SSE reply back into Responses-SSE events on the way out.
//
// Rationale: Copilot's /v1/responses endpoint is stateful (it session-checks
// tool_call_ids / previous_response_id against the connection that minted
// them). In the cross-relay migration case those ids were minted by another
// relay, so the upstream rejects them. /v1/chat/completions is stateless
// with respect to tool_call_ids, so translating to chat lets the orphan turn
// complete without upstream session bookkeeping.
//
// The wrapped response body is a Responses-SSE stream indistinguishable
// from Copilot's native /v1/responses output (modulo encrypted_content,
// which orphan migration can't preserve regardless of route).
// forwardResponsesStream and the absorbLine sticky-cache tee keep working
// unchanged downstream.
//
// When the resolved account has no WorkerURL, the gateway falls back to a
// direct upstream /chat/completions proxy using the translated payload.
func DoOrphanTranslateResponsesProxy(accountID string, state *config.State, bodyBytes []byte) (*http.Response, []byte, copilotTurnRequest, error) {
	turnRequest := newCopilotTurnRequest(copilotInteractionTypeUser)
	turnRequest.CacheSource = "orphan_translate_fresh"

	workerURL := ""
	if acct, _ := store.GetAccount(accountID); acct != nil {
		workerURL = strings.TrimSpace(acct.WorkerURL)
	}
	if workerURL == "" {
		log.Printf("[responses account=%s] orphan_translate worker unavailable — falling back to direct upstream chat/completions", accountID)
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
	chatBody, stats, translateErr := orphan_translate.ResponsesToChat(bodyBytes)
	translateMs := time.Since(translateStart).Milliseconds()
	if translateErr != nil {
		log.Printf("[responses account=%s] orphan_translate request-translation failed translate_ms=%d: %v",
			accountID, translateMs, translateErr)
		return nil, bodyBytes, turnRequest, translateErr
	}
	log.Printf("[responses account=%s] orphan_translate=on worker=%s input_items=%d messages=%d dropped_reasoning=%d tools_in=%d tools_out=%d dropped_tools=%d translate_ms=%d",
		accountID, workerURL,
		stats.InputItems, stats.Messages, stats.DroppedReasoning,
		stats.ToolsIn, stats.ToolsOut, stats.DroppedTools,
		translateMs)

	start := time.Now()
	var (
		resp   *http.Response
		err    error
		callMs int64
	)
	if workerURL != "" {
		resp, err = ProxyRequestViaWorker(context.Background(), workerURL, "POST", "/v1/chat/completions", chatBody, nil)
		callMs = time.Since(start).Milliseconds()
		if err != nil {
			log.Printf("[responses account=%s] orphan_translate worker call failed worker=%s worker_ms=%d: %v",
				accountID, workerURL, callMs, err)
			return resp, bodyBytes, turnRequest, err
		}
		log.Printf("[responses account=%s] orphan_translate worker=%s worker_ms=%d status=%d ct=%q",
			accountID, workerURL, callMs, resp.StatusCode, resp.Header.Get("Content-Type"))
	} else {
		normalizedBody, extraHeaders, hasVision, normErr := normalizeCompletionsPayload(state, chatBody)
		if normErr != nil {
			log.Printf("[responses account=%s] orphan_translate direct normalization failed: %v", accountID, normErr)
			return nil, bodyBytes, turnRequest, normErr
		}
		resp, err = ProxyRequestWithBytes(state, "POST", "/chat/completions", normalizedBody, mergeHeaderSets(turnRequest.Headers(), extraHeaders), hasVision)
		callMs = time.Since(start).Milliseconds()
		if err != nil {
			log.Printf("[responses account=%s] orphan_translate direct call failed direct_ms=%d: %v",
				accountID, callMs, err)
			return resp, bodyBytes, turnRequest, err
		}
		log.Printf("[responses account=%s] orphan_translate direct_ms=%d status=%d ct=%q",
			accountID, callMs, resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	// On non-2xx, let the caller see the raw error body. isRetryableStatus +
	// disableOnFatalUpstream inspect StatusCode and sometimes sniff the body,
	// and the response-translator would synthesize a fake `response.failed`
	// that hides the real 401 / 429 / 5xx from those branches.
	//
	// We drain the body once so we can log a snippet, then re-wrap it as an
	// io.NopCloser around the captured bytes so the retry loop still sees the
	// original payload verbatim. Without this snippet we are blind to worker
	// schema-validation failures (400 with application/json body) — those are
	// the signal that our synthesized chat/completions payload shape disagrees
	// with what the worker expects.
	if resp.StatusCode >= 300 {
		errBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		snippet := string(errBody)
		if len(snippet) > 512 {
			snippet = snippet[:512] + "…(truncated)"
		}
		log.Printf("[responses account=%s] orphan_translate worker non-2xx status=%d read_err=%v body=%s",
			accountID, resp.StatusCode, readErr, snippet)
		resp.Body = io.NopCloser(bytes.NewReader(errBody))
		return resp, bodyBytes, turnRequest, nil
	}

	resp.Body = orphan_translate.NewResponsesReader(resp.Body, model)
	resp.Header.Set("Content-Type", "text/event-stream")
	resp.Header.Del("Content-Length")

	return resp, bodyBytes, turnRequest, nil
}
