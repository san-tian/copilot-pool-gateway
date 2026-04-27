package instance

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/config"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

type responsesProxyMode int

const (
	responsesProxyModeDirect responsesProxyMode = iota
	responsesProxyModeDetachedSameAccount
	responsesProxyModeDetachedCrossAccount
)

// DoResponsesProxy forwards requests directly to GitHub Copilot /responses endpoint.
func DoResponsesProxy(accountID string, state *config.State, bodyBytes []byte, traceID string) (*http.Response, []byte, copilotTurnRequest, error) {
	return doResponsesProxy(accountID, state, bodyBytes, responsesProxyModeDirect, traceID)
}

func DoDetachedResponsesProxy(accountID string, state *config.State, bodyBytes []byte, traceID string) (*http.Response, []byte, copilotTurnRequest, error) {
	return doResponsesProxy(accountID, state, bodyBytes, responsesProxyModeDetachedSameAccount, traceID)
}

func DoDetachedResponsesProxyCrossAccount(accountID string, state *config.State, bodyBytes []byte, traceID string) (*http.Response, []byte, copilotTurnRequest, error) {
	return doResponsesProxy(accountID, state, bodyBytes, responsesProxyModeDetachedCrossAccount, traceID)
}

func doResponsesProxy(accountID string, state *config.State, bodyBytes []byte, mode responsesProxyMode, traceID string) (*http.Response, []byte, copilotTurnRequest, error) {
	turnRequest := newCopilotTurnRequest(copilotInteractionTypeUser)

	// Resolve per-account worker routing. When an account has WorkerURL set and
	// USE_WORKER_POOL is not forced off, requests go through caozhiyuan/copilot-api
	// running as a loopback sidecar. The worker ONLY does format translation
	// (OpenAI Responses-API ↔ Copilot chat). Session state — previous_response_id
	// expansion, compaction rewrite, call_id canonicalization, sticky-cache
	// lookups — stays in the Go router and runs unconditionally. Skipping those
	// in worker mode breaks continuation / cross-relay migration / orphan detection.
	var workerURL string
	if config.WorkerPoolMode() != "off" {
		if acct, _ := store.GetAccount(accountID); acct != nil {
			workerURL = strings.TrimSpace(acct.WorkerURL)
		}
	}
	useWorker := workerURL != ""

	// Convert model ID and normalize token limit fields for providers that reject max_tokens.
	var payload map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &payload); err == nil {
		previousResponseID, _ := payload["previous_response_id"].(string)
		switch mode {
		case responsesProxyModeDirect, responsesProxyModeDetachedSameAccount:
			turnRequest = buildResponsesTurnRequest(accountID, previousResponseID, payload["input"])
		default:
			turnRequest = newCopilotTurnRequest(copilotInteractionTypeUser)
			turnRequest.CacheSource = "cross_account_fresh"
		}
		if err := rewriteCompactionInput(payload); err != nil {
			return nil, nil, copilotTurnRequest{}, err
		}
		switch mode {
		case responsesProxyModeDetachedCrossAccount:
			if err := rewritePreviousResponseContinuationAnyAccount(payload); err != nil {
				return nil, nil, copilotTurnRequest{}, err
			}
		default:
			if err := rewritePreviousResponseContinuation(accountID, payload); err != nil {
				return nil, nil, copilotTurnRequest{}, err
			}
		}
		modelID := ""
		if model, ok := payload["model"].(string); ok {
			modelID = store.ToCopilotID(model)
			payload["model"] = modelID
		}

		if usesResponsesMaxCompletionTokens(modelID) {
			if _, hasMaxComp := payload["max_completion_tokens"]; !hasMaxComp {
				if maxOutput, ok := payload["max_output_tokens"]; ok {
					payload["max_completion_tokens"] = maxOutput
					delete(payload, "max_output_tokens")
				}
				if maxTokens, ok := payload["max_tokens"]; ok {
					payload["max_completion_tokens"] = maxTokens
				}
				if _, hasMaxCompNow := payload["max_completion_tokens"]; !hasMaxCompNow {
					if limit := lookupMaxOutputTokens(state, modelID); limit > 0 {
						payload["max_completion_tokens"] = limit
					}
				}
			}
			delete(payload, "max_tokens")
			delete(payload, "max_output_tokens")
		}

		bodyBytes, _ = json.Marshal(payload)
	}

	if useWorker {
		start := time.Now()
		resp, err := ProxyRequestViaWorker(context.Background(), workerURL, "POST", "/v1/responses", bodyBytes, turnRequest.Headers(), traceID)
		statusCode := 0
		if resp != nil {
			statusCode = resp.StatusCode
		}
		log.Printf("[responses account=%s trace=%s] worker=%s worker_ms=%d status=%d err=%v",
			accountID, traceID, workerURL, time.Since(start).Milliseconds(), statusCode, err)
		return resp, bodyBytes, turnRequest, err
	}

	resp, err := ProxyRequestWithBytes(state, "POST", "/responses", bodyBytes, turnRequest.Headers(), false)
	return resp, bodyBytes, turnRequest, err
}

func usesResponsesMaxCompletionTokens(modelID string) bool {
	return strings.HasPrefix(modelID, "gpt-5")
}

// ForwardResponsesResponse forwards the upstream response directly to client.
func ForwardResponsesResponse(c *gin.Context, accountID string, turnRequest copilotTurnRequest, requestBody []byte, resp *http.Response) {
	reqID := ""
	if v, ok := c.Get("respReqID"); ok {
		if s, _ := v.(string); s != "" {
			reqID = s
		}
	}
	defer func() {
		closeStart := time.Now()
		_ = resp.Body.Close()
		log.Printf("[responses rid=%s] post_close_ms=%d (resp.Body.Close)", reqID, time.Since(closeStart).Milliseconds())
	}()
	applyRoutingResponseHeaders(c, accountID)

	contentType := resp.Header.Get("Content-Type")
	isStream := strings.Contains(contentType, "text/event-stream")

	if isStream {
		forwardResponsesStream(c, accountID, turnRequest, requestBody, resp)
	} else {
		forwardResponsesNonStream(c, accountID, turnRequest, requestBody, resp)
	}
}

func forwardResponsesStream(c *gin.Context, accountID string, turnRequest copilotTurnRequest, requestBody []byte, resp *http.Response) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	// Close the TCP socket after the stream finishes. Some downstream agents
	// (notably the pi client) treat a turn as "done" only on TCP FIN, not on
	// the chunked-body terminator. Keeping the connection alive across the
	// terminal SSE event leaves them spinning in a working state even after
	// response.completed has been fully delivered.
	c.Header("Connection", "close")
	c.Request.Close = true
	c.Header("X-Accel-Buffering", "no")
	c.Status(resp.StatusCode)

	reqID := ""
	if v, ok := c.Get("respReqID"); ok {
		if s, _ := v.(string); s != "" {
			reqID = s
		}
	}

	reader := bufio.NewReaderSize(resp.Body, 10*1024*1024)
	var capture responsesStreamCapture
	var probe streamTailProbe
	startedAt := time.Now()
	writeErrOccurred := false
	var exitErr error
	// Track call_ids we've already stashed in the in-memory sticky cache so we
	// don't re-write them on every SSE line. Bind each new call_id to the
	// current account + turn context *as soon as we see the corresponding
	// function_call item* rather than waiting for stream terminal. If upstream
	// drops mid-stream, the ids we've already forwarded to the client stay in
	// the sticky cache — the next turn's canonical binding still resolves
	// instead of orphaning on an account we never stored.
	stashedFCIDs := map[string]bool{}
	stashNewFCIDs := func() {
		for _, id := range collectResponsesFunctionCallIDs(capture.ReplayItems) {
			if stashedFCIDs[id] {
				continue
			}
			stashedFCIDs[id] = true
			stashResponseFunctionCallTurnContextInMemory(accountID, []string{id}, turnRequest.Context)
		}
	}
	c.Stream(func(w io.Writer) bool {
		line, err := reader.ReadBytes(0x0A)
		if len(line) > 0 {
			capture.absorbLine(line)
			stashNewFCIDs()
			probe.observe(line)
			if _, writeErr := w.Write(line); writeErr != nil {
				writeErrOccurred = true
				exitErr = writeErr
				return false
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			// Some upstreams (Copilot /responses in particular) keep the TCP
			// socket open for keep-alive reuse after the terminal event. If we
			// blindly wait for EOF, downstream agents that rely on connection
			// close to mark a turn "done" stay stuck in a working state. Once
			// the terminal event has been fully delivered (event body + the
			// blank line that ends the SSE event), we stop reading so the
			// response finishes and the client socket closes.
			if probe.sawResponseDone && probe.endedWithBlankSep {
				return false
			}
		}
		if err != nil {
			if err != io.EOF {
				log.Printf("Responses stream read error: %v", err)
			}
			exitErr = err
			return false
		}
		return true
	})
	probe.log("responses", reqID, accountID, time.Since(startedAt), exitErr, writeErrOccurred)
	postStreamStart := time.Now()
	t0 := postStreamStart
	storeResponsesReplayFromStream(accountID, requestBody, capture)
	t1 := time.Now()
	storeResponseTurnContext(accountID, capture.ResponseID, turnRequest.Context)
	t2 := time.Now()
	storeResponseFunctionCallTurnContext(accountID, collectResponsesFunctionCallIDs(capture.ReplayItems), turnRequest.Context)
	t3 := time.Now()
	log.Printf("[responses rid=%s] post_stream store_replay_ms=%d store_turn_ms=%d store_fc_turn_ms=%d total_post_ms=%d",
		reqID,
		t1.Sub(t0).Milliseconds(),
		t2.Sub(t1).Milliseconds(),
		t3.Sub(t2).Milliseconds(),
		t3.Sub(postStreamStart).Milliseconds())
}

// streamTailProbe records basic stats and the last few SSE lines written to the client.
// Used to diagnose cases where the client reports that the stream "never ended" —
// e.g. upstream didn't emit a terminal event, or ended without the trailing blank line
// that SSE parsers use to commit the final event.
type streamTailProbe struct {
	tail              [][]byte
	totalLines        int
	totalBytes        int
	sawResponseDone   bool
	sawDataDONE       bool
	endedWithBlankSep bool
}

func (p *streamTailProbe) observe(line []byte) {
	p.totalLines++
	p.totalBytes += len(line)

	trimmed := bytes.TrimRight(line, "\r\n")
	if len(trimmed) == 0 {
		p.endedWithBlankSep = true
	} else {
		p.endedWithBlankSep = false
		if bytes.HasPrefix(trimmed, []byte("event:")) {
			ev := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("event:")))
			if bytes.Equal(ev, []byte("response.completed")) || bytes.Equal(ev, []byte("response.incomplete")) || bytes.Equal(ev, []byte("response.failed")) {
				p.sawResponseDone = true
			}
		} else if bytes.HasPrefix(trimmed, []byte("data:")) {
			payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
			if bytes.Equal(payload, []byte("[DONE]")) {
				p.sawDataDONE = true
			} else {
				// also catch event-name-in-data style ({"type":"response.completed", ...})
				if bytes.Contains(payload, []byte(`"type":"response.completed"`)) ||
					bytes.Contains(payload, []byte(`"type":"response.incomplete"`)) ||
					bytes.Contains(payload, []byte(`"type":"response.failed"`)) {
					p.sawResponseDone = true
				}
			}
		}
	}

	const maxTail = 12
	const maxLineLen = 200
	stored := line
	if len(stored) > maxLineLen {
		stored = append([]byte{}, line[:maxLineLen]...)
	} else {
		stored = append([]byte{}, stored...)
	}
	p.tail = append(p.tail, stored)
	if len(p.tail) > maxTail {
		p.tail = p.tail[len(p.tail)-maxTail:]
	}
}

func (p *streamTailProbe) log(kind, reqID, accountID string, elapsed time.Duration, exitErr error, writeErrOccurred bool) {
	exitReason := "eof"
	if writeErrOccurred {
		exitReason = "write_err"
	} else if exitErr != nil && exitErr != io.EOF {
		exitReason = "read_err"
	} else if exitErr == nil && p.endedWithBlankSep && (p.sawResponseDone || p.sawDataDONE) {
		// Early close: we proactively returned from the stream loop after the
		// terminal event's blank separator was flushed, instead of waiting for
		// upstream EOF. Downstream clients that gate on socket close now
		// unblock immediately.
		exitReason = "terminal_early_close"
	}

	// Render the last few lines as a single-line, quoted list with visible escapes.
	tailParts := make([]string, 0, len(p.tail))
	for _, line := range p.tail {
		esc := bytes.ReplaceAll(line, []byte("\\"), []byte("\\\\"))
		esc = bytes.ReplaceAll(esc, []byte("\n"), []byte("\\n"))
		esc = bytes.ReplaceAll(esc, []byte("\r"), []byte("\\r"))
		esc = bytes.ReplaceAll(esc, []byte("\""), []byte("\\\""))
		tailParts = append(tailParts, "\""+string(esc)+"\"")
	}
	tailStr := strings.Join(tailParts, " | ")

	log.Printf("[stream-tail kind=%s rid=%s account=%s] exit=%s err=%v elapsed=%s lines=%d bytes=%d saw_response_done=%v saw_data_DONE=%v ends_with_blank=%v tail=[%s]",
		kind, reqID, accountID, exitReason, exitErr, elapsed.Round(time.Millisecond),
		p.totalLines, p.totalBytes, p.sawResponseDone, p.sawDataDONE, p.endedWithBlankSep, tailStr)
}

func forwardResponsesNonStream(c *gin.Context, accountID string, turnRequest copilotTurnRequest, requestBody []byte, resp *http.Response) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read response"})
		return
	}

	storeResponsesReplayFromBody(accountID, requestBody, body)

	// Try to filter out empty reasoning items for better client compatibility.
	var respData map[string]interface{}
	if err := json.Unmarshal(body, &respData); err == nil {
		storeResponseTurnContext(accountID, extractResponseID(respData), turnRequest.Context)
		if output, ok := respData["output"].([]interface{}); ok {
			storeResponseFunctionCallTurnContext(accountID, collectResponsesFunctionCallIDs(output), turnRequest.Context)
			filtered := make([]interface{}, 0, len(output))
			for _, item := range output {
				if m, ok := item.(map[string]interface{}); ok {
					if itemType, _ := m["type"].(string); itemType == "reasoning" {
						if summary, _ := m["summary"].([]interface{}); len(summary) == 0 {
							continue
						}
					}
					filtered = append(filtered, item)
				}
			}
			respData["output"] = filtered
			if filteredBody, err := json.Marshal(respData); err == nil {
				body = filteredBody
			}
		}
	}

	c.Data(resp.StatusCode, "application/json", body)
}
