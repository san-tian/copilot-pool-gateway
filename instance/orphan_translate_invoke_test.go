package instance

import (
	"bufio"
	"io"
	"strings"
	"testing"

	"copilot-go/instance/orphan_translate"
)

// chatSSEReadCloser wraps a string as io.ReadCloser for translator input.
type chatSSEReadCloser struct{ *strings.Reader }

func (c *chatSSEReadCloser) Close() error { return nil }

func newChatSSE(s string) io.ReadCloser {
	return &chatSSEReadCloser{strings.NewReader(s)}
}

// TestOrphanTranslate_AbsorbLineCapturesFunctionCall verifies that the
// synthesized Responses-SSE stream produced by orphan_translate.NewResponsesReader
// is parseable by responsesStreamCapture.absorbLine — i.e. the sticky-cache
// tee continues to extract response_id and function_call items exactly as it
// does for native Copilot /v1/responses streams.
//
// This is the end-to-end contract check between the translator and the
// sticky-cache pipeline. Without this, a translator emit-format regression
// would silently break cross-relay continuation on the next turn.
func TestOrphanTranslate_AbsorbLineCapturesFunctionCall(t *testing.T) {
	chat := `data: {"id":"c1","choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"id":"c1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}

data: {"id":"c1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]}}]}

data: {"id":"c1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	r := orphan_translate.NewResponsesReader(newChatSSE(chat), "gpt-4o")
	defer r.Close()

	var capture responsesStreamCapture
	rd := bufio.NewReader(r)
	for {
		line, err := rd.ReadBytes('\n')
		if len(line) > 0 {
			capture.absorbLine(line)
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("read translator stream: %v", err)
		}
	}

	if !strings.HasPrefix(capture.ResponseID, "resp_") {
		t.Fatalf("expected synthetic ResponseID with resp_ prefix, got %q", capture.ResponseID)
	}
	if len(capture.ReplayItems) != 1 {
		t.Fatalf("expected 1 replay item, got %d: %+v", len(capture.ReplayItems), capture.ReplayItems)
	}
	item, ok := capture.ReplayItems[0].(map[string]interface{})
	if !ok {
		t.Fatalf("replay item is not a map: %+v", capture.ReplayItems[0])
	}
	if item["type"] != "function_call" {
		t.Fatalf("expected type=function_call, got %v", item["type"])
	}
	if item["call_id"] != "call_abc" {
		t.Fatalf("expected call_id=call_abc, got %v", item["call_id"])
	}
	if item["name"] != "get_weather" {
		t.Fatalf("expected name=get_weather, got %v", item["name"])
	}
	if item["arguments"] != `{"location":"NYC"}` {
		t.Fatalf("expected fully-assembled arguments, got %v", item["arguments"])
	}

	// collectResponsesFunctionCallIDs feeds stashResponseFunctionCallTurnContext.
	// If the translator's item shape diverges from the sticky-cache contract,
	// this extraction silently returns nothing and the next-turn cache lookup
	// misses — which would re-trigger the very orphan cycle we're fixing.
	ids := collectResponsesFunctionCallIDs(capture.ReplayItems)
	if len(ids) != 1 || ids[0] != "call_abc" {
		t.Fatalf("expected [call_abc], got %v", ids)
	}
}

// TestOrphanTranslate_AbsorbLineTextOnlyCapturesResponseID verifies that even
// a pure text stream (no tool calls, hence no replay items) still produces a
// stable ResponseID that storeResponseTurnContext can bind.
func TestOrphanTranslate_AbsorbLineTextOnlyCapturesResponseID(t *testing.T) {
	chat := `data: {"id":"c","choices":[{"delta":{"role":"assistant","content":"hi"}}]}

data: {"id":"c","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	r := orphan_translate.NewResponsesReader(newChatSSE(chat), "gpt-4o")
	defer r.Close()

	var capture responsesStreamCapture
	rd := bufio.NewReader(r)
	for {
		line, err := rd.ReadBytes('\n')
		if len(line) > 0 {
			capture.absorbLine(line)
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("read translator stream: %v", err)
		}
	}

	if !strings.HasPrefix(capture.ResponseID, "resp_") {
		t.Fatalf("expected synthetic ResponseID, got %q", capture.ResponseID)
	}
	// Text-only → no replayable items (the message item type is not in the
	// replayable set; only reasoning / function_call are).
	if len(capture.ReplayItems) != 0 {
		t.Fatalf("expected 0 replay items for text-only stream, got %d: %+v",
			len(capture.ReplayItems), capture.ReplayItems)
	}
}
