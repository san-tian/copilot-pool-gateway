package orphan_translate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

// fakeReadCloser wraps a byte buffer as io.ReadCloser for translator input.
type fakeReadCloser struct{ *bytes.Reader }

func (f *fakeReadCloser) Close() error { return nil }

func newFakeSrc(chatSSE string) io.ReadCloser {
	return &fakeReadCloser{bytes.NewReader([]byte(chatSSE))}
}

// parseSSEEvents reads a Responses SSE stream and returns parsed data-payloads
// in order, keyed by event name.
type sseEvent struct {
	Name string
	Data map[string]interface{}
}

func readAllEvents(t *testing.T, r io.Reader) []sseEvent {
	t.Helper()
	var events []sseEvent
	rd := bufio.NewReader(r)
	var currentEvent string
	for {
		line, err := rd.ReadBytes('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(string(line), "\r\n")
			switch {
			case strings.HasPrefix(trimmed, "event:"):
				currentEvent = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
			case strings.HasPrefix(trimmed, "data:"):
				payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				var data map[string]interface{}
				if err := json.Unmarshal([]byte(payload), &data); err != nil {
					t.Fatalf("parse sse data %q: %v", payload, err)
				}
				events = append(events, sseEvent{Name: currentEvent, Data: data})
				currentEvent = ""
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("sse read: %v", err)
		}
	}
	return events
}

func findEvents(events []sseEvent, name string) []sseEvent {
	var out []sseEvent
	for _, e := range events {
		if e.Name == name {
			out = append(out, e)
		}
	}
	return out
}

func TestTranslator_TextOnly_EmitsCreatedDeltaCompleted(t *testing.T) {
	chat := `data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	if len(findEvents(events, "response.created")) != 1 {
		t.Fatalf("expected 1 response.created, got %d", len(findEvents(events, "response.created")))
	}
	deltas := findEvents(events, "response.output_text.delta")
	if len(deltas) != 2 {
		t.Fatalf("expected 2 output_text.delta, got %d", len(deltas))
	}
	if deltas[0].Data["delta"] != "Hello" || deltas[1].Data["delta"] != " world" {
		t.Fatalf("unexpected delta sequence: %+v", deltas)
	}
	completed := findEvents(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("expected 1 response.completed, got %d", len(completed))
	}
	resp := completed[0].Data["response"].(map[string]interface{})
	if status, _ := resp["status"].(string); status != "completed" {
		t.Fatalf("expected status=completed, got %v", resp["status"])
	}
	output, _ := resp["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(output))
	}
	msg := output[0].(map[string]interface{})
	if msg["type"] != "message" || msg["role"] != "assistant" {
		t.Fatalf("unexpected message item: %+v", msg)
	}
}

func TestTranslator_ToolCall_EmitsFunctionCallItem(t *testing.T) {
	chat := `data: {"id":"chatcmpl-2","choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"id":"chatcmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}

data: {"id":"chatcmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]}}]}

data: {"id":"chatcmpl-2","choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	argDeltas := findEvents(events, "response.function_call_arguments.delta")
	if len(argDeltas) != 2 {
		t.Fatalf("expected 2 arguments.delta events, got %d", len(argDeltas))
	}
	argDone := findEvents(events, "response.function_call_arguments.done")
	if len(argDone) != 1 {
		t.Fatalf("expected 1 arguments.done, got %d", len(argDone))
	}
	if argDone[0].Data["arguments"] != `{"location":"NYC"}` {
		t.Fatalf("unexpected final arguments: %v", argDone[0].Data["arguments"])
	}

	completed := findEvents(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("expected 1 response.completed, got %d", len(completed))
	}
	resp := completed[0].Data["response"].(map[string]interface{})
	output := resp["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(output))
	}
	fc := output[0].(map[string]interface{})
	if fc["type"] != "function_call" || fc["call_id"] != "call_abc" || fc["name"] != "get_weather" {
		t.Fatalf("unexpected fc item: %+v", fc)
	}
	if fc["arguments"] != `{"location":"NYC"}` {
		t.Fatalf("unexpected fc arguments: %v", fc["arguments"])
	}
}

func TestTranslator_MultipleToolCalls(t *testing.T) {
	chat := `data: {"id":"c","choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"foo","arguments":"{}"}}]}}]}

data: {"id":"c","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"bar","arguments":"{}"}}]}}]}

data: {"id":"c","choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	completed := findEvents(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("expected 1 response.completed")
	}
	resp := completed[0].Data["response"].(map[string]interface{})
	output := resp["output"].([]interface{})
	if len(output) != 2 {
		t.Fatalf("expected 2 fc items, got %d", len(output))
	}
	ids := []string{
		output[0].(map[string]interface{})["call_id"].(string),
		output[1].(map[string]interface{})["call_id"].(string),
	}
	if ids[0] != "call_a" || ids[1] != "call_b" {
		t.Fatalf("unexpected call_ids: %v", ids)
	}
}

func TestTranslator_SyntheticResponseID_ResponsePrefix(t *testing.T) {
	chat := `data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	created := findEvents(events, "response.created")
	if len(created) != 1 {
		t.Fatalf("expected 1 response.created")
	}
	resp := created[0].Data["response"].(map[string]interface{})
	id, _ := resp["id"].(string)
	if !strings.HasPrefix(id, "resp_") {
		t.Fatalf("expected synthetic id with resp_ prefix, got %q", id)
	}
	// Same id must appear on the terminal event.
	completed := findEvents(events, "response.completed")
	respDone := completed[0].Data["response"].(map[string]interface{})
	if respDone["id"] != id {
		t.Fatalf("response_id changed mid-stream: created=%v completed=%v", id, respDone["id"])
	}
}

func TestTranslator_LengthFinish_EmitsIncomplete(t *testing.T) {
	chat := `data: {"id":"c","choices":[{"delta":{"content":"..."},"finish_reason":"length"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	if len(findEvents(events, "response.incomplete")) != 1 {
		t.Fatalf("expected response.incomplete, events=%v", events)
	}
}

func TestTranslator_UpstreamError_EmitsFailed(t *testing.T) {
	// Only partial input; simulate read error by wrapping a reader that errors.
	er := &errReader{err: io.ErrUnexpectedEOF}
	r := NewResponsesReader(er, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	if len(findEvents(events, "response.failed")) != 1 {
		t.Fatalf("expected response.failed, got events=%v", events)
	}
}

type errReader struct{ err error }

func (e *errReader) Read(p []byte) (int, error) { return 0, e.err }
func (e *errReader) Close() error                { return nil }

func TestTranslator_ContentAfterToolCalls_OpensFreshMessage(t *testing.T) {
	chat := `data: {"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"foo","arguments":"{}"}}]}}]}

data: {"choices":[{"delta":{"content":"trailing text"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()

	events := readAllEvents(t, r)
	completed := findEvents(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("expected response.completed")
	}
	resp := completed[0].Data["response"].(map[string]interface{})
	output := resp["output"].([]interface{})
	if len(output) != 2 {
		t.Fatalf("expected 2 output items (fc + message), got %d: %+v", len(output), output)
	}
	if output[0].(map[string]interface{})["type"] != "function_call" {
		t.Fatalf("expected first item function_call, got %v", output[0])
	}
	if output[1].(map[string]interface{})["type"] != "message" {
		t.Fatalf("expected second item message, got %v", output[1])
	}
}

// Ensure the absorbLine parser in responses_replay can recover function_call
// items from our synthesized stream without modification. This is a smoke check
// of SSE compatibility — exact behavioral verification happens in integration.
func TestTranslator_OutputCompatibleWithAbsorbLine_SmokeCheck(t *testing.T) {
	chat := `data: {"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	src := newFakeSrc(chat)
	r := NewResponsesReader(src, "gpt-4o")
	defer r.Close()
	buf, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if !strings.Contains(string(buf), `"type":"function_call"`) {
		t.Fatalf("output does not contain function_call item: %s", string(buf))
	}
	if !strings.Contains(string(buf), `"call_id":"call_xyz"`) {
		t.Fatalf("output missing call_id: %s", string(buf))
	}
	if !strings.Contains(string(buf), `"type":"response.completed"`) {
		t.Fatalf("output missing response.completed: %s", string(buf))
	}
}
