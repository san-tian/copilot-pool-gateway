package orphan_translate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/google/uuid"
)

// NewResponsesReader wraps a chat/completions SSE stream and returns an
// io.ReadCloser that emits OpenAI Responses API SSE events. Translation runs
// in a background goroutine; closing the returned reader also closes the
// underlying chat stream.
//
// On graceful end of chat stream (either [DONE] or EOF after a finish_reason),
// a terminal `response.completed` event is emitted. If the underlying stream
// errors, a `response.failed` event is emitted so downstream capture can still
// commit what we've seen.
func NewResponsesReader(src io.ReadCloser, model string) io.ReadCloser {
	pr, pw := io.Pipe()
	t := &translator{
		src:        src,
		dst:        pw,
		model:      model,
		responseID: "resp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		toolCalls:  map[int]*toolCallState{},
	}
	go t.run()
	return &pipedReader{PipeReader: pr, underlying: src}
}

type pipedReader struct {
	*io.PipeReader
	underlying io.Closer
	once       sync.Once
}

func (p *pipedReader) Close() error {
	var err error
	p.once.Do(func() {
		_ = p.PipeReader.Close()
		err = p.underlying.Close()
	})
	return err
}

type translator struct {
	src        io.ReadCloser
	dst        *io.PipeWriter
	model      string
	responseID string

	seq         int
	createdSent bool

	// current text-message item state (mutually exclusive with active tool_call stream).
	textItem *textItemState

	// tool_call deltas keyed by their streaming index.
	toolCalls map[int]*toolCallState
	toolOrder []int // index values in emit order

	// output items accumulated for the final response.completed event.
	// Each entry remembers its outputIndex so the final array can be sorted
	// regardless of the close order.
	output []finalOutputEntry

	finishReason string
}

type finalOutputEntry struct {
	outputIndex int
	item        map[string]interface{}
}

type textItemState struct {
	itemID      string
	outputIndex int
	contentIdx  int
	textBuf     strings.Builder
	closed      bool
}

type toolCallState struct {
	deltaIdx    int
	outputIndex int
	itemID      string
	callID      string
	name        string
	argsBuf     strings.Builder
	addedSent   bool
	closed      bool
}

// chatChunk is a single parsed chat/completions SSE data payload.
type chatChunk struct {
	ID             string
	ContentDelta   string
	ToolCallDeltas []chatToolCallDelta
	FinishReason   string
	Usage          map[string]interface{}
}

type chatToolCallDelta struct {
	Index     int
	ID        string
	Name      string
	Arguments string
}

func (t *translator) run() {
	defer func() {
		_ = t.dst.Close()
	}()

	reader := bufio.NewReaderSize(t.src, 128*1024)
	var scannerErr error
	done := false

ReadLoop:
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if stop := t.onLine(line); stop {
				done = true
				break ReadLoop
			}
		}
		if err != nil {
			if err != io.EOF {
				scannerErr = err
			}
			break
		}
	}

	if scannerErr != nil {
		t.emitFailed(scannerErr.Error())
		return
	}
	_ = done
	t.finalize()
}

// onLine returns true when [DONE] is seen (stream terminated cleanly).
func (t *translator) onLine(raw []byte) bool {
	trimmed := bytes.TrimSpace(raw)
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return false
	}
	payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
	if len(payload) == 0 {
		return false
	}
	if bytes.Equal(payload, []byte("[DONE]")) {
		return true
	}
	chunk, ok := parseChatChunk(payload)
	if !ok {
		return false
	}
	t.applyChunk(chunk)
	return false
}

func parseChatChunk(payload []byte) (chatChunk, bool) {
	var raw struct {
		ID      string `json:"id"`
		Choices []struct {
			Delta struct {
				Content   interface{} `json:"content"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage map[string]interface{} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return chatChunk{}, false
	}

	out := chatChunk{ID: raw.ID, Usage: raw.Usage}
	if len(raw.Choices) > 0 {
		ch := raw.Choices[0]
		if s, ok := ch.Delta.Content.(string); ok {
			out.ContentDelta = s
		}
		for _, tc := range ch.Delta.ToolCalls {
			out.ToolCallDeltas = append(out.ToolCallDeltas, chatToolCallDelta{
				Index:     tc.Index,
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
		if ch.FinishReason != nil {
			out.FinishReason = *ch.FinishReason
		}
	}
	return out, true
}

func (t *translator) applyChunk(chunk chatChunk) {
	if !t.createdSent {
		t.emitCreated()
		t.createdSent = true
	}

	if chunk.ContentDelta != "" {
		t.handleContentDelta(chunk.ContentDelta)
	}
	for _, tcd := range chunk.ToolCallDeltas {
		t.handleToolCallDelta(tcd)
	}
	if chunk.FinishReason != "" {
		t.finishReason = chunk.FinishReason
	}
}

func (t *translator) handleContentDelta(delta string) {
	// If a tool_call was in progress, close the current text item? No — chat
	// doesn't interleave content between tool_call chunks; but if content
	// arrives after tool_calls have been streamed, we still open a fresh
	// message item after them. Keeping text and tool-calls in separate items
	// matches Responses semantics.
	if t.textItem == nil || t.textItem.closed {
		item := &textItemState{
			itemID:      "msg_" + shortID(),
			outputIndex: t.nextOutputIndex(),
			contentIdx:  0,
		}
		t.textItem = item
		t.emitEvent("response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": item.outputIndex,
			"item": map[string]interface{}{
				"type":    "message",
				"id":      item.itemID,
				"status":  "in_progress",
				"role":    "assistant",
				"content": []interface{}{},
			},
		})
		t.emitEvent("response.content_part.added", map[string]interface{}{
			"type":          "response.content_part.added",
			"item_id":       item.itemID,
			"output_index":  item.outputIndex,
			"content_index": item.contentIdx,
			"part":          map[string]interface{}{"type": "output_text", "text": ""},
		})
	}
	t.textItem.textBuf.WriteString(delta)
	t.emitEvent("response.output_text.delta", map[string]interface{}{
		"type":          "response.output_text.delta",
		"item_id":       t.textItem.itemID,
		"output_index":  t.textItem.outputIndex,
		"content_index": t.textItem.contentIdx,
		"delta":         delta,
	})
}

func (t *translator) handleToolCallDelta(d chatToolCallDelta) {
	// Close any open text item first — tool_calls produce their own output
	// items and shouldn't nest inside a message.
	t.closeTextItem()

	tc, exists := t.toolCalls[d.Index]
	if !exists {
		tc = &toolCallState{
			deltaIdx:    d.Index,
			outputIndex: t.nextOutputIndex(),
			itemID:      "fc_" + shortID(),
		}
		t.toolCalls[d.Index] = tc
		t.toolOrder = append(t.toolOrder, d.Index)
	}
	if d.ID != "" && tc.callID == "" {
		tc.callID = d.ID
	}
	if d.Name != "" && tc.name == "" {
		tc.name = d.Name
	}

	if !tc.addedSent && (tc.callID != "" || tc.name != "") {
		t.emitEvent("response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": tc.outputIndex,
			"item": map[string]interface{}{
				"type":      "function_call",
				"id":        tc.itemID,
				"status":    "in_progress",
				"call_id":   tc.callID,
				"name":      tc.name,
				"arguments": "",
			},
		})
		tc.addedSent = true
	}

	if d.Arguments != "" {
		tc.argsBuf.WriteString(d.Arguments)
		if tc.addedSent {
			t.emitEvent("response.function_call_arguments.delta", map[string]interface{}{
				"type":         "response.function_call_arguments.delta",
				"item_id":      tc.itemID,
				"output_index": tc.outputIndex,
				"delta":        d.Arguments,
			})
		}
	}
}

func (t *translator) closeTextItem() {
	if t.textItem == nil || t.textItem.closed {
		return
	}
	item := t.textItem
	text := item.textBuf.String()
	t.emitEvent("response.output_text.done", map[string]interface{}{
		"type":          "response.output_text.done",
		"item_id":       item.itemID,
		"output_index":  item.outputIndex,
		"content_index": item.contentIdx,
		"text":          text,
	})
	t.emitEvent("response.content_part.done", map[string]interface{}{
		"type":          "response.content_part.done",
		"item_id":       item.itemID,
		"output_index":  item.outputIndex,
		"content_index": item.contentIdx,
		"part":          map[string]interface{}{"type": "output_text", "text": text},
	})
	msgItem := map[string]interface{}{
		"type":   "message",
		"id":     item.itemID,
		"status": "completed",
		"role":   "assistant",
		"content": []interface{}{
			map[string]interface{}{"type": "output_text", "text": text, "annotations": []interface{}{}},
		},
	}
	t.emitEvent("response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": item.outputIndex,
		"item":         msgItem,
	})
	t.output = append(t.output, finalOutputEntry{outputIndex: item.outputIndex, item: msgItem})
	item.closed = true
}

func (t *translator) closeToolCalls() {
	for _, idx := range t.toolOrder {
		tc := t.toolCalls[idx]
		if tc == nil || tc.closed {
			continue
		}
		if !tc.addedSent {
			t.emitEvent("response.output_item.added", map[string]interface{}{
				"type":         "response.output_item.added",
				"output_index": tc.outputIndex,
				"item": map[string]interface{}{
					"type":      "function_call",
					"id":        tc.itemID,
					"status":    "in_progress",
					"call_id":   tc.callID,
					"name":      tc.name,
					"arguments": "",
				},
			})
			tc.addedSent = true
		}
		args := tc.argsBuf.String()
		t.emitEvent("response.function_call_arguments.done", map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      tc.itemID,
			"output_index": tc.outputIndex,
			"arguments":    args,
		})
		fcItem := map[string]interface{}{
			"type":      "function_call",
			"id":        tc.itemID,
			"status":    "completed",
			"call_id":   tc.callID,
			"name":      tc.name,
			"arguments": args,
		}
		t.emitEvent("response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": tc.outputIndex,
			"item":         fcItem,
		})
		t.output = append(t.output, finalOutputEntry{outputIndex: tc.outputIndex, item: fcItem})
		tc.closed = true
	}
}

func (t *translator) finalize() {
	if !t.createdSent {
		// No chunks arrived at all — emit a minimal created so downstream
		// capture still sees a response_id.
		t.emitCreated()
		t.createdSent = true
	}
	t.closeTextItem()
	t.closeToolCalls()

	status := "completed"
	if t.finishReason == "length" {
		status = "incomplete"
	}
	t.emitEvent("response."+status, map[string]interface{}{
		"type":     "response." + status,
		"response": t.responseSnapshot(status),
	})
}

func (t *translator) emitFailed(detail string) {
	if !t.createdSent {
		t.emitCreated()
		t.createdSent = true
	}
	t.closeTextItem()
	t.closeToolCalls()
	snap := t.responseSnapshot("failed")
	snap["error"] = map[string]interface{}{"type": "server_error", "message": detail}
	t.emitEvent("response.failed", map[string]interface{}{
		"type":     "response.failed",
		"response": snap,
	})
}

func (t *translator) responseSnapshot(status string) map[string]interface{} {
	return map[string]interface{}{
		"id":     t.responseID,
		"object": "response",
		"status": status,
		"model":  t.model,
		"output": t.sortedOutput(),
	}
}

func (t *translator) sortedOutput() []interface{} {
	entries := append([]finalOutputEntry(nil), t.output...)
	sort.SliceStable(entries, func(i, j int) bool {
		return entries[i].outputIndex < entries[j].outputIndex
	})
	out := make([]interface{}, len(entries))
	for i, e := range entries {
		out[i] = e.item
	}
	return out
}

func (t *translator) emitCreated() {
	t.emitEvent("response.created", map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":     t.responseID,
			"object": "response",
			"status": "in_progress",
			"model":  t.model,
			"output": []interface{}{},
		},
	})
}

func (t *translator) emitEvent(name string, payload map[string]interface{}) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	var buf bytes.Buffer
	buf.WriteString("event: ")
	buf.WriteString(name)
	buf.WriteByte('\n')
	buf.WriteString("data: ")
	buf.Write(data)
	buf.WriteString("\n\n")
	if _, err := t.dst.Write(buf.Bytes()); err != nil {
		// downstream closed; further writes will fail silently
	}
	t.seq++
}

func (t *translator) nextOutputIndex() int {
	return len(t.output) + liveItemCount(t)
}

// liveItemCount counts items that have been added to the stream but not yet
// appended to t.output (i.e. currently open items).
func liveItemCount(t *translator) int {
	n := 0
	if t.textItem != nil && !t.textItem.closed {
		n++
	}
	for _, idx := range t.toolOrder {
		tc := t.toolCalls[idx]
		if tc != nil && !tc.closed {
			n++
		}
	}
	return n
}

func shortID() string {
	s := strings.ReplaceAll(uuid.NewString(), "-", "")
	if len(s) > 24 {
		return s[:24]
	}
	return s
}
