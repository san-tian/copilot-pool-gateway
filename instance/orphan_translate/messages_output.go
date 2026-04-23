package orphan_translate

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"sort"
	"strings"

	"github.com/google/uuid"
)

// NewResponsesReaderFromMessages wraps an Anthropic /v1/messages SSE stream
// and returns an io.ReadCloser that emits OpenAI Responses API SSE events.
// Translation runs in a background goroutine; closing the returned reader
// also closes the underlying Messages stream.
//
// Anthropic streaming emits these event types (§Streaming messages):
//   message_start         — initial message envelope (id, model, usage)
//   content_block_start   — opens a content block at index N
//                           (block.type = text | tool_use | thinking)
//   content_block_delta   — incremental update for the open block
//                           (delta.type = text_delta | input_json_delta |
//                                         thinking_delta | signature_delta)
//   content_block_stop    — closes the block at index N
//   message_delta         — stop_reason/usage patch on the envelope
//   message_stop          — terminal event
//   ping                  — keepalive (ignored)
//   error                 — upstream failure envelope
//
// Responses SSE expected by absorbLine / downstream:
//   response.created
//   response.output_item.added           (message OR function_call)
//   response.content_part.added          (message only)
//   response.output_text.delta/done      (message only)
//   response.content_part.done           (message only)
//   response.function_call_arguments.delta/done (function_call only)
//   response.output_item.done
//   response.completed / response.incomplete / response.failed
//
// thinking_delta / signature_delta are dropped — they carry no cross-relay
// signal we can replay, and the Responses reasoning shape requires
// encrypted_content which we can't re-mint.
func NewResponsesReaderFromMessages(src io.ReadCloser, model string) io.ReadCloser {
	pr, pw := io.Pipe()
	t := &msgTranslator{
		src:        src,
		dst:        pw,
		fallback:   model,
		responseID: "resp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		blocks:     map[int]*msgBlockState{},
	}
	go t.run()
	return &pipedReader{PipeReader: pr, underlying: src}
}

type msgTranslator struct {
	src      io.ReadCloser
	dst      *io.PipeWriter
	fallback string // model name from request, used if message_start omits it

	responseID string
	model      string

	seq         int
	createdSent bool

	// Per-Anthropic-content-block state. Keyed by the `index` field on
	// content_block_start events — Anthropic streams are sequential but the
	// index is authoritative for routing deltas.
	blocks    map[int]*msgBlockState
	blockOrder []int // emit order

	// Accumulated final items, ready for response.completed snapshot.
	output []finalOutputEntry

	stopReason string
}

type msgBlockState struct {
	blockIdx    int
	outputIndex int
	kind        string // "text" | "tool_use"
	itemID      string

	// For text blocks.
	textContentIdx int
	textBuf        strings.Builder

	// For tool_use blocks.
	callID  string
	name    string
	argsBuf strings.Builder

	addedSent bool
	closed    bool
}

func (t *msgTranslator) run() {
	defer func() {
		_ = t.dst.Close()
	}()

	reader := bufio.NewReaderSize(t.src, 128*1024)
	var scannerErr error

	var currentEvent string
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			trimmed := bytes.TrimRight(line, "\r\n")
			switch {
			case len(bytes.TrimSpace(trimmed)) == 0:
				// blank line — event terminator, already handled when data: arrived
			case bytes.HasPrefix(trimmed, []byte("event:")):
				currentEvent = string(bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("event:"))))
			case bytes.HasPrefix(trimmed, []byte("data:")):
				payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
				if len(payload) == 0 {
					continue
				}
				t.onEvent(currentEvent, payload)
				currentEvent = ""
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
	t.finalize()
}

func (t *msgTranslator) onEvent(name string, payload []byte) {
	// Many Anthropic streams also embed `"type"` inside the data payload —
	// prefer the explicit event header but fall back to the JSON type.
	if name == "" {
		var peek struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(payload, &peek); err == nil {
			name = peek.Type
		}
	}

	switch name {
	case "message_start":
		t.handleMessageStart(payload)
	case "content_block_start":
		t.handleContentBlockStart(payload)
	case "content_block_delta":
		t.handleContentBlockDelta(payload)
	case "content_block_stop":
		t.handleContentBlockStop(payload)
	case "message_delta":
		t.handleMessageDelta(payload)
	case "message_stop":
		// emitted on finalize — nothing to do here
	case "error":
		t.handleError(payload)
	case "ping":
		// keepalive
	}
}

func (t *msgTranslator) handleMessageStart(payload []byte) {
	if t.createdSent {
		return
	}
	var raw struct {
		Message struct {
			ID    string `json:"id"`
			Model string `json:"model"`
		} `json:"message"`
	}
	_ = json.Unmarshal(payload, &raw)
	if raw.Message.Model != "" {
		t.model = raw.Message.Model
	} else {
		t.model = t.fallback
	}
	t.emitCreated()
	t.createdSent = true
}

func (t *msgTranslator) handleContentBlockStart(payload []byte) {
	if !t.createdSent {
		// Some streams may elide message_start on reconnect — synthesize.
		if t.model == "" {
			t.model = t.fallback
		}
		t.emitCreated()
		t.createdSent = true
	}
	var raw struct {
		Index        int `json:"index"`
		ContentBlock struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Text  string          `json:"text"`
			Input json.RawMessage `json:"input"`
		} `json:"content_block"`
	}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return
	}

	bs := &msgBlockState{
		blockIdx:    raw.Index,
		outputIndex: t.nextOutputIndex(),
	}
	switch raw.ContentBlock.Type {
	case "text":
		bs.kind = "text"
		bs.itemID = "msg_" + shortID()
		bs.textBuf.WriteString(raw.ContentBlock.Text)
		t.blocks[raw.Index] = bs
		t.blockOrder = append(t.blockOrder, raw.Index)

		t.emitEvent("response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": bs.outputIndex,
			"item": map[string]interface{}{
				"type":    "message",
				"id":      bs.itemID,
				"status":  "in_progress",
				"role":    "assistant",
				"content": []interface{}{},
			},
		})
		t.emitEvent("response.content_part.added", map[string]interface{}{
			"type":          "response.content_part.added",
			"item_id":       bs.itemID,
			"output_index":  bs.outputIndex,
			"content_index": bs.textContentIdx,
			"part":          map[string]interface{}{"type": "output_text", "text": ""},
		})
		if raw.ContentBlock.Text != "" {
			t.emitEvent("response.output_text.delta", map[string]interface{}{
				"type":          "response.output_text.delta",
				"item_id":       bs.itemID,
				"output_index":  bs.outputIndex,
				"content_index": bs.textContentIdx,
				"delta":         raw.ContentBlock.Text,
			})
		}
		bs.addedSent = true

	case "tool_use":
		bs.kind = "tool_use"
		bs.itemID = "fc_syn_" + shortID()
		// Mint a synthetic call_id rather than forwarding Anthropic's toolu_XXX.
		// The client stores this call_id in its four sticky caches (keyed on
		// function_call ids) and replays it on the next turn via
		// rewritePreviousResponseContinuation, which expands it into upstream
		// /v1/responses input[]. Copilot /v1/responses recognizes toolu_-prefixed
		// ids as "belongs to a different (Messages API) session" and rejects the
		// continuation with "input item does not belong" — probe verified 2026-04-23
		// that a never-minted call_syn_XXX id is accepted. Forwarding the raw
		// Anthropic id is the sticky-cache poisoning vector that caused pi 502s.
		bs.callID = "call_syn_" + shortID()
		bs.name = raw.ContentBlock.Name
		if len(raw.ContentBlock.Input) > 0 && !bytes.Equal(bytes.TrimSpace(raw.ContentBlock.Input), []byte("{}")) {
			// Seed argsBuf with the initial input so the final `.done` has the
			// full payload even if no input_json_delta follows.
			bs.argsBuf.Write(raw.ContentBlock.Input)
		}
		t.blocks[raw.Index] = bs
		t.blockOrder = append(t.blockOrder, raw.Index)

		t.emitEvent("response.output_item.added", map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": bs.outputIndex,
			"item": map[string]interface{}{
				"type":      "function_call",
				"id":        bs.itemID,
				"status":    "in_progress",
				"call_id":   bs.callID,
				"name":      bs.name,
				"arguments": "",
			},
		})
		bs.addedSent = true

	case "thinking":
		// Drop — no replay-safe representation.
		bs.kind = "thinking"
		t.blocks[raw.Index] = bs
	}
}

func (t *msgTranslator) handleContentBlockDelta(payload []byte) {
	var raw struct {
		Index int `json:"index"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			PartialJSON string `json:"partial_json"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return
	}
	bs := t.blocks[raw.Index]
	if bs == nil || bs.closed {
		return
	}
	switch raw.Delta.Type {
	case "text_delta":
		if bs.kind != "text" || raw.Delta.Text == "" {
			return
		}
		bs.textBuf.WriteString(raw.Delta.Text)
		t.emitEvent("response.output_text.delta", map[string]interface{}{
			"type":          "response.output_text.delta",
			"item_id":       bs.itemID,
			"output_index":  bs.outputIndex,
			"content_index": bs.textContentIdx,
			"delta":         raw.Delta.Text,
		})
	case "input_json_delta":
		if bs.kind != "tool_use" || raw.Delta.PartialJSON == "" {
			return
		}
		bs.argsBuf.WriteString(raw.Delta.PartialJSON)
		t.emitEvent("response.function_call_arguments.delta", map[string]interface{}{
			"type":         "response.function_call_arguments.delta",
			"item_id":      bs.itemID,
			"output_index": bs.outputIndex,
			"delta":        raw.Delta.PartialJSON,
		})
	case "thinking_delta", "signature_delta":
		// dropped
	}
}

func (t *msgTranslator) handleContentBlockStop(payload []byte) {
	var raw struct {
		Index int `json:"index"`
	}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return
	}
	bs := t.blocks[raw.Index]
	if bs == nil || bs.closed {
		return
	}
	t.closeBlock(bs)
}

func (t *msgTranslator) handleMessageDelta(payload []byte) {
	var raw struct {
		Delta struct {
			StopReason string `json:"stop_reason"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(payload, &raw); err == nil {
		if raw.Delta.StopReason != "" {
			t.stopReason = raw.Delta.StopReason
		}
	}
}

func (t *msgTranslator) handleError(payload []byte) {
	var raw struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &raw)
	msg := raw.Error.Message
	if msg == "" {
		msg = string(payload)
	}
	t.emitFailed(msg)
}

func (t *msgTranslator) closeBlock(bs *msgBlockState) {
	switch bs.kind {
	case "text":
		text := bs.textBuf.String()
		t.emitEvent("response.output_text.done", map[string]interface{}{
			"type":          "response.output_text.done",
			"item_id":       bs.itemID,
			"output_index":  bs.outputIndex,
			"content_index": bs.textContentIdx,
			"text":          text,
		})
		t.emitEvent("response.content_part.done", map[string]interface{}{
			"type":          "response.content_part.done",
			"item_id":       bs.itemID,
			"output_index":  bs.outputIndex,
			"content_index": bs.textContentIdx,
			"part":          map[string]interface{}{"type": "output_text", "text": text},
		})
		msgItem := map[string]interface{}{
			"type":   "message",
			"id":     bs.itemID,
			"status": "completed",
			"role":   "assistant",
			"content": []interface{}{
				map[string]interface{}{"type": "output_text", "text": text, "annotations": []interface{}{}},
			},
		}
		t.emitEvent("response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": bs.outputIndex,
			"item":         msgItem,
		})
		t.output = append(t.output, finalOutputEntry{outputIndex: bs.outputIndex, item: msgItem})

	case "tool_use":
		args := bs.argsBuf.String()
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
		t.emitEvent("response.function_call_arguments.done", map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      bs.itemID,
			"output_index": bs.outputIndex,
			"arguments":    args,
		})
		fcItem := map[string]interface{}{
			"type":      "function_call",
			"id":        bs.itemID,
			"status":    "completed",
			"call_id":   bs.callID,
			"name":      bs.name,
			"arguments": args,
		}
		t.emitEvent("response.output_item.done", map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": bs.outputIndex,
			"item":         fcItem,
		})
		t.output = append(t.output, finalOutputEntry{outputIndex: bs.outputIndex, item: fcItem})
	}
	bs.closed = true
}

func (t *msgTranslator) finalize() {
	if !t.createdSent {
		if t.model == "" {
			t.model = t.fallback
		}
		t.emitCreated()
		t.createdSent = true
	}
	// Close any straggler blocks whose content_block_stop never arrived.
	for _, idx := range t.blockOrder {
		bs := t.blocks[idx]
		if bs != nil && !bs.closed && bs.kind != "thinking" {
			t.closeBlock(bs)
		}
	}

	status := "completed"
	if t.stopReason == "max_tokens" {
		status = "incomplete"
	}
	t.emitEvent("response."+status, map[string]interface{}{
		"type":     "response." + status,
		"response": t.responseSnapshot(status),
	})
}

func (t *msgTranslator) emitFailed(detail string) {
	if !t.createdSent {
		if t.model == "" {
			t.model = t.fallback
		}
		t.emitCreated()
		t.createdSent = true
	}
	for _, idx := range t.blockOrder {
		bs := t.blocks[idx]
		if bs != nil && !bs.closed && bs.kind != "thinking" {
			t.closeBlock(bs)
		}
	}
	snap := t.responseSnapshot("failed")
	snap["error"] = map[string]interface{}{"type": "server_error", "message": detail}
	t.emitEvent("response.failed", map[string]interface{}{
		"type":     "response.failed",
		"response": snap,
	})
}

func (t *msgTranslator) responseSnapshot(status string) map[string]interface{} {
	return map[string]interface{}{
		"id":     t.responseID,
		"object": "response",
		"status": status,
		"model":  t.model,
		"output": t.sortedOutput(),
	}
}

func (t *msgTranslator) sortedOutput() []interface{} {
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

func (t *msgTranslator) emitCreated() {
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

func (t *msgTranslator) emitEvent(name string, payload map[string]interface{}) {
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
	_, _ = t.dst.Write(buf.Bytes())
	t.seq++
}

func (t *msgTranslator) nextOutputIndex() int {
	return len(t.output) + liveMsgItemCount(t)
}

func liveMsgItemCount(t *msgTranslator) int {
	n := 0
	for _, idx := range t.blockOrder {
		bs := t.blocks[idx]
		if bs != nil && !bs.closed && bs.kind != "thinking" {
			n++
		}
	}
	return n
}

