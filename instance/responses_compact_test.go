package instance

import (
	"strings"
	"testing"
)

// TestBuildResponsesCompactTranscript_SplitsPrevSummary verifies that
// compaction / compaction_summary items are pulled into the previousSummary
// return value while normal turns stay in the conversation body.
func TestBuildResponsesCompactTranscript_SplitsPrevSummary(t *testing.T) {
	items := []interface{}{
		map[string]interface{}{"role": "user", "content": "first request"},
		map[string]interface{}{
			"type": "compaction_summary",
			"summary": []interface{}{
				map[string]interface{}{"type": "summary_text", "text": "earlier summary text"},
			},
		},
		map[string]interface{}{"role": "assistant", "content": "doing work"},
	}

	conv, prev := buildResponsesCompactTranscript(items)

	if !strings.Contains(conv, "first request") || !strings.Contains(conv, "doing work") {
		t.Fatalf("expected conversation to include user/assistant turns, got %q", conv)
	}
	if strings.Contains(conv, "earlier summary text") {
		t.Fatalf("previous summary leaked into conversation body: %q", conv)
	}
	if prev != "earlier summary text" {
		t.Fatalf("previousSummary = %q, want %q", prev, "earlier summary text")
	}
}

// TestBuildResponsesCompactTranscript_MultipleSummariesJoined joins multiple
// prior compaction items with a blank line.
func TestBuildResponsesCompactTranscript_MultipleSummariesJoined(t *testing.T) {
	items := []interface{}{
		map[string]interface{}{
			"type":    "compaction",
			"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "round one"}},
		},
		map[string]interface{}{
			"type":    "compaction_summary",
			"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "round two"}},
		},
	}

	conv, prev := buildResponsesCompactTranscript(items)
	if conv != "" {
		t.Fatalf("expected empty conversation when only summaries present, got %q", conv)
	}
	if prev != "round one\n\nround two" {
		t.Fatalf("prev = %q, want %q", prev, "round one\n\nround two")
	}
}

// TestBuildCompactUserContent_InitialPromptWhenNoPrevSummary verifies the
// initial prompt variant is chosen and no <previous-summary> tag appears when
// no prior summary is provided.
func TestBuildCompactUserContent_InitialPromptWhenNoPrevSummary(t *testing.T) {
	got := buildCompactUserContent("[USER]\nhello", "")

	if !strings.Contains(got, "<conversation>") || !strings.Contains(got, "</conversation>") {
		t.Fatalf("expected <conversation> wrapping, got %q", got)
	}
	if strings.Contains(got, "<previous-summary>") {
		t.Fatalf("unexpected <previous-summary> tag in initial-mode output: %q", got)
	}
	if !strings.Contains(got, compactInitialUserPrompt) {
		t.Fatal("expected compactInitialUserPrompt in body")
	}
	if strings.Contains(got, compactUpdateUserPrompt) {
		t.Fatal("update prompt should not be used when prev summary is empty")
	}
}

// TestBuildCompactUserContent_UpdatePromptWhenPrevSummaryPresent verifies the
// update prompt variant runs and both tags are present.
func TestBuildCompactUserContent_UpdatePromptWhenPrevSummaryPresent(t *testing.T) {
	got := buildCompactUserContent("[USER]\nfollow-up", "prior checkpoint")

	if !strings.Contains(got, "<conversation>\n[USER]\nfollow-up\n</conversation>") {
		t.Fatalf("conversation tag malformed: %q", got)
	}
	if !strings.Contains(got, "<previous-summary>\nprior checkpoint\n</previous-summary>") {
		t.Fatalf("previous-summary tag malformed: %q", got)
	}
	if !strings.Contains(got, compactUpdateUserPrompt) {
		t.Fatal("expected update prompt body when prev summary non-empty")
	}
	if strings.Contains(got, compactInitialUserPrompt) {
		t.Fatal("initial prompt should not be used when prev summary present")
	}
}

// TestBuildCompactUserContent_OnlyPrevSummary covers the edge case where the
// caller provides only a prior summary (no new conversation). The conversation
// tag should be omitted; previous-summary + update prompt still render.
func TestBuildCompactUserContent_OnlyPrevSummary(t *testing.T) {
	got := buildCompactUserContent("", "prior checkpoint")

	if strings.Contains(got, "<conversation>") {
		t.Fatalf("conversation tag should be omitted when conversation empty: %q", got)
	}
	if !strings.Contains(got, "<previous-summary>\nprior checkpoint\n</previous-summary>") {
		t.Fatalf("expected previous-summary tag, got %q", got)
	}
	if !strings.Contains(got, compactUpdateUserPrompt) {
		t.Fatal("expected update prompt")
	}
}

// TestDoResponsesCompactProxy_EmptyInputRejected verifies the validation path
// still rejects requests that carry no usable input whatsoever (neither
// conversation nor previous summary).
func TestDoResponsesCompactProxy_EmptyInputRejected(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":[]}`)

	resp, out, err := DoResponsesCompactProxy(nil, body)
	if resp != nil || out != nil {
		t.Fatalf("expected nil resp/body on rejection, got resp=%v body=%s", resp, string(out))
	}
	rerr, ok := err.(*ResponsesRewriteError)
	if !ok {
		t.Fatalf("expected *ResponsesRewriteError, got %T: %v", err, err)
	}
	if !strings.Contains(rerr.Message, "non-empty input") {
		t.Fatalf("unexpected error message: %q", rerr.Message)
	}
}

// TestDoResponsesCompactProxy_MissingModelRejected verifies the earlier
// validation path still fires when model is absent, so our input-partition
// code isn't masking the existing contract.
func TestDoResponsesCompactProxy_MissingModelRejected(t *testing.T) {
	body := []byte(`{"input":[{"role":"user","content":"hi"}]}`)

	resp, out, err := DoResponsesCompactProxy(nil, body)
	if resp != nil || out != nil {
		t.Fatalf("expected nil resp/body on rejection, got resp=%v body=%s", resp, string(out))
	}
	rerr, ok := err.(*ResponsesRewriteError)
	if !ok {
		t.Fatalf("expected *ResponsesRewriteError, got %T: %v", err, err)
	}
	if !strings.Contains(rerr.Message, "model") {
		t.Fatalf("unexpected error message: %q", rerr.Message)
	}
}

// TestTruncateToolResultForCompact_ShortPasses verifies small outputs are
// returned as-is (no elision, no marker).
func TestTruncateToolResultForCompact_ShortPasses(t *testing.T) {
	in := strings.Repeat("a", toolResultMaxChars)
	if got := truncateToolResultForCompact(in); got != in {
		t.Fatalf("short input was modified; len in=%d out=%d", len(in), len(got))
	}
}

// TestTruncateToolResultForCompact_LongTruncated verifies oversize outputs
// are reduced to head+tail samples with an elision marker embedding the
// count of dropped characters.
func TestTruncateToolResultForCompact_LongTruncated(t *testing.T) {
	head := strings.Repeat("H", 5000)
	tail := strings.Repeat("T", 5000)
	in := head + tail
	got := truncateToolResultForCompact(in)
	if len(got) >= len(in) {
		t.Fatalf("expected truncation; in=%d out=%d", len(in), len(got))
	}
	if !strings.Contains(got, "chars elided") {
		t.Fatalf("missing elision marker: %q", got)
	}
	if !strings.HasPrefix(got, "H") {
		t.Fatalf("head sample missing: %q", got[:30])
	}
	if !strings.HasSuffix(got, "T") {
		t.Fatalf("tail sample missing: %q", got[len(got)-30:])
	}
}

// TestBuildResponsesCompactTranscript_TruncatesLargeToolResult verifies the
// size cap is applied when walking function_call_output items, so a single
// huge tool result cannot blow up the compact request.
func TestBuildResponsesCompactTranscript_TruncatesLargeToolResult(t *testing.T) {
	bigOutput := strings.Repeat("x", 50000)
	items := []interface{}{
		map[string]interface{}{"role": "user", "content": "please read the big file"},
		map[string]interface{}{
			"type":    "function_call_output",
			"call_id": "call_abc",
			"output":  bigOutput,
		},
	}

	conv, _ := buildResponsesCompactTranscript(items)

	if strings.Count(conv, "x") >= 50000 {
		t.Fatalf("expected truncation of big output, got %d xs", strings.Count(conv, "x"))
	}
	if !strings.Contains(conv, "chars elided") {
		t.Fatalf("expected elision marker in conversation: %q", conv[:200])
	}
	if !strings.Contains(conv, "[TOOL RESULT call_abc]") {
		t.Fatalf("expected tool result header preserved: %q", conv[:200])
	}
}
