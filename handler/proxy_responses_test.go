package handler

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// Tests the pure header-parsing shim that gates opt-in orphan degrade. Must
// treat the header as case-insensitive (HTTP header values aren't canonical
// like names) and trim surrounding whitespace. Anything other than "orphan"
// (including empty, unrelated, or future-extension values) must return false
// so a client that misspells the opt-in still lands in strict mode.
func TestContinuationDegradeOptIn(t *testing.T) {
	cases := []struct {
		headerValue string
		want        bool
	}{
		{"orphan", true},
		{"ORPHAN", true},
		{"  orphan  ", true},
		{"Orphan", true},
		{"", false},
		{"split", false},
		{"orphan,split", false},
		{"1", false},
	}
	for _, tc := range cases {
		t.Run(strings.ReplaceAll(tc.headerValue, " ", "_"), func(t *testing.T) {
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			if tc.headerValue != "" {
				c.Request.Header.Set("X-Copilot-Continuation-Degrade", tc.headerValue)
			}
			if got := continuationDegradeOptIn(c); got != tc.want {
				t.Fatalf("continuationDegradeOptIn(%q) = %v, want %v", tc.headerValue, got, tc.want)
			}
		})
	}
}

func TestExtractResponseFunctionCallOutputIDsIncludesCustomToolOutputs(t *testing.T) {
	body := []byte(`{
		"input": [
			{"type":"function_call_output","call_id":"call_fn","output":"ok"},
			{"type":"custom_tool_call_output","call_id":"call_custom","output":"ok"},
			{"type":"message","role":"user","content":"ignore"}
		]
	}`)

	got := extractResponseFunctionCallOutputIDs(body)
	if len(got) != 2 || got[0] != "call_fn" || got[1] != "call_custom" {
		t.Fatalf("expected function and custom tool output ids, got %#v", got)
	}
}

func TestResponsesSessionAffinityKeyPrefersExplicitHeader(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Request.Header.Set("X-Copilot-Pool-Session", "session-header-123")

	key, source := responsesSessionAffinityKey(c, []byte(`{"metadata":{"session_id":"session-body-123"}}`))
	if key == "" {
		t.Fatalf("expected affinity key from header")
	}
	if source != "X-Copilot-Pool-Session" {
		t.Fatalf("source = %q, want X-Copilot-Pool-Session", source)
	}
	key2, _ := responsesSessionAffinityKey(c, []byte(`{"metadata":{"session_id":"session-body-123"}}`))
	if key2 != key {
		t.Fatalf("same session should hash to stable key")
	}
}

func TestResponsesSessionAffinityKeyReadsMetadata(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	key, source := responsesSessionAffinityKey(c, []byte(`{"metadata":{"conversation_id":"conv-123456789"}}`))
	if key == "" {
		t.Fatalf("expected affinity key from metadata")
	}
	if source != "metadata.conversation_id" {
		t.Fatalf("source = %q, want metadata.conversation_id", source)
	}
}

func TestResponsesSessionAffinityKeyIgnoresShortValues(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Request.Header.Set("X-Session-Id", "short")

	key, source := responsesSessionAffinityKey(c, []byte(`{"metadata":{"session_id":"tiny"}}`))
	if key != "" || source != "" {
		t.Fatalf("short affinity values must be ignored, got key=%q source=%q", key, source)
	}
}

func TestSetResponsesSessionAffinityContextPoolOnly(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Request.Header.Set("X-Copilot-Pool-Session", "session-header-123")

	setResponsesSessionAffinityContext(c, false, []byte(`{}`))
	if _, ok := c.Get("responsesSessionAffinityKey"); ok {
		t.Fatalf("direct account request must not install responses session affinity")
	}

	setResponsesSessionAffinityContext(c, true, []byte(`{}`))
	if key, ok := c.Get("responsesSessionAffinityKey"); !ok || key == "" {
		t.Fatalf("pool request should install responses session affinity key")
	}
	if source, _ := c.Get("responsesSessionAffinitySource"); source != "X-Copilot-Pool-Session" {
		t.Fatalf("source = %v, want X-Copilot-Pool-Session", source)
	}
}

func TestResponsesRoutingTelemetryCountersAndRecentLimit(t *testing.T) {
	resetResponsesRoutingTelemetryForTest()
	defer resetResponsesRoutingTelemetryForTest()

	recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
		Kind:          "account_switch_trigger",
		RequestID:     "rid-1",
		SessionKey:    "session-hash",
		SessionSource: "X-Copilot-Pool-Session",
		Model:         "gpt-5.4",
		FromAccount:   "acct-a",
		Reason:        "retryable_status_non_continuation",
		StatusCode:    http.StatusTooManyRequests,
	})
	recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
		Kind:        "account_switched",
		RequestID:   "rid-1",
		Model:       "gpt-5.4",
		FromAccount: "acct-a",
		ToAccount:   "acct-b",
		AccountID:   "acct-b",
	})

	snapshot := snapshotResponsesRoutingTelemetry(1)
	if snapshot.Counters["account_switch_trigger"] != 1 {
		t.Fatalf("account_switch_trigger counter = %d, want 1", snapshot.Counters["account_switch_trigger"])
	}
	if snapshot.Counters["account_switched"] != 1 {
		t.Fatalf("account_switched counter = %d, want 1", snapshot.Counters["account_switched"])
	}
	if snapshot.LastEventAt["account_switch_trigger"] == "" {
		t.Fatalf("expected last event timestamp for account_switch_trigger")
	}
	if len(snapshot.Recent) != 1 {
		t.Fatalf("recent length = %d, want 1", len(snapshot.Recent))
	}
	if got := snapshot.Recent[0].Kind; got != "account_switched" {
		t.Fatalf("recent[0].kind = %q, want account_switched", got)
	}
}

func TestChoosePinnedResponsesAccountOrder(t *testing.T) {
	oneShot := &resolvedAccount{AccountID: "acct-refresh"}
	continuation := &resolvedAccount{AccountID: "acct-continuation"}
	sameTurn := &resolvedAccount{AccountID: "acct-same-turn"}

	resolved, sticky, usedOneShot := choosePinnedResponsesAccount(oneShot, continuation, sameTurn, "")
	if resolved != oneShot || sticky != "pinned_retry" || !usedOneShot {
		t.Fatalf("one-shot pin = (%v, %q, %v), want acct-refresh pinned_retry true", resolved, sticky, usedOneShot)
	}

	resolved, sticky, usedOneShot = choosePinnedResponsesAccount(nil, continuation, sameTurn, "")
	if resolved != continuation || sticky != "session_binding_canonical" || usedOneShot {
		t.Fatalf("continuation pin = (%v, %q, %v), want acct-continuation session_binding_canonical false", resolved, sticky, usedOneShot)
	}

	resolved, sticky, usedOneShot = choosePinnedResponsesAccount(nil, nil, sameTurn, "")
	if resolved != sameTurn || sticky != "same_turn_pinned" || usedOneShot {
		t.Fatalf("same-turn pin = (%v, %q, %v), want acct-same-turn same_turn_pinned false", resolved, sticky, usedOneShot)
	}
}

// writeSessionBindingError shapes the three continuation-failure responses
// that codex / pi clients will observe when strict binding rejects a
// continuation. Verify status codes, the discriminating `type` field, and
// that the original binding diagnostics (accountID, split list, reason) make
// it into the JSON body so operators can grep logs and clients can branch.
func TestWriteSessionBindingErrorAccountUnavailable(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	writeSessionBindingError(c, "rid-test", continuationBindingResult{
		Kind:      continuationBindingAccountUnavailable,
		AccountID: "acct-paused",
		Reason:    "account acct-paused is paused for model gpt-5.4",
	})

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
	var body map[string]map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("expected JSON body, got: %s (err=%v)", w.Body.String(), err)
	}
	errObj := body["error"]
	if errObj["type"] != "session_bound_account_unavailable" {
		t.Fatalf("expected type session_bound_account_unavailable, got %v", errObj["type"])
	}
	if errObj["account_id"] != "acct-paused" {
		t.Fatalf("expected account_id acct-paused, got %v", errObj["account_id"])
	}
	if _, ok := errObj["message"].(string); !ok {
		t.Fatalf("expected message string, got %v", errObj["message"])
	}
}

func TestWriteSessionBindingErrorSplit(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	writeSessionBindingError(c, "rid-test", continuationBindingResult{
		Kind:          continuationBindingSplit,
		SplitAccounts: []string{"acct-a", "acct-b"},
		Reason:        "tool_call_output history spans 2 accounts: acct-a, acct-b",
	})

	if w.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d", w.Code)
	}
	var body map[string]map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("expected JSON body, got: %s", w.Body.String())
	}
	errObj := body["error"]
	if errObj["type"] != "session_split_history" {
		t.Fatalf("expected type session_split_history, got %v", errObj["type"])
	}
	accounts, _ := errObj["accounts"].([]interface{})
	if len(accounts) != 2 || accounts[0] != "acct-a" || accounts[1] != "acct-b" {
		t.Fatalf("expected split accounts [acct-a acct-b], got %v", accounts)
	}
}

func TestWriteSessionBindingErrorOrphan(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	writeSessionBindingError(c, "rid-test", continuationBindingResult{
		Kind:   continuationBindingOrphan,
		Reason: "no tool_call_output call_id matches any known session (hits=0 misses=5)",
	})

	if w.Code != http.StatusGone {
		t.Fatalf("expected 410, got %d", w.Code)
	}
	var body map[string]map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("expected JSON body, got: %s", w.Body.String())
	}
	errObj := body["error"]
	if errObj["type"] != "session_expired" {
		t.Fatalf("expected type session_expired, got %v", errObj["type"])
	}
	msg, _ := errObj["message"].(string)
	if !strings.Contains(msg, "no tool_call_output call_id matches") {
		t.Fatalf("expected reason text in message, got %q", msg)
	}
}

// resolveContinuationBinding with no previous_response_id and no fc ids must
// return Orphan directly — the caller (proxyResponses) only reaches this
// function when continuationRequested is true, but defensive callers should
// get a deterministic Orphan rather than a panic if they hit the empty-input
// edge case.
func TestResolveContinuationBindingEmptyInputsOrphan(t *testing.T) {
	result := resolveContinuationBinding("", nil, "gpt-5.4")
	if result.Kind != continuationBindingOrphan {
		t.Fatalf("expected Orphan on empty inputs, got kind=%v", result.Kind)
	}
	if result.AccountID != "" {
		t.Fatalf("expected empty AccountID on orphan, got %q", result.AccountID)
	}
	if result.Resolved != nil {
		t.Fatalf("expected Resolved nil on orphan, got %+v", result.Resolved)
	}
}

// A previous_response_id that isn't in either replay or turn cache must
// resolve as Orphan with a reason that includes the offending id so the
// client error body makes the problem actionable.
func TestResolveContinuationBindingPrevIdMissesCache(t *testing.T) {
	result := resolveContinuationBinding("resp_never_seen_xyz", nil, "gpt-5.4")
	if result.Kind != continuationBindingOrphan {
		t.Fatalf("expected Orphan on unknown prev_id, got kind=%v", result.Kind)
	}
	if !strings.Contains(result.Reason, "resp_never_seen_xyz") {
		t.Fatalf("expected prev_id echoed in reason, got %q", result.Reason)
	}
}

func TestOrphanTranslateRouteForModel(t *testing.T) {
	cases := []struct {
		model string
		want  orphanTranslateRoute
	}{
		{"gpt-5.4", orphanTranslateRouteMessages},
		{"claude-opus-4.7", orphanTranslateRouteMessages},
		{"gpt-4o-mini", orphanTranslateRouteChat},
		{"", orphanTranslateRouteNone},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := orphanTranslateRouteForModel(tc.model); got != tc.want {
				t.Fatalf("orphanTranslateRouteForModel(%q) = %q, want %q", tc.model, got, tc.want)
			}
		})
	}
}

func TestOrphanTranslateRouteModeNames(t *testing.T) {
	cases := []struct {
		route orphanTranslateRoute
		want  string
	}{
		{orphanTranslateRouteChat, "orphan_translate"},
		{orphanTranslateRouteMessages, "orphan_translate_messages"},
		{orphanTranslateRouteNone, "direct"},
	}
	for _, tc := range cases {
		if got := tc.route.modeName(); got != tc.want {
			t.Fatalf("modeName(%q) = %q, want %q", tc.route, got, tc.want)
		}
	}
}

func TestContinuationRecoveryStateArmed(t *testing.T) {
	if (continuationRecoveryState{}).armed() {
		t.Fatalf("zero recovery state must not be armed")
	}
	recovery := continuationRecoveryState{Route: orphanTranslateRouteMessages, Reason: "replay-invalid", FromAccount: "acct-a"}
	if !recovery.armed() {
		t.Fatalf("messages recovery route should be armed")
	}
}

func TestReadReplayInvalidResponseDetectsUnauthorizedAndRestoresBody(t *testing.T) {
	body := `{"error":{"message":"input item does not belong to this connection"}}`
	resp := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	replayInvalid, detail := readReplayInvalidResponse(resp)
	if !replayInvalid {
		t.Fatalf("expected 401 replay-invalid body to be detected")
	}
	if detail != body {
		t.Fatalf("detail = %q, want %q", detail, body)
	}
	restored, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll(restored body): %v", err)
	}
	if string(restored) != body {
		t.Fatalf("body was not restored, got %q", string(restored))
	}
}

func TestReadReplayInvalidResponseIgnoresPlainUnauthorized(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       io.NopCloser(strings.NewReader(`{"error":"token expired"}`)),
	}

	replayInvalid, _ := readReplayInvalidResponse(resp)
	if replayInvalid {
		t.Fatalf("plain 401 token failures must stay on the normal token refresh path")
	}
}
