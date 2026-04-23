package handler

import (
	"encoding/json"
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
		Reason:        "function_call_output history spans 2 accounts: acct-a, acct-b",
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
		Reason: "no function_call_output call_id matches any known session (hits=0 misses=5)",
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
	if !strings.Contains(msg, "no function_call_output call_id matches") {
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
