package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestMessagesSessionAffinityKeyPrefersExplicitHeader(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	c.Request.Header.Set("X-Copilot-Pool-Session", "message-session-123")

	key, source := messagesSessionAffinityKey(c, []byte(`{"metadata":{"user_id":"{\"device_id\":\"device-123\",\"session_id\":\"session-123456\"}"}}`))
	if key == "" {
		t.Fatalf("expected affinity key from header")
	}
	if source != "X-Copilot-Pool-Session" {
		t.Fatalf("source = %q, want X-Copilot-Pool-Session", source)
	}
}

func TestMessagesSessionAffinityKeyReadsMetadataUserIDJSON(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	body := []byte(`{"metadata":{"user_id":"{\"device_id\":\"device-123\",\"session_id\":\"session-123456\"}"}}`)
	key, source := messagesSessionAffinityKey(c, body)
	if key == "" {
		t.Fatalf("expected affinity key from metadata.user_id")
	}
	if source != "metadata.user_id.session_id" {
		t.Fatalf("source = %q, want metadata.user_id.session_id", source)
	}

	key2, _ := messagesSessionAffinityKey(c, body)
	if key2 != key {
		t.Fatalf("same metadata.user_id session should hash to stable key")
	}
}

func TestMessagesSessionAffinityKeyReadsLegacyMetadataUserID(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	key, source := messagesSessionAffinityKey(c, []byte(`{"metadata":{"user_id":"user_device123_account_demo_session_session-legacy-123"}}`))
	if key == "" {
		t.Fatalf("expected affinity key from legacy metadata.user_id")
	}
	if source != "metadata.user_id.session_id" {
		t.Fatalf("source = %q, want metadata.user_id.session_id", source)
	}
}

func TestMessagesSessionAffinityKeyIgnoresShortMetadataSession(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	key, source := messagesSessionAffinityKey(c, []byte(`{"metadata":{"user_id":"{\"device_id\":\"device-123\",\"session_id\":\"tiny\"}"}}`))
	if key != "" || source != "" {
		t.Fatalf("short metadata session must be ignored, got key=%q source=%q", key, source)
	}
}

func TestSetMessagesSessionAffinityContextPoolOnly(t *testing.T) {
	resetMessagesSessionAffinityForTest()
	defer resetMessagesSessionAffinityForTest()

	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	body := []byte(`{"metadata":{"user_id":"{\"device_id\":\"device-123\",\"session_id\":\"session-123456\"}"}}`)
	setMessagesSessionAffinityContext(c, false, body)
	if _, ok := c.Get("messagesSessionAffinityKey"); ok {
		t.Fatalf("direct account request must not install messages session affinity")
	}

	setMessagesSessionAffinityContext(c, true, body)
	if key, ok := c.Get("messagesSessionAffinityKey"); !ok || key == "" {
		t.Fatalf("pool request should install messages session affinity key")
	}
	if source, _ := c.Get("messagesSessionAffinitySource"); source != "metadata.user_id.session_id" {
		t.Fatalf("source = %v, want metadata.user_id.session_id", source)
	}
}

func TestShouldRetryMessagesDetachedSameAccount(t *testing.T) {
	resolved := &resolvedAccount{AccountID: "acct-1"}

	if got := shouldRetryMessagesDetachedSameAccount(true, resolved, true, false, 0, 3); got != "acct-1" {
		t.Fatalf("same-account detached retry = %q, want acct-1", got)
	}
	if got := shouldRetryMessagesDetachedSameAccount(true, resolved, true, true, 1, 3); got != "" {
		t.Fatalf("detached follow-up must not re-arm same-account retry, got %q", got)
	}
	if got := shouldRetryMessagesDetachedSameAccount(true, resolved, false, false, 0, 3); got != "" {
		t.Fatalf("non-canonical continuation must not arm same-account detached retry, got %q", got)
	}
	if got := shouldRetryMessagesDetachedSameAccount(false, resolved, true, false, 0, 3); got != "" {
		t.Fatalf("non-continuation request must not arm same-account detached retry, got %q", got)
	}
	if got := shouldRetryMessagesDetachedSameAccount(true, resolved, true, false, 0, 1); got != "" {
		t.Fatalf("no remaining retry budget must not arm same-account detached retry, got %q", got)
	}
}
