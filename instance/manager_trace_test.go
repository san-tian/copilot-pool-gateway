package instance

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProxyRequestViaWorkerForwardsTraceID(t *testing.T) {
	t.Parallel()

	var gotTraceID string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotTraceID = r.Header.Get("X-Trace-Id")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	resp, err := ProxyRequestViaWorker(
		context.Background(),
		srv.URL,
		http.MethodPost,
		"/v1/responses",
		[]byte(`{"model":"gpt-5.4"}`),
		http.Header{"Content-Type": []string{"application/json"}},
		"trace-test-123",
	)
	if err != nil {
		t.Fatalf("ProxyRequestViaWorker returned error: %v", err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	if gotTraceID != "trace-test-123" {
		t.Fatalf("expected worker trace header trace-test-123, got %q", gotTraceID)
	}
}
