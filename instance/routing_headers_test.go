package instance

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestApplyRoutingResponseHeadersIncludesAccountAndPoolStrategy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Set("poolStrategy", "smart")

	applyRoutingResponseHeaders(c, "acc-123")

	if got := rec.Header().Get(responseHeaderCopilotAccountID); got != "acc-123" {
		t.Fatalf("account header = %q, want acc-123", got)
	}
	if got := rec.Header().Get(responseHeaderCopilotPoolStrategy); got != "smart" {
		t.Fatalf("pool strategy header = %q, want smart", got)
	}
}

func TestApplyRoutingResponseHeadersSkipsEmptyValues(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)

	applyRoutingResponseHeaders(c, "")

	if got := rec.Header().Get(responseHeaderCopilotAccountID); got != "" {
		t.Fatalf("account header = %q, want empty", got)
	}
	if got := rec.Header().Get(responseHeaderCopilotPoolStrategy); got != "" {
		t.Fatalf("pool strategy header = %q, want empty", got)
	}
}
