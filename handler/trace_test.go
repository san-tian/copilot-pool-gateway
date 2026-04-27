package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestProxyTracePreservesValidHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(proxyTrace())
	r.GET("/trace", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"trace_id": currentTraceID(c)})
	})

	req := httptest.NewRequest(http.MethodGet, "/trace", nil)
	req.Header.Set(traceIDHeader, "trace-123_ABC")
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if got := rec.Header().Get(traceIDHeader); got != "trace-123_ABC" {
		t.Fatalf("expected response trace header to be preserved, got %q", got)
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if got := payload["trace_id"]; got != "trace-123_ABC" {
		t.Fatalf("expected trace_id trace-123_ABC, got %q", got)
	}
}

func TestProxyTraceGeneratesForInvalidHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(proxyTrace())
	r.GET("/trace", func(c *gin.Context) {
		c.Status(http.StatusNoContent)
	})

	req := httptest.NewRequest(http.MethodGet, "/trace", nil)
	req.Header.Set(traceIDHeader, "bad trace value")
	rec := httptest.NewRecorder()

	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected 204, got %d", rec.Code)
	}
	got := rec.Header().Get(traceIDHeader)
	if got == "" {
		t.Fatal("expected generated trace header")
	}
	if got == "bad trace value" {
		t.Fatal("expected invalid trace id to be replaced")
	}
}
