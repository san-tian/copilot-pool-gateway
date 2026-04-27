package handler

import (
	"crypto/rand"
	"encoding/hex"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	traceIDContextKey = "traceID"
	traceIDHeader     = "X-Trace-Id"
	traceIDMaxLength  = 64
)

var traceIDPattern = regexp.MustCompile(`^\w[\w.-]*$`)

func proxyTrace() gin.HandlerFunc {
	return func(c *gin.Context) {
		traceID := resolveTraceID(c.GetHeader(traceIDHeader))
		c.Set(traceIDContextKey, traceID)
		c.Request.Header.Set(traceIDHeader, traceID)
		c.Header(traceIDHeader, traceID)
		c.Next()
	}
}

func currentTraceID(c *gin.Context) string {
	if v, ok := c.Get(traceIDContextKey); ok {
		if traceID, _ := v.(string); traceID != "" {
			return traceID
		}
	}
	traceID := resolveTraceID(c.GetHeader(traceIDHeader))
	c.Set(traceIDContextKey, traceID)
	c.Request.Header.Set(traceIDHeader, traceID)
	c.Header(traceIDHeader, traceID)
	return traceID
}

func resolveTraceID(candidate string) string {
	traceID := strings.TrimSpace(candidate)
	if traceID == "" || len(traceID) > traceIDMaxLength || !traceIDPattern.MatchString(traceID) {
		return newTraceID()
	}
	return traceID
}

func newTraceID() string {
	var random [4]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "gw-" + strconv.FormatInt(time.Now().UnixMilli(), 36)
	}
	return "gw-" + strconv.FormatInt(time.Now().UnixMilli(), 36) + "-" + hex.EncodeToString(random[:])
}
