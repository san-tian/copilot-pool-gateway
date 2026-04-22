package instance

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"strings"

	"copilot-go/anthropic"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

const (
	internalMessagesUpstreamModeHeader = "X-Copilot2Api-Internal-Messages-Mode"
	messagesUpstreamModeNativeAnthropic = "native-anthropic"
)

type nativeAnthropicConfig struct {
	BaseURL   string
	AuthToken string
	APIKey    string
}

func loadNativeAnthropicConfig() nativeAnthropicConfig {
	return nativeAnthropicConfig{
		BaseURL:   strings.TrimRight(strings.TrimSpace(os.Getenv("COPILOT_NATIVE_ANTHROPIC_BASE_URL")), "/"),
		AuthToken: strings.TrimSpace(os.Getenv("COPILOT_NATIVE_ANTHROPIC_AUTH_TOKEN")),
		APIKey:    strings.TrimSpace(os.Getenv("COPILOT_NATIVE_ANTHROPIC_API_KEY")),
	}
}

func (cfg nativeAnthropicConfig) enabled() bool {
	return cfg.BaseURL != "" && (cfg.AuthToken != "" || cfg.APIKey != "")
}

func shouldUseNativeAnthropicMessages(payload anthropic.AnthropicMessagesPayload) bool {
	cfg := loadNativeAnthropicConfig()
	if !cfg.enabled() {
		return false
	}
	modelID := strings.ToLower(strings.TrimSpace(store.ToCopilotID(payload.Model)))
	return strings.HasPrefix(modelID, "claude-opus-")
}

func doNativeAnthropicMessagesProxy(ctx context.Context, c *gin.Context, bodyBytes []byte) (*http.Response, error) {
	cfg := loadNativeAnthropicConfig()
	url := cfg.BaseURL + "/v1/messages"
	if rawQuery := strings.TrimSpace(c.Request.URL.RawQuery); rawQuery != "" {
		url += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}

	for name, values := range c.Request.Header {
		switch strings.ToLower(name) {
		case "authorization", "x-api-key", "host", "content-length":
			continue
		default:
			req.Header[name] = append([]string(nil), values...)
		}
	}
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if req.Header.Get("Accept") == "" {
		req.Header.Set("Accept", "application/json")
	}
	if req.Header.Get("anthropic-version") == "" {
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	if cfg.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.AuthToken)
	}
	if cfg.APIKey != "" {
		req.Header.Set("x-api-key", cfg.APIKey)
	}

	resp, err := getStreamingClient().Do(req)
	if err != nil {
		return nil, err
	}
	resp.Header.Set(internalMessagesUpstreamModeHeader, messagesUpstreamModeNativeAnthropic)
	return resp, nil
}

func isNativeAnthropicMessagesResponse(resp *http.Response) bool {
	return resp != nil && resp.Header.Get(internalMessagesUpstreamModeHeader) == messagesUpstreamModeNativeAnthropic
}

func forwardNativeAnthropicResponse(c *gin.Context, resp *http.Response) {
	defer func() { _ = resp.Body.Close() }()

	contentType := resp.Header.Get("Content-Type")
	isStream := strings.Contains(contentType, "text/event-stream")
	if isStream {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		// Close the TCP socket after the stream finishes — see the matching
		// rationale in forwardResponsesStream.
		c.Header("Connection", "close")
		c.Request.Close = true
		c.Header("X-Accel-Buffering", "no")
		c.Status(resp.StatusCode)

		reader := bufio.NewReaderSize(resp.Body, 10*1024*1024)
		sawMessageStop := false
		lastBlank := false
		c.Stream(func(w io.Writer) bool {
			line, err := reader.ReadBytes('\n')
			if len(line) > 0 {
				trimmed := bytes.TrimRight(line, "\r\n")
				if len(trimmed) == 0 {
					lastBlank = true
				} else {
					lastBlank = false
					if bytes.HasPrefix(trimmed, []byte("event:")) {
						ev := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("event:")))
						if bytes.Equal(ev, []byte("message_stop")) {
							sawMessageStop = true
						}
					} else if bytes.HasPrefix(trimmed, []byte("data:")) {
						payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
						if bytes.Contains(payload, []byte(`"type":"message_stop"`)) {
							sawMessageStop = true
						}
					}
				}
				if _, writeErr := w.Write(line); writeErr != nil {
					return false
				}
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
				// Anthropic's terminal event is `message_stop`. Once it and
				// its trailing blank line have been forwarded, stop reading —
				// some upstreams hold the socket open for keep-alive reuse
				// after the final event, which would leave downstream agents
				// spinning in a working state waiting for close.
				if sawMessageStop && lastBlank {
					return false
				}
			}
			if err != nil {
				return false
			}
			return true
		})
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read response"})
		return
	}
	if contentType == "" {
		contentType = "application/json"
	}
	c.Data(resp.StatusCode, contentType, body)
}
