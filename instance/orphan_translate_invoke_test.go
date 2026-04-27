package instance

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"copilot-go/config"
	"copilot-go/store"
)

func setupOrphanInvokeTestEnv(t *testing.T) {
	t.Helper()
	oldAppDir := store.AppDir
	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	t.Cleanup(func() {
		store.AppDir = oldAppDir
	})
}

func mustReadBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll(resp.Body): %v", err)
	}
	return string(body)
}

func TestDoOrphanTranslateResponsesProxyFallsBackWithoutWorkerURL(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	var seenPath string
	withStreamingClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			seenPath = req.URL.Path
			if got := req.Header.Get("X-Interaction-Type"); got != copilotInteractionTypeUser {
				t.Fatalf("X-Interaction-Type = %q", got)
			}
			body, _ := io.ReadAll(req.Body)
			if !strings.Contains(string(body), `"messages"`) {
				t.Fatalf("translated chat body missing messages: %s", string(body))
			}
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"teapot"}`)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateResponsesProxy(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-4o-mini","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateResponsesProxy returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if turnRequest.CacheSource != "orphan_translate_fresh" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}
	if seenPath != "/chat/completions" {
		t.Fatalf("path = %q", seenPath)
	}
}

func TestDoOrphanTranslateResponsesProxyDirectFallbackWrapsStreamingSuccess(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	withStreamingClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/chat/completions" {
				t.Fatalf("path = %q", req.URL.Path)
			}
			if got := req.Header.Get("X-Interaction-Type"); got != copilotInteractionTypeUser {
				t.Fatalf("X-Interaction-Type = %q", got)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(
					"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Recovered\"}}]}\n\n" +
						"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\" answer\"},\"finish_reason\":\"stop\"}]}\n\n" +
						"data: [DONE]\n\n",
				)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateResponsesProxy(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-4o-mini","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateResponsesProxy returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q", got)
	}
	if turnRequest.CacheSource != "orphan_translate_fresh" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}

	body := mustReadBody(t, resp)
	for _, want := range []string{
		"event: response.created",
		"event: response.output_text.delta",
		"Recovered",
		"answer",
		"event: response.completed",
		"\"status\":\"completed\"",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected wrapped responses stream to contain %q, got:\n%s", want, body)
		}
	}
}

func TestDoOrphanTranslateResponsesProxyWithTurnReusesAgentHeaders(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	if _, err := store.UpdateAccount(account.ID, map[string]interface{}{"workerUrl": "http://127.0.0.1:1"}); err != nil {
		t.Fatalf("UpdateAccount workerUrl: %v", err)
	}

	baseTurn := copilotTurnRequest{
		Context: copilotTurnContext{
			InteractionID:   "interaction-1",
			ClientSessionID: "client-session-1",
			AgentTaskID:     "agent-task-1",
		},
		InteractionType: copilotInteractionTypeAgent,
		Initiator:       "agent",
		CacheSource:     "fc_output_id_hit",
	}

	withWorkerClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/v1/chat/completions" {
				t.Fatalf("path = %q", req.URL.Path)
			}
			assertHeaderValue(t, req.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
			assertHeaderValue(t, req.Header, "X-Initiator", "agent")
			assertHeaderValue(t, req.Header, "X-Interaction-Id", "interaction-1")
			assertHeaderValue(t, req.Header, "X-Client-Session-Id", "client-session-1")
			assertHeaderValue(t, req.Header, "X-Agent-Task-Id", "agent-task-1")
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"teapot"}`)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateResponsesProxyWithTurn(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-4o-mini","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), baseTurn, "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateResponsesProxyWithTurn returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if turnRequest.CacheSource != "orphan_translate_reuse_turn" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}
	if turnRequest.Context != baseTurn.Context {
		t.Fatalf("expected reused context %+v, got %+v", baseTurn.Context, turnRequest.Context)
	}
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("InteractionType = %q", turnRequest.InteractionType)
	}
}

func TestDoOrphanTranslateMessagesProxyFallsBackWithoutWorkerURL(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	var seenPath string
	withStreamingClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			seenPath = req.URL.Path
			if got := req.Header.Get("X-Interaction-Type"); got != copilotInteractionTypeUser {
				t.Fatalf("X-Interaction-Type = %q", got)
			}
			body, _ := io.ReadAll(req.Body)
			if !strings.Contains(string(body), `"messages"`) {
				t.Fatalf("translated direct-messages bridge body missing messages: %s", string(body))
			}
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"teapot"}`)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateMessagesProxy(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateMessagesProxy returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if turnRequest.CacheSource != "orphan_translate_messages_fresh" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}
	if seenPath != "/chat/completions" {
		t.Fatalf("path = %q", seenPath)
	}
}

func TestDoOrphanTranslateMessagesProxyDirectFallbackWrapsStreamingSuccess(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	withStreamingClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/chat/completions" {
				t.Fatalf("path = %q", req.URL.Path)
			}
			if got := req.Header.Get("X-Interaction-Type"); got != copilotInteractionTypeUser {
				t.Fatalf("X-Interaction-Type = %q", got)
			}
			body, _ := io.ReadAll(req.Body)
			if !strings.Contains(string(body), `"messages"`) {
				t.Fatalf("translated direct-messages bridge body missing messages: %s", string(body))
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(
					"data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"delta\":{\"content\":\"Recovered gpt-5\"}}]}\n\n" +
						"data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"delta\":{\"content\":\" via bridge\"},\"finish_reason\":\"stop\"}]}\n\n" +
						"data: [DONE]\n\n",
				)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateMessagesProxy(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateMessagesProxy returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q", got)
	}
	if turnRequest.CacheSource != "orphan_translate_messages_fresh" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}

	body := mustReadBody(t, resp)
	for _, want := range []string{
		"event: response.created",
		"event: response.output_text.delta",
		"Recovered gpt-5",
		"via bridge",
		"event: response.completed",
		"\"status\":\"completed\"",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected wrapped responses stream to contain %q, got:\n%s", want, body)
		}
	}
}

func TestDoOrphanTranslateMessagesProxyWithTurnReusesAgentHeaders(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	if _, err := store.UpdateAccount(account.ID, map[string]interface{}{"workerUrl": "http://127.0.0.1:1"}); err != nil {
		t.Fatalf("UpdateAccount workerUrl: %v", err)
	}

	baseTurn := copilotTurnRequest{
		Context: copilotTurnContext{
			InteractionID:   "interaction-2",
			ClientSessionID: "client-session-2",
			AgentTaskID:     "agent-task-2",
		},
		InteractionType: copilotInteractionTypeAgent,
		Initiator:       "agent",
		CacheSource:     "fc_output_id_hit",
	}

	withWorkerClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Path != "/v1/messages" {
				t.Fatalf("path = %q", req.URL.Path)
			}
			assertHeaderValue(t, req.Header, "X-Interaction-Type", copilotInteractionTypeAgent)
			assertHeaderValue(t, req.Header, "X-Initiator", "agent")
			assertHeaderValue(t, req.Header, "X-Interaction-Id", "interaction-2")
			assertHeaderValue(t, req.Header, "X-Client-Session-Id", "client-session-2")
			assertHeaderValue(t, req.Header, "X-Agent-Task-Id", "agent-task-2")
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"teapot"}`)),
			}, nil
		}),
	})

	resp, _, turnRequest, err := DoOrphanTranslateMessagesProxyWithTurn(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`), baseTurn, "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateMessagesProxyWithTurn returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if turnRequest.CacheSource != "orphan_translate_messages_reuse_turn" {
		t.Fatalf("CacheSource = %q", turnRequest.CacheSource)
	}
	if turnRequest.Context != baseTurn.Context {
		t.Fatalf("expected reused context %+v, got %+v", baseTurn.Context, turnRequest.Context)
	}
	if turnRequest.InteractionType != copilotInteractionTypeAgent {
		t.Fatalf("InteractionType = %q", turnRequest.InteractionType)
	}
}

func TestDoOrphanTranslateMessagesProxyDirectFallbackUsesMaxCompletionTokensForGpt5(t *testing.T) {
	setupOrphanInvokeTestEnv(t)

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	var gotBody []byte
	withStreamingClient(t, &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			gotBody, _ = io.ReadAll(req.Body)
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"teapot"}`)),
			}, nil
		}),
	})

	resp, _, _, err := DoOrphanTranslateMessagesProxy(account.ID, &config.State{
		CopilotToken:  "copilot-token",
		VSCodeVersion: "1.99.0",
		AccountType:   "individual",
	}, []byte(`{"model":"gpt-5.4","max_output_tokens":64,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},{"type":"function_call","call_id":"call_1","name":"noop","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"ok"}]}`), "")
	if err != nil {
		t.Fatalf("DoOrphanTranslateMessagesProxy returned error: %v", err)
	}
	if resp == nil {
		t.Fatalf("expected response, got nil")
	}
	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var forwarded map[string]interface{}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatalf("failed to unmarshal forwarded body: %v", err)
	}
	if _, ok := forwarded["max_tokens"]; ok {
		t.Fatalf("expected max_tokens to be removed for gpt-5 orphan direct fallback, got %s", string(gotBody))
	}
	if got, ok := forwarded["max_completion_tokens"].(float64); !ok || got != 64 {
		t.Fatalf("expected max_completion_tokens=64, got %#v from %s", forwarded["max_completion_tokens"], string(gotBody))
	}
}
