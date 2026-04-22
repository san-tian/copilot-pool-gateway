package instance

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"copilot-go/store"
)

func setupProbeTestEnv(t *testing.T, handler http.HandlerFunc) func() {
	t.Helper()

	oldAppDir := store.AppDir
	oldGithubURL := githubCopilotURL
	oldBaseURL := copilotBaseURL

	store.AppDir = t.TempDir()
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}

	srv := httptest.NewServer(handler)
	githubCopilotURL = srv.URL + "/copilot/token"
	copilotBaseURL = func(accountType string) string {
		return srv.URL + "/" + accountType
	}

	return func() {
		srv.Close()
		store.AppDir = oldAppDir
		githubCopilotURL = oldGithubURL
		copilotBaseURL = oldBaseURL
		mu.Lock()
		for _, inst := range instances {
			if inst.stopChan != nil {
				close(inst.stopChan)
			}
		}
		instances = make(map[string]*ProxyInstance)
		mu.Unlock()
	}
}

func writeToken(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"token":      "copilot-token",
		"expires_at": time.Now().Add(time.Hour).Unix(),
	})
}

func writeModels(w http.ResponseWriter, modelIDs ...string) {
	data := make([]map[string]any, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		data = append(data, map[string]any{
			"id":       modelID,
			"object":   "model",
			"owned_by": "github",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func decodePostedBody(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return body
}

func postedModel(t *testing.T, r *http.Request) string {
	t.Helper()
	body := decodePostedBody(t, r)
	model, _ := body["model"].(string)
	return model
}

func TestProbeAccountSelectsFallbackTypeWithPremiumModel(t *testing.T) {
	cleanup := setupProbeTestEnv(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/copilot/token":
			writeToken(w)
		case "/individual/models":
			http.Error(w, "unauthorized: chat not enabled for IDE token", http.StatusUnauthorized)
		case "/business/models":
			writeModels(w, "gpt-5.4", "gpt-4.1")
		case "/business/chat/completions":
			body := decodePostedBody(t, r)
			if model, _ := body["model"].(string); model == "gpt-5.4" {
				if _, hasWrong := body["max_tokens"]; hasWrong {
					t.Fatalf("gpt-5.4 probe should not send max_tokens: %#v", body)
				}
				if got, ok := body["max_completion_tokens"].(float64); !ok || got != 1 {
					t.Fatalf("gpt-5.4 probe should send max_completion_tokens=1, got %#v", body)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
				return
			}
			http.Error(w, "model_not_supported: requested model unavailable", http.StatusBadRequest)
		default:
			http.NotFound(w, r)
		}
	})
	defer cleanup()

	account, err := store.AddAccount("demo", "gh-token", "")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	probe, state := ProbeAccount(*account, "")
	if probe == nil {
		t.Fatal("expected probe result")
	}
	if !probe.Success {
		t.Fatalf("expected success, got error: %s", probe.Error)
	}
	if probe.AccountType != "business" {
		t.Fatalf("expected business fallback, got %s", probe.AccountType)
	}
	if len(probe.SupportedModels) != 1 || probe.SupportedModels[0] != "gpt-5.4" {
		t.Fatalf("expected gpt-5.4 support, got %#v", probe.SupportedModels)
	}
	if state == nil || state.AccountType != "business" {
		t.Fatalf("expected probed state for business, got %#v", state)
	}
}

func TestReconcileAccountDisablesLowTierOnlyAccount(t *testing.T) {
	cleanup := setupProbeTestEnv(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/copilot/token":
			writeToken(w)
		case "/individual/models":
			writeModels(w, "gpt-4.1", "gpt-4o")
		default:
			http.NotFound(w, r)
		}
	})
	defer cleanup()

	account, err := store.AddAccount("demo", "gh-token", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	updated, probe, err := ReconcileAccount(*account)
	if err != nil {
		t.Fatalf("ReconcileAccount: %v", err)
	}
	if probe == nil || probe.Success {
		t.Fatalf("expected failed probe, got %#v", probe)
	}
	if updated == nil || updated.Enabled {
		t.Fatalf("expected disabled account, got %#v", updated)
	}
	if updated.ProbeStatus != "failed" {
		t.Fatalf("expected failed probe status, got %q", updated.ProbeStatus)
	}
	if updated.ProbeError == "" || updated.ProbeError == probe.Error {
		// keep the test easy to read, just ensure the persisted message is populated
	}
	if got := GetInstanceStatus(account.ID); got != "error" {
		t.Fatalf("expected error instance status, got %s", got)
	}
	if probe.Error == "" || updated.ProbeError == "" {
		t.Fatal("expected premium probe failure details to be persisted")
	}
}

func TestReconcileAccountPersistsSupportedModelsAndStartsInstance(t *testing.T) {
	cleanup := setupProbeTestEnv(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/copilot/token":
			writeToken(w)
		case "/business/models":
			writeModels(w, "gpt-5.4", "claude-opus-4.6")
		case "/business/chat/completions":
			switch postedModel(t, r) {
			case "gpt-5.4":
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
			case "claude-opus-4.6":
				http.Error(w, "model_not_supported: requested model unavailable", http.StatusBadRequest)
			default:
				http.Error(w, "model_not_supported: requested model unavailable", http.StatusBadRequest)
			}
		default:
			http.NotFound(w, r)
		}
	})
	defer cleanup()

	account, err := store.AddAccount("demo", "gh-token", "business")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	updatedAccount, err := store.UpdateAccount(account.ID, map[string]interface{}{"blockedModels": []string{"legacy-model"}})
	if err != nil {
		t.Fatalf("UpdateAccount: %v", err)
	}

	updated, probe, err := ReconcileAccount(*updatedAccount)
	if err != nil {
		t.Fatalf("ReconcileAccount: %v", err)
	}
	if probe == nil || !probe.Success {
		t.Fatalf("expected successful probe, got %#v", probe)
	}
	if updated == nil || !updated.Enabled {
		t.Fatalf("expected enabled account, got %#v", updated)
	}
	if updated.AccountType != "business" {
		t.Fatalf("expected detected business type, got %s", updated.AccountType)
	}
	if len(updated.SupportedModels) != 1 || updated.SupportedModels[0] != "gpt-5.4" {
		t.Fatalf("expected persisted gpt-5.4 support, got %#v", updated.SupportedModels)
	}
	if got := GetInstanceStatus(account.ID); got != "running" {
		t.Fatalf("expected running instance, got %s", got)
	}

	stored, err := store.GetAccount(account.ID)
	if err != nil {
		t.Fatalf("GetAccount: %v", err)
	}
	if stored == nil || stored.AccountType != "business" || !stored.Enabled {
		t.Fatalf("expected persisted business+enabled account, got %#v", stored)
	}
	if len(stored.UnsupportedModels) == 0 {
		t.Fatalf("expected unsupported premium models to be recorded, got %#v", stored.UnsupportedModels)
	}
	foundLegacy := false
	foundBlockedPremium := false
	for _, model := range stored.BlockedModels {
		if model == "legacy-model" {
			foundLegacy = true
		}
		if model == "gpt-5-codex" || model == "claude-opus-4.7" || model == "claude-opus-4.6" {
			foundBlockedPremium = true
		}
	}
	if !foundLegacy || !foundBlockedPremium {
		t.Fatalf("expected blocked models to preserve legacy entry and premium unsupported entries, got %#v", stored.BlockedModels)
	}
}
