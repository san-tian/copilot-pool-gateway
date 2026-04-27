package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"testing"
	"time"

	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

func setupAccountsEnv(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(store.AppDirEnvVar, dir)
	prev := store.AppDir
	store.AppDir = dir
	t.Cleanup(func() { store.AppDir = prev })
	if err := store.EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
}

// clearDefaultSupervisor ensures no leftover supervisor leaks between tests.
// Phase 0c's handlers nil-check; the tests here deliberately run with no
// supervisor wired so we assert the "0c ships before 0d" fallthrough path.
func clearDefaultSupervisor(t *testing.T) {
	t.Helper()
	prev := instance.DefaultSupervisor()
	instance.SetDefaultSupervisor(nil)
	t.Cleanup(func() { instance.SetDefaultSupervisor(prev) })
}

func TestHandleRestartWorker_NoSupervisorReturns503(t *testing.T) {
	clearDefaultSupervisor(t)
	setupAccountsEnv(t)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Params = gin.Params{{Key: "id", Value: "acct-1"}}
	c.Request = httptest.NewRequest(http.MethodPost, "/api/accounts/acct-1/worker/restart", nil)

	handleRestartWorker(c)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body = %s", rec.Code, rec.Body.String())
	}
}

func TestHandleRestartWorker_AccountMissingReturns404(t *testing.T) {
	clearDefaultSupervisor(t)
	setupAccountsEnv(t)
	// Install a non-nil supervisor so we pass the nil-check and reach the
	// store lookup. Using a zero-value pointer is fine — the test never
	// actually calls Restart because the lookup fails first.
	instance.SetDefaultSupervisor(&instance.WorkerSupervisor{})

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Params = gin.Params{{Key: "id", Value: "acct-missing"}}
	c.Request = httptest.NewRequest(http.MethodPost, "/api/accounts/acct-missing/worker/restart", nil)

	handleRestartWorker(c)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body = %s", rec.Code, rec.Body.String())
	}
}

func TestHandleRestartWorker_NoTokenReturns400(t *testing.T) {
	clearDefaultSupervisor(t)
	setupAccountsEnv(t)
	instance.SetDefaultSupervisor(&instance.WorkerSupervisor{})

	// Seed an account with no githubToken so the handler rejects before it
	// would otherwise call Restart on the zero-value supervisor (which would
	// segfault). This exercises the "re-auth first" branch.
	acct, err := store.AddAccount("x", "", "")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Params = gin.Params{{Key: "id", Value: acct.ID}}
	c.Request = httptest.NewRequest(http.MethodPost, "/api/accounts/"+acct.ID+"/worker/restart", nil)

	handleRestartWorker(c)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", rec.Code, rec.Body.String())
	}
	var body map[string]string
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	if body["error"] == "" {
		t.Errorf("expected error message, got %s", rec.Body.String())
	}
}

func TestHandleDeleteAccount_NilSupervisorStillDeletes(t *testing.T) {
	clearDefaultSupervisor(t)
	setupAccountsEnv(t)

	acct, err := store.AddAccount("x", "", "")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Params = gin.Params{{Key: "id", Value: acct.ID}}
	c.Request = httptest.NewRequest(http.MethodDelete, "/api/accounts/"+acct.ID, nil)

	handleDeleteAccount(c)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	got, err := store.GetAccount(acct.ID)
	if err != nil {
		t.Fatalf("GetAccount: %v", err)
	}
	if got != nil {
		t.Fatalf("account not deleted: %+v", got)
	}
}

func TestHandleAddAccount_AutoAdoptsWorkerWhenSupervisorEnabled(t *testing.T) {
	clearDefaultSupervisor(t)
	setupAccountsEnv(t)

	sup, err := instance.NewSupervisor(instance.SupervisorOptions{
		PortRangeStart: 55340,
		PortRangeEnd:   55349,
		WorkersRoot:    t.TempDir(),
		ReadyTimeout:   2 * time.Second,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error { return nil },
		PersistOnReady: func(entry instance.WorkerEntry) error {
			return store.UpdateAccountWorker(entry.AccountID, entry.WorkerURL, entry.Home, entry.Port, entry.PID)
		},
		PersistOnStop: store.ClearAccountWorker,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	instance.SetDefaultSupervisor(sup)
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	body := []byte(`{"name":"worker-new","githubToken":"tok-new","accountType":"individual"}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/accounts", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	handleAddAccount(c)

	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body = %s", rec.Code, rec.Body.String())
	}
	var got store.Account
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("Unmarshal response: %v; body=%s", err, rec.Body.String())
	}
	if !got.WorkerManaged {
		t.Fatalf("WorkerManaged = false: %+v", got)
	}
	if got.WorkerURL == "" || got.WorkerPort == 0 || got.WorkerPID == 0 {
		t.Fatalf("missing worker fields after auto-adopt: %+v", got)
	}
	stored, err := store.GetAccount(got.ID)
	if err != nil || stored == nil {
		t.Fatalf("GetAccount: %v, %v", stored, err)
	}
	if !stored.WorkerManaged || stored.WorkerURL == "" {
		t.Fatalf("stored account not updated: %+v", stored)
	}
}
