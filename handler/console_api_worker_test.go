package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

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
