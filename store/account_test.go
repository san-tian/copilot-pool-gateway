package store

import (
	"path/filepath"
	"testing"
)

func seedAccountStore(t *testing.T, accounts ...Account) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(AppDirEnvVar, dir)
	withAppDir(t, dir)
	if err := EnsurePaths(); err != nil {
		t.Fatalf("EnsurePaths: %v", err)
	}
	// EnsurePaths seeds empty "{}"; if the caller handed us accounts, overwrite.
	if len(accounts) > 0 {
		if err := writeAccounts(accounts); err != nil {
			t.Fatalf("writeAccounts: %v", err)
		}
	}
}

func TestUpdateAccountWorker_SetsAllFieldsAtomically(t *testing.T) {
	seedAccountStore(t, Account{ID: "acct-1", Name: "test", Enabled: true})

	if err := UpdateAccountWorker("acct-1", "http://127.0.0.1:55300", "/home/workers/acct-1", 55300, 12345); err != nil {
		t.Fatalf("UpdateAccountWorker: %v", err)
	}
	got, err := GetAccount("acct-1")
	if err != nil || got == nil {
		t.Fatalf("GetAccount: %v, %v", got, err)
	}
	if got.WorkerURL != "http://127.0.0.1:55300" {
		t.Errorf("WorkerURL = %q", got.WorkerURL)
	}
	if got.WorkerHome != "/home/workers/acct-1" {
		t.Errorf("WorkerHome = %q", got.WorkerHome)
	}
	if got.WorkerPort != 55300 {
		t.Errorf("WorkerPort = %d", got.WorkerPort)
	}
	if got.WorkerPID != 12345 {
		t.Errorf("WorkerPID = %d", got.WorkerPID)
	}
	if !got.WorkerManaged {
		t.Errorf("WorkerManaged = false, want true")
	}
}

func TestUpdateAccountWorker_UnknownAccountIsNoOp(t *testing.T) {
	seedAccountStore(t, Account{ID: "acct-1"})
	if err := UpdateAccountWorker("acct-missing", "http://x", "/h", 55300, 1); err != nil {
		t.Fatalf("UpdateAccountWorker unknown: %v", err)
	}
	got, _ := GetAccount("acct-1")
	if got == nil || got.WorkerManaged {
		t.Fatalf("unrelated account mutated: %+v", got)
	}
}

func TestClearAccountWorker_ZeroesAllFields(t *testing.T) {
	seedAccountStore(t, Account{
		ID:            "acct-1",
		WorkerURL:     "http://127.0.0.1:55300",
		WorkerHome:    "/h",
		WorkerPort:    55300,
		WorkerPID:     999,
		WorkerManaged: true,
	})

	if err := ClearAccountWorker("acct-1"); err != nil {
		t.Fatalf("ClearAccountWorker: %v", err)
	}
	got, err := GetAccount("acct-1")
	if err != nil || got == nil {
		t.Fatalf("GetAccount: %v, %v", got, err)
	}
	if got.WorkerURL != "" || got.WorkerHome != "" || got.WorkerPort != 0 || got.WorkerPID != 0 || got.WorkerManaged {
		t.Errorf("worker fields not cleared: %+v", got)
	}
}

func TestUpdateAccount_DoesNotExposeNewWorkerFields(t *testing.T) {
	// Admin UpdateAccount(map) must not let callers set WorkerPort/PID/Home/Managed.
	// Only the existing workerUrl key stays, for backward compat with the admin UI.
	seedAccountStore(t, Account{ID: "acct-1"})
	_, err := UpdateAccount("acct-1", map[string]interface{}{
		"workerPort":    55300,
		"workerPid":     12345,
		"workerHome":    "/bad",
		"workerManaged": true,
	})
	if err != nil {
		t.Fatalf("UpdateAccount: %v", err)
	}
	got, _ := GetAccount("acct-1")
	if got == nil {
		t.Fatalf("account missing")
	}
	if got.WorkerPort != 0 || got.WorkerPID != 0 || got.WorkerHome != "" || got.WorkerManaged {
		t.Errorf("admin setter leaked supervisor-owned fields: %+v", got)
	}
}

func TestAccountsFile_UsesResolvedAppDir(t *testing.T) {
	// Guard against the seed helper drifting — confirms AccountsFile() lives under
	// the t.TempDir() we set, not some leftover global state.
	seedAccountStore(t, Account{ID: "acct-1"})
	want := filepath.Join(AppDir, "accounts.json")
	if AccountsFile() != want {
		t.Fatalf("AccountsFile = %q, want %q", AccountsFile(), want)
	}
}
