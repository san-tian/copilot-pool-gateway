package store

import (
	"path/filepath"
	"testing"
)

func withAppDir(t *testing.T, dir string) {
	t.Helper()
	prev := AppDir
	AppDir = dir
	t.Cleanup(func() { AppDir = prev })
}

func TestResolveAppDirFromEnvUsesOverride(t *testing.T) {
	override := t.TempDir()
	t.Setenv(AppDirEnvVar, override)
	if got := resolveAppDirFromEnv(); got != override {
		t.Fatalf("expected app dir override %q, got %q", override, got)
	}
}

func TestResolveAppDirFromEnvFallsBackToDefault(t *testing.T) {
	t.Setenv(AppDirEnvVar, "")
	got := resolveAppDirFromEnv()
	if got == "" {
		t.Fatal("expected non-empty default app dir")
	}
	if base := filepath.Base(got); base != "copilot-api" {
		t.Fatalf("expected default app dir to end with copilot-api, got %q", got)
	}
}

func TestWorkersRootDefault(t *testing.T) {
	t.Setenv(WorkersRootEnvVar, "")
	withAppDir(t, "/tmp/test-app-dir")
	got := WorkersRoot()
	want := filepath.Join("/tmp/test-app-dir", "workers")
	if got != want {
		t.Fatalf("WorkersRoot default = %q, want %q", got, want)
	}
}

func TestWorkersRootEnvOverride(t *testing.T) {
	override := t.TempDir()
	t.Setenv(WorkersRootEnvVar, override)
	if got := WorkersRoot(); got != override {
		t.Fatalf("WorkersRoot override = %q, want %q", got, override)
	}
}

func TestWorkerHomeForJoinsAccountID(t *testing.T) {
	t.Setenv(WorkersRootEnvVar, "/tmp/wroot")
	got := WorkerHomeFor("acct-xyz")
	want := filepath.Join("/tmp/wroot", "acct-xyz")
	if got != want {
		t.Fatalf("WorkerHomeFor = %q, want %q", got, want)
	}
}
