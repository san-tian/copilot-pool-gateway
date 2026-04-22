package store

import (
	"path/filepath"
	"testing"
)

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
