package handler

import (
	"context"
	"log"
	"strings"

	"copilot-go/config"
	"copilot-go/instance"
	"copilot-go/store"
)

// adoptManagedWorkerForAccount best-effort migrates the account onto a
// supervisor-owned worker when the runtime is configured for auto-adopt.
// Errors are logged and the original account remains usable through the legacy
// direct path, matching the existing fail-soft device-flow behavior.
func adoptManagedWorkerForAccount(ctx context.Context, account *store.Account, githubToken string) *store.Account {
	if account == nil {
		return nil
	}
	token := strings.TrimSpace(githubToken)
	if token == "" {
		return account
	}
	sup := instance.DefaultSupervisor()
	if sup == nil || !config.WorkerAutoAdopt() {
		return account
	}
	if err := sup.RemoveAndCleanup(ctx, account.ID); err != nil {
		log.Printf("supervisor: remove-and-cleanup %s: %v", account.ID, err)
	}
	if _, err := sup.Spawn(ctx, account.ID, token); err != nil {
		log.Printf("supervisor: spawn %s: %v (falling through to direct mode)", account.ID, err)
		return account
	}
	fresh, err := store.GetAccount(account.ID)
	if err != nil {
		log.Printf("supervisor: refresh account %s after spawn: %v", account.ID, err)
		return account
	}
	if fresh != nil {
		return fresh
	}
	return account
}
