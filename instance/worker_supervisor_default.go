package instance

import "sync/atomic"

// defaultSupervisor holds the process-wide WorkerSupervisor instance. It is
// set once at startup by main.go after NewSupervisor+RecoverFromStore and read
// by handlers (device-flow completion, account delete, manual restart) that
// need to spawn/stop workers in response to admin actions.
//
// atomic.Pointer gives lock-free reads and is nil-safe for the "supervisor not
// yet wired" window (tests, partial rollouts). Handlers MUST nil-check before
// dereferencing.
var defaultSupervisor atomic.Pointer[WorkerSupervisor]

// SetDefaultSupervisor installs the process-wide supervisor. Intended to be
// called exactly once from main.go during startup. Passing nil clears it
// (useful in tests).
func SetDefaultSupervisor(s *WorkerSupervisor) {
	defaultSupervisor.Store(s)
}

// DefaultSupervisor returns the process-wide supervisor, or nil if none has
// been installed yet. Callers MUST nil-check.
func DefaultSupervisor() *WorkerSupervisor {
	return defaultSupervisor.Load()
}
