package instance

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"copilot-go/store"
)

// WorkerStatus tracks the lifecycle of a supervised copilot-api child.
type WorkerStatus string

const (
	WorkerStatusStarting     WorkerStatus = "starting"
	WorkerStatusReady        WorkerStatus = "ready"
	WorkerStatusCrashed      WorkerStatus = "crashed"
	WorkerStatusStopping     WorkerStatus = "stopping"
	WorkerStatusStopped      WorkerStatus = "stopped"
	WorkerStatusStartTimeout WorkerStatus = "start_timeout"
)

// WorkerEntry is a snapshot of a supervised worker as exposed to callers.
type WorkerEntry struct {
	AccountID string
	Port      int
	Home      string
	PID       int
	StartedAt time.Time
	Status    WorkerStatus
	LastError string
	WorkerURL string
}

// SupervisorOptions configures a WorkerSupervisor. Zero-value fields fall
// back to sensible defaults (see resolveDefaults).
type SupervisorOptions struct {
	ExePath        string
	PortRangeStart int
	PortRangeEnd   int
	WorkersRoot    string
	ReadyTimeout   time.Duration
	HealthInterval time.Duration

	// CommandFactory builds the child process command. If nil, a default
	// factory is used that invokes `<exePath> start --port <p>` with
	// COPILOT_API_HOME=<home>. Tests override this to run `sleep`.
	CommandFactory func(exePath string, port int, home string) *exec.Cmd

	// ReadinessProbe returns nil once the child at 127.0.0.1:<port> is
	// ready. If nil, httpModelsReady is used (GET /models, 2xx).
	ReadinessProbe func(ctx context.Context, port int) error

	// PersistOnReady is called after the worker passes its readiness probe,
	// with the ready WorkerEntry. Intended to atomically update the account
	// store (WorkerURL, Port, PID, Home, Managed). Errors are logged but do
	// not fail Spawn — the worker is already healthy in-memory. nil = no-op.
	PersistOnReady func(WorkerEntry) error

	// PersistOnStop is called after Stop tears down a worker and after the
	// readiness-timeout rollback path. Intended to zero the account store's
	// worker fields. Errors are logged but do not fail Stop. nil = no-op.
	PersistOnStop func(accountID string) error

	// Now is injectable for deterministic timestamps in tests.
	Now func() time.Time
}

// WorkerSupervisor owns the lifecycle of per-account copilot-api children.
type WorkerSupervisor struct {
	opts    SupervisorOptions
	mu      sync.Mutex
	workers map[string]*workerState
	ports   *portPool
	closed  bool
}

type workerState struct {
	entry   WorkerEntry
	cmd     *exec.Cmd
	logFile *os.File
	doneCh  chan struct{}
	adopted bool
}

var (
	ErrSupervisorClosed   = errors.New("supervisor: closed")
	ErrAccountAlreadyLive = errors.New("supervisor: account already has a live worker")
	ErrAccountUnknown     = errors.New("supervisor: account has no worker")
	ErrPortExhausted      = errors.New("supervisor: port pool exhausted")
	ErrPortReserved       = errors.New("supervisor: port already reserved")
	ErrReadyTimeout       = errors.New("supervisor: worker did not become ready before deadline")
	ErrInvalidPortRange   = errors.New("supervisor: invalid port range")
)

// NewSupervisor constructs a supervisor with the given options. It validates
// options but does NOT spawn any children.
func NewSupervisor(opts SupervisorOptions) (*WorkerSupervisor, error) {
	resolved, err := resolveDefaults(opts)
	if err != nil {
		return nil, err
	}
	pool, err := newPortPool(resolved.PortRangeStart, resolved.PortRangeEnd)
	if err != nil {
		return nil, err
	}
	return &WorkerSupervisor{
		opts:    resolved,
		workers: make(map[string]*workerState),
		ports:   pool,
	}, nil
}

func resolveDefaults(o SupervisorOptions) (SupervisorOptions, error) {
	if o.PortRangeStart == 0 && o.PortRangeEnd == 0 {
		o.PortRangeStart = 9100
		o.PortRangeEnd = 9199
	}
	if o.PortRangeStart <= 0 || o.PortRangeEnd < o.PortRangeStart {
		return o, ErrInvalidPortRange
	}
	if o.WorkersRoot == "" {
		o.WorkersRoot = store.WorkersRoot()
	}
	if o.ReadyTimeout <= 0 {
		o.ReadyTimeout = 20 * time.Second
	}
	if o.HealthInterval <= 0 {
		o.HealthInterval = 30 * time.Second
	}
	if o.CommandFactory == nil {
		o.CommandFactory = defaultCommandFactory
	}
	if o.ReadinessProbe == nil {
		o.ReadinessProbe = httpModelsReady
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return o, nil
}

func defaultCommandFactory(exePath string, port int, home string) *exec.Cmd {
	cmd := exec.Command(exePath, "start", "--port", strconv.Itoa(port))
	cmd.Env = append(os.Environ(), "COPILOT_API_HOME="+home)
	return cmd
}

// WorkerURLFor returns the loopback URL used for a given port.
func WorkerURLFor(port int) string {
	return fmt.Sprintf("http://127.0.0.1:%d", port)
}

func minDuration(left, right time.Duration) time.Duration {
	if left <= 0 {
		return right
	}
	if right <= 0 || left < right {
		return left
	}
	return right
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

func waitForProcessExit(ctx context.Context, pid int) error {
	if pid <= 0 {
		return nil
	}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		if !processAlive(pid) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// Spawn starts a new copilot-api worker for the account. github_token is
// written to <workersRoot>/<accountID>/github_token before the child starts
// so copilot-api's setupGitHubToken() picks it up without device flow.
//
// Spawn blocks until the child is ready or the ready timeout elapses. On
// failure it kills any partially-started child, releases the port, and
// returns an error; the account remains absent from the supervisor map.
func (s *WorkerSupervisor) Spawn(ctx context.Context, accountID, githubToken string) (WorkerEntry, error) {
	return s.spawn(ctx, accountID, githubToken, 0)
}

func (s *WorkerSupervisor) spawn(ctx context.Context, accountID, githubToken string, preferredPort int) (WorkerEntry, error) {
	if strings.TrimSpace(accountID) == "" {
		return WorkerEntry{}, errors.New("supervisor: empty accountID")
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return WorkerEntry{}, ErrSupervisorClosed
	}
	if _, exists := s.workers[accountID]; exists {
		s.mu.Unlock()
		return WorkerEntry{}, ErrAccountAlreadyLive
	}
	s.mu.Unlock()

	home := filepath.Join(s.opts.WorkersRoot, accountID)
	if err := os.MkdirAll(home, 0o700); err != nil {
		return WorkerEntry{}, fmt.Errorf("supervisor: mkdir home: %w", err)
	}
	tokenPath := filepath.Join(home, "github_token")
	if err := os.WriteFile(tokenPath, []byte(githubToken), 0o600); err != nil {
		return WorkerEntry{}, fmt.Errorf("supervisor: write github_token: %w", err)
	}

	logPath := filepath.Join(home, "copilot-api.log")
	logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return WorkerEntry{}, fmt.Errorf("supervisor: open log: %w", err)
	}

	port, err := s.selectPort(preferredPort)
	if err != nil {
		_ = logFile.Close()
		return WorkerEntry{}, err
	}

	cmd := s.opts.CommandFactory(s.opts.ExePath, port, home)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true

	if err := cmd.Start(); err != nil {
		s.ports.Release(port)
		_ = logFile.Close()
		return WorkerEntry{}, fmt.Errorf("supervisor: start child: %w", err)
	}

	st := &workerState{
		entry: WorkerEntry{
			AccountID: accountID,
			Port:      port,
			Home:      home,
			PID:       cmd.Process.Pid,
			StartedAt: s.opts.Now(),
			Status:    WorkerStatusStarting,
			WorkerURL: WorkerURLFor(port),
		},
		cmd:     cmd,
		logFile: logFile,
		doneCh:  make(chan struct{}),
	}

	s.mu.Lock()
	s.workers[accountID] = st
	s.mu.Unlock()

	go s.reap(accountID, st)

	readyCtx, cancel := context.WithTimeout(ctx, s.opts.ReadyTimeout)
	defer cancel()
	if err := s.opts.ReadinessProbe(readyCtx, port); err != nil {
		log.Printf("supervisor: worker %s (port %d) failed ready check: %v", accountID, port, err)
		s.markStartTimeout(accountID, err)
		_ = s.Stop(context.Background(), accountID)
		if errors.Is(err, context.DeadlineExceeded) {
			return WorkerEntry{}, ErrReadyTimeout
		}
		return WorkerEntry{}, fmt.Errorf("supervisor: readiness probe: %w", err)
	}

	s.mu.Lock()
	if cur, ok := s.workers[accountID]; ok {
		cur.entry.Status = WorkerStatusReady
	}
	snap := s.snapshotLocked(accountID)
	s.mu.Unlock()
	log.Printf("supervisor: worker %s ready at %s (pid=%d)", accountID, snap.WorkerURL, snap.PID)
	if s.opts.PersistOnReady != nil {
		if err := s.opts.PersistOnReady(snap); err != nil {
			log.Printf("supervisor: persist-on-ready for %s: %v (worker stays live)", accountID, err)
		}
	}
	return snap, nil
}

func (s *WorkerSupervisor) selectPort(preferredPort int) (int, error) {
	if preferredPort > 0 && preferredPort >= s.ports.start && preferredPort <= s.ports.end && portBindable(preferredPort) {
		if err := s.ports.Reserve(preferredPort); err == nil {
			return preferredPort, nil
		}
	}
	return s.ports.Alloc()
}

func (s *WorkerSupervisor) markStartTimeout(accountID string, cause error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if st, ok := s.workers[accountID]; ok {
		st.entry.Status = WorkerStatusStartTimeout
		if cause != nil {
			st.entry.LastError = cause.Error()
		}
	}
}

func (s *WorkerSupervisor) snapshotLocked(accountID string) WorkerEntry {
	if st, ok := s.workers[accountID]; ok {
		return st.entry
	}
	return WorkerEntry{}
}

// Stop sends SIGTERM to the worker's process group and waits up to 5s for
// exit; if the child is still alive it escalates to SIGKILL. The port is
// released and the account is removed from the supervisor's map on return.
// Stop is idempotent — calling on an unknown account returns ErrAccountUnknown.
func (s *WorkerSupervisor) Stop(ctx context.Context, accountID string) error {
	s.mu.Lock()
	st, ok := s.workers[accountID]
	if !ok {
		s.mu.Unlock()
		return ErrAccountUnknown
	}
	if st.entry.Status != WorkerStatusStopping {
		st.entry.Status = WorkerStatusStopping
	}
	doneCh := st.doneCh
	pid := st.entry.PID
	port := st.entry.Port
	adopted := st.adopted || st.cmd == nil
	s.mu.Unlock()

	// SIGTERM the process group (Setpgid made pgid == pid).
	if pid > 0 {
		_ = syscall.Kill(-pid, syscall.SIGTERM)
	}

	if adopted {
		waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := waitForProcessExit(waitCtx, pid); err != nil {
			log.Printf("supervisor: adopted worker %s (pid=%d) did not exit on SIGTERM; sending SIGKILL", accountID, pid)
			if pid > 0 {
				_ = syscall.Kill(-pid, syscall.SIGKILL)
			}
			if err := waitForProcessExit(ctx, pid); err != nil {
				return err
			}
		}
	} else {
		select {
		case <-doneCh:
		case <-time.After(5 * time.Second):
			log.Printf("supervisor: worker %s (pid=%d) did not exit on SIGTERM; sending SIGKILL", accountID, pid)
			if pid > 0 {
				_ = syscall.Kill(-pid, syscall.SIGKILL)
			}
			select {
			case <-doneCh:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	s.mu.Lock()
	if cur, ok := s.workers[accountID]; ok {
		if cur.logFile != nil {
			_ = cur.logFile.Close()
		}
		delete(s.workers, accountID)
	}
	s.mu.Unlock()
	s.ports.Release(port)
	if s.opts.PersistOnStop != nil {
		if err := s.opts.PersistOnStop(accountID); err != nil {
			log.Printf("supervisor: persist-on-stop for %s: %v", accountID, err)
		}
	}
	return nil
}

// Restart stops the worker then spawns it again with the same github_token
// (read back from disk). Convenience for admin console.
func (s *WorkerSupervisor) Restart(ctx context.Context, accountID, githubToken string) (WorkerEntry, error) {
	if err := s.Stop(ctx, accountID); err != nil && !errors.Is(err, ErrAccountUnknown) {
		return WorkerEntry{}, err
	}
	return s.Spawn(ctx, accountID, githubToken)
}

// RemoveAndCleanup stops the worker and deletes the account's api-home dir.
// If the supervisor has no record of the account, it still attempts to remove
// the dir so callers can clean up after a crash.
func (s *WorkerSupervisor) RemoveAndCleanup(ctx context.Context, accountID string) error {
	_ = s.Stop(ctx, accountID)
	home := filepath.Join(s.opts.WorkersRoot, accountID)
	if err := os.RemoveAll(home); err != nil {
		return fmt.Errorf("supervisor: remove home: %w", err)
	}
	return nil
}

// Status returns a snapshot of the worker's state. ok=false if unknown.
func (s *WorkerSupervisor) Status(accountID string) (WorkerEntry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st, ok := s.workers[accountID]
	if !ok {
		return WorkerEntry{}, false
	}
	return st.entry, true
}

// All returns snapshots of every registered worker. The slice is a copy.
func (s *WorkerSupervisor) All() []WorkerEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]WorkerEntry, 0, len(s.workers))
	for _, st := range s.workers {
		out = append(out, st.entry)
	}
	return out
}

// Shutdown stops every worker and marks the supervisor closed. Subsequent
// Spawn calls return ErrSupervisorClosed.
func (s *WorkerSupervisor) Shutdown(ctx context.Context) {
	s.mu.Lock()
	s.closed = true
	ids := make([]string, 0, len(s.workers))
	for id := range s.workers {
		ids = append(ids, id)
	}
	s.mu.Unlock()
	for _, id := range ids {
		_ = s.Stop(ctx, id)
	}
}

// RecoverFromStore rehydrates managed workers after a gateway restart. If a
// persisted worker PID is still alive and responds on the stored port, the
// supervisor adopts it; otherwise it spawns a fresh worker, preferring the
// previously assigned port when that port is still available.
func (s *WorkerSupervisor) RecoverFromStore(ctx context.Context) {
	accounts, err := store.GetEnabledAccounts()
	if err != nil {
		log.Printf("supervisor: recover load accounts: %v", err)
		return
	}
	for _, account := range accounts {
		if !account.WorkerManaged {
			continue
		}
		token := strings.TrimSpace(account.GithubToken)
		if token == "" {
			log.Printf("supervisor: recover skip account=%s missing github token", account.ID)
			continue
		}
		adopted, adoptErr := s.tryAdopt(ctx, account)
		if adoptErr != nil {
			log.Printf("supervisor: recover adopt account=%s: %v", account.ID, adoptErr)
		}
		if adopted {
			log.Printf("supervisor: recovered existing worker account=%s port=%d pid=%d", account.ID, account.WorkerPort, account.WorkerPID)
			continue
		}
		if _, err := s.spawn(ctx, account.ID, token, account.WorkerPort); err != nil {
			log.Printf("supervisor: recover spawn account=%s: %v", account.ID, err)
		}
	}
}

// MigrateLegacyAccounts promotes enabled legacy accounts (`WorkerManaged=false`)
// onto supervisor-owned workers. This is an opt-in startup migration for
// environments that previously relied on direct mode or externally managed
// workerUrl values. Existing manual worker processes are not touched here; the
// account's persisted WorkerURL simply flips to the new supervised worker once
// the readiness probe passes.
func (s *WorkerSupervisor) MigrateLegacyAccounts(ctx context.Context) {
	accounts, err := store.GetEnabledAccounts()
	if err != nil {
		log.Printf("supervisor: migrate legacy load accounts: %v", err)
		return
	}
	for _, account := range accounts {
		if account.WorkerManaged {
			continue
		}
		token := strings.TrimSpace(account.GithubToken)
		if token == "" {
			log.Printf("supervisor: migrate legacy skip account=%s missing github token", account.ID)
			continue
		}
		legacyWorkerURL := strings.TrimSpace(account.WorkerURL)
		if legacyWorkerURL != "" {
			log.Printf("supervisor: migrate legacy account=%s replacing manual worker_url=%s", account.ID, legacyWorkerURL)
		} else {
			log.Printf("supervisor: migrate legacy account=%s from direct mode", account.ID)
		}
		if _, err := s.spawn(ctx, account.ID, token, 0); err != nil {
			if errors.Is(err, ErrAccountAlreadyLive) {
				continue
			}
			log.Printf("supervisor: migrate legacy account=%s: %v", account.ID, err)
		}
	}
}

func (s *WorkerSupervisor) tryAdopt(ctx context.Context, account store.Account) (bool, error) {
	if account.WorkerPort <= 0 || account.WorkerPID <= 0 {
		return false, nil
	}
	if !processAlive(account.WorkerPID) {
		return false, nil
	}
	readyCtx, cancel := context.WithTimeout(ctx, minDuration(s.opts.ReadyTimeout, 5*time.Second))
	defer cancel()
	if err := s.opts.ReadinessProbe(readyCtx, account.WorkerPort); err != nil {
		return false, err
	}
	if err := s.ports.Reserve(account.WorkerPort); err != nil {
		return false, err
	}

	home := strings.TrimSpace(account.WorkerHome)
	if home == "" {
		home = store.WorkerHomeFor(account.ID)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		s.ports.Release(account.WorkerPort)
		return false, ErrSupervisorClosed
	}
	if _, exists := s.workers[account.ID]; exists {
		s.ports.Release(account.WorkerPort)
		return false, ErrAccountAlreadyLive
	}
	s.workers[account.ID] = &workerState{
		entry: WorkerEntry{
			AccountID: account.ID,
			Port:      account.WorkerPort,
			Home:      home,
			PID:       account.WorkerPID,
			StartedAt: s.opts.Now(),
			Status:    WorkerStatusReady,
			WorkerURL: WorkerURLFor(account.WorkerPort),
		},
		adopted: true,
	}
	return true, nil
}

// HealthLoop periodically checks worker liveness and restarts managed workers
// whose child process died or stopped answering /models.
func (s *WorkerSupervisor) HealthLoop(ctx context.Context) {
	ticker := time.NewTicker(s.opts.HealthInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.runHealthPass(ctx)
		}
	}
}

func (s *WorkerSupervisor) runHealthPass(ctx context.Context) {
	for _, entry := range s.All() {
		account, err := store.GetAccount(entry.AccountID)
		if err != nil {
			log.Printf("supervisor: health load account=%s: %v", entry.AccountID, err)
			continue
		}
		if account == nil || !account.Enabled || !account.WorkerManaged {
			_ = s.Stop(context.Background(), entry.AccountID)
			continue
		}
		token := strings.TrimSpace(account.GithubToken)
		if token == "" {
			continue
		}
		if entry.Status == WorkerStatusStarting || entry.Status == WorkerStatusStopping {
			continue
		}
		needsRestart := entry.Status == WorkerStatusCrashed || entry.Status == WorkerStatusStartTimeout || !processAlive(entry.PID)
		if !needsRestart {
			probeCtx, cancel := context.WithTimeout(ctx, minDuration(s.opts.HealthInterval/2, 5*time.Second))
			err = s.opts.ReadinessProbe(probeCtx, entry.Port)
			cancel()
			if err != nil {
				needsRestart = true
				s.markCrashed(entry.AccountID, fmt.Errorf("health probe failed: %w", err))
			}
		}
		if !needsRestart {
			continue
		}
		if _, err := s.Restart(context.Background(), entry.AccountID, token); err != nil {
			log.Printf("supervisor: health restart account=%s: %v", entry.AccountID, err)
		}
	}
}

func (s *WorkerSupervisor) reap(accountID string, st *workerState) {
	waitErr := st.cmd.Wait()
	s.markCrashed(accountID, waitErr)
	close(st.doneCh)
}

func (s *WorkerSupervisor) markCrashed(accountID string, cause error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if st, ok := s.workers[accountID]; ok {
		if st.entry.Status == WorkerStatusStopping {
			return
		}
		st.entry.Status = WorkerStatusCrashed
		if cause != nil {
			st.entry.LastError = cause.Error()
		}
	}
}

// httpModelsReady polls GET /models on the worker's loopback port until it
// returns a 2xx or ctx is done.
func httpModelsReady(ctx context.Context, port int) error {
	url := fmt.Sprintf("http://127.0.0.1:%d/models", port)
	client := &http.Client{Timeout: 2 * time.Second}
	backoff := 200 * time.Millisecond
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		if backoff < 1500*time.Millisecond {
			backoff += 200 * time.Millisecond
		}
	}
}

// ─── Port pool ─────────────────────────────────────────────────────────────

type portPool struct {
	mu        sync.Mutex
	start     int
	end       int
	allocated map[int]bool
}

func newPortPool(start, end int) (*portPool, error) {
	if start <= 0 || end < start {
		return nil, ErrInvalidPortRange
	}
	return &portPool{
		start:     start,
		end:       end,
		allocated: make(map[int]bool),
	}, nil
}

// Alloc returns the first free port in the range whose TCP bind-probe
// succeeds. On systems where all ports are occupied or reserved, returns
// ErrPortExhausted.
func (p *portPool) Alloc() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for port := p.start; port <= p.end; port++ {
		if p.allocated[port] {
			continue
		}
		if !portBindable(port) {
			continue
		}
		p.allocated[port] = true
		return port, nil
	}
	return 0, ErrPortExhausted
}

// Reserve marks a specific port as allocated without a bind check. Used by
// RecoverFromStore to restore prior assignments before Spawn.
func (p *portPool) Reserve(port int) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if port < p.start || port > p.end {
		return ErrInvalidPortRange
	}
	if p.allocated[port] {
		return ErrPortReserved
	}
	p.allocated[port] = true
	return nil
}

// Release returns a port to the pool. Ports outside the range are ignored.
func (p *portPool) Release(port int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.allocated, port)
}

// portBindable returns true if we can bind TCP on 127.0.0.1:port right now.
// Immediately closes the listener; a racing process may bind before the
// child actually starts, but this removes the obvious collisions.
func portBindable(port int) bool {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	_ = l.Close()
	return true
}
