package instance

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"copilot-go/store"
)

// ─── Port pool ─────────────────────────────────────────────────────────────

func TestPortPool_AllocReleaseReserve(t *testing.T) {
	p, err := newPortPool(55100, 55103)
	if err != nil {
		t.Fatalf("newPortPool: %v", err)
	}

	p1, err := p.Alloc()
	if err != nil {
		t.Fatalf("Alloc #1: %v", err)
	}
	p2, err := p.Alloc()
	if err != nil {
		t.Fatalf("Alloc #2: %v", err)
	}
	if p1 == p2 {
		t.Fatalf("expected distinct ports, got %d twice", p1)
	}

	p.Release(p1)
	p3, err := p.Alloc()
	if err != nil {
		t.Fatalf("Alloc after release: %v", err)
	}
	if p3 != p1 {
		// Not required semantically, but p1 becoming free should allow it again
		// unless something external grabbed it first. Just assert reuse is possible.
		if _, err := p.Alloc(); err != nil && !errors.Is(err, ErrPortExhausted) {
			t.Fatalf("unexpected error on 3rd alloc: %v", err)
		}
	}
}

func TestPortPool_Reserve(t *testing.T) {
	p, _ := newPortPool(55200, 55202)
	if err := p.Reserve(55201); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if err := p.Reserve(55201); !errors.Is(err, ErrPortReserved) {
		t.Fatalf("expected ErrPortReserved, got %v", err)
	}
	if err := p.Reserve(55999); !errors.Is(err, ErrInvalidPortRange) {
		t.Fatalf("expected ErrInvalidPortRange for out-of-range, got %v", err)
	}
	p.Release(55201)
	if err := p.Reserve(55201); err != nil {
		t.Fatalf("Reserve after Release: %v", err)
	}
}

func TestPortPool_InvalidRange(t *testing.T) {
	if _, err := newPortPool(0, 100); !errors.Is(err, ErrInvalidPortRange) {
		t.Fatalf("expected ErrInvalidPortRange, got %v", err)
	}
	if _, err := newPortPool(100, 50); !errors.Is(err, ErrInvalidPortRange) {
		t.Fatalf("expected ErrInvalidPortRange, got %v", err)
	}
}

// ─── Supervisor happy path ─────────────────────────────────────────────────

// newTestSupervisor returns a supervisor whose child processes are `sleep 30`
// and whose readiness probe the test controls via the returned readyFn.
func newTestSupervisor(t *testing.T, ready func(ctx context.Context, port int) error) *WorkerSupervisor {
	t.Helper()
	root := t.TempDir()
	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55300,
		PortRangeEnd:   55309,
		WorkersRoot:    root,
		ReadyTimeout:   2 * time.Second,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: ready,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })
	return sup
}

func seedSupervisorAccountsEnv(t *testing.T) {
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

func TestSupervisor_SpawnHappyPath(t *testing.T) {
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error {
		return nil
	})

	entry, err := sup.Spawn(context.Background(), "acct-1", "gho_test_token")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}

	if entry.Status != WorkerStatusReady {
		t.Errorf("Status = %q, want ready", entry.Status)
	}
	if entry.Port < 55300 || entry.Port > 55309 {
		t.Errorf("Port = %d, out of range", entry.Port)
	}
	if entry.PID <= 0 {
		t.Errorf("PID = %d, want >0", entry.PID)
	}
	if entry.WorkerURL != fmt.Sprintf("http://127.0.0.1:%d", entry.Port) {
		t.Errorf("WorkerURL = %q", entry.WorkerURL)
	}

	// github_token written with 0o600 and the expected content.
	tokenPath := filepath.Join(entry.Home, "github_token")
	info, err := os.Stat(tokenPath)
	if err != nil {
		t.Fatalf("stat token: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("token perm = %o, want 600", info.Mode().Perm())
	}
	content, err := os.ReadFile(tokenPath)
	if err != nil {
		t.Fatalf("read token: %v", err)
	}
	if string(content) != "gho_test_token" {
		t.Errorf("token content = %q", string(content))
	}

	// Status lookup matches.
	got, ok := sup.Status("acct-1")
	if !ok {
		t.Fatalf("Status(acct-1) missing")
	}
	if got.Port != entry.Port {
		t.Errorf("Status port mismatch: %d vs %d", got.Port, entry.Port)
	}

	// All() has exactly one entry.
	all := sup.All()
	if len(all) != 1 || all[0].AccountID != "acct-1" {
		t.Errorf("All() = %+v", all)
	}
}

func TestSupervisor_SpawnRejectsDuplicate(t *testing.T) {
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error { return nil })

	if _, err := sup.Spawn(context.Background(), "acct-dup", "tok"); err != nil {
		t.Fatalf("first Spawn: %v", err)
	}
	_, err := sup.Spawn(context.Background(), "acct-dup", "tok")
	if !errors.Is(err, ErrAccountAlreadyLive) {
		t.Fatalf("second Spawn err = %v, want ErrAccountAlreadyLive", err)
	}
}

func TestSupervisor_Stop(t *testing.T) {
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error { return nil })

	entry, err := sup.Spawn(context.Background(), "acct-stop", "tok")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	port := entry.Port

	if err := sup.Stop(context.Background(), "acct-stop"); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	// Account is removed.
	if _, ok := sup.Status("acct-stop"); ok {
		t.Errorf("Status still returns account after Stop")
	}
	// Port is free (Reserve succeeds) — same port is now reusable.
	if err := sup.ports.Reserve(port); err != nil {
		t.Errorf("port %d not released: %v", port, err)
	}
	// Stop on unknown → ErrAccountUnknown.
	if err := sup.Stop(context.Background(), "acct-stop"); !errors.Is(err, ErrAccountUnknown) {
		t.Errorf("second Stop err = %v, want ErrAccountUnknown", err)
	}
}

func TestSupervisor_RemoveAndCleanup(t *testing.T) {
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error { return nil })

	entry, err := sup.Spawn(context.Background(), "acct-clean", "tok")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	if _, err := os.Stat(entry.Home); err != nil {
		t.Fatalf("home missing before cleanup: %v", err)
	}

	if err := sup.RemoveAndCleanup(context.Background(), "acct-clean"); err != nil {
		t.Fatalf("RemoveAndCleanup: %v", err)
	}
	if _, err := os.Stat(entry.Home); !os.IsNotExist(err) {
		t.Errorf("home still exists after cleanup: %v", err)
	}
}

func TestSupervisor_ReadyTimeout(t *testing.T) {
	// Probe blocks until context is cancelled; simulates a child that never
	// responds on /models. Expect Spawn to fail with ErrReadyTimeout, the
	// child to be killed, and the port to return to the pool.
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error {
		<-ctx.Done()
		return ctx.Err()
	})
	// Shorten timeout specifically for this test.
	sup.opts.ReadyTimeout = 150 * time.Millisecond

	_, err := sup.Spawn(context.Background(), "acct-slow", "tok")
	if !errors.Is(err, ErrReadyTimeout) {
		t.Fatalf("Spawn err = %v, want ErrReadyTimeout", err)
	}
	if _, ok := sup.Status("acct-slow"); ok {
		t.Errorf("Status returns account after failed Spawn")
	}
	// Pool is fully restored.
	if len(sup.All()) != 0 {
		t.Errorf("All() = %d entries after failed Spawn", len(sup.All()))
	}
}

// ─── Persistence callbacks ─────────────────────────────────────────────────

func TestSupervisor_PersistOnReadyFiresWithEntry(t *testing.T) {
	root := t.TempDir()
	var persisted []WorkerEntry
	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55400,
		PortRangeEnd:   55409,
		WorkersRoot:    root,
		ReadyTimeout:   2 * time.Second,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error { return nil },
		PersistOnReady: func(e WorkerEntry) error {
			persisted = append(persisted, e)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	entry, err := sup.Spawn(context.Background(), "acct-persist", "tok")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	if len(persisted) != 1 {
		t.Fatalf("PersistOnReady called %d times, want 1", len(persisted))
	}
	if persisted[0].AccountID != "acct-persist" || persisted[0].Port != entry.Port || persisted[0].PID != entry.PID {
		t.Errorf("PersistOnReady entry mismatch: %+v vs %+v", persisted[0], entry)
	}
	if persisted[0].Status != WorkerStatusReady {
		t.Errorf("PersistOnReady status = %q, want ready", persisted[0].Status)
	}
}

func TestSupervisor_PersistOnStopFiresOnStopAndRollback(t *testing.T) {
	root := t.TempDir()
	var cleared []string
	probeCh := make(chan error, 1)
	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55410,
		PortRangeEnd:   55419,
		WorkersRoot:    root,
		ReadyTimeout:   150 * time.Millisecond,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error {
			select {
			case err := <-probeCh:
				return err
			case <-ctx.Done():
				return ctx.Err()
			}
		},
		PersistOnReady: func(WorkerEntry) error { return nil },
		PersistOnStop: func(accountID string) error {
			cleared = append(cleared, accountID)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	// Happy path: probe returns nil → Spawn succeeds → explicit Stop fires PersistOnStop.
	probeCh <- nil
	if _, err := sup.Spawn(context.Background(), "acct-a", "tok"); err != nil {
		t.Fatalf("Spawn a: %v", err)
	}
	if err := sup.Stop(context.Background(), "acct-a"); err != nil {
		t.Fatalf("Stop a: %v", err)
	}

	// Rollback path: probe blocks until ctx deadline → Spawn fails → PersistOnStop fires.
	_, err = sup.Spawn(context.Background(), "acct-b", "tok")
	if !errors.Is(err, ErrReadyTimeout) {
		t.Fatalf("Spawn b err = %v, want ErrReadyTimeout", err)
	}

	if len(cleared) != 2 {
		t.Fatalf("PersistOnStop called %d times, want 2 (stop + rollback); got %v", len(cleared), cleared)
	}
	if cleared[0] != "acct-a" || cleared[1] != "acct-b" {
		t.Errorf("PersistOnStop order = %v, want [acct-a acct-b]", cleared)
	}
}

func TestSupervisor_PersistOnReadyErrorDoesNotFailSpawn(t *testing.T) {
	root := t.TempDir()
	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55420,
		PortRangeEnd:   55429,
		WorkersRoot:    root,
		ReadyTimeout:   2 * time.Second,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error { return nil },
		PersistOnReady: func(WorkerEntry) error { return errors.New("boom") },
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	entry, err := sup.Spawn(context.Background(), "acct-peristerr", "tok")
	if err != nil {
		t.Fatalf("Spawn should succeed despite persist error: %v", err)
	}
	if entry.Status != WorkerStatusReady {
		t.Errorf("Status = %q, want ready", entry.Status)
	}
}

func TestSupervisor_Shutdown(t *testing.T) {
	sup := newTestSupervisor(t, func(ctx context.Context, port int) error { return nil })

	for _, id := range []string{"a", "b", "c"} {
		if _, err := sup.Spawn(context.Background(), id, "tok"); err != nil {
			t.Fatalf("Spawn %s: %v", id, err)
		}
	}
	if len(sup.All()) != 3 {
		t.Fatalf("pre-shutdown count = %d, want 3", len(sup.All()))
	}

	sup.Shutdown(context.Background())

	if len(sup.All()) != 0 {
		t.Errorf("post-shutdown count = %d, want 0", len(sup.All()))
	}
	// Post-shutdown Spawn is rejected.
	if _, err := sup.Spawn(context.Background(), "d", "tok"); !errors.Is(err, ErrSupervisorClosed) {
		t.Errorf("Spawn after Shutdown err = %v, want ErrSupervisorClosed", err)
	}
}

func TestSupervisor_RecoverFromStoreRespawnsManagedAccountOnStoredPort(t *testing.T) {
	seedSupervisorAccountsEnv(t)

	acct, err := store.AddAccount("recover", "tok", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	home := store.WorkerHomeFor(acct.ID)
	if err := store.UpdateAccountWorker(acct.ID, WorkerURLFor(55302), home, 55302, 43210); err != nil {
		t.Fatalf("UpdateAccountWorker: %v", err)
	}

	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55300,
		PortRangeEnd:   55309,
		WorkersRoot:    store.WorkersRoot(),
		ReadyTimeout:   2 * time.Second,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error { return nil },
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	sup.RecoverFromStore(context.Background())

	got, ok := sup.Status(acct.ID)
	if !ok {
		t.Fatalf("expected recovered worker for %s", acct.ID)
	}
	if got.Port != 55302 {
		t.Fatalf("recovered port = %d, want 55302", got.Port)
	}
	if got.Status != WorkerStatusReady {
		t.Fatalf("recovered status = %q, want ready", got.Status)
	}
}

func TestSupervisor_HealthLoopRestartsCrashedWorker(t *testing.T) {
	seedSupervisorAccountsEnv(t)

	acct, err := store.AddAccount("health", "tok", "individual")
	if err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	sup, err := NewSupervisor(SupervisorOptions{
		PortRangeStart: 55320,
		PortRangeEnd:   55329,
		WorkersRoot:    store.WorkersRoot(),
		ReadyTimeout:   2 * time.Second,
		HealthInterval: 50 * time.Millisecond,
		CommandFactory: func(_ string, _ int, _ string) *exec.Cmd {
			return exec.Command("sleep", "30")
		},
		ReadinessProbe: func(ctx context.Context, port int) error { return nil },
		PersistOnReady: func(entry WorkerEntry) error {
			return store.UpdateAccountWorker(entry.AccountID, entry.WorkerURL, entry.Home, entry.Port, entry.PID)
		},
		PersistOnStop: store.ClearAccountWorker,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	t.Cleanup(func() { sup.Shutdown(context.Background()) })

	entry, err := sup.Spawn(context.Background(), acct.ID, "tok")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}

	healthCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sup.HealthLoop(healthCtx)

	if err := syscall.Kill(-entry.PID, syscall.SIGKILL); err != nil {
		t.Fatalf("kill worker: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		got, ok := sup.Status(acct.ID)
		if ok && got.Status == WorkerStatusReady && got.PID > 0 && got.PID != entry.PID {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	got, _ := sup.Status(acct.ID)
	t.Fatalf("worker was not restarted, last state: %+v", got)
}

// ─── Default readiness probe ───────────────────────────────────────────────

func TestHttpModelsReady_Accepts2xx(t *testing.T) {
	// Stand up a local HTTP server that returns 200 on /models.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer srv.Close()

	port := portFromTestServer(t, srv)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := httpModelsReady(ctx, port); err != nil {
		t.Fatalf("httpModelsReady: %v", err)
	}
}

func TestHttpModelsReady_TimesOutWhenRefused(t *testing.T) {
	// Grab a free port, bind & release so nothing listens on it.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	_ = l.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	err = httpModelsReady(ctx, port)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
}

func portFromTestServer(t *testing.T, srv *httptest.Server) int {
	t.Helper()
	// srv.URL is like http://127.0.0.1:PORT
	u := srv.URL
	idx := strings.LastIndex(u, ":")
	if idx < 0 {
		t.Fatalf("unexpected server URL %q", u)
	}
	p, err := strconv.Atoi(u[idx+1:])
	if err != nil {
		t.Fatalf("parse port from %q: %v", u, err)
	}
	return p
}
