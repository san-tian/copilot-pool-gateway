package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"copilot-go/config"
	"copilot-go/handler"
	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

func getEnv(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func parseDurationEnv(key string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		log.Printf("invalid %s=%q: %v; using default %s", key, v, err, fallback)
		return fallback
	}
	return d
}

func parsePortRangeEnv(key string, fallbackStart, fallbackEnd int) (int, int) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallbackStart, fallbackEnd
	}
	parts := strings.SplitN(v, "-", 2)
	if len(parts) != 2 {
		log.Printf("invalid %s=%q: expected START-END; using default %d-%d", key, v, fallbackStart, fallbackEnd)
		return fallbackStart, fallbackEnd
	}
	start := strings.TrimSpace(parts[0])
	end := strings.TrimSpace(parts[1])
	var startPort, endPort int
	if _, err := fmt.Sscanf(start, "%d", &startPort); err != nil {
		log.Printf("invalid %s=%q: %v; using default %d-%d", key, v, err, fallbackStart, fallbackEnd)
		return fallbackStart, fallbackEnd
	}
	if _, err := fmt.Sscanf(end, "%d", &endPort); err != nil {
		log.Printf("invalid %s=%q: %v; using default %d-%d", key, v, err, fallbackStart, fallbackEnd)
		return fallbackStart, fallbackEnd
	}
	if startPort <= 0 || endPort < startPort {
		log.Printf("invalid %s=%q: bad range; using default %d-%d", key, v, fallbackStart, fallbackEnd)
		return fallbackStart, fallbackEnd
	}
	return startPort, endPort
}

func buildWorkerCommandFactory(extraArgs []string) func(string, int, string) *exec.Cmd {
	return func(exePath string, port int, home string) *exec.Cmd {
		args := append([]string{}, extraArgs...)
		args = append(args, "start", "--port", fmt.Sprintf("%d", port))
		cmd := exec.Command(exePath, args...)
		cmd.Env = append(os.Environ(), "COPILOT_API_HOME="+home)
		return cmd
	}
}

func initWorkerSupervisor() *instance.WorkerSupervisor {
	workerExe := strings.TrimSpace(getEnv("COPILOT_WORKER_EXE", "copilot-api"))
	workerPath, err := exec.LookPath(workerExe)
	if err != nil {
		log.Printf("WorkerSupervisor disabled: cannot find %q in PATH; set COPILOT_WORKER_EXE (and optional COPILOT_WORKER_ARGS) to enable managed copilot-api workers", workerExe)
		return nil
	}

	startPort, endPort := parsePortRangeEnv("COPILOT_WORKER_PORT_RANGE", 9100, 9199)
	readyTimeout := parseDurationEnv("COPILOT_WORKER_READY_TIMEOUT", 20*time.Second)
	healthInterval := parseDurationEnv("COPILOT_WORKER_HEALTH_INTERVAL", 30*time.Second)
	workerArgs := strings.Fields(strings.TrimSpace(os.Getenv("COPILOT_WORKER_ARGS")))

	sup, err := instance.NewSupervisor(instance.SupervisorOptions{
		ExePath:        workerPath,
		PortRangeStart: startPort,
		PortRangeEnd:   endPort,
		WorkersRoot:    store.WorkersRoot(),
		ReadyTimeout:   readyTimeout,
		HealthInterval: healthInterval,
		CommandFactory: buildWorkerCommandFactory(workerArgs),
		PersistOnReady: func(entry instance.WorkerEntry) error {
			return store.UpdateAccountWorker(entry.AccountID, entry.WorkerURL, entry.Home, entry.Port, entry.PID)
		},
		PersistOnStop: store.ClearAccountWorker,
	})
	if err != nil {
		log.Printf("WorkerSupervisor disabled: %v", err)
		return nil
	}
	log.Printf("WorkerSupervisor enabled: exe=%s args=%q workers_root=%s ports=%d-%d", workerPath, workerArgs, store.WorkersRoot(), startPort, endPort)
	return sup
}

// normalizeAndValidatePublicPath enforces that user-configured paths for the
// proxy-port SPA aliases (login / supplier-auth) are safe to mount: leading
// slash, not root, no collision with proxy routes or the public API surface.
// Returns the canonical (trailing-slash-stripped) form.
func normalizeAndValidatePublicPath(name, path string) (string, error) {
	p := strings.TrimSpace(path)
	if p == "" {
		return "", fmt.Errorf("%s is empty", name)
	}
	if !strings.HasPrefix(p, "/") {
		return "", fmt.Errorf("%s=%q must start with '/'", name, path)
	}
	if len(p) > 1 {
		p = strings.TrimRight(p, "/")
	}
	if p == "/" {
		return "", fmt.Errorf("%s=%q cannot be root '/'", name, path)
	}
	reserved := []string{
		"/v1",
		"/api",
		"/assets",
		"/reauth",
		"/responses/compact",
		"/chat/completions",
		"/models",
		"/embeddings",
	}
	for _, r := range reserved {
		if p == r || strings.HasPrefix(p, r+"/") {
			return "", fmt.Errorf("%s=%q collides with reserved prefix %q", name, path, r)
		}
	}
	return p, nil
}

func main() {
	webPort := flag.Int("web-port", 3000, "Web console port")
	proxyPort := flag.Int("proxy-port", 4141, "Proxy server port")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	autoStart := flag.Bool("auto-start", true, "Auto-start enabled accounts")
	proxyLoginPath := flag.String(
		"proxy-login-path",
		getEnv("PROXY_LOGIN_PATH", "/login"),
		"URL path that exposes the login page on the proxy port (env PROXY_LOGIN_PATH)",
	)
	proxySupplierAuthPath := flag.String(
		"proxy-supplier-auth-path",
		getEnv("PROXY_SUPPLIER_AUTH_PATH", "/supplier-auth"),
		"URL path that exposes the supplier-auth page on the proxy port (env PROXY_SUPPLIER_AUTH_PATH)",
	)
	flag.Parse()

	loginPath, err := normalizeAndValidatePublicPath("proxy-login-path", *proxyLoginPath)
	if err != nil {
		log.Fatalf("invalid --proxy-login-path: %v", err)
	}
	supplierAuthPath, err := normalizeAndValidatePublicPath("proxy-supplier-auth-path", *proxySupplierAuthPath)
	if err != nil {
		log.Fatalf("invalid --proxy-supplier-auth-path: %v", err)
	}
	if loginPath == supplierAuthPath {
		log.Fatalf("--proxy-login-path and --proxy-supplier-auth-path must differ (both resolve to %q)", loginPath)
	}

	if !*verbose {
		gin.SetMode(gin.ReleaseMode)
	}

	ctx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	// Ensure data directories exist
	if err := store.EnsurePaths(); err != nil {
		log.Fatalf("Failed to initialize data paths: %v", err)
	}

	// Load proxy config and apply to HTTP clients
	if proxyCfg, err := store.GetProxyConfig(); err == nil && proxyCfg.ProxyURL != "" {
		config.SetProxyURL(proxyCfg.ProxyURL)
		instance.RebuildHTTPClients()
		log.Printf("Using HTTP proxy: %s", proxyCfg.ProxyURL)
	}

	if sup := initWorkerSupervisor(); sup != nil {
		instance.SetDefaultSupervisor(sup)
		sup.RecoverFromStore(ctx)
		go sup.HealthLoop(ctx)
	}

	// Auto-start enabled accounts
	if *autoStart {
		accounts, err := store.GetEnabledAccounts()
		if err != nil {
			log.Printf("Warning: failed to load accounts: %v", err)
		} else {
			for _, account := range accounts {
				go func(a store.Account) {
					if err := instance.StartInstance(a); err != nil {
						log.Printf("Failed to auto-start account %s: %v", a.Name, err)
					}
				}(account)
			}
		}
	}

	webEngine := gin.New()
	if *verbose {
		webEngine.Use(gin.Logger())
	}
	webEngine.Use(gin.Recovery())
	handler.RegisterConsoleAPI(webEngine, *proxyPort)

	proxyEngine := gin.New()
	proxyEngine.Use(gin.Logger())
	proxyEngine.Use(gin.Recovery())
	handler.RegisterProxyPublic(proxyEngine, *proxyPort, loginPath, supplierAuthPath)
	handler.RegisterProxy(proxyEngine)

	webServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *webPort),
		Handler: webEngine,
	}
	proxyServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *proxyPort),
		Handler: proxyEngine,
	}

	serverErrs := make(chan error, 2)
	go func() {
		log.Printf("Web Console listening on :%d", *webPort)
		if err := webServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErrs <- fmt.Errorf("web console failed: %w", err)
		}
	}()
	go func() {
		log.Printf("Proxy listening on :%d (login=%s, supplier-auth=%s)", *proxyPort, loginPath, supplierAuthPath)
		if err := proxyServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErrs <- fmt.Errorf("proxy failed: %w", err)
		}
	}()

	var serverErr error
	select {
	case <-ctx.Done():
	case serverErr = <-serverErrs:
		stopSignals()
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := webServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("Web Console shutdown error: %v", err)
	}
	if err := proxyServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("Proxy shutdown error: %v", err)
	}
	if sup := instance.DefaultSupervisor(); sup != nil {
		sup.Shutdown(shutdownCtx)
		instance.SetDefaultSupervisor(nil)
	}
	if serverErr != nil {
		log.Fatalf("%v", serverErr)
	}
}
