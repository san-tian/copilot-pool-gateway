package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

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

	var wg sync.WaitGroup
	wg.Add(2)

	// Start Web Console
	go func() {
		defer wg.Done()
		webEngine := gin.New()
		if *verbose {
			webEngine.Use(gin.Logger())
		}
		webEngine.Use(gin.Recovery())

		handler.RegisterConsoleAPI(webEngine, *proxyPort)

		log.Printf("Web Console listening on :%d", *webPort)
		if err := webEngine.Run(fmt.Sprintf(":%d", *webPort)); err != nil {
			log.Fatalf("Web Console failed: %v", err)
		}
	}()

	// Start Proxy
	go func() {
		defer wg.Done()
		proxyEngine := gin.New()
		proxyEngine.Use(gin.Logger())
		proxyEngine.Use(gin.Recovery())

		handler.RegisterProxyPublic(proxyEngine, *proxyPort, loginPath, supplierAuthPath)
		handler.RegisterProxy(proxyEngine)

		log.Printf("Proxy listening on :%d (login=%s, supplier-auth=%s)", *proxyPort, loginPath, supplierAuthPath)
		if err := proxyEngine.Run(fmt.Sprintf(":%d", *proxyPort)); err != nil {
			log.Fatalf("Proxy failed: %v", err)
		}
	}()

	wg.Wait()
}
