package handler

import (
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"copilot-go/web"

	"github.com/gin-gonic/gin"
)

// RegisterProxyPublic mounts the surface needed to serve the login,
// supplier-auth, and (optionally) full admin console SPA on the proxy port:
//
//   - /api/*        — both the public API subset (config, setup/login, public
//                     auth and reauth flows) and the protected admin API
//                     (accounts, pool, model-map, etc.) so the console loaded
//                     from :38000 can drive everything with one session token.
//   - /assets/*     — Vite-built JS/CSS bundles (dev: filesystem; prod: embed).
//   - loginPath     — GET alias that serves index.html with a
//                     window.__CC_VIEW__ = "login" hint injected.
//   - supplierPath  — GET alias that serves index.html with a
//                     window.__CC_VIEW__ = "supplier-auth" hint injected.
//
// The hint lets App.tsx pick the right view even when the user configures a
// non-canonical path (e.g. /s-login instead of /supplier-auth).
//
// Must be called BEFORE RegisterProxy on the same engine so the engine-level
// ordering matters: the public GETs and /assets/* static won't be shadowed by
// the proxyAuth middleware which RegisterProxy scopes to its own sub-group.
func RegisterProxyPublic(r *gin.Engine, proxyPort int, loginPath, supplierAuthPath string) {
	webDist := findWebDist()

	var indexBytes []byte
	if webDist != "" {
		data, err := os.ReadFile(filepath.Join(webDist, "index.html"))
		if err != nil {
			log.Printf("Warning: proxy-port SPA mount failed to read dev index.html: %v", err)
		} else {
			indexBytes = data
		}
		r.Static("/assets", filepath.Join(webDist, "assets"))
	} else {
		distFS, err := fs.Sub(web.Dist, "dist")
		if err != nil {
			log.Printf("Warning: proxy-port SPA mount failed to access embedded dist: %v", err)
		} else {
			if data, err := fs.ReadFile(distFS, "index.html"); err != nil {
				log.Printf("Warning: proxy-port SPA mount failed to read embedded index.html: %v", err)
			} else {
				indexBytes = data
			}
			if assetsFS, err := fs.Sub(distFS, "assets"); err == nil {
				r.StaticFS("/assets", http.FS(assetsFS))
			}
		}
	}

	api := r.Group("/api")
	registerPublicConsoleAPI(api, proxyPort)

	// Mount the admin surface too so the SPA loaded from :38000 can drive the
	// full console with a single session token. adminAuthMiddleware enforces
	// the same admin-token / X-Admin-Token requirement as on :3000.
	protected := api.Group("")
	protected.Use(adminAuthMiddleware())
	registerProtectedConsoleAPI(protected, proxyPort)

	if len(indexBytes) == 0 {
		// No SPA bundle available — public alias routes would 500 anyway.
		return
	}

	loginHTML := injectViewHint(indexBytes, "login")
	supplierHTML := injectViewHint(indexBytes, "supplier-auth")

	if loginPath != "" {
		r.GET(loginPath, func(c *gin.Context) {
			c.Data(http.StatusOK, "text/html; charset=utf-8", loginHTML)
		})
	}
	if supplierAuthPath != "" {
		r.GET(supplierAuthPath, func(c *gin.Context) {
			c.Data(http.StatusOK, "text/html; charset=utf-8", supplierHTML)
		})
	}
}

// injectViewHint rewrites index.html to include a window.__CC_VIEW__ global
// right after <head>. Falls back to the original bytes if <head> isn't found.
func injectViewHint(indexBytes []byte, view string) []byte {
	const marker = "<head>"
	snippet := fmt.Sprintf("%s\n    <script>window.__CC_VIEW__ = %q;</script>", marker, view)
	rewritten := strings.Replace(string(indexBytes), marker, snippet, 1)
	if rewritten == string(indexBytes) {
		return indexBytes
	}
	return []byte(rewritten)
}
