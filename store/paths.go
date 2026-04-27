package store

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	AppDirEnvVar      = "COPILOT_API_APP_DIR"
	WorkersRootEnvVar = "COPILOT_WORKERS_HOME"
)

var AppDir = resolveAppDirFromEnv()

func defaultAppDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".local", "share", "copilot-api")
}

func resolveAppDirFromEnv() string {
	if override := strings.TrimSpace(os.Getenv(AppDirEnvVar)); override != "" {
		return override
	}
	return defaultAppDir()
}

func AccountsFile() string {
	return filepath.Join(AppDir, "accounts.json")
}

func PoolConfigFile() string {
	return filepath.Join(AppDir, "pool-config.json")
}

func AdminFile() string {
	return filepath.Join(AppDir, "admin.json")
}

func ModelMapFile() string {
	return filepath.Join(AppDir, "model_map.json")
}

func ProxyConfigFile() string {
	return filepath.Join(AppDir, "proxy-config.json")
}

func WorkersRoot() string {
	if override := strings.TrimSpace(os.Getenv(WorkersRootEnvVar)); override != "" {
		return override
	}
	return filepath.Join(AppDir, "workers")
}

func WorkerHomeFor(accountID string) string {
	return filepath.Join(WorkersRoot(), accountID)
}

func EnsurePaths() error {
	if err := os.MkdirAll(AppDir, 0755); err != nil {
		return err
	}
	files := []string{AccountsFile(), PoolConfigFile(), AdminFile(), ModelMapFile(), ProxyConfigFile(), PublicReauthFile()}
	for _, f := range files {
		if _, err := os.Stat(f); os.IsNotExist(err) {
			if err := os.WriteFile(f, []byte("{}"), 0644); err != nil {
				return err
			}
		}
	}
	return nil
}
