package store

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type Account struct {
	ID                string   `json:"id"`
	Name              string   `json:"name"`
	GithubToken       string   `json:"githubToken"`
	GithubLogin       string   `json:"githubLogin,omitempty"`
	GithubUserID      int64    `json:"githubUserId,omitempty"`
	BlockedModels     []string `json:"blockedModels,omitempty"`
	SupportedModels   []string `json:"supportedModels,omitempty"`
	UnsupportedModels []string `json:"unsupportedModels,omitempty"`
	AccountType       string   `json:"accountType"`
	ApiKey            string   `json:"apiKey"`
	Enabled           bool     `json:"enabled"`
	CreatedAt         string   `json:"createdAt"`
	Priority          int      `json:"priority"`
	ProbeStatus       string   `json:"probeStatus,omitempty"`
	ProbeCheckedAt    string   `json:"probeCheckedAt,omitempty"`
	ProbeError        string   `json:"probeError,omitempty"`
}

type PoolConfig struct {
	Enabled      bool   `json:"enabled"`
	Strategy     string `json:"strategy"`
	ApiKey       string `json:"apiKey"`
	RateLimitRPM int    `json:"rateLimitRPM,omitempty"` // Per-account rate limit (requests per minute), 0 = no limit
}

type accountStore struct {
	Accounts []Account `json:"accounts"`
}

var (
	accountMu sync.RWMutex
	poolMu    sync.RWMutex
)

func readAccounts() ([]Account, error) {
	data, err := os.ReadFile(AccountsFile())
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || string(data) == "{}" {
		return []Account{}, nil
	}
	var s accountStore
	if err := json.Unmarshal(data, &s); err != nil {
		var accounts []Account
		if err2 := json.Unmarshal(data, &accounts); err2 != nil {
			return []Account{}, nil
		}
		return accounts, nil
	}
	return s.Accounts, nil
}

func writeAccounts(accounts []Account) error {
	s := accountStore{Accounts: accounts}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(AccountsFile(), data, 0644)
}

func GetAccounts() ([]Account, error) {
	accountMu.RLock()
	defer accountMu.RUnlock()
	return readAccounts()
}

func GetAccount(id string) (*Account, error) {
	accounts, err := GetAccounts()
	if err != nil {
		return nil, err
	}
	for _, a := range accounts {
		if a.ID == id {
			return &a, nil
		}
	}
	return nil, nil
}

func GetAccountByApiKey(apiKey string) (*Account, error) {
	accounts, err := GetAccounts()
	if err != nil {
		return nil, err
	}
	for _, a := range accounts {
		if a.ApiKey == apiKey {
			return &a, nil
		}
	}
	return nil, nil
}

func GetEnabledAccounts() ([]Account, error) {
	accounts, err := GetAccounts()
	if err != nil {
		return nil, err
	}
	var enabled []Account
	for _, a := range accounts {
		if a.Enabled {
			enabled = append(enabled, a)
		}
	}
	return enabled, nil
}

func normalizeModelKey(model string) string {
	return strings.TrimSpace(strings.ToLower(model))
}

func hasBlockedModel(account Account, model string) bool {
	normalized := normalizeModelKey(model)
	if normalized == "" {
		return false
	}
	for _, blocked := range account.BlockedModels {
		if normalizeModelKey(blocked) == normalized {
			return true
		}
	}
	for _, blocked := range account.UnsupportedModels {
		if normalizeModelKey(blocked) == normalized {
			return true
		}
	}
	return false
}

func stringSliceFromValue(value interface{}) []string {
	switch v := value.(type) {
	case []string:
		return append([]string(nil), v...)
	case []interface{}:
		result := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				result = append(result, s)
			}
		}
		return result
	default:
		return nil
	}
}

func IsModelBlocked(accountID, model string) bool {
	accountMu.RLock()
	defer accountMu.RUnlock()

	accounts, err := readAccounts()
	if err != nil {
		return false
	}
	for _, account := range accounts {
		if account.ID == accountID {
			return hasBlockedModel(account, model)
		}
	}
	return false
}

func GetBlockedAccountIDs(model string) map[string]bool {
	blocked := map[string]bool{}
	normalized := normalizeModelKey(model)
	if normalized == "" {
		return blocked
	}

	accountMu.RLock()
	defer accountMu.RUnlock()

	accounts, err := readAccounts()
	if err != nil {
		return blocked
	}
	for _, account := range accounts {
		if hasBlockedModel(account, normalized) {
			blocked[account.ID] = true
		}
	}
	return blocked
}

func BlockModelForAccount(accountID, model string) error {
	normalized := normalizeModelKey(model)
	if normalized == "" {
		return nil
	}

	accountMu.Lock()
	defer accountMu.Unlock()

	accounts, err := readAccounts()
	if err != nil {
		return err
	}
	for i, account := range accounts {
		if account.ID != accountID {
			continue
		}
		if hasBlockedModel(account, normalized) {
			return nil
		}
		accounts[i].BlockedModels = append(accounts[i].BlockedModels, normalized)
		return writeAccounts(accounts)
	}
	return nil
}

func AddAccount(name, githubToken, accountType string) (*Account, error) {
	account, _, err := UpsertAccount("", name, githubToken, accountType, "", 0)
	return account, err
}

func matchesGithubIdentity(account Account, githubLogin string, githubUserID int64) bool {
	if githubUserID != 0 && account.GithubUserID != 0 && account.GithubUserID == githubUserID {
		return true
	}
	if githubLogin != "" && account.GithubLogin != "" && strings.EqualFold(account.GithubLogin, githubLogin) {
		return true
	}
	return false
}

func UpsertAccount(existingID, name, githubToken, accountType, githubLogin string, githubUserID int64) (*Account, bool, error) {
	accountMu.Lock()
	defer accountMu.Unlock()

	accounts, err := readAccounts()
	if err != nil {
		return nil, false, err
	}

	for i, a := range accounts {
		if (existingID != "" && a.ID == existingID) || (existingID == "" && matchesGithubIdentity(a, githubLogin, githubUserID)) {
			if name != "" {
				accounts[i].Name = name
			}
			accounts[i].GithubToken = githubToken
			accounts[i].BlockedModels = nil
			accounts[i].SupportedModels = nil
			accounts[i].UnsupportedModels = nil
			accounts[i].ProbeStatus = ""
			accounts[i].ProbeCheckedAt = ""
			accounts[i].ProbeError = ""
			if accountType != "" {
				accounts[i].AccountType = accountType
			}
			if githubLogin != "" {
				accounts[i].GithubLogin = githubLogin
			}
			if githubUserID != 0 {
				accounts[i].GithubUserID = githubUserID
			}
			if err := writeAccounts(accounts); err != nil {
				return nil, false, err
			}
			return &accounts[i], false, nil
		}
	}

	account := Account{
		ID:                uuid.New().String(),
		Name:              name,
		GithubToken:       githubToken,
		GithubLogin:       githubLogin,
		GithubUserID:      githubUserID,
		BlockedModels:     nil,
		SupportedModels:   nil,
		UnsupportedModels: nil,
		AccountType:       accountType,
		ApiKey:            "sk-" + uuid.New().String(),
		Enabled:           true,
		CreatedAt:         time.Now().UTC().Format(time.RFC3339),
		Priority:          0,
	}

	accounts = append(accounts, account)
	if err := writeAccounts(accounts); err != nil {
		return nil, false, err
	}
	return &account, true, nil
}

func UpdateAccount(id string, updates map[string]interface{}) (*Account, error) {
	accountMu.Lock()
	defer accountMu.Unlock()

	accounts, err := readAccounts()
	if err != nil {
		return nil, err
	}

	for i, a := range accounts {
		if a.ID == id {
			if v, ok := updates["name"].(string); ok {
				accounts[i].Name = v
			}
			if v, ok := updates["githubToken"].(string); ok {
				accounts[i].GithubToken = v
				accounts[i].BlockedModels = nil
			}
			if v, ok := updates["githubLogin"].(string); ok {
				accounts[i].GithubLogin = v
			}
			if v, ok := updates["githubUserId"]; ok {
				switch iv := v.(type) {
				case float64:
					accounts[i].GithubUserID = int64(iv)
				case int64:
					accounts[i].GithubUserID = iv
				case int:
					accounts[i].GithubUserID = int64(iv)
				}
			}
			if v, ok := updates["accountType"].(string); ok {
				accounts[i].AccountType = v
			}
			if v, ok := updates["enabled"].(bool); ok {
				accounts[i].Enabled = v
			}
			if v, ok := updates["probeStatus"].(string); ok {
				accounts[i].ProbeStatus = v
			}
			if v, ok := updates["probeCheckedAt"].(string); ok {
				accounts[i].ProbeCheckedAt = v
			}
			if v, ok := updates["probeError"].(string); ok {
				accounts[i].ProbeError = v
			}
			if v, ok := updates["blockedModels"]; ok {
				accounts[i].BlockedModels = stringSliceFromValue(v)
			}
			if v, ok := updates["supportedModels"]; ok {
				accounts[i].SupportedModels = stringSliceFromValue(v)
			}
			if v, ok := updates["unsupportedModels"]; ok {
				accounts[i].UnsupportedModels = stringSliceFromValue(v)
			}
			if v, ok := updates["priority"]; ok {
				switch pv := v.(type) {
				case float64:
					accounts[i].Priority = int(pv)
				case int:
					accounts[i].Priority = pv
				}
			}
			if err := writeAccounts(accounts); err != nil {
				return nil, err
			}
			return &accounts[i], nil
		}
	}
	return nil, nil
}

func DeleteAccount(id string) error {
	accountMu.Lock()
	defer accountMu.Unlock()

	accounts, err := readAccounts()
	if err != nil {
		return err
	}

	var filtered []Account
	for _, a := range accounts {
		if a.ID != id {
			filtered = append(filtered, a)
		}
	}
	return writeAccounts(filtered)
}

func RegenerateApiKey(id string) (string, error) {
	accountMu.Lock()
	defer accountMu.Unlock()

	accounts, err := readAccounts()
	if err != nil {
		return "", err
	}

	for i, a := range accounts {
		if a.ID == id {
			newKey := "sk-" + uuid.New().String()
			accounts[i].ApiKey = newKey
			if err := writeAccounts(accounts); err != nil {
				return "", err
			}
			return newKey, nil
		}
	}
	return "", nil
}

func GetPoolConfig() (*PoolConfig, error) {
	poolMu.RLock()
	defer poolMu.RUnlock()

	data, err := os.ReadFile(PoolConfigFile())
	if err != nil {
		return &PoolConfig{Strategy: "round-robin"}, nil
	}
	var cfg PoolConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return &PoolConfig{Strategy: "round-robin"}, nil
	}
	if cfg.Strategy == "" {
		cfg.Strategy = "round-robin"
	}
	return &cfg, nil
}

func UpdatePoolConfig(cfg *PoolConfig) error {
	poolMu.Lock()
	defer poolMu.Unlock()

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(PoolConfigFile(), data, 0644)
}

func RegeneratePoolApiKey() (string, error) {
	poolMu.Lock()
	defer poolMu.Unlock()

	data, err := os.ReadFile(PoolConfigFile())
	if err != nil {
		return "", err
	}
	var cfg PoolConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", err
	}
	cfg.ApiKey = "sk-pool-" + uuid.New().String()
	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(PoolConfigFile(), out, 0644); err != nil {
		return "", err
	}
	return cfg.ApiKey, nil
}
