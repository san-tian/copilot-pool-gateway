package instance

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/config"
	"copilot-go/copilot"
	"copilot-go/store"
)

var premiumProbeModels = []string{
	"gpt-5.4",
	"gpt-5-codex",
	"claude-opus-4.7",
	"claude-opus-4.6",
}

type AccountProbeResult struct {
	Success           bool     `json:"success"`
	AccountType       string   `json:"accountType,omitempty"`
	Error             string   `json:"error,omitempty"`
	CheckedAt         string   `json:"checkedAt"`
	SupportedModels   []string `json:"supportedModels,omitempty"`
	UnsupportedModels []string `json:"unsupportedModels,omitempty"`
}

func StartInstance(account store.Account) error {
	mu.Lock()
	if inst, ok := instances[account.ID]; ok {
		if inst.Status == "running" {
			mu.Unlock()
			return nil
		}
	}
	mu.Unlock()

	if canStartFromStoredProbe(account) {
		if err := startCachedAccount(account); err == nil {
			return nil
		} else {
			log.Printf("Cached start failed for %s, falling back to reprobe: %v", account.Name, err)
		}
	}

	_, probe, err := ReconcileAccount(account)
	if err != nil {
		return err
	}
	if probe != nil && !probe.Success {
		return fmt.Errorf("%s", probe.Error)
	}
	return nil
}

func ReprobeAccount(account store.Account) (*store.Account, *AccountProbeResult, error) {
	return ReconcileAccount(account)
}

func canStartFromStoredProbe(account store.Account) bool {
	return normalizeAccountType(account.AccountType) != "" && account.ProbeStatus == "passed" && len(account.SupportedModels) > 0
}

func startCachedAccount(account store.Account) error {
	state := newProbeState(account, account.AccountType)
	if err := refreshCopilotToken(state); err != nil {
		SetAccountError(account, err.Error())
		return err
	}
	if err := fetchModels(state); err != nil {
		SetAccountError(account, err.Error())
		return err
	}
	return startInstanceState(account, state)
}

func ReconcileAccount(account store.Account) (*store.Account, *AccountProbeResult, error) {
	probe, state := ProbeAccount(account, account.AccountType)
	updates := map[string]interface{}{
		"enabled":           probe.Success,
		"probeStatus":       probeStatus(probe.Success),
		"probeCheckedAt":    probe.CheckedAt,
		"probeError":        probe.Error,
		"supportedModels":   probe.SupportedModels,
		"unsupportedModels": probe.UnsupportedModels,
		"blockedModels":     mergeBlockedModels(account.BlockedModels, probe.UnsupportedModels),
	}
	if probe.Success && probe.AccountType != "" {
		updates["accountType"] = probe.AccountType
	}

	updated, err := store.UpdateAccount(account.ID, updates)
	if err != nil {
		return nil, probe, err
	}
	if updated == nil {
		updated = &account
		updated.Enabled = probe.Success
		updated.ProbeStatus = probeStatus(probe.Success)
		updated.ProbeCheckedAt = probe.CheckedAt
		updated.ProbeError = probe.Error
		updated.SupportedModels = append([]string(nil), probe.SupportedModels...)
		updated.UnsupportedModels = append([]string(nil), probe.UnsupportedModels...)
		updated.BlockedModels = mergeBlockedModels(account.BlockedModels, probe.UnsupportedModels)
		if probe.Success && probe.AccountType != "" {
			updated.AccountType = probe.AccountType
		}
	}

	if probe.Success {
		if err := startInstanceState(*updated, state); err != nil {
			return updated, probe, err
		}
		return updated, probe, nil
	}

	SetAccountError(*updated, probe.Error)
	return updated, probe, nil
}

func normalizeProbeModelKey(model string) string {
	return strings.TrimSpace(strings.ToLower(model))
}

func mergeBlockedModels(existing []string, unsupported []string) []string {
	premiumSet := make(map[string]bool, len(premiumProbeModels))
	for _, model := range premiumProbeModels {
		premiumSet[normalizeProbeModelKey(model)] = true
	}

	result := make([]string, 0, len(existing)+len(unsupported))
	seen := map[string]bool{}
	appendModel := func(model string) {
		normalized := normalizeProbeModelKey(model)
		if normalized == "" || seen[normalized] {
			return
		}
		seen[normalized] = true
		result = append(result, normalized)
	}

	for _, model := range existing {
		normalized := normalizeProbeModelKey(model)
		if premiumSet[normalized] {
			continue
		}
		appendModel(normalized)
	}
	for _, model := range unsupported {
		appendModel(model)
	}
	return result
}

func ProbeAccount(account store.Account, preferredType string) (*AccountProbeResult, *config.State) {
	checkedAt := time.Now().UTC().Format(time.RFC3339)
	candidates := orderedAccountTypes(preferredType)
	attempts := make([]string, 0, len(candidates))
	for _, accountType := range candidates {
		state := newProbeState(account, accountType)
		if err := refreshCopilotToken(state); err != nil {
			attempts = append(attempts, formatProbeAttemptError(accountType, "token", err))
			continue
		}
		if err := fetchModels(state); err != nil {
			attempts = append(attempts, formatProbeAttemptError(accountType, "models", err))
			continue
		}
		premiumResult, err := probePremiumModels(state)
		if err != nil {
			attempts = append(attempts, formatProbeAttemptError(accountType, "premium", err))
			continue
		}
		return &AccountProbeResult{
			Success:           true,
			AccountType:       accountType,
			CheckedAt:         checkedAt,
			SupportedModels:   premiumResult.Supported,
			UnsupportedModels: premiumResult.Unsupported,
		}, state
	}

	errMsg := strings.Join(attempts, "; ")
	if errMsg == "" {
		errMsg = "no account type probe candidates"
	}
	return &AccountProbeResult{
		Success:   false,
		Error:     errMsg,
		CheckedAt: checkedAt,
	}, nil
}

type premiumModelProbe struct {
	Supported   []string
	Unsupported []string
}

func probePremiumModels(state *config.State) (*premiumModelProbe, error) {
	state.RLock()
	models := state.Models
	state.RUnlock()

	if models == nil || len(models.Data) == 0 {
		return nil, fmt.Errorf("no models returned from upstream")
	}

	available := make(map[string]bool, len(models.Data))
	for _, model := range models.Data {
		available[model.ID] = true
	}

	result := &premiumModelProbe{
		Supported:   []string{},
		Unsupported: []string{},
	}
	failures := make([]string, 0, len(premiumProbeModels))
	for _, modelID := range premiumProbeModels {
		if !available[modelID] {
			result.Unsupported = append(result.Unsupported, modelID)
			failures = append(failures, fmt.Sprintf("%s => not_listed_by_upstream", modelID))
			continue
		}
		if err := probeChatCompletionsForModel(state, modelID); err != nil {
			result.Unsupported = append(result.Unsupported, modelID)
			failures = append(failures, fmt.Sprintf("%s => %s", modelID, classifyProbeError("smoke", err)))
			continue
		}
		result.Supported = append(result.Supported, modelID)
	}

	if len(result.Supported) == 0 {
		if len(failures) == 0 {
			return result, fmt.Errorf("no premium models passed")
		}
		return result, fmt.Errorf("no premium models passed: %s", strings.Join(failures, "; "))
	}
	return result, nil
}

func startInstanceState(account store.Account, state *config.State) error {
	if state == nil {
		return fmt.Errorf("missing probed state")
	}
	stopChan := make(chan struct{})
	inst := &ProxyInstance{
		Account:  account,
		State:    state,
		Status:   "running",
		stopChan: stopChan,
	}

	mu.Lock()
	if existing, ok := instances[account.ID]; ok {
		if existing.Status == "running" && existing.stopChan != nil {
			close(existing.stopChan)
			existing.stopChan = nil
		}
	}
	instances[account.ID] = inst
	mu.Unlock()

	go tokenRefreshLoop(inst)
	log.Printf("Instance started for account: %s", account.Name)
	return nil
}

func newProbeState(account store.Account, accountType string) *config.State {
	state := config.NewState()
	state.Lock()
	state.GithubToken = account.GithubToken
	state.AccountType = accountType
	state.VSCodeVersion = copilot.GetVSCodeVersion()
	state.Unlock()
	return state
}

func orderedAccountTypes(preferredType string) []string {
	base := []string{"individual", "business", "enterprise"}
	preferred := normalizeAccountType(preferredType)
	if preferred == "" {
		return base
	}
	ordered := []string{preferred}
	for _, accountType := range base {
		if accountType != preferred {
			ordered = append(ordered, accountType)
		}
	}
	return ordered
}

func normalizeAccountType(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "individual", "business", "enterprise":
		return strings.ToLower(strings.TrimSpace(value))
	default:
		return ""
	}
}

func probeStatus(success bool) string {
	if success {
		return "passed"
	}
	return "failed"
}

func formatProbeAttemptError(accountType, stage string, err error) string {
	return fmt.Sprintf("%s => %s", accountType, classifyProbeError(stage, err))
}

func classifyProbeError(stage string, err error) string {
	message := strings.TrimSpace(err.Error())
	lower := strings.ToLower(message)
	switch {
	case strings.Contains(lower, "chat not enabled for ide token"):
		return "chat_not_enabled: " + message
	case strings.Contains(lower, "quota_exceeded"):
		return "quota_exceeded: " + message
	case strings.Contains(lower, "model_not_supported"):
		return "model_not_supported: " + message
	case strings.Contains(lower, "no premium models passed"):
		return "premium_models_unavailable: " + message
	case strings.Contains(lower, "copilot token request failed") || strings.Contains(lower, "failed to get copilot token") || strings.Contains(lower, "failed to decode copilot token"):
		return "token_refresh_failed: " + message
	case strings.Contains(lower, "failed to parse models response"):
		return "invalid_models_response: " + message
	case strings.Contains(lower, "models request failed") && strings.Contains(lower, "status 401"):
		return "models_unauthorized: " + message
	case strings.Contains(lower, "status 5"):
		if stage == "smoke" || stage == "premium" {
			return "smoke_failed: " + message
		}
		return "upstream_5xx: " + message
	case stage == "smoke" || stage == "premium":
		return "smoke_failed: " + message
	default:
		return message
	}
}

func probeChatCompletionsForModel(state *config.State, modelID string) error {
	payloadBody := map[string]interface{}{
		"model": modelID,
		"messages": []map[string]string{{
			"role":    "user",
			"content": "hi",
		}},
		"stream": false,
	}
	if strings.HasPrefix(modelID, "gpt-5") {
		payloadBody["max_completion_tokens"] = 1
	} else {
		payloadBody["max_tokens"] = 1
	}

	payload, err := json.Marshal(payloadBody)
	if err != nil {
		return err
	}

	resp, err := ProxyRequestWithBytes(state, http.MethodPost, "/chat/completions", payload, nil, false)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("smoke request failed (status %d): %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}
