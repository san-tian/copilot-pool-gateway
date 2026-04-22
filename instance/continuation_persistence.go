package instance

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"copilot-go/store"
)

const durableContinuationStateVersion = 1

var continuationStateLoadOnce sync.Once
var continuationStateWriteMu sync.Mutex
var durableContinuationPersistenceDisabled bool

type durableCopilotTurnCacheEntry struct {
	AccountID  string             `json:"account_id"`
	Context    copilotTurnContext `json:"context"`
	CreatedAt  time.Time          `json:"created_at"`
	AccessedAt time.Time          `json:"accessed_at"`
}

type durableResponsesReplayEntry struct {
	AccountID   string        `json:"account_id"`
	Input       []interface{} `json:"input"`
	ReplayItems []interface{} `json:"replay_items"`
	CreatedAt   time.Time     `json:"created_at"`
	AccessedAt  time.Time     `json:"accessed_at"`
}

type durableContinuationState struct {
	Version                   int                                     `json:"version"`
	SavedAt                   time.Time                               `json:"saved_at"`
	ClientMachineID           string                                  `json:"client_machine_id,omitempty"`
	ResponseTurns             map[string]durableCopilotTurnCacheEntry `json:"response_turns,omitempty"`
	ResponseFunctionCallTurns map[string]durableCopilotTurnCacheEntry `json:"response_function_call_turns,omitempty"`
	MessageToolTurns          map[string]durableCopilotTurnCacheEntry `json:"message_tool_turns,omitempty"`
	ResponsesReplay           map[string]durableResponsesReplayEntry  `json:"responses_replay,omitempty"`
}

func continuationStateFile() string {
	return filepath.Join(store.AppDir, "continuation-state.json.gz")
}

func legacyContinuationStateFile() string {
	return filepath.Join(store.AppDir, "continuation-state.json")
}

func ensureDurableContinuationStateLoaded() {
	if durableContinuationPersistenceDisabled {
		return
	}
	continuationStateLoadOnce.Do(func() {
		if err := loadDurableContinuationState(); err != nil {
			log.Printf("Failed to load durable continuation state: %v", err)
		}
	})
}

func loadDurableContinuationState() error {
	body, err := readDurableContinuationStateFile()
	if err != nil {
		return err
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return nil
	}

	var snapshot durableContinuationState
	if err := json.Unmarshal(body, &snapshot); err != nil {
		return err
	}

	now := time.Now()
	if clientMachineID := strings.TrimSpace(snapshot.ClientMachineID); clientMachineID != "" {
		copilotClientMachineID = clientMachineID
	}
	restoreCopilotTurnCacheEntries(&responseTurnCache.mu, responseTurnCache.entries, snapshot.ResponseTurns, copilotTurnCacheTTL, now)
	restoreCopilotTurnCacheEntries(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, snapshot.ResponseFunctionCallTurns, copilotTurnCacheTTL, now)
	restoreCopilotTurnCacheEntries(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, snapshot.MessageToolTurns, copilotTurnCacheTTL, now)
	restoreResponsesReplayEntries(snapshot.ResponsesReplay, now)
	return nil
}

func readDurableContinuationStateFile() ([]byte, error) {
	body, err := readGzipFile(continuationStateFile())
	if err == nil {
		return body, nil
	}
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	legacyPath := legacyContinuationStateFile()
	body, err = os.ReadFile(legacyPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return body, nil
}

func readGzipFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()

	return io.ReadAll(reader)
}

func restoreCopilotTurnCacheEntries(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, snapshot map[string]durableCopilotTurnCacheEntry, ttl time.Duration, now time.Time) {
	if len(snapshot) == 0 {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	for key, entry := range snapshot {
		key = strings.TrimSpace(key)
		entry.AccountID = strings.TrimSpace(entry.AccountID)
		if key == "" || entry.AccountID == "" {
			continue
		}
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		if now.Sub(createdAt) > ttl {
			continue
		}
		accessedAt := entry.AccessedAt
		if accessedAt.IsZero() {
			accessedAt = createdAt
		}
		entries[key] = &copilotTurnCacheEntry{
			AccountID:  entry.AccountID,
			Context:    entry.Context,
			CreatedAt:  createdAt,
			AccessedAt: accessedAt,
		}
	}
	pruneCopilotTurnCacheLocked(entries, now)
}

func restoreResponsesReplayEntries(snapshot map[string]durableResponsesReplayEntry, now time.Time) {
	if len(snapshot) == 0 {
		return
	}
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()
	for responseID, entry := range snapshot {
		responseID = strings.TrimSpace(responseID)
		entry.AccountID = strings.TrimSpace(entry.AccountID)
		if responseID == "" || entry.AccountID == "" {
			continue
		}
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		if now.Sub(createdAt) > responsesReplayTTL {
			continue
		}
		accessedAt := entry.AccessedAt
		if accessedAt.IsZero() {
			accessedAt = createdAt
		}
		responsesReplayCache.entries[responseID] = &responsesReplayEntry{
			AccountID:   entry.AccountID,
			Input:       cloneJSONArray(entry.Input),
			ReplayItems: cloneJSONArray(entry.ReplayItems),
			CreatedAt:   createdAt,
			AccessedAt:  accessedAt,
		}
	}
	pruneResponsesReplayLocked(now)
}

func persistDurableContinuationState() {
	if durableContinuationPersistenceDisabled {
		return
	}
	ensureDurableContinuationStateLoaded()

	continuationStateWriteMu.Lock()
	defer continuationStateWriteMu.Unlock()

	snapshot := captureDurableContinuationState()
	body, err := json.Marshal(snapshot)
	if err != nil {
		log.Printf("Failed to marshal durable continuation state: %v", err)
		return
	}

	path := continuationStateFile()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		log.Printf("Failed to create durable continuation state dir: %v", err)
		return
	}
	tmpPath := path + ".tmp"
	if err := writeGzipFile(tmpPath, body); err != nil {
		log.Printf("Failed to write durable continuation state: %v", err)
		return
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		log.Printf("Failed to replace durable continuation state: %v", err)
		return
	}
	_ = os.Remove(legacyContinuationStateFile())
}

func writeGzipFile(path string, body []byte) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	gzipWriter := gzip.NewWriter(file)
	if _, err := gzipWriter.Write(body); err != nil {
		_ = gzipWriter.Close()
		_ = file.Close()
		_ = os.Remove(path)
		return err
	}
	if err := gzipWriter.Close(); err != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return err
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}

func captureDurableContinuationState() durableContinuationState {
	now := time.Now()
	return durableContinuationState{
		Version:                   durableContinuationStateVersion,
		SavedAt:                   now,
		ClientMachineID:           strings.TrimSpace(copilotClientMachineID),
		ResponseTurns:             captureCopilotTurnCacheEntries(&responseTurnCache.mu, responseTurnCache.entries, now),
		ResponseFunctionCallTurns: captureCopilotTurnCacheEntries(&responseFunctionCallTurnCache.mu, responseFunctionCallTurnCache.entries, now),
		MessageToolTurns:          captureCopilotTurnCacheEntries(&messageToolCallTurnCache.mu, messageToolCallTurnCache.entries, now),
		ResponsesReplay:           captureResponsesReplayEntries(now),
	}
}

func captureCopilotTurnCacheEntries(mu *sync.Mutex, entries map[string]*copilotTurnCacheEntry, now time.Time) map[string]durableCopilotTurnCacheEntry {
	mu.Lock()
	defer mu.Unlock()
	pruneCopilotTurnCacheLocked(entries, now)
	if len(entries) == 0 {
		return nil
	}
	snapshot := make(map[string]durableCopilotTurnCacheEntry, len(entries))
	for key, entry := range entries {
		snapshot[key] = durableCopilotTurnCacheEntry{
			AccountID:  entry.AccountID,
			Context:    entry.Context,
			CreatedAt:  entry.CreatedAt,
			AccessedAt: entry.AccessedAt,
		}
	}
	return snapshot
}

func captureResponsesReplayEntries(now time.Time) map[string]durableResponsesReplayEntry {
	responsesReplayCache.mu.Lock()
	defer responsesReplayCache.mu.Unlock()
	pruneResponsesReplayLocked(now)
	if len(responsesReplayCache.entries) == 0 {
		return nil
	}
	snapshot := make(map[string]durableResponsesReplayEntry, len(responsesReplayCache.entries))
	for responseID, entry := range responsesReplayCache.entries {
		snapshot[responseID] = durableResponsesReplayEntry{
			AccountID:   entry.AccountID,
			Input:       cloneJSONArray(entry.Input),
			ReplayItems: cloneJSONArray(entry.ReplayItems),
			CreatedAt:   entry.CreatedAt,
			AccessedAt:  entry.AccessedAt,
		}
	}
	return snapshot
}
