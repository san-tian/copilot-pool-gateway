package instance

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"copilot-go/store"

	_ "github.com/duckdb/duckdb-go/v2"
)

var responsesReplayArchiveDisabled bool

var responsesReplayArchiveStore = struct {
	mu   sync.Mutex
	db   *sql.DB
	path string
}{}

func InitResponsesReplayArchiveStore() error {
	_, err := ensureResponsesReplayArchiveStore()
	return err
}

func CloseResponsesReplayArchiveStore() error {
	responsesReplayArchiveStore.mu.Lock()
	defer responsesReplayArchiveStore.mu.Unlock()
	if responsesReplayArchiveStore.db == nil {
		responsesReplayArchiveStore.path = ""
		return nil
	}
	err := responsesReplayArchiveStore.db.Close()
	responsesReplayArchiveStore.db = nil
	responsesReplayArchiveStore.path = ""
	return err
}

func resetResponsesReplayArchiveStoreForTests() {
	_ = CloseResponsesReplayArchiveStore()
}

func responsesReplayArchiveEnabled() bool {
	return !responsesReplayArchiveDisabled
}

func ensureResponsesReplayArchiveStore() (*sql.DB, error) {
	if !responsesReplayArchiveEnabled() {
		return nil, nil
	}
	path := store.PayloadDuckDBFile()
	responsesReplayArchiveStore.mu.Lock()
	defer responsesReplayArchiveStore.mu.Unlock()
	if responsesReplayArchiveStore.db != nil && responsesReplayArchiveStore.path == path {
		return responsesReplayArchiveStore.db, nil
	}
	if responsesReplayArchiveStore.db != nil {
		_ = responsesReplayArchiveStore.db.Close()
		responsesReplayArchiveStore.db = nil
		responsesReplayArchiveStore.path = ""
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	db, err := sql.Open("duckdb", path)
	if err != nil {
		return nil, err
	}
	if err := migrateResponsesReplayArchiveDB(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	responsesReplayArchiveStore.db = db
	responsesReplayArchiveStore.path = path
	return db, nil
}

func migrateResponsesReplayArchiveDB(db *sql.DB) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS archived_responses_replay_payloads (
			response_id VARCHAR PRIMARY KEY,
			account_id VARCHAR NOT NULL,
			input_json VARCHAR NOT NULL,
			replay_items_json VARCHAR NOT NULL,
			created_at TIMESTAMP NOT NULL,
			last_seen_at TIMESTAMP NOT NULL
		)`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func storeResponsesReplayPayloadCold(responseID string, accountID string, input []interface{}, replayItems []interface{}) bool {
	start := time.Now()
	db, err := ensureResponsesReplayArchiveStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("responses replay archive init failed: %v", err)
		}
		return false
	}
	responseID = strings.TrimSpace(responseID)
	accountID = strings.TrimSpace(accountID)
	if responseID == "" || accountID == "" {
		return false
	}
	inputJSON, err := json.Marshal(cloneJSONArray(input))
	if err != nil {
		log.Printf("responses replay archive marshal input failed: response_id=%s err=%v", responseID, err)
		return false
	}
	replayJSON, err := json.Marshal(cloneJSONArray(replayItems))
	if err != nil {
		log.Printf("responses replay archive marshal replay failed: response_id=%s err=%v", responseID, err)
		return false
	}
	now := time.Now().UTC()
	_, err = db.Exec(`
		INSERT INTO archived_responses_replay_payloads (
			response_id,
			account_id,
			input_json,
			replay_items_json,
			created_at,
			last_seen_at
		)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(response_id) DO UPDATE SET
			account_id = excluded.account_id,
			input_json = excluded.input_json,
			replay_items_json = excluded.replay_items_json,
			created_at = excluded.created_at,
			last_seen_at = excluded.last_seen_at
	`, responseID, accountID, string(inputJSON), string(replayJSON), now, now)
	if err != nil {
		log.Printf("responses replay archive upsert failed: response_id=%s input_bytes=%d replay_bytes=%d elapsed_ms=%d err=%v", responseID, len(inputJSON), len(replayJSON), time.Since(start).Milliseconds(), err)
		return false
	}
	log.Printf("responses replay archive write response_id=%s account=%s input_bytes=%d replay_bytes=%d elapsed_ms=%d", responseID, accountID, len(inputJSON), len(replayJSON), time.Since(start).Milliseconds())
	return true
}

func loadResponsesReplayPayloadCold(responseID string) (*responsesReplayEntry, bool) {
	db, err := ensureResponsesReplayArchiveStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("responses replay archive init failed: %v", err)
		}
		return nil, false
	}
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return nil, false
	}
	row := db.QueryRow(`
		SELECT account_id, input_json, replay_items_json, created_at, last_seen_at
		FROM archived_responses_replay_payloads
		WHERE response_id = ?
	`, responseID)
	var accountID string
	var inputJSON string
	var replayJSON string
	var createdAt time.Time
	var lastSeenAt time.Time
	if err := row.Scan(&accountID, &inputJSON, &replayJSON, &createdAt, &lastSeenAt); err != nil {
		if err != sql.ErrNoRows {
			log.Printf("responses replay archive lookup failed: response_id=%s err=%v", responseID, err)
		}
		return nil, false
	}
	var input []interface{}
	if err := json.Unmarshal([]byte(inputJSON), &input); err != nil {
		log.Printf("responses replay archive unmarshal input failed: response_id=%s err=%v", responseID, err)
		return nil, false
	}
	var replayItems []interface{}
	if err := json.Unmarshal([]byte(replayJSON), &replayItems); err != nil {
		log.Printf("responses replay archive unmarshal replay failed: response_id=%s err=%v", responseID, err)
		return nil, false
	}
	now := time.Now().UTC()
	_, _ = db.Exec(`UPDATE archived_responses_replay_payloads SET last_seen_at = ? WHERE response_id = ?`, now, responseID)
	return &responsesReplayEntry{
		AccountID:   accountID,
		Input:       cloneJSONArray(input),
		ReplayItems: cloneJSONArray(replayItems),
		CreatedAt:   createdAt,
		AccessedAt:  now,
	}, true
}

func deleteResponsesReplayPayloadsColdForAccount(accountID string) error {
	db, err := ensureResponsesReplayArchiveStore()
	if err != nil || db == nil {
		return err
	}
	_, err = db.Exec(`DELETE FROM archived_responses_replay_payloads WHERE account_id = ?`, strings.TrimSpace(accountID))
	return err
}

func PingResponsesReplayArchiveStore() error {
	db, err := ensureResponsesReplayArchiveStore()
	if err != nil {
		return err
	}
	if db == nil {
		return fmt.Errorf("responses replay archive disabled")
	}
	var n int
	if err := db.QueryRow(`SELECT 42`).Scan(&n); err != nil {
		return err
	}
	if n != 42 {
		return fmt.Errorf("unexpected duckdb probe result: %d", n)
	}
	return nil
}
