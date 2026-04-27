package instance

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"copilot-go/config"
	"copilot-go/store"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

var continuationMetadataPersistenceDisabled bool

var continuationMetadataStore = struct {
	mu   sync.Mutex
	db   *sql.DB
	path string
}{}

const (
	continuationBindingResponse     = "response"
	continuationBindingFunctionCall = "function_call"
	continuationBindingToolUse      = "tool_use"
	continuationMetaClientMachineID = "client_machine_id"
	legacyMetadataImportMarker      = "legacy_metadata_imported_v1"

	replayArchiveStateHotOnly = "hot_only"
	replayArchiveStateReady   = "ready"
	replayArchiveStateFailed  = "failed"
)

type continuationBindingRecord struct {
	BindingType     string
	BindingID       string
	AccountID       string
	InteractionID   string
	ClientSessionID string
	AgentTaskID     string
	CreatedAt       time.Time
	LastSeenAt      time.Time
	ExpiresAt       time.Time
}

type responsesReplayMetadataRecord struct {
	ResponseID   string
	AccountID    string
	ArchiveState string
	DuckDBRef    string
	CreatedAt    time.Time
	LastSeenAt   time.Time
	ExpiresAt    time.Time
}

func InitContinuationMetadataStore() error {
	_, err := ensureContinuationMetadataStore()
	return err
}

func CloseContinuationMetadataStore() error {
	continuationMetadataStore.mu.Lock()
	defer continuationMetadataStore.mu.Unlock()
	if continuationMetadataStore.db == nil {
		continuationMetadataStore.path = ""
		return nil
	}
	err := continuationMetadataStore.db.Close()
	continuationMetadataStore.db = nil
	continuationMetadataStore.path = ""
	return err
}

func resetContinuationMetadataStoreForTests() {
	_ = CloseContinuationMetadataStore()
}

func continuationMetadataEnabled() bool {
	return !continuationMetadataPersistenceDisabled
}

func ensureContinuationMetadataStore() (*sql.DB, error) {
	if !continuationMetadataEnabled() {
		return nil, nil
	}
	path := store.ContinuationSQLiteFile()
	continuationMetadataStore.mu.Lock()
	defer continuationMetadataStore.mu.Unlock()
	if continuationMetadataStore.db != nil && continuationMetadataStore.path == path {
		return continuationMetadataStore.db, nil
	}
	if continuationMetadataStore.db != nil {
		_ = continuationMetadataStore.db.Close()
		continuationMetadataStore.db = nil
		continuationMetadataStore.path = ""
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	if err := configureContinuationMetadataDB(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := migrateContinuationMetadataDB(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := cleanupExpiredContinuationMetadata(db, time.Now()); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := importLegacyContinuationMetadataIfNeeded(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := loadOrCreateClientMachineID(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	continuationMetadataStore.db = db
	continuationMetadataStore.path = path
	return db, nil
}

func configureContinuationMetadataDB(db *sql.DB) error {
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA foreign_keys=ON",
		"PRAGMA busy_timeout=5000",
	}
	for _, stmt := range pragmas {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("%s: %w", stmt, err)
		}
	}
	return nil
}

func migrateContinuationMetadataDB(db *sql.DB) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS continuation_meta (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			updated_at_unix INTEGER NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS continuation_bindings (
			binding_type TEXT NOT NULL,
			binding_id TEXT NOT NULL,
			account_id TEXT NOT NULL,
			interaction_id TEXT NOT NULL,
			client_session_id TEXT NOT NULL,
			agent_task_id TEXT NOT NULL,
			created_at_unix INTEGER NOT NULL,
			last_seen_at_unix INTEGER NOT NULL,
			expires_at_unix INTEGER NOT NULL,
			PRIMARY KEY (binding_type, binding_id)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_continuation_bindings_expires ON continuation_bindings(expires_at_unix)`,
		`CREATE INDEX IF NOT EXISTS idx_continuation_bindings_account ON continuation_bindings(binding_type, account_id)`,
		`CREATE TABLE IF NOT EXISTS responses_replay_metadata (
			response_id TEXT PRIMARY KEY,
			account_id TEXT NOT NULL,
			archive_state TEXT NOT NULL,
			duckdb_ref TEXT NOT NULL,
			created_at_unix INTEGER NOT NULL,
			last_seen_at_unix INTEGER NOT NULL,
			expires_at_unix INTEGER NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_responses_replay_metadata_expires ON responses_replay_metadata(expires_at_unix)`,
		`CREATE INDEX IF NOT EXISTS idx_responses_replay_metadata_account ON responses_replay_metadata(account_id)`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func loadOrCreateClientMachineID(db *sql.DB) error {
	machineID, ok, err := getContinuationMeta(db, continuationMetaClientMachineID)
	if err != nil {
		return err
	}
	if ok && strings.TrimSpace(machineID) != "" {
		copilotClientMachineID = strings.TrimSpace(machineID)
		return nil
	}
	machineID = strings.TrimSpace(copilotClientMachineID)
	if machineID == "" {
		machineID = uuid.NewString()
		copilotClientMachineID = machineID
	}
	return setContinuationMeta(db, continuationMetaClientMachineID, machineID)
}

func importLegacyContinuationMetadataIfNeeded(db *sql.DB) error {
	if _, ok, err := getContinuationMeta(db, legacyMetadataImportMarker); err != nil {
		return err
	} else if ok {
		return nil
	}
	body, err := readDurableContinuationStateFile()
	if err != nil {
		return err
	}
	if len(strings.TrimSpace(string(body))) == 0 {
		return setContinuationMeta(db, legacyMetadataImportMarker, time.Now().UTC().Format(time.RFC3339Nano))
	}
	var snapshot durableContinuationState
	if err := json.Unmarshal(body, &snapshot); err != nil {
		return err
	}
	if err := importLegacyBindingMap(db, continuationBindingResponse, snapshot.ResponseTurns); err != nil {
		return err
	}
	if err := importLegacyBindingMap(db, continuationBindingFunctionCall, snapshot.ResponseFunctionCallTurns); err != nil {
		return err
	}
	if err := importLegacyBindingMap(db, continuationBindingToolUse, snapshot.MessageToolTurns); err != nil {
		return err
	}
	if len(snapshot.ResponsesReplay) > 0 {
		if err := importLegacyReplayMetadata(db, snapshot.ResponsesReplay); err != nil {
			return err
		}
	}
	if clientMachineID := strings.TrimSpace(snapshot.ClientMachineID); clientMachineID != "" {
		if err := setContinuationMeta(db, continuationMetaClientMachineID, clientMachineID); err != nil {
			return err
		}
	}
	return setContinuationMeta(db, legacyMetadataImportMarker, time.Now().UTC().Format(time.RFC3339Nano))
}

func importLegacyBindingMap(db *sql.DB, bindingType string, snapshot map[string]durableCopilotTurnCacheEntry) error {
	if len(snapshot) == 0 {
		return nil
	}
	now := time.Now()
	retention := config.ContinuationMetadataRetention()
	for key, entry := range snapshot {
		key = strings.TrimSpace(key)
		accountID := strings.TrimSpace(entry.AccountID)
		if key == "" || accountID == "" {
			continue
		}
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		if now.Sub(createdAt) > retention {
			continue
		}
		lastSeenAt := entry.AccessedAt
		if lastSeenAt.IsZero() {
			lastSeenAt = createdAt
		}
		rec := continuationBindingRecord{
			BindingType:     bindingType,
			BindingID:       key,
			AccountID:       accountID,
			InteractionID:   entry.Context.InteractionID,
			ClientSessionID: entry.Context.ClientSessionID,
			AgentTaskID:     entry.Context.AgentTaskID,
			CreatedAt:       createdAt,
			LastSeenAt:      lastSeenAt,
			ExpiresAt:       createdAt.Add(retention),
		}
		if err := upsertContinuationBinding(db, rec); err != nil {
			return err
		}
	}
	return nil
}

func getContinuationMeta(db *sql.DB, key string) (string, bool, error) {
	row := db.QueryRow(`SELECT value FROM continuation_meta WHERE key = ?`, key)
	var value string
	if err := row.Scan(&value); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", false, nil
		}
		return "", false, err
	}
	return value, true, nil
}

func setContinuationMeta(db *sql.DB, key string, value string) error {
	_, err := db.Exec(`
		INSERT INTO continuation_meta (key, value, updated_at_unix)
		VALUES (?, ?, ?)
		ON CONFLICT(key) DO UPDATE SET
			value = excluded.value,
			updated_at_unix = excluded.updated_at_unix
	`, key, value, time.Now().Unix())
	return err
}

func cleanupExpiredContinuationMetadata(db *sql.DB, now time.Time) error {
	if _, err := db.Exec(`DELETE FROM continuation_bindings WHERE expires_at_unix < ?`, now.Unix()); err != nil {
		return err
	}
	_, err := db.Exec(`DELETE FROM responses_replay_metadata WHERE expires_at_unix < ?`, now.Unix())
	return err
}

func upsertContinuationBinding(db *sql.DB, rec continuationBindingRecord) error {
	if rec.CreatedAt.IsZero() {
		rec.CreatedAt = time.Now()
	}
	if rec.LastSeenAt.IsZero() {
		rec.LastSeenAt = rec.CreatedAt
	}
	if rec.ExpiresAt.IsZero() {
		rec.ExpiresAt = rec.CreatedAt.Add(config.ContinuationMetadataRetention())
	}
	_, err := db.Exec(`
		INSERT INTO continuation_bindings (
			binding_type,
			binding_id,
			account_id,
			interaction_id,
			client_session_id,
			agent_task_id,
			created_at_unix,
			last_seen_at_unix,
			expires_at_unix
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(binding_type, binding_id) DO UPDATE SET
			account_id = excluded.account_id,
			interaction_id = excluded.interaction_id,
			client_session_id = excluded.client_session_id,
			agent_task_id = excluded.agent_task_id,
			created_at_unix = excluded.created_at_unix,
			last_seen_at_unix = excluded.last_seen_at_unix,
			expires_at_unix = excluded.expires_at_unix
	`, rec.BindingType, rec.BindingID, rec.AccountID, rec.InteractionID, rec.ClientSessionID, rec.AgentTaskID, rec.CreatedAt.Unix(), rec.LastSeenAt.Unix(), rec.ExpiresAt.Unix())
	return err
}

func deleteContinuationBindingsForAccount(accountID string) error {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		return err
	}
	_, err = db.Exec(`DELETE FROM continuation_bindings WHERE account_id = ?`, accountID)
	return err
}

func importLegacyReplayMetadata(db *sql.DB, snapshot map[string]durableResponsesReplayEntry) error {
	if len(snapshot) == 0 {
		return nil
	}
	now := time.Now()
	retention := config.ContinuationMetadataRetention()
	for responseID, entry := range snapshot {
		responseID = strings.TrimSpace(responseID)
		accountID := strings.TrimSpace(entry.AccountID)
		if responseID == "" || accountID == "" {
			continue
		}
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = now
		}
		if now.Sub(createdAt) > retention {
			continue
		}
		lastSeenAt := entry.AccessedAt
		if lastSeenAt.IsZero() {
			lastSeenAt = createdAt
		}
		if err := upsertResponsesReplayMetadata(db, responsesReplayMetadataRecord{
			ResponseID:   responseID,
			AccountID:    accountID,
			ArchiveState: replayArchiveStateHotOnly,
			CreatedAt:    createdAt,
			LastSeenAt:   lastSeenAt,
			ExpiresAt:    createdAt.Add(retention),
		}); err != nil {
			return err
		}
	}
	return nil
}

func lookupContinuationBinding(bindingType string, bindingID string) (continuationBindingRecord, bool) {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("continuation metadata lookup init failed: %v", err)
		}
		return continuationBindingRecord{}, false
	}
	now := time.Now()
	row := db.QueryRow(`
		SELECT account_id, interaction_id, client_session_id, agent_task_id, created_at_unix, last_seen_at_unix, expires_at_unix
		FROM continuation_bindings
		WHERE binding_type = ? AND binding_id = ? AND expires_at_unix >= ?
	`, bindingType, bindingID, now.Unix())
	var rec continuationBindingRecord
	var createdUnix, lastSeenUnix, expiresUnix int64
	if err := row.Scan(&rec.AccountID, &rec.InteractionID, &rec.ClientSessionID, &rec.AgentTaskID, &createdUnix, &lastSeenUnix, &expiresUnix); err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("continuation metadata lookup failed: type=%s id=%s err=%v", bindingType, bindingID, err)
		}
		return continuationBindingRecord{}, false
	}
	rec.BindingType = bindingType
	rec.BindingID = bindingID
	rec.CreatedAt = time.Unix(createdUnix, 0)
	rec.LastSeenAt = time.Unix(lastSeenUnix, 0)
	rec.ExpiresAt = time.Unix(expiresUnix, 0)
	_, _ = db.Exec(`UPDATE continuation_bindings SET last_seen_at_unix = ? WHERE binding_type = ? AND binding_id = ?`, now.Unix(), bindingType, bindingID)
	return rec, true
}

func storeContinuationBinding(bindingType string, bindingID string, accountID string, ctx copilotTurnContext) {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("continuation metadata init failed: %v", err)
		}
		return
	}
	now := time.Now()
	rec := continuationBindingRecord{
		BindingType:     bindingType,
		BindingID:       bindingID,
		AccountID:       strings.TrimSpace(accountID),
		InteractionID:   strings.TrimSpace(ctx.InteractionID),
		ClientSessionID: strings.TrimSpace(ctx.ClientSessionID),
		AgentTaskID:     strings.TrimSpace(ctx.AgentTaskID),
		CreatedAt:       now,
		LastSeenAt:      now,
		ExpiresAt:       now.Add(config.ContinuationMetadataRetention()),
	}
	if rec.BindingID == "" || rec.AccountID == "" {
		return
	}
	if err := upsertContinuationBinding(db, rec); err != nil {
		log.Printf("continuation metadata upsert failed: type=%s id=%s err=%v", bindingType, bindingID, err)
	}
}

func bindingRecordContext(rec continuationBindingRecord) copilotTurnContext {
	return copilotTurnContext{
		InteractionID:   rec.InteractionID,
		ClientSessionID: rec.ClientSessionID,
		AgentTaskID:     rec.AgentTaskID,
	}
}

func upsertResponsesReplayMetadata(db *sql.DB, rec responsesReplayMetadataRecord) error {
	if rec.ResponseID == "" || rec.AccountID == "" {
		return nil
	}
	if rec.CreatedAt.IsZero() {
		rec.CreatedAt = time.Now()
	}
	if rec.LastSeenAt.IsZero() {
		rec.LastSeenAt = rec.CreatedAt
	}
	if rec.ExpiresAt.IsZero() {
		rec.ExpiresAt = rec.CreatedAt.Add(config.ContinuationMetadataRetention())
	}
	if strings.TrimSpace(rec.ArchiveState) == "" {
		rec.ArchiveState = replayArchiveStateHotOnly
	}
	_, err := db.Exec(`
		INSERT INTO responses_replay_metadata (
			response_id,
			account_id,
			archive_state,
			duckdb_ref,
			created_at_unix,
			last_seen_at_unix,
			expires_at_unix
		)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(response_id) DO UPDATE SET
			account_id = excluded.account_id,
			archive_state = excluded.archive_state,
			duckdb_ref = excluded.duckdb_ref,
			created_at_unix = excluded.created_at_unix,
			last_seen_at_unix = excluded.last_seen_at_unix,
			expires_at_unix = excluded.expires_at_unix
	`, rec.ResponseID, rec.AccountID, rec.ArchiveState, rec.DuckDBRef, rec.CreatedAt.Unix(), rec.LastSeenAt.Unix(), rec.ExpiresAt.Unix())
	return err
}

func storeResponsesReplayMetadata(responseID string, accountID string, archiveState string, duckDBRef string) {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("responses replay metadata init failed: %v", err)
		}
		return
	}
	now := time.Now()
	if err := upsertResponsesReplayMetadata(db, responsesReplayMetadataRecord{
		ResponseID:   strings.TrimSpace(responseID),
		AccountID:    strings.TrimSpace(accountID),
		ArchiveState: strings.TrimSpace(archiveState),
		DuckDBRef:    strings.TrimSpace(duckDBRef),
		CreatedAt:    now,
		LastSeenAt:   now,
		ExpiresAt:    now.Add(config.ContinuationMetadataRetention()),
	}); err != nil {
		log.Printf("responses replay metadata upsert failed: response_id=%s err=%v", responseID, err)
	}
}

func lookupResponsesReplayMetadata(responseID string) (responsesReplayMetadataRecord, bool) {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		if err != nil {
			log.Printf("responses replay metadata lookup init failed: %v", err)
		}
		return responsesReplayMetadataRecord{}, false
	}
	now := time.Now()
	row := db.QueryRow(`
		SELECT account_id, archive_state, duckdb_ref, created_at_unix, last_seen_at_unix, expires_at_unix
		FROM responses_replay_metadata
		WHERE response_id = ? AND expires_at_unix >= ?
	`, strings.TrimSpace(responseID), now.Unix())
	var rec responsesReplayMetadataRecord
	var createdUnix, lastSeenUnix, expiresUnix int64
	if err := row.Scan(&rec.AccountID, &rec.ArchiveState, &rec.DuckDBRef, &createdUnix, &lastSeenUnix, &expiresUnix); err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			log.Printf("responses replay metadata lookup failed: response_id=%s err=%v", responseID, err)
		}
		return responsesReplayMetadataRecord{}, false
	}
	rec.ResponseID = strings.TrimSpace(responseID)
	rec.CreatedAt = time.Unix(createdUnix, 0)
	rec.LastSeenAt = time.Unix(lastSeenUnix, 0)
	rec.ExpiresAt = time.Unix(expiresUnix, 0)
	_, _ = db.Exec(`UPDATE responses_replay_metadata SET last_seen_at_unix = ? WHERE response_id = ?`, now.Unix(), rec.ResponseID)
	return rec, true
}

func deleteResponsesReplayMetadataForAccount(accountID string) error {
	db, err := ensureContinuationMetadataStore()
	if err != nil || db == nil {
		return err
	}
	_, err = db.Exec(`DELETE FROM responses_replay_metadata WHERE account_id = ?`, accountID)
	return err
}
