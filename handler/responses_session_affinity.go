package handler

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

const (
	responsesSessionAffinityTTL        = 6 * time.Hour
	responsesSessionAffinityMaxEntries = 4096
	responsesRoutingTelemetryMaxEvents = 256
)

type responsesSessionAffinityEntry struct {
	AccountID  string
	AccessedAt time.Time
}

var responsesSessionAffinity = struct {
	mu      sync.Mutex
	entries map[string]responsesSessionAffinityEntry
}{
	entries: map[string]responsesSessionAffinityEntry{},
}

type responsesRoutingTelemetryEvent struct {
	Time          string `json:"time"`
	Kind          string `json:"kind"`
	RequestID     string `json:"request_id,omitempty"`
	SessionKey    string `json:"session_key,omitempty"`
	SessionSource string `json:"session_source,omitempty"`
	Model         string `json:"model,omitempty"`
	AccountID     string `json:"account_id,omitempty"`
	FromAccount   string `json:"from_account,omitempty"`
	ToAccount     string `json:"to_account,omitempty"`
	Attempt       int    `json:"attempt,omitempty"`
	StatusCode    int    `json:"status_code,omitempty"`
	Reason        string `json:"reason,omitempty"`
	StickyKind    string `json:"sticky_kind,omitempty"`
	RecoveryRoute string `json:"recovery_route,omitempty"`
	Mode          string `json:"mode,omitempty"`
	Continuation  bool   `json:"continuation,omitempty"`
}

type responsesRoutingTelemetrySnapshot struct {
	StartedAt   string                           `json:"started_at"`
	UpdatedAt   string                           `json:"updated_at,omitempty"`
	Counters    map[string]uint64                `json:"counters"`
	LastEventAt map[string]string                `json:"last_event_at"`
	Recent      []responsesRoutingTelemetryEvent `json:"recent"`
}

var responsesRoutingTelemetry = struct {
	mu        sync.Mutex
	startedAt time.Time
	updatedAt time.Time
	counters  map[string]uint64
	lastAt    map[string]time.Time
	recent    []responsesRoutingTelemetryEvent
}{
	startedAt: time.Now().UTC(),
	counters:  map[string]uint64{},
	lastAt:    map[string]time.Time{},
	recent:    []responsesRoutingTelemetryEvent{},
}

func recordResponsesRoutingEvent(event responsesRoutingTelemetryEvent) {
	event.Kind = strings.TrimSpace(event.Kind)
	if event.Kind == "" {
		return
	}
	now := time.Now().UTC()
	event.Time = now.Format(time.RFC3339Nano)
	responsesRoutingTelemetry.mu.Lock()
	responsesRoutingTelemetry.updatedAt = now
	responsesRoutingTelemetry.counters[event.Kind]++
	responsesRoutingTelemetry.lastAt[event.Kind] = now
	responsesRoutingTelemetry.recent = append(responsesRoutingTelemetry.recent, event)
	if len(responsesRoutingTelemetry.recent) > responsesRoutingTelemetryMaxEvents {
		copy(responsesRoutingTelemetry.recent, responsesRoutingTelemetry.recent[len(responsesRoutingTelemetry.recent)-responsesRoutingTelemetryMaxEvents:])
		responsesRoutingTelemetry.recent = responsesRoutingTelemetry.recent[:responsesRoutingTelemetryMaxEvents]
	}
	responsesRoutingTelemetry.mu.Unlock()

	log.Printf("[responses metric] kind=%s time=%s rid=%s session_source=%q session_key=%q model=%q account=%q from=%q to=%q attempt=%d status=%d reason=%q sticky=%q recovery_route=%q mode=%q continuation=%v",
		event.Kind, event.Time, event.RequestID, event.SessionSource, event.SessionKey, event.Model, event.AccountID,
		event.FromAccount, event.ToAccount, event.Attempt, event.StatusCode, event.Reason, event.StickyKind,
		event.RecoveryRoute, event.Mode, event.Continuation)
}

func snapshotResponsesRoutingTelemetry(limit int) responsesRoutingTelemetrySnapshot {
	if limit <= 0 || limit > responsesRoutingTelemetryMaxEvents {
		limit = responsesRoutingTelemetryMaxEvents
	}
	responsesRoutingTelemetry.mu.Lock()
	defer responsesRoutingTelemetry.mu.Unlock()

	counters := make(map[string]uint64, len(responsesRoutingTelemetry.counters))
	for k, v := range responsesRoutingTelemetry.counters {
		counters[k] = v
	}
	lastAt := make(map[string]string, len(responsesRoutingTelemetry.lastAt))
	for k, v := range responsesRoutingTelemetry.lastAt {
		lastAt[k] = v.Format(time.RFC3339Nano)
	}
	recent := responsesRoutingTelemetry.recent
	if len(recent) > limit {
		recent = recent[len(recent)-limit:]
	}
	recentCopy := make([]responsesRoutingTelemetryEvent, len(recent))
	copy(recentCopy, recent)
	updatedAt := ""
	if !responsesRoutingTelemetry.updatedAt.IsZero() {
		updatedAt = responsesRoutingTelemetry.updatedAt.Format(time.RFC3339Nano)
	}
	return responsesRoutingTelemetrySnapshot{
		StartedAt:   responsesRoutingTelemetry.startedAt.Format(time.RFC3339Nano),
		UpdatedAt:   updatedAt,
		Counters:    counters,
		LastEventAt: lastAt,
		Recent:      recentCopy,
	}
}

func resetResponsesRoutingTelemetryForTest() {
	responsesRoutingTelemetry.mu.Lock()
	defer responsesRoutingTelemetry.mu.Unlock()
	responsesRoutingTelemetry.startedAt = time.Now().UTC()
	responsesRoutingTelemetry.updatedAt = time.Time{}
	responsesRoutingTelemetry.counters = map[string]uint64{}
	responsesRoutingTelemetry.lastAt = map[string]time.Time{}
	responsesRoutingTelemetry.recent = []responsesRoutingTelemetryEvent{}
}

func responsesSessionAffinityKey(c *gin.Context, body []byte) (string, string) {
	for _, h := range []string{
		"X-Copilot-Pool-Session",
		"X-Copilot-Session-Id",
		"X-Client-Session-Id",
		"X-Session-Id",
		"X-Conversation-Id",
		"OpenAI-Conversation-Id",
	} {
		if v := usableAffinityValue(c.GetHeader(h)); v != "" {
			return hashAffinityKey("header:"+strings.ToLower(h), v), h
		}
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", ""
	}
	for _, k := range []string{"session_id", "conversation_id", "thread_id"} {
		if v := usableAffinityValue(fmt.Sprint(payload[k])); v != "" && v != "<nil>" {
			return hashAffinityKey("body:"+k, v), "body." + k
		}
	}
	if metadata, ok := payload["metadata"].(map[string]interface{}); ok {
		for _, k := range []string{"session_id", "conversation_id", "thread_id", "trace_id"} {
			if v := usableAffinityValue(fmt.Sprint(metadata[k])); v != "" && v != "<nil>" {
				return hashAffinityKey("metadata:"+k, v), "metadata." + k
			}
		}
	}
	switch conv := payload["conversation"].(type) {
	case string:
		if v := usableAffinityValue(conv); v != "" {
			return hashAffinityKey("body:conversation", v), "body.conversation"
		}
	case map[string]interface{}:
		for _, k := range []string{"id", "conversation_id"} {
			if v := usableAffinityValue(fmt.Sprint(conv[k])); v != "" && v != "<nil>" {
				return hashAffinityKey("body:conversation."+k, v), "body.conversation." + k
			}
		}
	}
	return "", ""
}

func setResponsesSessionAffinityContext(c *gin.Context, isPool interface{}, body []byte) {
	if isPool != true {
		return
	}
	if key, source := responsesSessionAffinityKey(c, body); key != "" {
		c.Set("responsesSessionAffinityKey", key)
		c.Set("responsesSessionAffinitySource", source)
	}
}

func usableAffinityValue(v string) string {
	v = strings.TrimSpace(v)
	if len(v) < 8 || len(v) > 512 {
		return ""
	}
	return v
}

func hashAffinityKey(namespace, value string) string {
	sum := sha256.Sum256([]byte(namespace + "\x00" + value))
	return hex.EncodeToString(sum[:])
}

func lookupResponsesSessionAffinity(key, requestedModel string, exclude map[string]bool) *resolvedAccount {
	key = strings.TrimSpace(key)
	if key == "" {
		return nil
	}
	now := time.Now()
	responsesSessionAffinity.mu.Lock()
	entry, ok := responsesSessionAffinity.entries[key]
	if !ok || now.Sub(entry.AccessedAt) > responsesSessionAffinityTTL {
		if ok {
			delete(responsesSessionAffinity.entries, key)
			recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
				Kind:       "session_affinity_forget",
				SessionKey: key,
				Model:      requestedModel,
				AccountID:  entry.AccountID,
				Reason:     "expired",
			})
		}
		responsesSessionAffinity.mu.Unlock()
		return nil
	}
	entry.AccessedAt = now
	responsesSessionAffinity.entries[key] = entry
	responsesSessionAffinity.mu.Unlock()

	if exclude != nil && exclude[entry.AccountID] {
		recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
			Kind:       "session_affinity_ignored",
			SessionKey: key,
			Model:      requestedModel,
			AccountID:  entry.AccountID,
			Reason:     "excluded",
		})
		return nil
	}
	if store.IsModelBlocked(entry.AccountID, requestedModel) {
		forgetResponsesSessionAffinity(key, entry.AccountID)
		recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
			Kind:       "session_affinity_forget",
			SessionKey: key,
			Model:      requestedModel,
			AccountID:  entry.AccountID,
			Reason:     "model_blocked",
		})
		return nil
	}
	account, _ := store.GetAccount(entry.AccountID)
	if account == nil || !account.Enabled {
		forgetResponsesSessionAffinity(key, entry.AccountID)
		recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
			Kind:       "session_affinity_forget",
			SessionKey: key,
			Model:      requestedModel,
			AccountID:  entry.AccountID,
			Reason:     "account_disabled_or_missing",
		})
		return nil
	}
	state := instance.GetInstanceState(entry.AccountID)
	if state == nil {
		forgetResponsesSessionAffinity(key, entry.AccountID)
		recordResponsesRoutingEvent(responsesRoutingTelemetryEvent{
			Kind:       "session_affinity_forget",
			SessionKey: key,
			Model:      requestedModel,
			AccountID:  entry.AccountID,
			Reason:     "instance_not_running",
		})
		return nil
	}
	return &resolvedAccount{State: state, AccountID: entry.AccountID}
}

func rememberResponsesSessionAffinity(key, accountID string) {
	key = strings.TrimSpace(key)
	accountID = strings.TrimSpace(accountID)
	if key == "" || accountID == "" {
		return
	}
	now := time.Now()
	responsesSessionAffinity.mu.Lock()
	defer responsesSessionAffinity.mu.Unlock()
	responsesSessionAffinity.entries[key] = responsesSessionAffinityEntry{AccountID: accountID, AccessedAt: now}
	if len(responsesSessionAffinity.entries) <= responsesSessionAffinityMaxEntries {
		return
	}
	var oldestKey string
	var oldest time.Time
	for k, entry := range responsesSessionAffinity.entries {
		if oldestKey == "" || entry.AccessedAt.Before(oldest) {
			oldestKey = k
			oldest = entry.AccessedAt
		}
	}
	if oldestKey != "" {
		delete(responsesSessionAffinity.entries, oldestKey)
	}
}

func forgetResponsesSessionAffinity(key, accountID string) {
	responsesSessionAffinity.mu.Lock()
	defer responsesSessionAffinity.mu.Unlock()
	if entry, ok := responsesSessionAffinity.entries[key]; ok && entry.AccountID == accountID {
		delete(responsesSessionAffinity.entries, key)
	}
}
