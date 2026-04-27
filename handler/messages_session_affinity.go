package handler

import (
	"encoding/json"
	"regexp"
	"strings"
	"sync"
	"time"

	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

const (
	messagesSessionAffinityTTL        = 6 * time.Hour
	messagesSessionAffinityMaxEntries = 4096
)

type messagesSessionAffinityEntry struct {
	AccountID  string
	AccessedAt time.Time
}

var messagesSessionAffinity = struct {
	mu      sync.Mutex
	entries map[string]messagesSessionAffinityEntry
}{
	entries: map[string]messagesSessionAffinityEntry{},
}

func messagesSessionAffinityKey(c *gin.Context, body []byte) (string, string) {
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

	var payload struct {
		Metadata *struct {
			UserID string `json:"user_id"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || payload.Metadata == nil {
		return "", ""
	}

	deviceID, sessionID := parseMessagesUserIDMetadata(payload.Metadata.UserID)
	if usableAffinityValue(sessionID) == "" {
		return "", ""
	}
	anchor := sessionID
	if usableAffinityValue(deviceID) != "" {
		anchor = deviceID + "\x00" + sessionID
	}
	return hashAffinityKey("messages:metadata.user_id", anchor), "metadata.user_id.session_id"
}

func parseMessagesUserIDMetadata(userID string) (deviceID string, sessionID string) {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return "", ""
	}

	legacyDeviceID := ""
	legacySessionID := ""
	if match := legacyUserIDDevicePattern.FindStringSubmatch(userID); len(match) == 2 {
		legacyDeviceID = strings.TrimSpace(match[1])
	}
	if match := legacyUserIDSessionPattern.FindStringSubmatch(userID); len(match) == 2 {
		legacySessionID = strings.TrimSpace(match[1])
	}
	if legacySessionID != "" {
		return legacyDeviceID, legacySessionID
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(userID), &parsed); err != nil {
		return "", ""
	}

	deviceID = trimmedJSONField(parsed, "device_id")
	if deviceID == "" {
		deviceID = trimmedJSONField(parsed, "account_uuid")
	}
	sessionID = trimmedJSONField(parsed, "session_id")
	return deviceID, sessionID
}

func trimmedJSONField(payload map[string]interface{}, key string) string {
	v, _ := payload[key].(string)
	return strings.TrimSpace(v)
}

var (
	legacyUserIDDevicePattern  = regexpMustCompile(`user_([^_]+)_account`)
	legacyUserIDSessionPattern = regexpMustCompile(`_session_(.+)$`)
)

func regexpMustCompile(expr string) *regexp.Regexp {
	return regexp.MustCompile(expr)
}

func setMessagesSessionAffinityContext(c *gin.Context, isPool interface{}, body []byte) {
	if isPool != true {
		return
	}
	if key, source := messagesSessionAffinityKey(c, body); key != "" {
		c.Set("messagesSessionAffinityKey", key)
		c.Set("messagesSessionAffinitySource", source)
	}
}

func lookupMessagesSessionAffinity(key, requestedModel string, exclude map[string]bool) *resolvedAccount {
	key = strings.TrimSpace(key)
	if key == "" {
		return nil
	}
	now := time.Now()
	messagesSessionAffinity.mu.Lock()
	entry, ok := messagesSessionAffinity.entries[key]
	if !ok || now.Sub(entry.AccessedAt) > messagesSessionAffinityTTL {
		if ok {
			delete(messagesSessionAffinity.entries, key)
		}
		messagesSessionAffinity.mu.Unlock()
		return nil
	}
	entry.AccessedAt = now
	messagesSessionAffinity.entries[key] = entry
	messagesSessionAffinity.mu.Unlock()

	if exclude != nil && exclude[entry.AccountID] {
		return nil
	}
	if store.IsModelBlocked(entry.AccountID, requestedModel) {
		forgetMessagesSessionAffinity(key, entry.AccountID)
		return nil
	}
	state := instance.GetInstanceState(entry.AccountID)
	if state == nil {
		forgetMessagesSessionAffinity(key, entry.AccountID)
		return nil
	}
	return &resolvedAccount{State: state, AccountID: entry.AccountID}
}

func bindMessagesSessionAffinity(key, accountID string) {
	key = strings.TrimSpace(key)
	accountID = strings.TrimSpace(accountID)
	if key == "" || accountID == "" {
		return
	}
	now := time.Now()
	messagesSessionAffinity.mu.Lock()
	defer messagesSessionAffinity.mu.Unlock()
	messagesSessionAffinity.entries[key] = messagesSessionAffinityEntry{AccountID: accountID, AccessedAt: now}
	if len(messagesSessionAffinity.entries) <= messagesSessionAffinityMaxEntries {
		return
	}
	oldestKey := ""
	var oldestTime time.Time
	for k, entry := range messagesSessionAffinity.entries {
		if oldestKey == "" || entry.AccessedAt.Before(oldestTime) {
			oldestKey = k
			oldestTime = entry.AccessedAt
		}
	}
	if oldestKey != "" {
		delete(messagesSessionAffinity.entries, oldestKey)
	}
}

func forgetMessagesSessionAffinity(key, accountID string) {
	key = strings.TrimSpace(key)
	if key == "" {
		return
	}
	messagesSessionAffinity.mu.Lock()
	defer messagesSessionAffinity.mu.Unlock()
	if entry, ok := messagesSessionAffinity.entries[key]; ok {
		if accountID == "" || entry.AccountID == accountID {
			delete(messagesSessionAffinity.entries, key)
		}
	}
}

func resetMessagesSessionAffinityForTest() {
	messagesSessionAffinity.mu.Lock()
	defer messagesSessionAffinity.mu.Unlock()
	messagesSessionAffinity.entries = map[string]messagesSessionAffinityEntry{}
}
