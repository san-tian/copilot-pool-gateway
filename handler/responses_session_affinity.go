package handler

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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
		}
		responsesSessionAffinity.mu.Unlock()
		return nil
	}
	entry.AccessedAt = now
	responsesSessionAffinity.entries[key] = entry
	responsesSessionAffinity.mu.Unlock()

	if exclude != nil && exclude[entry.AccountID] {
		return nil
	}
	if store.IsModelBlocked(entry.AccountID, requestedModel) {
		forgetResponsesSessionAffinity(key, entry.AccountID)
		return nil
	}
	account, _ := store.GetAccount(entry.AccountID)
	if account == nil || !account.Enabled {
		forgetResponsesSessionAffinity(key, entry.AccountID)
		return nil
	}
	state := instance.GetInstanceState(entry.AccountID)
	if state == nil {
		forgetResponsesSessionAffinity(key, entry.AccountID)
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
