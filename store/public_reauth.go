package store

import (
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
)

type PublicReauthSession struct {
	ID              string `json:"id"`
	AccountID       string `json:"accountId"`
	AccountName     string `json:"accountName"`
	AccountType     string `json:"accountType"`
	Status          string `json:"status"`
	AuthSessionID   string `json:"authSessionId,omitempty"`
	UserCode        string `json:"userCode,omitempty"`
	VerificationURI string `json:"verificationUri,omitempty"`
	CreatedAt       string `json:"createdAt"`
	ExpiresAt       string `json:"expiresAt"`
	CompletedAt     string `json:"completedAt,omitempty"`
	Error           string `json:"error,omitempty"`
}

type publicReauthStore struct {
	Sessions []PublicReauthSession `json:"sessions"`
}

var publicReauthMu sync.RWMutex

func PublicReauthFile() string {
	return AppDir + "/public-reauth-sessions.json"
}

func readPublicReauthSessions() ([]PublicReauthSession, error) {
	data, err := os.ReadFile(PublicReauthFile())
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || string(data) == "{}" {
		return []PublicReauthSession{}, nil
	}
	var s publicReauthStore
	if err := json.Unmarshal(data, &s); err != nil {
		var sessions []PublicReauthSession
		if err2 := json.Unmarshal(data, &sessions); err2 != nil {
			return []PublicReauthSession{}, nil
		}
		return sessions, nil
	}
	return s.Sessions, nil
}

func writePublicReauthSessions(sessions []PublicReauthSession) error {
	s := publicReauthStore{Sessions: sessions}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(PublicReauthFile(), data, 0644)
}

func CreatePublicReauthSession(account Account, ttl time.Duration) (*PublicReauthSession, error) {
	publicReauthMu.Lock()
	defer publicReauthMu.Unlock()

	sessions, err := readPublicReauthSessions()
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()
	session := PublicReauthSession{
		ID:          uuid.New().String(),
		AccountID:   account.ID,
		AccountName: account.Name,
		AccountType: account.AccountType,
		Status:      "ready",
		CreatedAt:   now.Format(time.RFC3339),
		ExpiresAt:   now.Add(ttl).Format(time.RFC3339),
	}

	sessions = append(sessions, session)
	if err := writePublicReauthSessions(sessions); err != nil {
		return nil, err
	}
	return &session, nil
}

func GetPublicReauthSession(id string) (*PublicReauthSession, error) {
	publicReauthMu.RLock()
	defer publicReauthMu.RUnlock()

	sessions, err := readPublicReauthSessions()
	if err != nil {
		return nil, err
	}
	for _, session := range sessions {
		if session.ID == id {
			return &session, nil
		}
	}
	return nil, nil
}

func UpdatePublicReauthSession(updated PublicReauthSession) error {
	publicReauthMu.Lock()
	defer publicReauthMu.Unlock()

	sessions, err := readPublicReauthSessions()
	if err != nil {
		return err
	}
	for i, session := range sessions {
		if session.ID == updated.ID {
			sessions[i] = updated
			return writePublicReauthSessions(sessions)
		}
	}
	sessions = append(sessions, updated)
	return writePublicReauthSessions(sessions)
}
