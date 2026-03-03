package auth

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"copilot-go/config"

	"github.com/google/uuid"
)

type AuthSession struct {
	ID              string    `json:"id"`
	DeviceCode      string    `json:"deviceCode"`
	UserCode        string    `json:"userCode"`
	VerificationURI string    `json:"verificationUri"`
	ExpiresAt       time.Time `json:"expiresAt"`
	Interval        int       `json:"interval"`
	Status          string    `json:"status"` // "pending", "completed", "expired", "error"
	AccessToken     string    `json:"accessToken,omitempty"`
	Error           string    `json:"error,omitempty"`
}

var (
	authSessions sync.Map
)

type deviceCodeResponse struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
}

type tokenResponse struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	Error       string `json:"error"`
	Interval    int    `json:"interval,omitempty"`
}

// errSlowDown is returned when GitHub asks to increase the polling interval.
type errSlowDown struct {
	Interval int
}

func (e *errSlowDown) Error() string {
	return fmt.Sprintf("slow_down: interval=%d", e.Interval)
}

func StartDeviceFlow() (*AuthSession, error) {
	log.Printf("[DeviceFlow] Starting device flow, client_id=%s", config.GithubClientID)

	body, _ := json.Marshal(map[string]string{
		"client_id": config.GithubClientID,
		"scope":     "read:user",
	})

	req, err := http.NewRequest("POST", config.GithubDeviceURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[DeviceFlow] Request to %s failed: %v", config.GithubDeviceURL, err)
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	log.Printf("[DeviceFlow] Device code response (status %d): %s", resp.StatusCode, string(respBody))

	var dcResp deviceCodeResponse
	if err := json.Unmarshal(respBody, &dcResp); err != nil {
		return nil, fmt.Errorf("failed to parse device code response: %w", err)
	}

	session := &AuthSession{
		ID:              uuid.New().String(),
		DeviceCode:      dcResp.DeviceCode,
		UserCode:        dcResp.UserCode,
		VerificationURI: dcResp.VerificationURI,
		ExpiresAt:       time.Now().Add(time.Duration(dcResp.ExpiresIn) * time.Second),
		Interval:        dcResp.Interval,
		Status:          "pending",
	}

	if session.Interval < 5 {
		session.Interval = 5
	}

	log.Printf("[DeviceFlow] Session created: id=%s userCode=%s interval=%d expiresIn=%d", session.ID, session.UserCode, session.Interval, dcResp.ExpiresIn)

	authSessions.Store(session.ID, session)

	go pollForToken(session)

	return session, nil
}

func pollForToken(session *AuthSession) {
	interval := time.Duration(session.Interval) * time.Second
	log.Printf("[DeviceFlow] Poll goroutine started for session %s, interval=%v", session.ID, interval)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if time.Now().After(session.ExpiresAt) {
				log.Printf("[DeviceFlow] Session %s expired", session.ID)
				session.Status = "expired"
				session.Error = "device code expired"
				authSessions.Store(session.ID, session)
				return
			}

			token, err := requestToken(session.DeviceCode)
			if err != nil {
				// Handle slow_down: increase interval and reset ticker
				var sd *errSlowDown
				if errors.As(err, &sd) {
					interval = time.Duration(sd.Interval) * time.Second
					log.Printf("[DeviceFlow] Session %s slow_down, new interval=%v", session.ID, interval)
					ticker.Reset(interval)
				} else {
					log.Printf("[DeviceFlow] Session %s poll: %v", session.ID, err)
				}
				continue
			}

			if token != "" {
				log.Printf("[DeviceFlow] Session %s got token (len=%d)", session.ID, len(token))
				session.Status = "completed"
				session.AccessToken = token
				authSessions.Store(session.ID, session)
				return
			}
			log.Printf("[DeviceFlow] Session %s poll: empty token, no error", session.ID)
		}
	}
}

func requestToken(deviceCode string) (string, error) {
	body, _ := json.Marshal(map[string]string{
		"client_id":   config.GithubClientID,
		"device_code": deviceCode,
		"grant_type":  "urn:ietf:params:oauth:grant-type:device_code",
	})

	req, err := http.NewRequest("POST", config.GithubTokenURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[DeviceFlow] Token request HTTP error: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	log.Printf("[DeviceFlow] Token response (status %d): %s", resp.StatusCode, string(respBody))

	var tokenResp tokenResponse
	if err := json.Unmarshal(respBody, &tokenResp); err != nil {
		log.Printf("[DeviceFlow] Token response parse error: %v, body: %s", err, string(respBody))
		return "", err
	}

	if tokenResp.Error == "slow_down" {
		interval := tokenResp.Interval
		if interval == 0 {
			interval = 10
		}
		return "", &errSlowDown{Interval: interval}
	}

	if tokenResp.Error != "" {
		return "", fmt.Errorf("%s", tokenResp.Error)
	}

	return tokenResp.AccessToken, nil
}

func GetSession(sessionID string) *AuthSession {
	v, ok := authSessions.Load(sessionID)
	if !ok {
		return nil
	}
	return v.(*AuthSession)
}

func CleanupSession(sessionID string) {
	authSessions.Delete(sessionID)
}
