package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"copilot-go/auth"
	"copilot-go/config"
	"copilot-go/instance"
	"copilot-go/store"

	"github.com/gin-gonic/gin"
)

type githubUserProfile struct {
	ID    int64  `json:"id"`
	Login string `json:"login"`
}

type authCompletionResult struct {
	Account *store.Account
	Created bool
	Probe   *instance.AccountProbeResult
}

func fetchGithubUserProfile(accessToken string) (*githubUserProfile, error) {
	req, err := http.NewRequest("GET", config.GithubUserURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "token "+accessToken)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", fmt.Sprintf("GitHubCopilotChat/%s", config.CopilotVersion))

	resp, err := config.NewHTTPClient(10 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("github user lookup failed: %d", resp.StatusCode)
	}

	var profile githubUserProfile
	if err := jsonNewDecoder(resp.Body).Decode(&profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

func completeSessionToAccount(sessionID, name, existingAccountID string) (*authCompletionResult, error) {
	session := auth.GetSession(sessionID)
	if session == nil {
		return nil, fmt.Errorf("session not found")
	}
	if session.Status != "completed" || session.AccessToken == "" {
		return nil, fmt.Errorf("auth not completed")
	}

	profile, err := fetchGithubUserProfile(session.AccessToken)
	if err != nil {
		return nil, err
	}

	resolvedName := strings.TrimSpace(name)
	if resolvedName == "" {
		resolvedName = profile.Login
		if resolvedName == "" {
			resolvedName = "GitHub Account"
		}
	}

	account, created, err := store.UpsertAccount(existingAccountID, resolvedName, session.AccessToken, "", profile.Login, profile.ID)
	if err != nil {
		return nil, err
	}

	auth.CleanupSession(sessionID)
	instance.StopInstance(account.ID)

	// Phase 0c: auto-adopt worker via supervisor when configured. Fail-soft —
	// any supervisor error is logged but does not block device-flow completion.
	// The account remains usable via the direct /v1/responses path.
	if sup := instance.DefaultSupervisor(); sup != nil && config.WorkerAutoAdopt() {
		ctx := context.Background()
		if err := sup.RemoveAndCleanup(ctx, account.ID); err != nil {
			log.Printf("supervisor: remove-and-cleanup %s: %v", account.ID, err)
		}
		if _, err := sup.Spawn(ctx, account.ID, session.AccessToken); err != nil {
			log.Printf("supervisor: spawn %s: %v (falling through to direct mode)", account.ID, err)
		} else if fresh, freshErr := store.GetAccount(account.ID); freshErr == nil && fresh != nil {
			// Spawn's PersistOnReady wrote WorkerURL/WorkerHome/etc back to
			// store; refresh in-memory copy so ReconcileAccount sees it.
			account = fresh
		}
	}

	updated, probe, err := instance.ReconcileAccount(*account)
	if err != nil {
		return nil, err
	}
	if updated != nil {
		account = updated
	}

	return &authCompletionResult{Account: account, Created: created, Probe: probe}, nil
}

func publicPathURL(c *gin.Context, path string) string {
	scheme := "http"
	if proto := c.GetHeader("X-Forwarded-Proto"); proto != "" {
		scheme = proto
	} else if c.Request.TLS != nil {
		scheme = "https"
	}
	return fmt.Sprintf("%s://%s%s", scheme, c.Request.Host, path)
}

func publicSessionURL(c *gin.Context, sessionID string) string {
	return publicPathURL(c, fmt.Sprintf("/reauth/%s", sessionID))
}

func publicSupplierAuthURL(c *gin.Context) string {
	return publicPathURL(c, "/supplier-auth")
}

func handleCreatePublicReauthSession(c *gin.Context) {
	accountID := c.Param("id")
	account, err := store.GetAccount(accountID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if account == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "account not found"})
		return
	}

	session, err := store.CreatePublicReauthSession(*account, 24*time.Hour)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"id":         session.ID,
		"sessionUrl": publicSessionURL(c, session.ID),
		"expiresAt":  session.ExpiresAt,
	})
}

func publicSessionResponse(session *store.PublicReauthSession) gin.H {
	return gin.H{
		"id":              session.ID,
		"accountName":     session.AccountName,
		"status":          session.Status,
		"userCode":        session.UserCode,
		"verificationUri": session.VerificationURI,
		"expiresAt":       session.ExpiresAt,
		"completedAt":     session.CompletedAt,
		"error":           session.Error,
	}
}

func loadPublicReauthSession(c *gin.Context) (*store.PublicReauthSession, bool) {
	sessionID := c.Param("sessionId")
	session, err := store.GetPublicReauthSession(sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return nil, false
	}
	if session == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
		return nil, false
	}
	if expiresAt, err := time.Parse(time.RFC3339, session.ExpiresAt); err == nil && time.Now().After(expiresAt) && session.Status != "completed" {
		session.Status = "expired"
		session.Error = "reauth session expired"
		if saveErr := store.UpdatePublicReauthSession(*session); saveErr != nil {
			log.Printf("failed to persist expired reauth session %s: %v", session.ID, saveErr)
		}
	}
	return session, true
}

func handleGetPublicReauthSession(c *gin.Context) {
	session, ok := loadPublicReauthSession(c)
	if !ok {
		return
	}
	c.JSON(http.StatusOK, publicSessionResponse(session))
}

func handleStartPublicReauth(c *gin.Context) {
	session, ok := loadPublicReauthSession(c)
	if !ok {
		return
	}
	if session.Status == "expired" {
		c.JSON(http.StatusBadRequest, gin.H{"error": session.Error})
		return
	}
	if session.Status == "completed" {
		c.JSON(http.StatusOK, publicSessionResponse(session))
		return
	}
	if session.AuthSessionID != "" {
		if authSession := auth.GetSession(session.AuthSessionID); authSession != nil {
			session.UserCode = authSession.UserCode
			session.VerificationURI = authSession.VerificationURI
			session.Status = authSession.Status
			if authSession.Status == "pending" || authSession.Status == "completed" {
				_ = store.UpdatePublicReauthSession(*session)
				c.JSON(http.StatusOK, publicSessionResponse(session))
				return
			}
		}
	}

	authSession, err := auth.StartDeviceFlow()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	session.AuthSessionID = authSession.ID
	session.UserCode = authSession.UserCode
	session.VerificationURI = authSession.VerificationURI
	session.Status = authSession.Status
	session.Error = ""
	if err := store.UpdatePublicReauthSession(*session); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, publicSessionResponse(session))
}

func handlePollPublicReauth(c *gin.Context) {
	session, ok := loadPublicReauthSession(c)
	if !ok {
		return
	}
	if session.Status == "expired" || session.Status == "completed" {
		c.JSON(http.StatusOK, publicSessionResponse(session))
		return
	}
	if session.AuthSessionID == "" {
		c.JSON(http.StatusOK, publicSessionResponse(session))
		return
	}

	authSession := auth.GetSession(session.AuthSessionID)
	if authSession == nil {
		session.Status = "error"
		session.Error = "auth session unavailable"
		_ = store.UpdatePublicReauthSession(*session)
		c.JSON(http.StatusOK, publicSessionResponse(session))
		return
	}

	session.UserCode = authSession.UserCode
	session.VerificationURI = authSession.VerificationURI
	session.Status = authSession.Status
	session.Error = authSession.Error

	if authSession.Status == "completed" && authSession.AccessToken != "" {
		result, err := completeSessionToAccount(authSession.ID, session.AccountName, session.AccountID)
		if err != nil {
			session.Status = "error"
			session.Error = err.Error()
		} else if result.Probe != nil && !result.Probe.Success {
			session.Status = "error"
			session.Error = result.Probe.Error
			if result.Account != nil {
				session.AccountName = result.Account.Name
			}
		} else {
			session.Status = "completed"
			session.CompletedAt = time.Now().UTC().Format(time.RFC3339)
			session.Error = ""
			if result.Account != nil {
				session.AccountName = result.Account.Name
			}
		}
	}

	if authSession.Status == "expired" && session.Error == "" {
		session.Error = "device code expired"
	}
	if authSession.Status == "error" && session.Error == "" {
		session.Error = authSession.Error
	}

	if err := store.UpdatePublicReauthSession(*session); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, publicSessionResponse(session))
}

func handleStartPublicAuth(c *gin.Context) {
	authSession, err := auth.StartDeviceFlow()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"sessionId":       authSession.ID,
		"status":          authSession.Status,
		"userCode":        authSession.UserCode,
		"verificationUri": authSession.VerificationURI,
		"expiresAt":       authSession.ExpiresAt,
	})
}

func handlePollPublicAuth(c *gin.Context) {
	sessionID := c.Param("sessionId")
	authSession := auth.GetSession(sessionID)
	if authSession == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"sessionId":       authSession.ID,
		"status":          authSession.Status,
		"userCode":        authSession.UserCode,
		"verificationUri": authSession.VerificationURI,
		"expiresAt":       authSession.ExpiresAt,
		"error":           authSession.Error,
	})
}

func handleCompletePublicAuth(c *gin.Context) {
	var body struct {
		SessionID string `json:"sessionId"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
		return
	}

	result, err := completeSessionToAccount(body.SessionID, "", "")
	if err != nil {
		if err.Error() == "session not found" {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		if err.Error() == "auth not completed" {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	status := http.StatusOK
	if result.Created {
		status = http.StatusCreated
	}
	c.JSON(status, gin.H{
		"account": result.Account,
		"created": result.Created,
		"probe":   result.Probe,
	})
}

var jsonNewDecoder = func(r io.Reader) *json.Decoder {
	return json.NewDecoder(r)
}
