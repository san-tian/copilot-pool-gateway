package instance

import (
	"strings"

	"github.com/gin-gonic/gin"
)

const (
	responseHeaderCopilotAccountID    = "X-Copilot-Account-Id"
	responseHeaderCopilotPoolStrategy = "X-Copilot-Pool-Strategy"
)

func applyRoutingResponseHeaders(c *gin.Context, accountID string) {
	if c == nil {
		return
	}
	if accountID = strings.TrimSpace(accountID); accountID != "" {
		c.Header(responseHeaderCopilotAccountID, accountID)
	}
	if poolStrategy := strings.TrimSpace(c.GetString("poolStrategy")); poolStrategy != "" {
		c.Header(responseHeaderCopilotPoolStrategy, poolStrategy)
	}
}
