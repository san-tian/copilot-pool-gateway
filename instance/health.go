package instance

import (
	"log"

	"copilot-go/store"
)

func SetAccountError(account store.Account, reason string) {
	mu.Lock()
	defer mu.Unlock()

	if inst, ok := instances[account.ID]; ok {
		if inst.Status == "running" && inst.stopChan != nil {
			close(inst.stopChan)
			inst.stopChan = nil
		}
		inst.Account = account
		inst.Status = "error"
		inst.Error = reason
		return
	}

	instances[account.ID] = &ProxyInstance{
		Account: account,
		Status:  "error",
		Error:   reason,
	}
}

func DisableAccount(accountID, reason string) error {
	account, err := store.UpdateAccount(accountID, map[string]interface{}{"enabled": false})
	if err != nil {
		return err
	}
	if account != nil {
		SetAccountError(*account, reason)
	}

	name := accountID
	if account != nil && account.Name != "" {
		name = account.Name
	}
	log.Printf("Account %s disabled automatically: %s", name, reason)
	return nil
}
