#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-macaron-dev}"
GRAY_UNIT_NAME="${GRAY_UNIT_NAME:-copilot-go-gray.service}"

ssh "${REMOTE}" "systemctl --user stop '${GRAY_UNIT_NAME}' >/dev/null 2>&1 || true; systemctl --user reset-failed '${GRAY_UNIT_NAME}' >/dev/null 2>&1 || true"
echo "stopped ${GRAY_UNIT_NAME} on ${REMOTE}"
