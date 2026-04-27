#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-macaron-dev}"
GRAY_UNIT_NAME="${GRAY_UNIT_NAME:-copilot-go-gray.service}"
GRAY_BINARY_PATH="${GRAY_BINARY_PATH:-/home/dev/services/copilot2api-go-gray/copilot-go}"
GRAY_PROXY_PORT="${GRAY_PROXY_PORT:-39100}"
GRAY_WEB_PORT="${GRAY_WEB_PORT:-39101}"
LIVE_UNIT_NAME="${LIVE_UNIT_NAME:-copilot-go.service}"
LIVE_BINARY_PATH="${LIVE_BINARY_PATH:-/home/dev/services/copilot2api-go/copilot-go}"
LIVE_PROXY_PORT="${LIVE_PROXY_PORT:-38000}"
LIVE_WEB_PORT="${LIVE_WEB_PORT:-3000}"
TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"

ssh "${REMOTE}" bash -s -- \
  "${GRAY_UNIT_NAME}" \
  "${GRAY_BINARY_PATH}" \
  "${GRAY_PROXY_PORT}" \
  "${GRAY_WEB_PORT}" \
  "${LIVE_UNIT_NAME}" \
  "${LIVE_BINARY_PATH}" \
  "${LIVE_PROXY_PORT}" \
  "${LIVE_WEB_PORT}" \
  "${TAG}" <<'EOF'
set -euo pipefail

gray_unit_name="$1"
gray_binary_path="$2"
gray_proxy_port="$3"
gray_web_port="$4"
live_unit_name="$5"
live_binary_path="$6"
live_proxy_port="$7"
live_web_port="$8"
tag="$9"

backup_path="${live_binary_path}.backup-${tag}"

gray_ready() {
  local config_json
  config_json="$(curl -fsS "http://127.0.0.1:${gray_web_port}/api/config" 2>/dev/null || true)"
  [[ "${config_json}" == *"\"proxyPort\":${gray_proxy_port}"* ]]
}

live_ready() {
  local config_json
  config_json="$(curl -fsS "http://127.0.0.1:${live_web_port}/api/config" 2>/dev/null || true)"
  [[ "${config_json}" == *"\"proxyPort\":${live_proxy_port}"* ]]
}

systemctl --user is-active "${gray_unit_name}" >/dev/null
gray_ready
ss -ltnp 2>/dev/null | grep -E ":${gray_proxy_port}\b|:${gray_web_port}\b" >/dev/null

cp "${live_binary_path}" "${backup_path}"
install -m 755 "${gray_binary_path}" "${live_binary_path}"
systemctl --user restart "${live_unit_name}"

for _ in $(seq 1 30); do
  if systemctl --user is-active "${live_unit_name}" >/dev/null \
    && live_ready \
    && ss -ltnp 2>/dev/null | grep -E ":${live_proxy_port}\b|:${live_web_port}\b" >/dev/null; then
    printf 'live_ready unit=%s proxy=%s web=%s backup=%s\n' \
      "${live_unit_name}" "${live_proxy_port}" "${live_web_port}" "${backup_path}"
    exit 0
  fi
  sleep 1
done

echo "live promotion failed" >&2
systemctl --user status "${live_unit_name}" --no-pager >&2 || true
exit 1
EOF

echo "promoted gray candidate on ${REMOTE}; backup saved with tag ${TAG}"
