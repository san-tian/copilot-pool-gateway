#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_SRC=$(cd -- "${SCRIPT_DIR}/.." && pwd)

REMOTE="${REMOTE:-macaron-dev}"
REMOTE_SRC="${REMOTE_SRC:-/home/dev/src/copilot2api-go}"
GRAY_RUNTIME_DIR="${GRAY_RUNTIME_DIR:-/home/dev/services/copilot2api-go-gray}"
GRAY_APP_DIR="${GRAY_APP_DIR:-/home/dev/.local/share/copilot-api-gray}"
GRAY_WORKERS_ROOT="${GRAY_WORKERS_ROOT:-${GRAY_APP_DIR}/workers}"
GRAY_BINARY_PATH="${GRAY_BINARY_PATH:-${GRAY_RUNTIME_DIR}/copilot-go}"
GRAY_UNIT_NAME="${GRAY_UNIT_NAME:-copilot-go-gray.service}"
GRAY_PROXY_PORT="${GRAY_PROXY_PORT:-39100}"
GRAY_WEB_PORT="${GRAY_WEB_PORT:-39101}"
GRAY_WORKER_PORT_RANGE="${GRAY_WORKER_PORT_RANGE:-9390-9399}"
GRAY_WORKER_EXE="${GRAY_WORKER_EXE:-__disabled__}"
GRAY_WORKER_ARGS="${GRAY_WORKER_ARGS:-}"
GRAY_USE_WORKER_POOL="${GRAY_USE_WORKER_POOL:-off}"
GRAY_WORKER_AUTO_ADOPT="${GRAY_WORKER_AUTO_ADOPT:-}"
GRAY_WORKER_MIGRATE_LEGACY="${GRAY_WORKER_MIGRATE_LEGACY:-}"
GRAY_READY_RETRIES="${GRAY_READY_RETRIES:-90}"
TAG="${TAG:-$(date +%Y%m%d%H%M%S)}"
SKIP_LOCAL_CHECKS="${SKIP_LOCAL_CHECKS:-0}"

if [[ "${SKIP_LOCAL_CHECKS}" != "1" ]]; then
  (
    cd "${LOCAL_SRC}"
    go test ./...
    go build ./...
  )
fi

rsync -av --delete \
  --exclude .git \
  --exclude '*.log' \
  "${LOCAL_SRC}/" \
  "${REMOTE}:${REMOTE_SRC}/"

ssh "${REMOTE}" bash -s -- \
  "${REMOTE_SRC}" \
  "${GRAY_RUNTIME_DIR}" \
  "${GRAY_APP_DIR}" \
  "${GRAY_WORKERS_ROOT}" \
  "${GRAY_BINARY_PATH}" \
  "${GRAY_UNIT_NAME}" \
  "${GRAY_PROXY_PORT}" \
  "${GRAY_WEB_PORT}" \
  "${GRAY_WORKER_PORT_RANGE}" \
  "${GRAY_WORKER_EXE}" \
  "${GRAY_WORKER_ARGS}" \
  "${GRAY_USE_WORKER_POOL}" \
  "${GRAY_WORKER_AUTO_ADOPT}" \
  "${GRAY_WORKER_MIGRATE_LEGACY}" \
  "${GRAY_READY_RETRIES}" \
  "${TAG}" <<'EOF'
set -euo pipefail

remote_src="$1"
gray_runtime_dir="$2"
gray_app_dir="$3"
gray_workers_root="$4"
gray_binary_path="$5"
gray_unit_name="$6"
gray_proxy_port="$7"
gray_web_port="$8"
gray_worker_port_range="$9"
gray_worker_exe="${10}"
gray_worker_args="${11}"
gray_use_worker_pool="${12}"
gray_worker_auto_adopt="${13}"
gray_worker_migrate_legacy="${14}"
gray_ready_retries="${15}"
tag="${16}"

tmp_binary="/tmp/copilot-go-gray-${tag}"
live_app_dir="${HOME}/.local/share/copilot-api"

port_is_in_use() {
  local port="$1"
  ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

assert_port_free() {
  local port="$1"
  local label="$2"
  if port_is_in_use "${port}"; then
    echo "gray ${label} port ${port} is already in use" >&2
    ss -ltnp 2>/dev/null | grep -E ":${port}\b" >&2 || true
    exit 1
  fi
}

gray_ready() {
  local config_json
  config_json="$(curl -fsS "http://127.0.0.1:${gray_web_port}/api/config" 2>/dev/null || true)"
  [[ "${config_json}" == *"\"proxyPort\":${gray_proxy_port}"* ]]
}

install -d "${gray_runtime_dir}" "${gray_app_dir}"
rsync -a --delete --exclude 'workers/' "${live_app_dir}/" "${gray_app_dir}/"
rm -rf "${gray_workers_root}"
mkdir -p "${gray_workers_root}"

cd "${remote_src}"
/usr/local/bin/go build -o "${tmp_binary}" .
install -m 755 "${tmp_binary}" "${gray_binary_path}"
rm -f "${tmp_binary}"

systemctl --user stop "${gray_unit_name}" 2>/dev/null || true
systemctl --user reset-failed "${gray_unit_name}" 2>/dev/null || true

assert_port_free "${gray_proxy_port}" "proxy"
assert_port_free "${gray_web_port}" "web"

cmd=(
  systemd-run --user --unit "${gray_unit_name}"
  --property=WorkingDirectory="${gray_runtime_dir}"
  --property=Restart=on-failure
  --property=RestartSec=3
  --property=Environment="COPILOT_API_APP_DIR=${gray_app_dir}"
  --property=Environment="COPILOT_WORKERS_HOME=${gray_workers_root}"
  --property=Environment="COPILOT_WORKER_PORT_RANGE=${gray_worker_port_range}"
  --property=Environment="COPILOT_WORKER_EXE=${gray_worker_exe}"
  --property=Environment="USE_WORKER_POOL=${gray_use_worker_pool}"
  --property=Environment="RESPONSES_ORPHAN_TRANSLATE=on"
)
if [[ -n "${gray_worker_args}" ]]; then
  cmd+=(--property=Environment="COPILOT_WORKER_ARGS=${gray_worker_args}")
fi
if [[ -n "${gray_worker_auto_adopt}" ]]; then
  cmd+=(--property=Environment="COPILOT_WORKER_AUTO_ADOPT=${gray_worker_auto_adopt}")
fi
if [[ -n "${gray_worker_migrate_legacy}" ]]; then
  cmd+=(--property=Environment="COPILOT_WORKER_MIGRATE_LEGACY=${gray_worker_migrate_legacy}")
fi
cmd+=("${gray_binary_path}" --web-port "${gray_web_port}" --proxy-port "${gray_proxy_port}")
"${cmd[@]}" >/dev/null

for _ in $(seq 1 "${gray_ready_retries}"); do
  if gray_ready; then
    systemctl --user is-active "${gray_unit_name}" >/dev/null
    ss -ltnp 2>/dev/null | grep -E ":${gray_proxy_port}\b|:${gray_web_port}\b" >/dev/null
    printf 'gray_ready unit=%s proxy=%s web=%s app_dir=%s runtime_dir=%s\n' \
      "${gray_unit_name}" "${gray_proxy_port}" "${gray_web_port}" "${gray_app_dir}" "${gray_runtime_dir}"
    exit 0
  fi
  sleep 1
done

echo "gray candidate failed to become ready" >&2
systemctl --user status "${gray_unit_name}" --no-pager >&2 || true
exit 1
EOF

cat <<MSG
Candidate deployed on ${REMOTE}.
  Unit: ${GRAY_UNIT_NAME}
  Proxy: http://127.0.0.1:${GRAY_PROXY_PORT}
  Web:   http://127.0.0.1:${GRAY_WEB_PORT}
  AppDir: ${GRAY_APP_DIR}

Next:
  $(basename "$0" | sed 's/deploy/smoke/') 
MSG
