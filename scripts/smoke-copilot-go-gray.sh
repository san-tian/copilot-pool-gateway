#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-macaron-dev}"
GRAY_UNIT_NAME="${GRAY_UNIT_NAME:-copilot-go-gray.service}"
GRAY_PROXY_PORT="${GRAY_PROXY_PORT:-39100}"
GRAY_WEB_PORT="${GRAY_WEB_PORT:-39101}"
GRAY_APP_DIR="${GRAY_APP_DIR:-/home/dev/.local/share/copilot-api-gray}"
ORPHAN_MODEL="${ORPHAN_MODEL:-gpt-5.4}"
ORPHAN_TEXT="${ORPHAN_TEXT:-Return the single word OK.}"

ssh "${REMOTE}" bash -s -- \
  "${GRAY_UNIT_NAME}" \
  "${GRAY_PROXY_PORT}" \
  "${GRAY_WEB_PORT}" \
  "${GRAY_APP_DIR}" \
  "${ORPHAN_MODEL}" \
  "${ORPHAN_TEXT}" <<'EOF'
set -euo pipefail

gray_unit_name="$1"
gray_proxy_port="$2"
gray_web_port="$3"
gray_app_dir="$4"
orphan_model="$5"
orphan_text="$6"

systemctl --user is-active "${gray_unit_name}" >/dev/null
ss -ltnp 2>/dev/null | grep -E ":${gray_proxy_port}\b|:${gray_web_port}\b" >/dev/null
config_json="$(curl -fsS "http://127.0.0.1:${gray_web_port}/api/config")"
if [[ "${config_json}" != *"\"proxyPort\":${gray_proxy_port}"* ]]; then
  echo "gray web probe did not identify the expected candidate: ${config_json}" >&2
  exit 1
fi

python3 - "${gray_proxy_port}" "${gray_app_dir}" "${orphan_model}" "${orphan_text}" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

proxy_port = int(sys.argv[1])
app_dir = pathlib.Path(sys.argv[2])
orphan_model = sys.argv[3]
orphan_text = sys.argv[4]

data = json.loads((app_dir / "accounts.json").read_text())
accounts = data.get("accounts", []) if isinstance(data, dict) else []

enabled = [a for a in accounts if isinstance(a, dict) and a.get("enabled") and a.get("apiKey")]
worker_no = next(
    (
        a for a in enabled
        if not str(a.get("workerUrl", "") or "").strip()
        and orphan_model in set(a.get("supportedModels") or [])
    ),
    None,
)
worker_yes = next(
    (
        a for a in enabled
        if str(a.get("workerUrl", "") or "").strip()
        and orphan_model in set(a.get("supportedModels") or [])
    ),
    None,
)

if worker_no is None:
    raise SystemExit(f"missing enabled worker=no account with {orphan_model} support in {app_dir}/accounts.json")

def request(path, api_key, body=None, timeout=60):
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}{path}",
        data=body,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()

def wait_models(account):
    deadline = time.time() + 30
    last = None
    while time.time() < deadline:
        status, headers, body = request("/v1/models", account["apiKey"], body=None, timeout=20)
        last = (status, headers, body)
        if status == 200:
            return last
        time.sleep(2)
    raise SystemExit(
        f"/v1/models never became ready for {account['name']} ({account['id']}): "
        f"last_status={last[0]} last_body={last[2][:300].decode('utf-8', 'replace')}"
    )

for sample in [worker_yes, worker_no]:
    if sample is None:
        continue
    status, _, _ = wait_models(sample)
    print(f"models_ok name={sample['name']} worker={'yes' if str(sample.get('workerUrl', '') or '').strip() else 'no'} status={status}")

call_id = f"call_gray_smoke_{int(time.time())}"
payload = {
    "model": orphan_model,
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": orphan_text}]},
        {"type": "function_call", "call_id": call_id, "name": "noop", "arguments": "{}"},
        {"type": "function_call_output", "call_id": call_id, "output": "ok"},
    ],
    "stream": True,
}
status, headers, body = request("/v1/responses", worker_no["apiKey"], body=json.dumps(payload).encode(), timeout=60)
content_type = headers.get("Content-Type", "")
snippet = body[:4000].decode("utf-8", "replace")
print(f"orphan_sample account={worker_no['name']} status={status} ct={content_type}")
print(snippet)

if status != 200:
    raise SystemExit(f"expected 200 from orphan sample, got {status}")
if "text/event-stream" not in content_type:
    raise SystemExit(f"expected text/event-stream, got {content_type}")
if "event: response.completed" not in snippet:
    raise SystemExit("expected response.completed in orphan sample output")
PY
EOF

echo "gray smoke passed on ${REMOTE}:${GRAY_PROXY_PORT}"
