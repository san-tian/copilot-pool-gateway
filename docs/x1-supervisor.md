# X1 Worker Supervisor — Design Doc

**Status:** Draft · 2026-04-23
**Owner:** san-tian
**Baseline:** tag `pre-x1-supervisor` (commit `5cf2833`)

## 1. 目标与范围

让 gateway 在 UI 点击「登录 GitHub」即可完成新账号全链路上线：device flow → 写 token → 拉起独立 `copilot-api` 子进程 → 自动填 `WorkerURL`。**不改 copilot-api 代码**，全部协议知识由 copilot-api 承担；gateway 只做 GitHub OAuth + 子进程生命周期 + 会话编排。

### 1.1 In scope
- 每账号一个 `copilot-api` 子进程（X1 路线）
- 子进程生命周期：spawn / health check / restart / kill / cleanup
- 端口分配与冲突避免
- `github_token` 预写到 api-home，复用 copilot-api `setupGitHubToken()` 文件读取路径
- gateway 重启时从 DB 恢复所有 `Enabled=true` 账号的子进程
- admin console 删除账号时连带清理子进程与 api-home 目录
- 观测：每子进程日志独立落盘；`/api/accounts/:id` 返回供给 UI 展示的 supervisor 状态

### 1.2 Non-goals（本阶段不做）
- 删除 `anthropic/` 翻译器、删除 direct `/v1/chat/completions` 等（属于后续阶段 1–4）
- 多机部署（workers-root 跨主机）
- 子进程资源隔离（cgroup/namespace）
- 端口复用（每账号独占一个端口，不做多账号共享端口）
- 替换 copilot-api 的 token refresh loop（由子进程自己跑）

## 2. 前置依据

- **copilot-api 支持无交互启动**（验证 2026-04-23）：`src/lib/token.ts:128-142` 若 `$COPILOT_API_HOME/github_token` 文件非空则直接加载，跳过 device flow
- **copilot-api 路径解析**：`src/lib/paths.ts` 用 `COPILOT_API_HOME` env 决定 APP_DIR
- **gateway device flow 已就绪**：`auth/device_flow.go:StartDeviceFlow()` 返回 `*AuthSession`，后台 goroutine `pollForToken` 完成后把 `AccessToken` 填入 session
- **gateway 已具备 public reauth 闭环**：`handler/public_reauth.go:completeSessionToAccount` 拿到 token → `store.UpsertAccount` → `instance.ReconcileAccount`
- **Account 数据模型有 `WorkerURL` 字段**：`store/account.go:30` — 当前手填，X1 下由 supervisor 自动填

## 3. 架构总览

```
┌──────────────────────── gateway process ────────────────────────┐
│                                                                 │
│  handler/public_reauth.go                                       │
│    completeSessionToAccount()                                   │
│       │                                                         │
│       ▼                                                         │
│  instance/worker_supervisor.go  (NEW)                           │
│    ├─ Spawn(accountID, githubToken) → workerURL                 │
│    ├─ Stop(accountID)                                           │
│    ├─ Restart(accountID)                                        │
│    ├─ HealthLoop()  ←── periodic probe on WorkerURL/models      │
│    └─ RecoverFromStore()  ←── called on gateway startup         │
│                                                                 │
│       │ exec.Cmd with env COPILOT_API_HOME=<per-account-dir>    │
│       ▼                                                         │
│  os-level child: copilot-api --port <p> start                   │
│                                                                 │
│  handler/proxy.go  (UNCHANGED in this phase)                    │
│    resolves account → reads account.WorkerURL → ProxyRequestViaWorker
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Per-account on disk:
  $WORKERS_ROOT/<account-id>/
    ├─ github_token          ← gho_XXX (0600)
    ├─ config.json           ← copilot-api 自己写
    └─ copilot-api.log       ← stdout + stderr
```

## 4. 数据模型

### 4.1 `store.Account` 新增字段（向后兼容 omitempty）

```go
// store/account.go
type Account struct {
    // ... 现有字段保持不变 ...

    // Supervisor-managed fields (X1). When the supervisor manages this
    // account, WorkerURL is auto-filled by Spawn and SHOULD NOT be edited
    // by hand through the admin console.
    WorkerManaged bool   `json:"workerManaged,omitempty"` // true ⇒ supervisor owns WorkerURL
    WorkerPort    int    `json:"workerPort,omitempty"`    // assigned port (for restart after gateway restart)
    WorkerPID     int    `json:"workerPid,omitempty"`     // last-known child pid (diagnostic only; authoritative state is in supervisor memory)
    WorkerHome    string `json:"workerHome,omitempty"`    // absolute path to api-home (derived from account id, stored for explicit cleanup)
}
```

**迁移**：零迁移。老账号字段为零值，`WorkerManaged=false`，走原有手填 `WorkerURL` 路径，行为不变。用户可在 UI 上对老账号触发「迁移到 supervisor」动作（后续 UI 任务）。

### 4.2 `store/paths.go` 新增

```go
func WorkersRoot() string {
    if v := strings.TrimSpace(os.Getenv("COPILOT_WORKERS_HOME")); v != "" {
        return v
    }
    return filepath.Join(AppDir(), "workers")
}

func WorkerHomeFor(accountID string) string {
    return filepath.Join(WorkersRoot(), accountID)
}
```

默认 `~/.local/share/copilot-api/workers/<account-id>/`，与现有 `AppDir()` 同根。

### 4.3 端口分配

- Range 由 env 决定：`COPILOT_WORKER_PORT_RANGE=9100-9199`（默认）
- 分配策略：supervisor 启动时扫描已知 `WorkerPort`，池化剩余端口。新账号 `Spawn` 时从池里取第一个空闲端口，分配后立刻 `UpdateAccount` 持久化，避免重复
- 冲突处理：绑 port 失败 → 从池里移除、重试下一个；连续失败 3 次 → 记 `WorkerStatus="port_exhausted"` 返回 UI

## 5. 组件

### 5.1 `instance/worker_supervisor.go`（新文件）

核心 API：

```go
type WorkerSupervisor struct {
    mu       sync.Mutex
    workers  map[string]*workerEntry  // accountID → entry
    ports    *portPool
    exePath  string                   // resolved copilot-api binary path
    stopCh   chan struct{}
}

type workerEntry struct {
    AccountID string
    Port      int
    Home      string
    Cmd       *exec.Cmd
    LogFile   *os.File
    StartedAt time.Time
    Status    string   // "starting" | "ready" | "crashed" | "stopping"
    LastError string
}

func NewSupervisor() *WorkerSupervisor

// Lifecycle
func (s *WorkerSupervisor) Spawn(accountID, githubToken string) (workerURL string, err error)
func (s *WorkerSupervisor) Stop(accountID string) error
func (s *WorkerSupervisor) Restart(accountID string) error
func (s *WorkerSupervisor) RemoveAndCleanup(accountID string) error  // Stop + rm -rf home

// Introspection
func (s *WorkerSupervisor) Status(accountID string) (workerEntry, bool)
func (s *WorkerSupervisor) All() []workerEntry

// Background
func (s *WorkerSupervisor) RecoverFromStore(ctx context.Context)   // on gateway startup
func (s *WorkerSupervisor) HealthLoop(ctx context.Context)         // periodic /models probe
func (s *WorkerSupervisor) Shutdown(ctx context.Context)           // on gateway SIGTERM
```

#### 5.1.1 `Spawn` 实现要点

```go
// 1. mkdir home, write github_token
os.MkdirAll(home, 0700)
os.WriteFile(filepath.Join(home, "github_token"), []byte(githubToken), 0600)

// 2. alloc port
port := s.ports.Alloc()

// 3. build command
cmd := exec.Command(s.exePath, "--port", strconv.Itoa(port), "start")
cmd.Env = append(os.Environ(), "COPILOT_API_HOME="+home)
cmd.Stdout = logFile
cmd.Stderr = logFile
cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // 让子进程独立 pgid，便于 SIGTERM 整组

// 4. start, register
cmd.Start()

// 5. persist to store
store.UpdateAccountWorker(accountID, "http://127.0.0.1:"+port, port, cmd.Process.Pid, home)

// 6. wait for readiness: poll http://127.0.0.1:<port>/models with timeout
//    — copilot-api 需要先 setupCopilotToken（~1-2 秒），/models 返回 200 才算 ready
waitForReady(ctx, "http://127.0.0.1:"+port+"/models", 20*time.Second)

// 7. spawn a goroutine that cmd.Wait()-s and transitions to "crashed" on exit
go s.reapLoop(accountID, cmd)
```

#### 5.1.2 `reapLoop`

```go
func (s *WorkerSupervisor) reapLoop(accountID string, cmd *exec.Cmd) {
    err := cmd.Wait()
    // child exited — update status, trigger HealthLoop to decide restart
    s.markCrashed(accountID, err)
    // Backoff restart: 5s, 30s, 2m, 5m (capped). Reset counter after 1 min stable.
}
```

#### 5.1.3 `waitForReady`

copilot-api 启动后需要 `setupGitHubToken` → `setupCopilotToken` 才能处理请求。用 `/models` 做就绪探测（无状态 GET）。超时 20s 给足 token 交换的空间；未 ready 则视为启动失败，`Spawn` 返回 error。

#### 5.1.4 `RecoverFromStore`

启动流程（插入 `main.go` 现有 `autoStart` 块前）：

```go
accounts := store.GetEnabledAccounts()
for _, a := range accounts {
    if !a.WorkerManaged { continue }      // 手填 WorkerURL 的老账号跳过
    if a.GithubToken == "" { mark degraded; continue }
    // 端口使用账号持久化的 WorkerPort 以避免冲突
    s.ports.Reserve(a.WorkerPort)
    go s.spawnWithExistingPort(a)         // 错开并发启动
}
```

### 5.2 `store/account.go` 新增

```go
// UpdateAccountWorker persists the supervisor-owned fields atomically.
// Called by Spawn; NOT exposed via admin console API.
func UpdateAccountWorker(accountID, workerURL string, port, pid int, home string) error

// ClearAccountWorker resets supervisor fields; called on Stop / RemoveAndCleanup.
func ClearAccountWorker(accountID string) error
```

### 5.3 `handler/public_reauth.go` 改动

`completeSessionToAccount` 的现有顺序：

```
UpsertAccount → StopInstance → ReconcileAccount
```

改为：

```
UpsertAccount →
  if WorkerManaged-by-default-for-new-accounts (config flag) {
    StopInstance
    supervisor.RemoveAndCleanup(oldIfAny)
    workerURL, err := supervisor.Spawn(account.ID, token)
    store.UpdateAccountWorker(...)   // 已由 Spawn 内部完成
  }
  ReconcileAccount  ← 现有 probe/识别流程不变
```

关键：**`ReconcileAccount` 的调用保持**，它负责 probe / 模型识别 / status 展示。supervisor 只负责子进程存在，不侵入账号健康语义。

### 5.4 `handler/console_api.go` 改动

- 删除账号 (`DELETE /api/accounts/:id`) 时额外调 `supervisor.RemoveAndCleanup`
- 新增 `POST /api/accounts/:id/worker/restart` — 手动重启子进程（运维口子）
- `GET /api/accounts/:id` 返回 payload 追加 `workerStatus` / `workerPid` / `workerStartedAt` 字段供 UI 展示

### 5.5 `main.go` 改动

```go
// 现有 autoStart 块前插入：
sup := instance.NewSupervisor()
sup.RecoverFromStore(ctx)
go sup.HealthLoop(ctx)
instance.SetDefaultSupervisor(sup)     // 让其他包按需拿到全局实例

// 现有 go func StartInstance() 不变

// SIGTERM 处理追加 sup.Shutdown(shutdownCtx)
```

## 6. 配置

新增 env：

| Name | Default | 说明 |
|---|---|---|
| `COPILOT_WORKERS_HOME` | `$COPILOT_API_APP_DIR/workers` | 每账号 api-home 根目录 |
| `COPILOT_WORKER_PORT_RANGE` | `9100-9199` | 端口池范围 |
| `COPILOT_WORKER_EXE` | `copilot-api`（从 PATH 找） | 指定 copilot-api 可执行路径 |
| `COPILOT_WORKER_AUTO_ADOPT` | `true` | 新创建账号默认 `WorkerManaged=true` |
| `COPILOT_WORKER_HEALTH_INTERVAL` | `30s` | HealthLoop 轮询间隔 |
| `COPILOT_WORKER_READY_TIMEOUT` | `20s` | Spawn 后 `/models` 就绪超时 |

全部可选；不设时用默认值，与现有部署语义兼容。

## 7. 失败模式与观测

| 场景 | 行为 |
|---|---|
| 子进程启动超时（`/models` 20s 内不 ready） | 记 `WorkerStatus="start_timeout"`，kill 子进程，端口归还池子，向调用方返回 error，账号入库但 `WorkerURL=""` |
| 子进程运行中退出 | `reapLoop` 捕获 `cmd.Wait()`，按 backoff 重启；3 次失败后标 `WorkerStatus="crashed"` 停止自动重启，等运维干预 |
| 端口耗尽 | `Spawn` 返回 `ErrPortExhausted`；UI 展示「端口池已满」，指导扩 range |
| `github_token` 被 copilot-api 拒绝（device flow 过期或 revoke） | HealthLoop 捕获 `/models` 401 → 标 `WorkerStatus="token_invalid"`，不自动重启，由 public reauth 触发刷新 |
| gateway 崩溃 | 子进程变孤儿进程继续服务；gateway 重启后 `RecoverFromStore` 重新认领 (pid 能对上就不 respawn，对不上 kill 后 respawn) |
| copilot-api 可执行不在 PATH | 启动时 `exec.LookPath` 校验失败，`NewSupervisor` 返回 error，gateway 不启动（fail-fast） |

每子进程独立日志：`$WORKERS_ROOT/<account-id>/copilot-api.log`，带 `lumberjack` 做按大小 rotate（50MB × 3 份）。

## 8. 现有优化的保留清单 ⚠️

这节是本次改动的**硬约束**。每项都要有对应的回归测试或手工验证点。

| 模块 | 位置 | 为什么不受 supervisor 影响 |
|---|---|---|
| Session state machine（`previous_response_id` 展开、canonical binding 等） | `handler/proxy.go` + `instance/continuation_*.go` | 生命周期完全在 gateway 内，supervisor 不改请求路径 |
| 4-kind continuation classification | `handler/proxy.go:continuationBinding*` | 同上 |
| sticky caches | `instance/continuation_persistence.go` | 独立于 worker 层 |
| cross-account rollover | `handler/proxy.go:crossAccountRollover` | 基于账号池调度，WorkerURL 填自动填手动无差别 |
| orphan_translate flatten | `instance/orphan_translate/input.go` + `messages_input.go` | 进 gateway 内 in-process 翻译，不跨进程 |
| `call_syn_` synthetic id 铸造 | `instance/orphan_translate/messages_output.go` | 同上 |
| replay-invalid in-loop fallback + pre-loop Split/Unavailable fallback | `handler/proxy.go:1337 / 1530` | 同上 |
| continuation persistence (async+debounced) | `instance/continuation_persistence.go` | 独立快照线程，supervisor 不干预 |
| `/v1/messages` native path via `anthropic/` translator | `instance/handler.go:168+` via `ProxyRequestWithBytesCtx` | **本阶段保持直连**，supervisor 管的是 `/v1/responses` 的 WorkerURL 路径；`/v1/messages` 继续走 direct，`anthropic/` 1002 行不删 |
| admin console + 一键加账号 device flow | `handler/public_reauth.go` + `handler/console_api.go` | device flow 语义不变，只是 `completeSessionToAccount` 内部在 `ReconcileAccount` 前插入 supervisor.Spawn |
| ReconcileAccount / ProbeAccount / 账号识别 | `instance/reconcile.go` + `instance/account_probe.go` | 仍然直连打 `api.githubcopilot.com/models`，**保持不变**；后续阶段再决定是否改走 worker |
| Token refresh (`refreshCopilotToken`) | `instance/manager.go:202` | 老账号（WorkerManaged=false）继续用；WorkerManaged=true 账号的 `copilot_token` 改由子进程持有，gateway 不刷新 |

**硬性验证步骤**（部署前必跑）：
1. `go test ./... -count=1` 全绿
2. 对 WorkerManaged=false 的现有账号发一次 `/v1/responses` 续对话 → 命中 canonical binding → 成功；对同账号构造 split/orphan case → orphan_translate flatten 走通
3. 对 WorkerManaged=true 的新账号发一次 `/v1/responses` → 走 WorkerURL → 子进程 `copilot-api.log` 有对应记录
4. kill -9 子进程 → 1 分钟内 HealthLoop 捕获并 respawn（观察 gateway log）
5. 重启 gateway → `RecoverFromStore` 对所有 WorkerManaged 账号重建子进程，无端口冲突

## 9. 分阶段实施计划

| 阶段 | 内容 | 交付 |
|---|---|---|
| **0a** | supervisor 骨架：Spawn / Stop / waitForReady / 端口池 | `instance/worker_supervisor.go` + 单测 |
| **0b** | Account 字段 + store API + paths | `store/account.go` 扩字段；`store/paths.go` 加 `WorkersRoot` |
| **0c** | 接 device flow + 删账号清理 | 改 `handler/public_reauth.go` + `handler/console_api.go` |
| **0d** | `RecoverFromStore` + `HealthLoop` + `Shutdown` | 改 `main.go` |
| **0e** | admin UI 状态展示（WorkerPID / WorkerStatus / 手动 restart 按钮） | `web/` 改动 |
| **0f** | 文档 + 迁移指南：现有账号如何打开 `WorkerManaged` | `docs/x1-supervisor-migration.md` |

每个子阶段一个独立 commit，独立部署，可独立回滚。

## 10. 验收标准

- [ ] `go test ./... -count=1` 通过
- [ ] 新建账号：UI 点登录 → 2 分钟内出现 `WorkerURL=http://127.0.0.1:91xx`，发请求成功
- [ ] 删除账号：子进程 SIGTERM 内退出 + api-home 目录清理（或保留并打 tombstone，视策略）
- [ ] 重启 gateway：`RecoverFromStore` 让所有 `WorkerManaged=true` 账号在 30s 内重新 ready
- [ ] 杀子进程：60s 内自动 respawn
- [ ] 端口耗尽：返回 `ErrPortExhausted` 且 UI 可读
- [ ] 现有 `WorkerManaged=false` 账号的所有优化（见 §8）行为不变，live log grep 0 回归

## 11. 回滚路径

- **单阶段回滚**：`git revert <stage-commit>`，重新 build & deploy
- **全回滚**：`git reset --hard pre-x1-supervisor` + `git push --force-with-lease pool-gateway master`（仅在整条路线放弃时使用）
- **数据兼容**：`Account.WorkerManaged/Port/Pid/Home` 是 `omitempty` 新字段；回滚到旧 binary 时自动忽略，不破坏账号文件

## 12. 后续阶段（本 doc 不实施）

按 §8 底部的迁移路径：
- **阶段 1**：`/v1/messages` 改走 worker → 删 `anthropic/` 1002 行
- **阶段 2**：`/v1/chat/completions` + `/v1/embeddings` + compaction + probe 全改 worker → 删 `ProxyRequestWithBytes*` 等
- **阶段 3**：`fetchModels` + `refreshCopilotToken` 迁到子进程 → gateway 不再持有 `copilot_token`
- **阶段 4**：删 `/v1/responses` direct fallback

每个后续阶段依赖上一个 + 本 doc 的 supervisor 基础设施。

## 13. 开放问题

- **端口占用冲突策略**：同主机若已有其他服务占用 9100-9199 区间，是否给个黑名单机制？当前方案只做"试绑失败就跳下一个"
- **api-home 保留还是清理**：删除账号时 `rm -rf` 目录（干净）vs 保留并打 `tombstone`（便于审计 / 恢复）。默认清理；提供 `COPILOT_WORKER_PRESERVE_ON_DELETE=true` 开关
- **bun vs node vs 打包二进制**：`copilot-api` 用什么形式发布 / 查找？要不要让 supervisor 支持 `bun run` 作为 fallback？本 doc 假设 `copilot-api` 是 PATH 上的单一可执行文件（bun 打包产物或 npm global install）
