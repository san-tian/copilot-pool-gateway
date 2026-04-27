# Storage architecture

Status: draft for local implementation work. Not committed.

## Decision summary

- SQLite is the only online authority.
- SQLite stores metadata only. SQLite does not store request or response payload bodies.
- DuckDB stores cold payload and transcript data only.
- Normal request handling does not read DuckDB.
- Recovery paths may read DuckDB when SQLite metadata says a replayable transcript exists but the hot in-memory payload cache does not have it.
- In-memory LRU caches are accelerators only. They are not authoritative.

## Resource envelope

- RAM budget for this feature: up to 24 GB on a 32 GB host.
- Disk budget for archived payload data: configurable within a 64 GB local disk budget.
- Metadata retention target: 14 days, hard retention when possible.
- Payload retention target: 14 days, soft target subject to disk budget and LRU eviction.

## Data classes

### 1. Metadata in SQLite

Metadata is small, queried frequently, and must survive process restart. Keep it in SQLite for 14 days.

Examples:

- `response_id -> account_id + turn_context`
- `call_id -> account_id + turn_context`
- `tool_use_id -> account_id + turn_context`
- `session_affinity_key -> account_id`
- `response_id -> transcript_ref`
- `created_at`
- `last_seen_at`
- `expires_at`
- `archive_state`
- `duckdb_ref`
- model, route, source, status flags needed by recovery logic

This metadata remains the basis for:

- canonical continuation binding
- split vs orphan classification
- session affinity lookup
- deciding whether recovery is possible
- deciding which recovery path to try next

### 2. Hot payload in memory only

Large request and response payloads can be cached in memory for fast replay, but memory is only a cache.

Examples:

- request body bytes
- replay transcript items
- response body fragments needed for transcript replay

Eviction policy:

- bounded by configurable RAM budget
- LRU by `last_seen_at`
- process restart drops the hot cache without changing authoritative metadata

### 3. Cold payload in DuckDB

DuckDB stores payload and transcript data for recovery when the hot cache has already evicted them.

Examples:

- canonicalized request body
- replay transcript body
- optional archived response body for debugging or forensic replay

DuckDB is not used for:

- normal routing
- normal continuation binding
- session affinity
- rate-limit or account selection

DuckDB is used for:

- transcript replay during recovery after hot-cache miss
- historical incident replay
- storage-efficient cold retention of large payloads

## Read path

### Normal path

Normal requests use only:

- in-memory hot cache when present
- SQLite metadata for binding and recovery decisions

Normal requests do not read DuckDB.

### Recovery path

Recovery path order:

1. use SQLite metadata to determine whether replay or recovery should happen
2. try in-memory hot payload cache
3. if hot payload is missing and metadata says archived payload exists, load payload from DuckDB
4. if DuckDB payload is missing, fall back to existing non-transcript recovery paths
   - orphan translate
   - degrade opt-in
   - typed error

This means DuckDB miss does not invalidate SQLite metadata. It only reduces recovery capability.

## Write path

### Synchronous request-path writes

During request handling, write only metadata to SQLite.

Examples:

- new response turn bindings
- function call ownership
- message tool ownership
- session affinity updates
- replay index rows
- payload archive intent rows
- `last_seen_at` updates

Do not write payload bodies into SQLite.

### Asynchronous archive writes

A background archiver moves payload or transcript bodies into DuckDB.

Flow:

1. request path records metadata and archive intent in SQLite
2. request path may place payload into in-memory hot cache
3. background worker writes canonicalized payload into DuckDB
4. background worker updates SQLite metadata
   - `archive_state=ready`
   - `duckdb_ref=<stable key>`
   - size and checksum fields if needed

If DuckDB write fails:

- SQLite metadata remains valid
- normal path remains unaffected
- only transcript replay after hot-cache miss may fail

## Retention and eviction

### SQLite metadata retention

- retain for 14 days
- do not capacity-evict metadata under normal operation
- expire by time window only

### Hot in-memory payload retention

- bounded by RAM budget
- evict by LRU
- optional secondary TTL if needed for stale cleanup, but capacity pressure is primary

### DuckDB payload retention

- target 14 days
- bounded by configurable disk budget
- evict by LRU or least-recently-recovered semantics using metadata tracked in SQLite
- eviction removes cold replay capability only

## Compression stance

SQLite does not store payload.

Payload compression belongs in DuckDB. The expected shape is:

- metadata columns in SQLite
- payload/transcript bodies in DuckDB
- optional canonicalization and deduplication before DuckDB insert

DuckDB is acceptable here because it sits off the normal request path and is only used on recovery or historical replay.

## Consistency model

There is one authority only: SQLite metadata.

DuckDB is a derived payload store.

Therefore:

- SQLite commit success defines the authoritative state
- DuckDB lag or failure does not change continuation ownership facts
- recovery consults SQLite first and DuckDB second
- missing DuckDB payload means `recovery degraded`, not `metadata missing`

## Migration and branch strategy

Implementation should start from `origin/master`, not from `nolanho/fix`.

Reason:

- `nolanho/fix` diverged before worker supervisor, responses session affinity, routing telemetry, and trace header changes
- rebase cost is higher than rewriting on the current architecture

Cherry-pick or manually port only the low-conflict parts of `nolanho/fix` that still match current semantics:

- configurable cache capacities and durations
- logging improvements where still relevant

Do not port `handler/proxy.go` continuation logic from `nolanho/fix` blindly.

## Immediate implementation direction

1. add SQLite schema for metadata authority
2. replace JSON and gzip continuation snapshots with SQLite metadata writes
3. add hot in-memory payload cache with LRU
4. add DuckDB archival schema for payload and transcript bodies
5. wire recovery path to read DuckDB only after hot-cache miss
6. keep existing orphan translate and degrade behavior as fallback

## Current implementation status on `sqlite-metadata`

Implemented:

- SQLite authority for continuation bindings
  - `response_id`
  - `function_call call_id`
  - `tool_use_id`
  - `client_machine_id`
- SQLite authority for responses replay metadata
  - `response_id -> account_id`
  - `archive_state`
  - `duckdb_ref`
- DuckDB cold payload store for replay request body and replay transcript body
- recovery path order in `responses_replay.go`
  - hot in-memory payload cache first
  - DuckDB on hot-cache miss
  - existing degrade or typed error after replay miss
- normal routing still uses SQLite metadata only; it does not read DuckDB
- legacy JSON snapshot writes removed; `continuation-state.json(.gz)` is now read-only for one-time compatibility import
- Dockerfile updated for DuckDB CGO build and runtime linkage
- DuckDB write-time logging added in `responses_replay_duckdb.go`

Not implemented yet:

- async archive worker; current DuckDB payload write is synchronous in `storeResponsesReplay`
- DuckDB retention worker and disk-budget enforcement
- migration of old replay payload bodies into DuckDB; legacy JSON is imported only into SQLite metadata
