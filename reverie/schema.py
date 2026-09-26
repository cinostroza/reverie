"""SQL schema and forward-only migrations (HLD 6.2, 14.3)."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# v1
# --------------------------------------------------------------------------
_V1 = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
  id             TEXT PRIMARY KEY,
  scope_id       TEXT NOT NULL,
  type           TEXT NOT NULL,
  memory_class   TEXT NOT NULL,
  label          TEXT NOT NULL,
  body           TEXT NOT NULL DEFAULT '',
  valence        REAL NOT NULL DEFAULT 0.0,
  provenance     TEXT NOT NULL,
  source_ids     TEXT NOT NULL DEFAULT '[]',
  state          TEXT NOT NULL DEFAULT 'active',
  superseded_by  TEXT REFERENCES nodes(id),
  merged_into    TEXT REFERENCES nodes(id),
  learned_under  TEXT NOT NULL DEFAULT '{}',
  community_id   TEXT,
  centrality     REAL NOT NULL DEFAULT 0.0,
  alpha          REAL NOT NULL DEFAULT 1.0,
  beta           REAL NOT NULL DEFAULT 1.0,
  attributions   INTEGER NOT NULL DEFAULT 0,
  utility_at     INTEGER NOT NULL,
  strength       REAL NOT NULL DEFAULT 1.0,
  activations    INTEGER NOT NULL DEFAULT 0,
  last_activated INTEGER,
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_scope_state ON nodes(scope_id, state, type);
CREATE INDEX IF NOT EXISTS idx_nodes_label       ON nodes(scope_id, label);
CREATE INDEX IF NOT EXISTS idx_nodes_community   ON nodes(scope_id, community_id, centrality DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_class       ON nodes(scope_id, memory_class, state);

CREATE TABLE IF NOT EXISTS edges (
  id             TEXT PRIMARY KEY,
  scope_id       TEXT NOT NULL,
  src            TEXT NOT NULL REFERENCES nodes(id),
  dst            TEXT NOT NULL REFERENCES nodes(id),
  relation       TEXT NOT NULL,
  weight         REAL NOT NULL DEFAULT 0.5,
  provenance     TEXT NOT NULL,
  confidence     REAL NOT NULL DEFAULT 0.5,
  traversals     INTEGER NOT NULL DEFAULT 0,
  last_traversed INTEGER,
  created_at     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(scope_id, src, weight DESC);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(scope_id, dst, weight DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_uniq ON edges(scope_id, src, dst, relation);

CREATE TABLE IF NOT EXISTS aliases (
  scope_id TEXT NOT NULL,
  alias    TEXT NOT NULL,
  node_id  TEXT NOT NULL REFERENCES nodes(id),
  PRIMARY KEY (scope_id, alias)
);

CREATE TABLE IF NOT EXISTS communities (
  id          TEXT PRIMARY KEY,
  scope_id    TEXT NOT NULL,
  label       TEXT,
  summary     TEXT,
  cohesion    REAL,
  node_count  INTEGER NOT NULL DEFAULT 0,
  prev_id     TEXT,
  updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS node_vec (
  node_id   TEXT PRIMARY KEY REFERENCES nodes(id),
  scope_id  TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vec_scope ON node_vec(scope_id);

CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(
  node_id UNINDEXED,
  scope_id UNINDEXED,
  label,
  body,
  tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS recalls (
  id           TEXT PRIMARY KEY,
  scope_id     TEXT NOT NULL,
  cue          TEXT NOT NULL,
  node_ids     TEXT NOT NULL,
  activations  TEXT NOT NULL,
  withheld_ids TEXT NOT NULL DEFAULT '[]',
  is_ablation  INTEGER NOT NULL DEFAULT 0,
  token_cost   INTEGER NOT NULL DEFAULT 0,
  latency_ms   REAL NOT NULL DEFAULT 0.0,
  created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recalls_created ON recalls(scope_id, created_at);

CREATE TABLE IF NOT EXISTS outcomes (
  id          TEXT PRIMARY KEY,
  scope_id    TEXT NOT NULL,
  recall_id   TEXT REFERENCES recalls(id),
  episode_id  TEXT,
  idem_key    TEXT UNIQUE,
  signal_tier INTEGER NOT NULL,
  outcome     TEXT NOT NULL,
  detail      TEXT NOT NULL DEFAULT '',
  attributed  INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_pending ON outcomes(attributed, created_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_recall  ON outcomes(recall_id);

-- Per-node ledger of applied attributions. Exists so a later higher-tier
-- outcome can *reverse* an earlier lower-tier one (HLD 9.3) rather than
-- stacking a contradiction on top of it.
CREATE TABLE IF NOT EXISTS attribution_log (
  id          TEXT PRIMARY KEY,
  outcome_id  TEXT NOT NULL REFERENCES outcomes(id),
  node_id     TEXT NOT NULL,
  d_alpha     REAL NOT NULL,
  d_beta      REAL NOT NULL,
  reversed    INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attrlog_outcome ON attribution_log(outcome_id);
CREATE INDEX IF NOT EXISTS idx_attrlog_node    ON attribution_log(node_id, reversed);

CREATE TABLE IF NOT EXISTS episode_buffer (
  id           TEXT PRIMARY KEY,
  scope_id     TEXT NOT NULL,
  client_key   TEXT,
  payload      TEXT NOT NULL,
  task_type    TEXT NOT NULL DEFAULT 'unknown',
  consolidated INTEGER NOT NULL DEFAULT 0,
  created_at   INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_buffer_client
  ON episode_buffer(scope_id, client_key) WHERE client_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_buffer_pending
  ON episode_buffer(scope_id, consolidated, created_at);

-- Unsampled counters. Backpressure sampling (HLD 7) deliberately skews the
-- buffer toward failure, so every base rate in the system reads from here
-- instead -- including tier-3 baselines and the "N of M" figures in briefs.
CREATE TABLE IF NOT EXISTS counters (
  scope_id   TEXT NOT NULL,
  task_type  TEXT NOT NULL,
  outcome    TEXT NOT NULL,
  n          INTEGER NOT NULL DEFAULT 0,
  sum_ms     INTEGER NOT NULL DEFAULT 0,
  sum_steps  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (scope_id, task_type, outcome)
);

-- Duration samples for tier-3 behavioural baselines. Reservoir-capped.
CREATE TABLE IF NOT EXISTS duration_samples (
  scope_id  TEXT NOT NULL,
  task_type TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  n_steps   INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dursamp ON duration_samples(scope_id, task_type);
"""

_MIGRATIONS: dict[int, str] = {1: _V1}


class SchemaError(RuntimeError):
    pass


def migrate(conn: sqlite3.Connection) -> int:
    """Apply forward-only migrations. Refuses to open a future database."""
    current = _read_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaError(
            f"database schema v{current} is newer than this build (v{SCHEMA_VERSION}). "
            "Upgrade reverie."
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(_MIGRATIONS[version])
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(version),),
        )
        conn.commit()
    return SCHEMA_VERSION


def _read_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row[0]) if row else 0


def check_embedding_model(conn: sqlite3.Connection, model: str, dim: int) -> None:
    """HLD 14.3: refuse to start rather than silently mix vector spaces."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key='embedding_model'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (model,),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('embedding_dim', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(dim),),
        )
        conn.commit()
        return

    stored = row[0]
    if stored != model:
        raise SchemaError(
            f"embedding model mismatch: database was built with {stored!r}, "
            f"config says {model!r}. Mixing vector spaces silently corrupts recall.\n"
            f"Fix with:  reverie reembed --model {model}"
        )
