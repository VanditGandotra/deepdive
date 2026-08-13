"""SQLite caching layer: general cache, LLM content-hash cache, run snapshots, call logging."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Optional

from config import DB_PATH

_local = threading.local()
_write_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db() -> None:
    con = _conn()
    with _write_lock:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                key       TEXT PRIMARY KEY,
                value     TEXT NOT NULL,
                expires_at REAL NOT NULL,
                source    TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS llm_cache (
                input_hash TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_calls (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT    NOT NULL DEFAULT '',
                model               TEXT    NOT NULL,
                prompt_version      TEXT    NOT NULL,
                input_hash          TEXT    NOT NULL,
                tokens_input        INTEGER NOT NULL DEFAULT 0,
                tokens_output       INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
                cost_est_usd        REAL    NOT NULL DEFAULT 0,
                was_cached          INTEGER NOT NULL DEFAULT 0,
                created_at          REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_or_url TEXT    NOT NULL,
                run_at        REAL    NOT NULL,
                snapshot_json TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS freshness (
                key            TEXT PRIMARY KEY,
                source         TEXT NOT NULL,
                last_fetched   REAL NOT NULL,
                ttl_seconds    INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_key ON run_snapshots(ticker_or_url);
        """)
        con.commit()


# ── General cache ─────────────────────────────────────────────────────────────

def get_cache(key: str) -> Optional[str]:
    con = _conn()
    row = con.execute(
        "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
        (key, time.time()),
    ).fetchone()
    return row["value"] if row else None


def set_cache(key: str, value: str, ttl: int, source: str = "") -> None:
    with _write_lock:
        _conn().execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at, source) VALUES (?,?,?,?)",
            (key, value, time.time() + ttl, source),
        )
        _conn().commit()


def delete_cache(key: str) -> None:
    with _write_lock:
        _conn().execute("DELETE FROM cache WHERE key = ?", (key,))
        _conn().commit()


def get_cache_obj(key: str) -> Optional[Any]:
    raw = get_cache(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def set_cache_obj(key: str, value: Any, ttl: int, source: str = "") -> None:
    set_cache(key, json.dumps(value, default=str), ttl, source)


# ── LLM content-hash cache ────────────────────────────────────────────────────

def get_llm_cache(input_hash: str) -> Optional[str]:
    row = _conn().execute(
        "SELECT value FROM llm_cache WHERE input_hash = ?", (input_hash,)
    ).fetchone()
    return row["value"] if row else None


def set_llm_cache(input_hash: str, value: str) -> None:
    with _write_lock:
        _conn().execute(
            "INSERT OR REPLACE INTO llm_cache (input_hash, value, created_at) VALUES (?,?,?)",
            (input_hash, value, time.time()),
        )
        _conn().commit()


# ── LLM call logging ──────────────────────────────────────────────────────────

def log_llm_call(
    model: str,
    prompt_version: str,
    input_hash: str,
    tokens_input: int,
    tokens_output: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cost_est: float,
    was_cached: bool,
    session_id: str = "",
) -> None:
    with _write_lock:
        _conn().execute(
            """INSERT INTO llm_calls
               (session_id, model, prompt_version, input_hash,
                tokens_input, tokens_output, cache_write_tokens, cache_read_tokens,
                cost_est_usd, was_cached, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, model, prompt_version, input_hash,
                tokens_input, tokens_output, cache_write_tokens, cache_read_tokens,
                cost_est, int(was_cached), time.time(),
            ),
        )
        _conn().commit()


def get_session_stats(session_id: str) -> dict:
    rows = _conn().execute(
        """SELECT model, SUM(tokens_input) ti, SUM(tokens_output) to_,
                  SUM(cache_read_tokens) cr, SUM(cache_write_tokens) cw,
                  SUM(cost_est_usd) cost, SUM(was_cached) hits, COUNT(*) calls
           FROM llm_calls WHERE session_id = ?
           GROUP BY model""",
        (session_id,),
    ).fetchall()

    stats: dict = {
        "total_calls": 0, "cached_calls": 0,
        "tokens_input": 0, "tokens_output": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cost_est_usd": 0.0, "by_model": {},
    }
    for r in rows:
        stats["tokens_input"]       += r["ti"] or 0
        stats["tokens_output"]      += r["to_"] or 0
        stats["cache_read_tokens"]  += r["cr"] or 0
        stats["cache_write_tokens"] += r["cw"] or 0
        stats["cost_est_usd"]       += r["cost"] or 0
        stats["cached_calls"]       += r["hits"] or 0
        stats["total_calls"]        += r["calls"] or 0
        stats["by_model"][r["model"]] = {
            "calls": r["calls"],
            "cost": r["cost"] or 0,
        }
    return stats


# ── Run snapshots ─────────────────────────────────────────────────────────────

def save_run_snapshot(ticker_or_url: str, snapshot_json: str) -> None:
    with _write_lock:
        _conn().execute(
            "INSERT INTO run_snapshots (ticker_or_url, run_at, snapshot_json) VALUES (?,?,?)",
            (ticker_or_url.lower(), time.time(), snapshot_json),
        )
        _conn().commit()


def get_last_run_snapshot(ticker_or_url: str) -> Optional[dict]:
    row = _conn().execute(
        """SELECT run_at, snapshot_json FROM run_snapshots
           WHERE ticker_or_url = ? ORDER BY run_at DESC LIMIT 1""",
        (ticker_or_url.lower(),),
    ).fetchone()
    if not row:
        return None
    snap = json.loads(row["snapshot_json"])
    snap["_run_at"] = datetime.utcfromtimestamp(row["run_at"]).isoformat()
    return snap


# ── Freshness tracking ────────────────────────────────────────────────────────

def record_freshness(key: str, source: str, ttl_seconds: int) -> None:
    with _write_lock:
        _conn().execute(
            "INSERT OR REPLACE INTO freshness (key, source, last_fetched, ttl_seconds) VALUES (?,?,?,?)",
            (key, source, time.time(), ttl_seconds),
        )
        _conn().commit()


def get_freshness(key: str) -> Optional[dict]:
    row = _conn().execute(
        "SELECT source, last_fetched, ttl_seconds FROM freshness WHERE key = ?", (key,)
    ).fetchone()
    if not row:
        return None
    last_fetched = datetime.utcfromtimestamp(row["last_fetched"])
    age_secs = (datetime.utcnow() - last_fetched).total_seconds()
    is_stale = age_secs > row["ttl_seconds"]
    if age_secs < 3600:
        age_str = f"{int(age_secs / 60)}m ago"
    elif age_secs < 86400:
        age_str = f"{int(age_secs / 3600)}h ago"
    else:
        age_str = f"{int(age_secs / 86400)}d ago"
    return {
        "source": row["source"],
        "last_fetched_at": last_fetched,
        "is_stale": is_stale,
        "age_description": age_str,
    }


# ── Cleanup ───────────────────────────────────────────────────────────────────

def purge_expired_cache() -> int:
    with _write_lock:
        cur = _conn().execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        _conn().commit()
        return cur.rowcount


# ── Portfolio CRUD ────────────────────────────────────────────────────────────

def init_portfolio_tables() -> None:
    con = _conn()
    with _write_lock:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL UNIQUE,
                created_at REAL    NOT NULL,
                updated_at REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS holdings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id  INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
                ticker        TEXT    NOT NULL,
                shares        REAL    NOT NULL DEFAULT 0,
                cost_basis    REAL,
                account       TEXT,
                notes         TEXT,
                is_cash       INTEGER NOT NULL DEFAULT 0,
                UNIQUE(portfolio_id, ticker)
            );

            PRAGMA foreign_keys = ON;
        """)
        con.commit()


def create_portfolio(name: str) -> int:
    """Create a new portfolio. Returns portfolio_id."""
    now = time.time()
    with _write_lock:
        cur = _conn().execute(
            "INSERT INTO portfolios (name, created_at, updated_at) VALUES (?, ?, ?)",
            (name, now, now),
        )
        _conn().commit()
        return cur.lastrowid


def get_portfolio_names() -> list:
    """Return list of portfolio names."""
    rows = _conn().execute(
        "SELECT name FROM portfolios ORDER BY created_at"
    ).fetchall()
    return [r["name"] for r in rows]


def get_portfolio_id(name: str) -> Optional[int]:
    """Return portfolio_id for name, or None if not found."""
    row = _conn().execute(
        "SELECT id FROM portfolios WHERE name = ?", (name,)
    ).fetchone()
    return row["id"] if row else None


def save_holdings(portfolio_id: int, holdings: list) -> None:
    """
    Replace all holdings for this portfolio.
    Each holding dict: {ticker, shares, cost_basis, account, notes, is_cash}
    """
    with _write_lock:
        con = _conn()
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
        for h in holdings:
            con.execute(
                """INSERT OR REPLACE INTO holdings
                   (portfolio_id, ticker, shares, cost_basis, account, notes, is_cash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    portfolio_id,
                    h["ticker"],
                    h.get("shares", 0),
                    h.get("cost_basis"),
                    h.get("account"),
                    h.get("notes"),
                    int(bool(h.get("is_cash", False))),
                ),
            )
        con.execute(
            "UPDATE portfolios SET updated_at = ? WHERE id = ?",
            (time.time(), portfolio_id),
        )
        con.commit()


def get_holdings(portfolio_id: int) -> list:
    """Return holdings as list of dicts."""
    rows = _conn().execute(
        """SELECT ticker, shares, cost_basis, account, notes, is_cash
           FROM holdings WHERE portfolio_id = ? ORDER BY is_cash, ticker""",
        (portfolio_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_portfolio(portfolio_id: int) -> None:
    """Delete portfolio and all its holdings."""
    with _write_lock:
        con = _conn()
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
        con.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        con.commit()


# Initialise tables on first import
init_db()
init_portfolio_tables()
