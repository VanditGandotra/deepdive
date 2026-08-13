"""Durable portfolio persistence — portfolios and holdings survive cache clears."""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Optional

from config import DB_PATH

_local = threading.local()
_write_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def _init() -> None:
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
        """)
        con.commit()


def create_portfolio(name: str) -> int:
    """Create a new portfolio. Returns portfolio_id."""
    now = time.time()
    with _write_lock:
        con = _conn()
        cur = con.execute(
            "INSERT INTO portfolios (name, created_at, updated_at) VALUES (?, ?, ?)",
            (name, now, now),
        )
        con.commit()
        return cur.lastrowid


def get_portfolio_names() -> list:
    """Return portfolio names ordered by creation time."""
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
    Idempotent full-replace of holdings for this portfolio.
    Each dict: {ticker, shares, cost_basis?, account?, notes?, is_cash?}
    """
    with _write_lock:
        con = _conn()
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
    """Return holdings as list of dicts, cash rows last."""
    rows = _conn().execute(
        """SELECT ticker, shares, cost_basis, account, notes, is_cash
           FROM holdings WHERE portfolio_id = ? ORDER BY is_cash, ticker""",
        (portfolio_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_portfolio(portfolio_id: int) -> None:
    """Delete portfolio and all its holdings (cascade via FK)."""
    with _write_lock:
        con = _conn()
        con.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
        con.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        con.commit()


_init()
