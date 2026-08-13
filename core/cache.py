from __future__ import annotations

import time
from datetime import date
from typing import Optional

from config import TTL_PRICES
from data.cache import _conn, _write_lock
from core.schemas import TickerAnalysis

ANALYSIS_VERSION = "v1"


def _init_table() -> None:
    with _write_lock:
        _conn().execute("""
            CREATE TABLE IF NOT EXISTS ticker_analysis (
                ticker           TEXT NOT NULL,
                as_of_date       TEXT NOT NULL,
                analysis_version TEXT NOT NULL,
                value_json       TEXT NOT NULL,
                expires_at       REAL NOT NULL,
                PRIMARY KEY (ticker, as_of_date, analysis_version)
            )
        """)
        _conn().commit()


_init_table()


def get_cached_analysis(
    ticker: str,
    as_of: date,
    version: str = ANALYSIS_VERSION,
) -> Optional[TickerAnalysis]:
    row = _conn().execute(
        "SELECT value_json FROM ticker_analysis WHERE ticker=? AND as_of_date=? AND analysis_version=? AND expires_at>?",
        (ticker.upper(), str(as_of), version, time.time()),
    ).fetchone()
    if not row:
        return None
    try:
        return TickerAnalysis.model_validate_json(row[0])
    except Exception:
        return None


def save_analysis(analysis: TickerAnalysis) -> None:
    with _write_lock:
        _conn().execute(
            "INSERT OR REPLACE INTO ticker_analysis (ticker, as_of_date, analysis_version, value_json, expires_at) VALUES (?,?,?,?,?)",
            (
                analysis.ticker.upper(),
                str(analysis.as_of),
                analysis.analysis_version,
                analysis.model_dump_json(),
                time.time() + TTL_PRICES,
            ),
        )
        _conn().commit()
