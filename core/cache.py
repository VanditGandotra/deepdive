from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date
from typing import Optional

from config import TTL_PRICES
from data.cache import _conn, _write_lock
from core.schemas import TickerAnalysis

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "v1"


def _config_hash(config: Optional[dict]) -> str:
    if not config:
        return "default"
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _init_table() -> None:
    with _write_lock:
        _conn().execute("""
            CREATE TABLE IF NOT EXISTS ticker_analysis (
                ticker           TEXT NOT NULL,
                as_of_date       TEXT NOT NULL,
                analysis_version TEXT NOT NULL,
                config_hash      TEXT NOT NULL DEFAULT 'default',
                value_json       TEXT NOT NULL,
                expires_at       REAL NOT NULL,
                PRIMARY KEY (ticker, as_of_date, analysis_version, config_hash)
            )
        """)
        _conn().commit()
        # Migration: add config_hash column if upgrading from old schema
        try:
            _conn().execute("ALTER TABLE ticker_analysis ADD COLUMN config_hash TEXT NOT NULL DEFAULT 'default'")
            _conn().commit()
        except Exception:
            pass  # column already exists


_init_table()


def get_cached_analysis(
    ticker: str,
    as_of: date,
    version: str = ANALYSIS_VERSION,
    config: Optional[dict] = None,
) -> Optional[TickerAnalysis]:
    ch = _config_hash(config)
    row = _conn().execute(
        "SELECT value_json FROM ticker_analysis WHERE ticker=? AND as_of_date=? AND analysis_version=? AND config_hash=? AND expires_at>?",
        (ticker.upper(), str(as_of), version, ch, time.time()),
    ).fetchone()
    if not row:
        return None
    try:
        return TickerAnalysis.model_validate_json(row[0])
    except Exception as exc:
        logger.warning("Failed to deserialize cached analysis for %s: %s", ticker, exc)
        return None


def save_analysis(analysis: TickerAnalysis, config: Optional[dict] = None) -> None:
    ch = _config_hash(config)
    with _write_lock:
        _conn().execute(
            "INSERT OR REPLACE INTO ticker_analysis (ticker, as_of_date, analysis_version, config_hash, value_json, expires_at) VALUES (?,?,?,?,?,?)",
            (
                analysis.ticker.upper(),
                str(analysis.as_of),
                analysis.analysis_version,
                ch,
                analysis.model_dump_json(),
                time.time() + TTL_PRICES,
            ),
        )
        _conn().commit()
