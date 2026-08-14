"""Tests for portfolio persistence and portfolio model. No API calls required."""
from __future__ import annotations

import sqlite3
import threading
import time as _time
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ── In-memory DB helpers ───────────────────────────────────────────────────────

class _InMemoryCache:
    """
    Self-contained in-memory SQLite backend for testing portfolio CRUD.
    Mirrors the public API of data.cache portfolio functions.
    """

    def __init__(self) -> None:
        self._con = sqlite3.connect(":memory:", check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL UNIQUE,
                created_at REAL    NOT NULL,
                updated_at REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS holdings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id  INTEGER NOT NULL,
                ticker        TEXT    NOT NULL,
                shares        REAL    NOT NULL DEFAULT 0,
                cost_basis    REAL,
                account       TEXT,
                notes         TEXT,
                is_cash       INTEGER NOT NULL DEFAULT 0,
                UNIQUE(portfolio_id, ticker)
            );
        """)
        self._con.commit()

    def create_portfolio(self, name: str) -> int:
        now = _time.time()
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO portfolios (name, created_at, updated_at) VALUES (?,?,?)",
                (name, now, now),
            )
            self._con.commit()
            return cur.lastrowid

    def get_portfolio_names(self) -> list:
        rows = self._con.execute("SELECT name FROM portfolios ORDER BY created_at").fetchall()
        return [r["name"] for r in rows]

    def get_portfolio_id(self, name: str):
        row = self._con.execute("SELECT id FROM portfolios WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None

    def save_holdings(self, portfolio_id: int, holdings: list) -> None:
        with self._lock:
            self._con.execute("DELETE FROM holdings WHERE portfolio_id=?", (portfolio_id,))
            for h in holdings:
                self._con.execute(
                    """INSERT OR REPLACE INTO holdings
                       (portfolio_id, ticker, shares, cost_basis, account, notes, is_cash)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        portfolio_id, h["ticker"], h.get("shares", 0),
                        h.get("cost_basis"), h.get("account"), h.get("notes"),
                        int(bool(h.get("is_cash", False))),
                    ),
                )
            self._con.execute(
                "UPDATE portfolios SET updated_at=? WHERE id=?",
                (_time.time(), portfolio_id),
            )
            self._con.commit()

    def get_holdings(self, portfolio_id: int) -> list:
        rows = self._con.execute(
            "SELECT ticker, shares, cost_basis, account, notes, is_cash "
            "FROM holdings WHERE portfolio_id=? ORDER BY is_cash, ticker",
            (portfolio_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_portfolio(self, portfolio_id: int) -> None:
        with self._lock:
            self._con.execute("DELETE FROM holdings WHERE portfolio_id=?", (portfolio_id,))
            self._con.execute("DELETE FROM portfolios WHERE id=?", (portfolio_id,))
            self._con.commit()


@pytest.fixture()
def db() -> _InMemoryCache:
    """Fresh in-memory DB per test — no filesystem, no cross-test contamination."""
    return _InMemoryCache()


# ── Tests: Portfolio CRUD ──────────────────────────────────────────────────────

class TestPortfolioCRUD:

    def test_create_and_get_portfolio(self, db: _InMemoryCache) -> None:
        pid = db.create_portfolio("My Test Portfolio")
        assert isinstance(pid, int)
        assert pid > 0
        names = db.get_portfolio_names()
        assert "My Test Portfolio" in names

    def test_get_portfolio_id(self, db: _InMemoryCache) -> None:
        pid = db.create_portfolio("Alpha Fund")
        fetched_id = db.get_portfolio_id("Alpha Fund")
        assert fetched_id == pid

    def test_get_portfolio_id_missing(self, db: _InMemoryCache) -> None:
        result = db.get_portfolio_id("DoesNotExist")
        assert result is None

    def test_save_and_get_holdings(self, db: _InMemoryCache) -> None:
        pid = db.create_portfolio("Hold Test")
        holdings_in = [
            {"ticker": "AAPL", "shares": 10.0, "cost_basis": 150.0, "account": "Taxable", "notes": "Core", "is_cash": False},
            {"ticker": "CASH", "shares": 5000.0, "cost_basis": None, "account": None, "notes": None, "is_cash": True},
        ]
        db.save_holdings(pid, holdings_in)
        holdings_out = db.get_holdings(pid)
        assert len(holdings_out) == 2
        tickers = [h["ticker"] for h in holdings_out]
        assert "AAPL" in tickers
        assert "CASH" in tickers
        aapl = next(h for h in holdings_out if h["ticker"] == "AAPL")
        assert aapl["shares"] == pytest.approx(10.0)
        assert aapl["cost_basis"] == pytest.approx(150.0)
        assert aapl["account"] == "Taxable"
        assert aapl["notes"] == "Core"
        assert not bool(aapl["is_cash"])
        cash_row = next(h for h in holdings_out if h["ticker"] == "CASH")
        assert bool(cash_row["is_cash"])

    def test_save_holdings_replaces_existing(self, db: _InMemoryCache) -> None:
        pid = db.create_portfolio("Replace Test")
        db.save_holdings(pid, [{"ticker": "NVDA", "shares": 5.0, "cost_basis": None, "account": None, "notes": None, "is_cash": False}])
        # Replace with entirely different holdings
        db.save_holdings(pid, [{"ticker": "MSFT", "shares": 20.0, "cost_basis": 300.0, "account": None, "notes": None, "is_cash": False}])
        holdings = db.get_holdings(pid)
        assert len(holdings) == 1
        assert holdings[0]["ticker"] == "MSFT"

    def test_delete_portfolio(self, db: _InMemoryCache) -> None:
        pid = db.create_portfolio("To Delete")
        db.save_holdings(pid, [{"ticker": "TSLA", "shares": 3.0, "cost_basis": None, "account": None, "notes": None, "is_cash": False}])
        db.delete_portfolio(pid)
        assert "To Delete" not in db.get_portfolio_names()
        assert db.get_portfolio_id("To Delete") is None
        # Holdings should also be gone
        holdings = db.get_holdings(pid)
        assert holdings == []

    def test_multiple_portfolios(self, db: _InMemoryCache) -> None:
        db.create_portfolio("Portfolio A")
        db.create_portfolio("Portfolio B")
        names = db.get_portfolio_names()
        assert "Portfolio A" in names
        assert "Portfolio B" in names


# ── Tests: Portfolio model ─────────────────────────────────────────────────────

from core.portfolio import EnrichmentResult, Holding, Portfolio, enrich_with_prices


class TestPortfolioModel:

    def _make_portfolio(self) -> Portfolio:
        holdings = [
            Holding(ticker="AAPL", shares=10, market_value=2000.0, sector="Technology"),
            Holding(ticker="MSFT", shares=5, market_value=1500.0, sector="Technology"),
            Holding(ticker="JPM", shares=8, market_value=500.0, sector="Financials"),
        ]
        pf = Portfolio(name="Test", holdings=holdings)
        return pf

    def test_total_value(self) -> None:
        pf = self._make_portfolio()
        assert pf.total_value == pytest.approx(4000.0)

    def test_compute_weights(self) -> None:
        pf = self._make_portfolio()
        pf.compute_weights()
        total_weight = sum(h.weight for h in pf.holdings)
        assert total_weight == pytest.approx(1.0, abs=1e-6)
        aapl = next(h for h in pf.holdings if h.ticker == "AAPL")
        assert aapl.weight == pytest.approx(2000.0 / 4000.0)

    def test_compute_weights_zero_value(self) -> None:
        pf = Portfolio(name="Empty", holdings=[Holding(ticker="AAPL", shares=10, market_value=0.0)])
        pf.compute_weights()
        assert pf.holdings[0].weight == 0.0

    def test_sector_concentration(self) -> None:
        pf = self._make_portfolio()
        pf.compute_weights()
        conc = pf.sector_concentration()
        total = sum(conc.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert "Technology" in conc
        assert "Financials" in conc
        assert conc["Technology"] == pytest.approx((2000.0 + 1500.0) / 4000.0)

    def test_sector_concentration_unknown_sector(self) -> None:
        pf = Portfolio(name="No Sector", holdings=[
            Holding(ticker="XYZ", shares=1, market_value=100.0, sector=None, weight=1.0),
        ])
        conc = pf.sector_concentration()
        assert "Unknown" in conc

    def test_to_dataframe_columns(self) -> None:
        pf = self._make_portfolio()
        pf.compute_weights()
        df = pf.to_dataframe()
        expected_cols = [
            "Ticker", "Shares", "Cost Basis", "Account", "Notes",
            "Price", "Market Value", "Unrealized P&L", "Unrealized P&L %",
            "Weight", "Sector",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_to_dataframe_rows(self) -> None:
        pf = self._make_portfolio()
        df = pf.to_dataframe()
        assert len(df) == 3
        assert set(df["Ticker"].tolist()) == {"AAPL", "MSFT", "JPM"}

    def test_to_dataframe_empty(self) -> None:
        pf = Portfolio(name="Empty")
        df = pf.to_dataframe()
        assert len(df) == 0


# ── Tests: CSV parser ──────────────────────────────────────────────────────────

import io as _io
from ui.portfolio_ui import _parse_csv


class _FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""
    def __init__(self, content: str) -> None:
        self._content = content.encode("utf-8")

    def read(self) -> bytes:
        return self._content


class TestCsvParser:

    def test_basic_csv(self) -> None:
        csv = "Ticker,Shares,Cost Basis\nAAPL,10,150.0\nMSFT,5,300.0\n"
        df = _parse_csv(_FakeUpload(csv))
        assert df is not None
        assert len(df) == 2
        assert set(df["Ticker"].tolist()) == {"AAPL", "MSFT"}

    def test_csv_with_symbol_column(self) -> None:
        csv = "Symbol,Quantity,Avg Price\nNVDA,20,450.0\nAMD,15,100.0\n"
        df = _parse_csv(_FakeUpload(csv))
        assert df is not None
        assert "NVDA" in df["Ticker"].tolist()

    def test_csv_cash_detection(self) -> None:
        csv = "Ticker,Shares\nAAPL,10\nCASH,5000\n"
        df = _parse_csv(_FakeUpload(csv))
        assert df is not None
        cash_rows = df[df["Cash"] == True]
        assert len(cash_rows) == 1
        assert cash_rows.iloc[0]["Ticker"] == "CASH"

    def test_csv_missing_required_columns(self) -> None:
        csv = "Name,Price\nApple,150\n"
        with pytest.raises(ValueError, match="Cannot find Ticker and Shares columns"):
            _parse_csv(_FakeUpload(csv))

    def test_csv_junk_header_rows(self) -> None:
        csv = "Account: Brokerage\nDate: 2026-01-01\nTicker,Shares,Cost Basis\nAAPL,10,150\n"
        df = _parse_csv(_FakeUpload(csv))
        assert df is not None
        assert "AAPL" in df["Ticker"].tolist()

    def test_csv_strips_whitespace(self) -> None:
        csv = "Ticker,Shares\n AAPL ,10\n"
        df = _parse_csv(_FakeUpload(csv))
        assert df is not None
        assert df.iloc[0]["Ticker"] == "AAPL"


# ── Tests: EnrichmentResult contract ─────────────────────────────────────────

class TestEnrichmentResult:

    def test_enrich_always_returns_enrichment_result(self) -> None:
        """enrich_with_prices must return EnrichmentResult on every path."""
        from unittest.mock import patch, MagicMock
        fund = MagicMock()
        fund.current_price = 150.0
        fund.sector = "Technology"
        pf = Portfolio(name="T", holdings=[Holding(ticker="AAPL", shares=10)])
        with patch("data.market.get_fundamentals", return_value=fund):
            result = enrich_with_prices(pf)
        assert isinstance(result, EnrichmentResult)
        assert result.failed == []
        assert result.portfolio.holdings[0].market_value == pytest.approx(1500.0)

    def test_enrich_collects_failed_tickers(self) -> None:
        """Tickers whose fetch raises must appear in result.failed, not crash."""
        from unittest.mock import patch
        pf = Portfolio(name="T", holdings=[
            Holding(ticker="GOOD", shares=5),
            Holding(ticker="BAD", shares=3),
        ])
        def fake_get(ticker, **_):
            if ticker == "BAD":
                raise RuntimeError("network error")
            m = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            m.current_price = 100.0
            m.sector = "X"
            return m
        with patch("data.market.get_fundamentals", side_effect=fake_get):
            result = enrich_with_prices(pf)
        assert isinstance(result, EnrichmentResult)
        assert "BAD" in result.failed
        assert "GOOD" not in result.failed

    def test_enrich_empty_portfolio(self) -> None:
        pf = Portfolio(name="Empty", holdings=[])
        result = enrich_with_prices(pf)
        assert isinstance(result, EnrichmentResult)
        assert result.failed == []

    def test_enrich_cash_row_never_fails(self) -> None:
        pf = Portfolio(name="T", holdings=[Holding(ticker="CASH", shares=5000, is_cash=True)])
        result = enrich_with_prices(pf)
        assert result.failed == []
        assert result.portfolio.holdings[0].market_value == pytest.approx(5000.0)


# ── Tests: Partial-data correctness (AMZN regression) ─────────────────────────

class TestPartialDataFixes:
    """Regression suite for the AMZN/partial-portfolio correctness bugs."""

    def test_failed_holding_weight_is_none(self) -> None:
        """Failed ticker weight must be None — not 0.0 silently renormalized away."""
        pf = Portfolio(name="T", holdings=[
            Holding(ticker="AMD", shares=10),
            Holding(ticker="AMZN", shares=5),
        ])

        def fake_get(ticker):
            if ticker == "AMZN":
                raise RuntimeError("rate limited")
            m = MagicMock()
            m.current_price = 100.0
            m.sector = "Technology"
            return m

        with patch("data.market.get_fundamentals", side_effect=fake_get):
            result = enrich_with_prices(pf)

        amzn = next(h for h in result.portfolio.holdings if h.ticker == "AMZN")
        amd = next(h for h in result.portfolio.holdings if h.ticker == "AMD")
        assert amzn.weight is None, "Failed holding must have weight=None, not 0.0"
        assert amd.weight == pytest.approx(1.0)   # 100% of the priced portion
        assert "AMZN" in result.failed

    def test_fetch_errors_include_exception_detail(self) -> None:
        """fetch_errors must record ticker → 'ExcType: message' for every failure."""
        pf = Portfolio(name="T", holdings=[Holding(ticker="AMZN", shares=5)])

        with patch("data.market.get_fundamentals",
                   side_effect=RuntimeError("Yahoo returned 0 fields")):
            result = enrich_with_prices(pf)

        assert "AMZN" in result.fetch_errors
        assert "RuntimeError" in result.fetch_errors["AMZN"]

    def test_failed_fetch_does_not_write_cache(self) -> None:
        """_check_info raises before set_cache_obj is reached — cache must stay clean."""
        with patch("data.cache.set_cache_obj") as mock_set, \
             patch("data.cache.get_cache_obj", return_value=None), \
             patch("data.cache.record_freshness"), \
             patch("time.sleep"):   # suppress retry delays
            mock_ticker = MagicMock()
            mock_ticker.info = {}  # empty dict → _check_info raises SourceUnavailable
            with patch("yfinance.Ticker", return_value=mock_ticker):
                from data.market import get_fundamentals
                with pytest.raises(Exception):
                    get_fundamentals("_TEST_NO_CACHE_")

        mock_set.assert_not_called()

    def test_yfinance_no_session_override(self) -> None:
        """yf.Ticker must be called without a session= kwarg so yfinance can
        use its own curl_cffi session internally (required since yfinance 0.2.50+)."""
        import inspect
        import data.market as mkt
        src = inspect.getsource(mkt)
        # Confirm _yf_session is gone and no session= is passed to Ticker
        assert "_yf_session" not in src, "_yf_session must be removed from data.market"
        assert "session=_yf_session" not in src, "session=_yf_session() must not appear in data.market"
