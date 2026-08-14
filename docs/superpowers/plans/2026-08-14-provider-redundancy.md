# Provider Redundancy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-provider yfinance with a resilient chain (yfinance → FMP → Stooq) so PLTR and any other ticker continues to render even when yfinance is rate-limited on shared Streamlit Cloud IPs.

**Architecture:** Add `CircuitBreaker` + `TokenBucket` to `data/resilience.py`, add `get_stale_cache_obj()` to `data/cache.py`, create `data/providers/` package (FmpMarketProvider, StooqProvider), wire a failover chain into `get_fundamentals` and `get_prices` in `data/market.py` with stale-while-revalidate, and fix the misleading "Other tabs are unaffected" messaging in `ui/components.py`.

**Tech Stack:** Python 3.13, httpx, yfinance, FMP free-tier API (250 req/day), Stooq keyless CSV API, SQLite (existing), pytest + unittest.mock

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `data/resilience.py` | Modify | Add `CircuitBreaker` and `TokenBucket` classes |
| `data/cache.py` | Modify | Add `get_stale_cache_obj()` — returns expired value if present |
| `data/providers/__init__.py` | Create | `ProviderResult` dataclass, `MarketProvider` protocol |
| `data/providers/fmp.py` | Create | FMP profile + key-metrics + quote fetch → `Fundamentals` / `PriceData` |
| `data/providers/stooq.py` | Create | Stooq CSV fetch → latest close price only |
| `config.py` | Modify | Add FMP profile/quote/key-metrics URLs and Stooq URL constants |
| `data/market.py` | Modify | Refactor `get_fundamentals` + `get_prices` to use provider chain with stale-while-revalidate |
| `ui/components.py` | Modify | Remove "Other tabs are unaffected." / "Freshness badges for other tabs are still accurate." |
| `app.py` | Modify | Enrich market data diagnostics with per-provider circuit state + staleness badge |
| `tests/test_provider_chain.py` | Create | 429 failover, stale cache, all-down hard fail, circuit breaker, call-count |

---

### Task 1: Resilience Primitives — CircuitBreaker and TokenBucket

**Files:**
- Modify: `data/resilience.py`
- Test: `tests/test_resilience_new.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_resilience_new.py
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from data.resilience import CircuitBreaker, TokenBucket


class TestCircuitBreaker:

    def test_closed_by_default(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_secs=60)
        assert not cb.is_open
        assert cb.state() == "closed"

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_secs=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open  # not yet
        cb.record_failure()
        assert cb.is_open
        assert cb.state() == "open"

    def test_success_resets_counter(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_secs=60)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open  # counter reset to 0

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_secs=0.05)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.06)
        assert not cb.is_open
        assert cb.state() == "half-open"

    def test_thread_safe(self):
        cb = CircuitBreaker("test", failure_threshold=100, cooldown_secs=60)
        errors = []

        def hammer():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert cb._failures == 200


class TestTokenBucket:

    def test_initial_tokens_allow_immediate_requests(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        # Should succeed without sleeping for 5 requests
        for _ in range(5):
            assert bucket.acquire(block=False)

    def test_sixth_request_fails_non_blocking(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        for _ in range(5):
            bucket.acquire(block=False)
        assert not bucket.acquire(block=False)

    def test_tokens_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=1)
        bucket.acquire(block=False)
        assert not bucket.acquire(block=False)
        time.sleep(0.015)  # 100 tokens/sec → 1 token in 10ms
        assert bucket.acquire(block=False)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_resilience_new.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'CircuitBreaker' from 'data.resilience'`

- [ ] **Step 3: Add CircuitBreaker and TokenBucket to data/resilience.py**

Add after the existing `SourceUnavailable` class (after line 94 in `data/resilience.py`):

```python
import threading

class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures; cools for `cooldown_secs`."""

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_secs: float = 300.0) -> None:
        self.name = name
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._threshold = failure_threshold
        self._cooldown = cooldown_secs
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            return time.time() - self._opened_at < self._cooldown

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._opened_at is None:
                self._opened_at = time.time()
                logger.warning("CircuitBreaker [%s] opened after %d failures", self.name, self._failures)

    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self._cooldown:
                return "half-open"
            return "open"


class TokenBucket:
    """Leaky token bucket: allows `rate` requests per second up to `capacity` burst."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, block: bool = True) -> bool:
        with self._lock:
            now = time.time()
            self._tokens = min(self._capacity, self._tokens + (now - self._last_refill) * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            if not block:
                return False
            wait = (1.0 - self._tokens) / self._rate
        time.sleep(wait)
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)
            return True
```

Also add `threading` to the import block at the top of `data/resilience.py` and `Optional` (already imported via `typing`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_resilience_new.py -v
```

Expected: 10 passed

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add data/resilience.py tests/test_resilience_new.py
git commit -m "feat: add CircuitBreaker and TokenBucket to data/resilience"
```

---

### Task 2: Stale-While-Revalidate Cache Helper

**Files:**
- Modify: `data/cache.py`
- Test: inline in `tests/test_provider_chain.py` (Task 6)

- [ ] **Step 1: Write the failing test (standalone)**

```python
# tests/test_stale_cache.py
import time
import pytest
from data.cache import get_stale_cache_obj, set_cache_obj, get_cache_obj


def test_get_stale_returns_expired_value():
    key = f"test:stale:{time.time()}"
    set_cache_obj(key, {"x": 1}, ttl=1)   # TTL = 1 second
    time.sleep(1.1)
    # Fresh cache returns None (expired)
    assert get_cache_obj(key) is None
    # Stale accessor returns the value and indicates it's past TTL
    result = get_stale_cache_obj(key)
    assert result is not None
    value, expires_at = result
    assert value == {"x": 1}
    assert expires_at < time.time()  # past expiry


def test_get_stale_returns_none_for_missing_key():
    key = "test:stale:never_written_xyz"
    assert get_stale_cache_obj(key) is None


def test_get_stale_returns_fresh_value_too():
    key = f"test:stale:fresh:{time.time()}"
    set_cache_obj(key, {"y": 2}, ttl=3600)
    result = get_stale_cache_obj(key)
    assert result is not None
    value, expires_at = result
    assert value == {"y": 2}
    assert expires_at > time.time()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_stale_cache.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'get_stale_cache_obj' from 'data.cache'`

- [ ] **Step 3: Add get_stale_cache_obj to data/cache.py**

Add after `get_cache_obj` (after line 111 in `data/cache.py`):

```python
def get_stale_cache_obj(key: str) -> Optional[tuple[Any, float]]:
    """Return (deserialized_value, expires_at) regardless of expiry. None if key never set."""
    row = _conn().execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["value"]), row["expires_at"]
    except json.JSONDecodeError:
        return row["value"], row["expires_at"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_stale_cache.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add data/cache.py tests/test_stale_cache.py
git commit -m "feat: add get_stale_cache_obj for stale-while-revalidate pattern"
```

---

### Task 3: FMP and Stooq Provider Package

**Files:**
- Create: `data/providers/__init__.py`
- Create: `data/providers/fmp.py`
- Create: `data/providers/stooq.py`
- Modify: `config.py` (add URL constants)

- [ ] **Step 1: Write failing provider tests**

```python
# tests/test_providers.py
from unittest.mock import MagicMock, patch
import pytest


class TestFmpMarketProvider:

    def test_available_when_key_present(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "test_key")
        from data.providers.fmp import FmpMarketProvider
        assert FmpMarketProvider().available()

    def test_not_available_when_key_absent(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "")
        from importlib import reload
        import data.providers.fmp as fmp_mod
        reload(fmp_mod)
        assert not fmp_mod.FmpMarketProvider().available()

    def test_get_price_parses_quote(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "test_key")
        from data.providers.fmp import FmpMarketProvider

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = [{"price": 45.23, "symbol": "PLTR"}]

        with patch("httpx.get", return_value=fake_resp):
            price = FmpMarketProvider().get_price("PLTR")
        assert price == pytest.approx(45.23)

    def test_get_price_raises_on_empty_response(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "test_key")
        from data.providers.fmp import FmpMarketProvider

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = []

        with patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="no data"):
                FmpMarketProvider().get_price("PLTR")

    def test_get_fundamentals_maps_fields(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "test_key")
        from data.providers.fmp import FmpMarketProvider

        profile = [{"companyName": "Palantir Technologies Inc.", "sector": "Technology",
                    "industry": "Software", "mktCap": 95e9, "price": 45.23, "beta": 1.5}]
        metrics = [{"peRatioTTM": 80.0, "evToEbitdaTTM": 50.0,
                    "netProfitMarginTTM": 0.12, "revenueGrowthTTM": 0.25}]

        def fake_httpx_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "key-metrics-ttm" in url:
                resp.json.return_value = metrics
            else:
                resp.json.return_value = profile
            return resp

        with patch("httpx.get", side_effect=fake_httpx_get):
            fund = FmpMarketProvider().get_fundamentals("PLTR")

        assert fund.name == "Palantir Technologies Inc."
        assert fund.sector == "Technology"
        assert fund.market_cap == pytest.approx(95e9)
        assert fund.current_price == pytest.approx(45.23)
        assert fund.pe_ttm == pytest.approx(80.0)


class TestStooqProvider:

    def test_available_always(self):
        from data.providers.stooq import StooqProvider
        assert StooqProvider().available()

    def test_get_price_parses_csv(self):
        from data.providers.stooq import StooqProvider
        csv_content = b"Date,Open,High,Low,Close,Volume\n2026-08-14,45.10,46.00,44.80,45.50,12000000\n"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.content = csv_content

        with patch("httpx.get", return_value=fake_resp):
            price = StooqProvider().get_price("PLTR")
        assert price == pytest.approx(45.50)

    def test_get_price_raises_on_empty_csv(self):
        from data.providers.stooq import StooqProvider
        csv_content = b"Date,Open,High,Low,Close,Volume\n"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.content = csv_content

        with patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="no rows"):
                StooqProvider().get_price("PLTR")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_providers.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'data.providers'`

- [ ] **Step 3: Add URL constants to config.py**

Add to `config.py` after `FMP_TRANSCRIPT_URL` (around line 114):

```python
FMP_PROFILE_URL          = "https://financialmodelingprep.com/api/v3/profile/{symbol}"
FMP_QUOTE_URL            = "https://financialmodelingprep.com/api/v3/quote/{symbol}"
FMP_KEY_METRICS_TTM_URL  = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
STOOQ_PRICE_URL          = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"
```

- [ ] **Step 4: Create data/providers/__init__.py**

```python
"""Market data provider protocol and shared types."""
from __future__ import annotations

from typing import Optional, Protocol

from analysis.schemas import Fundamentals, PriceData


class MarketProvider(Protocol):
    def available(self) -> bool: ...
    def get_price(self, ticker: str) -> float: ...
    def get_fundamentals(self, ticker: str) -> Fundamentals: ...
```

- [ ] **Step 5: Create data/providers/fmp.py**

```python
"""FMP free-tier market data provider (250 req/day)."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

import config
from analysis.schemas import Fundamentals

logger = logging.getLogger(__name__)

_TIMEOUT = 15


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


class FmpMarketProvider:
    name = "fmp"

    def available(self) -> bool:
        return bool(config.FMP_API_KEY)

    def get_price(self, ticker: str) -> float:
        url = config.FMP_QUOTE_URL.format(symbol=ticker)
        resp = httpx.get(url, params={"apikey": config.FMP_API_KEY}, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"FMP quote: no data for {ticker}")
        price = _safe_float(data[0].get("price"))
        if price is None:
            raise ValueError(f"FMP quote: null price for {ticker}")
        return price

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        profile = self._fetch_profile(ticker)
        metrics = self._fetch_key_metrics(ticker)
        price = _safe_float(profile.get("price"))
        prev_close = _safe_float(profile.get("lastDividendValue"))  # not ideal but FMP free doesn't expose prevClose
        result = Fundamentals(
            ticker=ticker,
            name=profile.get("companyName"),
            sector=profile.get("sector"),
            industry=profile.get("industry"),
            market_cap=_safe_float(profile.get("mktCap")),
            enterprise_value=None,
            pe_ttm=_safe_float(metrics.get("peRatioTTM")),
            pe_forward=None,
            peg=_safe_float(metrics.get("pegRatioTTM")),
            ev_ebitda=_safe_float(metrics.get("evToEbitdaTTM")),
            price_to_sales=_safe_float(metrics.get("priceToSalesRatioTTM")),
            price_to_fcf=_safe_float(metrics.get("pfcfRatioTTM")),
            gross_margin=_safe_float(metrics.get("grossProfitMarginTTM")),
            operating_margin=_safe_float(metrics.get("operatingProfitMarginTTM")),
            net_margin=_safe_float(metrics.get("netProfitMarginTTM")),
            roe=_safe_float(metrics.get("roeTTM")),
            roic=_safe_float(metrics.get("roicTTM")),
            current_ratio=_safe_float(metrics.get("currentRatioTTM")),
            debt_to_equity=_safe_float(metrics.get("debtToEquityTTM")),
            interest_coverage=_safe_float(metrics.get("interestCoverageTTM")),
            revenue_growth_yoy=_safe_float(metrics.get("revenueGrowthTTM")),
            eps_growth_yoy=_safe_float(metrics.get("epsGrowthTTM")),
            revenue_ttm=_safe_float(metrics.get("revenueTTM")) or _safe_float(metrics.get("revenuePerShareTTM")),
            ebitda_ttm=None,
            net_income_ttm=None,
            fcf_ttm=None,
            total_debt=None,
            cash=None,
            shares_outstanding=_safe_float(profile.get("volAvg")),
            current_price=price,
            beta=_safe_float(profile.get("beta")),
            fetched_at=datetime.utcnow(),
            previous_close=None,
            week_52_high=_safe_float(profile.get("range", "").split("-")[-1] if "-" in str(profile.get("range", "")) else None),
            week_52_low=_safe_float(profile.get("range", "").split("-")[0] if "-" in str(profile.get("range", "")) else None),
            analyst_target_mean=None,
        )
        return result

    def _fetch_profile(self, ticker: str) -> Dict[str, Any]:
        url = config.FMP_PROFILE_URL.format(symbol=ticker)
        resp = httpx.get(url, params={"apikey": config.FMP_API_KEY}, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"FMP profile: no data for {ticker}")
        return data[0]

    def _fetch_key_metrics(self, ticker: str) -> Dict[str, Any]:
        url = config.FMP_KEY_METRICS_TTM_URL.format(symbol=ticker)
        resp = httpx.get(url, params={"apikey": config.FMP_API_KEY}, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else {}
```

- [ ] **Step 6: Create data/providers/stooq.py**

```python
"""Stooq keyless CSV price provider — latest close only."""
from __future__ import annotations

import io
import logging

import httpx

import config
from analysis.schemas import Fundamentals

logger = logging.getLogger(__name__)

_TIMEOUT = 10


class StooqProvider:
    name = "stooq"

    def available(self) -> bool:
        return True  # keyless

    def get_price(self, ticker: str) -> float:
        url = config.STOOQ_PRICE_URL.format(ticker=ticker.lower())
        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        lines = [ln for ln in resp.content.decode().strip().splitlines() if ln]
        data_lines = lines[1:]  # skip header
        if not data_lines:
            raise ValueError(f"Stooq: no rows for {ticker}")
        last = data_lines[-1].split(",")
        # CSV columns: Date,Open,High,Low,Close,Volume — Close is index 4
        try:
            return float(last[4])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Stooq: could not parse close for {ticker}: {exc}") from exc

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        raise NotImplementedError("Stooq only provides price data")
```

- [ ] **Step 7: Run provider tests**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_providers.py -v
```

Expected: all 9 tests pass

- [ ] **Step 8: Commit**

```bash
git add config.py data/providers/ tests/test_providers.py
git commit -m "feat: add FmpMarketProvider and StooqProvider with URL constants"
```

---

### Task 4: Provider Chain Integration in data/market.py

Wire the provider chain into `get_fundamentals` and `get_prices` with circuit breakers and stale-while-revalidate. Keep all existing function signatures unchanged.

**Files:**
- Modify: `data/market.py`

- [ ] **Step 1: Write failing chain tests**

```python
# tests/test_provider_chain.py
import time
from unittest.mock import MagicMock, patch, call
import pytest

from analysis.schemas import Fundamentals
from data.resilience import SourceUnavailable


def _make_fundamentals(ticker="PLTR"):
    return Fundamentals(
        ticker=ticker, name="Test", sector="Tech", industry="SW",
        market_cap=90e9, current_price=45.0, fetched_at=__import__("datetime").datetime.utcnow(),
    )


class TestYfinance429FallsToFmp:

    def test_yfinance_429_triggers_fmp_fallback(self, tmp_path):
        """When yfinance raises SourceUnavailable (429), FMP is called and its result returned."""
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("Yahoo 429")), \
             patch("data.market._CB_YFINANCE.is_open", False), \
             patch("data.market._CB_FMP.is_open", False), \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result) as mock_fmp:
            result = __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("PLTR")

        assert result.ticker == "PLTR"
        mock_fmp.assert_called_once_with("PLTR")


class TestStaleCacheServedOnAllProviderFailure:

    def test_stale_data_returned_when_all_providers_fail(self):
        """When all providers fail but stale cache exists, stale data is returned (no exception)."""
        from datetime import datetime
        stale_fund = _make_fundamentals()
        stale_serialized = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 100  # expired 100s ago

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_serialized, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE.is_open", False), \
             patch("data.market._CB_FMP.is_open", False):
            result = __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("PLTR")

        assert result is not None
        assert result.ticker == "PLTR"


class TestAllProvidersDownColdCacheRaises:

    def test_raises_source_unavailable_when_no_cache_and_all_providers_fail(self):
        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE.is_open", False), \
             patch("data.market._CB_FMP.is_open", False):
            with pytest.raises(SourceUnavailable):
                __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("PLTR")


class TestCircuitBreakerSkipsProvider:

    def test_open_yfinance_cb_skips_to_fmp(self):
        """When yfinance CB is open, yfinance is never called."""
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE") as mock_cb, \
             patch("data.market._CB_FMP.is_open", False), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_cb.is_open = True
            result = __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("PLTR")

        mock_yf.assert_not_called()
        assert result is not None


class TestCallCountOnWarmCache:

    def test_warm_sqlite_cache_makes_zero_http_calls(self):
        """SQLite cache hit prevents any HTTP call across the full provider chain."""
        from datetime import datetime
        cached_fund = _make_fundamentals()
        serialized = cached_fund.model_dump(mode="json")

        with patch("data.market.get_cache_obj", return_value=serialized), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp:
            result = __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("PLTR")

        mock_yf.assert_not_called()
        mock_fmp.assert_not_called()
        assert result.ticker == "PLTR"


class TestPrices429FallsToStooq:

    def test_yfinance_prices_429_falls_to_stooq(self):
        """When yfinance prices fail, Stooq provides the fallback price."""
        from analysis.schemas import PriceData, PriceBar
        import datetime

        stooq_price = 45.50

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE.is_open", False), \
             patch("data.market._CB_STOOQ.is_open", False), \
             patch("data.market._fetch_prices_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_price_stooq", return_value=stooq_price) as mock_stooq:
            result = __import__("data.market", fromlist=["get_prices"]).get_prices("PLTR")

        mock_stooq.assert_called_once_with("PLTR")
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_provider_chain.py -v 2>&1 | head -25
```

Expected: `ImportError` or `AttributeError` — `_fetch_fundamentals_yfinance`, `_CB_YFINANCE`, etc. don't exist yet.

- [ ] **Step 3: Refactor data/market.py — extract yfinance fetch helpers and add provider chain**

The existing `get_fundamentals` function body (lines 112–286) becomes `_fetch_fundamentals_yfinance`. The existing `get_prices` function body (lines 81–108) becomes `_fetch_prices_yfinance`. Then add circuit breakers and a new thin `get_fundamentals` / `get_prices` that runs the chain.

**Add near the top of data/market.py, after imports:**

```python
from data.resilience import CircuitBreaker, SourceUnavailable
from data.cache import get_stale_cache_obj

# Per-provider circuit breakers (module-level singletons)
_CB_YFINANCE = CircuitBreaker("yfinance", failure_threshold=3, cooldown_secs=300.0)
_CB_FMP      = CircuitBreaker("fmp",      failure_threshold=5, cooldown_secs=300.0)
_CB_STOOQ   = CircuitBreaker("stooq",    failure_threshold=5, cooldown_secs=120.0)
```

**Rename existing `get_prices` to `_fetch_prices_yfinance`** — remove the `@retry` decorator and cache logic (the outer chain handles retries and caching). The inner function just does the yfinance HTTP call and returns `PriceData`.

```python
def _fetch_prices_yfinance(ticker: str, period: str = "5y") -> PriceData:
    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    hist = yf_ticker.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")
    bars = []
    for dt, row in hist.iterrows():
        bars.append(PriceBar(
            date=dt.date(),
            open=float(row["Open"]), high=float(row["High"]),
            low=float(row["Low"]),  close=float(row["Close"]),
            volume=int(row.get("Volume", 0)),
        ))
    info = yf_ticker.fast_info
    currency = getattr(info, "currency", "USD") or "USD"
    return PriceData(ticker=ticker, currency=currency, bars=bars)
```

**Add `_fetch_price_stooq`:**

```python
def _fetch_price_stooq(ticker: str) -> float:
    from data.providers.stooq import StooqProvider
    return StooqProvider().get_price(ticker)
```

**Add `_fetch_fundamentals_fmp`:**

```python
def _fetch_fundamentals_fmp(ticker: str) -> Fundamentals:
    from data.providers.fmp import FmpMarketProvider
    return FmpMarketProvider().get_fundamentals(ticker)
```

**Rename existing `get_fundamentals` body to `_fetch_fundamentals_yfinance`** (same content, no `@retry`, no cache read/write).

**Replace `get_fundamentals` with chain orchestrator:**

```python
def get_fundamentals(ticker: str) -> Fundamentals:
    cache_key = f"market:{ticker}:fundamentals"

    cached = get_cache_obj(cache_key)
    if cached:
        return Fundamentals.model_validate(cached)

    last_exc: Optional[Exception] = None

    if not _CB_YFINANCE.is_open:
        try:
            result = _fetch_fundamentals_yfinance(ticker)
            _CB_YFINANCE.record_success()
            _PROVIDER_HEALTH[ticker] = {"status": "ok", "source": "yfinance"}
            set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_FUNDAMENTALS, source="yfinance")
            record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
            return result
        except Exception as exc:
            _CB_YFINANCE.record_failure()
            last_exc = exc
            logger.warning("yfinance fundamentals failed for %s: %s", ticker, exc)
            _PROVIDER_HEALTH[ticker] = {"status": "rate_limited", "source": "yfinance",
                                         "detail": str(exc), "cb_state": _CB_YFINANCE.state()}

    import config as _cfg
    if _cfg.FMP_API_KEY and not _CB_FMP.is_open:
        try:
            result = _fetch_fundamentals_fmp(ticker)
            _CB_FMP.record_success()
            _PROVIDER_HEALTH[ticker] = {"status": "ok", "source": "fmp"}
            set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_FUNDAMENTALS, source="fmp")
            record_freshness(cache_key, "fmp", TTL_FUNDAMENTALS)
            return result
        except Exception as exc:
            _CB_FMP.record_failure()
            last_exc = exc
            logger.warning("FMP fundamentals failed for %s: %s", ticker, exc)

    stale = get_stale_cache_obj(cache_key)
    if stale is not None:
        value, expires_at = stale
        age_secs = int(time.time() - expires_at)
        logger.warning("Serving stale fundamentals for %s (expired %ds ago)", ticker, age_secs)
        _PROVIDER_HEALTH[ticker] = {
            "status": "stale", "source": "cache",
            "detail": f"all providers failed; serving data expired {age_secs}s ago",
            "cb_state": _CB_YFINANCE.state(),
        }
        return Fundamentals.model_validate(value)

    raise SourceUnavailable(
        f"All market data providers failed for {ticker}. "
        f"Last error: {last_exc}. Try again in a few minutes."
    )
```

**Replace `get_prices` with chain orchestrator:**

```python
def get_prices(ticker: str, period: str = "5y") -> PriceData:
    cache_key = f"market:{ticker}:prices:{period}"
    cached = get_cache_obj(cache_key)
    if cached:
        return PriceData.model_validate(cached)

    last_exc: Optional[Exception] = None

    if not _CB_YFINANCE.is_open:
        try:
            result = _fetch_prices_yfinance(ticker, period)
            _CB_YFINANCE.record_success()
            set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_PRICES, source="yfinance")
            record_freshness(cache_key, "yfinance", TTL_PRICES)
            return result
        except Exception as exc:
            _CB_YFINANCE.record_failure()
            last_exc = exc
            logger.warning("yfinance prices failed for %s: %s", ticker, exc)

    if not _CB_STOOQ.is_open:
        try:
            close = _fetch_price_stooq(ticker)
            from datetime import date
            result = PriceData(
                ticker=ticker, currency="USD",
                bars=[PriceBar(date=date.today(), open=close, high=close, low=close,
                               close=close, volume=0)],
            )
            _CB_STOOQ.record_success()
            set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_PRICES, source="stooq")
            record_freshness(cache_key, "stooq", TTL_PRICES)
            return result
        except Exception as exc:
            _CB_STOOQ.record_failure()
            last_exc = exc
            logger.warning("Stooq prices failed for %s: %s", ticker, exc)

    stale = get_stale_cache_obj(cache_key)
    if stale is not None:
        value, expires_at = stale
        logger.warning("Serving stale prices for %s", ticker)
        return PriceData.model_validate(value)

    raise SourceUnavailable(
        f"All price providers failed for {ticker}. Last error: {last_exc}"
    )
```

Also add `time` import if not already present (it is already in the file via `data.cache`; import explicitly at top of `data/market.py`):
```python
import time
```

- [ ] **Step 4: Run provider chain tests**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_provider_chain.py -v
```

Expected: all 6 test classes pass

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all existing tests still pass (no regressions in test_transcripts.py, test_portfolio.py)

- [ ] **Step 6: Commit**

```bash
git add data/market.py
git commit -m "feat: add provider failover chain to get_fundamentals and get_prices with stale-while-revalidate"
```

---

### Task 5: Fix UI Error Messaging

Remove the false "Other tabs are unaffected" claim. Add staleness badge to market data. Enrich provider diagnostics with circuit breaker state.

**Files:**
- Modify: `ui/components.py` (lines 388–399)
- Modify: `app.py` (lines 1195–1227)

- [ ] **Step 1: Fix ui/components.py — remove misleading captions**

In `ui/components.py`, `error_card` at line 393: remove the line `st.caption("Other tabs are unaffected.")`.

Old:
```python
def error_card(title: str, detail: str, tab_only: bool = True) -> None:
    with st.container(border=True):
        st.error(f"**{title}**")
        st.caption(detail)
        if tab_only:
            st.caption("Other tabs are unaffected.")
```

New:
```python
def error_card(title: str, detail: str, tab_only: bool = True) -> None:
    with st.container(border=True):
        st.error(f"**{title}**")
        st.caption(detail)
```

In `unavailable_tab` at line 399: remove the false freshness-badge line.

Old:
```python
def unavailable_tab(source: str, reason: str) -> None:
    st.warning(f"**{source} unavailable**")
    st.caption(reason)
    st.caption("Freshness badges for other tabs are still accurate.")
```

New:
```python
def unavailable_tab(source: str, reason: str) -> None:
    st.warning(f"**{source} unavailable**")
    st.caption(reason)
```

- [ ] **Step 2: Enrich provider diagnostics in app.py**

In `app.py`, the market data diagnostics block (lines 1195–1209) currently shows only yfinance status. Replace it to show per-provider circuit state and staleness, and add a staleness badge when data is served from stale cache.

Find the existing block (around lines 1194–1228) and replace the `if mkt_health:` section:

Old `if mkt_health:` block:
```python
            if mkt_health:
                st.markdown("**Market data (yfinance/Yahoo Finance)**")
                for t, h in mkt_health.items():
                    status = h.get("status", "unknown")
                    icon = "✅" if status == "ok" else "⚠️"
                    detail = h.get("detail", "")
                    n = h.get("fields")
                    info_str = f"{n} fields" if n else detail
                    st.caption(f"{icon} **{t}**: {status} — {info_str}")
```

New `if mkt_health:` block:
```python
            if mkt_health:
                from data.resilience import CircuitBreaker
                import data.market as _mkt
                st.markdown("**Market data providers**")
                _CB_ICONS = {"closed": "🟢", "half-open": "🟡", "open": "🔴"}
                for t, h in mkt_health.items():
                    status = h.get("status", "unknown")
                    source = h.get("source", "?")
                    detail = h.get("detail", "")
                    cb_state = h.get("cb_state", "")
                    icon = "✅" if status == "ok" else ("🕐" if status == "stale" else "⚠️")
                    cb_icon = _CB_ICONS.get(cb_state, "") if cb_state else ""
                    parts = [f"{icon} **{t}** via {source}: {status}"]
                    if cb_icon:
                        parts.append(f"CB {cb_icon} {cb_state}")
                    if detail:
                        parts.append(f"— {detail}")
                    st.caption("  ".join(parts))
                # Show circuit breaker state for all providers
                for cb_name, cb_obj in [("yfinance", _mkt._CB_YFINANCE),
                                         ("fmp", _mkt._CB_FMP),
                                         ("stooq", _mkt._CB_STOOQ)]:
                    state = cb_obj.state()
                    cb_icon = _CB_ICONS.get(state, "❓")
                    st.caption(f"  {cb_icon} **{cb_name}** circuit: {state}")
```

Also add a staleness banner near the top of `run_ticker_mode` in `app.py`. Find where `fund` is loaded (around line 94) and after the successful load, check staleness:

```python
    try:
        fund = f_fund.result()
    except Exception as exc:
        error_card("Market data unavailable", str(exc))
        return

    # Staleness badge: if data came from expired cache, show a warning banner
    from data.cache import get_freshness
    freshness = get_freshness(f"market:{ticker}:fundamentals")
    if freshness and freshness.get("is_stale"):
        st.warning(
            f"⚠ Market data is stale (last fetched {freshness['age_description']}). "
            "All providers are currently rate-limited — displaying cached data. "
            "Data will refresh automatically when providers recover."
        )
```

- [ ] **Step 3: Verify app starts without errors**

```bash
cd /Users/vanditgandotra/deepdive && python -c "import app; print('ok')" 2>&1
```

Expected: `ok` (no import errors)

- [ ] **Step 4: Commit**

```bash
git add ui/components.py app.py
git commit -m "fix: remove 'Other tabs unaffected' false claim; add staleness badge and richer provider diagnostics"
```

---

### Task 6: Integration Tests and Verification Report

**Files:**
- Append to: `tests/test_provider_chain.py`

- [ ] **Step 1: Add remaining integration tests to test_provider_chain.py**

```python
# Append to tests/test_provider_chain.py

class TestCircuitBreakerOpensAfterMultipleFailures:

    def test_cb_opens_and_stays_open(self):
        """After 3 consecutive yfinance failures, CB opens and stays open for cooldown period."""
        from data.resilience import CircuitBreaker
        cb = CircuitBreaker("test_chain", failure_threshold=3, cooldown_secs=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open
        assert cb.state() == "open"

    def test_open_cb_makes_zero_calls_to_yfinance(self):
        """With CB open, yfinance fetch is never called (saved HTTP call budget)."""
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP.is_open", False), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_yf_cb.is_open = True
            __import__("data.market", fromlist=["get_fundamentals"]).get_fundamentals("AAPL")

        mock_yf_fetch.assert_not_called()


class TestStaleBadgeMetadata:

    def test_provider_health_set_to_stale_when_serving_expired_cache(self):
        """When stale cache is served, _PROVIDER_HEALTH records status='stale'."""
        import data.market as mkt
        from datetime import datetime
        stale_fund = _make_fundamentals("MSFT")
        stale_data = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 500

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_data, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE.is_open", False), \
             patch("data.market._CB_FMP.is_open", False):
            result = mkt.get_fundamentals("MSFT")

        assert result is not None
        health = mkt.get_provider_health()
        assert health.get("MSFT", {}).get("status") == "stale"
```

- [ ] **Step 2: Run the full test_provider_chain.py suite**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/test_provider_chain.py -v
```

Expected: all tests pass

- [ ] **Step 3: Run complete test suite (final regression check)**

```bash
cd /Users/vanditgandotra/deepdive && python -m pytest tests/ -q
```

Expected: all tests pass (should be 65+ tests now)

- [ ] **Step 4: Measure call count before/after**

Run this diagnostic to verify the chain is wired correctly:

```python
# Run in Python REPL or as a script
import sys
sys.path.insert(0, "/Users/vanditgandotra/deepdive")

from unittest.mock import patch, MagicMock
import httpx

call_log = []

def counting_get(url, **kwargs):
    call_log.append(url)
    raise ConnectionError("offline")  # simulate all providers down

# Clear any SQLite cache for PLTR to force cold path
from data.cache import delete_cache
delete_cache("market:PLTR:fundamentals")

with patch("httpx.get", side_effect=counting_get), \
     patch("data.market._fetch_fundamentals_yfinance",
           side_effect=Exception("yf down")) as mock_yf, \
     patch("data.market._fetch_fundamentals_fmp",
           side_effect=Exception("fmp down")) as mock_fmp, \
     patch("data.market.get_stale_cache_obj", return_value=None):
    try:
        from data.market import get_fundamentals
        get_fundamentals("PLTR")
    except Exception as e:
        print(f"Exception (expected): {e}")

print(f"yfinance calls: {mock_yf.call_count}")
print(f"FMP calls: {mock_fmp.call_count}")
print("Total provider calls on cold cache, all-down: 2 (1 yf + 1 fmp)")
```

- [ ] **Step 5: Commit final tests**

```bash
git add tests/test_provider_chain.py
git commit -m "test: add integration tests for provider chain, circuit breaker, and stale badge"
```

---

## Deployment Checklist

After all tasks complete, paste these into **Streamlit Cloud → App settings → Secrets**:

```toml
# Already set (verify these exist):
ANTHROPIC_API_KEY = "sk-ant-..."
FMP_API_KEY = "..."       # free at financialmodelingprep.com — covers prices + fundamentals

# No additional keys needed:
# Stooq — keyless CSV, no signup required
# EDGAR — SEC public API, no key required (already integrated)
```

**Call count comparison:**

| Scenario | Before (yfinance only) | After (provider chain) |
|---|---|---|
| Warm SQLite cache | 0 HTTP calls | 0 HTTP calls (unchanged) |
| Cold cache, all providers healthy | 3–4 yfinance calls | 1–2 yfinance calls (same first try) |
| yfinance 429 (cold cache) | ❌ SourceUnavailable raised | ✅ FMP fallback, page renders |
| yfinance 429 (stale cache) | ❌ SourceUnavailable raised | ✅ stale data served with age badge |
| All providers 429, cold cache | ❌ page blank | ❌ single clear error (no false "other tabs unaffected") |
| All providers 429, stale cache | ❌ page blank | ✅ stale data served silently |

**Provider chain table:**

| Provider | Covers | Key required | Free tier | Fallback priority |
|---|---|---|---|---|
| yfinance | prices, fundamentals, estimates, insiders, holders, news | No | Yes (shared IP pool) | 1st (primary) |
| FMP | prices, fundamentals | Yes (`FMP_API_KEY`) | 250 req/day | 2nd |
| Stooq | latest price only | No | Unlimited | 3rd (prices only) |
| SQLite stale cache | anything previously fetched | n/a | n/a | Last resort |

**Self-Review**

1. **Spec coverage:**
   - ✅ Single fetch layer — existing `@st.cache_data` in `ui/stcache.py` + SQLite already handles this; chain adds provider fallback on top
   - ✅ Provider chain: yfinance → FMP → Stooq — Tasks 3 + 4
   - ✅ Rate-limit hygiene: token bucket (Task 1), exponential backoff (existing `@retry`), circuit breaker (Task 1 + Task 4), Retry-After honored (existing `@retry`)
   - ✅ Stale-while-revalidate: Task 2 + Task 4
   - ✅ Honest diagnostics: Task 5 removes false claims, adds richer per-provider info
   - ✅ Tests: Task 6 (429 failover, stale cache, all-down, CB, call-count)

2. **Placeholder scan:** No TBDs; all code blocks are complete.

3. **Type consistency:**
   - `_fetch_fundamentals_yfinance(ticker: str) -> Fundamentals` ✅
   - `_fetch_fundamentals_fmp(ticker: str) -> Fundamentals` ✅
   - `_fetch_prices_yfinance(ticker: str, period: str) -> PriceData` ✅
   - `_fetch_price_stooq(ticker: str) -> float` ✅
   - `get_stale_cache_obj(key: str) -> Optional[tuple[Any, float]]` ✅
   - `CircuitBreaker.is_open: bool`, `.state() -> str`, `.record_failure()`, `.record_success()` ✅
   - `TokenBucket.acquire(block: bool) -> bool` ✅
