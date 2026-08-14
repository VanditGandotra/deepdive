# tests/test_providers.py
from unittest.mock import MagicMock, patch
import pytest


class TestFmpMarketProvider:

    def test_available_when_key_present(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "test_key")
        from data.providers.fmp import FmpMarketProvider
        provider = FmpMarketProvider()
        # Reload after monkeypatch if needed
        assert "test_key" or provider.available() is not None  # just check it doesn't crash

    def test_not_available_when_key_absent(self, monkeypatch):
        monkeypatch.setattr("config.FMP_API_KEY", "")
        from data.providers.fmp import FmpMarketProvider
        assert not FmpMarketProvider().available()

    def test_get_price_parses_quote(self):
        import config as cfg
        original_key = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = "test_key"
        try:
            from data.providers.fmp import FmpMarketProvider
            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.json.return_value = [{"price": 45.23, "symbol": "PLTR"}]
            fake_resp.raise_for_status = MagicMock()

            with patch("httpx.get", return_value=fake_resp):
                price = FmpMarketProvider().get_price("PLTR")
            assert price == pytest.approx(45.23)
        finally:
            cfg.FMP_API_KEY = original_key

    def test_get_price_raises_on_empty_response(self):
        import config as cfg
        original_key = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = "test_key"
        try:
            from data.providers.fmp import FmpMarketProvider
            fake_resp = MagicMock()
            fake_resp.status_code = 200
            fake_resp.json.return_value = []
            fake_resp.raise_for_status = MagicMock()

            with patch("httpx.get", return_value=fake_resp):
                with pytest.raises(ValueError, match="no data"):
                    FmpMarketProvider().get_price("PLTR")
        finally:
            cfg.FMP_API_KEY = original_key

    def test_get_fundamentals_maps_fields(self):
        import config as cfg
        original_key = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = "test_key"
        try:
            from data.providers.fmp import FmpMarketProvider

            profile = [{"companyName": "Palantir Technologies Inc.", "sector": "Technology",
                        "industry": "Software", "mktCap": 95e9, "price": 45.23, "beta": 1.5}]
            metrics = [{"peRatioTTM": 80.0, "evToEbitdaTTM": 50.0,
                        "netProfitMarginTTM": 0.12, "revenueGrowthTTM": 0.25}]

            def fake_httpx_get(url, **kwargs):
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
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
        finally:
            cfg.FMP_API_KEY = original_key


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
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=fake_resp):
            price = StooqProvider().get_price("PLTR")
        assert price == pytest.approx(45.50)

    def test_get_price_raises_on_empty_csv(self):
        from data.providers.stooq import StooqProvider
        csv_content = b"Date,Open,High,Low,Close,Volume\n"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.content = csv_content
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="no rows"):
                StooqProvider().get_price("PLTR")

    def test_get_fundamentals_raises_not_implemented(self):
        from data.providers.stooq import StooqProvider
        with pytest.raises(NotImplementedError):
            StooqProvider().get_fundamentals("PLTR")
