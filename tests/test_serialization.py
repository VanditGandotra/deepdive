"""Tests for core/serialization.py — to_jsonable() round-trip safety."""
from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from core.serialization import to_jsonable


def _roundtrip(obj):
    """Convert to jsonable, dump with allow_nan=False, load back."""
    safe = to_jsonable(obj)
    return json.loads(json.dumps(safe, allow_nan=False))


class TestPrimitives:
    def test_python_bool_unchanged(self):
        assert to_jsonable(True) is True
        assert to_jsonable(False) is False

    def test_python_int_unchanged(self):
        assert to_jsonable(42) == 42
        assert isinstance(to_jsonable(42), int)

    def test_python_float_unchanged(self):
        v = to_jsonable(3.14)
        assert v == pytest.approx(3.14)
        assert isinstance(v, float)

    def test_python_string_unchanged(self):
        assert to_jsonable("hello") == "hello"

    def test_none_unchanged(self):
        assert to_jsonable(None) is None


class TestNumpyScalars:
    def test_np_bool_true(self):
        v = to_jsonable(np.bool_(True))
        assert v is True
        assert isinstance(v, bool)

    def test_np_bool_false(self):
        v = to_jsonable(np.bool_(False))
        assert v is False
        assert isinstance(v, bool)

    def test_np_bool_json_safe(self):
        result = _roundtrip({"flag": np.bool_(True)})
        assert result == {"flag": True}

    def test_np_int8(self):
        assert to_jsonable(np.int8(5)) == 5
        assert isinstance(to_jsonable(np.int8(5)), int)

    def test_np_int64(self):
        assert to_jsonable(np.int64(10**9)) == 10**9
        assert isinstance(to_jsonable(np.int64(1)), int)

    def test_np_float32(self):
        v = to_jsonable(np.float32(1.5))
        assert isinstance(v, float)
        assert v == pytest.approx(1.5, abs=1e-4)

    def test_np_float64(self):
        v = to_jsonable(np.float64(2.718))
        assert isinstance(v, float)
        assert v == pytest.approx(2.718)

    def test_np_nan_becomes_none(self):
        assert to_jsonable(np.float64(float("nan"))) is None

    def test_np_inf_becomes_none(self):
        assert to_jsonable(np.float64(float("inf"))) is None

    def test_np_neginf_becomes_none(self):
        assert to_jsonable(np.float64(float("-inf"))) is None

    def test_python_nan_becomes_none(self):
        assert to_jsonable(float("nan")) is None

    def test_python_inf_becomes_none(self):
        assert to_jsonable(float("inf")) is None


class TestNumpyArrays:
    def test_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _roundtrip(arr)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_2d_array(self):
        arr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = _roundtrip(arr)
        assert result == [[1.0, 0.5], [0.5, 1.0]]

    def test_bool_array(self):
        arr = np.array([True, False, True])
        result = _roundtrip(arr)
        assert result == [True, False, True]
        assert all(isinstance(v, bool) for v in result)

    def test_nan_in_array_becomes_none(self):
        arr = np.array([1.0, float("nan"), 3.0])
        result = to_jsonable(arr)
        assert result[1] is None
        # Must round-trip with allow_nan=False
        json.dumps(result, allow_nan=False)

    def test_integer_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = _roundtrip(arr)
        assert result == [1, 2, 3]


class TestNestedStructures:
    def test_dict_with_np_values(self):
        d = {
            "a": np.float64(0.4),
            "b": np.bool_(True),
            "c": np.int32(7),
        }
        result = _roundtrip(d)
        assert result == {"a": pytest.approx(0.4), "b": True, "c": 7}

    def test_non_string_dict_keys_become_strings(self):
        d = {0: "zero", 1: "one"}
        result = to_jsonable(d)
        assert "0" in result
        assert "1" in result

    def test_nested_dict(self):
        d = {"outer": {"inner": np.float64(1.23)}}
        result = _roundtrip(d)
        assert result["outer"]["inner"] == pytest.approx(1.23)

    def test_list_of_mixed(self):
        lst = [np.float64(1.0), np.bool_(False), 42, "text"]
        result = _roundtrip(lst)
        assert result == [pytest.approx(1.0), False, 42, "text"]

    def test_tuple_becomes_list(self):
        t = (np.float64(1.0), np.int64(2))
        result = to_jsonable(t)
        assert isinstance(result, list)
        assert result == pytest.approx([1.0, 2])


class TestPandasObjects:
    def test_series(self):
        s = pd.Series({"a": 1.0, "b": 2.0})
        result = to_jsonable(s)
        assert isinstance(result, dict)
        assert _roundtrip(s) == {"a": 1.0, "b": 2.0}

    def test_index(self):
        idx = pd.Index(["x", "y", "z"])
        result = to_jsonable(idx)
        assert result == ["x", "y", "z"]


class TestSpecialTypes:
    def test_decimal(self):
        d = Decimal("3.14159")
        result = to_jsonable(d)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14159)

    def test_set_becomes_sorted_list(self):
        s = {"b", "a", "c"}
        result = to_jsonable(s)
        assert isinstance(result, list)
        assert set(result) == {"a", "b", "c"}

    def test_frozenset(self):
        fs = frozenset([3, 1, 2])
        result = to_jsonable(fs)
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_dataclass(self):
        import dataclasses

        @dataclasses.dataclass
        class Foo:
            x: np.float64
            y: np.bool_

        f = Foo(x=np.float64(1.5), y=np.bool_(True))
        result = _roundtrip(f)
        assert result == {"x": pytest.approx(1.5), "y": True}

    def test_enum(self):
        import enum

        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        assert to_jsonable(Color.RED) == "red"


class TestOptimizerPayloadShape:
    """Simulate the exact payload structure _build_optimizer_payload produces."""

    def _make_payload(self):
        proposed_arr = np.array([0.4, 0.35, 0.25])
        return {
            "method": "max_sharpe",
            "risk_free_rate_pct": round(np.float64(5.0), 2),
            "n_assets": 3,
            "constraints": {
                "max_position_pct": round(np.float64(40.0), 1),
                "binding": ["AAPL at position cap (40%)"],
            },
            "holdings": [
                {
                    "ticker": "AAPL",
                    "expected_return_pct": round(np.float64(12.34), 2),
                    "at_position_cap": bool(proposed_arr[0] >= 0.40 - 1e-3),
                    "risk_contribution_pct": round(np.float64(33.3), 2),
                }
            ],
            "portfolio": {
                "sharpe": round(np.float64(1.234), 4),
                "expected_return_pct": round(np.float64(14.5), 2),
                "vol_pct": round(np.float64(18.2), 2),
            },
            "sensitivity": [
                {
                    "ticker": "AAPL",
                    "return_shock_pp": round(np.float64(2.0), 2),
                    "weight_before_pct": round(np.float64(40.0), 2),
                    "weight_after_pct": round(np.float64(40.0), 2),
                    "weight_delta_pp": round(np.float64(0.0), 2),
                }
            ],
        }

    def test_payload_roundtrip(self):
        payload = self._make_payload()
        safe = to_jsonable(payload)
        json.dumps(safe, allow_nan=False)

    def test_np_bool_in_at_position_cap(self):
        # Specifically test the np.bool_ case that caused the reported crash
        proposed_arr = np.array([0.4, 0.35, 0.25])
        raw_flag = proposed_arr[0] >= 0.40 - 1e-3  # np.bool_
        assert isinstance(raw_flag, np.bool_)
        assert not isinstance(raw_flag, bool)
        # to_jsonable must convert it
        safe_flag = to_jsonable(raw_flag)
        assert isinstance(safe_flag, bool)
        json.dumps({"flag": safe_flag}, allow_nan=False)

    def test_nan_in_sharpe_becomes_none(self):
        payload = {"sharpe": float("nan")}
        safe = to_jsonable(payload)
        assert safe["sharpe"] is None
        json.dumps(safe, allow_nan=False)
