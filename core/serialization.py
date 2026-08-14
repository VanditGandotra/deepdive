"""JSON-safe serialization helpers.

to_jsonable() converts any value produced by NumPy / Pandas / SciPy into a
type that json.dumps can handle.  It is the safety net — the source of truth
is that every boundary (OptimizeResult fields, payload dicts) should already
use Python-native types.  Use to_jsonable() + allow_nan=False everywhere.

NumPy 2.x note: isinstance(np.bool_(), bool) is False, and np.bool_.__name__
is 'bool', so json.dumps raises "Object of type bool is not JSON serializable"
when it encounters np.bool_ — even though the error message looks like a Python
bool.  to_jsonable() handles this before json.dumps ever sees it.
"""
from __future__ import annotations

import dataclasses
import enum
import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce obj to JSON-safe Python types.

    Conversion table:
      np.bool_                    → bool
      np.integer                  → int
      np.floating / float (±Inf, NaN) → None  (JSON has no NaN/Inf)
      float (finite)              → float  (unchanged)
      np.ndarray                  → list   (recursively converted)
      pd.Series                   → {str(k): v} dict
      pd.DataFrame                → {col: {str(idx): v}} dict
      pd.Index                    → list
      Decimal                     → float
      set / frozenset             → sorted list
      dict                        → {str(k): to_jsonable(v)}
      list / tuple                → list   (recursively converted)
      dataclass instance          → dict   (via dataclasses.asdict)
      Enum                        → .value
      everything else             → unchanged (int, str, None, …)
    """
    # bool must be checked before int — Python bool is a subclass of int,
    # but np.bool_ is NOT a subclass of bool (NumPy 2.x).
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {
            str(col): {str(idx): to_jsonable(val) for idx, val in obj[col].items()}
            for col in obj.columns
        }
    if isinstance(obj, pd.Index):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj
