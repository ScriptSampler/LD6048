"""Small helpers for writing results consistently."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RESULTS


def _to_native(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_to_native(v) for v in o.tolist()]
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def save_json(obj: Any, name: str, subdir: str | None = None) -> Path:
    d = RESULTS / subdir if subdir else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_to_native(obj), f, indent=2, ensure_ascii=False)
    return p


def save_csv(df: pd.DataFrame, name: str, subdir: str | None = None, index: bool = False,
             float_format: str = "%.6g") -> Path:
    d = RESULTS / subdir if subdir else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    df.to_csv(p, index=index, float_format=float_format)
    return p


def load_json(name: str, subdir: str | None = None) -> Any:
    d = RESULTS / subdir if subdir else RESULTS
    with open(d / name, encoding="utf-8") as f:
        return json.load(f)
