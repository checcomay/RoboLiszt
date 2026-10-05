from __future__ import annotations

from pathlib import Path
import csv
import numpy as np


def save_timeseries(path: str | Path, t, columns: dict[str, np.ndarray]) -> Path:
    """Save numeric time-series data as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(columns)
    n = len(t)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *keys])
        for i in range(n):
            writer.writerow([float(t[i]), *[float(np.asarray(columns[k])[i]) for k in keys]])

    return path
