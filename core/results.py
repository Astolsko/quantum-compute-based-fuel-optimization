"""
core/results.py  --  every script's numbers, saved as JSON next to its plots.

The READMEs used to carry hand-copied numbers, which drifted from the code. Now
each script also writes what it printed to <outputs>/results/<name>.json, and
tools/make_report.py builds the results tables from those files. A table in a
README is therefore always traceable to one run folder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import config


def _plain(x):
    """numpy / tuples / non-finite floats -> things json can write and read back."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None                          # infeasible / failed: shown as "-"
    if isinstance(x, Path):
        return str(x)
    return x


def save(name: str, data: dict, directory: Path | None = None) -> Path:
    """Write `data` to <directory or outputs>/results/<name>.json; returns the path."""
    out = Path(directory or config.OUTPUTS_DIR) / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(_plain(data), indent=1), encoding="utf-8")
    return path


def load_all(run_dir: Path) -> dict:
    """{name: data} for every results/*.json anywhere under a run folder."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(run_dir).rglob("results/*.json"))}
