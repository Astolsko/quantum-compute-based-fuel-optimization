"""
manifest.py  --  record WHAT a run ran on: package versions, machine, settings.

run_all.sh calls this at the start of every run; it writes manifest.json into
the run folder. Two runs that disagree can then be told apart by their
manifests instead of by memory.

    python -m tools.manifest runs/2026-10-28_0400
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm",
            "pyqubo", "dwave-neal", "dwave-samplers", "qiskit", "qiskit-aer",
            "qiskit-aer-gpu", "qiskit-algorithms", "qiskit-optimization",
            "matplotlib", "streamlit"]


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def main() -> None:
    run_dir = Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "os": platform.platform(),
        "python": sys.version.split()[0], "cpu": _cpu(), "cores": os.cpu_count(),
        "gpu": _gpu(),
        "env": {k: os.environ.get(k) for k in ("FUEL_JOBS", "FUEL_QDEVICE",
                                                "FUEL_RUN_DIR", "CONDA_DEFAULT_ENV")},
        "packages": {p: v for p in PACKAGES if (v := _version(p))},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest -> {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
