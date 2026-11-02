"""Record the inputs and effective settings of a run before training begins."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import torch


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def run_manifest(name, config, device):
    inputs = [file_record(config.data.path)]
    reference = getattr(config, "reference", None)
    if reference is not None and reference.path:
        inputs += [file_record(path) for path in sorted(Path(reference.path).glob("*.npy"))]
    return dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(), pipeline=name,
                configuration=asdict(config), device=str(device), inputs=inputs,
                source=[file_record(path) for path in sorted(Path(__file__).parent.glob("*.py"))],
                environment=dict(python=platform.python_version(), platform=platform.platform(),
                                 packages={p: importlib.metadata.version(p) for p in
                                           ("torch", "numpy", "scipy", "kneed", "scikit-learn")},
                                 cuda=torch.version.cuda,
                                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
