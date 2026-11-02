"""Provenance and metrics shared by the executable benchmark scripts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepgs.library import CandidateLibrary

TRUE_TERMS = ["theta_z", "theta_zz", "theta_z^2"]


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=digest.hexdigest())


def environment():
    return dict(python=sys.version, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in
                          ("numpy", "scipy", "torch", "scikit-learn", "kneed")})


def relative_l2(estimate, reference):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if estimate.shape != reference.shape or not np.isfinite(estimate).all() or not np.isfinite(reference).all():
        raise ValueError("Metric arrays must have matching shapes and finite entries")
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(estimate - reference) / norm) if norm > 0 else None


def support_score(names, expected=TRUE_TERMS):
    if names is None:
        return dict(exact=False, missing=sorted(expected), extra=[], status="no_knee")
    return dict(exact=set(names) == set(expected), missing=sorted(set(expected) - set(names)),
                extra=sorted(set(names) - set(expected)), status="selected")


def subset(library, rows, names=None):
    names = library.names if names is None else names
    return CandidateLibrary(library.theta[rows], library.terms[rows][:, [library.names.index(n) for n in names]],
                            library.theta_t[rows], names, None if library.depth is None else library.depth[rows])


def normalized_condition(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(norms > 0, norms, 1)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(scaled.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    return dict(rank=rank, columns=matrix.shape[1],
                condition=float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None,
                singular_values=singular.tolist())
