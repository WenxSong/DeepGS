"""Stage contracts for the version-2, fixed-structure workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json

import numpy as np

from .library import CandidateLibrary
from .sparse import StructureResult

CANONICAL_UNITS = {"time": "day", "depth": "m", "water_content": "fraction"}


@dataclass(frozen=True)
class LibraryArtifact:
    library: CandidateLibrary
    sample_z: np.ndarray
    sample_t: np.ndarray
    units: dict = field(default_factory=lambda: dict(CANONICAL_UNITS))
    fingerprint: str = field(init=False)

    def __post_init__(self):
        library = deepcopy(self.library)
        z = np.asarray(self.sample_z, dtype=np.float64).reshape(-1).copy()
        t = np.asarray(self.sample_t, dtype=np.float64).reshape(-1).copy()
        n = library.terms.shape[0]
        if len(set(library.feature_names)) != len(library.feature_names):
            raise ValueError("Library term IDs must be unique")
        if z.size != n or t.size != n or not np.isfinite(z).all() or not np.isfinite(t).all():
            raise ValueError("Library row coordinates must be finite and match the sample count")
        for name, actual in (("depth", z), ("time", t)):
            source = getattr(library, name, None)
            if source is not None and not np.array_equal(np.asarray(source).ravel(), actual):
                raise ValueError(f"Library {name} coordinates do not match row coordinates")
        if self.units != CANONICAL_UNITS:
            raise ValueError("LibraryArtifact requires canonical units (day, m, fraction)")
        digest = hashlib.sha256(json.dumps(dict(names=list(library.feature_names), units=self.units), sort_keys=True).encode())
        for array in (library.theta, library.terms, library.theta_t, z, t):
            value = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        for array in (library.theta, library.terms, library.theta_t, library.depth, getattr(library, "time", None), z, t):
            if array is not None:
                array.setflags(write=False)
        object.__setattr__(self, "library", library)
        object.__setattr__(self, "sample_z", z)
        object.__setattr__(self, "sample_t", t)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @classmethod
    def create(cls, library):
        if library.depth is None or getattr(library, "time", None) is None:
            raise ValueError("Explicit depth and time coordinates are required")
        return cls(library, library.depth, library.time)

    def assert_unchanged(self):
        current = LibraryArtifact(self.library, self.sample_z, self.sample_t, self.units)
        if current.fingerprint != self.fingerprint:
            raise ValueError("Library artifact changed after fingerprinting")


@dataclass(frozen=True)
class SelectionResult:
    status: str
    terms: tuple[str, ...]
    term_indices: tuple[int, ...]
    coarse: StructureResult | None
    diagnostics: tuple[dict, ...]
    library_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "term_indices", tuple(self.term_indices))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.status not in ("selected", "empty", "no_knee"):
            raise ValueError("Unknown selection status")
        if len(self.terms) != len(self.term_indices) or len(set(self.terms)) != len(self.terms):
            raise ValueError("Selected terms must be unique and aligned with indices")
        if (self.status == "selected") != bool(self.terms):
            raise ValueError("Selection status and selected terms disagree")
        if self.coarse is not None and (tuple(self.coarse.names) != self.terms or tuple(self.coarse.active) != self.term_indices):
            raise ValueError("Coarse estimates and frozen selected terms disagree")

    def summary(self):
        return dict(status=self.status, terms=list(self.terms), term_indices=list(self.term_indices),
                    library_fingerprint=self.library_fingerprint, diagnostics=list(self.diagnostics),
                    criterion=self.coarse.criterion if self.coarse is not None else None)

    @classmethod
    def supplied(cls, terms):
        """Explicit fixed support for inversion, recorded separately from discovery."""
        terms = tuple(terms)
        signature = hashlib.sha256(json.dumps(terms).encode()).hexdigest()
        return cls("selected" if terms else "empty", terms, tuple(range(len(terms))), None, (), "supplied:" + signature)

    def assert_unchanged(self):
        if self.coarse is not None and (tuple(self.coarse.names) != self.terms or tuple(self.coarse.active) != self.term_indices):
            raise ValueError("Selected structure changed after freezing")


@dataclass(frozen=True)
class CoefficientSpec:
    form: str
    inputs: tuple[str, ...]
    terms: tuple[str, ...]


@dataclass(frozen=True)
class InversionResult:
    status: str = "not_requested"
    terms: tuple[str, ...] = ()
    field: object | None = None
    coefficients: object | None = None
    history: tuple = ()
    domain: dict | None = None
    metrics: dict | None = None
    metadata: dict | None = None

    def summary(self):
        return dict(status=self.status, terms=list(self.terms), domain=self.domain, metrics=self.metrics,
                    metadata=self.metadata)


@dataclass(frozen=True)
class RunResult:
    execution_status: str
    selection: SelectionResult
    inversion: InversionResult
    artifacts: dict
