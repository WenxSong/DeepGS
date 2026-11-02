"""Structure-only adapters for the two-stage workflow.

The numerical selection kernels remain in sparse.py. The result freezes term
order and carries a candidate-library fingerprint into the inversion stage.
"""
from __future__ import annotations

from dataclasses import asdict

from .contracts import LibraryArtifact, SelectionResult
from .grouping import build_system
from .library import term_names
from .sparse import select_sgtr, select_spline_with_diagnostics


def select(library: LibraryArtifact, config, soil: str = "homogeneous") -> SelectionResult:
    """Select once from an aligned library, without performing any inversion."""
    library.assert_unchanged()
    supported = {"group": "homogeneous", "spline": "homogeneous", "spatial_group": "heterogeneous"}
    if config.selector not in supported:
        raise ValueError(f"Unknown structure selector: {config.selector}")
    if soil != supported[config.selector]:
        raise ValueError(f"Selector {config.selector!r} requires soil={supported[config.selector]!r}")
    names = library.library.names
    if len(set(names)) != len(names) or not set(names).issubset(term_names(4)):
        raise ValueError("Library terms must be unique supported derivative IDs; the constant is added by grouping")
    system = build_system(library.library, by_depth=config.selector == "spatial_group",
                          **asdict(config.grouping))
    if config.selector == "spline":
        settings = asdict(config.spline)
        settings["n_knots"] = settings.pop("n_internal_knots")
        coarse, path = select_spline_with_diagnostics(system, **settings)
    else:
        coarse = select_sgtr(system, **asdict(config.sgtr))
        path = coarse.path
    if coarse is None:
        return SelectionResult(status="no_knee", terms=(), term_indices=(), coarse=None,
                               diagnostics=tuple(path), library_fingerprint=library.fingerprint)
    return SelectionResult(status="selected" if coarse.names else "empty", terms=tuple(coarse.names),
                           term_indices=tuple(coarse.active), coarse=coarse, diagnostics=tuple(path),
                           library_fingerprint=library.fingerprint)
