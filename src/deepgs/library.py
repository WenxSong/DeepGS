"""Candidate term library: naming, finite differences and automatic differentiation.

Terms are ordered as spatial derivatives, their squares, then cross products. The constant
term is not stored here; it is added as feature 0 when the regression system is built.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

SYMBOLS = ("theta_z", "theta_zz", "theta_zzz", "theta_zzzz")
CONSTANT = "1"


def term_names(max_order: int) -> list[str]:
    base = SYMBOLS[:max_order]
    return [*base, *(f"{s}^2" for s in base), *(f"{a}*{b}" for a, b in combinations(base, 2))]


def combine_terms(derivatives: list) -> list:
    """Assemble library columns from [theta_z, theta_zz, ...]; works on arrays and tensors."""
    return [*derivatives, *(d * d for d in derivatives), *(a * b for a, b in combinations(derivatives, 2))]


def required_order(names: list[str]) -> int:
    """Highest spatial derivative needed by the named terms; a constant needs none."""
    orders = [part.count("z") for name in names for part in name.replace("^2", "").split("*")]
    return max(orders, default=0)


@dataclass
class CandidateLibrary:
    """Rows are sample points. depth is present when coefficients may depend on depth."""

    theta: np.ndarray
    terms: np.ndarray
    theta_t: np.ndarray
    names: list[str]
    depth: np.ndarray | None = None
    time: np.ndarray | None = None

    def __post_init__(self):
        if self.terms.ndim != 2 or self.theta.ndim != 2 or self.theta_t.ndim != 2:
            raise ValueError("Library arrays must be two-dimensional")
        if self.theta.shape[1] != 1 or self.theta_t.shape[1] != 1:
            raise ValueError("Library theta and theta_t must have one column")
        if self.terms.shape[1] != len(self.names):
            raise ValueError("Library columns and term names differ")
        for array in (self.theta, self.terms, self.theta_t):
            if array.shape[0] != self.terms.shape[0]:
                raise ValueError("Library arrays have different row counts")
            if not np.isfinite(array).all():
                raise ValueError("Library contains non-finite values")
        for name, coordinate in (("depth", self.depth), ("time", self.time)):
            if coordinate is not None:
                if coordinate.shape != self.theta.shape or not np.isfinite(coordinate).all():
                    raise ValueError(f"Library {name} must be finite with shape {self.theta.shape}")

    @property
    def feature_names(self) -> list[str]:
        return [CONSTANT, *self.names]

    def save(self, path) -> None:
        extra = {} if self.depth is None else {"depth": self.depth}
        if self.time is not None:
            extra["time"] = self.time
        np.savez_compressed(path, theta=self.theta, terms=self.terms, theta_t=self.theta_t,
                            names=np.asarray(self.names), **extra)


def average_libraries(libraries: list[CandidateLibrary]) -> CandidateLibrary:
    """Mean of libraries evaluated on one grid, e.g. networks trained from different seeds."""
    if not libraries:
        raise ValueError("Cannot average an empty library collection")
    first = libraries[0]
    for library in libraries[1:]:
        if library.names != first.names:
            raise ValueError("Cannot average libraries with different term names or order")
        if any(getattr(library, key).shape != getattr(first, key).shape for key in ("theta", "terms", "theta_t")):
            raise ValueError("Cannot average libraries with different row counts or shapes")
        for coordinate in ("depth", "time"):
            a, b = getattr(first, coordinate), getattr(library, coordinate)
            if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
                raise ValueError(f"Cannot average libraries with different {coordinate} coordinates or row order")
    return CandidateLibrary(
        theta=np.mean([lib.theta for lib in libraries], axis=0),
        terms=np.mean([lib.terms for lib in libraries], axis=0),
        theta_t=np.mean([lib.theta_t for lib in libraries], axis=0),
        names=first.names.copy(),
        depth=None if first.depth is None else first.depth.copy(),
        time=None if first.time is None else first.time.copy(),
    )


def finite_difference_library(theta: np.ndarray, depths: np.ndarray, times: np.ndarray,
                              max_order: int = 3) -> CandidateLibrary:
    """Backward difference in time, central stencils in depth, for theta of shape (n_depths, n_times)."""
    steps = np.diff(depths)
    if not np.allclose(steps, steps[0]):
        raise ValueError("Finite differences need equally spaced depths")
    if theta.shape[0] < 5:
        raise ValueError("The stencils need at least five depths")
    h = steps[0]
    u = theta
    uz = (u[3:-1] - u[1:-3]) / (2 * h)
    uzz = (u[3:-1] - 2 * u[2:-2] + u[1:-3]) / h**2
    uzzz = (0.5 * u[4:] - u[3:-1] + u[1:-3] - 0.5 * u[:-4]) / h**3
    uzzzz = (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4]) / h**4
    derivatives = [uz, uzz, uzzz, uzzzz][:max_order]
    ut = (u[2:-2, 1:] - u[2:-2, :-1]) / np.diff(times)
    columns = [column[:, 1:].ravel() for column in combine_terms(derivatives)]
    grid_depth = np.repeat(depths[2:-2], times.size - 1)
    grid_time = np.tile(times[1:], depths.size - 4)
    return CandidateLibrary(
        theta=u[2:-2, 1:].reshape(-1, 1),
        terms=np.column_stack(columns),
        theta_t=ut.reshape(-1, 1),
        names=term_names(max_order),
        depth=grid_depth.reshape(-1, 1),
        time=grid_time.reshape(-1, 1),
    )


def spatial_derivatives(theta: torch.Tensor, z: torch.Tensor, max_order: int) -> list[torch.Tensor]:
    derivatives, current = [], theta
    for _ in range(max_order):
        current = torch.autograd.grad(current.sum(), z, create_graph=True)[0]
        derivatives.append(current)
    return derivatives


def evaluate_term(name: str, derivatives: list[torch.Tensor], like: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluate one library term; a constant needs an explicit shape when no derivatives exist."""
    if name == CONSTANT:
        template = like if like is not None else (derivatives[0] if derivatives else None)
        if template is None:
            raise ValueError("Constant term evaluation requires a like tensor")
        return torch.ones_like(template)
    by_symbol = dict(zip(SYMBOLS, derivatives))
    if name.endswith("^2"):
        return by_symbol[name[:-2]] ** 2
    if "*" in name:
        left, right = name.split("*")
        return by_symbol[left] * by_symbol[right]
    return by_symbol[name]


def library_at_points(model, z: np.ndarray, t: np.ndarray, max_order: int,
                      batch_size: int = 50000) -> CandidateLibrary:
    """Evaluate the library by automatic differentiation of a trained field at given points."""
    parameter = next(model.parameters())
    outputs = []
    for start in range(0, z.shape[0], batch_size):
        zb = torch.as_tensor(z[start:start + batch_size], dtype=parameter.dtype, device=parameter.device).requires_grad_(True)
        tb = torch.as_tensor(t[start:start + batch_size], dtype=parameter.dtype, device=parameter.device).requires_grad_(True)
        u = model(zb, tb)
        ut = torch.autograd.grad(u.sum(), tb, create_graph=True)[0]
        columns = combine_terms(spatial_derivatives(u, zb, max_order))
        outputs.append([tensor.detach().cpu().numpy().astype(np.float64) for tensor in (u, ut, *columns)])
    stacked = [np.concatenate(group, axis=0) for group in zip(*outputs)]
    return CandidateLibrary(
        theta=stacked[0],
        theta_t=stacked[1],
        terms=np.concatenate(stacked[2:], axis=1),
        names=term_names(max_order),
        depth=np.asarray(z, dtype=np.float64).reshape(-1, 1),
        time=np.asarray(t, dtype=np.float64).reshape(-1, 1),
    )


def library_on_grid(model, depths: np.ndarray, times: np.ndarray, max_order: int) -> CandidateLibrary:
    time_grid, depth_grid = np.meshgrid(times, depths)
    return library_at_points(model, depth_grid.reshape(-1, 1), time_grid.reshape(-1, 1), max_order)
