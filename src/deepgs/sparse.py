"""Structure selection: sequential grouped threshold ridge and spline backward elimination."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from kneed import KneeLocator
from scipy.interpolate import BSpline

from .grouping import GroupedSystem


@dataclass
class StructureResult:
    active: list[int]
    names: list[str]
    theta_centers: np.ndarray
    depth_centers: np.ndarray | None
    coefficients: np.ndarray  # physical values, shape (n_features, n_groups)
    path: list[dict] = field(default_factory=list)
    criterion: str = ""
    spline: dict | None = None

    def summary(self) -> dict:
        return dict(active=self.active, terms=self.names, criterion=self.criterion, path=self.path)


# --- criteria ---------------------------------------------------------------------------------

def aic_score(rss: float, n_samples: int, k: int, epsilon: float, corrected: bool = False) -> float:
    """AIC-inspired loss N ln(RSS/N + epsilon) + 2k.

    With normalized data RSS/N is much smaller than epsilon, so this behaves like RSS/epsilon + 2k:
    a term must lower the normalized RSS by about 2 epsilon to be kept.
    """
    score = n_samples * np.log(rss / n_samples + epsilon) + 2 * k
    if corrected:
        score += 2 * (k + 1) * (k + 2) / (n_samples - k - 2)
    return float(score)


def knee_complexity(complexities: list[int], errors: list[float]) -> int | None:
    """Kneedle point of the convex decreasing log10 error against model complexity."""
    log_errors = np.log10(np.maximum(errors, np.finfo(float).tiny))
    return KneeLocator(complexities, log_errors, curve="convex", direction="decreasing").knee


def _names(system: GroupedSystem, active) -> list[str]:
    return [system.feature_names[i] for i in active]


# --- sequential grouped threshold ridge -------------------------------------------------------

def ridge(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)


def _fit_groups(system: GroupedSystem, keep: np.ndarray, lam: float) -> np.ndarray:
    W = np.zeros((system.n_features, system.n_groups))
    for g, (X, y) in enumerate(zip(system.X, system.y)):
        W[keep, g] = ridge(X[:, keep], y, lam)[:, 0]
    return W


def group_rss(system: GroupedSystem, W: np.ndarray) -> float:
    return float(sum(np.sum((y[:, 0] - X @ W[:, g]) ** 2) for g, (X, y) in enumerate(zip(system.X, system.y))))


def threshold_support(W: np.ndarray, system: GroupedSystem, tol: float, lam: float, maxit: int) -> np.ndarray:
    """Remove whole terms whose group norm is below tol, refitting with ridge between passes."""
    keep = np.ones(system.n_features, dtype=bool)
    current = W
    for iteration in range(maxit):
        survivors = keep & (np.linalg.norm(current, axis=1) >= tol)
        if not survivors.any():
            return survivors
        changed = not np.array_equal(survivors, keep)
        keep = survivors
        if not changed or iteration == maxit - 1:
            break
        current = _fit_groups(system, keep, lam)
    return keep


def sgtr_path(system: GroupedSystem, num_tols: int, lam: float, maxit: int, epsilon: float,
              corrected: bool = False) -> list[dict]:
    """Scan log-spaced thresholds. Every distinct support is refitted without ridge bias once."""
    W_ridge = _fit_groups(system, np.ones(system.n_features, dtype=bool), lam)
    norms = np.linalg.norm(W_ridge, axis=1)
    lowest, highest = max(norms.min(), np.finfo(float).tiny), norms.max()
    tolerances = [0.0, *np.exp(np.linspace(np.log(lowest), np.log(highest), num_tols))[:-1]]

    seen: dict[tuple, dict] = {}
    for tol in tolerances:
        keep = threshold_support(W_ridge, system, tol, lam, maxit)
        key = tuple(np.flatnonzero(keep).tolist())
        if key in seen:
            continue
        W = _fit_groups(system, keep, 0.0) if keep.any() else np.zeros_like(W_ridge)
        rss = group_rss(system, W)
        seen[key] = dict(tol=float(tol), active=list(key), k=len(key), rss=rss, W=W,
                         aic=aic_score(rss, system.n_samples, len(key), epsilon, corrected))
    return list(seen.values())


def select_sgtr(system: GroupedSystem, num_tols: int = 200, lam: float = 1e-4, maxit: int = 1,
                epsilon: float = 1e-2, corrected: bool = False, min_energy_gain: float | None = None) -> StructureResult:
    """Select the structure with the lowest AIC-inspired loss on the threshold path.

    A term must lower the normalized RSS by about 2 epsilon. Under group normalization the total
    energy equals the number of groups, so a fixed epsilon becomes stricter as groups get fewer.
    min_energy_gain rescales the approximate penalty to a fraction of normalized response energy
    and overrides epsilon. Grouping and normalization still affect weights and the selection path.
    """
    if min_energy_gain is not None:
        epsilon = min_energy_gain * sum(float(np.sum(y**2)) for y in system.y) / 2
    path = sgtr_path(system, num_tols, lam, maxit, epsilon, corrected)
    best = min(path, key=lambda entry: entry["aic"])
    return StructureResult(
        active=best["active"],
        names=_names(system, best["active"]),
        theta_centers=system.theta_centers,
        depth_centers=system.depth_centers,
        coefficients=system.to_physical(best["W"]),
        path=[dict(k=e["k"], rss=e["rss"], aic=e["aic"], tol=e["tol"], terms=_names(system, e["active"])) for e in path],
        criterion="minimum AIC-inspired loss",
    )


# --- spline-regularized backward elimination --------------------------------------------------

def spline_knots(theta: np.ndarray, n_knots: int, degree: int) -> np.ndarray:
    low, high = float(np.min(theta)), float(np.max(theta))
    if low == high:
        low, high = low - 1e-3, high + 1e-3
    inner = np.linspace(low, high, n_knots + 2)[1:-1]
    return np.r_[[low] * (degree + 1), inner, [high] * (degree + 1)]


def spline_basis(points: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    n_basis = knots.size - degree - 1
    return np.column_stack([BSpline(knots, np.eye(n_basis)[i], degree)(points) for i in range(n_basis)])


def expand_design(system: GroupedSystem, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every feature column is multiplied by the basis evaluated at its group centre."""
    blocks = [(X[:, :, None] * basis[g][None, None, :]).reshape(X.shape[0], -1) for g, X in enumerate(system.X)]
    return np.vstack(blocks), np.vstack(system.y)


def solve_active(gram: np.ndarray, moment: np.ndarray, active: list[int], n_features: int, n_basis: int,
                 lam: float) -> np.ndarray:
    """Ridge solution restricted to the spline blocks of the active features."""
    C = np.zeros((n_features, n_basis))
    if not active:
        return C
    columns = np.concatenate([np.arange(j * n_basis, (j + 1) * n_basis) for j in active])
    solution = np.linalg.solve(gram[np.ix_(columns, columns)] + lam * np.eye(columns.size), moment[columns])
    C[active] = solution.reshape(len(active), n_basis)
    return C


def mean_group_mse(system: GroupedSystem, W: np.ndarray) -> float:
    return float(np.mean([np.mean((y[:, 0] - X @ W[:, g]) ** 2) for g, (X, y) in enumerate(zip(system.X, system.y))]))


def elimination_path(system: GroupedSystem, n_knots: int = 6, degree: int = 3, lam: float = 0.01) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Drop the term with the smallest spline-coefficient block norm until nothing remains."""
    knots = spline_knots(system.theta_centers, n_knots, degree)
    basis = spline_basis(system.theta_centers, knots, degree)
    n_basis = basis.shape[1]
    design, response = expand_design(system, basis)
    gram, moment = design.T @ design, design.T @ response

    active = list(range(system.n_features))
    path = []
    while True:
        C = solve_active(gram, moment, active, system.n_features, n_basis, lam)
        W = (basis @ C.T).T
        path.append(dict(active=active.copy(), k=len(active), error=mean_group_mse(system, W), C=C, W=W))
        if not active:
            break
        block_norms = np.linalg.norm(C[active], axis=1)
        active.pop(int(np.argmin(block_norms)))
    return path, knots, basis


def select_spline_with_diagnostics(system: GroupedSystem, n_knots: int = 6, degree: int = 3,
                                   lam: float = 0.01, refit_lam: float | None = None
                                   ) -> tuple[StructureResult | None, list[dict]]:
    """Select a spline model and preserve its path even when there is no knee.

    Ridge at this strength shrinks the curves strongly. refit_lam refits the selected
    support with a smaller weight when the intermediate curves themselves are of interest.
    """
    if system.depth_centers is not None:
        raise ValueError("Spline selection expects water-content groups without depth")
    if system.normalization != "global":
        raise ValueError("Spline selection requires global normalization for shared physical coefficient curves")
    path, knots, basis = elimination_path(system, n_knots, degree, lam)
    ordered = sorted(path, key=lambda entry: entry["k"])
    knee = knee_complexity([e["k"] for e in ordered], [e["error"] for e in ordered])
    summary = [dict(k=e["k"], error=e["error"], terms=_names(system, e["active"])) for e in ordered]
    if knee is None:
        return None, summary
    chosen = next(e for e in ordered if e["k"] == knee)
    C = chosen["C"]
    if refit_lam is not None:
        design, response = expand_design(system, basis)
        C = solve_active(design.T @ design, design.T @ response, chosen["active"], system.n_features,
                         basis.shape[1], refit_lam)
    # Spline selection uses global normalization, so one scale holds for all groups.
    C_physical = C / system.x_scale[0][:, None] * system.y_scale[0]
    result = StructureResult(
        active=chosen["active"],
        names=_names(system, chosen["active"]),
        theta_centers=system.theta_centers,
        depth_centers=None,
        coefficients=(basis @ C_physical.T).T,
        path=summary,
        criterion="Kneedle on log10 error along backward elimination",
        spline=dict(knots=knots, degree=degree, control_points=C_physical),
    )
    return result, summary


def select_spline(system: GroupedSystem, n_knots: int = 6, degree: int = 3, lam: float = 0.01,
                  refit_lam: float | None = None) -> StructureResult | None:
    """Compatibility entry point returning None when the curve has no knee."""
    result, _ = select_spline_with_diagnostics(system, n_knots, degree, lam, refit_lam)
    return result


def spline_curve(result: StructureResult, feature: int, theta: np.ndarray) -> np.ndarray:
    """Evaluate a selected coefficient function from its spline control points."""
    spline = result.spline
    return BSpline(spline["knots"], spline["control_points"][feature], spline["degree"])(theta)
