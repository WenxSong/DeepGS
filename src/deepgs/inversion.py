"""Coefficient inversion after the structure has been fixed, with two coefficient forms.

'network' uses the DeepGS branch networks xi_j(theta). They work with any selected structure and
impose no sign, which suits coefficients that span orders of magnitude.

'polynomial' uses f_j(theta) = s_j * sum_k c_jk q^k, with c_jk >= 0 and q in [0, 1]. Nonnegative
powers of a nonnegative variable give the sign s_j and monotonicity by construction, and also
restrict curvature, so the form is limited over a wide water-content range. At every step the
controls are solved by bounded least squares and held constant while Adam updates the field, so the
gradient of the outer loss follows the envelope theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import lsq_linear

from .data import Observations
from .library import evaluate_term, required_order, spatial_derivatives
from .networks import FieldNetwork, StateCoefficient, observation_bounds
from .pinn import train_pinn

N_CONTROLS = 4


@dataclass(frozen=True)
class InversionSettings:
    coefficient_form: str = "network"  # 'network' or 'polynomial'
    coefficient_hidden: tuple[int, ...] = (20, 20)
    hidden_sizes: tuple[int, ...] = (50, 50, 50, 50, 50)
    epochs: int = 20000
    data_weight: float = 100.0
    pde_weight: float = 1.0
    collocation_points: int = 2000
    dynamic_sampling_fraction: float = 0.9
    learning_rate: float = 0.001
    log_every: int = 1000


def theta_bounds(observations: Observations, envelope: tuple[float, float] | None = None) -> tuple[float, float]:
    """Field output range: observed values with a margin, kept inside a known envelope if there is one."""
    low, high = float(observations.theta.min()), float(observations.theta.max())
    if envelope is None:
        return low - 0.01, high + 0.01
    margin = 0.005 * (envelope[1] - envelope[0])
    lower, upper = max(envelope[0] + margin, low - 0.01), min(envelope[1] - margin, high + 0.01)
    return (lower, upper) if lower < upper else (low, high)


def sample_collocation(observations: Observations, settings: InversionSettings, seed: int) -> np.ndarray:
    """Latin-hypercube depths; times biased toward periods when the observations change quickly."""
    n = settings.collocation_points
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, n + 1)
    unit = rng.random(n) * (edges[1:] - edges[:-1]) + edges[:-1]
    rng.shuffle(unit)
    depth = observations.depths.min() + unit * (observations.depths.max() - observations.depths.min())

    times = observations.times
    changes = np.max(np.abs(np.diff(observations.theta, axis=0)), axis=1)
    scores = np.zeros(times.size)
    scores[0], scores[-1] = changes[0], changes[-1]
    scores[1:-1] = np.maximum(changes[:-1], changes[1:])
    intervals = 0.5 * (scores[:-1] + scores[1:])

    rng = np.random.default_rng(seed + 193)
    count = int(round(n * settings.dynamic_sampling_fraction))
    parts = []
    if count:
        if intervals.sum() > 0:
            index = rng.choice(intervals.size, size=count, replace=True, p=intervals / intervals.sum())
            parts.append(times[index] + rng.random(count) * (times[index + 1] - times[index]))
        else:
            parts.append(rng.uniform(times[0], times[-1], count))
    if count < n:
        parts.append(rng.uniform(times[0], times[-1], n - count))
    sampled = np.concatenate(parts)
    rng.shuffle(sampled)
    return np.column_stack((depth, sampled))


def pde_matrix(field: FieldNetwork, names: Sequence[str], signs: Sequence[int], bounds: tuple[float, float],
               z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows are residual points, columns the signed basis q^k times each active term."""
    theta = field(z, t)
    theta_t = torch.autograd.grad(theta.sum(), t, create_graph=True)[0]
    derivatives = spatial_derivatives(theta, z, required_order(list(names)))
    q = (theta - bounds[0]) / (bounds[1] - bounds[0])
    basis = torch.cat([q**k for k in range(N_CONTROLS)], dim=1)
    columns = [sign * basis * evaluate_term(name, derivatives, like=theta) for name, sign in zip(names, signs)]
    return torch.cat(columns, dim=1), theta_t.ravel()


def project_controls(matrix: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Bounded least squares for the nonnegative controls, through QR on scaled columns."""
    x, y = matrix.detach().cpu().numpy(), response.detach().cpu().numpy()
    norms = np.linalg.norm(x, axis=0)
    usable = norms > np.finfo(float).tiny
    controls = np.zeros(x.shape[1])
    if usable.any():
        q, r = np.linalg.qr(x[:, usable] / norms[usable], mode="reduced")
        fit = lsq_linear(r, q.T @ y, bounds=(0, np.inf), method="bvls", tol=1e-11, max_iter=500)
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f"Coefficient projection failed: {fit.message}")
        # The solver can return rounding-level negatives at active bounds.
        controls[usable] = np.maximum(fit.x, 0.0) / norms[usable]
    return torch.as_tensor(controls, dtype=matrix.dtype, device=matrix.device)


def cubic_curves(controls: np.ndarray, signs: Sequence[int], bounds: tuple[float, float], theta: np.ndarray) -> np.ndarray:
    """Coefficient values with shape (n_terms, n_theta)."""
    q = (np.asarray(theta) - bounds[0]) / (bounds[1] - bounds[0])
    basis = np.stack([q**k for k in range(N_CONTROLS)], axis=1)
    return (controls @ basis.T) * np.asarray(signs)[:, None]


def run_network_inversion(observations: Observations, names: Sequence[str], device: torch.device,
                          settings: InversionSettings | None = None, seed: int = 2026,
                          envelope: tuple[float, float] | None = None) -> dict:
    """Jointly train the field and one branch network per selected term."""
    settings = settings or InversionSettings()
    torch.manual_seed(seed)
    bounds = theta_bounds(observations, envelope)
    field = FieldNetwork(settings.hidden_sizes, "silu", observation_bounds(observations), bounds, xavier=True).to(
        device=device, dtype=torch.float64)
    nets = torch.nn.ModuleList(StateCoefficient(settings.coefficient_hidden, *bounds) for _ in names).to(
        device=device, dtype=torch.float64)
    collocation = sample_collocation(observations, settings, seed + 97)
    optimizer = torch.optim.Adam(list(field.parameters()) + list(nets.parameters()), lr=settings.learning_rate)
    history = train_pinn(field, nets, names, observations, collocation, settings.epochs, settings.data_weight,
                         settings.pde_weight, optimizer, log_every=settings.log_every, label="inversion")

    grid = np.linspace(bounds[0], bounds[1], 200)
    points = torch.as_tensor(observations.points(), device=device, dtype=torch.float64)
    with torch.no_grad():
        curves = np.stack([net(torch.as_tensor(grid[:, None], device=device, dtype=torch.float64)).cpu().numpy().ravel() for net in nets])
        fitted = field(points[:, :1], points[:, 1:]).cpu().numpy().reshape(observations.theta.shape)
    return dict(field=field, coefficients=nets, controls=None, bounds=bounds, theta_grid=grid, curves=curves, fitted=fitted,
                history=history, theta_rmse=float(np.sqrt(np.mean((fitted - observations.theta) ** 2))))


def run_inversion(observations: Observations, names: Sequence[str], signs: Sequence[int] | None, device: torch.device,
                  settings: InversionSettings | None = None, seed: int = 2026,
                  envelope: tuple[float, float] | None = None) -> dict:
    """Jointly reconstruct the field and the coefficient functions of a fixed structure."""
    settings = settings or InversionSettings()
    if settings.coefficient_form == "network":
        return run_network_inversion(observations, names, device, settings, seed, envelope)
    if settings.coefficient_form != "polynomial":
        raise ValueError(f"Unknown coefficient form: {settings.coefficient_form}")
    if signs is None:
        raise ValueError("The polynomial form needs a sign for each term")
    torch.manual_seed(seed)
    bounds = theta_bounds(observations, envelope)
    field = FieldNetwork(settings.hidden_sizes, "silu", observation_bounds(observations), bounds, xavier=True).to(
        device=device, dtype=torch.float64)
    collocation = torch.as_tensor(sample_collocation(observations, settings, seed + 97), device=device, dtype=torch.float64)
    z_r, t_r = (collocation[:, i:i + 1].clone().requires_grad_(True) for i in range(2))
    points = torch.as_tensor(observations.points(), device=device, dtype=torch.float64)
    target = torch.as_tensor(observations.values(), device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(field.parameters(), lr=settings.learning_rate)

    history = []
    for epoch in range(1, settings.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        matrix, response = pde_matrix(field, names, signs, bounds, z_r, t_r)
        controls = project_controls(matrix, response)
        data_loss = (field(points[:, :1], points[:, 1:]) - target).square().mean()
        pde_loss = (matrix @ controls - response).square().mean()
        loss = settings.data_weight * data_loss + settings.pde_weight * pde_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % settings.log_every == 0 or epoch == settings.epochs:
            row = dict(epoch=epoch, data_loss=float(data_loss.detach()), pde_loss=float(pde_loss.detach()))
            history.append(row)
            print(f"[inversion] epoch {epoch}: L_data {row['data_loss']:.3e}, L_pde {row['pde_loss']:.3e}", flush=True)

    matrix, response = pde_matrix(field, names, signs, bounds, z_r, t_r)
    controls = project_controls(matrix, response).cpu().numpy().reshape(len(names), N_CONTROLS)
    grid = np.linspace(bounds[0], bounds[1], 200)
    curves = cubic_curves(controls, signs, bounds, grid)
    with torch.no_grad():
        fitted = field(points[:, :1], points[:, 1:]).cpu().numpy().reshape(observations.theta.shape)
    signed = curves * np.asarray(signs)[:, None]
    if signed.min() < -1e-10 or np.diff(signed, axis=1).min() < -1e-10:
        raise RuntimeError("Coefficient sign or monotonicity audit failed")
    return dict(field=field, controls=controls, bounds=bounds, theta_grid=grid, curves=curves, fitted=fitted, history=history,
                theta_rmse=float(np.sqrt(np.mean((fitted - observations.theta) ** 2))))
