"""Data fitting and physics-informed training shared by the pipelines."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

from .data import Observations
from .library import CONSTANT, evaluate_term, required_order, spatial_derivatives


def to_tensors(points: np.ndarray, like: torch.Tensor, requires_grad: bool = False):
    """Split (z, t) points into two column tensors."""
    tensor = torch.as_tensor(points, dtype=like.dtype, device=like.device)
    z, t = tensor[:, :1].clone(), tensor[:, 1:].clone()
    return z.requires_grad_(requires_grad), t.requires_grad_(requires_grad)


def halton_points(n: int, observations: Observations, seed: int = 0) -> np.ndarray:
    """Quasi-random residual points over the observed depth-time domain; the origin is skipped."""
    unit = qmc.Halton(d=2, scramble=False, seed=seed).random(n + 1)[1:]
    low = [observations.depths.min(), observations.times.min()]
    high = [observations.depths.max(), observations.times.max()]
    return qmc.scale(unit, low, high)


def fit_field(model: nn.Module, observations: Observations, iterations: int, learning_rate: float = 1e-3,
              log_every: int = 0, label: str = "field", lbfgs_iterations: int = 0) -> list[dict]:
    """Train a field on observations alone: Adam, then optionally L-BFGS to tighten the fit."""
    parameter = next(model.parameters())
    z, t = to_tensors(observations.points(), parameter)
    target = torch.as_tensor(observations.values(), dtype=parameter.dtype, device=parameter.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(1, iterations + 1):
        optimizer.zero_grad()
        loss = (model(z, t) - target).square().mean()
        loss.backward()
        optimizer.step()
        if log_every and (step % log_every == 0 or step == iterations):
            history.append(dict(step=step, data_loss=float(loss.detach())))
            print(f"[{label}] step {step}: data loss {float(loss.detach()):.3e}", flush=True)
    if lbfgs_iterations:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iterations, max_eval=5 * lbfgs_iterations,
                                  history_size=50, tolerance_grad=0.0, tolerance_change=0.0, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            value = (model(z, t) - target).square().mean()
            value.backward()
            return value

        lbfgs.step(closure)
        final = float((model(z, t) - target).square().mean().detach())
        history.append(dict(step=iterations + lbfgs_iterations, data_loss=final))
        print(f"[{label}] after L-BFGS: data loss {final:.3e}", flush=True)
    return history


def pde_residual(field: nn.Module, coefficients: Sequence[nn.Module], names: Sequence[str], z: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    """theta_t minus the sum of the active terms, each weighted by its coefficient network."""
    theta = field(z, t)
    theta_t = torch.autograd.grad(theta.sum(), t, create_graph=True)[0]
    derivatives = spatial_derivatives(theta, z, required_order(list(names)))
    residual = theta_t
    for name, coefficient in zip(names, coefficients):
        term = torch.ones_like(theta) if name == CONSTANT else evaluate_term(name, derivatives)
        residual = residual - term * coefficient(theta, z)
    return residual


def train_pinn(field: nn.Module, coefficients: Sequence[nn.Module], names: Sequence[str],
               observations: Observations, collocation: np.ndarray, steps: int, data_weight: float,
               pde_weight: float, optimizer: torch.optim.Optimizer,
               scheduler: Callable[[], None] | None = None, log_every: int = 0, label: str = "pinn") -> list[dict]:
    """Minimize data_weight * L_data + pde_weight * L_pde with a fixed set of residual points."""
    parameter = next(field.parameters())
    z, t = to_tensors(observations.points(), parameter)
    target = torch.as_tensor(observations.values(), dtype=parameter.dtype, device=parameter.device)
    z_r, t_r = to_tensors(collocation, parameter, requires_grad=True)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        data_loss = (field(z, t) - target).square().mean()
        pde_loss = pde_residual(field, coefficients, names, z_r, t_r).square().mean()
        loss = data_weight * data_loss + pde_weight * pde_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler()
        if log_every and (step % log_every == 0 or step == steps):
            row = dict(step=step, loss=float(loss.detach()), data_loss=float(data_loss.detach()),
                       pde_loss=float(pde_loss.detach()))
            history.append(row)
            print(f"[{label}] step {step}: L_data {row['data_loss']:.3e}, L_pde {row['pde_loss']:.3e}", flush=True)
    return history
