"""Differentiable cubic B-spline coefficients for fixed-structure inversion."""
from __future__ import annotations

import math

import torch
from torch import nn


class BSplineCoefficient(nn.Module):
    """A matrix of homogeneous coefficient curves on a fixed theta domain.

    Knots are clamped and uniform in normalized water content q. Controls are
    unconstrained trainable parameters; no shape prior is imposed implicitly.
    All basis operations remain in Torch so gradients flow into the field.
    """

    degree = 3

    def __init__(self, theta_bounds, n_terms: int, n_internal_knots: int = 6,
                 *, dtype=torch.float64, device=None):
        super().__init__()
        if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
            raise ValueError("n_terms must be a positive integer")
        if (isinstance(n_internal_knots, bool) or not isinstance(n_internal_knots, int)
                or n_internal_knots < 0):
            raise ValueError("n_internal_knots must be a nonnegative integer")
        if len(theta_bounds) != 2:
            raise ValueError("theta_bounds must contain two finite increasing values")
        low, high = map(float, theta_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("theta_bounds must contain two finite increasing values")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("B-spline coefficients require float32 or float64")
        self.n_terms = n_terms
        self.n_internal_knots = n_internal_knots
        self.n_basis = n_internal_knots + self.degree + 1
        options = dict(dtype=dtype, device=device)
        self.register_buffer("theta_bounds", torch.tensor([low, high], **options))
        interior = torch.linspace(0., 1., n_internal_knots + 2, **options)[1:-1]
        self.register_buffer("knots", torch.cat((torch.zeros(4, **options), interior,
                                                 torch.ones(4, **options))))
        self.controls = nn.Parameter(1e-3 * torch.randn(n_terms, self.n_basis, **options))

    def basis(self, theta: torch.Tensor) -> torch.Tensor:
        """Return basis values (n_points,n_basis), with one-sided endpoint derivatives."""
        if not isinstance(theta, torch.Tensor) or theta.ndim not in (1, 2):
            raise ValueError("theta must be a one-dimensional tensor or an (n,1) tensor")
        if theta.ndim == 2 and theta.shape[1] != 1:
            raise ValueError("theta must have one column")
        theta = theta.to(device=self.controls.device, dtype=self.controls.dtype).reshape(-1, 1)
        if not torch.isfinite(theta).all():
            raise ValueError("B-spline queries must be finite")
        low, high = self.theta_bounds.unbind()
        if ((theta < low) | (theta > high)).any():
            raise ValueError("B-spline query is outside the fixed theta domain")
        q = (theta - low) / (high - low)
        left, right = self.knots[:-1], self.knots[1:]
        intervals = (q >= left) & (q < right)
        # Include q=1 in the last nonempty degree-zero interval. Recursing from
        # that interval preserves the polynomial's left derivative at q=1;
        # replacing the final cubic basis with a one-hot tensor would cut it.
        endpoint = (q == self.knots[-1]) & (right == self.knots[-1]) & (left < right)
        values = (intervals | endpoint).to(dtype=q.dtype)
        for order in range(1, self.degree + 1):
            count = self.knots.numel() - order - 1
            denominator_left = self.knots[order:order + count] - self.knots[:count]
            denominator_right = self.knots[order + 1:order + count + 1] - self.knots[1:count + 1]
            # Repeated clamped knots have zero spans and contribute exactly zero.
            # Avoid computing a 0/0 branch, whose backward can otherwise be NaN.
            nonzero_left, nonzero_right = denominator_left != 0, denominator_right != 0
            safe_left = torch.where(nonzero_left, denominator_left, torch.ones_like(denominator_left))
            safe_right = torch.where(nonzero_right, denominator_right, torch.ones_like(denominator_right))
            weight_left = (q - self.knots[:count]) / safe_left * nonzero_left
            weight_right = (self.knots[order + 1:order + count + 1] - q) / safe_right * nonzero_right
            values = weight_left * values[:, :count] + weight_right * values[:, 1:count + 1]
        return values

    def forward(self, theta: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        return self.basis(theta) @ self.controls.T
