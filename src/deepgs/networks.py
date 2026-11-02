"""Water-content field networks and coefficient representations."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import Observations

ACTIVATIONS = {"tanh": torch.tanh, "sin": torch.sin, "silu": F.silu}


def _xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


class FieldNetwork(nn.Module):
    """theta(z, t) as a fully connected network.

    bounds scales inputs to [-1, 1] from ((z_min, t_min), (z_max, t_max)); output_bounds maps
    the output into a fixed water-content range through a sigmoid.
    """

    def __init__(self, hidden_sizes: Sequence[int], activation: str = "tanh", bounds=None, output_bounds=None,
                 xavier: bool = False, bounded_lerp: bool = False):
        super().__init__()
        self.activation = ACTIVATIONS[activation]
        widths = [2, *hidden_sizes]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.output = nn.Linear(widths[-1], 1)
        self.normalize_inputs = bounds is not None
        if bounds is not None:
            lower, upper = (torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(1, 2) for b in bounds)
            self.register_buffer("lower", lower)
            self.register_buffer("span", (upper - lower).clamp(min=1e-6))
        self.output_bounds = output_bounds
        self.bounded_lerp = bounded_lerp
        if xavier:
            self.apply(_xavier)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, t], dim=1)
        if self.normalize_inputs:
            x = 2 * (x - self.lower) / self.span - 1
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.output(x)
        if self.output_bounds is not None:
            if self.bounded_lerp:
                low, high = x.new_tensor(self.output_bounds).unbind()
                return torch.lerp(low, high, torch.sigmoid(x))
            low, high = self.output_bounds
            return low + (high - low) * torch.sigmoid(x)
        return x


def observation_bounds(observations: Observations):
    return ((observations.depths.min(), observations.times.min()), (observations.depths.max(), observations.times.max()))


def _stack(widths: Sequence[int]) -> nn.ModuleList:
    return nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))


class StateCoefficient(nn.Module):
    """xi(theta) for homogeneous soil, with theta scaled to [-1, 1]."""

    def __init__(self, hidden_sizes: Sequence[int], theta_min: float, theta_max: float):
        super().__init__()
        self.layers = _stack([1, *hidden_sizes])
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.theta_min, self.theta_span = theta_min, max(theta_max - theta_min, 1e-6)

    def forward(self, theta: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        x = 2 * (theta - self.theta_min) / self.theta_span - 1
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.output(x)


class HeterogeneousCoefficient(nn.Module):
    """Two branches with optional nonlinear fusion.

    Empty fusion_hidden preserves the original additive a(theta) + b(z) model.
    Nonempty fusion_hidden enables interactions; it is an experimental extension.
    """

    def __init__(self, theta_hidden: Sequence[int], depth_hidden: Sequence[int], fusion_hidden: Sequence[int] = ()):
        super().__init__()
        self.theta_layers = _stack([1, *theta_hidden])
        self.depth_layers = _stack([1, *depth_hidden])
        merged_width = theta_hidden[-1] + depth_hidden[-1]
        self.fusion_layers = _stack([merged_width, *fusion_hidden])
        self.output = nn.Linear(fusion_hidden[-1] if fusion_hidden else merged_width, 1)

    def forward(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        a, b = theta, z
        for layer in self.theta_layers:
            a = torch.tanh(layer(a))
        for layer in self.depth_layers:
            b = torch.tanh(layer(b))
        merged = torch.cat([a, b], dim=1)
        for layer in self.fusion_layers:
            merged = torch.tanh(layer(merged))
        return self.output(merged)
