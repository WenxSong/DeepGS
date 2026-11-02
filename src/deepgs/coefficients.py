"""Fixed-structure coefficient representations with one ordered matrix interface."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .experiment import BSplineConfig, InversionConfig, NetworkConfig, PolynomialConfig
from .networks import HeterogeneousCoefficient, StateCoefficient


def _input(theta, bounds):
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError("theta must have shape (n_points,1)")
    if not torch.isfinite(theta).all():
        raise ValueError("theta query must be finite")
    if (theta < bounds[0]).any() or (theta > bounds[1]).any():
        raise ValueError("Coefficient query lies outside theta_bounds")


class NetworkCoefficient(nn.Module):
    def __init__(self, terms, soil, bounds, settings: NetworkConfig):
        super().__init__()
        self.terms = tuple(terms)
        self.soil = soil
        self.register_buffer("theta_bounds", torch.as_tensor(bounds, dtype=torch.float64))
        if soil == "homogeneous":
            self.branches = nn.ModuleList(StateCoefficient(settings.hidden_sizes, *bounds) for _ in terms)
        elif soil == "heterogeneous":
            fusion = settings.fusion_hidden if settings.architecture == "fusion" else ()
            self.branches = nn.ModuleList(HeterogeneousCoefficient(settings.theta_hidden, settings.depth_hidden, fusion) for _ in terms)
        else:
            raise ValueError("Unsupported soil type")
        self._spec = dict(form="network", soil=soil, terms=list(terms), theta_bounds=list(bounds),
                          settings=deepcopy(settings.__dict__))

    @property
    def spec(self):
        return deepcopy(self._spec)

    def forward(self, theta, z=None):
        _input(theta, self.theta_bounds)
        if self.soil == "heterogeneous":
            if z is None or z.shape != theta.shape or not torch.isfinite(z).all():
                raise ValueError("Heterogeneous coefficients require aligned finite z")
        return torch.cat([branch(theta, z) for branch in self.branches], dim=1)


class PolynomialCoefficient(nn.Module):
    def __init__(self, terms, bounds, signs):
        super().__init__()
        self.terms = tuple(terms)
        self.register_buffer("theta_bounds", torch.as_tensor(bounds, dtype=torch.float64))
        self.register_buffer("signs", torch.as_tensor(signs, dtype=torch.float64))
        self.register_buffer("controls", torch.zeros(len(terms), 4, dtype=torch.float64))
        self._spec = dict(form="polynomial", soil="homogeneous", terms=list(terms), theta_bounds=list(bounds), signs=list(signs))

    @property
    def spec(self):
        return deepcopy(self._spec)

    def forward(self, theta, z=None):
        _input(theta, self.theta_bounds)
        q = (theta - self.theta_bounds[0]) / (self.theta_bounds[1] - self.theta_bounds[0])
        basis = torch.cat([q**k for k in range(4)], dim=1)
        return (basis @ self.controls.T) * self.signs[None, :]


class SplineCoefficient(nn.Module):
    def __init__(self, terms, bounds, settings: BSplineConfig):
        super().__init__()
        from .bspline import BSplineCoefficient
        self.terms = tuple(terms)
        self.basis_model = BSplineCoefficient(bounds, len(terms), settings.n_internal_knots)
        self._spec = dict(form="bspline", soil="homogeneous", terms=list(terms), theta_bounds=list(bounds),
                          degree=3, n_internal_knots=settings.n_internal_knots)

    @property
    def spec(self):
        return deepcopy(self._spec)

    @property
    def controls(self):
        return self.basis_model.controls

    def forward(self, theta, z=None):
        return self.basis_model(theta, z)


def build_coefficient_model(terms, soil, config: InversionConfig, bounds):
    """Build a model; the caller owns independent initialization RNG and dtype."""
    if not terms or len(set(terms)) != len(terms):
        raise ValueError("Coefficient terms must be nonempty and unique")
    if soil not in ("homogeneous", "heterogeneous"):
        raise ValueError("Unsupported soil type")
    if soil == "heterogeneous" and config.coefficient_form != "network":
        raise ValueError("Heterogeneous inversion supports only network")
    if config.coefficient_form == "network":
        return NetworkCoefficient(terms, soil, bounds, config.network or NetworkConfig())
    if config.coefficient_form == "polynomial":
        settings = config.polynomial or PolynomialConfig()
        missing = set(terms) - settings.signs.keys()
        if missing:
            raise ValueError(f"Missing polynomial signs for {sorted(missing)}")
        return PolynomialCoefficient(terms, bounds, [settings.signs[n] for n in terms])
    if config.coefficient_form == "bspline":
        return SplineCoefficient(terms, bounds, config.bspline or BSplineConfig())
    raise ValueError("Unsupported coefficient_form")


def coefficient_from_spec(spec):
    """Reconstruct architecture before a checkpoint loader applies state_dict."""
    form, terms, soil, bounds = spec["form"], spec["terms"], spec["soil"], spec["theta_bounds"]
    if form == "network":
        return NetworkCoefficient(terms, soil, bounds, NetworkConfig(**spec["settings"]))
    if form == "polynomial" and soil == "homogeneous":
        return PolynomialCoefficient(terms, bounds, spec["signs"])
    if form == "bspline" and soil == "homogeneous":
        if spec.get("degree") != 3:
            raise ValueError("Only cubic B-splines are supported")
        return SplineCoefficient(terms, bounds, BSplineConfig(degree=3, n_internal_knots=spec["n_internal_knots"]))
    raise ValueError("Unsupported checkpoint coefficient specification")
