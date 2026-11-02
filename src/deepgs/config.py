"""Experiment configuration. Each published method has its own defaults, overridable from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .inversion import InversionSettings


@dataclass
class DataConfig:
    path: str = ""
    format: str | None = None
    depths: list[float] | None = None
    noise: float = 0.0
    noise_seed: int = 0
    depth_max: float = 1.0
    resample_minutes: bool = False
    minutes_per_row: float = 1.0


@dataclass
class FieldConfig:
    hidden_layers: int = 5
    width: int = 50
    activation: str = "tanh"
    normalize_inputs: bool = True
    xavier: bool = True
    iterations: int = 20000
    learning_rate: float = 1e-3
    lbfgs_iterations: int = 0
    dtype: str = "float32"


@dataclass
class LibraryConfig:
    max_order: int = 3
    depth_count: int = 20
    time_count: int | None = None  # None uses the observation times
    depth_min: float | None = None
    depth_max: float | None = None


@dataclass
class GroupConfig:
    win_size: float = 0.001
    min_group_size: int = 200
    normalization: str = "global"
    by_depth: bool = False
    merge_tail: float = 0.01


@dataclass
class SGTRConfig:
    num_tols: int = 200
    lam: float = 1e-4
    maxit: int = 1
    epsilon: float = 1e-2
    corrected: bool = True
    min_energy_gain: float | None = None  # overrides epsilon; see sparse.select_sgtr


@dataclass
class SplineConfig:
    n_knots: int = 6
    degree: int = 3
    lam: float = 0.01
    refit_lam: float | None = None


@dataclass
class GSRConfig:
    """Song et al. (2022): grouped sparse regression on a finite-difference or DNN library."""

    data: DataConfig = dc_field(default_factory=DataConfig)
    differentiation: str = "dnn"  # 'fd' or 'dnn'
    seeds: list[int] = dc_field(default_factory=lambda: [1, 2, 3, 4, 5])
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    library: LibraryConfig = dc_field(default_factory=LibraryConfig)
    group: GroupConfig = dc_field(default_factory=GroupConfig)
    sgtr: SGTRConfig = dc_field(default_factory=SGTRConfig)
    poly_degree: int = 10


@dataclass
class DeepGSConfig:
    """Song et al. (2023): group sparsity and physics-informed training in alternation."""

    data: DataConfig = dc_field(default_factory=DataConfig)
    seed: int = 0
    field: FieldConfig = dc_field(default_factory=lambda: FieldConfig(hidden_layers=7, iterations=50000))
    coefficient_hidden: list[int] = dc_field(default_factory=lambda: [20, 20])
    group: GroupConfig = dc_field(default_factory=GroupConfig)
    sgtr: SGTRConfig = dc_field(default_factory=lambda: SGTRConfig(num_tols=500, lam=1e-2, maxit=10, epsilon=10**-1.5, corrected=False))
    max_order: int = 3
    data_weight: float = 100.0
    pde_weight: float = 1.0
    collocation_points: int = 10000
    sparse_points: int = 100000
    ado_rounds: int = 5
    ado_steps: int = 100000
    post_steps: int = 100000
    lr_decay: float = 0.95
    lr_decay_every: int = 1000
    min_lr: float = 1e-6
    log_every: int = 2000


@dataclass
class ReferenceSoilConfig:
    """Known parameters of a synthetic heterogeneous soil, used only to score the results."""

    path: str | None = None
    theta_s: float | None = None
    theta_r: float | None = None
    depth_max: float = 1.0


@dataclass
class ExtendedDeepGSConfig:
    """Song et al. (2025): coarse-grained regression and coefficients that depend on theta and depth.

    In YAML this is 'pipeline: deepgs' with 'soil: heterogeneous'.
    """

    data: DataConfig = dc_field(default_factory=DataConfig)
    reference: ReferenceSoilConfig = dc_field(default_factory=ReferenceSoilConfig)
    seed: int = 0
    field: FieldConfig = dc_field(default_factory=lambda: FieldConfig(hidden_layers=6, activation="sin", normalize_inputs=False, xavier=False))
    library: LibraryConfig = dc_field(default_factory=lambda: LibraryConfig(depth_count=10))
    group: GroupConfig = dc_field(default_factory=lambda: GroupConfig(min_group_size=100, normalization="group", by_depth=True))
    sgtr: SGTRConfig = dc_field(default_factory=lambda: SGTRConfig(num_tols=500, lam=1e-2, epsilon=10**1.2, corrected=False))
    theta_hidden: list[int] = dc_field(default_factory=lambda: [40, 40, 40, 40])
    depth_hidden: list[int] = dc_field(default_factory=lambda: [10])
    fusion_hidden: list[int] = dc_field(default_factory=list)  # empty: original additive branches
    data_weight: float = 10.0
    pde_weight: float = 1.0
    collocation_points: int = 20000
    train_steps: int = 200000
    log_every: int = 2000


@dataclass
class SplineSRConfig:
    """Spline-regularized sparse regression followed by coefficient inversion."""

    data: DataConfig = dc_field(default_factory=DataConfig)
    seed: int = 2026
    field: FieldConfig = dc_field(default_factory=lambda: FieldConfig(activation="silu", normalize_inputs=False, xavier=False))
    library: LibraryConfig = dc_field(default_factory=lambda: LibraryConfig(max_order=4, depth_count=6, time_count=5000))
    group: GroupConfig = dc_field(default_factory=lambda: GroupConfig(min_group_size=100, merge_tail=0.0))
    spline: SplineConfig = dc_field(default_factory=SplineConfig)
    invert: bool = True
    # Physical prior for the polynomial form, which runs only when exactly these terms are selected.
    # Values are the sign and direction. The network form inverts any selected structure.
    prior: dict[str, int] = dc_field(default_factory=lambda: {"theta_z": -1, "theta_zz": 1, "theta_z^2": 1})
    envelope: list[float] | None = None
    inversion: InversionSettings = dc_field(default_factory=InversionSettings)


PIPELINES = {"gsr": GSRConfig, "deepgs": DeepGSConfig, "extended_deepgs": ExtendedDeepGSConfig, "spline_sr": SplineSRConfig}
# 'pipeline: deepgs' selects the workflow for the soil type.
SOIL_TYPES = {"homogeneous": "deepgs", "heterogeneous": "extended_deepgs"}


def _build(cls, values: dict[str, Any], base=None):
    """Create a dataclass from a nested dict, starting from the defaults of its parent, and reject unknown keys."""
    instance = base if base is not None else cls()
    known = {f.name: f for f in fields(instance)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown option '{key}' for {cls.__name__}")
        current = getattr(instance, key)
        if is_dataclass(current):
            value = _build(type(current), value, current)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        object.__setattr__(instance, key, value)
    return instance


def load_config(path: str | Path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    name = raw.pop("pipeline", None)
    if name not in PIPELINES:
        raise ValueError(f"'pipeline' must be one of {sorted(PIPELINES)}")
    soil = raw.pop("soil", None)
    if soil is not None:
        if name != "deepgs":
            raise ValueError("'soil' applies only to the deepgs pipeline")
        if soil not in SOIL_TYPES:
            raise ValueError(f"'soil' must be one of {sorted(SOIL_TYPES)}")
        name = SOIL_TYPES[soil]
    return name, _build(PIPELINES[name], raw)
