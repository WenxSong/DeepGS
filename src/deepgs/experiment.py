"""Validated schema-2 configuration for the two-stage workflow.

Historical pipeline configurations remain in :mod:`deepgs.config`.  This module
does not infer a published method from a combination of settings.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field as dc_field, fields, is_dataclass
import math
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from .config import DataConfig, FieldConfig, LibraryConfig, SGTRConfig
from .library import term_names


@dataclass
class UnitsConfig:
    time: str = "day"
    depth: str = "m"
    water_content: str = "fraction"


@dataclass
class ExperimentDataConfig(DataConfig):
    units: UnitsConfig = dc_field(default_factory=UnitsConfig)
    depth_positive: str = "down"


@dataclass
class ProblemConfig:
    soil: str = "homogeneous"


@dataclass
class PreprocessConfig:
    differentiation: str = "dnn"
    seeds: list[int] = dc_field(default_factory=lambda: [0])
    field: FieldConfig = dc_field(default_factory=lambda: FieldConfig(activation="silu", dtype="float64"))


@dataclass
class GroupingConfig:
    win_size: float = 0.001
    min_group_size: int = 100
    normalization: str = "global"
    merge_tail: float = 0.0


@dataclass
class SelectionSplineConfig:
    degree: int = 3
    n_internal_knots: int = 6
    lam: float = 0.01
    refit_lam: float | None = None


@dataclass
class StructureConfig:
    selector: str = "group"
    grouping: GroupingConfig = dc_field(default_factory=GroupingConfig)
    sgtr: SGTRConfig = dc_field(default_factory=SGTRConfig)
    spline: SelectionSplineConfig = dc_field(default_factory=SelectionSplineConfig)


@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = dc_field(default_factory=lambda: [20, 20])
    architecture: str = "additive"
    theta_hidden: list[int] = dc_field(default_factory=lambda: [40, 40, 40, 40])
    depth_hidden: list[int] = dc_field(default_factory=lambda: [10])
    fusion_hidden: list[int] = dc_field(default_factory=lambda: [40, 40])


@dataclass
class PolynomialConfig:
    degree: int = 3
    signs: dict[str, int] = dc_field(default_factory=dict)


@dataclass
class BSplineConfig:
    degree: int = 3
    n_internal_knots: int = 6


@dataclass
class DomainConfig:
    theta_bounds: str | list[float] = "auto"
    margin: float = 0.01


@dataclass
class TrainingConfig:
    epochs: int = 20000
    learning_rate: float = 0.001
    data_weight: float = 100.0
    pde_weight: float = 1.0
    log_every: int = 1000


@dataclass
class CollocationConfig:
    sampler: str = "dynamic_lhs"
    points: int = 2000
    dynamic_fraction: float = 0.9


@dataclass
class InversionConfig:
    enabled: bool = False
    coefficient_form: str = "network"
    seed: int = 0
    field: FieldConfig = dc_field(default_factory=lambda: FieldConfig(activation="silu", dtype="float64"))
    domain: DomainConfig = dc_field(default_factory=DomainConfig)
    network: NetworkConfig | None = None
    polynomial: PolynomialConfig | None = None
    bspline: BSplineConfig | None = None
    training: TrainingConfig = dc_field(default_factory=TrainingConfig)
    collocation: CollocationConfig = dc_field(default_factory=CollocationConfig)


@dataclass
class RuntimeConfig:
    device: str = "auto"
    save_library: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = 2
    problem: ProblemConfig = dc_field(default_factory=ProblemConfig)
    data: ExperimentDataConfig = dc_field(default_factory=ExperimentDataConfig)
    preprocess: PreprocessConfig = dc_field(default_factory=PreprocessConfig)
    library: LibraryConfig = dc_field(default_factory=LibraryConfig)
    structure: StructureConfig = dc_field(default_factory=StructureConfig)
    inversion: InversionConfig = dc_field(default_factory=InversionConfig)
    runtime: RuntimeConfig = dc_field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "ExperimentConfig":
        return validate_experiment(self)


def _value(expected, value, name):
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        nonnull = [choice for choice in args if choice is not type(None)]
        if len(nonnull) == 1:
            return _value(nonnull[0], value, name)
        for choice in args:
            try:
                return _value(choice, value, name)
            except ValueError:
                pass
        raise ValueError(f"{name}: invalid value/type {value!r}")
    if expected is type(None):
        if value is None:
            return None
    elif is_dataclass(expected):
        return _build(expected, value, name)
    elif origin is list:
        if isinstance(value, list):
            return [_value(args[0], v, f"{name}[{i}]") for i, v in enumerate(value)]
    elif origin is dict:
        if isinstance(value, dict):
            return {_value(args[0], k, name): _value(args[1], v, f"{name}.{k}") for k, v in value.items()}
    elif expected is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif type(value) is expected:
        return value
    raise ValueError(f"{name}: expected {expected}, got {value!r}")


def _build(cls, raw, name="config"):
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown option at {name}: {sorted(unknown, key=str)}")
    hints = get_type_hints(cls)
    return cls(**{key: _value(hints[key], value, f"{name}.{key}") for key, value in raw.items()})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, name, allow_zero=False):
    _require(math.isfinite(value) and (value >= 0 if allow_zero else value > 0),
             f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")


def _field(config, name):
    _require(config.activation in ("tanh", "sin", "silu"), f"{name}.activation is unsupported")
    _require(config.dtype in ("float32", "float64"), f"{name}.dtype must be float32 or float64")
    for key in ("hidden_layers", "width"):
        _positive(getattr(config, key), f"{name}.{key}")
    for key in ("iterations", "lbfgs_iterations"):
        _positive(getattr(config, key), f"{name}.{key}", True)
    _positive(config.learning_rate, f"{name}.learning_rate")


def _validate_data(d):
    _require(bool(d.path.strip()), "data.path is required")
    _require(d.format in (None, "txt", "npz", "csv"), "data.format must be txt, npz or csv")
    _require(d.units.time in ("day", "hour", "minute", "second"), "Unsupported data.units.time")
    _require(d.units.depth in ("m", "cm", "mm"), "Unsupported data.units.depth")
    _require(d.units.water_content in ("fraction", "percent"), "Unsupported data.units.water_content")
    _require(d.depth_positive == "down", "data.depth_positive must be down")
    _positive(d.depth_max, "data.depth_max")
    _positive(d.minutes_per_row, "data.minutes_per_row")
    _positive(d.noise, "data.noise", True)
    _positive(d.noise_seed, "data.noise_seed", True)
    if d.depths is not None:
        _require(len(d.depths) >= 1, "data.depths cannot be empty")
        _require(all(a < b for a, b in zip(d.depths, d.depths[1:])), "data.depths must be strictly increasing")
    is_csv = d.format == "csv" or (d.format is None and Path(d.path).suffix.lower() == ".csv")
    if is_csv:
        _require(d.units.time == "day" and d.units.depth == "m",
                 "CSV coordinates require day and m units; explicit time or minutes_per_row defines sampling")


def _validate_preprocess(p):
    _require(p.differentiation in ("fd", "dnn"), "preprocess.differentiation must be fd or dnn")
    _require(bool(p.seeds) and all(seed >= 0 for seed in p.seeds), "preprocess.seeds must be nonempty nonnegative integers")
    _require(len(set(p.seeds)) == len(p.seeds), "preprocess.seeds must be unique")
    _field(p.field, "preprocess.field")


def _validate_library(lib):
    _require(lib.max_order in (1, 2, 3, 4), "library.max_order must be 1 through 4")
    _require(lib.depth_count >= 2, "library.depth_count must be at least 2")
    if lib.time_count is not None:
        _require(lib.time_count >= 2, "library.time_count must be at least 2")
    if lib.depth_min is not None and lib.depth_max is not None:
        _require(lib.depth_min < lib.depth_max, "library.depth_min must be below depth_max")


def _validate_structure(s, soil):
    _require(s.selector in ("group", "spline", "spatial_group"), "Unsupported structure.selector")
    expected_soil = "heterogeneous" if s.selector == "spatial_group" else "homogeneous"
    _require(soil == expected_soil, f"{s.selector} requires {expected_soil} soil")
    _require(s.grouping.normalization in ("global", "group"), "Unsupported grouping.normalization")
    _require(s.selector != "spline" or s.grouping.normalization == "global", "spline selection requires global normalization")
    _positive(s.grouping.win_size, "structure.grouping.win_size")
    _positive(s.grouping.min_group_size, "structure.grouping.min_group_size")
    _require(0 <= s.grouping.merge_tail < 0.5, "structure.grouping.merge_tail must be in [0, 0.5)")
    for key in ("num_tols", "maxit"):
        _positive(getattr(s.sgtr, key), f"structure.sgtr.{key}")
    _require(s.sgtr.num_tols >= 2, "structure.sgtr.num_tols must be at least 2")
    for key in ("lam", "epsilon"):
        _positive(getattr(s.sgtr, key), f"structure.sgtr.{key}", key == "lam")
    if s.sgtr.min_energy_gain is not None:
        _require(0 < s.sgtr.min_energy_gain <= 1, "structure.sgtr.min_energy_gain must be in (0,1]")
    _require(s.spline.degree == 3, "structure.spline.degree must be 3 in this release")
    _positive(s.spline.n_internal_knots, "structure.spline.n_internal_knots", True)
    _positive(s.spline.lam, "structure.spline.lam")
    if s.spline.refit_lam is not None:
        _positive(s.spline.refit_lam, "structure.spline.refit_lam")


def _validate_inversion(inv, selector, soil, max_order):
    _require(inv.coefficient_form in ("network", "polynomial", "bspline"), "Unsupported inversion.coefficient_form")
    from .capabilities import validate_combination
    validate_combination(selector, inv.coefficient_form, soil)
    if soil == "heterogeneous":
        _require(inv.coefficient_form == "network", "heterogeneous inversion supports only network")
    for name in ("network", "polynomial", "bspline"):
        _require(getattr(inv, name) is None or inv.coefficient_form == name, f"inversion.{name} does not apply to {inv.coefficient_form}")
    _positive(inv.seed, "inversion.seed", True)
    _field(inv.field, "inversion.field")
    _positive(inv.domain.margin, "inversion.domain.margin", True)
    bounds = inv.domain.theta_bounds
    _require(bounds == "auto" or (isinstance(bounds, list) and len(bounds) == 2 and bounds[0] < bounds[1]),
             "inversion.domain.theta_bounds must be auto or [low,high]")
    if inv.network:
        _require(inv.network.architecture in ("additive", "fusion"), "Unsupported inversion.network.architecture")
        _require(soil != "homogeneous" or inv.network.architecture == "additive",
                 "inversion.network.architecture=fusion applies only to heterogeneous soil")
        for name in ("hidden_sizes", "theta_hidden", "depth_hidden", "fusion_hidden"):
            widths = getattr(inv.network, name)
            _require(bool(widths) and all(w > 0 for w in widths), f"inversion.network.{name} needs positive widths")
    if inv.polynomial:
        _require(inv.polynomial.degree == 3, "inversion.polynomial.degree must be 3")
        known_terms = {"1", *term_names(max_order)}
        _require(all(k in known_terms and v in (-1, 1) for k, v in inv.polynomial.signs.items()),
                 "polynomial signs require known candidate term IDs with +/-1 values")
    if inv.bspline:
        _require(inv.bspline.degree == 3, "inversion.bspline.degree must be 3")
        _positive(inv.bspline.n_internal_knots, "inversion.bspline.n_internal_knots", True)
    for key in ("epochs", "log_every"):
        _positive(getattr(inv.training, key), f"inversion.training.{key}")
    for key in ("learning_rate", "data_weight", "pde_weight"):
        _positive(getattr(inv.training, key), f"inversion.training.{key}")
    _require(inv.collocation.sampler in ("dynamic_lhs", "halton"), "Unsupported inversion.collocation.sampler")
    _positive(inv.collocation.points, "inversion.collocation.points")
    _require(0 <= inv.collocation.dynamic_fraction <= 1, "inversion.collocation.dynamic_fraction must be in [0,1]")


def validate_experiment(config: ExperimentConfig) -> ExperimentConfig:
    """Validate every field, also when a caller constructed/mutated dataclasses."""
    _require(isinstance(config, ExperimentConfig), "Expected ExperimentConfig")
    _build(ExperimentConfig, asdict(config))  # strict recursive scalar/container types
    _require(config.schema_version == 2, "schema_version must be 2")
    _require(config.problem.soil in ("homogeneous", "heterogeneous"), "Unsupported problem.soil")
    _validate_data(config.data)
    _validate_preprocess(config.preprocess)
    _validate_library(config.library)
    _validate_structure(config.structure, config.problem.soil)
    _validate_inversion(config.inversion, config.structure.selector, config.problem.soil, config.library.max_order)
    device = config.runtime.device
    _require(device in ("auto", "cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit()),
             "runtime.device must be auto, cpu or cuda[:index]")
    return config


def load_experiment(source: str | Path | dict | ExperimentConfig) -> ExperimentConfig:
    """Load schema 2 without mutating input; resolve paths against its own source."""
    base = Path.cwd()
    if isinstance(source, ExperimentConfig):
        raw = source.to_dict()
    elif isinstance(source, dict):
        raw = deepcopy(source)
    else:
        path = Path(source).resolve()
        base = path.parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "Experiment configuration must be a mapping")
    _require(raw.get("schema_version") == 2, "schema_version: 2 is required; use the legacy entry for pipeline configs")
    raw = deepcopy(raw)
    if isinstance(raw.get("structure"), dict) and raw["structure"].get("selector") == "spatial_group":
        # Defaults associated with this explicit selector preserve CGSR scaling.
        grouping = raw["structure"].setdefault("grouping", {})
        if isinstance(grouping, dict):
            grouping.setdefault("normalization", "group")
        sgtr = raw["structure"].setdefault("sgtr", {})
        if isinstance(sgtr, dict):
            for key, value in dict(num_tols=500, lam=0.01, epsilon=10**1.2, corrected=False).items():
                sgtr.setdefault(key, value)
    config = _build(ExperimentConfig, raw)
    validate_experiment(config)
    config.data.path = str((base / config.data.path).resolve())
    return config
