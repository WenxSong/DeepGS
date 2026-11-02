"""The four published methods, assembled from the shared components."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures

from .config import DataConfig, DeepGSConfig, ExtendedDeepGSConfig, FieldConfig, GSRConfig, LibraryConfig, SplineSRConfig
from .data import Observations, load_observations
from .grouping import build_system
from .inversion import run_inversion
from .library import CandidateLibrary, average_libraries, finite_difference_library, library_at_points, library_on_grid
from .networks import FieldNetwork, HeterogeneousCoefficient, StateCoefficient, observation_bounds
from .physics import coefficient_errors, load_heterogeneous_soil
from .pinn import fit_field, halton_points, train_pinn
from .sparse import StructureResult, select_sgtr, select_spline
from .utils import resolve_device, save_json, set_seed


def load_data(config: DataConfig) -> Observations:
    return load_observations(config.path, config.format, config.depths, config.noise, config.noise_seed,
                             config.depth_max, config.resample_minutes, config.minutes_per_row)


def build_field(config: FieldConfig, observations: Observations, device: torch.device) -> FieldNetwork:
    bounds = observation_bounds(observations) if config.normalize_inputs else None
    model = FieldNetwork([config.width] * config.hidden_layers, config.activation, bounds, xavier=config.xavier)
    return model.to(device=device, dtype=getattr(torch, config.dtype))


def candidate_grid(config: LibraryConfig, observations: Observations) -> tuple[np.ndarray, np.ndarray]:
    low = observations.depths.min() if config.depth_min is None else config.depth_min
    high = observations.depths.max() if config.depth_max is None else config.depth_max
    depths = np.linspace(low, high, config.depth_count)
    times = observations.times
    if config.time_count is not None:
        times = np.linspace(times.min(), times.max(), config.time_count)
    return depths, times


def _save_structure(output: Path, structure: StructureResult) -> None:
    save_json(output / "structure.json", structure.summary())
    extra = {} if structure.depth_centers is None else {"depth_centers": structure.depth_centers}
    if structure.spline is not None:
        extra.update(knots=structure.spline["knots"], control_points=structure.spline["control_points"])
    np.savez(output / "coarse_coefficients.npz", coefficients=structure.coefficients,
             theta_centers=structure.theta_centers, active=np.asarray(structure.active), **extra)


# --- polynomial coefficient curves (Song et al., 2022) ----------------------------------------

def polynomial_curve(theta: np.ndarray, values: np.ndarray, degree: int = 10, n_points: int = 50):
    """Remove local outliers among the grouped coefficients and fit a ridge polynomial."""
    pairs = np.column_stack((theta, values))
    keep = np.ones(theta.size, dtype=bool)
    if theta.size > 20:
        keep = LocalOutlierFactor(n_neighbors=20, contamination=0.001).fit_predict(pairs) == 1
    features = PolynomialFeatures(degree=degree)
    model = Ridge(alpha=1e-8).fit(features.fit_transform(theta[keep, None]), values[keep])
    grid = np.linspace(theta[keep].min(), theta[keep].max(), n_points)
    return grid, model.predict(features.transform(grid[:, None]))


# --- Song et al. (2022) -----------------------------------------------------------------------

def run_gsr(config: GSRConfig, output: str | Path | None = None, device: str | None = None) -> StructureResult:
    observations = load_data(config.data)
    if config.differentiation == "fd":
        library = finite_difference_library(observations.theta.T, observations.depths, observations.times, config.library.max_order)
    elif config.differentiation == "dnn":
        target = resolve_device(device)
        libraries = []
        for seed in config.seeds:
            set_seed(seed)
            model = build_field(config.field, observations, target)
            fit_field(model, observations, config.field.iterations, config.field.learning_rate,
                      log_every=max(config.field.iterations // 10, 1), label=f"field seed {seed}",
                      lbfgs_iterations=config.field.lbfgs_iterations)
            depths, times = candidate_grid(config.library, observations)
            libraries.append(library_on_grid(model, depths, times, config.library.max_order))
        library = average_libraries(libraries)
    else:
        raise ValueError(f"Unknown differentiation: {config.differentiation}")

    system = build_system(library, **asdict(config.group))
    structure = select_sgtr(system, **asdict(config.sgtr))
    if output is not None:
        output = Path(output)
        _save_structure(output, structure)
        curves = {}
        for feature, name in zip(structure.active, structure.names):
            grid, fitted = polynomial_curve(structure.theta_centers, structure.coefficients[feature], config.poly_degree)
            curves[f"{name}/theta"], curves[f"{name}/coefficient"] = grid, fitted
        np.savez(output / "coefficient_curves.npz", **curves)
    return structure


# --- Song et al. (2023) -----------------------------------------------------------------------

def _select_deepgs(model, observations: Observations, config: DeepGSConfig, seed_offset: int) -> tuple[StructureResult, list[str], tuple[float, float]]:
    """Group sparse regression on candidates predicted over quasi-random points."""
    points = halton_points(config.sparse_points, observations, seed_offset)
    library = library_at_points(model, points[:, :1], points[:, 1:], config.max_order)
    structure = select_sgtr(build_system(library, **asdict(config.group)), **asdict(config.sgtr))
    return structure, structure.names, (float(library.theta.min()), float(library.theta.max()))


def run_deepgs(config: DeepGSConfig, output: str | Path | None = None, device: str | None = None) -> dict:
    target = resolve_device(device)
    set_seed(config.seed)
    observations = load_data(config.data)
    field = build_field(config.field, observations, target)
    collocation = halton_points(config.collocation_points, observations)
    history = fit_field(field, observations, config.field.iterations, config.field.learning_rate,
                        log_every=config.log_every, label="pre-training", lbfgs_iterations=config.field.lbfgs_iterations)

    def coefficient_nets(names, bounds):
        nets = torch.nn.ModuleList(StateCoefficient(config.coefficient_hidden, *bounds) for _ in names)
        return nets.to(device=target, dtype=getattr(torch, config.field.dtype))

    def adam_with_decay(parameters):
        optimizer = torch.optim.Adam(parameters, lr=config.field.learning_rate)
        floor = config.min_lr / config.field.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(config.lr_decay ** (step / config.lr_decay_every), floor))
        return optimizer, scheduler.step

    structure, names, bounds = _select_deepgs(field, observations, config, 0)
    selections = [names]
    print(f"[deepgs] initial structure: {names}", flush=True)
    nets = coefficient_nets(names, bounds)

    parameters = list(field.parameters()) + list(nets.parameters())
    optimizer, scheduler = adam_with_decay(parameters)
    for round_index in range(config.ado_rounds):
        history += train_pinn(field, nets, names, observations, collocation, config.ado_steps, config.data_weight,
                              config.pde_weight, optimizer, scheduler, config.log_every, f"ADO {round_index + 1}")
        structure, new_names, bounds = _select_deepgs(field, observations, config, round_index + 1)
        selections.append(new_names)
        print(f"[deepgs] round {round_index + 1} structure: {new_names}", flush=True)
        if new_names != names:
            # A changed structure needs a matching set of coefficient networks.
            names = new_names
            nets = coefficient_nets(names, bounds)
            optimizer, scheduler = adam_with_decay(list(field.parameters()) + list(nets.parameters()))

    optimizer, scheduler = adam_with_decay(list(field.parameters()) + list(nets.parameters()))
    history += train_pinn(field, nets, names, observations, collocation, config.post_steps, config.data_weight,
                          config.pde_weight, optimizer, scheduler, config.log_every, "post-training")

    grid = np.linspace(*bounds, 200)
    with torch.no_grad():
        theta = torch.as_tensor(grid[:, None], dtype=getattr(torch, config.field.dtype), device=target)
        curves = np.stack([net(theta).cpu().numpy().ravel() for net in nets]) if names else np.empty((0, grid.size))
    result = dict(structure=structure, terms=names, selections=selections, theta_grid=grid, curves=curves, history=history,
                  field=field, coefficients=nets)
    if output is not None:
        output = Path(output)
        _save_structure(output, structure)
        save_json(output / "training.json", dict(terms=names, selections=selections, history=history))
        np.savez(output / "coefficient_curves.npz", theta=grid, curves=curves, terms=np.asarray(names))
        torch.save(dict(field=field.state_dict(), coefficients=nets.state_dict()), output / "model.pt")
    return result


# --- Song et al. (2025) -----------------------------------------------------------------------

def run_extended_deepgs(config: ExtendedDeepGSConfig, output: str | Path | None = None, device: str | None = None) -> dict:
    target = resolve_device(device)
    set_seed(config.seed)
    observations = load_data(config.data)

    # Algorithm 1: coarse-grained group sparse regression fixes the structure.
    pretrained = build_field(config.field, observations, target)
    history = fit_field(pretrained, observations, config.field.iterations, config.field.learning_rate,
                        log_every=config.log_every, label="pre-training", lbfgs_iterations=config.field.lbfgs_iterations)
    depths, times = candidate_grid(config.library, observations)
    library = library_on_grid(pretrained, depths, times, config.library.max_order)
    structure = select_sgtr(build_system(library, **asdict(config.group)), **asdict(config.sgtr))
    names = structure.names
    print(f"[extended] structure: {names}", flush=True)

    # Algorithm 2: fresh networks trained under the identified structure.
    field = build_field(config.field, observations, target)
    dtype = getattr(torch, config.field.dtype)
    nets = torch.nn.ModuleList(HeterogeneousCoefficient(config.theta_hidden, config.depth_hidden, config.fusion_hidden) for _ in names)
    nets = nets.to(device=target, dtype=dtype)
    optimizer = torch.optim.Adam(list(field.parameters()) + list(nets.parameters()))
    collocation = halton_points(config.collocation_points, observations)
    history += train_pinn(field, nets, names, observations, collocation, config.train_steps, config.data_weight,
                          config.pde_weight, optimizer, log_every=config.log_every, label="extended")

    grid_z, grid_t = np.meshgrid(depths, times)
    z = torch.as_tensor(grid_z.reshape(-1, 1), dtype=dtype, device=target)
    t = torch.as_tensor(grid_t.reshape(-1, 1), dtype=dtype, device=target)
    with torch.no_grad():
        theta = field(z, t)
        coefficient_fields = (np.stack([net(theta, z).cpu().numpy().reshape(grid_z.shape) for net in nets])
                              if names else np.empty((0, *grid_z.shape)))
        theta_field = theta.cpu().numpy().reshape(grid_z.shape)
    errors = None
    if config.reference.path:
        soil = load_heterogeneous_soil(config.reference.path, config.reference.theta_s, config.reference.theta_r,
                                       config.reference.depth_max)
        reference = soil.coefficients(theta_field, grid_z)
        errors = coefficient_errors(dict(zip(names, coefficient_fields)), reference)
        print(f"[extended] coefficient errors against the reference soil: {errors}", flush=True)
    result = dict(structure=structure, terms=names, depths=depths, times=times, theta=theta_field,
                  coefficient_fields=coefficient_fields, history=history, field=field, coefficients=nets,
                  coefficient_errors=errors)
    if output is not None:
        output = Path(output)
        _save_structure(output, structure)
        save_json(output / "training.json", dict(terms=names, coefficient_errors=errors, history=history))
        np.savez(output / "fields.npz", depths=depths, times=times, theta=theta_field,
                 coefficient_fields=coefficient_fields, terms=np.asarray(names))
        torch.save(dict(field=field.state_dict(), coefficients=nets.state_dict()), output / "model.pt")
    return result


# --- Spline-SR --------------------------------------------------------------------------------

def spline_library(config: SplineSRConfig, observations: Observations, device: torch.device) -> CandidateLibrary:
    set_seed(config.seed)
    model = build_field(config.field, observations, device)
    fit_field(model, observations, config.field.iterations, config.field.learning_rate,
              log_every=max(config.field.iterations // 10, 1), label="field",
              lbfgs_iterations=config.field.lbfgs_iterations)
    depths, times = candidate_grid(config.library, observations)
    return library_on_grid(model, depths, times, config.library.max_order)


def run_spline_sr(config: SplineSRConfig, output: str | Path | None = None, device: str | None = None) -> dict:
    target = resolve_device(device)
    observations = load_data(config.data)
    library = spline_library(config, observations, target)
    structure = select_spline(build_system(library, **asdict(config.group)), **asdict(config.spline))
    result = dict(structure=structure, inversion=None, status="no knee on the error-complexity curve")
    if structure is not None:
        result["status"] = "structure selected"
        print(f"[spline-sr] structure: {structure.names}", flush=True)
        if output is not None:
            _save_structure(Path(output), structure)
        if config.invert:
            polynomial = config.inversion.coefficient_form == "polynomial"
            if not structure.names:
                result["status"] = "empty structure selected; inversion skipped"
            elif polynomial and set(structure.names) != set(config.prior):
                result["status"] = "structure differs from the prior; polynomial inversion skipped"
            else:
                signs = [config.prior[name] for name in structure.names] if polynomial else None
                envelope = tuple(config.envelope) if config.envelope else None
                result["inversion"] = run_inversion(observations, structure.names, signs, target,
                                                    config.inversion, config.seed, envelope)
                result["status"] = "inverted"
    if output is not None:
        output = Path(output)
        summary = dict(status=result["status"])
        inversion = result["inversion"]
        if inversion is not None:
            controls = inversion["controls"]
            summary.update(theta_rmse=inversion["theta_rmse"], bounds=list(inversion["bounds"]),
                           coefficient_form=config.inversion.coefficient_form, history=inversion["history"])
            extra = {}
            if controls is not None:
                summary["controls"] = controls.tolist()
                extra["controls"] = controls
            np.savez(output / "inversion.npz", theta_grid=inversion["theta_grid"], curves=inversion["curves"],
                     fitted=inversion["fitted"], terms=np.asarray(structure.names), **extra)
            state = dict(field=inversion["field"].state_dict())
            if inversion.get("coefficients") is not None:
                state["coefficients"] = inversion["coefficients"].state_dict()
            torch.save(state, output / "model.pt")
        save_json(output / "summary.json", summary)
    return result


RUNNERS = {"gsr": run_gsr, "deepgs": run_deepgs, "extended_deepgs": run_extended_deepgs, "spline_sr": run_spline_sr}
