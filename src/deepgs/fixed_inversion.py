"""PINN inversion of a frozen selected structure; never invokes a selector."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib

import numpy as np
import torch

from .coefficients import build_coefficient_model
from .contracts import InversionResult, SelectionResult
from .data import Observations
from .experiment import ExperimentConfig, InversionConfig, load_experiment
from .inversion import InversionSettings, pde_matrix, project_controls, sample_collocation
from .library import evaluate_term, required_order, spatial_derivatives, term_names
from .networks import FieldNetwork, observation_bounds
from .pinn import halton_points, to_tensors


@contextmanager
def _rng(seed, device):
    indices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=indices):
        torch.random.default_generator.manual_seed(seed)
        if indices:
            torch.cuda.default_generators[indices[0]].manual_seed(seed)
        yield


def _state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bounds(observations, config):
    observed_low, observed_high = float(observations.theta.min()), float(observations.theta.max())
    if config.domain.theta_bounds == "auto":
        bounds = (observed_low - config.domain.margin, observed_high + config.domain.margin)
    else:
        bounds = tuple(config.domain.theta_bounds)
    if not bounds[0] < bounds[1] or observed_low < bounds[0] or observed_high > bounds[1]:
        raise ValueError("theta_bounds must have positive width and contain all training observations")
    return bounds


def _validated(config, soil):
    # Validate the public stand-alone entry with the same checks as YAML runs.
    raw = dict(schema_version=2, problem=dict(soil=soil), data=dict(path="supplied-observations.npz"), library=dict(max_order=4),
               structure=dict(selector="spatial_group" if soil == "heterogeneous" else "group"),
               inversion=asdict(config))
    return load_experiment(raw).inversion


def _residual(field, coefficients, terms, z, t):
    theta = field(z, t)
    theta_t = torch.autograd.grad(theta.sum(), t, create_graph=True)[0]
    derivatives = spatial_derivatives(theta, z, required_order(list(terms)))
    columns = torch.cat([evaluate_term(name, derivatives, like=theta) for name in terms], dim=1)
    values = coefficients(theta, z)
    if values.shape != columns.shape:
        raise ValueError("Coefficient matrix and frozen term columns disagree")
    return theta_t - (values * columns).sum(dim=1, keepdim=True)


def invert(observations: Observations, selection: SelectionResult, config: InversionConfig,
           soil="homogeneous", device="cpu") -> InversionResult:
    config = _validated(config, soil)
    terms = tuple(selection.terms)
    if selection.coarse is not None and (tuple(selection.coarse.names) != terms or tuple(selection.coarse.active) != selection.term_indices):
        raise ValueError("Coarse structure changed after selection was frozen")
    if not config.enabled:
        return InversionResult(status="not_requested", terms=terms)
    if selection.status == "no_knee":
        return InversionResult(status="skipped_no_structure", terms=terms)
    if not terms:
        return InversionResult(status="skipped_empty", terms=terms)
    if len(set(terms)) != len(terms) or any(n not in {"1", *term_names(4)} for n in terms):
        raise ValueError("Unsupported or duplicate selected term IDs")
    if config.coefficient_form == "polynomial":
        signs = config.polynomial.signs if config.polynomial else {}
        missing = [name for name in terms if name not in signs]
        if missing:
            return InversionResult(status="blocked_missing_prior", terms=terms, metadata=dict(missing_signs=missing))
    if observations.times.size < 2 or observations.depths.size < 2:
        raise ValueError("Inversion requires at least two times and two depths")
    if (not np.isfinite(observations.times).all() or not np.isfinite(observations.depths).all()
            or not np.all(np.diff(observations.times) > 0) or not np.all(np.diff(observations.depths) > 0)):
        raise ValueError("Inversion coordinates must be finite and strictly increasing")
    target_device = torch.device(device)
    dtype = getattr(torch, config.field.dtype)
    bounds = _bounds(observations, config)
    input_bounds = (tuple(tuple(float(value) for value in edge) for edge in observation_bounds(observations))
                    if config.field.normalize_inputs else None)
    field_spec = dict(hidden_sizes=[config.field.width] * config.field.hidden_layers,
                      activation=config.field.activation, bounds=input_bounds,
                      output_bounds=bounds, xavier=config.field.xavier, bounded_lerp=True)
    with _rng(config.seed, target_device):
        field = FieldNetwork(**field_spec).to(device=target_device, dtype=dtype)
    initial_field_sha256 = _state_hash(field)
    with _rng(config.seed + 1000003, target_device):
        coefficients = build_coefficient_model(terms, soil, config, bounds).to(device=target_device, dtype=dtype)
    collocation_settings = InversionSettings(collocation_points=config.collocation.points,
        dynamic_sampling_fraction=config.collocation.dynamic_fraction)
    if config.collocation.sampler == "dynamic_lhs":
        collocation = sample_collocation(observations, collocation_settings, config.seed + 97)
    else:
        collocation = halton_points(config.collocation.points, observations, config.seed + 97)
    collocation_sha256 = hashlib.sha256(np.asarray(collocation, dtype="<f8").tobytes()).hexdigest()
    polynomial = config.coefficient_form == "polynomial"
    parameters = list(field.parameters()) + ([] if polynomial else list(coefficients.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=config.training.learning_rate)
    z, t = to_tensors(observations.points(), next(field.parameters()))
    observed = torch.as_tensor(observations.values(), dtype=dtype, device=target_device)
    z_r, t_r = to_tensors(collocation, next(field.parameters()), requires_grad=True)
    history = []
    for step in range(1, config.training.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        if polynomial:
            matrix, response = pde_matrix(field, terms, coefficients.signs.tolist(), bounds, z_r, t_r)
            controls = project_controls(matrix, response)
            pde_loss = (matrix @ controls - response).square().mean()
            coefficients.controls.copy_(controls.reshape(len(terms), 4))
        else:
            pde_loss = _residual(field, coefficients, terms, z_r, t_r).square().mean()
        data_loss = (field(z, t) - observed).square().mean()
        loss = config.training.data_weight * data_loss + config.training.pde_weight * pde_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite inversion loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError(f"Non-finite inversion gradient at step {step}")
        optimizer.step()
        if step == 1 or step % config.training.log_every == 0 or step == config.training.epochs:
            history.append(dict(step=step, loss=float(loss.detach()), data_loss=float(data_loss.detach()),
                                pde_loss=float(pde_loss.detach())))
    if polynomial:
        matrix, response = pde_matrix(field, terms, coefficients.signs.tolist(), bounds, z_r, t_r)
        coefficients.controls.copy_(project_controls(matrix, response).reshape(len(terms), 4))
    with torch.no_grad():
        if any(not torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Non-finite final model parameters")
        fitted = field(z, t)
        final_values = coefficients(fitted, z)
        if not torch.isfinite(fitted).all() or not torch.isfinite(final_values).all():
            raise RuntimeError("Non-finite final field or coefficients")
        rmse = float((fitted - observed).square().mean().sqrt())
    if tuple(selection.terms) != terms or tuple(coefficients.terms) != terms:
        raise RuntimeError("Frozen structure changed during inversion")
    metadata = dict(field_spec=field_spec, coefficient_spec=coefficients.spec, soil=soil,
        dtype=config.field.dtype, seed=config.seed, coefficient_seed=config.seed + 1000003,
        initial_field_sha256=initial_field_sha256, collocation_sha256=collocation_sha256,
        collocation_points=config.collocation.points, epochs=config.training.epochs,
        structure_fingerprint=selection.library_fingerprint)
    return InversionResult(status="completed", terms=terms, field=field, coefficients=coefficients,
        history=tuple(history), domain=dict(theta_bounds=list(bounds)), metrics=dict(theta_rmse=rmse), metadata=metadata)
