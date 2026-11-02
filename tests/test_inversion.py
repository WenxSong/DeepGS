import numpy as np
import pytest
import torch

from deepgs.data import Observations
from deepgs.inversion import (InversionSettings, cubic_curves, project_controls, run_inversion, sample_collocation,
                              theta_bounds)


def small_observations():
    times = np.linspace(0, 1, 60)
    depths = np.array([0.05, 0.25, 0.45, 0.65])
    theta = 0.2 + 0.05 * np.sin(4 * times)[:, None] * np.exp(-depths)[None, :]
    return Observations(times, depths, theta)


def test_projection_recovers_nonnegative_controls():
    rng = np.random.default_rng(0)
    matrix = torch.as_tensor(rng.normal(size=(200, 4)))
    truth = torch.tensor([0.5, 0.0, 2.0, 1.0], dtype=matrix.dtype)
    controls = project_controls(matrix, matrix @ truth)
    assert controls.numpy() == pytest.approx(truth.numpy(), abs=1e-8)


def test_projection_keeps_controls_nonnegative():
    rng = np.random.default_rng(1)
    matrix = torch.as_tensor(rng.normal(size=(200, 4)))
    truth = torch.tensor([0.5, -3.0, 2.0, 1.0], dtype=matrix.dtype)
    assert (project_controls(matrix, matrix @ truth).numpy() >= 0).all()


def test_curves_follow_sign_and_are_monotonic():
    controls = np.array([[1.0, 0.5, 0.0, 0.2], [0.1, 0.0, 2.0, 0.0]])
    curves = cubic_curves(controls, [-1, 1], (0.1, 0.3), np.linspace(0.1, 0.3, 20))
    assert (curves[0] < 0).all() and (np.diff(curves[0]) < 0).all()
    assert (curves[1] > 0).all() and (np.diff(curves[1]) >= 0).all()


def test_theta_bounds_respects_envelope():
    observations = small_observations()
    low, high = theta_bounds(observations)
    assert low < observations.theta.min() and high > observations.theta.max()
    assert theta_bounds(observations, (0.19, 0.205))[0] > 0.19


def test_collocation_stays_in_domain():
    observations = small_observations()
    points = sample_collocation(observations, InversionSettings(collocation_points=100), 0)
    assert points.shape == (100, 2)
    assert points[:, 0].min() >= 0.05 and points[:, 0].max() <= 0.65
    assert points[:, 1].min() >= 0 and points[:, 1].max() <= 1


def test_polynomial_inversion_runs_and_keeps_constraints():
    settings = InversionSettings(coefficient_form="polynomial", hidden_sizes=(10, 10), epochs=5, collocation_points=50, log_every=5)
    result = run_inversion(small_observations(), ["theta_z", "theta_zz", "theta_z^2"], [-1, 1, 1],
                           torch.device("cpu"), settings)
    assert result["controls"].shape == (3, 4)
    assert (result["controls"] >= 0).all()
    assert np.isfinite(result["theta_rmse"])


def test_polynomial_inversion_needs_signs():
    with pytest.raises(ValueError, match="sign"):
        run_inversion(small_observations(), ["theta_z"], None, torch.device("cpu"), InversionSettings(coefficient_form="polynomial"))


def test_network_is_the_default_and_accepts_a_constant_term():
    settings = InversionSettings(hidden_sizes=(10, 10), coefficient_hidden=(5,), epochs=3, collocation_points=50, log_every=3)
    result = run_inversion(small_observations(), ["1", "theta_z", "theta_zz"], None, torch.device("cpu"), settings)
    assert result["controls"] is None
    assert result["curves"].shape == (3, 200)
    assert np.isfinite(result["theta_rmse"])


def test_unknown_coefficient_form_is_rejected():
    with pytest.raises(ValueError, match="coefficient form"):
        run_inversion(small_observations(), ["theta_z"], None, torch.device("cpu"), InversionSettings(coefficient_form="spline"))
