"""Numerical and gradient checks for the new coefficient representation."""
import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from deepgs.bspline import BSplineCoefficient


@pytest.mark.parametrize("n_internal_knots", [0, 1, 6])
def test_basis_and_first_derivative_match_scipy_including_knots_and_endpoints(n_internal_knots):
    model = BSplineCoefficient((.1, .4), 2, n_internal_knots)
    knots = model.knots.cpu().numpy()
    points = np.unique(np.r_[0., 1., np.linspace(0, 1, 21), knots,
                             np.maximum(knots - 1e-9, 0), np.minimum(knots + 1e-9, 1)])
    theta = torch.tensor((.1 + .3 * points)[:, None], dtype=torch.float64, requires_grad=True)
    expected_q = (theta.detach().numpy()[:, 0] - .1) / (.4 - .1)
    reference = BSpline(knots, np.eye(model.n_basis), 3)
    basis = model.basis(theta)
    np.testing.assert_allclose(basis.detach().numpy(), reference(expected_q), atol=1e-12, rtol=1e-10)
    derivative = torch.stack([torch.autograd.grad(basis[:, i].sum(), theta, retain_graph=True)[0][:, 0]
                              for i in range(model.n_basis)], dim=1)
    np.testing.assert_allclose(derivative.detach().numpy(), reference(expected_q, nu=1) / .3,
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(basis.detach().sum(1).numpy(), 1., atol=1e-14)
    assert (basis >= -1e-14).all()


def test_endpoint_coefficient_gradient_remains_connected():
    model = BSplineCoefficient((0., 1.), 1, 2)
    with torch.no_grad():
        model.controls.copy_(torch.arange(model.n_basis, dtype=torch.float64).reshape(1, -1))
    theta = torch.tensor([[0.], [1.]], dtype=torch.float64, requires_grad=True)
    values = model(theta)
    derivative = torch.autograd.grad(values.sum(), theta)[0]
    reference = BSpline(model.knots.numpy(), model.controls.detach().numpy()[0], 3)
    np.testing.assert_allclose(values.detach().numpy()[:, 0], reference([0., 1.]), atol=1e-12)
    np.testing.assert_allclose(derivative.detach().numpy()[:, 0], reference([0., 1.], nu=1), atol=1e-12)
    assert (derivative.abs() > 0).all()


def test_autograd_gradcheck_for_theta_and_controls():
    model = BSplineCoefficient((.1, .4), 2, 3)
    theta = torch.tensor([[.123], [.217], [.386]], dtype=torch.float64, requires_grad=True)
    controls = model.controls.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x, c: model.basis(x) @ c.T, (theta, controls),
                                    eps=1e-6, atol=1e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(lambda x, c: model.basis(x) @ c.T, (theta, controls),
                                        eps=1e-6, atol=1e-6, rtol=1e-4)


def test_fixed_field_fit_recovers_known_representable_curves():
    model = BSplineCoefficient((.1, .4), 3, 6)
    theta = torch.linspace(.1, .4, 160, dtype=torch.float64)[:, None]
    q = (theta[:, 0] - .1) / .3
    truth = torch.stack((1 + q, q.square(), -.3 + q.pow(3)), dim=1)
    basis = model.basis(theta)
    fitted = torch.linalg.lstsq(basis, truth).solution
    with torch.no_grad():
        model.controls.copy_(fitted.T)
    np.testing.assert_allclose(model(theta).detach().numpy(), truth.numpy(), atol=1e-12)
    held_theta = torch.linspace(.1005, .3995, 121, dtype=torch.float64)[:, None]
    held_q = (held_theta[:, 0] - .1) / .3
    held_truth = torch.stack((1 + held_q, held_q.square(), -.3 + held_q.pow(3)), dim=1)
    np.testing.assert_allclose(model(held_theta).detach().numpy(), held_truth.numpy(), atol=1e-12)


def test_coefficient_controls_and_field_inputs_receive_gradients():
    model = BSplineCoefficient((.1, .4), 2)
    field_parameter = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
    theta = field_parameter.expand(8, 1)
    model(theta).square().sum().backward()
    assert model.controls.grad is not None and torch.isfinite(model.controls.grad).all()
    assert model.controls.grad.abs().sum() > 0
    assert field_parameter.grad is not None and torch.isfinite(field_parameter.grad)
    assert field_parameter.grad.abs() > 0


def test_state_dict_roundtrip_preserves_values_and_buffers(tmp_path):
    first = BSplineCoefficient((.1, .4), 2, 3)
    destination = tmp_path / "bspline.pt"
    torch.save(first.state_dict(), destination)
    second = BSplineCoefficient((.1, .4), 2, 3)
    second.load_state_dict(torch.load(destination, weights_only=True))
    theta = torch.linspace(.1, .4, 20, dtype=torch.float64)
    torch.testing.assert_close(first(theta), second(theta), rtol=0, atol=0)
    assert not second.knots.requires_grad and not second.theta_bounds.requires_grad


@pytest.mark.parametrize("theta", [torch.tensor([.09]), torch.tensor([.41]), torch.tensor([float("nan")]),
                                  torch.ones(2, 2)])
def test_invalid_queries_are_rejected(theta):
    with pytest.raises(ValueError):
        BSplineCoefficient((.1, .4), 1)(theta)


@pytest.mark.parametrize("kwargs", [dict(theta_bounds=(.2, .1), n_terms=1),
                                    dict(theta_bounds=(.1, float("inf")), n_terms=1),
                                    dict(theta_bounds=(.1, .4), n_terms=0),
                                    dict(theta_bounds=(.1, .4), n_terms=1, n_internal_knots=-1),
                                    dict(theta_bounds=(.1, .4), n_terms=1, n_internal_knots=True)])
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BSplineCoefficient(**kwargs)


def test_float32_conversion_is_finite_and_partition_preserving():
    model = BSplineCoefficient((.1, .4), 2).float()
    theta = torch.linspace(model.theta_bounds[0], model.theta_bounds[1], 20, requires_grad=True)
    values = model(theta)
    torch.testing.assert_close(model.basis(theta).sum(1), torch.ones(20), rtol=1e-6, atol=1e-6)
    values.sum().backward()
    assert torch.isfinite(theta.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_values_and_theta_gradients():
    cpu = BSplineCoefficient((.1, .4), 2)
    gpu = BSplineCoefficient((.1, .4), 2).cuda()
    gpu.load_state_dict(cpu.state_dict())
    theta_cpu = torch.tensor([[.1], [.17], [.3], [.4]], dtype=torch.float64, requires_grad=True)
    theta_gpu = theta_cpu.detach().cuda().requires_grad_(True)
    a, b = cpu(theta_cpu), gpu(theta_gpu)
    torch.testing.assert_close(a, b.cpu(), rtol=1e-10, atol=1e-12)
    a.sum().backward()
    b.sum().backward()
    torch.testing.assert_close(theta_cpu.grad, theta_gpu.grad.cpu(), rtol=1e-9, atol=1e-10)
