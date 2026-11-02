"""V2 bounded fields and coefficient domains share the same dtype endpoints."""
import pytest
import torch

from deepgs.coefficients import build_coefficient_model
from deepgs.experiment import InversionConfig, PolynomialConfig
from deepgs.networks import FieldNetwork


@pytest.mark.parametrize("bounds", [(.26833593849423576, .4913317399105574), (.1, .4)])
@pytest.mark.parametrize("form", ["network", "polynomial", "bspline"])
def test_float32_saturated_fields_stay_in_coefficient_domain(bounds, form):
    field = FieldNetwork([4], output_bounds=bounds, bounded_lerp=True).float()
    config = InversionConfig(coefficient_form=form)
    if form == "polynomial":
        config.polynomial = PolynomialConfig(signs={"1": 1})
    coefficients = build_coefficient_model(("1",), "homogeneous", config, bounds).float()
    z, t = torch.zeros(3, 1), torch.zeros(3, 1)
    for logit, expected in ((-100., bounds[0]), (100., bounds[1])):
        with torch.no_grad():
            field.output.weight.zero_()
            field.output.bias.fill_(logit)
        theta = field(z, t)
        torch.testing.assert_close(theta, torch.full_like(theta, expected), rtol=0, atol=0)
        assert torch.isfinite(coefficients(theta)).all()


@pytest.mark.parametrize("form", ["network", "polynomial", "bspline"])
def test_bounded_lerp_keeps_coefficient_to_field_gradients_finite(form):
    bounds = (.26833593849423576, .4913317399105574)
    field = FieldNetwork([4], output_bounds=bounds, bounded_lerp=True).float()
    config = InversionConfig(coefficient_form=form)
    if form == "polynomial":
        config.polynomial = PolynomialConfig(signs={"1": 1})
    coefficients = build_coefficient_model(("1",), "homogeneous", config, bounds).float()
    z, t = torch.zeros(3, 1), torch.zeros(3, 1)
    theta = field(z, t)
    (theta.square().mean() + coefficients(theta).square().mean()).backward()
    gradients = [p.grad for p in field.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


def test_legacy_output_mapping_remains_the_default():
    field = FieldNetwork([4], output_bounds=(.1, .4))
    assert field.bounded_lerp is False
