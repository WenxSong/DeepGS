import numpy as np
import pytest

from deepgs.physics import SOILS

THETA = np.linspace(0.12, 0.40, 15)


def numerical(fn, theta, eps=1e-6):
    return (fn(theta + eps) - fn(theta - eps)) / (2 * eps)


@pytest.mark.parametrize("name", sorted(SOILS))
def test_dK_matches_numerical_derivative(name):
    soil = SOILS[name]
    assert soil.dK(THETA) == pytest.approx(numerical(soil.K, THETA), rel=1e-5)


@pytest.mark.parametrize("name", sorted(SOILS))
def test_D_equals_K_times_retention_slope(name):
    soil = SOILS[name]

    def suction(theta):
        return (soil.Se(theta) ** (-1 / soil.m) - 1) ** (1 / soil.n) / soil.alpha

    assert soil.D(THETA) == pytest.approx(-soil.K(THETA) * numerical(suction, THETA), rel=1e-5)


@pytest.mark.parametrize("name", sorted(SOILS))
def test_dD_matches_numerical_derivative(name):
    soil = SOILS[name]
    assert soil.dD(THETA) == pytest.approx(numerical(soil.D, THETA, 1e-6), rel=1e-4)


def test_table_values_use_published_texture_labels():
    assert SOILS["sandy_loam"].Ks == 1.061
    assert SOILS["silt_loam"].Ks == 0.108


def test_reference_signs():
    coefficients = SOILS["loam"].coefficients(THETA)
    assert (coefficients["theta_z"] < 0).all()
    assert (coefficients["theta_zz"] > 0).all()
    assert (coefficients["theta_z^2"] > 0).all()
