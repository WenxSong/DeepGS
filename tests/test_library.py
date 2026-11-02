import numpy as np
import pytest
import torch

from deepgs.library import (CandidateLibrary, average_libraries, finite_difference_library, library_at_points,
                            required_order, term_names)


def test_term_order_matches_published_libraries():
    assert term_names(3) == ["theta_z", "theta_zz", "theta_zzz", "theta_z^2", "theta_zz^2", "theta_zzz^2",
                             "theta_z*theta_zz", "theta_z*theta_zzz", "theta_zz*theta_zzz"]
    assert len(term_names(4)) == 14
    # Spline-SR feature index 5 (after the constant) must be theta_z^2.
    assert term_names(4)[4] == "theta_z^2"


def test_required_order():
    assert required_order(["1"]) == 0
    assert required_order(["theta_z", "theta_z^2"]) == 1
    assert required_order(["1", "theta_z*theta_zzz"]) == 3


def test_finite_differences_on_polynomial_field():
    depths = np.linspace(0, 1, 11)
    times = np.linspace(0, 1, 5)
    z, t = np.meshgrid(depths, times, indexing="ij")
    theta = z**3 + 2 * t * z
    library = finite_difference_library(theta, depths, times, max_order=3)
    zi, ti = np.meshgrid(depths[2:-2], times[1:], indexing="ij")
    assert library.theta[:, 0] == pytest.approx((zi**3 + 2 * ti * zi).ravel())
    assert library.theta_t[:, 0] == pytest.approx((2 * zi).ravel())
    assert library.terms[:, 0] == pytest.approx((3 * zi**2 + 0.01 + 2 * ti).ravel(), abs=1e-9)
    assert library.terms[:, 1] == pytest.approx((6 * zi).ravel(), abs=1e-9)
    assert library.terms[:, 2] == pytest.approx(np.full(zi.size, 6.0), abs=1e-7)
    assert library.depth[:, 0] == pytest.approx(zi.ravel())


class Cubic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, z, t):
        return self.scale * z**3 + 2 * t * z


def test_autodiff_library_on_polynomial_field():
    z = np.array([[0.5], [1.0], [2.0]])
    t = np.array([[0.0], [1.0], [3.0]])
    library = library_at_points(Cubic(), z, t, max_order=4)
    assert library.theta_t[:, 0] == pytest.approx(2 * z[:, 0])
    assert library.terms[:, 0] == pytest.approx(3 * z[:, 0] ** 2 + 2 * t[:, 0])
    assert library.terms[:, 1] == pytest.approx(6 * z[:, 0])
    assert library.terms[:, 2] == pytest.approx(np.full(3, 6.0))
    assert library.terms[:, 3] == pytest.approx(np.zeros(3))
    assert library.terms[:, 4] == pytest.approx(library.terms[:, 0] ** 2)
    assert library.feature_names[0] == "1"


def test_non_finite_library_is_rejected():
    with pytest.raises(ValueError):
        CandidateLibrary(np.array([[np.nan]]), np.zeros((1, 1)), np.zeros((1, 1)), ["theta_z"])


def test_average_libraries():
    a = CandidateLibrary(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)), ["theta_z"])
    b = CandidateLibrary(3 * np.ones((2, 1)), 3 * np.ones((2, 1)), 3 * np.ones((2, 1)), ["theta_z"])
    assert average_libraries([a, b]).terms == pytest.approx(2 * np.ones((2, 1)))
