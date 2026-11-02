"""Structure recovery on the loam benchmark, whose parameters are known."""
import numpy as np
import pytest

from deepgs.grouping import build_system
from deepgs.library import CandidateLibrary, finite_difference_library
from deepgs.physics import SOILS
from deepgs.sparse import aic_score, select_sgtr, select_spline, spline_curve

from conftest import TRUE_TERMS

LOAM = SOILS["loam"]


def relative_error(estimate, truth):
    return np.linalg.norm(estimate - truth) / np.linalg.norm(truth)


def subset(library, depths, every=1):
    rows = np.flatnonzero(np.isin(np.round(library.depth[:, 0], 6), np.round(depths, 6)))[::every]
    return CandidateLibrary(library.theta[rows], library.terms[rows], library.theta_t[rows], library.names, library.depth[rows])


def test_aic_behaves_like_linear_penalty():
    epsilon = 1e-2
    gain = aic_score(0.10, 30000, 3, epsilon) - aic_score(0.08, 30000, 3, epsilon)
    assert gain == pytest.approx(0.02 / epsilon, rel=1e-2)


def test_sgtr_recovers_richards_terms(loam):
    observed = loam.select_depths([0.10, 0.12, 0.14, 0.16, 0.18])
    library = finite_difference_library(observed.theta.T, observed.depths, observed.times)
    system = build_system(library, 0.001, 200, "global", merge_tail=0.01)
    result = select_sgtr(system, num_tols=200, lam=1e-4, epsilon=1e-2, corrected=True)
    assert set(result.names) == TRUE_TERMS

    centers = result.theta_centers
    index = {name: feature for feature, name in zip(result.active, result.names)}
    for name, truth in LOAM.coefficients(centers).items():
        assert relative_error(result.coefficients[index[name]], truth) < 0.5
    assert relative_error(result.coefficients[index["theta_z"]], -LOAM.dK(centers)) < 0.1
    assert relative_error(result.coefficients[index["theta_zz"]], LOAM.D(centers)) < 0.15


def test_coarse_grained_regression_recovers_richards_terms(loam):
    library = finite_difference_library(loam.theta.T, loam.depths, loam.times)
    depths = loam.depths[np.round(np.linspace(5, 25, 10)).astype(int)]
    system = build_system(subset(library, depths), 0.001, 100, "group", by_depth=True, merge_tail=0.01)
    result = select_sgtr(system, num_tols=100, lam=1e-2, epsilon=10**1.2)
    assert set(result.names) == TRUE_TERMS
    assert result.depth_centers is not None


def test_group_scales_are_not_mixed_between_groups(loam):
    """Each group must convert back with its own scale, or physical coefficients are wrong."""
    library = finite_difference_library(loam.theta.T, loam.depths, loam.times)
    depths = loam.depths[np.round(np.linspace(5, 25, 10)).astype(int)]
    system = build_system(subset(library, depths), 0.001, 100, "group", by_depth=True, merge_tail=0.01)
    result = select_sgtr(system, num_tols=100, lam=1e-2, epsilon=10**1.2)
    feature = result.active[result.names.index("theta_z")]
    error = relative_error(result.coefficients[feature], -LOAM.dK(result.theta_centers))
    assert error < 0.2


def test_spline_selection_and_refit(loam):
    library = finite_difference_library(loam.theta.T, loam.depths, loam.times, max_order=4)
    system = build_system(subset(library, loam.depths[5:26:5], every=5), 0.001, 100, "global")
    result = select_spline(system, n_knots=6, lam=1e-2)
    assert set(result.names) == TRUE_TERMS

    refit = select_spline(system, n_knots=6, lam=1e-2, refit_lam=1e-6)
    assert set(refit.names) == TRUE_TERMS
    centers = system.theta_centers
    feature = refit.active[refit.names.index("theta_z")]
    shrunk = relative_error(result.coefficients[feature], -LOAM.dK(centers))
    unbiased = relative_error(refit.coefficients[feature], -LOAM.dK(centers))
    assert unbiased < shrunk
    assert spline_curve(refit, feature, centers) == pytest.approx(refit.coefficients[feature])


def test_spline_rejects_incompatible_local_scaling(loam):
    library = finite_difference_library(loam.theta.T, loam.depths, loam.times, max_order=4)
    system = build_system(subset(library, loam.depths[5:26:5], every=5), normalization="group")
    with pytest.raises(ValueError, match="global normalization"):
        select_spline(system)
