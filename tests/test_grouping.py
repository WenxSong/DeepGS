import numpy as np
import pytest

from deepgs.grouping import build_system, merge_tails, theta_windows
from deepgs.library import CandidateLibrary


def make_library(n=3000, seed=0, depth=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.1, 0.3, n)[:, None]
    terms = rng.normal(size=(n, 2)) * [1.0, 5.0]
    theta_t = 2 * terms[:, :1] - terms[:, 1:] + 0.5
    return CandidateLibrary(theta, terms, theta_t, ["theta_z", "theta_zz"], depth)


def test_windows_cover_each_sample_once():
    theta = np.sort(np.random.default_rng(0).uniform(0.1, 0.3, 5000))
    windows = theta_windows(theta, 0.001, 20)
    assert np.array_equal(np.concatenate(windows), np.arange(theta.size))
    assert all(window.size >= 20 for window in windows)


def test_merge_tails_keeps_all_samples():
    windows = [np.arange(i * 5, (i + 1) * 5) for i in range(200)]
    merged = merge_tails(windows, 0.01)
    assert len(merged) == 198
    assert np.array_equal(np.concatenate(merged), np.arange(1000))
    assert len(merge_tails(windows[:2], 0.01)) == 2


@pytest.mark.parametrize("normalization", ["global", "group"])
def test_scales_recover_physical_coefficients(normalization):
    system = build_system(make_library(), 0.005, 50, normalization)
    # The true physical coefficients in normalized units are x_scale / y_scale times the values.
    truth = np.array([0.5, 2.0, -1.0])
    W = truth[:, None] * system.x_scale.T / system.y_scale
    assert system.to_physical(W) == pytest.approx(np.tile(truth[:, None], system.n_groups))
    assert system.n_samples == 3000


def test_group_normalization_gives_each_group_unit_norms():
    system = build_system(make_library(), 0.005, 50, "group")
    for X, y in zip(system.X, system.y):
        assert np.linalg.norm(X, axis=0) == pytest.approx(np.ones(3))
        assert np.linalg.norm(y) == pytest.approx(1.0)


def test_grouping_by_depth_separates_depths():
    depth = np.repeat([0.1, 0.2, 0.3], 1000)[:, None]
    system = build_system(make_library(depth=depth), 0.02, 50, "group", by_depth=True)
    assert set(np.round(system.depth_centers, 6)) == {0.1, 0.2, 0.3}
    assert system.n_samples == 3000


def test_by_depth_needs_depth_values():
    with pytest.raises(ValueError):
        build_system(make_library(), by_depth=True)
