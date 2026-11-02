"""New selector adapters must preserve numerical kernels and failure diagnostics."""
from dataclasses import dataclass, field, asdict

import numpy as np
import pytest

from deepgs.contracts import LibraryArtifact
from deepgs.config import SGTRConfig
from deepgs.grouping import build_system
from deepgs.library import CandidateLibrary
from deepgs.selection import select
from deepgs.sparse import select_sgtr, select_spline


@dataclass
class Grouping:
    win_size: float = .01
    min_group_size: int = 20
    normalization: str = "global"
    merge_tail: float = 0.


@dataclass
class Spline:
    degree: int = 3
    n_internal_knots: int = 2
    lam: float = .01
    refit_lam: float | None = None


@dataclass
class Config:
    selector: str = "group"
    grouping: Grouping = field(default_factory=Grouping)
    sgtr: SGTRConfig = field(default_factory=lambda: SGTRConfig(num_tols=12))
    spline: Spline = field(default_factory=Spline)


def artifact():
    rng = np.random.default_rng(4)
    theta = np.tile(np.linspace(.15, .35, 200), 4)[:, None]
    z = np.repeat([.05, .10, .15, .20], 200)[:, None]
    t = np.tile(np.arange(200.), 4)[:, None]
    x = rng.normal(size=(800, 2))
    y = ((1 + theta[:, 0]) * x[:, 0] + .4 * x[:, 1])[:, None]
    return LibraryArtifact.create(CandidateLibrary(theta, x, y, ["theta_z", "theta_zz"], z, t))


@pytest.mark.parametrize("selector,soil", [("group", "homogeneous"), ("spatial_group", "heterogeneous")])
def test_group_adapters_are_numerically_identical(selector, soil):
    item = artifact()
    cfg = Config(selector=selector)
    if selector == "spatial_group":
        cfg.grouping.normalization = "group"
    old = select_sgtr(build_system(item.library, by_depth=selector == "spatial_group", **asdict(cfg.grouping)),
                      **asdict(cfg.sgtr))
    result = select(item, cfg, soil)
    assert result.terms == tuple(old.names)
    assert result.term_indices == tuple(old.active)
    assert result.diagnostics == tuple(old.path)
    assert result.library_fingerprint == item.fingerprint
    np.testing.assert_array_equal(result.coarse.coefficients, old.coefficients)
    assert result.status == ("selected" if old.names else "empty")


def test_spline_adapter_preserves_path_coefficients_and_term_order(monkeypatch):
    item, cfg = artifact(), Config(selector="spline")
    monkeypatch.setattr("deepgs.sparse.knee_complexity", lambda ks, errors: 2)
    old = select_spline(build_system(item.library, **asdict(cfg.grouping)), n_knots=2, degree=3, lam=.01)
    result = select(item, cfg)
    assert result.terms == tuple(old.names)
    assert result.term_indices == tuple(old.active)
    assert result.diagnostics == tuple(old.path)
    np.testing.assert_array_equal(result.coarse.coefficients, old.coefficients)
    np.testing.assert_array_equal(result.coarse.spline["control_points"], old.spline["control_points"])


def test_no_knee_retains_complete_elimination_diagnostics(monkeypatch):
    monkeypatch.setattr("deepgs.sparse.knee_complexity", lambda ks, errors: None)
    result = select(artifact(), Config(selector="spline"))
    assert result.status == "no_knee" and result.terms == () and result.coarse is None
    assert [step["k"] for step in result.diagnostics] == [0, 1, 2, 3]


def test_empty_support_is_a_distinct_selected_model(monkeypatch):
    from deepgs.sparse import StructureResult
    monkeypatch.setattr("deepgs.selection.select_sgtr", lambda system, **kwargs: StructureResult(
        [], [], system.theta_centers, system.depth_centers, np.zeros((system.n_features, system.n_groups))))
    result = select(artifact(), Config())
    assert result.status == "empty" and result.terms == () and result.coarse is not None


@pytest.mark.parametrize("selector,soil", [("group", "heterogeneous"), ("spatial_group", "homogeneous"),
                                           ("spline", "heterogeneous"), ("missing", "homogeneous")])
def test_selection_boundary_rejects_invalid_combinations(selector, soil):
    with pytest.raises(ValueError):
        select(artifact(), Config(selector=selector), soil)


def test_new_selector_rejects_ambiguous_term_ids():
    item = artifact()
    item.library.names[1] = item.library.names[0]
    with pytest.raises(ValueError):
        select(item, Config())


def test_new_selector_rejects_reassigned_arrays_after_fingerprinting():
    item = artifact()
    item.library.terms = item.library.terms + .1
    with pytest.raises(ValueError, match="changed after fingerprinting"):
        select(item, Config())
