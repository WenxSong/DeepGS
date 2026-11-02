"""Short runs through every pipeline. These check wiring and outputs, not scientific accuracy."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from deepgs.config import DataConfig, DeepGSConfig, ExtendedDeepGSConfig, FieldConfig, GroupConfig, GSRConfig, InversionSettings, LibraryConfig, SGTRConfig, SplineSRConfig, load_config
from deepgs.pipelines import polynomial_curve, run_deepgs, run_extended_deepgs, run_gsr, run_spline_sr

from conftest import ROOT, TH_TXT, TRUE_TERMS

SMALL_FIELD = dict(hidden_layers=2, width=10, iterations=5)
SENSORS = DataConfig(path=str(TH_TXT), depths=[0.05, 0.25, 0.45, 0.65])


@pytest.fixture(autouse=True)
def optional_profile(request):
    if request.node.name != "test_unknown_option_is_rejected" and not TH_TXT.exists():
        pytest.skip("This historical integration check requires the optional local research profile")


def test_gsr_finite_difference_end_to_end(tmp_path):
    config = GSRConfig(data=DataConfig(path=str(TH_TXT), depths=[0.10, 0.12, 0.14, 0.16, 0.18]), differentiation="fd")
    result = run_gsr(config, tmp_path)
    assert set(result.names) == TRUE_TERMS
    assert (tmp_path / "structure.json").exists()
    with np.load(tmp_path / "coefficient_curves.npz") as curves:
        assert "theta_zz/coefficient" in curves.files


def test_gsr_dnn_averages_seeds(tmp_path):
    config = GSRConfig(data=SENSORS, seeds=[1, 2], field=FieldConfig(**SMALL_FIELD),
                       library=LibraryConfig(depth_count=5, time_count=300), group=GroupConfig(min_group_size=20), sgtr=SGTRConfig(num_tols=10))
    assert run_gsr(config, tmp_path, "cpu").coefficients.shape[0] == 10


def test_deepgs_alternates_and_saves(tmp_path):
    config = DeepGSConfig(data=SENSORS, field=FieldConfig(**SMALL_FIELD), coefficient_hidden=[5],
                          group=GroupConfig(min_group_size=50), sgtr=SGTRConfig(num_tols=10, lam=1e-2),
                          collocation_points=50, sparse_points=2000, ado_rounds=1, ado_steps=3, post_steps=2, log_every=1)
    result = run_deepgs(config, tmp_path, "cpu")
    assert len(result["selections"]) == 2
    assert result["curves"].shape[0] == len(result["terms"])
    assert (tmp_path / "model.pt").exists()


def test_extended_deepgs_saves_depth_dependent_fields(tmp_path):
    config = ExtendedDeepGSConfig(data=SENSORS, field=FieldConfig(activation="sin", normalize_inputs=False, **SMALL_FIELD),
                                  library=LibraryConfig(depth_count=3, time_count=300), group=GroupConfig(min_group_size=20, normalization="group", by_depth=True),
                                  sgtr=SGTRConfig(num_tols=10, lam=1e-2, epsilon=10**1.2, corrected=False),
                                  theta_hidden=[5, 5], depth_hidden=[3, 4], collocation_points=30, train_steps=2, log_every=1)
    result = run_extended_deepgs(config, tmp_path, "cpu")
    assert result["coefficient_fields"].shape == (len(result["terms"]), 300, 3)
    assert result["structure"].depth_centers is not None


def spline_config(**inversion):
    return SplineSRConfig(data=SENSORS, field=FieldConfig(activation="silu", normalize_inputs=False, **SMALL_FIELD),
                          library=LibraryConfig(max_order=4, depth_count=5, time_count=100), group=GroupConfig(min_group_size=20, merge_tail=0.0),
                          prior={"theta_zzzz": 1},
                          inversion=InversionSettings(hidden_sizes=(10,), coefficient_hidden=(5,), epochs=1, collocation_points=20, **inversion))


def test_spline_sr_polynomial_skips_inversion_when_structure_differs(tmp_path):
    result = run_spline_sr(spline_config(coefficient_form="polynomial"), tmp_path, "cpu")
    assert result["inversion"] is None
    assert (tmp_path / "summary.json").exists()


def test_spline_sr_network_inverts_any_selected_structure(tmp_path):
    result = run_spline_sr(spline_config(), tmp_path, "cpu")
    assert result["inversion"] is not None
    assert result["inversion"]["curves"].shape[0] == len(result["structure"].names)
    with np.load(tmp_path / "inversion.npz") as saved:
        assert "controls" not in saved.files
    assert (tmp_path / "model.pt").exists()


def test_polynomial_curve_tracks_smooth_function():
    theta = np.linspace(0.1, 0.4, 100)
    grid, fitted = polynomial_curve(theta, theta**2, degree=3)
    assert fitted == pytest.approx(grid**2, abs=1e-3)


def test_spline_empty_support_is_saved_without_inversion(tmp_path, monkeypatch):
    from deepgs.sparse import StructureResult
    empty = StructureResult([], [], np.array([0.2, 0.3]), None, np.zeros((15, 2)))
    monkeypatch.setattr("deepgs.pipelines.select_spline", lambda *args, **kwargs: empty)
    result = run_spline_sr(spline_config(), tmp_path, "cpu")
    assert result["inversion"] is None
    assert result["status"] == "empty structure selected; inversion skipped"
    assert (tmp_path / "structure.json").exists()


@pytest.mark.parametrize("path", [p for p in sorted((ROOT / "configs").glob("*.yaml"))
                                  if "pipeline" in yaml.safe_load(p.read_text(encoding="utf-8"))], ids=lambda p: p.stem)
def test_shipped_configs_load(path):
    name, config = load_config(path)
    expected = "intermediate/data/loam_all_0.2/th.txt" if name == "extended_deepgs" else "intermediate/data/th.txt"
    assert config.data.path == expected, name
    assert Path(ROOT / config.data.path).exists(), name


def test_unknown_option_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("pipeline: gsr\nsgtr:\n  thresholds: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds"):
        load_config(path)
