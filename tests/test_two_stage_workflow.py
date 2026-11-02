"""Two-stage wiring on a deterministic library; not a scientific recovery trial."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

import deepgs
from deepgs.artifacts import load_model, save_inversion
from deepgs.contracts import LibraryArtifact, SelectionResult
from deepgs.data import Observations
from deepgs.experiment import load_experiment
from deepgs.library import CandidateLibrary


def fixture_data(tmp_path):
    times = np.linspace(0, 1, 20)
    depths = np.linspace(.05, .25, 7)
    theta = .24 + .02 * np.sin(times[:, None] * 4 + depths[None, :])
    path = tmp_path / "data.npz"
    np.savez(path, times=times, depths=depths, theta=theta)
    return path, Observations(times, depths, theta)


def fixture_library():
    rng = np.random.default_rng(8)
    n = 800
    theta = np.linspace(.2, .3, n)[:, None]
    terms = rng.normal(size=(n, 3))
    response = terms @ np.array([-.3, .5, .7])[:, None]
    raw = CandidateLibrary(theta, terms, response, ["theta_z", "theta_zz", "theta_z^2"],
                           np.repeat([.1, .2], n // 2)[:, None], np.tile(np.linspace(0, 1, n // 2), 2)[:, None])
    return LibraryArtifact.create(raw)


def config(path, selector="group", form="network"):
    return dict(schema_version=2, data=dict(path=str(path)), problem=dict(soil="heterogeneous" if selector == "spatial_group" else "homogeneous"),
        preprocess=dict(differentiation="fd"), structure=dict(selector=selector, grouping=dict(min_group_size=100),
        sgtr=dict(num_tols=20, epsilon=.01)), inversion=dict(enabled=True, coefficient_form=form,
        field=dict(hidden_layers=2, width=5, dtype="float64"), training=dict(epochs=2, log_every=1),
        collocation=dict(points=20), **({"polynomial":dict(signs={"theta_z":-1,"theta_zz":1,"theta_z^2":1})} if form == "polynomial" else {})),
        runtime=dict(device="cpu"))


CASES = [(s, f) for s in ("group", "spline") for f in ("network", "polynomial", "bspline")] + [("spatial_group", "network")]


@pytest.mark.parametrize("selector,form", CASES)
def test_all_seven_paths_invert_save_and_reload(tmp_path, monkeypatch, selector, form):
    import deepgs.workflow as workflow
    path, observations = fixture_data(tmp_path)
    item = fixture_library()
    monkeypatch.setattr(workflow, "build_library", lambda *args: item)
    # A deterministic path position makes this a wiring test, independent of Kneedle tuning.
    if selector == "spline":
        monkeypatch.setattr("deepgs.sparse.knee_complexity", lambda *args: 3)
    real_select = workflow.select
    calls = []
    def counted(*args):
        calls.append(1)
        return real_select(*args)
    monkeypatch.setattr(workflow, "select", counted)
    result = deepgs.run(config(path, selector, form), output=tmp_path / "run")
    assert result.execution_status == result.inversion.status == "completed"
    assert calls == [1] and result.selection.terms == result.inversion.terms
    assert result.selection.terms == ("theta_z", "theta_zz", "theta_z^2")
    loaded = load_model(tmp_path / "run/inversion/model.pt")
    with np.load(tmp_path / "run/inversion/fields.npz") as saved:
        z, t = np.meshgrid(observations.depths, observations.times)
        np.testing.assert_allclose(loaded.predict(z, t), saved["theta"], atol=1e-14)
    with np.load(tmp_path / "run/inversion/coefficients.npz") as saved:
        if "theta_grid" in saved:
            actual = loaded.evaluate_coefficients(saved["theta_grid"]).T
            np.testing.assert_allclose(actual, saved["curves"], atol=1e-14)
        else:
            assert str(saved["coordinate_kind"]) == "trajectory_pairs"
            actual = loaded.evaluate_coefficients(saved["theta"], saved["z"]).T
            np.testing.assert_allclose(actual, saved["coefficient_fields"], atol=1e-14)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["stages"]["inversion"] == "completed"
    assert manifest["stage_timings"]["inversion"]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("status,expected", [("empty", "skipped_empty"), ("no_knee", "skipped_no_structure")])
def test_empty_and_no_knee_are_explicit(tmp_path, monkeypatch, status, expected):
    path, _ = fixture_data(tmp_path)
    monkeypatch.setattr("deepgs.workflow.select", lambda *args: SelectionResult(status, (), (), None, (), "test"))
    result = deepgs.run(config(path), output=tmp_path / "run")
    assert result.inversion.status == expected
    assert not (tmp_path / "run/inversion/model.pt").exists()


def test_missing_prior_is_incomplete_and_keeps_selection(tmp_path, monkeypatch):
    path, _ = fixture_data(tmp_path)
    monkeypatch.setattr("deepgs.workflow.select", lambda *args: SelectionResult.supplied(["1", "theta_z"]))
    result = deepgs.run(config(path, form="polynomial"), output=tmp_path / "run")
    assert result.execution_status == "incomplete" and result.inversion.status == "blocked_missing_prior"
    assert (tmp_path / "run/selection/structure.json").exists()
    assert result.selection.terms == ("1", "theta_z")


def test_inversion_failure_preserves_selection_status(tmp_path, monkeypatch):
    path, _ = fixture_data(tmp_path)
    monkeypatch.setattr("deepgs.workflow.select", lambda *args: SelectionResult.supplied(["theta_z"]))
    def fail(*args):
        raise RuntimeError("inversion failure")
    monkeypatch.setattr("deepgs.workflow.invert", fail)
    with pytest.raises(RuntimeError, match="inversion failure"):
        deepgs.run(config(path), output=tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert manifest["stages"] == dict(selection="selected", inversion="failed")
    assert summary["selection"]["terms"] == ["theta_z"]
    assert summary["inversion"]["status"] == "failed"


def test_supplied_support_is_not_marked_as_discovery(tmp_path):
    _, observations = fixture_data(tmp_path)
    effective = load_experiment(config(tmp_path / "unused.npz"))
    selection = SelectionResult.supplied(["1"])
    result = deepgs.invert(observations, selection, effective.inversion)
    save_inversion(tmp_path, result, observations, selection)
    loaded = deepgs.load_model(tmp_path / "inversion/model.pt")
    assert loaded.structure_source == "supplied"
    with pytest.raises(ValueError, match="domain"):
        loaded.evaluate_coefficients([4.])


@pytest.mark.parametrize("form", ["network", "polynomial", "bspline"])
def test_loaded_float32_predictions_are_valid_coefficient_queries(tmp_path, form):
    _, observations = fixture_data(tmp_path)
    raw = config(tmp_path / "unused.npz", form=form)
    raw["inversion"]["field"]["dtype"] = "float32"
    raw["inversion"]["domain"] = dict(theta_bounds=[.1, .4])
    effective = load_experiment(raw)
    selection = SelectionResult.supplied(["theta_zz"])
    result = deepgs.invert(observations, selection, effective.inversion)
    with torch.no_grad():
        result.field.output.weight.zero_()
        result.field.output.bias.fill_(100)
    save_inversion(tmp_path, result, observations, selection)
    loaded = deepgs.load_model(tmp_path / "inversion/model.pt")
    theta = loaded.predict([.1], [.5])
    assert np.isfinite(loaded.evaluate_coefficients(theta)).all()
