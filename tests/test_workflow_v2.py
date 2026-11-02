"""Selection-only integration checks, including units and row provenance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from deepgs import run
from deepgs.artifacts import load_library
from deepgs.cli import main
from deepgs.experiment import load_experiment
from deepgs.workflow import build_library, load_data


def profile(path, time_scale=1, depth_scale=1, theta_scale=1):
    times = np.linspace(0, 1, 160)
    depths = np.linspace(.05, .25, 9)
    theta = .24 + .03 * np.sin(4 * times[:, None] + depths[None, :] * 7)
    np.savez(path, times=times * time_scale, depths=depths * depth_scale, theta=theta * theta_scale)


def configuration(path, selector="group"):
    return dict(schema_version=2, data=dict(path=str(path)), preprocess=dict(differentiation="fd"),
                problem=dict(soil="heterogeneous" if selector == "spatial_group" else "homogeneous"),
                structure=dict(selector=selector, grouping=dict(min_group_size=20), sgtr=dict(num_tols=12)),
                inversion=dict(enabled=False), runtime=dict(device="cpu", save_library=True))


def test_unit_conversion_precedes_differentiation_and_noise(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    profile(a)
    profile(b, time_scale=24, depth_scale=100, theta_scale=100)
    cfg1, cfg2 = configuration(a), configuration(b)
    cfg2["data"]["units"] = dict(time="hour", depth="cm", water_content="percent")
    import torch
    conf1, conf2 = load_experiment(cfg1), load_experiment(cfg2)
    raw1, raw2 = load_data(conf1), load_data(conf2)
    np.testing.assert_allclose(raw1.theta, raw2.theta, atol=1e-15)
    lib1, lib2 = [build_library(c, obs, torch.device("cpu")) for c, obs in ((conf1, raw1), (conf2, raw2))]
    np.testing.assert_allclose(lib1.library.terms, lib2.library.terms, rtol=1e-8, atol=1e-9)
    for cfg in (cfg1, cfg2):
        cfg["data"].update(noise=.1, noise_seed=7)
    np.testing.assert_allclose(load_data(load_experiment(cfg1)).theta, load_data(load_experiment(cfg2)).theta, atol=1e-15)


@pytest.mark.parametrize("selector", ["group", "spline", "spatial_group"])
def test_cli_three_selectors_and_verified_cache(tmp_path, selector):
    data = tmp_path / "input.npz"
    profile(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(configuration(data, selector)))
    out = tmp_path / "run"
    main(["run", str(path), "--output", str(out), "--device", "cpu"])
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    artifact = load_library(out / "selection/library.npz")
    assert manifest["workflow"] == "two_stage" and manifest["status"] == "completed"
    assert artifact.fingerprint == manifest["library_fingerprint"]
    assert summary["inversion"]["status"] == "not_requested"
    assert not (out / "inversion").exists()
    assert summary["selection"]["status"] in ("selected", "empty", "no_knee")
    assert manifest["inputs"][0]["sha256"] and manifest["source"]
    assert manifest["units"] == dict(time="day", depth="m", water_content="fraction")
    coarse = out / "selection/coarse_coefficients.npz"
    if coarse.exists():
        with np.load(coarse) as values:
            assert len(values["terms"]) == values["coefficients"].shape[0]
            if "control_points" in values:
                assert values["control_points"].shape[0] == len(values["terms"])


def test_selection_called_once_and_cache_tampering_rejected(tmp_path, monkeypatch):
    import deepgs.workflow as workflow
    data = tmp_path / "input.npz"
    profile(data)
    original = workflow.select
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(workflow, "select", counted)
    result = run(configuration(data), output=tmp_path / "run")
    assert calls == [1] and isinstance(result.selection.terms, tuple)
    cache = tmp_path / "run/selection/library.npz"
    with np.load(cache) as arrays:
        values = dict(arrays)
    values["sample_t"] = values["sample_t"] + .01
    np.savez(cache, **values)
    with pytest.raises(ValueError, match="fingerprint"):
        load_library(cache)


def test_inversion_configuration_still_validates_before_reading(tmp_path):
    config = configuration(tmp_path / "nonexistent.npz")
    config["inversion"]["enabled"] = True
    config["inversion"]["coefficient_form"] = "unknown"
    with pytest.raises(ValueError, match="coefficient_form"):
        run(config, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_selection_failure_records_status(tmp_path, monkeypatch):
    data = tmp_path / "input.npz"
    profile(data)
    def fail(*args):
        raise RuntimeError("controlled selector failure")
    monkeypatch.setattr("deepgs.workflow.select", fail)
    with pytest.raises(RuntimeError, match="controlled"):
        run(configuration(data), output=tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert manifest["status"] == summary["execution_status"] == "failed"
    assert manifest["stages"]["inversion"] == "not_requested"


def test_dnn_ensemble_workflow_saves_explicit_coordinates(tmp_path):
    path = tmp_path / "input.npz"
    profile(path)
    config = configuration(path)
    config["preprocess"] = dict(differentiation="dnn", seeds=[3, 7],
                                field=dict(hidden_layers=1, width=4, iterations=1, dtype="float64"))
    config["library"] = dict(depth_count=5, time_count=20)
    result = run(config, output=tmp_path / "run")
    library = load_library(tmp_path / "run/selection/library.npz")
    assert library.sample_t.size == 100
    np.testing.assert_array_equal(library.sample_t, np.tile(np.linspace(0, 1, 20), 5))
    np.testing.assert_array_equal(library.sample_z, np.repeat(np.linspace(.05, .25, 5), 20))
    assert result.execution_status == "completed"


def test_config_relative_path_ignores_calling_directory(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    profile(source / "input.npz")
    config = configuration("input.npz")
    path = source / "input.yaml"
    path.write_text(yaml.safe_dump(config))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = run(path)
    assert result.execution_status == "completed"
    assert Path(result.artifacts["directory"]) == elsewhere / "intermediate/runs/input"


@pytest.mark.parametrize("bad", ["nan", "duplicate", "reversed"])
def test_bad_coordinates_rejected_before_output(tmp_path, bad):
    path = tmp_path / "bad.npz"
    profile(path)
    with np.load(path) as arrays:
        values = dict(arrays)
    if bad == "nan":
        values["times"][0] = np.nan
    elif bad == "duplicate":
        values["times"][1] = values["times"][0]
    else:
        values["depths"] = values["depths"][::-1]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="coordinates"):
        run(configuration(path), output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_percent_csv_and_strict_columns(tmp_path):
    path = tmp_path / "sensors.csv"
    pd.DataFrame({"0.05m": [20, 21], "0.1m": [24, 25]}).to_csv(path, index=False)
    config = configuration(path)
    config["data"].update(depths=[.05, .1], minutes_per_row=60, units=dict(water_content="percent"))
    raw = load_data(load_experiment(config))
    assert raw.times[-1] == pytest.approx(1 / 24)
    assert raw.theta[0, 0] == pytest.approx(.2)
    config["data"]["depths"] = [.05, .15]
    with pytest.raises(ValueError, match="named sensor"):
        load_data(load_experiment(config))
