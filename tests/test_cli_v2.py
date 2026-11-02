"""The public CLI and installed-package templates share the capability contract."""
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest
import yaml

from deepgs.capabilities import CAPABILITIES, format_capabilities, validate_combination
from deepgs.cli import main
from deepgs.experiment import load_experiment
from deepgs.template_api import build_template, write_template


COMBINATIONS = [(item.selector, item.soil, form) for item in CAPABILITIES for form in item.coefficient_forms]


def test_capability_registry_describes_seven_legal_combinations():
    assert len(COMBINATIONS) == 7
    assert {item.selector for item in CAPABILITIES} == {"group", "spline", "spatial_group"}
    assert "fixed-structure" in format_capabilities()
    with pytest.raises(ValueError, match="supports coefficient forms"):
        validate_combination("spatial_group", "bspline")


@pytest.mark.parametrize("selector,soil,form", COMBINATIONS)
@pytest.mark.parametrize("budget", ["quickstart", "full"])
def test_bundled_templates_cover_legal_combinations(selector, soil, form, budget, tmp_path):
    config = tmp_path / "config.yaml"
    write_template(config, selector, form, budget=budget)
    parsed = load_experiment(config)
    assert parsed.problem.soil == soil
    assert parsed.structure.selector == selector
    assert parsed.inversion.coefficient_form == form
    assert parsed.inversion.enabled
    assert parsed.data.path == str(tmp_path / "YOUR_DATA.npz")
    assert not (tmp_path / "YOUR_DATA.npz").exists()
    assert parsed.preprocess.field.iterations == (20 if budget == "quickstart" else 20000)
    assert parsed.inversion.training.epochs == (20 if budget == "quickstart" else 20000)
    if form == "polynomial":
        assert parsed.inversion.polynomial.signs == {"theta_z": -1, "theta_zz": 1, "theta_z^2": 1}
        assert "explicit Richards-equation assumptions" in config.read_text(encoding="utf-8")


@pytest.mark.parametrize("selector", [item.selector for item in CAPABILITIES])
def test_selection_only_template(selector):
    raw = build_template(selector, selection_only=True)
    assert raw["inversion"]["enabled"] is False
    assert raw["data"]["path"] == "YOUR_DATA.npz"


def test_templates_are_fresh_and_refuse_overwrite(tmp_path):
    first = build_template("group")
    first["data"]["path"] = "changed"
    assert build_template("group")["data"]["path"] == "YOUR_DATA.npz"
    output = tmp_path / "config.yaml"
    output.write_text("existing user configuration", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_template(output, "group")
    assert output.read_text() == "existing user configuration"


def test_methods_command_never_runs_a_workflow(monkeypatch, capsys):
    monkeypatch.setattr("deepgs.api.run", lambda *args, **kwargs: pytest.fail("methods ran a workflow"))
    main(["methods"])
    output = capsys.readouterr().out
    assert all(selector in output for selector in ("group", "spline", "spatial_group"))
    assert "polynomial" in output and "bspline" in output


def test_init_creates_config_and_rejects_invalid_soil_and_existing_file(tmp_path, capsys):
    output = tmp_path / "created.yaml"
    main(["init", "--selector", "spline", "--coefficient-form", "bspline", "--output", str(output)])
    assert yaml.safe_load(output.read_text())["inversion"]["coefficient_form"] == "bspline"
    assert "Set data.path" in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["init", "--selector", "group", "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == before
    bad = tmp_path / "invalid.yaml"
    with pytest.raises(SystemExit):
        main(["init", "--selector", "group", "--soil", "heterogeneous", "--output", str(bad)])
    assert not bad.exists()
    with pytest.raises(SystemExit):
        main(["init", "--selector", "spatial_group", "--coefficient-form", "polynomial", "--output", str(bad)])
    assert not bad.exists()


@pytest.mark.parametrize("execution,status,exit_code", [("completed", "completed", None),
    ("completed", "skipped_empty", None), ("completed", "not_requested", None),
    ("incomplete", "blocked_missing_prior", 1), ("failed", "failed", 1)])
def test_run_routes_public_api_and_returns_nonzero_on_incomplete(monkeypatch, capsys, execution, status, exit_code):
    calls = []
    def execute(config, *, output, device):
        calls.append((config, output, device))
        return SimpleNamespace(execution_status=execution, selection=SimpleNamespace(status="selected", terms=("theta_z",)),
                               inversion=SimpleNamespace(status=status), artifacts={"directory": "runs/test"})
    monkeypatch.setattr("deepgs.api.run", execute)
    arguments = ["run", "config.yaml", "--output", "runs/test", "--device", "cpu"]
    if exit_code is None:
        main(arguments)
    else:
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == exit_code
    assert calls == [(Path("config.yaml"), Path("runs/test"), "cpu")]
    assert f"Inversion {status}" in capsys.readouterr().out


def test_unknown_command_uses_cli_error_not_legacy_runner(monkeypatch):
    monkeypatch.setattr("deepgs.cli.legacy_main", lambda *args: pytest.fail("unknown command entered legacy"))
    with pytest.raises(SystemExit) as error:
        main(["rn", "input.yaml"])
    assert error.value.code == 2


def test_bare_yaml_remains_explicitly_warned_legacy_route(monkeypatch):
    calls = []
    monkeypatch.setattr("deepgs.cli.legacy_main", lambda arguments: calls.append(arguments))
    with pytest.warns(FutureWarning, match="legacy workflow"):
        main(["old.yaml", "--device", "cpu"])
    assert calls == [["old.yaml", "--device", "cpu"]]


def test_generated_config_runs_real_selection_cli_from_another_directory(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "selection.yaml"
    main(["init", "--selector", "group", "--selection-only", "--output", str(path)])
    times, depths = np.linspace(0, 1, 60), np.linspace(.05, .35, 7)
    theta = .2 + .02 * np.sin(4 * times[:, None]) * np.exp(-depths[None, :])
    np.savez(configs / "observations.npz", times=times, depths=depths, theta=theta)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["data"]["path"] = "observations.npz"
    raw["preprocess"]["differentiation"] = "fd"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "result"
    main(["run", str(path), "--device", "cpu", "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_status"] == "completed"
    assert summary["inversion"]["status"] == "not_requested"
    assert (output / "selection/structure.json").exists()
    assert not (output / "inversion").exists()
