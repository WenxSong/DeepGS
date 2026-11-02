"""Offline evaluation requires an explicit scientific reference in the new API."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def evaluator(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("deepgs_benchmark_evaluate", directory / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_never_assumes_loam_reference(tmp_path, monkeypatch):
    module = evaluator(monkeypatch)
    (tmp_path / "selection").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(dict(schema_version=2, status="completed", configuration=dict(problem=dict(soil="homogeneous")))))
    (tmp_path / "summary.json").write_text(json.dumps(dict(inversion=dict(status="completed", metrics=dict(theta_rmse=.01)))))
    (tmp_path / "selection/structure.json").write_text(json.dumps(dict(status="selected", terms=["theta_z", "theta_zz", "theta_z^2"])))
    score = module.score_run(tmp_path)
    assert score["structure"]["status"] == "not_evaluated"
    assert score["coefficient_errors"] == {}
    assert module.score_run(tmp_path, reference_soil="loam")["structure"]["exact"]


def test_legacy_evaluator_remains_compatible(tmp_path, monkeypatch):
    module = evaluator(monkeypatch)
    (tmp_path / "structure.json").write_text(json.dumps(dict(terms=["theta_z", "theta_zz", "theta_z^2"])))
    assert module.score_run(tmp_path)["structure"]["exact"]
