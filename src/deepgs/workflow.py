"""Select a structure once, then invert its fixed coefficients with a PINN."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .artifacts import finite_json, save_selection, save_inversion
from .contracts import CANONICAL_UNITS, InversionResult, LibraryArtifact, RunResult
from .data import Observations, load_profile_npz, load_profile_txt
from .datasets import csv_has_explicit_time, load_standard_csv, validate_dataset, validate_observations
from .experiment import load_experiment
from .fixed_inversion import invert
from .library import average_libraries, finite_difference_library, library_on_grid
from .pipelines import build_field, candidate_grid
from .pinn import fit_field
from .provenance import file_record
from .selection import select
from .utils import resolve_device, save_json, set_seed

DEPTH_TO_METRES = {"m": 1., "cm": .01, "mm": .001}
TIME_TO_DAYS = {"day": 1., "hour": 1 / 24, "minute": 1 / 1440, "second": 1 / 86400}


def _load_legacy_csv(data):
    """Read a sensor CSV without a time column, using the configured row interval."""
    if data.depths is None:
        raise ValueError("CSV without a time column requires explicit data.depths and minutes_per_row")
    table = pd.read_csv(data.path)
    names = [f"{depth:g}m" for depth in data.depths]
    if not all(n in table.columns for n in names):
        raise ValueError(f"CSV needs named sensor columns: {names}")
    return Observations(np.arange(len(table)) * data.minutes_per_row / 1440,
                        np.asarray(data.depths), table.loc[:, names].to_numpy(dtype=float))


def load_data(config):
    """Normalize before depth selection, resampling and synthetic noise."""
    data = config.data
    unit = data.units
    fmt = data.format or {".npz": "npz", ".csv": "csv"}.get(Path(data.path).suffix.lower(), "txt")
    depth_factor = DEPTH_TO_METRES[unit.depth]
    time_factor = TIME_TO_DAYS[unit.time]
    explicit_csv = fmt == "csv" and csv_has_explicit_time(data.path)
    if explicit_csv:
        raw = load_standard_csv(data.path)
    elif fmt == "csv":
        raw = _load_legacy_csv(data)
    elif fmt == "npz":
        raw = load_profile_npz(data.path)
    else:
        raw = load_profile_txt(data.path, data.depth_max)
    observations = Observations(raw.times * time_factor, raw.depths * depth_factor,
                                raw.theta / (100 if unit.water_content == "percent" else 1))
    validate_observations(observations)
    if data.resample_minutes:
        observations = observations.resample_to_minutes()
    if data.depths is not None and (fmt != "csv" or explicit_csv):
        requested = np.asarray(data.depths) * depth_factor
        observations = observations.select_depths(requested)
    validate_observations(observations)
    return observations.add_noise(data.noise, data.noise_seed)


def _dataset_provenance(config):
    """Attach only metadata explicitly naming this input; never change settings."""
    import yaml
    path = Path(config.data.path).resolve()
    metadata_file = path.parent / "metadata.yaml"
    if not metadata_file.is_file():
        return None, None
    metadata_record = file_record(metadata_file)
    try:
        metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Adjacent dataset metadata must be valid YAML") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("observations"), str):
        return None, None
    target = (metadata_file.parent / metadata["observations"]).resolve()
    if target != path:
        return None, None
    report = validate_dataset(metadata_file.parent)
    if report["observations"] != str(path) or file_record(metadata_file)["sha256"] != metadata_record["sha256"]:
        raise ValueError("Dataset metadata changed while validating")
    if asdict(config.data.units) != report["units"] or config.data.depth_positive != report["depth_positive"]:
        raise ValueError("Configuration units/depth direction conflict with the dataset metadata")
    return report, metadata_record


def build_library(config, observations, device):
    """Build the candidate library by finite differences or a DNN field ensemble."""
    prep = config.preprocess
    if prep.differentiation == "fd":
        raw = finite_difference_library(observations.theta.T, observations.depths, observations.times,
                                        config.library.max_order)
    else:
        depths, times = candidate_grid(config.library, observations)
        libraries = []
        for seed in prep.seeds:
            set_seed(seed)
            model = build_field(prep.field, observations, device)
            fit_field(model, observations, prep.field.iterations, prep.field.learning_rate,
                      log_every=max(prep.field.iterations // 10, 1), label=f"selection seed {seed}",
                      lbfgs_iterations=prep.field.lbfgs_iterations)
            libraries.append(library_on_grid(model, depths, times, config.library.max_order))
        raw = average_libraries(libraries)
    return LibraryArtifact(raw, raw.depth, raw.time)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _output_folder(source, output):
    """Resolve the run folder, refusing one that already holds results."""
    name = Path(source).stem if isinstance(source, (str, Path)) else "selection"
    folder = Path(output or Path("intermediate") / "runs" / name).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Output directory is not empty: {folder}")
    return folder


def _check_differentiation_inputs(config, observations):
    """Reject observation depths the configured differentiation cannot use."""
    depths = observations.depths
    if config.preprocess.differentiation == "fd":
        if depths.size < 5 or not np.allclose(np.diff(depths), np.diff(depths)[0]):
            raise ValueError("Finite differences require at least five equally spaced observed depths")
    if config.preprocess.differentiation == "dnn":
        for bound, value in (("depth_min", config.library.depth_min), ("depth_max", config.library.depth_max)):
            if value is not None and not depths.min() <= value <= depths.max():
                raise ValueError(f"library.{bound} must be inside the observed depth interval")
        low = depths.min() if config.library.depth_min is None else config.library.depth_min
        high = depths.max() if config.library.depth_max is None else config.library.depth_max
        if low >= high:
            raise ValueError("Candidate depth interval must have positive width")


def _load_checked_observations(config):
    """Load observations, failing if the input or its metadata changed meanwhile."""
    input_record = file_record(config.data.path)
    dataset_report, metadata_record = _dataset_provenance(config)
    observations = load_data(config)
    if file_record(config.data.path)["sha256"] != input_record["sha256"]:
        raise ValueError("Input file changed while loading")
    if metadata_record is not None and file_record(metadata_record["path"])["sha256"] != metadata_record["sha256"]:
        raise ValueError("Dataset metadata changed while loading")
    inputs = [input_record] + ([metadata_record] if metadata_record else [])
    return observations, inputs, dataset_report


def _save_config(source, config, folder):
    if isinstance(source, (str, Path)):
        shutil.copyfile(source, folder / "config.yaml")
    else:
        import yaml
        (folder / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")


def _start_manifest(config, device, inputs, dataset_report):
    """Record the settings, inputs, source and environment before training begins."""
    package_dir = Path(__file__).parent
    return dict(schema_version=2, workflow="two_stage", status="running", configuration=asdict(config),
                started_utc=_utc_now(), device=str(device),
                units=dict(CANONICAL_UNITS), inputs=inputs,
                dataset=dataset_report,
                source=[file_record(p) for p in sorted(package_dir.rglob("*"))
                        if p.is_file() and p.suffix in (".py", ".yaml")],
                environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                 packages={p: importlib.metadata.version(p)
                                           for p in ("scipy", "kneed", "scikit-learn", "pyyaml")}),
                stages=dict(selection="running", inversion="pending" if config.inversion.enabled else "not_requested"),
                stage_timings=dict(selection=dict(started_utc=_utc_now())))


def _invert_frozen_structure(config, observations, result, target):
    """Invert the selected structure and verify that it stayed frozen throughout."""
    result.assert_unchanged()
    signature = (result.terms, result.term_indices, result.library_fingerprint)
    inversion = invert(observations, result, config.inversion, config.problem.soil, str(target))
    result.assert_unchanged()
    if signature != (result.terms, result.term_indices, result.library_fingerprint) or inversion.terms != result.terms:
        raise RuntimeError("Frozen structure changed during inversion")
    return inversion


def _equation_text(result):
    if result.terms:
        right_side = " + ".join(f"f[{n}] * ({n})" for n in result.terms)
    else:
        right_side = "0" if result.status == "empty" else "unavailable"
    return "theta_t = " + right_side


def _save_summary(folder, execution_status, result, inversion):
    summary = dict(schema_version=2, execution_status=execution_status, units=dict(CANONICAL_UNITS),
                   structure_source="selected", selection=result.summary(), inversion=inversion.summary(),
                   equation=_equation_text(result),
                   reference_metrics=dict(status="unavailable",
                                          reason="No reference supplied; use explicit offline evaluation"),
                   nonfinite_metric_policy="Non-finite diagnostic values are written as null")
    save_json(folder / "summary.json", finite_json(summary))


def run(source, *, device=None, output=None):
    """Run selection and any configured inversion, saving results to a new folder."""
    config = load_experiment(source)
    requested_device = device or config.runtime.device
    target = resolve_device(None if requested_device == "auto" else requested_device)
    folder = _output_folder(source, output)
    observations, inputs, dataset_report = _load_checked_observations(config)
    _check_differentiation_inputs(config, observations)
    folder.mkdir(parents=True, exist_ok=True)
    _save_config(source, config, folder)
    manifest = _start_manifest(config, target, inputs, dataset_report)
    save_json(folder / "manifest.json", manifest)
    started = time.monotonic()
    phase = "selection"
    result = None
    inversion = InversionResult()
    try:
        library = build_library(config, observations, target)
        result = select(library, config.structure, config.problem.soil)
        save_selection(folder, result, library, config.runtime.save_library)
        manifest.update(library_fingerprint=library.fingerprint)
        manifest["stages"]["selection"] = result.status
        manifest["stage_timings"]["selection"].update(finished_utc=_utc_now(),
                                                      elapsed_seconds=time.monotonic() - started)
        if config.inversion.enabled:
            phase = "inversion"
            phase_started = time.monotonic()
            manifest["stages"]["inversion"] = "running"
            manifest["stage_timings"]["inversion"] = dict(started_utc=_utc_now())
            save_json(folder / "manifest.json", finite_json(manifest))
            inversion = _invert_frozen_structure(config, observations, result, target)
            save_inversion(folder, inversion, observations, result)
            manifest["stages"]["inversion"] = inversion.status
            manifest["stage_timings"]["inversion"].update(finished_utc=_utc_now(),
                                                          elapsed_seconds=time.monotonic() - phase_started)
        execution_status = "incomplete" if inversion.status == "blocked_missing_prior" else "completed"
        _save_summary(folder, execution_status, result, inversion)
        manifest.update(status=execution_status)
        return RunResult(execution_status, result, inversion, dict(directory=str(folder)))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest["stages"][phase] = "failed"
        manifest["stage_timings"][phase].update(finished_utc=_utc_now())
        save_json(folder / "summary.json", dict(schema_version=2, execution_status="failed",
                  error=manifest["error"], selection=finite_json(result.summary()) if result else dict(status="failed"),
                  inversion=dict(status="failed" if phase == "inversion" else "not_requested")))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started, finished_utc=_utc_now())
        save_json(folder / "manifest.json", finite_json(manifest))
