"""Public dataset metadata and explicit-coordinate observation formats."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from .data import Observations

CANONICAL_UNITS = dict(time="day", depth="m", water_content="fraction")
METADATA_FIELDS = {"schema_version", "name", "kind", "observations", "units", "depth_positive",
                   "time_origin", "source", "license", "details"}
REQUIRED_METADATA_FIELDS = METADATA_FIELDS - {"details"}


def _numeric(array, name):
    values = np.asarray(array)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        label = f"{name} coordinates" if name in ("times", "depths") else name
        raise ValueError(f"Observation {label} must contain finite real numeric values")
    return values


def _validate(observations, *, canonical):
    times = _numeric(observations.times, "times")
    depths = _numeric(observations.depths, "depths")
    theta = _numeric(observations.theta, "theta")
    for name, values in (("times", times), ("depths", depths)):
        if values.ndim != 1 or values.size < 2 or not (values[1:] > values[:-1]).all() or (values < 0).any():
            raise ValueError(f"Observation {name} must have at least two nonnegative, strictly increasing coordinates")
    if theta.ndim != 2 or theta.shape != (times.size, depths.size):
        raise ValueError("Observation theta must have shape (n_times,n_depths)")
    if canonical and ((theta < 0).any() or (theta > 1).any()):
        raise ValueError("Water content must be a fraction in [0,1] after unit conversion")
    return observations


def validate_observations(observations: Observations) -> Observations:
    """Check canonical coordinates and fractions without changing or filling data."""
    if not isinstance(observations, Observations):
        raise ValueError("Expected Observations")
    return _validate(observations, canonical=True)


def csv_has_explicit_time(path: str | Path) -> bool:
    """Inspect the literal header before pandas can rename duplicated columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    return "time" in header


def load_standard_csv(path: str | Path) -> Observations:
    """Read time,<depth>m,..., preserving explicit irregular times.

    The theta range is checked after unit conversion by the caller, allowing
    the workflow's explicit percent input option. Dataset validation itself
    requires canonical fractions and calls validate_observations afterwards.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if not header or header[0] != "time" or len(header) < 3:
            raise ValueError("Standard CSV requires first column time and at least two '<depth>m' columns")
        if len(set(header)) != len(header):
            raise ValueError("Standard CSV has duplicate column names")
        depths = []
        for name in header[1:]:
            if name != name.strip() or not name.endswith("m"):
                raise ValueError("Standard CSV depth columns must be numeric '<depth>m' labels")
            try:
                depth = float(name[:-1])
            except ValueError as error:
                raise ValueError("Standard CSV depth columns must be numeric '<depth>m' labels") from error
            if not np.isfinite(depth) or depth < 0:
                raise ValueError("Standard CSV depths must be finite and nonnegative")
            depths.append(depth)
        if not np.all(np.diff(depths) > 0):
            raise ValueError("Standard CSV depths must be unique and strictly increasing")
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Standard CSV row {line} has an unexpected number of columns")
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise ValueError(f"Standard CSV row {line} must contain only numeric values") from error
    if len(rows) < 2:
        raise ValueError("Standard CSV requires at least two time rows")
    values = np.asarray(rows, dtype=np.float64)
    observations = Observations(values[:, 0], np.asarray(depths, dtype=np.float64), values[:, 1:])
    return _validate(observations, canonical=False)


def load_standard_npz(path: str | Path) -> Observations:
    """Read required numeric arrays, never loading pickled or extra arrays."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = {"times", "depths", "theta"} - set(archive.files)
            if missing:
                raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
            values = {name: _numeric(archive[name], name).copy() for name in ("times", "depths", "theta")}
    except (TypeError, AttributeError) as error:
        raise ValueError("Expected an NPZ archive with times, depths and theta arrays") from error
    return validate_observations(Observations(**values))


def validate_dataset(folder: str | Path) -> dict:
    """Validate metadata and observations; return a JSON-compatible summary."""
    folder = Path(folder).resolve()
    try:
        metadata = yaml.safe_load((folder / "metadata.yaml").read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Dataset metadata must be valid YAML") from error
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a mapping")
    missing, unknown = REQUIRED_METADATA_FIELDS - metadata.keys(), metadata.keys() - METADATA_FIELDS
    if missing or unknown:
        raise ValueError(f"Dataset metadata fields: missing={sorted(missing)}, unknown={sorted(unknown, key=str)}")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("Dataset schema_version must be 1")
    for name in ("name", "time_origin", "source", "license", "observations"):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"Dataset {name} must be a nonempty string")
    if metadata["kind"] not in ("synthetic", "real"):
        raise ValueError("Dataset kind must be synthetic or real")
    if metadata["units"] != CANONICAL_UNITS:
        raise ValueError("Dataset units must be exactly day, m and fraction")
    if metadata["depth_positive"] != "down":
        raise ValueError("Dataset depth_positive must be down")
    if "details" in metadata and not isinstance(metadata["details"], dict):
        raise ValueError("Dataset details must be a mapping")
    relative = Path(metadata["observations"])
    path = (folder / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(folder) or not path.is_file():
        raise ValueError("Dataset observations must name an existing file contained within its folder")
    if path.suffix.lower() == ".npz":
        observations = load_standard_npz(path)
        with np.load(path, allow_pickle=False) as archive:
            extra_arrays = sorted(set(archive.files) - {"times", "depths", "theta"})
    elif path.suffix.lower() == ".csv":
        observations = validate_observations(load_standard_csv(path))
        extra_arrays = []
    else:
        raise ValueError("Dataset observations must be .npz or .csv")
    return dict(status="valid", name=metadata["name"], kind=metadata["kind"], path=str(folder),
        observations=str(path), shape=[int(v) for v in observations.theta.shape], units=dict(CANONICAL_UNITS),
        depth_positive="down", time_origin=metadata["time_origin"], source=metadata["source"],
        license=metadata["license"], extra_arrays=extra_arrays,
        summary=dict(time_range=[float(observations.times.min()), float(observations.times.max())],
                     depth_range=[float(observations.depths.min()), float(observations.depths.max())],
                     water_content_range=[float(observations.theta.min()), float(observations.theta.max())]))
