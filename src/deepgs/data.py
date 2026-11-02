"""Water-content observations at fixed depths."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Observations:
    """theta has shape (n_times, n_depths); depths are in metres, positive downward."""

    times: np.ndarray
    depths: np.ndarray
    theta: np.ndarray

    def __post_init__(self):
        if self.theta.shape != (self.times.size, self.depths.size):
            raise ValueError(f"theta shape {self.theta.shape} does not match times and depths")
        if not np.isfinite(self.theta).all():
            raise ValueError("Observations contain non-finite values")

    def points(self) -> np.ndarray:
        """(z, t) of every entry in the row order of values()."""
        z, t = np.meshgrid(self.depths, self.times)
        return np.column_stack((z.ravel(), t.ravel()))

    def values(self) -> np.ndarray:
        return self.theta.reshape(-1, 1)

    def select_depths(self, depths: Sequence[float], tolerance: float = 5e-4) -> "Observations":
        columns = []
        for depth in depths:
            index = int(np.argmin(np.abs(self.depths - depth)))
            if abs(self.depths[index] - depth) > tolerance:
                raise ValueError(f"Depth {depth} m is not on the grid; nearest is {self.depths[index]} m")
            columns.append(index)
        return replace(self, depths=self.depths[columns], theta=self.theta[:, columns])

    def add_noise(self, sigma: float, seed: int = 0) -> "Observations":
        """theta* = theta + sigma * std(theta) * N(0, 1), clipped to [0, 1]."""
        if sigma < 0:
            raise ValueError("Noise level must be non-negative")
        if sigma == 0:
            return self
        rng = np.random.default_rng(seed)
        noisy = self.theta + sigma * np.std(self.theta) * rng.standard_normal(self.theta.shape)
        return replace(self, theta=np.clip(noisy, 0.0, 1.0))

    def resample_to_minutes(self) -> "Observations":
        """Keep the last record of each minute. Variable solver steps otherwise bunch up in time."""
        minutes = np.rint(self.times * 1440.0).astype(np.int64)
        last = {int(minute): index for index, minute in enumerate(minutes)}
        keys = sorted(last)
        return replace(self, times=np.asarray(keys, dtype=float) / 1440.0, theta=self.theta[[last[k] for k in keys]])


def load_profile_txt(path: str | Path, depth_max: float = 1.0) -> Observations:
    """Solver output: first column time, remaining columns equally spaced nodes from 0 to depth_max."""
    raw = np.loadtxt(path, dtype=np.float64)
    theta = raw[:, 1:]
    return Observations(raw[:, 0], np.linspace(0.0, depth_max, theta.shape[1]), theta)


def load_profile_npz(path: str | Path) -> Observations:
    with np.load(path) as archive:
        return Observations(archive["times"], archive["depths"], archive["theta"])


def load_sensor_csv(path: str | Path, depths: Sequence[float], minutes_per_row: float = 1.0) -> Observations:
    """Sensor table with one row per record. Columns named like '0.05m' are matched to depths."""
    table = pd.read_csv(path)
    names = [f"{depth:g}m" for depth in depths]
    if all(name in table.columns for name in names):
        values = table.loc[:, names].to_numpy(dtype=np.float64)
    else:
        values = table.iloc[:, : len(depths)].to_numpy(dtype=np.float64)
    times = np.arange(len(values), dtype=np.float64) * minutes_per_row / 1440.0
    return Observations(times, np.asarray(depths, dtype=np.float64), values)


def load_observations(
    path: str | Path,
    file_format: str | None = None,
    depths: Sequence[float] | None = None,
    noise: float = 0.0,
    noise_seed: int = 0,
    depth_max: float = 1.0,
    resample_minutes: bool = False,
    minutes_per_row: float = 1.0,
) -> Observations:
    path = Path(path)
    file_format = file_format or {".npz": "npz", ".csv": "csv"}.get(path.suffix, "txt")
    if file_format == "csv":
        if depths is None:
            raise ValueError("Sensor tables need explicit depths")
        observations = load_sensor_csv(path, depths, minutes_per_row)
    else:
        observations = load_profile_npz(path) if file_format == "npz" else load_profile_txt(path, depth_max)
        if resample_minutes:
            observations = observations.resample_to_minutes()
        if depths is not None:
            observations = observations.select_depths(depths)
    return observations.add_noise(noise, noise_seed)
