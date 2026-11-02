"""Normalize a soil-column CSV with an explicitly supplied sampling interval."""
import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def convert(source, output, *, minutes_per_row, data_license="not_specified"):
    source, output = Path(source), Path(output)
    if not math.isfinite(minutes_per_row) or minutes_per_row <= 0:
        raise ValueError("minutes_per_row must be positive and finite")
    table = pd.read_csv(source)
    if "time" in table.columns:
        raise ValueError("This converter expects legacy CSV without a time column")
    try:
        depths = np.array([float(name[:-1]) if name.endswith("m") else float("nan") for name in table.columns])
        theta = table.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("Expected numeric water content and '<depth>m' headers") from error
    if (depths.size < 2 or len(table) < 2 or not np.isfinite(depths).all() or (depths < 0).any()
            or not (depths[1:] > depths[:-1]).all()):
        raise ValueError("At least two increasing nonnegative sensor depths and two records are required")
    if not np.isfinite(theta).all() or (theta < 0).any() or (theta > 1).any():
        raise ValueError("Water content must be finite fractions in [0,1]; handle missing values explicitly")
    times = np.arange(len(table), dtype=np.float64) * minutes_per_row / 1440.
    metadata = dict(schema_version=1, name=output.name, kind="real", observations="observations.npz",
        units=dict(time="day", depth="m", water_content="fraction"), depth_positive="down",
        time_origin="first_recorded_sample; absolute start time not supplied",
        source=f"User-provided soil-column observations from {source.name}", license=data_license,
        details=dict(original_file=source.name, original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     sampling_interval_minutes=float(minutes_per_row),
                     time_conversion="time[i] = i * sampling_interval_minutes / 1440",
                     preprocessing="No interpolation, resampling, missing-value filling or water-content alteration; all sensor columns retained"))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, times=times, depths=depths, theta=theta)
    with (output / "observations.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", *table.columns])
        writer.writerows([[format(t, ".17g"), *(format(v, ".17g") for v in row)] for t, row in zip(times, theta)])
    with (output / "metadata.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return dict(shape=list(theta.shape), original_sha256=metadata["details"]["original_sha256"], output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes-per-row", type=float, required=True)
    parser.add_argument("--data-license", default="not_specified")
    args = parser.parse_args()
    result = convert(args.source, args.output_dir, minutes_per_row=args.minutes_per_row, data_license=args.data_license)
    print(f"Saved {result['shape']} real observations to {args.output_dir}")


if __name__ == "__main__":
    main()
