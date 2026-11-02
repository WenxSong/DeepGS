"""Write a manufactured heat-equation profile; independent of the research datasets."""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", help="legacy single NPZ output")
    parser.add_argument("--dataset-dir", type=Path, help="write a standard synthetic dataset folder")
    args = parser.parse_args()
    if (args.output is None) == (args.dataset_dir is None):
        parser.error("choose either an output NPZ path or --dataset-dir")
    times = np.linspace(0, 1, 120)
    depths = np.linspace(.05, .25, 21)
    t, z = np.meshgrid(times, depths, indexing="ij")
    k = np.pi / .2
    theta = .25 + .02 * np.exp(-.002 * k**2 * t) * np.sin(k * (z - .05))
    theta += .01 * np.exp(-.002 * (2 * k)**2 * t) * np.sin(2 * k * (z - .05))
    if args.dataset_dir is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, times=times, depths=depths, theta=theta)
        print(f"Saved {args.output}; manufactured theta_t=0.002 theta_zz, units day/m/fraction")
    else:
        folder = args.dataset_dir
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "observations.npz").open("xb") as stream:
            np.savez_compressed(stream, times=times, depths=depths, theta=theta)
        with (folder / "observations.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", *(f"{z:g}m" for z in depths)])
            writer.writerows([[format(t, ".17g"), *(format(v, ".17g") for v in row)] for t, row in zip(times, theta)])
        metadata = dict(schema_version=1, name="heat_demo", kind="synthetic", observations="observations.npz",
                        units=dict(time="day", depth="m", water_content="fraction"), depth_positive="down",
                        time_origin="start_of_simulation", source="Generated independently by examples/make_demo.py",
                        license="MIT", details=dict(generator="examples/make_demo.py",
                        reference_equation="theta_t = 0.002 theta_zz", purpose="manufactured heat-equation execution demo; not a Richards recovery benchmark"))
        with (folder / "metadata.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        print(f"Saved standard synthetic dataset at {folder}")


if __name__ == "__main__":
    main()
