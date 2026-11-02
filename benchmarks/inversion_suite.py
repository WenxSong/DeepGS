"""Paired known-structure joint inversion, with matched field initialization and samples.

This evaluates coefficient recovery conditional on a supplied structure, not
structure discovery. Both backends use the same field architecture, noise, seed,
collocation sampler and epoch budget. Wall time is reported separately.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, TRUE_TERMS, environment, fingerprint, relative_l2
from deepgs.data import Observations, load_profile_txt
from deepgs.inversion import InversionSettings, run_inversion
from deepgs.physics import SOILS
from deepgs.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--noise", nargs="+", type=float, default=[0, 0.05])
    parser.add_argument("--epochs", type=int, default=20000)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_profile_txt(ROOT / "intermediate/data/th.txt")
    clean = base.select_depths([0.05, 0.10, 0.15, 0.20, 0.25])
    clean = Observations(clean.times[::5], clean.depths, clean.theta[::5])
    held = base.select_depths([0.07, 0.12, 0.17, 0.22])
    held = Observations(held.times[::5], held.depths, held.theta[::5])
    manifest = dict(mode="known_structure_inversion", supplied_terms=TRUE_TERMS, epoch_budget=args.epochs,
                    seeds=args.seeds, noise=args.noise, device=args.device, environment=environment(),
                    data=fingerprint(ROOT / "intermediate/data/th.txt"), time_stride=5, sensor_depths=clean.depths.tolist(),
                    held_sensor_depths=held.depths.tolist(),
                    source=[fingerprint(p) for p in sorted((ROOT / "src/deepgs").glob("*.py"))] + [fingerprint(Path(__file__))])
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Changed provenance; choose a new output directory")
    save_json(path, manifest)
    records = []
    for noise in args.noise:
        for seed in args.seeds:
            observed = clean.add_noise(noise, seed)
            for form in ("network", "polynomial"):
                name = f"{form}_noise{noise:g}_seed{seed}"
                destination = args.output / (name + ".json")
                if destination.exists():
                    records.append(json.loads(destination.read_text(encoding="utf-8")))
                    continue
                save_json(args.output / "status.json", dict(state="running", active=name, completed=len(records)))
                settings = InversionSettings(coefficient_form=form, epochs=args.epochs, log_every=max(1, args.epochs // 10))
                started = time.monotonic()
                result = run_inversion(observed, TRUE_TERMS, [-1, 1, 1], torch.device(args.device), settings, seed)
                coefficients = {}
                for low, high in ((0.16, 0.38), (0.16, 0.30)):
                    grid = result["theta_grid"]
                    mask = (grid >= low) & (grid <= high)
                    coefficients[f"{low}:{high}"] = dict(points=int(mask.sum()), errors={})
                    if mask.any():
                        truth = SOILS["loam"].coefficients(grid[mask])
                        coefficients[f"{low}:{high}"]["errors"] = {n: relative_l2(c[mask], truth[n]) for n, c in zip(TRUE_TERMS, result["curves"])}
                with torch.no_grad():
                    points = torch.as_tensor(held.points(), device=args.device, dtype=torch.float64)
                    predicted = result["field"](points[:, :1], points[:, 1:]).cpu().numpy().reshape(held.theta.shape)
                record = dict(case=name, mode="known_structure_inversion", settings=asdict(settings), noise=noise, seed=seed,
                              elapsed_seconds=time.monotonic() - started, coefficients=coefficients,
                              observed_theta_rmse=result["theta_rmse"],
                              clean_training_sensor_theta_relative_error=relative_l2(result["fitted"], clean.theta),
                              held_sensor_theta_relative_error=relative_l2(predicted, held.theta), history=result["history"])
                np.savez_compressed(args.output / (name + ".npz"), theta_grid=result["theta_grid"], curves=result["curves"],
                                    fitted=result["fitted"], held_fitted=predicted, terms=np.asarray(TRUE_TERMS))
                save_json(destination, record)
                records.append(record)
                print(f"finished {name} in {record['elapsed_seconds']:.1f}s", flush=True)
                save_json(args.output / "evaluation.json", dict(mode="known_structure_inversion", records=records))
    save_json(args.output / "status.json", dict(state="completed", completed=len(records)))


if __name__ == "__main__":
    main()
