"""Generate matched-grid DNN libraries serially for fixed_library.py.

Defaults match the three historical clean-loam diagnostics (seed 1, five sensors).
Optional seed/noise overrides create new exploratory data; keep their directories separate.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, environment, fingerprint
from deepgs.data import load_profile_txt
from deepgs.library import library_at_points
from deepgs.networks import FieldNetwork, observation_bounds
from deepgs.pinn import fit_field
from deepgs.utils import save_json, set_seed

VARIANTS = {"tanh_norm_lbfgs": ("tanh", 5, True, 50000, 10000),
            "silu_norm_lbfgs": ("silu", 5, True, 50000, 10000),
            "sin_raw_adam": ("sin", 6, False, 20000, 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--output", type=Path, default=ROOT / "intermediate/runs/library_cache")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lbfgs", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    full = load_profile_txt(ROOT / "intermediate/data/th.txt")
    observed = full.select_depths([0.05, 0.10, 0.15, 0.20, 0.25]).add_noise(args.noise, args.noise_seed)
    times, depths = full.times[1:], full.depths[5:26]
    t, z = np.meshgrid(times, depths)
    for name in args.variants:
        activation, layers, normalize, steps, lbfgs = VARIANTS[name]
        settings = dict(activation=activation, layers=layers, width=50, normalize=normalize,
                        steps=steps if args.steps is None else args.steps, lbfgs=lbfgs if args.lbfgs is None else args.lbfgs,
                        seed=args.seed, noise=args.noise, noise_seed=args.noise_seed, device=args.device,
                        inputs=[fingerprint(ROOT / "intermediate/data/th.txt")],
                        code=[fingerprint(p) for p in sorted((ROOT / "src/deepgs").glob("*.py"))])
        stem = f"lib_{name}_noise{float(args.noise)}"
        path = args.output / (stem + ".npz")
        metadata = args.output / (stem + ".json")
        if metadata.exists():
            old = json.loads(metadata.read_text(encoding="utf-8"))
            if old["settings"] != settings:
                raise ValueError("Cache settings changed; use a new output directory")
            if path.exists() and old["status"] == "completed":
                print(f"cached {name}", flush=True)
                continue
        elif path.exists():
            raise ValueError(f"Existing cache has no provenance: {path}")
        record = dict(settings=settings, environment=environment(), status="running")
        save_json(metadata, record)
        set_seed(args.seed)
        field = FieldNetwork([50] * layers, activation, observation_bounds(observed) if normalize else None, xavier=normalize).to(args.device)
        history = fit_field(field, observed, settings["steps"], lbfgs_iterations=settings["lbfgs"],
                            log_every=max(1, settings["steps"] // 10), label=name)
        torch.save(dict(field=field.state_dict(), settings=settings, history=history), args.output / (stem + ".pt"))
        lib = library_at_points(field, z.reshape(-1, 1), t.reshape(-1, 1), 4, batch_size=4096)
        np.savez_compressed(path, theta=lib.theta, terms=lib.terms, theta_t=lib.theta_t, depth=lib.depth,
                            time=t.reshape(-1, 1), names=np.asarray(lib.names))
        record.update(status="completed", cache=fingerprint(path))
        save_json(metadata, record)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
