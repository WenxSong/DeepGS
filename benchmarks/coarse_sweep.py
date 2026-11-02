"""Complete the three-sensor DNN energy-gain scan, serially and resumably.

Matches the DNN conditions in the legacy diag_energy_gain_wide.py: sin 6x50,
20k Adam, raw inputs, 10 depths, two noise levels and two seeds. FD comparisons
remain in fixed_library.py; this script trains only the missing DNN cases.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, TRUE_TERMS, environment, fingerprint, support_score
from deepgs.data import load_profile_txt
from deepgs.grouping import build_system
from deepgs.library import CandidateLibrary, library_at_points, term_names
from deepgs.networks import FieldNetwork
from deepgs.pinn import fit_field
from deepgs.sparse import sgtr_path, aic_score
from deepgs.utils import save_json, set_seed


def scan(lib, truth):
    records = []
    for window in (0.001, 0.002, 0.005, 0.01):
        system = build_system(lib, window, 100, "group", by_depth=True, merge_tail=0.01)
        path = sgtr_path(system, 200, 1e-2, 1, 10**1.2)
        energy = sum(float(np.sum(y**2)) for y in system.y)
        for gain in (None, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.15):
            epsilon = 10**1.2 if gain is None else gain * energy / 2
            best = min(path, key=lambda p: aic_score(p["rss"], system.n_samples, p["k"], epsilon))
            names = [system.feature_names[j] for j in best["active"]]
            records.append(dict(win_size=window, groups=system.n_groups, min_energy_gain=gain,
                                epsilon=epsilon, terms=names, score=support_score(names, truth)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise", type=float, nargs="+", default=[0, 0.1])
    parser.add_argument("--soils", nargs="+", choices=["homogeneous", "heterogeneous"], default=["homogeneous", "heterogeneous"])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(kind="development_sweep", steps=args.steps, device=args.device, seeds=args.seeds, noise=args.noise,
                      soils=args.soils, environment=environment(),
                      source=[fingerprint(p) for p in sorted((ROOT / "src/deepgs").glob("*.py"))] + [fingerprint(Path(__file__))],
                      data=[fingerprint(ROOT / "intermediate/data/th.txt"), fingerprint(ROOT / "intermediate/data/loam_all_0.2/th.txt")])
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != provenance:
        raise ValueError("Sweep provenance changed; choose another directory")
    save_json(manifest, provenance)
    completed = []
    for soil in args.soils:
        file = ROOT / ("intermediate/data/th.txt" if soil == "homogeneous" else "intermediate/data/loam_all_0.2/th.txt")
        full = load_profile_txt(file)
        base = full.select_depths([0.05, 0.15, 0.25])
        t, z = np.meshgrid(base.times, np.linspace(0.05, 0.25, 10))
        truth = TRUE_TERMS if soil == "homogeneous" else ["1", *TRUE_TERMS]
        for noise in args.noise:
            for seed in args.seeds:
                name = f"{soil}_noise{noise:g}_seed{seed}"
                result_path = args.output / (name + ".json")
                cache = args.output / (name + ".npz")
                checkpoint = args.output / (name + ".pt")
                save_json(args.output / "status.json", dict(state="running", active=name, completed=completed))
                if result_path.exists():
                    completed.append(name)
                    print(f"cached result {name}", flush=True)
                    continue
                if cache.exists():
                    with np.load(cache) as stored:
                        lib = CandidateLibrary(stored["theta"], stored["terms"], stored["theta_t"], term_names(3), stored["depth"])
                else:
                    set_seed(seed)
                    field = FieldNetwork([50] * 6, "sin").to(args.device)
                    if checkpoint.exists():
                        field.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)["field"])
                    else:
                        print(f"train {name} for {args.steps} steps", flush=True)
                        history = fit_field(field, base.add_noise(noise, seed), args.steps,
                                            log_every=max(1, args.steps // 10), label=name)
                        torch.save(dict(field=field.state_dict(), history=history), checkpoint)
                    lib = library_at_points(field, z.reshape(-1, 1), t.reshape(-1, 1), 3, batch_size=4096)
                    lib.save(cache)
                selections = scan(lib, truth)
                save_json(result_path, dict(case=name, soil=soil, noise=noise, seed=seed, steps=args.steps,
                          cache=fingerprint(cache), rows=len(lib.theta), selections=selections))
                completed.append(name)
                print(f"finished {name}: {sum(r['score']['exact'] for r in selections)}/{len(selections)} settings recover exact support", flush=True)
    save_json(args.output / "status.json", dict(state="completed", completed=completed))


if __name__ == "__main__":
    main()
