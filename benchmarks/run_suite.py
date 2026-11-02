"""Serial CLI benchmark runner with explicit budgets, timeout and durable status.

smoke checks wiring; pilot is a bounded diagnostic; full uses shipped schedules.
Completed cases can be resumed only when config and source hashes still match.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from common import ROOT, environment, fingerprint
from evaluate import score_run
from deepgs.config import load_config
from deepgs.utils import save_json

CASES = {"gsr_fd": "gsr_fd_loam", "gsr_dnn": "gsr_dnn_loam", "spline": "spline_sr_loam",
         "deepgs": "deepgs_homogeneous", "heterogeneous": "deepgs_heterogeneous"}


def configure(case, seed, preset, fusion):
    name, config = load_config(ROOT / "configs" / (CASES[case] + ".yaml"))
    raw = json.loads(json.dumps(asdict(config)))  # normalize tuple/list representation for resume checks
    raw["pipeline"] = name
    raw["data"]["path"] = str(ROOT / raw["data"]["path"])
    raw["data"]["noise_seed"] = seed
    if "seed" in raw:
        raw["seed"] = seed
    if "seeds" in raw:
        # A full DNN-GSR case still averages five initializations; repeat independent ensembles.
        raw["seeds"] = list(range(seed * 5 + 1, seed * 5 + 6)) if preset == "full" else [seed]
    if case == "heterogeneous":
        raw["reference"]["path"] = str(ROOT / raw["reference"]["path"])
        raw["fusion_hidden"] = fusion
    if preset != "full" and case != "gsr_fd":
        smoke = preset == "smoke"
        raw["field"].update(iterations=10 if smoke else 1000, lbfgs_iterations=0)
        if smoke:
            raw["field"].update(hidden_layers=2, width=12)
        if "library" in raw:
            raw["library"].update(depth_count=5 if smoke else 10, time_count=200 if smoke else 2000)
        if "sgtr" in raw:
            raw["sgtr"]["num_tols"] = 30 if smoke else 200
        raw["group"]["min_group_size"] = 20 if smoke else 100
        if case == "deepgs":
            raw.update(collocation_points=50 if smoke else 2000, sparse_points=2000 if smoke else 20000,
                       ado_rounds=1, ado_steps=5 if smoke else 2000, post_steps=5 if smoke else 1000,
                       log_every=5 if smoke else 200)
        if case == "heterogeneous":
            raw.update(collocation_points=50 if smoke else 2000, train_steps=10 if smoke else 3000,
                       log_every=5 if smoke else 200)
        if case == "spline":
            raw["inversion"].update(epochs=5 if smoke else 2000, collocation_points=50 if smoke else 1000,
                                    log_every=5 if smoke else 200)
            if smoke:
                raw["inversion"].update(hidden_sizes=[12, 12], coefficient_hidden=[6])
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fusion-hidden", nargs="*", type=int, default=[])
    parser.add_argument("--timeout", type=float, default=21600, help="seconds per case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    code = [fingerprint(p) for p in sorted((ROOT / "src/deepgs").glob("*.py"))]
    records = []
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for case in args.cases:
        for seed in ([args.seeds[0]] if case == "gsr_fd" else args.seeds):
            raw = configure(case, seed, args.preset, args.fusion_hidden)
            label = f"{case}_seed{seed}"
            directory = args.output / label
            case_manifest = args.output / f"{label}.case.json"
            provenance = dict(preset=args.preset, configuration=raw, source=code, device=args.device,
                              input=fingerprint(raw["data"]["path"]))
            if case_manifest.exists() and json.loads(case_manifest.read_text(encoding="utf-8")) != provenance:
                raise ValueError(f"Changed provenance for {label}; choose a new output directory")
            save_json(case_manifest, provenance)
            manifest_path = directory / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "completed":
                print(f"resume completed {label}", flush=True)
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"Incomplete {directory}; preserve it and choose a new suite output directory")
                config_path = args.output / f"{label}.yaml"
                config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
                command = [sys.executable, "-m", "deepgs.cli", str(config_path), "--output", str(directory), "--device", args.device]
                start = time.monotonic()
                save_json(args.output / "status.json", dict(state="running", active=label, completed=records,
                          command=command, timeout_seconds=args.timeout, preset=args.preset))
                print(f"start {label} ({args.preset})", flush=True)
                with (args.output / f"{label}.log").open("w", encoding="utf-8") as log:
                    try:
                        run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        save_json(args.output / "status.json", dict(state="timeout", active=label, completed=records))
                        raise
                if run.returncode:
                    save_json(args.output / "status.json", dict(state="failed", active=label, completed=records, returncode=run.returncode))
                    raise RuntimeError(f"{label} exited {run.returncode}; see its log")
                print(f"finished {label} in {time.monotonic()-start:.1f}s", flush=True)
            scored = score_run(directory, case == "heterogeneous")
            save_json(directory / "evaluation.json", scored)
            records.append(scored)
            save_json(args.output / "evaluation.json", dict(preset=args.preset, environment=environment(), records=records,
                      evaluation_source=[fingerprint(ROOT / "benchmarks" / file) for file in ("common.py", "evaluate.py", "run_suite.py")],
                      interpretation="Execution success is not scientific accuracy; pilot and smoke do not validate published performance."))
    save_json(args.output / "status.json", dict(state="completed", completed=records, preset=args.preset))


if __name__ == "__main__":
    main()
