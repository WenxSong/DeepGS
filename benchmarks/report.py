"""Export compact, source-linked evidence from the local diagnostic records."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import ROOT, fingerprint
from deepgs.utils import save_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path, default=ROOT / "intermediate/runs/fixed_library_v2")
    parser.add_argument("--inversion", type=Path, default=ROOT / "intermediate/runs/fixed_inversion_v2")
    parser.add_argument("--pilot", type=Path, default=ROOT / "intermediate/runs/heterogeneous_pilot_20261031")
    parser.add_argument("--coarse", type=Path, default=ROOT / "intermediate/runs/coarse_sweep_20261031")
    parser.add_argument("--joint", type=Path, default=ROOT / "intermediate/runs/inversion_comparison_20261031")
    parser.add_argument("--output", type=Path, default=ROOT / "intermediate/history",
                        help="report directory; existing historical reports are never overwritten")
    args = parser.parse_args()
    args.output = args.output.resolve()
    report_file = args.output / "validation_20261031.md"
    evidence_file = args.output / "validation_evidence.json"
    if report_file.exists() or evidence_file.exists():
        raise ValueError("Report output already exists; choose a new --output directory to preserve historical evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    libraries = {p.stem: read(p) for p in sorted(args.libraries.glob("*.json")) if p.stem != "manifest"}
    for record in libraries.values():
        for selection in record["selections"]:
            selection.pop("path", None)
    inverse = read(args.inversion / "results.json")
    evidence = dict(date="2026-10-31", fixed_library_manifest=read(args.libraries / "manifest.json"),
                    libraries=libraries, fixed_inversion=inverse,
                    fixed_inversion_manifest=read(args.inversion / "manifest.json"))
    pilot = read(args.pilot / "evaluation.json") if (args.pilot / "evaluation.json").exists() else None
    if pilot:
        for record in pilot["records"]:
            record["directory"] = Path(record["directory"]).relative_to(ROOT).as_posix()
        evidence["heterogeneous_pilot"] = pilot
    coarse = [read(p) for p in sorted(args.coarse.glob("*noise*_seed*.json"))]
    if coarse:
        evidence["coarse_sweep"] = dict(manifest=read(args.coarse / "manifest.json"), records=coarse,
                                        status=read(args.coarse / "status.json"))
    joint = read(args.joint / "evaluation.json") if (args.joint / "evaluation.json").exists() else None
    if joint:
        evidence["joint_inversion"] = dict(manifest=read(args.joint / "manifest.json"), results=joint,
                                          status=read(args.joint / "status.json"))
        checks = []
        for row in joint["records"]:
            file = args.joint / (row["case"] + ".npz")
            with np.load(file) as arrays:
                grid, curves = arrays["theta_grid"], arrays["curves"]
                names = arrays["terms"].tolist()
                diffusion = curves[names.index("theta_zz")]
                curvature = curves[names.index("theta_z^2")]
                derivative = np.gradient(diffusion, grid)
                mask = (grid > 0.16) & (grid < 0.38)
                mask[[0, -1]] = False
                norm = np.linalg.norm(derivative[mask])
                checks.append(dict(case=row["case"], source=fingerprint(file), points=int(mask.sum()),
                    f3_negative_points=int((curvature[mask] < 0).sum()),
                    f3_vs_df2_relative_gap=float(np.linalg.norm(curvature[mask] - derivative[mask]) / norm) if norm > 0 else None))
        evidence["joint_inversion"]["coefficient_shape_checks"] = checks
        if (args.joint / "execution_scope.json").exists():
            evidence["joint_inversion"]["execution_scope"] = read(args.joint / "execution_scope.json")
    evidence["data_provenance"] = read(ROOT / "docs/data_manifest.json")
    evidence["metadata_update"] = ("Author confirmed on 2026-10-31 that this loam_all_0.2 case uses "
        "constant theta_s=0.43 and theta_r=0.078. Historical result caveats retain their original wording; "
        "the constants and numerical results are unchanged. Table 1 describes other cases.")
    evidence["source_records"] = [fingerprint(p) for p in sorted(args.libraries.glob("*.json"))]
    save_json(evidence_file, evidence)
    benchmark_link = Path(os.path.relpath(ROOT / "benchmarks/README.md", args.output)).as_posix()
    variants = ["tanh_norm_lbfgs", "silu_norm_lbfgs", "sin_raw_adam"]
    text = f"""# Integration validation record (2026-10-31)

This is a development diagnostic record for the current code, not a claim of reproducing
published accuracy. Numerical values and input and code hashes are in
[validation_evidence.json](validation_evidence.json); the execution method is in
[benchmarks/README.md]({benchmark_link}). Historical runs stay local in `intermediate/`.

## Fixed candidate-library comparison

The three existing network variants all use noiseless loam, 5 sensors and training seed 1.
The comparison uses 21 depths in 0.05–0.25 m at the same time points, 209,979 rows in total,
with no network retraining. These are development caches, not three independent datasets or
independent noise trials. The old caches saved no explicit time coordinates, so time order was
recovered from the original generating script and checked against depth and row count; the new
cache generator saves time coordinates.

The table counts how many of the three network variants selected the correct three-term Richards
structure. Global SGTR fixes lambda=1e-4, epsilon=0.01 with the AIC correction; CGSR fixes
lambda=0.01, epsilon=10^1.2, per-group normalization and window width 0.001; Spline fixes
lambda=0.01. Composite libraries with substituted derivatives are for diagnosis only and need
not come from one integrable field; FD is also a comparator with discretization error.

| Substitution | Global SGTR | CGSR | Spline-SR |
|---|---:|---:|---:|
"""
    for suffix, label in [("", "None"), ("_fd_theta", "Water content only"), ("_fd_time", "Time derivative only"),
                          ("_fd_space", "All spatial derivative terms"),
                          ("_fd_derivatives", "Time and spatial derivatives, keeping DNN water content")]:
        counts = {"global": 0, "group": 0, "spline": 0}
        for variant in variants:
            for s in libraries[variant + suffix]["selections"]:
                chosen = ((s.get("normalization") == "global" and s["value"] == 0.01) or
                          (s.get("normalization") == "group" and s["win_size"] == 0.001 and s["rule"] == "epsilon") or
                          (s["selector"] == "spline" and s["lam"] == 0.01))
                if chosen:
                    counts[s.get("normalization", "spline")] += int(s["score"]["exact"])
        text += f"| {label} | {counts['global']}/3 | {counts['group']}/3 | {counts['spline']}/3 |\n"
    text += """
Spatial-derivative substitution supports an important role in SGTR/CGSR misselection, but not
all failures can be attributed to one source. Spline-SR still fails with FD derivatives when DNN
water content is kept for grouping, which shows that grouping and regularization also need attention.
All three spline lambdas (0.01, 0.001, 0.0001) select correctly in the all-FD comparison.

In the CGSR development sweep, 4%–5% gain recovers the three terms on all three DNN libraries and
three window widths, but 4% misses a term on the all-FD library at window width 0.001. Fixed epsilon
fails at the DNN window width 0.001 and succeeds at 0.005/0.01. These results do not support globally
replacing the default, so `min_energy_gain` stays an experimental option: neither removed nor promoted
to the default. The historical three-sensor noise experiments differ from these five-sensor noiseless
experiments, and their recovery rates cannot be merged directly.

## Fixed-field inversion and representation error

The true three-term structure is given; every fifth time point is used, with consistent FD spatial
derivatives. The oracle response is the reference coefficients times this spatial library; the FD
response comes from numerical time differences. Evaluation uses 200 equally spaced water-content
points in each stated interval.

| Interval | Coefficient form | Direct-fit error f1/f2/f3 | Fixed-FD-field inversion error f1/f2/f3 |
|---|---|---|---|
"""
    for row in inverse["records"]:
        representation = "/".join(f"{v:.4f}" for v in row["representation_errors"].values())
        errors = "/".join(f"{v:.4f}" for v in row["inverse"]["fd_time"]["coefficient_errors"].values())
        text += f"| {row['bounds'][0]}–{row['bounds'][1]} | {row['form']} | {representation} | {errors} |\n"
    text += """
Nonnegative cubic polynomials have clear representation limits on wide intervals. The fixed-field
spline diagnostic reaches lower error, but it is an unconstrained linear fit, not an implemented or
validated spline PINN inversion, and it does not show that network inversion beats polynomials. The
current network default stays for now; comparing them still needs matched budgets, multiple seeds and
noise. This experiment redefines polynomial coordinates per evaluation interval, so it is not the
complete configuration of the old joint inversion.

## Expressiveness of the heterogeneous network

The original code and the compatible default concatenate the two branches and output linearly, so
they can represent only `a(theta)+b(z)`. On the rectangle theta=0.16–0.38, z=0.05–0.25 m, the best
additive projection of the current reference coefficients has these relative errors for the four terms:

"""
    text += ", ".join(f"`{k}`: {v:.4f}" for k, v in inverse["additive_surface_lower_bound"].items()) + ".\n"
    text += """

This is a lower bound for additive approximation on this rectangular grid regardless of network
capacity, not a lower bound on error along observed trajectories. The author confirmed θ_s=0.43 and
θ_r=0.078 as constants over depth in this case; the values used in existing diagnostics are unchanged.
An optional nonlinear `fusion_hidden` layer was added, and its ability to represent interaction passes
unit tests; the default empty list preserves the original method. This extension is not claimed as an
accuracy improvement.

## Short heterogeneous runs

"""
    if pilot:
        text += "Run on the heterogeneous data with 10% noise, 1,000 pretraining steps and 3,000 joint training steps, with the original additive structure.\n\n"
        text += "| Run | Selected terms | Correct structure |\n|---|---|---|\n"
        for row in pilot["records"]:
            text += f"| {Path(row['directory']).name} | {', '.join(row['terms'] or [])} | {row['structure']['exact']} |\n"
    else:
        text += "Results have not yet been written to a complete evaluation record.\n"
    text += """

Short training checks execution and exposes faults; it cannot replace complete training. Insufficient
training can still give a very small data loss with a wrong structure, and extending later fixed-structure
training cannot restore terms that were already deleted.
"""
    if coarse:
        status = read(args.coarse / "status.json")["state"]
        text += f"\n## Three-sensor DNN gain sweep\n\nExecution status: `{status}`; {len(coarse)}/8 data/noise/seed combinations recorded.\n"
        text += """
Matches the DNN conditions of the earlier `diag_energy_gain_wide.py`: sin 6×50, raw inputs,
20,000 Adam steps, 3 sensors, 10 candidate depths, and noise 0/10% × seeds 0/1 for each soil.
After caching, the same SGTR path is rescored. The denominator of each cell counts only
data/noise/seed combinations and does not count window width or threshold settings as independent
trials. This is still a development sweep, not an independent soil test set.

| Soil | Window | Fixed ε | 2% | 3% | 4% | 5% | 6.5% | 8% | 10% | 15% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
        for soil in ("homogeneous", "heterogeneous"):
            rows = [r for r in coarse if r["soil"] == soil]
            if not rows:
                continue
            for window in (0.001, 0.002, 0.005, 0.01):
                counts = [sum(s["score"]["exact"] for r in rows for s in r["selections"]
                              if s["win_size"] == window and s["min_energy_gain"] == gain)
                          for gain in (None, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.15)]
                text += f"| {soil} | {window} | " + " | ".join(f"{n}/{len(rows)}" for n in counts) + " |\n"
        text += """

The effective relative threshold of a fixed ε is about 2ε/(number of groups). The 21-depth cache above
has about 2,000 groups, so ε=10^1.2 is about 1.6%; the 10-depth library here has about 900 groups, which
gives about 3.4%–3.7%. This explains why the penalty strength differs between the two diagnostic settings,
but it is not the only cause of all differences. Window width also changes sample weights, the local
constant-coefficient approximation and the threshold path, which energy normalization cannot remove.

No gain ratio in this test grid covers all soils and window widths. 4%–5% gives 4/4 for both soils at the
small windows 0.001/0.002 but still fails on the homogeneous case at larger windows. `min_energy_gain`
therefore stays an experimental option, and the default ε is unchanged for now. A next round could freeze
small windows and 4%–5% as candidates and validate them on seeds not used in development; the recovery
rates tuned here must not be reported as independent validation. Successful initial selection on a
complete candidate library also does not mean that later joint coefficient training has passed.
"""
    if joint:
        status = read(args.joint / "status.json")["state"]
        manifest = evidence["joint_inversion"]["manifest"]
        planned = 2 * len(manifest["seeds"]) * len(manifest["noise"])
        text += f"\n## Joint inversion on a matched budget\n\nStatus: `{status}`; {len(joint['records'])}/{planned} completed.\n"
        text += "As the author requested, this round compares only the two forms at zero noise and seed 0; the next batch depends on this pair.\n"
        text += f"""
With the true three-term structure given, both forms use the same field-network initialization,
observation noise, collocation points and a budget of {manifest['epoch_budget']:,} training steps.
The table gives coefficient relative L2 errors on theta=0.16–0.38, and clean water-content relative
errors at four sensor depths not used for fitting observations. This is a spatial interpolation check
within one simulation, not prediction from an external source, and no structure discovery runs here.
Complete training curves per seed, narrow-interval coefficient errors and evaluation point counts are
in the JSON evidence. The polynomial uses given signs and nonnegative cubic coefficients, solving bounded
least squares at each step; the network coefficient branch has no sign constraint and is updated by Adam.
The comparison is therefore between the two complete backends, including their constraints and
optimizers. Equal step counts do not mean equal computational cost, and differences cannot be attributed
to the representation alone.

| Noise | Seed | Form | f1 | f2 | f3 | Held-out depth θ | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
"""
        for row in joint["records"]:
            errors = row["coefficients"]["0.16:0.38"]["errors"]
            values = [errors.get(n) for n in ("theta_z", "theta_zz", "theta_z^2")]
            cells = " | ".join("NA" if v is None else f"{v:.4f}" for v in values)
            text += (f"| {row['noise']:.0%} | {row['seed']} | {row['settings']['coefficient_form']} | {cells} | "
                     f"{row['held_sensor_theta_relative_error']:.4f} | {row['elapsed_seconds']:.1f} |\n")
        text += "\nAn additional check of the homogeneous Richards relation f3=df2/dθ, with the derivative from numerical differences of saved curves; it was not a training constraint.\n\n"
        text += "| Case | Relative gap f3 vs df2/dθ | Points with f3<0 / evaluated |\n|---|---:|---:|\n"
        for row in evidence["joint_inversion"]["coefficient_shape_checks"]:
            gap = row["f3_vs_df2_relative_gap"]
            gap_text = "NA" if gap is None else f"{gap:.4f}"
            text += f"| {row['case']} | {gap_text} | {row['f3_negative_points']}/{row['points']} |\n"
        if status == "completed" and manifest["seeds"] == [0] and manifest["noise"] == [0]:
            text += """

On the wide interval in this round, the polynomial has lower f1/f2 and held-out-depth water-content
errors and the network has lower f3 error. On the narrow interval 0.16–0.30, the network has lower f1/f3
and the polynomial lower f2. Neither form wins for all coefficients and intervals. Both have water-content
relative errors below 1%, yet wide-interval coefficient errors remain about 27%–69%, which shows that a
good field fit is not enough to prove accurate recovery of equation coefficients.

As a next step, first pair the network and polynomial under fixed fields and derivatives, with consistent
water-content intervals and evaluation grids, to separate representation and optimization limits from
coupling in joint learning; then decide on independent seed and noise batches. The existing fixed-field
spline diagnostic cannot replace this comparison for the network coefficient backend. f3=df2/dθ is a
candidate for later constrained inversion, but it is a new physical constraint and needs separate
validation. This round keeps the existing defaults and starts none of this follow-up training; 20,000
steps is the end of a budget, not a proof of convergence.
"""
    text += """

## Release and remaining work

All failed results are retained. The next step uses this round's single-seed noiseless comparison to decide
on field-reconstruction improvements and later independent seed and noise validation; this round does not
extend the training queue automatically. Complete neural training configurations and the new fusion layer
have not yet passed scientific acceptance.

The author confirmed that Ks holds nodal values and that the final node may be missing; padding with the
last value does not affect the upper diagnostic interval. The author also confirmed theta_s=0.43 and
theta_r=0.078 as constants over depth in this case; the spatial standard deviations in the paper's table
belong to other cases. The code license, the public license of the current local data and the target GitHub
repository were undecided at the time, and nothing was uploaded to an external repository in this round.
"""
    report_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
