"""Separate coefficient representation from fixed-field inverse identifiability.

Oracle response uses known coefficients on FD spatial terms. FD response uses
the numerical time derivative. Neither experiment learns the water-content field.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from common import ROOT, TRUE_TERMS, environment, fingerprint, normalized_condition, relative_l2
from deepgs.data import load_profile_txt
from deepgs.library import finite_difference_library
from deepgs.physics import SOILS, load_heterogeneous_soil
from deepgs.sparse import spline_basis, spline_knots
from deepgs.utils import save_json


def fit(matrix, response, nonnegative):
    norms = np.linalg.norm(matrix, axis=0)
    scale = np.where(norms > 0, norms, 1)
    q, r = np.linalg.qr(matrix / scale, mode="reduced")
    if nonnegative:
        solved = lsq_linear(r, q.T @ response, bounds=(0, np.inf), method="bvls", tol=1e-12, max_iter=1000)
        if not solved.success:
            raise RuntimeError(solved.message)
        value = solved.x
    else:
        value = np.linalg.lstsq(r, q.T @ response, rcond=None)[0]
    return value / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "intermediate/runs/fixed_inversion")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    full = load_profile_txt(ROOT / "intermediate/data/th.txt")
    observed = full.select_depths(full.depths[3:28])
    library = finite_difference_library(observed.theta.T, observed.depths, observed.times, 3)
    rows = np.arange(len(library.theta)).reshape(21, -1)[:, ::5].ravel()
    theta = library.theta[rows, 0]
    spatial = library.terms[rows][:, [library.names.index(n) for n in TRUE_TERMS]]
    ut = library.theta_t[rows, 0]
    records = []
    for bounds in ((0.16, 0.38), (0.16, 0.30)):
        mask = (theta >= bounds[0]) & (theta <= bounds[1])
        state, terms, response = theta[mask], spatial[mask], ut[mask]
        grid = np.linspace(*bounds, 200)
        truth = np.stack(list(SOILS["loam"].coefficients(grid).values()))
        actual = np.stack(list(SOILS["loam"].coefficients(state).values()), axis=1)
        oracle = np.sum(actual * terms, axis=1)
        for form in ("nonnegative_cubic", "unconstrained_cubic", "spline"):
            signs = np.array([-1, 1, 1]) if form == "nonnegative_cubic" else np.ones(3)
            if form == "spline":
                knots = spline_knots(grid, 6, 3)
                basis, evaluation = spline_basis(state, knots, 3), spline_basis(grid, knots, 3)
            else:
                basis = np.column_stack([((state - bounds[0]) / (bounds[1] - bounds[0]))**k for k in range(4)])
                evaluation = np.column_stack([((grid - bounds[0]) / (bounds[1] - bounds[0]))**k for k in range(4)])
            constrained = form == "nonnegative_cubic"
            controls = [fit(evaluation * signs[j], truth[j], constrained) for j in range(3)]
            curves = np.stack([signs[j] * evaluation @ controls[j] for j in range(3)])
            representation = {n: relative_l2(curves[j], truth[j]) for j, n in enumerate(TRUE_TERMS)}
            design = np.hstack([signs[j] * terms[:, j:j+1] * basis for j in range(3)])
            inverse = {}
            for label, y in (("oracle", oracle), ("fd_time", response)):
                controls = fit(design, y, constrained).reshape(3, -1)
                curves = (controls @ evaluation.T) * signs[:, None]
                inverse[label] = dict(relative_residual=relative_l2(design @ controls.ravel(), y),
                                      coefficient_errors={n: relative_l2(curves[j], truth[j]) for j, n in enumerate(TRUE_TERMS)})
            records.append(dict(bounds=list(bounds), rows=len(state), form=form, representation_errors=representation,
                                condition=normalized_condition(design), inverse=inverse))
    # Least-squares additive surface is the best unrestricted a(theta)+b(z) on this rectangular grid.
    soil = load_heterogeneous_soil(ROOT / "intermediate/data/loam_all_0.2", theta_s=0.43, theta_r=0.078)
    state, depth = np.meshgrid(np.linspace(0.16, 0.38, 100), np.linspace(0.05, 0.25, 21))
    additive = {}
    for name, surface in soil.coefficients(state, depth).items():
        projection = surface.mean(axis=0, keepdims=True) + surface.mean(axis=1, keepdims=True) - surface.mean()
        additive[name] = relative_l2(projection, surface)
    save_json(args.output / "results.json", dict(records=records, additive_surface_lower_bound=additive,
              caveat="Additive error is on a rectangular theta-z domain, not the observed trajectory; parameter provenance in docs/data_manifest.json."))
    save_json(args.output / "manifest.json", dict(kind="diagnostic_with_known_structure_and_reference", environment=environment(),
              inputs=[fingerprint(ROOT / "intermediate/data/th.txt"), *[fingerprint(p) for p in sorted((ROOT / "intermediate/data/loam_all_0.2").glob("*.npy"))]],
              time_stride=5, evaluation="200 equally spaced theta values within each closed interval",
              code=[fingerprint(Path(__file__)), fingerprint(ROOT / "benchmarks/common.py")] +
                   [fingerprint(p) for p in sorted((ROOT / "src/deepgs").glob("*.py"))]))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
