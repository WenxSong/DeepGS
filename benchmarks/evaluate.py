"""Score saved outputs without treating successful execution as scientific validation."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, TRUE_TERMS, relative_l2, support_score
from deepgs.physics import SOILS, load_heterogeneous_soil
from deepgs.utils import save_json


def _score_v2(folder, manifest, bounds, reference_soil):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    structure_file = folder / "selection/structure.json"
    structure = json.loads(structure_file.read_text(encoding="utf-8")) if structure_file.exists() else {}
    names = structure.get("terms")
    result = dict(directory=str(folder), schema_version=2, execution_status=manifest.get("status"),
                  terms=names, selection_status=structure.get("status"),
                  inversion_status=summary.get("inversion", {}).get("status"),
                  fit_metrics=summary.get("inversion", {}).get("metrics"),
                  structure=dict(status="not_evaluated", reason="No reference structure supplied"),
                  coefficient_interval=list(bounds), coefficient_errors={}, evaluation_points={})
    if reference_soil is None:
        return result
    if reference_soil != "loam" or manifest["configuration"]["problem"]["soil"] != "homogeneous":
        raise ValueError("This explicit reference evaluator supports homogeneous loam only")
    result["structure"] = support_score(names, TRUE_TERMS)
    result["reference_soil"] = reference_soil
    path = folder / "inversion/coefficients.npz"
    if path.exists():
        with np.load(path) as saved:
            grid = saved["theta_grid"]
            for name, curve in zip(saved["terms"].tolist(), saved["curves"]):
                if name not in TRUE_TERMS:
                    continue
                valid = (grid >= bounds[0]) & (grid <= bounds[1]) & np.isfinite(curve)
                result["evaluation_points"][name] = dict(valid=int(valid.sum()), total=int(grid.size))
                if valid.any():
                    result["coefficient_errors"][name] = relative_l2(curve[valid], SOILS[reference_soil].coefficients(grid[valid])[name])
    return result


def score_run(folder, heterogeneous=False, bounds=(0.16, 0.38), reference_soil=None):
    folder = Path(folder)
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    if manifest.get("schema_version") == 2:
        return _score_v2(folder, manifest, bounds, reference_soil)
    expected = ["1", *TRUE_TERMS] if heterogeneous else TRUE_TERMS
    result = dict(directory=str(folder), execution_status=manifest.get("status", "legacy_unknown"),
                  coefficient_interval=list(bounds), coefficient_errors={}, evaluation_points={})
    structure_file = folder / "structure.json"
    names = json.loads(structure_file.read_text(encoding="utf-8"))["terms"] if structure_file.exists() else None
    result.update(terms=names, structure=support_score(names, expected))
    if not structure_file.exists() and not (folder / "summary.json").exists():
        result["structure"]["status"] = "missing_structure"
    if heterogeneous:
        file = folder / "fields.npz"
        ref = manifest.get("configuration", {}).get("reference", {})
        if file.exists() and ref.get("path"):
            soil = load_heterogeneous_soil(ref["path"], ref.get("theta_s"), ref.get("theta_r"), ref.get("depth_max", 1))
            with np.load(file) as data:
                theta = data["theta"]
                depth = np.broadcast_to(data["depths"], theta.shape)
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    coefficients = soil.coefficients(theta, depth)
                finite_reference = np.logical_and.reduce([np.isfinite(c) for c in coefficients.values()])
                result["reference_domain_points"] = dict(total=int(theta.size),
                    finite_all_reference_terms=int(finite_reference.sum()),
                    finite_in_interval=int((finite_reference & (theta >= bounds[0]) & (theta <= bounds[1])).sum()))
                for name, curve in zip(data["terms"].tolist(), data["coefficient_fields"]):
                    if name not in coefficients:
                        continue
                    valid = ((theta >= bounds[0]) & (theta <= bounds[1]) & np.isfinite(coefficients[name]) & np.isfinite(curve))
                    result["evaluation_points"][name] = dict(valid=int(valid.sum()), total=int(valid.size))
                    if valid.any():
                        result["coefficient_errors"][name] = relative_l2(curve[valid], coefficients[name][valid])
            result["reference_status"] = "configured reference; coefficients evaluated at predicted theta on saved trajectories; parameter provenance in docs/data_manifest.json"
    else:
        curves = {}
        for file in (folder / "coefficient_curves.npz", folder / "inversion.npz"):
            if not file.exists():
                continue
            with np.load(file) as data:
                if "curves" in data:
                    grid = data["theta_grid"] if "theta_grid" in data else data["theta"]
                    curves.update({str(n): (grid.copy(), c.copy()) for n, c in zip(data["terms"], data["curves"])})
                else:
                    curves.update({n: (data[f"{n}/theta"], data[f"{n}/coefficient"]) for n in names or []
                                   if f"{n}/theta" in data})
        for name, (grid, curve) in curves.items():
            if name not in TRUE_TERMS:
                continue
            valid = (grid >= bounds[0]) & (grid <= bounds[1]) & np.isfinite(curve)
            result["evaluation_points"][name] = dict(valid=int(valid.sum()), total=int(valid.size))
            if valid.any():
                result["coefficient_errors"][name] = relative_l2(curve[valid], SOILS["loam"].coefficients(grid[valid])[name])
        if "theta_zz" in curves and "theta_z^2" in curves:
            grid, diffusion = curves["theta_zz"]
            other_grid, derivative = curves["theta_z^2"]
            if len(grid) > 2 and np.array_equal(grid, other_grid):
                valid = (grid > bounds[0]) & (grid < bounds[1])
                valid[[0, -1]] = False
                if valid.any():
                    result["f3_vs_df2_relative_gap"] = relative_l2(derivative[valid], np.gradient(diffusion, grid)[valid])
    for filename in ("training.json", "summary.json"):
        if (folder / filename).exists():
            extra = json.loads((folder / filename).read_text(encoding="utf-8"))
            for key in ("selections", "theta_rmse", "status", "coefficient_form"):
                if key in extra:
                    result[key] = extra[key]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--reference-soil", choices=["loam"], help="explicit homogeneous reference for schema-2 results")
    args = parser.parse_args()
    result = score_run(args.folder, args.heterogeneous, reference_soil=args.reference_soil)
    save_json(args.folder / "evaluation.json", result)
    print(json.dumps(result, indent=2))
