"""Version-2 selection artifacts, separate from legacy output formats."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass

from .utils import save_json


def finite_json(value):
    """Convert numerical diagnostics to strict JSON without non-finite literals."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_selection(output, result, library, save_library=False):
    folder = Path(output) / "selection"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "structure.json", finite_json(result.summary()))
    if result.coarse is not None:
        coarse = result.coarse
        values = dict(coefficients=coarse.coefficients[coarse.active], theta_centers=coarse.theta_centers,
                      active=np.asarray(coarse.active), terms=np.asarray(coarse.names))
        if coarse.depth_centers is not None:
            values["depth_centers"] = coarse.depth_centers
        if coarse.spline is not None:
            values.update(coarse.spline)
            values["control_points"] = coarse.spline["control_points"][coarse.active]
        np.savez_compressed(folder / "coarse_coefficients.npz", **values)
    if save_library:
        raw = library.library
        np.savez_compressed(folder / "library.npz", theta=raw.theta, terms=raw.terms, theta_t=raw.theta_t,
                            names=np.asarray(raw.names), sample_z=library.sample_z, sample_t=library.sample_t,
                            units=json.dumps(library.units), fingerprint=library.fingerprint)


def load_library(path):
    """Load and verify a v2 candidate cache, including explicit row coordinates."""
    from .contracts import LibraryArtifact
    from .library import CandidateLibrary

    with np.load(path, allow_pickle=False) as data:
        raw = CandidateLibrary(data["theta"], data["terms"], data["theta_t"], data["names"].tolist(),
                               data["sample_z"].reshape(-1, 1), data["sample_t"].reshape(-1, 1))
        result = LibraryArtifact(raw, data["sample_z"], data["sample_t"], json.loads(str(data["units"])))
        if result.fingerprint != str(data["fingerprint"]):
            raise ValueError("Library cache fingerprint mismatch")
        return result


def _predict_in_batches(field, observations):
    parameter = next(field.parameters())
    points = observations.points()
    values = []
    with torch.no_grad():
        for start in range(0, len(points), 20000):
            batch = torch.as_tensor(points[start:start + 20000], dtype=parameter.dtype, device=parameter.device)
            values.append(field(batch[:, :1], batch[:, 1:]).cpu().numpy().ravel())
    return np.concatenate(values).reshape(observations.theta.shape)


def save_inversion(output, result, observations, selection):
    """Store final coefficients separately from stage-one coarse estimates."""
    if result.status != "completed":
        return
    selection.assert_unchanged()
    if result.terms != selection.terms:
        raise ValueError("Inversion changed the frozen term order")
    folder = Path(output) / "inversion"
    folder.mkdir(parents=True, exist_ok=True)
    fitted = _predict_in_batches(result.field, observations)
    np.savez_compressed(folder / "fields.npz", times=observations.times, depths=observations.depths, theta=fitted,
                        observed_theta=observations.theta)
    model = result.coefficients
    parameter = next(result.field.parameters())
    if model.spec["soil"] == "homogeneous":
        low, high = model.spec["theta_bounds"]
        grid = np.linspace(low, high, 200)
        with torch.no_grad():
            curves = model(torch.as_tensor(grid[:, None], dtype=parameter.dtype, device=parameter.device)).cpu().numpy().T
        np.savez_compressed(folder / "coefficients.npz", theta_grid=grid, curves=curves, terms=np.asarray(result.terms),
                            coordinate_kind="theta_curve")
    else:
        z = np.broadcast_to(observations.depths, fitted.shape).ravel()
        theta = fitted.ravel()
        chunks = []
        with torch.no_grad():
            for start in range(0, theta.size, 20000):
                a = torch.as_tensor(theta[start:start + 20000, None], dtype=parameter.dtype, device=parameter.device)
                b = torch.as_tensor(z[start:start + 20000, None], dtype=parameter.dtype, device=parameter.device)
                chunks.append(model(a, b).cpu().numpy())
        np.savez_compressed(folder / "coefficients.npz", theta=theta, z=z,
                            coefficient_fields=np.concatenate(chunks).T, terms=np.asarray(result.terms),
                            coordinate_kind="trajectory_pairs")
    save_json(folder / "history.json", finite_json(list(result.history)))
    from .contracts import CANONICAL_UNITS
    checkpoint = dict(format_version=1, schema_version=2, terms=list(result.terms), units=CANONICAL_UNITS,
                      structure_fingerprint=selection.library_fingerprint,
                      structure_source="supplied" if selection.library_fingerprint.startswith("supplied:") else "selected",
                      dtype=str(parameter.dtype).removeprefix("torch."),
                      field_spec=finite_json(result.metadata["field_spec"]), coefficient_spec=finite_json(model.spec),
                      field_state=result.field.state_dict(), coefficient_state=model.state_dict())
    torch.save(checkpoint, folder / "model.pt")


@dataclass(frozen=True)
class LoadedModel:
    field: object
    coefficients: object
    terms: tuple
    units: dict
    structure_source: str

    def predict(self, z, t):
        z, t = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(t, dtype=float))
        if not np.isfinite(z).all() or not np.isfinite(t).all():
            raise ValueError("Query coordinates must be finite")
        p = next(self.field.parameters())
        with torch.no_grad():
            a, b = [torch.as_tensor(v.reshape(-1, 1), dtype=p.dtype, device=p.device) for v in (z, t)]
            return self.field(a, b).cpu().numpy().reshape(z.shape)

    def evaluate_coefficients(self, theta, z=None):
        theta = np.asarray(theta, dtype=float)
        p = next(self.field.parameters())
        query_dtype = np.float32 if p.dtype == torch.float32 else np.float64
        theta = theta.astype(query_dtype)
        low, high = np.asarray(self.coefficients.spec["theta_bounds"], dtype=query_dtype)
        if not np.isfinite(theta).all() or np.any(theta < low) or np.any(theta > high):
            raise ValueError("Coefficient query is outside the saved theta domain")
        if self.coefficients.spec["soil"] == "heterogeneous":
            if z is None:
                raise ValueError("Heterogeneous coefficients require depth")
            theta, z = np.broadcast_arrays(theta, np.asarray(z, dtype=float))
            if not np.isfinite(z).all():
                raise ValueError("Depth query must be finite")
        with torch.no_grad():
            a = torch.as_tensor(theta.reshape(-1, 1), dtype=p.dtype, device=p.device)
            b = None if z is None else torch.as_tensor(np.asarray(z).reshape(-1, 1), dtype=p.dtype, device=p.device)
            return self.coefficients(a, b).cpu().numpy()


def load_model(path, device="cpu"):
    """Reconstruct field and coefficients from a versioned tensor-only checkpoint."""
    from .coefficients import coefficient_from_spec
    from .networks import FieldNetwork
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("schema_version") != 2:
        raise ValueError("Unsupported model checkpoint format")
    dtype_name = checkpoint["dtype"]
    if dtype_name not in ("float32", "float64"):
        raise ValueError("Unsupported checkpoint dtype")
    dtype = getattr(torch, dtype_name)
    spec = checkpoint["field_spec"]
    field = FieldNetwork(spec["hidden_sizes"], spec["activation"], spec["bounds"], spec["output_bounds"],
                         xavier=spec.get("xavier", False), bounded_lerp=spec.get("bounded_lerp", False)).to(device=device, dtype=dtype)
    coefficients = coefficient_from_spec(checkpoint["coefficient_spec"]).to(device=device, dtype=dtype)
    if tuple(checkpoint["terms"]) != coefficients.terms:
        raise ValueError("Checkpoint coefficient terms disagree")
    field.load_state_dict(checkpoint["field_state"], strict=True)
    coefficients.load_state_dict(checkpoint["coefficient_state"], strict=True)
    field.eval()
    coefficients.eval()
    return LoadedModel(field, coefficients, coefficients.terms, checkpoint["units"], checkpoint["structure_source"])
