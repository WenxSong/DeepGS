"""Create editable configurations from resources bundled with an installed wheel."""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path

import yaml

from .capabilities import BUDGETS, validate_combination


def _resource(name: str) -> dict:
    content = resources.files("deepgs").joinpath("templates", name).read_text(encoding="utf-8")
    return yaml.safe_load(content)


def _merge(target: dict, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def build_template(selector: str, coefficient_form: str = "network", *, soil: str | None = None,
                   budget: str = "quickstart", selection_only: bool = False) -> dict:
    """Return a fresh config; data remains the explicit YOUR_DATA.npz placeholder."""
    capability = validate_combination(selector, coefficient_form, soil)
    if budget not in BUDGETS:
        raise ValueError(f"Unknown budget {budget!r}; choose from {', '.join(BUDGETS)}")
    raw = _resource("base.yaml")
    _merge(raw, _resource(f"{budget}.yaml"))
    raw["problem"]["soil"] = capability.soil
    raw["structure"]["selector"] = selector
    if selector == "spatial_group":
        raw["structure"]["grouping"]["normalization"] = "group"
        raw["structure"]["sgtr"].update(lam=.01, epsilon=10**1.2, corrected=False)
    if selector == "spline":
        raw["library"]["max_order"] = 4
    raw["inversion"]["enabled"] = not selection_only
    raw["inversion"]["coefficient_form"] = coefficient_form
    _merge(raw["inversion"], _resource(f"{coefficient_form}.yaml"))
    if coefficient_form == "network" and budget == "quickstart":
        raw["inversion"]["network"].update(hidden_sizes=[10], theta_hidden=[10], depth_hidden=[5], fusion_hidden=[10])
    # Validate without persisting the loader's resolved placeholder path.
    from .experiment import load_experiment
    load_experiment(raw)
    return raw


def write_template(output: str | Path, selector: str, coefficient_form: str = "network", *,
                   soil: str | None = None, budget: str = "quickstart", selection_only: bool = False) -> Path:
    """Write a config exclusively; an existing file is never overwritten."""
    raw = build_template(selector, coefficient_form, soil=soil, budget=budget, selection_only=selection_only)
    path = Path(output)
    header = ["# DeepGS Toolkit configuration (schema 2).",
              "# Replace YOUR_DATA.npz with your observations; no data are generated or downloaded.",
              "# NPZ keys: times (nt,), depths (nz,), theta (nt,nz).",
              "# Relative paths are resolved from this configuration's directory."]
    if budget == "quickstart":
        header += ["# QUICKSTART: tiny training budgets check execution only; accuracy is not validated."]
    else:
        header += ["# FULL: 20,000 field steps and 20,000 inversion steps; runtime depends on data/hardware.",
                   "# These are documented starting settings, not tuned or guaranteed-converged defaults."]
    if selection_only:
        header += ["# Selection only: the inversion settings below are inactive."]
    if coefficient_form == "polynomial":
        header += ["# POLYNOMIAL PRIOR: the signs below are explicit Richards-equation assumptions.",
                   "# Review them for your problem. Any selected term without a sign blocks inversion."]
    if coefficient_form == "bspline":
        header += ["# B-spline coefficient inversion is a new experimental backend."]
    if raw["problem"]["soil"] == "heterogeneous":
        header += ["# The additive network is a(theta)+b(z); general interactions require architecture: fusion."]
    text = "\n".join(header) + "\n" + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return path
