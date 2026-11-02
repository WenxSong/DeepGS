"""Public combinations supported by the two-stage workflow."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    selector: str
    soil: str
    coefficient_forms: tuple[str, ...]
    description: str


CAPABILITIES = (
    Capability("group", "homogeneous", ("network", "polynomial", "bspline"),
               "Water-content grouped sparse regression"),
    Capability("spline", "homogeneous", ("network", "polynomial", "bspline"),
               "Continuous spline sparse structure selection"),
    Capability("spatial_group", "heterogeneous", ("network",),
               "Depth and water-content grouped sparse regression"),
)
SELECTORS = tuple(item.selector for item in CAPABILITIES)
COEFFICIENT_FORMS = ("network", "polynomial", "bspline")
SOIL_TYPES = ("homogeneous", "heterogeneous")
BUDGETS = ("quickstart", "full")


def capability_for(selector: str) -> Capability:
    for item in CAPABILITIES:
        if item.selector == selector:
            return item
    raise ValueError(f"Unknown structure selector {selector!r}; choose from {', '.join(SELECTORS)}")


def validate_combination(selector: str, coefficient_form: str, soil: str | None = None) -> Capability:
    item = capability_for(selector)
    if soil is not None and soil != item.soil:
        raise ValueError(f"Selector {selector!r} requires soil={item.soil!r}, got {soil!r}")
    if coefficient_form not in item.coefficient_forms:
        raise ValueError(f"Selector {selector!r} supports coefficient forms: {', '.join(item.coefficient_forms)}")
    return item


def format_capabilities() -> str:
    lines = ["DeepGS Toolkit: structure selection -> fixed-structure PINN inversion",
             "selector       soil           coefficient forms",
             "-------------- -------------- ------------------------------"]
    lines += [f"{item.selector:<14} {item.soil:<14} {', '.join(item.coefficient_forms)}" for item in CAPABILITIES]
    lines += ["", "Use --selection-only with init to omit coefficient inversion.",
              "Polynomial inversion needs explicit coefficient signs; B-spline inversion is experimental.",
              "Quickstart budgets check execution; they do not establish recovery accuracy."]
    return "\n".join(lines)
