"""Public entry points for the version-2 workflow."""


def run(config, *, device=None, output=None):
    """Run the two-stage workflow from a configuration path, mapping or dataclass."""
    from .workflow import run as execute
    return execute(config, device=device, output=output)


def select(library, config, soil="homogeneous"):
    """Select a structure from an existing candidate library."""
    from .selection import select as execute
    return execute(library, config, soil)


def invert(observations, selection, config, *, soil="homogeneous", device="cpu"):
    """Invert an explicitly fixed support; reference equations are not inferred."""
    from dataclasses import asdict
    from .experiment import load_experiment, InversionConfig
    from .contracts import SelectionResult
    from .datasets import validate_observations
    from .fixed_inversion import invert as execute
    raw = asdict(config) if isinstance(config, InversionConfig) else config
    selector = "spatial_group" if soil == "heterogeneous" else "group"
    # Reuse schema-2 validation by wrapping the inversion section in a minimal experiment.
    validated = load_experiment(dict(schema_version=2, data=dict(path="__provided_observations__"),
                                     library=dict(max_order=4), problem=dict(soil=soil),
                                     structure=dict(selector=selector), inversion=raw)).inversion
    validate_observations(observations)
    if not isinstance(selection, SelectionResult):
        selection = SelectionResult.supplied(selection)
    return execute(observations, selection, validated, soil=soil, device=device)


def load_model(path, device="cpu"):
    """Reload a saved inversion model for coefficient evaluation and prediction."""
    from .artifacts import load_model as execute
    return execute(path, device=device)
