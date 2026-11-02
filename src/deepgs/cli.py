"""Configure and run DeepGS structure selection and fixed-structure inversion."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .pipelines import RUNNERS
from .provenance import run_manifest
from .utils import resolve_device, save_json


def legacy_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="defaults to intermediate/runs/<config name>")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to cuda when available")
    args = parser.parse_args(argv)

    name, config = load_config(args.config)
    output = args.output or Path("intermediate") / "runs" / args.config.stem
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. Choose a new --output to preserve previous results.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output / "config.yaml")
    device = resolve_device(args.device)
    manifest = run_manifest(name, config, device)
    manifest["workflow"] = "legacy"
    save_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        RUNNERS[name](config, output, str(device))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / "manifest.json", manifest)
    print(f"Results saved to {output}")


def _build_parser():
    from .capabilities import BUDGETS, COEFFICIENT_FORMS, SELECTORS, SOIL_TYPES
    parser = argparse.ArgumentParser(description="DeepGS: structure selection followed by fixed-structure PINN inversion")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="list supported selectors and coefficient representations")
    validate = commands.add_parser("validate-data", help="validate a standard dataset folder before training")
    validate.add_argument("directory", type=Path)
    initialize = commands.add_parser("init", help="write an editable schema-2 configuration; does not create data")
    initialize.add_argument("--selector", required=True, choices=SELECTORS)
    initialize.add_argument("--coefficient-form", default="network", choices=COEFFICIENT_FORMS)
    initialize.add_argument("--soil", choices=SOIL_TYPES, default=None, help="derived from the selector when omitted")
    initialize.add_argument("--budget", choices=BUDGETS, default="quickstart")
    initialize.add_argument("--selection-only", action="store_true", help="disable coefficient inversion")
    initialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="run a schema_version: 2 workflow")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--device", default=None)
    commands.add_parser("legacy", help="run a historical pipeline configuration")
    return parser


def _methods_command(parser, args):
    from .capabilities import format_capabilities
    print(format_capabilities())


def _validate_data_command(parser, args):
    import yaml
    from .datasets import validate_dataset
    try:
        report = validate_dataset(args.directory)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


def _init_command(parser, args):
    from .template_api import write_template
    try:
        path = write_template(args.output, args.selector, args.coefficient_form, soil=args.soil,
                              budget=args.budget, selection_only=args.selection_only)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Configuration saved to {path}")
    print("Set data.path to your observations before running. Relative paths follow the configuration directory.")


def _run_command(parser, args):
    from .api import run as execute
    try:
        result = execute(args.config, output=args.output, device=args.device)
    except (ValueError, OSError, NotImplementedError) as error:
        parser.error(str(error))
    print(f"Selection {result.selection.status}: {list(result.selection.terms)}")
    print(f"Inversion {result.inversion.status}")
    print(f"Results saved to {result.artifacts['directory']}")
    if result.execution_status != "completed":
        raise SystemExit(1)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "legacy":
        legacy_main(argv[1:])
        return
    if argv and not argv[0].startswith("-") and Path(argv[0]).suffix.lower() in (".yaml", ".yml"):
        warnings.warn("Bare configuration invocation is a legacy workflow. Use 'deepgs legacy CONFIG' or the v2 'deepgs run CONFIG'.", FutureWarning)
        legacy_main(argv)
        return
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {"methods": _methods_command, "validate-data": _validate_data_command,
                "init": _init_command, "run": _run_command}
    handlers[args.command](parser, args)


if __name__ == "__main__":
    main()
