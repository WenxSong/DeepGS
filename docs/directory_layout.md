# Directory layout and local archive

The repository root holds the maintained source of the tool. Raw research data, experiment outputs, build products, old reports and planning records are all kept in `intermediate/`, which Git ignores.

```text
DeepGS Toolkit/
├── src/deepgs/          # Package code and configuration templates installed with the wheel
├── configs/             # Runnable example configurations
├── examples/            # Independent demonstration data generators
├── data/                # Publishable data specification, synthetic demonstration, real-data template
│   ├── synthetic/       # Synthetic data; each case has its own metadata.yaml
│   └── real/            # Real observations: six local soil-column cases, public instructions and template
├── benchmarks/          # Reusable diagnostic and evaluation scripts
├── tests/               # Test code
├── docs/                # Guides, algorithm notes, release checks and data provenance metadata
├── .github/workflows/   # CI configuration
├── README.md
├── pyproject.toml
├── MANIFEST.in
├── LICENSE
├── CITATION.cff
└── intermediate/        # Local materials; never in Git or distributions
    ├── data/            # Original simulated data/, including two th.txt files, parameter arrays and metadata
    ├── data_imports/    # Newly imported raw experimental CSVs, archived byte for byte
    ├── runs/            # Original runs/ plus later run outputs
    ├── build/           # Temporary build files from the original build/
    ├── build_metadata/  # Original src/*.egg-info
    ├── cache/           # Existing __pycache__ and similar caches
    ├── releases/        # Packages; old packages archived by phase, new ones in current/
    ├── history/         # Old diagnostic reports, original numerical evidence, P0/P1 phase notes
    ├── planning/        # Framework design and implementation plans
    ├── references/      # Original code and paper materials, in their original structure
    ├── logs/            # Logs
    ├── backups/         # Backups taken before modifications
    ├── progress.md      # Current progress and historical work record
    └── layout_migration_20261031.json  # Old and new paths with file hashes
```

## Outputs and builds

The CLI and API write to `intermediate/runs/<config name>/` by default. An explicit `--output` can select another folder, and examples in this project all use `intermediate/runs/`.

To build distributions, run `python -m build --outdir intermediate/releases/current`. Installation tooling may leave temporary `egg-info` and build caches, which are build materials that can be archived.

## Data paths

Older data now lives in `intermediate/data/`. Legacy configurations use the project root as the working directory. Relative data paths in schema-2 configurations are based on the configuration's folder, so examples in `configs/` use `../intermediate/data/`. Public sources and hashes are in [data provenance](data_provenance.md) and the [data manifest](data_manifest.json).

## Preserved records

Manifests, configuration snapshots, numerical evidence and old packages from saved experiments keep their original bytes. They may record the original `data/`, `runs/` or `dist/` paths. Find the new locations through the migration list, and do not rewrite these records to look like new executions. Resume checks compare sources and configurations, so moved old results count only as historical records; a re-execution should use a new output folder.

## What is distributed

Raw research data does not ship with the code package. From `data/`, only the synthetic demonstration and standard templates explicitly listed in the manifest are packaged. Without the local research data, the corresponding legacy data tests are explicitly skipped, while the independent manufactured-solution and synthetic-data engineering checks still run.
