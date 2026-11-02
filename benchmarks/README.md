# Reproducible validation

Run from the repository root after `pip install -e ".[dev]"`. Each command writes
to a separate directory. Keep one GPU training job active at a time.

Local datasets belong in `intermediate/data/`; generated runs, caches and
checkpoints belong in `intermediate/runs/`. These local directories are excluded
from publication. Dataset metadata is recorded in
[docs/data_manifest.json](../docs/data_manifest.json). Historical records retain
their original embedded paths as provenance; those strings do not define the
current directory layout.

## Execution levels

```bash
python benchmarks/run_suite.py --preset smoke --device cpu --output intermediate/runs/smoke
python benchmarks/run_suite.py --preset pilot --device cuda --seeds 0 1 2 --output intermediate/runs/pilot
python benchmarks/run_suite.py --preset full --device cuda --seeds 0 1 2 --output intermediate/runs/full
```

`smoke` checks CLI wiring with tiny networks and training budgets. `pilot` uses
1,000 field steps and reduced training schedules; failures are diagnostic, not
evidence about a converged method. `full` uses the shipped schedules, which can
take hours per case. Select individual cases with `--cases heterogeneous` etc.
The default timeout is six hours **per case**, configurable with `--timeout`.
Completed runs resume only with matching source, configuration and input hashes.
Incomplete outputs are preserved; choose another suite directory to retry them.

Each run saves the input YAML, expanded defaults, source/data hashes, environment,
elapsed time and execution outcome. The suite writes `evaluation.json` separately
from execution status. A wrong or empty selected structure is reported as such;
successful execution is never an accuracy pass. A killed process may retain
`running` in its run manifest; the suite's `status.json` records timeouts.

## Fixed-library diagnosis

```bash
python benchmarks/generate_library_cache.py --device cuda --output intermediate/runs/library_cache
python benchmarks/fixed_library.py --cache-dir intermediate/runs/library_cache --output intermediate/runs/fixed_library
python benchmarks/fixed_inversion.py --output intermediate/runs/fixed_inversion
```

The three-sensor DNN gain scan from the earlier progress record is available as
`python benchmarks/coarse_sweep.py --device cuda --output intermediate/runs/coarse_sweep`.
It uses two soils, noise 0/10%, seeds 0/1, 20,000 field-training steps and four
grouping windows. It caches fields/libraries and reuses SGTR paths. Run it
sequentially with other GPU commands, not alongside them.

For matched known-structure joint inversion:

```bash
python benchmarks/inversion_suite.py --device cuda --output intermediate/runs/inversion_comparison
```

This runs both coefficient forms for noise 0/5% and seeds 0/1/2 at 20,000 epochs.
The author's first-batch scope on 2026-10-31 is only the noiseless seed-0 pair:
`python benchmarks/inversion_suite.py --device cuda --seeds 0 --noise 0 --output intermediate/runs/inversion_pair`.
Further seeds/noise are deferred until that pair is reviewed.
Field initialization, training observations and collocation sampling are paired.
The polynomial backend uses supplied coefficient signs and nonnegative cubic
controls solved by bounded least squares; the network backend has unrestricted
coefficient branches trained by Adam. This compares the implemented backends,
including their different constraints and optimizers, not representation alone.
It also scores four withheld sensor depths; this is spatial interpolation within
the same simulation, not source-external prediction. `--epochs 2 --seeds 0 --noise 0`
is only an execution check. Do not compare its errors as converged performance.

Cache generation is the expensive step. Fixed-library diagnosis performs no
training. The generator records explicit time coordinates; the older caches in
`intermediate/diag` omit time, whose alignment is inferred from their generating
script and checked against row count/depth. That historical provenance limit is
retained in the diagnostic manifest. `fixed_library.py` currently scores the
clean seed-1 loam comparison only; other seeds/noises require a separately
specified comparison protocol, not renaming caches to make them appear identical.

The fixed-library experiment uses 21 depths from 0.05 to 0.25 m at every available
time after the first (209,979 rows for the bundled data), with identical rows in
all substitutions. `--time-stride` changes that protocol and is recorded.
Counterfactual substitutions need not form derivatives of one consistent field.
Finite differences of the clean simulation are numerical comparators, not exact
derivatives. Sparse-path scoring holds ridge and threshold settings fixed while
changing epsilon; it does not mix epsilon with a simultaneous ridge change.

The fixed-inversion experiment supplies the true three-term structure and uses
every fifth time sample. It separates:

1. Approximation of known coefficient functions on 200 equally spaced theta values.
2. Coefficient inversion with FD spatial derivatives and an oracle PDE response.
3. Coefficient inversion with the same spatial derivatives and FD time derivative.

Its spline fit is an unconstrained linear diagnostic, not a new PINN inversion
backend. Its nonnegative cubic diagnostic uses each stated evaluation interval
as the polynomial domain; this is not an exact replay of joint-training bounds.

## Evaluation contract

- Exact support, missing terms and extra terms are always reported separately.
- Coefficient errors are relative L2, with an explicit theta interval and the
  number of valid points. Missing/undefined references do not count as success.
- Historical neural results and fixed-library experiments are development data.
  Do not report selection rates on them as independent test-set performance.
- Fit seeds and noise seeds must be identified. A DNN-GSR ensemble of five
  initializations is one experiment, not five independent recovery trials.
- For homogeneous curves, `f3_vs_df2_relative_gap` checks the Richards identity
  on interior saved grid points; it is a diagnostic, not an imposed constraint.
- Heterogeneous scores currently use reference coefficients at predicted theta
  along saved trajectories. This is distinct from reconstructing a coefficient
  surface over a rectangular theta-z domain or scoring held-out observations.
- No numerical coefficient-error acceptance threshold has been chosen from the
  current outcomes. Published-accuracy reproduction remains a separate task.

The optional `--fusion-hidden 40 40` in heterogeneous suite runs enables an
experimental nonlinear merger. Default `[]` preserves the original additive
two-branch architecture. Compare them using identical data and budgets before
promoting a new default.

## Local diagnostic reports

`report.py` reads local diagnostic runs from `intermediate/runs/` and writes its
generated Markdown and JSON evidence under `intermediate/history/`. To create a
fresh report without replacing preserved historical evidence, use a new folder:

```bash
python benchmarks/report.py --output intermediate/history/report_refresh
```

Existing report/evidence files are never overwritten. Source metadata is read
from `docs/data_manifest.json`; result files and historical embedded paths are
not migrated or rewritten by the reporter.
