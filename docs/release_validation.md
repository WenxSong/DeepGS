# Engineering validation of the two-stage tool

Initial validation: 2026-10-31. Data interface update: 2026-11-01. This record verifies the tool's implementation, interfaces and packaging. Validation of full paper accuracy is in separate historical diagnostics, and the two sets of results are not mixed.

## Scope

P2–P5 are the internal development phases.

| Phase | Implemented | Engineering checks |
|---|---|---|
| P2 | Fixed-structure network/polynomial; heterogeneous additive/fusion networks | Constant term, fourth-order terms, missing priors, replay of old numerical checks, pairing of random state and collocation points |
| P3 | Differentiable cubic B-spline inversion | SciPy basis and derivative comparison, endpoints and knots, gradcheck/gradgradcheck, control-point recovery, CPU/CUDA |
| P4 | methods/init/run/legacy, API, packaged templates, model loading | Support matrix, invalid combinations, relative paths, output protection, template availability in an independent installation |
| P5 | Full regression, two-stage combinations, packaging, MIT license, citation and CI configuration | All 218 tests passed; package checks recorded separately in a local verification list |

**Seven combinations.** The six homogeneous paths (group/spline × network/polynomial/bspline) and the heterogeneous spatial_group × network path each completed structure selection, actual inversion, saving and loading on a small deterministic candidate library. Spline combinations fix the position on the elimination path in tests, which keeps Kneedle parameter choice out of the scheduler check. These are engineering fixtures, not seven independent scientific recovery trials.

**Other paths covered.** The real-data FD entry, a very short DNN ensemble library build, empty structures, missing knees, missing sign priors, preservation of first-stage outputs when inversion fails, and independent inversion of an explicitly supplied structure. Models with a supplied structure are labelled `supplied` and do not count as automatic discovery.

**Pairing.** With the same observations, field configuration and seed, the three coefficient representations have identical initial-field and collocation fingerprints. Polynomial inversion solves BVLS at each step, while networks and B-splines update coefficients with Adam, so equal training steps do not mean equal optimization constraints or computational cost.

**Precision.** Saturated float32 field outputs risk endpoint rounding. The new two-stage field uses endpoints at the same precision as the coefficient representation and a `torch.lerp` mapping; the legacy workflow keeps its original mapping. Regression tests cover the upper and lower saturation bounds, querying coefficients at predicted values after save and load, and finite gradients for all three representations.

## Environment and packaging

Local verification used the dependencies of the existing `torch_env`. No claim is made about a fresh dependency environment or remote GitHub Actions. The CI configuration includes CPU tests, sdist and wheel builds, and an installed-resource check outside the repository. Actual remote results can only be obtained after publishing the repository.

| Check | Result |
|---|---|
| Local source working folder | 218 passed |
| Unpacked source package, excluding raw research data | 194 passed, 24 explicitly skipped |
| Wheel built from the source package and installed to an independent folder, run in isolated mode from outside the repository: `methods`, generating seven templates, actual inversion on manufactured data, loading a model and predicting | All passed |

Distributions contain no papers, model caches, run logs or raw research data. Package hashes and local log paths are recorded in `intermediate/p25_verification.json`, which keeps local absolute paths out of public records.

## Licensing

The author confirmed the code license as MIT. The public license for the current research data still awaits separate confirmation from the author, so the original files stay local, and default distributions and Git exclude `th.txt` and `npy` files. Public format documentation, data hashes and the independent manufactured-solution generator are kept. Without the research data, only the corresponding legacy data tests are explicitly skipped, and synthetic-data engineering checks continue to run.

## Not in this first release

Later research extensions include two-dimensional B-splines, shape constraints, the f3=df2/dθ constraint, initializing from pretrained field weights, and full-budget accuracy evaluation. None of these is a validated feature of this first release.

## Data interface update (2026-11-01)

Public datasets are organized as `data/synthetic/<case>/` and `data/real/<case>/`, sharing NPZ or CSV with explicit time and depth, plus `metadata.yaml`. `deepgs validate-data` was added. Training reads matching metadata and records its source and hash, and metadata does not inject reference equations or parameters. The soil-column CSV converter requires an explicit sampling interval; the legacy reading entry keeps its original configuration defaults.

| Check | Result |
|---|---|
| Local full regression | 270 passed |
| Source package without local legacy inputs | 246 passed, 24 skipped |

New tests cover the standard format, coordinates and units, metadata path restrictions, explicit-time CSV, soil-column conversion and source recording.

Each of the six user-provided soil-column datasets is 251×5. The 1-minute row interval was confirmed by the user. All original water contents and five depths are retained, the converted NPZ/CSV values match the originals, and the original files are archived byte for byte.

The release scope includes the independent manufactured `heat_demo`, format documentation and the real-data template. The six soil-column observations and their case metadata are excluded from the default source package and the upload copy, because public data terms have not been set. Packaging and out-of-repository installation results, current package hashes and the actual commands are recorded locally in `intermediate/dataset_update_verification.json`. These engineering checks do not replace complete training and scientific evaluation of the soil-column data.

## English documentation and readability update (2026-11-02)

Documentation and generated report text were translated to English. Long functions in `workflow.py`, `cli.py` and `experiment.py` were split without changing settings, messages or output formats.

| Check | Result |
|---|---|
| Local full regression | 270 passed |
| Source package without local legacy inputs | 246 passed, 24 skipped |
| Wheel installed outside the repository: `methods`, `init`, demo run with inversion | All passed |

Remote CI results are on the repository's Actions page.
