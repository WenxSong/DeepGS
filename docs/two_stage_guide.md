# Two-stage workflow guide

The main workflow is *build a candidate library → select a structure → invert coefficients with a fixed-structure PINN*. This guide describes the implemented interface. For the origin of each method, see the [algorithm overview](algorithm_overview.md); for local archive locations, see the [directory layout](directory_layout.md).

## 1. Choose the components

```bash
deepgs methods
```

| Structure selector | Soil | Available inversion representations |
|---|---|---|
| `group` | homogeneous | `network`, `polynomial`, `bspline` |
| `spline` | homogeneous | `network`, `polynomial`, `bspline` |
| `spatial_group` | heterogeneous | `network` |

- The `spline` selector uses B-splines to select a structure, while `bspline` is a representation for the inversion stage. The two stages are configured independently and share neither knots nor training controls.
- The original Spline-SR inversion uses a sign-constrained, nonnegative cubic polynomial. B-spline inversion is a new, experimental option.
- Heterogeneous networks default to `architecture: additive`, which represents a(θ)+b(z). `architecture: fusion` uses nonlinear fusion to allow interaction, and its performance needs independent evaluation.
- This first release has no two-dimensional splines or spatial polynomials.

## 2. Configuration and budgets

```bash
deepgs init --selector spline --coefficient-form bspline --budget quickstart --output config.yaml
deepgs init --selector group --selection-only --output selection.yaml
```

The selector determines the soil type, and an incompatible explicit `--soil` is an error. Templates ship with the wheel, so `init` also works outside the repository. It neither generates nor downloads data: replace `data.path: YOUR_DATA.npz` with your own path. It refuses to overwrite an existing configuration.

| Budget | Settings | Purpose |
|---|---|---|
| `quickstart` | 20 pre-selection field-fit steps, 20 inversion steps, small networks, few collocation points | Checks that the workflow runs |
| `full` | 20,000 / 20,000 steps | A starting budget |

Neither budget is a convergence guarantee or an independently validated optimal setting.

The source tree and source distribution include an independent demonstration:

```bash
python examples/make_demo.py --dataset-dir intermediate/synthetic_demo
```

It generates a standard synthetic dataset (`observations.npz`, `observations.csv`, `metadata.yaml`) satisfying θ_t = 0.002 θ_zz, in day/m/fraction units. `configs/demo.yaml` reads the bundled `data/synthetic/heat_demo/observations.npz` by default, so regenerating it is unnecessary. The demonstration does not depend on local research data and is not a Richards-equation recovery benchmark. Wheel users supply their own data or take the demo script from the source distribution.

| Configuration section | Purpose |
|---|---|
| `problem` | Whether the soil is homogeneous |
| `data` | Format, units, observation depths, noise |
| `preprocess` | FD or DNN differentiation, pre-selection field network and seeds |
| `library` | Candidate order and sampling grid |
| `structure` | Selector, grouping, scoring parameters |
| `inversion` | Whether to invert, representation, inversion field, domain and optimization budget |
| `runtime` | Device, whether to save the candidate library |

`preprocess.field` and `inversion.field` configure independent networks. The inversion field is initialized independently, because an ensemble candidate library is not an averaged network that can be inherited. The structure-selection entry is called exactly once, and the set and order of terms stay fixed during inversion.

Standard datasets live in `data/synthetic/` and `data/real/` and share one [data format](data_format.md). Run `deepgs validate-data DATASET_DIR` first. Training configurations still point to the observations file inside the folder.

## 3. Data conventions

The standard NPZ contains `times (nt,)`, `depths (nz,)` and `theta (nt,nz)`. Coordinates must be finite and strictly increasing, with at least two times and two depths. θ must be a complete rectangular array with no NaN, Inf or missing values, and must lie in [0,1] after unit conversion.

| Format | Reading rule |
|---|---|
| `npz` | `times`, `depths`, `theta`, keeping explicit coordinates |
| `txt` | First column is time; the rest are equally spaced nodes from 0 to `data.depth_max` |
| `csv` | The standard form has a `time` column of explicit days and depth columns such as `0.05m`, `0.15m`; older CSVs without `time` are still adapted through `depths` / `minutes_per_row` |

**Units**

- CSV sensors are never guessed from "the first few columns". A CSV's unit configuration must be day/m; implicit minute intervals are converted to days.
- TXT and NPZ support day/hour/minute/second and m/cm/mm. θ supports fraction/percent.
- Input is converted to day/m/fraction first; depths are then selected, data resampled and synthetic noise added.
- `data.depths` and `data.depth_max` use the input depth unit. `library.depth_min` and `library.depth_max` use the internal unit, metres.

**Differentiation**

- Only `down` is supported as the positive depth direction.
- FD needs at least five equally spaced observed depths, and missing difference nodes are not filled by interpolation.
- DNN multi-seed candidate libraries may be averaged only when their coordinates, row order and term order agree.

**Paths and device**

- Relative data paths in YAML resolve against the configuration's folder; those in dictionary configurations resolve against the working directory at call time.
- Output paths resolve against the working directory at call time.
- `--device` overrides the configured device, and the effective value is written to the manifest.

## 4. Running and representations

```bash
deepgs run config.yaml --device cpu --output intermediate/runs/experiment_01
```

The output folder must be empty. Supported devices are `cpu`, `cuda`, `cuda:index`, or `auto` in the configuration.

| Representation | Training | Configuration |
|---|---|---|
| `network` | Adam jointly updates the field and coefficient networks | `hidden_sizes`; heterogeneous soil uses two branches and additive/fusion |
| `polynomial` | Bounded least squares solves nonnegative controls at each step; Adam updates the field | `degree=3`, explicit `signs` |
| `bspline` | Adam jointly updates the field and spline controls | `degree=3`, number of internal knots |

The polynomial template's `theta_z=-1`, `theta_zz=1`, `theta_z^2=1` are explicit Richards sign priors. They take no part in structure selection and do not add terms. If a selected term has no specified sign, inversion reports `blocked_missing_prior`, the run is `incomplete` and the command exits non-zero; the structure result is kept.

B-spline knots are fixed and controls are trainable; the coarse selection-stage curve is not copied as the final curve. This first release does not automatically add smoothness penalties, monotonicity constraints or f3=df2/dθ.

The θ domain is derived from the observed range plus a margin, or specified explicitly as a range containing all observations. It is fixed before training, field outputs are constrained to it, and querying coefficients outside it raises an error. With the same field configuration and inversion seed, all backends share field initialization and collocation points; coefficient initialization uses an independent random state.

## 5. Outputs and status

```text
run/
├── config.yaml
├── manifest.json
├── summary.json
├── selection/
│   ├── structure.json
│   ├── coarse_coefficients.npz
│   └── library.npz                 # when save_library=true
└── inversion/                      # when inversion completes
    ├── coefficients.npz
    ├── fields.npz
    ├── history.json
    └── model.pt
```

| Output | Contents |
|---|---|
| `manifest.json` | Configuration, units, seeds, data and source hashes, stage timings and status |
| `summary.json` | Ordered terms, the symbolic equation and each stage's status; scientific error metrics are `unavailable` without a reference |
| `coarse_coefficients.npz` | First-stage coarse estimates; coefficient rows match term names, and `active` stores original candidate indices |
| `coefficients.npz` (homogeneous) | `theta_grid` and `curves` with shape `(n_terms,n_theta)` |
| `coefficients.npz` (heterogeneous) | θ/z paired points and `(n_terms,n_points)` coefficients; `coordinate_kind=trajectory_pairs` means values along the reconstructed trajectory, not a complete rectangular surface |
| `fields.npz` | `theta` / `observed_theta` of shape `(nt,nz)`, with `times` and `depths` |

Empty structures and missing knees have explicit statuses and skip inversion. A completed execution does not mean that inversion succeeded or that the equation is correct. If training fails, the first-stage outputs are kept, and non-finite diagnostic values are written as JSON `null`.

## 6. Python API and model reloading

```python
import deepgs

result = deepgs.run("config.yaml", device="cpu", output="intermediate/runs/python_case")
print(result.selection.terms)
print(result.inversion.status)
```

Load a model only when the inversion status is `completed`:

```python
import numpy as np
import deepgs

model = deepgs.load_model("intermediate/runs/python_case/inversion/model.pt", device="cpu")
low, high = model.coefficients.spec["theta_bounds"]
theta = np.linspace(low, high, 100)
values = model.evaluate_coefficients(theta)  # (100,n_terms), ordered as model.terms
prediction = model.predict(z=[0.05, 0.10, 0.15], t=0.5)
```

- **Heterogeneous models** also take depth, for example `model.evaluate_coefficients(theta, z=0.15)`. θ and z broadcast, and evaluation returns `(n_points,n_terms)` for the flattened list of points. θ is converted to the saved float32/float64 precision before the domain check; NaN, Inf and out-of-domain values raise errors.
- **`predict(z,t)`** broadcasts, and its output shape equals the broadcast coordinate shape. Coordinates must be finite. It does not clip queries outside the observed time and depth ranges; these are extrapolations, and the current checks make no claim about their accuracy.
- **Loading** rebuilds the model, units and term order. Optimizer-level checkpoint resume is not provided.

Lower-level interface:

```python
from deepgs.contracts import LibraryArtifact

library = LibraryArtifact.create(candidate_library)  # includes per-row depth/time
selected = deepgs.select(library, structure_config, soil="homogeneous")
inverted = deepgs.invert(observations, selected, inversion_config,
                        soil="homogeneous", device="cpu")
```

`invert` also accepts an explicit term list such as `["theta_z", "theta_zz", "theta_z^2"]`. This is marked as a supplied structure and must not be counted as automatic discovery. `Observations` passed directly must already use day/m/fraction.

## 7. Legacy entry and release status

```bash
deepgs legacy configs/gsr_fd_loam.yaml --device cpu --output intermediate/runs/history
```

Legacy configurations are read under their original rules, and the original alternating optimization exists only in this entry. Schema-2 runs fix the structure and carry a different workflow label, so their results are not automatically pooled with legacy results. Calling `deepgs` with a bare configuration path prints a compatibility warning. Legacy configurations may need research data that is not distributed.

The code is under the [MIT license](../LICENSE). The publication scope of the local research data has not been confirmed, so it is not in public packages; an independent demo script and format documentation support a first run. Tests that need the research data are explicitly skipped when it is absent. The repository is WenxSong/DeepGS, renamed from the original DNN_GSR repository. No new DOI deposit is claimed.

The short seven-combination check verifies engineering connections and does not replace a complete scientific evaluation. Historical diagnostics are archived locally in `intermediate/history/validation_20261031.md`; for release status, see the [checklist](release_checklist.md).
