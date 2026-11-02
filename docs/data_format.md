# Standard data format

Synthetic data and real observations share one structure: each dataset folder contains `metadata.yaml` and the one observations file it points to. `kind` only describes where the data came from. It does not determine soil homogeneity, the chosen method or the equation terms.

```text
<dataset>/
├── metadata.yaml
└── observations.npz          # or observations.csv
```

## Observations file

### NPZ

| Array | Shape | Units and conventions |
|---|---|---|
| `times` | `(nt,)` | Elapsed days since `time_origin`; finite, nonnegative, strictly increasing |
| `depths` | `(nz,)` | m, positive downward; finite, nonnegative, strictly increasing |
| `theta` | `(nt,nz)` | Volumetric water content; finite and in [0,1] |

The grid must be complete and rectangular, with at least two times and two depths. NaN, Inf, duplicate coordinates and missing cells are rejected. DNN differentiation accepts unequal time and depth spacing; FD also requires at least five equally spaced observed depths.

```python
import numpy as np

# times, depths and theta are already in the units and shapes above
np.savez("observations.npz", times=times, depths=depths, theta=theta)
```

### CSV

CSV uses a wide table with explicit time: the first column is `time`, and the remaining column names are depths ending in `m`. This example only illustrates the format:

```csv
time,0.05m,0.10m
0.0,0.20,0.21
0.5,0.22,0.23
```

`time` is a numeric count of elapsed days, not a date string. Depth columns must increase numerically, with no duplicates or gaps. A CSV expresses the same rectangular observations as an NPZ. The standard CSV never infers time from row numbers; older sensor CSVs without a time column are still read through the original adapter but are not a standard for new datasets.

## Metadata

```yaml
schema_version: 1
name: heat_demo
kind: synthetic
observations: observations.npz
units:
  time: day
  depth: m
  water_content: fraction
depth_positive: down
time_origin: start_of_simulation
source: Manufactured heat-equation profile
license: MIT
```

| Field | Contents |
|---|---|
| `schema_version` | Always 1; separate from the training configuration's `schema_version=2` |
| `name` | Dataset name |
| `kind` | `synthetic` or `real` |
| `observations` | Path of the observations file relative to this folder, NPZ or CSV |
| `units` | Always `day`, `m`, `fraction` |
| `depth_positive` | Always `down` |
| `time_origin` | Start of elapsed time; state explicitly when the actual start and timezone are unknown, and do not invent a date |
| `source` | Generation method or observation source, with citation or record |
| `license` | Applicable license or permission; use `not_specified` when undecided. Passing format validation does not make a dataset publishable |
| `details` | Optional mapping for instruments, generation settings, seeds and preprocessing records |

If a dataset provides equivalent NPZ and CSV files, the metadata picks one as the entry point and the other stays as a format example. A template for real observations is in [metadata.example.yaml](../data/real/metadata.example.yaml); its placeholders do not mean that data or distribution permission has been obtained.

**Real data.** Convert absolute datetimes to elapsed days first, keeping the actual `time_origin` and timezone. Record any aggregation of duplicate records, removal of missing values and manual imputation explicitly; the tool does not fill with zeros or interpolate on its own. Instruments, calibration and processing steps can go in `details`, with the original inputs kept separately.

**Synthetic data.** `details` may point to an additional analytic reference file for explicit offline evaluation. Analytic equations, reference coefficients and source descriptions in metadata never enter structure selection or inversion constraints automatically.

## Validation and training

```bash
deepgs validate-data data/synthetic/heat_demo
```

Validation checks metadata, file structure, coordinates and value ranges. It does not establish the quality of the observations or the correctness of a recovered equation. The public example is manufactured heat-equation data, not a Richards recovery benchmark.

Training still points to a specific file, for example from a configuration in `configs/`:

```yaml
data:
  path: ../data/synthetic/heat_demo/observations.npz
  format: npz
  units:
    time: day
    depth: m
    water_content: fraction
```

For CSV, point to the corresponding `.csv` file and set `format: csv`. Training does not accept a dataset folder, and it never chooses a method, reads reference values or fills in sign priors from metadata alone. Standard datasets have uniform units; for the unit rules of other legacy inputs, see the [user guide](two_stage_guide.md).

## Imported soil-column records

The six local cases in `data/real/` were provided by the user. Each has 251 rows and all five depths (251×5), at one row per minute. The absolute start time was not provided, and distribution terms are unset.

Time is generated from the sampling interval the user confirmed: `times[i] = i / 1440` days, with the last record at minute 250. The original CSVs are in `intermediate/data_imports/soil_column_20261101/`, and the conversion script is `examples/convert_soil_column.py`. There was no interpolation, resampling, zero-filling or modification of water-content values.

An example standard observations path is `data/real/loamy_sand_wetting_and_drying/observations.npz`, with the matching short configuration `configs/real_column.yaml`. That configuration is only an execution entry point and does not mean that an accuracy evaluation on real data has been done.
