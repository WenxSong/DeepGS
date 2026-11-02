# DeepGS Toolkit

DeepGS Toolkit discovers soil water flow equations from water-content observations. It extends the original DNN-GSR project into a shared PyTorch tool that integrates DNN-GSR, DeepGS, Extended-DeepGS and Spline-SR.

**Water-content observations → sparse structure selection → fixed-structure PINN coefficient inversion.**

| Structure selector | Soil | Coefficient representations |
|---|---|---|
| `group` | Homogeneous | `network`, `polynomial`, `bspline` |
| `spline` | Homogeneous | `network`, `polynomial`, `bspline` |
| `spatial_group` | Heterogeneous | `network` |

The selected terms stay fixed during inversion. Polynomial inversion requires explicit coefficient signs, and B-spline inversion is an experimental option.

## Quick start

Python 3.10 or newer is required. From the repository folder:

```bash
pip install -e .
deepgs run configs/demo.yaml --device cpu
```

Results are saved to `intermediate/runs/demo/`. Start with `summary.json`. To repeat a run, use a new output path, because the output folder must be empty.

This short example uses the bundled manufactured heat-equation profile in `data/synthetic/heat_demo/`. It checks that the tool runs; it is not a Richards-equation accuracy benchmark.

## Use your own data

Datasets live in `data/synthetic/<case>/` or `data/real/<case>/`. Each case contains an observations file and `metadata.yaml`, following one [data format](docs/data_format.md).

| NPZ array | Shape | Default unit |
|---|---|---|
| `times` | `(nt,)` | Days |
| `depths` | `(nz,)` | Metres, positive downward |
| `theta` | `(nt, nz)` | Volumetric water-content fraction |

Check a dataset before training:

```bash
deepgs validate-data data/synthetic/heat_demo
```

Create a configuration, then replace its `data.path` with your observations file:

```bash
deepgs methods
deepgs init --selector spline --coefficient-form bspline --output configs/my_case.yaml
deepgs run configs/my_case.yaml --device cpu
```

## Repository layout

| Path | Contents |
|---|---|
| `src/deepgs/` | Package code and configuration templates |
| `configs/` | Runnable example configurations |
| `data/` | Dataset format notes, synthetic demonstration, real-data template |
| `examples/` | Demonstration data generator and soil-column converter |
| `benchmarks/` | [Diagnostic and evaluation scripts](benchmarks/README.md) |
| `tests/` | Test suite (`pip install -e ".[dev]"`, then `pytest`) |
| `docs/` | Guides and records listed below |

## Documentation

| Topic | Document |
|---|---|
| Configuration, outputs and Python API | [User guide](docs/two_stage_guide.md) |
| Synthetic and real dataset format | [Data format](docs/data_format.md) |
| Methods and publication references | [Algorithm overview](docs/algorithm_overview.md) |
| What has and has not been verified | [Release validation](docs/release_validation.md) |
| Source and licensing of research data | [Data provenance](docs/data_provenance.md) |
| Folders and the local archive | [Directory layout](docs/directory_layout.md) |
| Release status and remaining work | [Release checklist](docs/release_checklist.md) |
| Publishing to GitHub | [GitHub update](docs/github_update.md) |

The original DNN-GSR code is preserved in [the previous repository revision](https://github.com/WenxSong/DeepGS/tree/32058cc16ae3493598d7a775f75d8b390eda4c01). New combinations of methods do not imply reproduction of all published accuracy results.

## License and citation

Code is licensed under [MIT](LICENSE). The independent synthetic demonstration and real-data templates are included. Original research data and intermediate files stay local, and six standardized soil-column cases are also kept out of public packages while their redistribution terms remain unset.

Please cite the publications corresponding to the components you use; see the [algorithm overview](docs/algorithm_overview.md) and [CITATION.cff](CITATION.cff).

## References

1. Song, W., Shi, L., Wang, L., Wang, Y., & Hu, X. (2022). Data-driven discovery of soil moisture flow governing equation: A sparse regression framework. *Water Resources Research, 58*(8), e2022WR031926. [https://doi.org/10.1029/2022WR031926](https://doi.org/10.1029/2022WR031926)

2. Song, W., Shi, L., Hu, X., Wang, Y., & Wang, L. (2023). Reconstructing the unsaturated flow equation from sparse and noisy data: Leveraging the synergy of group sparsity and physics-informed deep learning. *Water Resources Research, 59*(5), e2022WR034122. [https://doi.org/10.1029/2022WR034122](https://doi.org/10.1029/2022WR034122)

3. Song, W., Shi, L., He, L., Zha, Y., Hu, X., Rahmati, M., & Vereecken, H. (2025). Deep learning identification of the governing equation for water flow in heterogeneous soils from data. *Water Resources Research, 61*(3), e2024WR037786. [https://doi.org/10.1029/2024WR037786](https://doi.org/10.1029/2024WR037786)

4. Song, W., Shi, L., Wang, Y., He, L., Rahmati, M., & Vereecken, H. (2026). From soil moisture observations to governing flow equations: What data-driven equation discovery reveals. *Water Resources Research*, e2026WR045026.
