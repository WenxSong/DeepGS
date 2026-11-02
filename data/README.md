# Dataset layout

Synthetic and real observations use the same files and units:

```text
data/
├── synthetic/
│   └── heat_demo/
│       ├── metadata.yaml
│       ├── observations.npz
│       └── observations.csv
└── real/
    ├── README.md
    └── metadata.example.yaml
```

| Category | Contents |
|---|---|
| [synthetic](synthetic/README.md) | Independent manufactured heat-equation demonstration |
| [real](real/README.md) | Six standardized local soil-column cases; public instructions and metadata template |

Each dataset has `metadata.yaml` pointing to one observations file. The included
NPZ and CSV representations of the synthetic demonstration contain the same
observations. Time is elapsed **days**, depth is **metres positive downward**,
and water content is a **volume fraction**.

```bash
deepgs validate-data data/synthetic/heat_demo
```

Validation checks the dataset format and metadata. It does not establish the
correct governing equation or data quality. To train, set `data.path` to the
observations **file**, for example `../data/synthetic/heat_demo/observations.npz`
from a configuration in `configs/`.

See [data format](../docs/data_format.md) for NPZ/CSV fields and the metadata
contract. Synthetic/real describes the source of observations; homogeneous/
heterogeneous is a separate model setting. No method or coefficient signs are
chosen automatically from a dataset's category or metadata.

Historical research inputs remain local under `intermediate/data/` pending
redistribution confirmation. Their provenance is documented in
[data provenance](../docs/data_provenance.md); they are separate from this public
demonstration and its MIT license.
