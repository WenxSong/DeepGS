# Real observations

Six user-provided soil-column cases have been standardized locally. Each has
251 time samples at 1-minute spacing and five sensor depths: 0.05, 0.25, 0.45,
0.65 and 0.85 m. The user confirmed the sampling interval on 2026-11-01.
The absolute clock origin was not supplied; time starts at the first record.

| Local case | Material | Process |
|---|---|---|
| loamy_sand_drying | Loamy sand | Drying |
| loamy_sand_wetting | Loamy sand | Wetting |
| loamy_sand_wetting_and_drying | Loamy sand | Wetting and drying |
| sand_drying | Sand | Drying |
| sand_wetting | Sand | Wetting |
| sand_wetting_and_drying | Sand | Wetting and drying |

Original CSV bytes are retained in `intermediate/data_imports/soil_column_20261101/`.
Conversion only adds explicit elapsed-day time, converts headers to coordinates,
and stores all water-content values unchanged. No sensor column was dropped,
including 0.85 m. The original-file SHA256 and conversion are recorded in each
metadata file.

Redistribution terms for these measurements have not been specified. Their
metadata uses `license: not_specified`; local observation files stay ignored by
Git and are not included in the prepared publication folder or installation
packages. The public files here are instructions and a
[metadata example](metadata.example.yaml). Format validation does not grant a
data license.

Reproduce the conversion for a source CSV with a confirmed interval:

```bash
python examples/convert_soil_column.py RAW.csv --output-dir data/real/your_dataset --minutes-per-row 1
```

A short local configuration is provided as `configs/real_column.yaml`.

For an authorized dataset, create one directory containing:

```text
your_dataset/
├── metadata.yaml
└── observations.npz          # or observations.csv
```

Copy `metadata.example.yaml` into that directory as `metadata.yaml`, replace
every placeholder with actual information, and record the applicable license or
permission. Local real-data files are excluded from version control until their
redistribution scope is confirmed.

Use the [common NPZ/CSV format](../../docs/data_format.md): elapsed days, metres
positive downward and volumetric fractions. Convert absolute datetimes to
elapsed days and record the actual origin and timezone in metadata. Record
instrument/source information and preprocessing in `details`.

Handle duplicate times, missing values and incomplete sensor grids explicitly.
The tool does not silently interpolate missing records or replace them with
zero. If records are excluded, aggregated or imputed before import, document
that operation and retain the original inputs separately.

```bash
deepgs validate-data data/real/your_dataset
```

Training still uses `data.path` pointing to the observations file, not the
directory. The `real` category does not determine whether soil is homogeneous.
