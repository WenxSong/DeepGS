# Research data provenance and location

The public data folder is divided into `data/synthetic/` (synthetic) and `data/real/` (real observations), which share one [standard format](data_format.md). What is currently public is an independent manufactured-solution demonstration and a template for real data.

The earlier research inputs are all synthetic simulations and were archived from the original `data/` to `intermediate/data/`. The original files, the original `data/README.md` and the original `manifest.json` keep their original bytes. The public list of current locations and hashes is in [data_manifest.json](data_manifest.json). That list only updates location information and does not alter the original data.

## Local research inputs

- `intermediate/data/th.txt`: homogeneous loam simulation from the DNN-GSR materials.
- `intermediate/data/loam_all_0.2/th.txt`: heterogeneous simulation, with `ks.npy`, `alpha.npy` and `n.npy`.

## Known assumptions and inconsistencies

- The author confirmed on 2026-10-31 that Ks holds nodal values. With 100 values for 101 nodes, the final node may be missing. The current loader pads with the last available value, and this assumption does not affect reference diagnostics in the upper z = 0.05–0.25 m.
- The author confirmed that θ_s = 0.43 and θ_r = 0.078 are constant over depth in this case. The spatial standard deviations in Table 1 of the paper belong to other cases, and the constants used in existing diagnostics are unchanged.
- A closure error of about 0.09–0.10 over the full profile is a recorded inconsistency. It must not be taken as numerical rounding or as proof that all the metadata is correct.

## Licensing

The code is MIT-licensed. The public license for the research inputs is still awaiting separate confirmation, so the data stays in `intermediate/`, which is ignored by Git and excluded from distributions. Installation packages contain only the tool, format documentation and metadata, not this raw research data.

The [public Extended-DeepGS record](https://zenodo.org/records/14551650) is marked CC BY 4.0 and its download checks passed. However, none of its five `th.txt` files is identical to the two local files, and it has no parameter arrays, so no license was assigned to the local data on that basis. The original verification record is kept in `intermediate/public_reference_14551650/identity_check.json`.

## Demonstration data

`examples/make_demo.py` generates an independent manufactured heat-equation profile, and the repository provides the same standard demonstration in `data/synthetic/heat_demo/`. The demonstration does not depend on the research inputs above and is not a scientific Richards-equation recovery result.
