# Synthetic observations

`heat_demo/` contains an independent manufactured profile satisfying
`theta_t = 0.002 theta_zz`. Its NPZ and explicit-time CSV files are equivalent;
`metadata.yaml` records the source, time origin and MIT license.

```bash
deepgs validate-data data/synthetic/heat_demo
```

This is a small tool demonstration, not a Richards-equation recovery benchmark.
Short training on it checks execution and does not establish recovery accuracy.

Additional synthetic datasets should each have a separate directory with
`metadata.yaml` and one canonical observations file. Record the generator,
settings and seed in `details` where applicable. Analytic coefficients or
reference terms may be supplied for explicit evaluation; they must not be used
to infer selected terms or constraints automatically during training.

The common format is described in [data format](../../docs/data_format.md).
