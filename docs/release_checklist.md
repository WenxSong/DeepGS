# Release preparation

The implemented interface follows the [two-stage algorithm](algorithm_overview.md): structure selection once, then optional fixed-structure PINN inversion. Usage is in the [two-stage guide](two_stage_guide.md).

## Implemented engineering scope

- P0/P1: schema-2 configuration, strict validation, units and coordinates, three selectors, frozen term order and stage artifacts.
- P2: network and signed polynomial fixed-structure inversion, heterogeneous network coefficients, missing-prior states and preserved selection on inversion failure.
- P3: differentiable cubic B-spline coefficient inversion, fixed domains, gradient/numerical checks and versioned prediction round trips.
- P4: shared Python/CLI entry, methods/init/run, templates in installed wheels, clear quickstart/full budgets and an independent source demonstration.
- P5: regression and seven-combination engineering checks, packaging and installed-package verification. Final test counts and artifact checks are recorded with the build in the [release validation record](release_validation.md); old logs do not establish the current result.

Tests distinguish real inversion from empty/no-knee skips. Short runs check execution and numerical contracts, not converged scientific accuracy. Historical regression coverage is retained through the legacy entry.

## Distribution contents

- Software license: author-confirmed MIT, recorded in LICENSE and package metadata.
- Include templates, format docs, citation information, the independent manufactured-profile script and its bundled data/synthetic/heat_demo dataset in the source distribution.
- Organize datasets as data/synthetic/<case> or data/real/<case>, and validate them with deepgs validate-data. Keep the six local soil-column observation cases out of public packages until redistribution terms are set.
- Keep local research th.txt/parameter arrays out of Git and source/wheel artifacts pending redistribution authorization; retain metadata and hashes.
- Keep logs, article PDFs, caches, checkpoints and local analyses out of public distribution.
- Verify templates after wheel installation outside the checkout and source examples without research inputs.

## Remaining deployment items

- Repository: https://github.com/WenxSong/DeepGS (main), renamed from the existing DNN_GSR repository.
- Confirm scope and license for each research dataset before public data distribution.
- Check the CPU CI result on the repository's Actions page.
- Review version/tag and citation metadata once the repository identity is settled. No new DOI deposition is claimed.

These deployment items do not block a code-only package with documented data formats and an independent demonstration.

## Separate scientific validation

- Freeze settings before independent seed/noise validation.
- Assess complete heterogeneous training and additive/fusion representations with matched data and budgets.
- Compare network, polynomial and B-spline inversion across independent cases; fixed-field diagnostics and small coupled tests answer different questions.
- Use held-out depths or forcing records for prediction, identifying development and independent evidence.
- Check the possible missing final Ks node before full-depth endpoint reference evaluation; theta_s=0.43 and theta_r=0.078 are author-confirmed constants for the current local heterogeneous case.

The local historical report at `intermediate/history/validation_20261031.md` retains failures and limitations. B-spline inversion and nonlinear fusion are experimental. Full budgets are not convergence proofs; the package does not claim reproduction of all published figures.
