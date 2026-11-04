# Contributor and agent instructions

Read README.md, description.md, docs/models.md, docs/methods/index.md and the
relevant method page before editing. Keep the package portable and standalone.

- Preserve statistical definitions, target scales, data laws, seeds, candidate
  grids and ties unless a requested change explicitly covers them.
- Reuse the numerical kernels through adapters. Record kernel changes in
  source_map.json and compare full curves and selections in paired tests.
- Keep region/sampler/risk variants explicit. Describe bivariate Jones as sliced W2.
- Preserve every failed-path record and distinguish numerical failure from
  the CP unmatched outcome.
- All method presets use the common runner. Lin targets are population
  parameters in its registered built-ins. R/W run uses mean risk;
  risk-summary reports paired mean/median reductions.
- Distinguish posterior computation from pseudo-data generation.
- The Stan-only posterior command takes Stan/JSON/YAML; general is NUTS and
  importance is BridgeStan IS. Keep its schema distinct from the EPSS runner,
  which still uses built-ins or adapters for data operations and generation.
- Preserve aligned row IDs/design metadata and the declared EPSS unit.
- Use fresh outputs for new runs and the common missing-task recovery workflow
  for resumption. Preserve completed results and failed-attempt evidence.
- Run tests with python -m unittest discover -s tests, integrity checks with
  python tools/check_release.py, and plots into a fresh output directory.
- Refresh release_manifest.json with tools/update_manifest.py after reviewing
  shipped-file changes. Keep generated build/results/cache files out of the repo.
- Obtain explicit authorization for production batches, scheduler submissions,
  public uploads and software downloads.
- Keep runtime paths configurable and distribute source/configuration/compact
  data rather than machine-specific binaries, accounts or credentials.
- On shared HPC login nodes use small checks; schedule substantive work through
  the site's scheduler and follow its monitoring limits.
