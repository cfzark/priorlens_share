# Development instructions

## Scope

This repository accompanies three prior-impact case studies: normal sampling
with unknown variance (NIG), Lupus logistic regression, and Eight Schools.
Read `README.md`, `description.md`, and the relevant configuration before editing.
Work from this repository root. Do not assume access to the authors' original
workspace or a particular computer, account, scheduler, or filesystem mount.

## Scientific invariants

1. Keep model, baseline, informative prior, target coordinates, pseudo-data law,
   loss, candidate grid, selection order, and tie rule separate and explicit.
2. Primary columns are Reimherr, Wiesenfarth, Jones, and **CP**. Label the fourth
   column simply `CP`; retain its complete definition in configurations.
3. R/W use mean risk for NIG and Eight Schools and median risk for Lupus.
   Median reduces internal losses before matching. Simulation plots still show
   arithmetic means of dataset-specific EPSSs.
4. CP here uses bootstrap deletion and H1. Do not change it to augmentation,
   D0, or an adequacy-gated estimator as a plotting edit. Historical NIG region
   definitions are preserved; do not claim an unperformed central-region rerun.
5. A joint posterior and its scalar marginals are different EPSS targets.
   Nonlinear target transformations and coordinate scaling are scientific choices.
6. Preserve seeds, stable prior-grid IDs, and observed-data generation. Changing
   a computation backend or configuration requires a new output root and tests.
7. Do not label these multidimensional Jones adaptations as exact reproductions
   of an original one-dimensional conjugate experiment.
8. Keep numerical failure distinct from unmatched; do not silently drop paths,
   candidate fits, replications, or nonfinite risks.

## Source map

- `v2/src/prioreffect_v2/models.py`, `quadrature.py`, `inference.py`: models and posteriors.
- `paths.py`, `risk_components.py`: matching paths and risk pseudo-data.
- `losses.py`, `regions.py`, `selection.py`: discrepancies, regions, matching.
- `methods.py`: method composition.
- `school_acceleration.py`, `lupus_experiment.py`, `lupus_acceleration.py`:
  specialized numerical workflows; Lupus also uses the retained package/C++ source.
- `scripts/run_nig*.py`: retained NIG simulation entry points.
- `compute/`: portable wrappers; not alternative scientific definitions.
- `plotting/layouts.py`, `plotting/plot_figures.py`: publication plotting only.

## Safe workflow

- Preserve bundled source tables, original figure assets, and completed outputs.
  Never overwrite them to make a validation pass. Use a fresh directory.
- Run `python tests/check_bundle.py` and the relevant small computation after
  changes. Full production grids require explicit authorization.
- A successful smoke establishes executable integration, not production Monte
  Carlo precision, statistical validity, or exact historical equality.
- Use relative paths or a path derived from `__file__`; do not add personal paths.
- Do not commit generated binaries, credentials, raw private data, large run
  archives, environment folders, or scheduler logs.
- Keep finite-grid endpoints visible; real data have no known-truth star.
  Use shared zero-centered heatmap scales. Schools shading is between-dataset
  SD, never relabel it as a confidence interval.
- Save numerical source tables and provenance beside new plots. Hash checks
  establish file integrity, not correctness of an estimator.
- No automatic uploads, repository publication, new license, or expensive
  simulations without the user's explicit request.
