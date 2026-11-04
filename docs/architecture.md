# Architecture and extension

| Layer | Files | Responsibility |
|---|---|---|
| Public API | priorlens/api.py, cli.py, workflows.py | Run/validate, compatibility translation, paired-risk summaries |
| Stan-only posterior | priorlens/stan_posterior.py | Stan/JSON/YAML loading, NUTS/BridgeStan IS, shared-target posterior output |
| Contracts | prioreffect_v2/core.py | Targets, coordinates, errors and seed utilities |
| Models | models.py, quadrature.py | Likelihood/prior interfaces and optimized posteriors |
| Inference | inference.py, school_acceleration.py | Backend dispatch, fit caches and diagnostics |
| Pseudo-data | paths.py, risk_components.py, focal.py | Paths, risk data banks and focal-school design |
| Loss / region | losses.py, regions.py, central_tkl.py | Wasserstein, risk, coverage and KL |
| Methods | methods.py, morita_versions.py | Preset resolution and family dispatch |
| Lin | lin.py, lin_runner.py | Nested data/selection kernels and common-task integration |
| Selection | selection.py | Bidirectional signs, ties and CP adequacy |
| Experiments | experiments.py, calibration.py | Plans, workers, recovery, merge and delta calibration |
| Specialized engines | nig_cp.py, lupus_acceleration.py, prioreffect/ | Model-specific numerical acceleration |

## Lin grouping

Manifest leaves remain per prior/method/target. Each Lin task carries its
method's target group. The first target evaluates the group together; following
targets retrieve their curves and selections from a bounded result cache.
Baseline regions and path banks have a separate bounded cache shared across
prior cells and the two variants. Independent workers compute the same values
from the recorded seeds.

## Adding components

For posterior-only models, use the [Stan interface](stan_posterior.md). For
general EPSS models, implement the [adapter contract](general_tutorial.md). For an
accelerator, compare full candidate objectives, pathwise selections and
diagnostics against the reference on a fixed data bank.

Cache identity includes model, data, prior, target definitions, grid and
numerical settings. A region defines its own-posterior mass, cross-probability,
density coordinates and boundary-tie handling. A method defines its data law,
loss, aggregation order, sign and tie rule.

source_map.json records retained source provenance and integration changes;
release_manifest.json inventories the current distribution.
