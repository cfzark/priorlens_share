# Data, targets and configuration

This page describes EPSS specifications. The separate
[posterior-only schema](stan_posterior.md) takes Stan, JSON and YAML without
methods or candidate grids.

Specifications use schema_version: 1. Data may be inline or in JSON.
Data, prior-grid, adapter and Stan paths are relative to the YAML file;
output_root is relative to the working directory. The CLI --output overrides it.

## Fields

| Field | Meaning |
|---|---|
| model | Built-in name/options or Python adapter and Stan source |
| data | Aligned y, design/metadata and row IDs |
| priors | Complete baseline and interest dictionaries |
| prior_grid | Unique prior_id plus fields overriding the interest prior |
| targets | Unique target IDs with one or two coordinates |
| methods | Presets and their controls |
| candidates | Values, meaning and unit; method-local entries override this block |
| inference | Posterior backend and numerical controls |
| seed / seed_scheme | Random seed and derivation scheme for generic methods |
| replication_id | Dataset identity; also keys Lin streams |
| run_mode | smoke / production; production requires explicit method budgets |
| execution | Operational metadata |

See [model names](models.md), [presets](methods/index.md) and
[method options](options.md).

## Rows and covariates

Each row is one EPSS unit: observation for NIG/Lupus and school for Schools.
Expand grouped Lupus counts to patient rows. Grouped/longitudinal data use an
adapter that implements the intended resampling unit.

Preprocess covariates once. Bootstrap resamples aligned rows; likelihood/PPD
resamples design rows and generates fresh responses using the fixed transformation.
Lupus prediction targets and Lin design records use x_center/x_divisor.
Coefficient-only targets can omit both. The example includes complete metadata.

## Target coordinates

```yaml
targets:
  - id: joint_log
    coordinates:
      - {variable: mu, scale: 0.1}
      - {variable: sigma2, transform: log, reference: 0.01}
```

| transform | Coordinate |
|---|---|
| identity | theta/scale |
| log | log(theta/reference)/scale |
| square | theta²/scale on nonnegative support |
| logit | log(theta/(1-theta))/scale |

scale and reference are positive. Posterior point transformations retain their
probability weights; density transformations include the Jacobian.
The declared coordinates define the geometry used by distances and regions.

## Candidate meanings

| Method | meaning | Candidate size |
|---|---|---|
| Reimherr | baseline_size_offset | baseline N+m |
| Wiesenfarth | informative_size_reduction | informative N-m |
| Jones/CP augmentation | number_augmented | N+m, with directional sign selection |
| Jones/CP deletion | number_deleted | N-m, with directional sign selection |
| Clarke/Morita | pseudo_sample_size | m |
| Lin | informative_fresh_size_reduction | informative fresh N-m, signed m in [-N,N] |

Values are sorted unique integers. min_retained bounds retained sample size for
the generic grids; model checks separately establish posterior admissibility
and required moments. Lin checks its signed range and the informative prior at
the zero-data endpoint.

Focal Schools retain the original observation exactly once and change auxiliary
schools, with focal_protocol: fixed_observed_anchor_v1. theta[k] uses the original
1-based school index and agrees with focal_unit_id.
