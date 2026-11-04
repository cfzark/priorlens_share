# Coverage-probability matching

For posterior distributions P,Q and their nominal-c regions R_P,R_Q, define
c1=P(R_Q), c2=Q(R_P), c=1-alpha. CP compares these cross-posterior probabilities.

## Regions

| region | Definition | Input |
|---|---|---|
| equal_tail | Scalar equal-tail interval | PPF/CDF or weighted support |
| central | Scalar interval; joint rectangle calibrated to posterior mass >=c | Marginal quantiles and joint weighted support |
| pdf_hdr | Rank actual log density and accumulate mass to c, including boundary ties | Target logpdf/kernel and posterior weights |
| exact_density | Density threshold at an interpolated alpha-quantile of log density | Target logpdf/kernel |
| kde | Density threshold using weighted Gaussian KDE | Nonsingular posterior support |

The joint central rectangle uses a common marginal-tail probability calibrated
against joint mass, preserving posterior dependence. PDF-HDR records its cutoff,
strict mass, boundary mass and total own mass. Density evaluation uses target
coordinates and their Jacobians; marginal HDR requires the marginal density.

Lupus joint PDF-HDR uses affine coefficient coordinates; its marginal examples
use central intervals. numpy_linear quantiles use uniform weights. PDF-HDR uses
weighted density ranks. KDE uses SciPy's default bandwidth.

## Losses

With gap_weight=lambda:

- historical: lambda*(c1-c2)^2 + (min(c1,c2)-c)^2.
- direct: lambda*(c1-c2)^2 + (c1-c)^2 + (c2-c)^2.
- cap_nominal: true replaces each ci by min(ci,c) before forming every term.

H1 is historical with lambda=1 and no cap.
Capped D0 is direct with lambda=0 and cap_nominal=true.
alphas and alpha_weights specify a weighted multi-level objective.

```yaml
methods:
  - preset: cp_bootstrap_deletion_v2
    paths: 2
    posterior_draws: 512
    region: central
    alphas: [0.05]
    loss_family: historical
    gap_weight: 1
    cap_nominal: false
    cache_regions: true
candidates: {meaning: number_deleted, unit: observation, values: [0, 1]}
```

## Paths and signs

Deletion uses nested prefixes D_(N-m) of a length-N bootstrap:
positive m compares q_b(X) vs q_pi(D), negative m compares q_pi(X) vs q_b(D).
At m=0, D is the full resampled vector.

Augmentation appends nested additions. Positive m compares q_pi(X) vs q_b(X+Y);
the negative direction reverses priors. Empirical presets break ties by negative
direction, then smallest m. Selection precedes averaging over paths.

## Absolute adequacy

Optional delta requires A=max(|c1-c|,|c2-c|)<=delta. Among eligible candidates,
secondary: gap or min_m determines selection; an empty eligible set gives
unmatched. delta_artifact loads a setting-matched no-conflict calibration.
Use `priorlens calibrate-cp --help` for calibration inputs.

Without delta, selection minimizes the specified H1/D0 objective. Selected
cross-coverages describe the resulting match.
