# Jones and Wasserstein computation

For observed data X of size N, source-style paths draw one parameter from the
informative observed posterior and generate additions independently across
candidate sizes and directions, conditional on that parameter.

- Positive m: q_pi(theta|X) vs q_b(theta|X+Y_plus_m).
- Negative m: q_b(theta|X) vs q_pi(theta|X+Y_minus_m).

Select within each path, then average signed m. Source-style ties prefer the
positive direction and then the smallest addition. Empirical/nested presets
use their declared negative-direction tie preference.

## Scalar W2

W2(P,Q)^2 = integral_0^1 [F_P^-1(u)-F_Q^-1(u)]^2 du.

| Representation | Calculation | Control |
|---|---|---|
| Two recognized Normal measures | sqrt((muP-muQ)^2+(sdP-sdQ)^2) | Analytic identity |
| Two PPF-equipped measures | Trapezoidal quantile integral over [1e-6,1-1e-6] | quantile_grid_size |
| IS/GH weighted clouds | Interpolated weighted quantiles | Fixed 256 midpoint probabilities |
| NUTS equal-weight draws | Empirical quantile approximation | Fixed 256 midpoint probabilities |
| integration_rule: empirical_sort | Sorted point pairing | Exact empirical W2 for equal counts and uniform weights |

The low-level Wasserstein object accepts order=1 or2; Jones uses order2.
squared: true returns the squared distance. The PPF integration and finite
cloud calculations have their respective integration and sampling errors.

## Bivariate targets

Joint comparisons use sliced W2: project onto directions, average the
one-dimensional squared distances, then take the square root.
projection_rule: random uses seeded random unit directions;
projection_rule: nig uses equally spaced angles with a fixed half-step offset.
projections sets the direction count. Cloud projections use 256 midpoint
quantiles, or sorted pairing under empirical_sort.

Target transformations and scales define the geometry. The generic runner uses
the declared coordinates. The saved Lupus joint comparison's baseline-SD
geometry is recorded in [case settings](../cases.md).

```yaml
methods:
  - preset: jones_source_adaptation_v2
    paths: 2
    posterior_draws: 512
    quantile_grid_size: 128
    projections: 8
    projection_rule: nig
    integration_rule: quantile
candidates: {meaning: number_augmented, unit: observation, values: [0, 1]}
```

posterior_draws controls generated target support. IS/GH targets retain their
full weighted support. The model checks the finite second target moments used
by Wasserstein comparison.
