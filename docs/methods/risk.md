# Reimherr and Wiesenfarth

For target Z=T(theta), reference z*=E_b[Z|X], posterior mean a(D) and
covariance V(D), the available losses are:

| risk | Loss |
|---|---|
| posterior_mse | trace(V(D))+||a(D)-z*||² |
| squared_mean | ||a(D)-z*||² |
| posterior_variance | trace(V(D)) |

Reimherr defaults to bootstrap and posterior_mse, comparing informative risk
at N with baseline risk at N+m. Wiesenfarth defaults to plug-in likelihood and
squared_mean, comparing baseline risk at N with informative risk at N-m.
Both minimize the absolute risk gap, breaking ties by the smallest grid value.

## Components

Both presets accept sampling: bootstrap or likelihood. The default
reference_rule: baseline_posterior_mean uses E_b[T(theta)|X].
target_reference explicitly sets z*. theta_star sets the natural parameter
for likelihood generation; its default is E_b[theta|X].

For nonlinear targets, E_b[T(theta)|X] and T(E_b[theta|X]) are distinct
quantities. legacy_plugin supplies the transform-of-plug-in reference.

risk_moments: marginal uses analytic/weighted coordinate moments.
legacy_joint uses the joint Measure calculation.
risk_coupling: shared_by_size shares compatible data laws at each size across
methods/priors; different sizes have independent streams. independent uses
separate method streams. risk_data_seed controls the shared data bank.

## Mean and median

`priorlens run` aggregates mean risk. For both reductions:

```bash
priorlens risk-summary --spec examples/nig/quick.yaml --output results/nig_both
```

The workflow saves each raw loss bank and separately reduces it by mean and
median before matching the fixed/candidate risk curves. Median refers to the
internal risk sample. The command selects the R/W entries from the supplied
specification.

First-moment losses check first moments; variance losses check second moments.
NIG bootstrap can use risk_invalid_policy: redraw_baseline_proper to condition
on a proper baseline posterior.

Focal-school risk uses the marginal theta_j posterior, integrating global
hyperparameters and retaining the original focal observation once.
