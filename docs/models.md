# Model names, priors and presets

Three names in a configuration serve different purposes:

```yaml
model: {name: nig}
methods:
  - preset: lin_ppd_v1
    paths: 2
    posterior_draws: 512
inference: {backend: optimized}
```

Here `nig` selects the statistical model, `lin_ppd_v1` selects the EPSS
construction, and `optimized` selects how its posteriors are computed.

## Built-in model names

| model.name | Likelihood | Natural parameters | Data | EPSS unit | Optimized posterior |
|---|---|---|---|---|---|
| normal_known | y_i ~ Normal(mu, sigma²) | mu | y | observation | Normal conjugacy |
| poisson_gamma | y_i ~ Poisson(lambda) | lambda | y: nonnegative counts | observation | Gamma conjugacy |
| beta_binomial | y_i ~ Binomial(trials, p) | p | y: counts in 0:trials | observation | Beta conjugacy |
| nig | y_i ~ Normal(mu, sigma²) | mu, sigma2 | y | observation | Normal–inverse-Gamma conjugacy |
| eight_schools | y_j ~ Normal(mu, s_j²+tau²), after integrating theta_j | mu, tau | aligned y, s | school | Collapsed log-tau quadrature |
| lupus_logistic | y_i ~ Bernoulli(logistic(beta0+beta1*x_i)) | beta0, beta1 | aligned binary y, processed x | observation | Adaptive Gauss–Hermite quadrature |

`unit_ids` supplies unique row identities. Model options are
`normal_known.options.sigma` (known SD, default 1) and
`beta_binomial.options.trials` (trials per row, default 20).
Each Binomial row counts as one EPSS unit. Use trials=1 and individual rows
when the unit is an individual Bernoulli observation.

All six built-ins also expose the density/coordinate and Stan interfaces used
by `importance` and `general` backends. `auto` selects their optimized provider.

## Prior dictionaries

Use these dictionaries for both `priors.interest` and `priors.baseline`.

### normal_known

```yaml
model: {name: normal_known, options: {sigma: 1}}
priors:
  interest: {family: normal, mean: 0, sd: 0.5}
  baseline: {family: flat}
```

`normal` means mu~Normal(mean,sd²). `flat` and `jeffreys` both represent a
constant density in mu. Targets use the variable `mu`.

### poisson_gamma

```yaml
model: {name: poisson_gamma}
priors:
  interest: {family: gamma, shape: 2, rate: 1}
  baseline: {family: jeffreys}
```

The Gamma density is proportional to lambda^(shape-1)*exp(-rate*lambda).
`flat` is constant in lambda>0; `jeffreys` is proportional to lambda^(-1/2).
The target variable is `lambda`.

### beta_binomial

```yaml
model: {name: beta_binomial, options: {trials: 20}}
priors:
  interest: {family: beta, a: 2, b: 3}
  baseline: {family: jeffreys}
```

The prior is p~Beta(a,b). `flat` gives Beta(1,1), and `jeffreys` gives
Beta(1/2,1/2). The target variable is `p`.

### nig

```yaml
model: {name: nig}
priors:
  interest: {family: nig, mu0: 1.9, tau0_sq: 1, kappa: 0.5, eta: 0.005}
  baseline: {family: jeffreys}
```

For v=sigma², `nig` specifies

- v~InverseGamma(kappa,eta), with density proportional to v^(-kappa-1)*exp(-eta/v);
- mu|v~Normal(mu0,tau0_sq*v).

`tau0_sq` is a conditional variance multiplier; its reciprocal is the
conditional prior precision multiplier. `flat_mu_ig` uses a constant mu density
and the same inverse-Gamma v density, with fields kappa and eta. The built-in
`jeffreys` label denotes density 1/v with respect to dmu dv.

Targets use `mu` and `sigma2`, individually or jointly. Transformations include
log variance and scaled raw variance. See examples/nig/ for five targets.

### eight_schools

```yaml
model: {name: eight_schools}
priors:
  interest: {family: schools, mu_mean: 0, mu_sd: 25, tau_kind: half_t, tau_scale: 6, tau_df: 4}
  baseline: {family: schools, mu_mean: 0, mu_sd: 25, tau_kind: half_t, tau_scale: 25, tau_df: 4}
```

mu~Normal(mu_mean,mu_sd²), independent of tau. The tau choices are:

| tau_kind | Distribution | Additional fields |
|---|---|---|
| half_t | Positive half of a centered Student-t | tau_scale, tau_df |
| half_cauchy | Positive half of a centered Cauchy | tau_scale |
| lognormal | log(tau)~Normal(log(tau_scale),tau_log_sd²) | tau_scale, tau_log_sd |
| bounded_uniform | Uniform(0,tau_scale) | tau_scale |

Population targets use `mu` and `tau`, including log(tau) and tau².
A focal target uses `theta[k]` with a 1-based original-school index and matching
`focal_unit_id`. R/W, Jones/CP and Clarke implement the fixed-observed-anchor
focal protocol. Lin uses population targets. Versioned Morita Case2 uses
lognormal tau and population targets; its example is configs/morita_schools_smoke.yaml.

### lupus_logistic

```yaml
model: {name: lupus_logistic}
priors:
  interest: {family: student_t, loc: [0, 3], scale: [10, 2.5], df: 7}
  baseline: {family: student_t, loc: [0, 0], scale: [10, 1000], df: 7}
```

The coefficients have independent Student-t priors. Vector order is
[beta0,beta1]; `scale` is the Student-t scale parameter. Supply preprocessed x.
Metadata x_center and x_divisor defines raw-to-processed conversion
(x_raw-x_center)/x_divisor and is used for raw-x prediction targets and Lin's
design records.

Targets include `beta0`, `beta1`, their joint pair, `p_x_0`, `p_x_0.5`,
`p_x_1`, `p_x_1.5`, `p_x_2`, and `risk_difference` (p_x_2 minus p_x_0).
The real-data tutorial uses the three coefficient targets. Joint PDF-HDR uses
affine coefficient coordinates; marginal coefficient examples use central regions.

## Model–method map

| Model | R/W | Jones | CP | Lin | Versioned Morita |
|---|---|---|---|---|---|
| normal_known | Scalar | Source-faithful or adaptation | Scalar | PPD / plug-in | 2008 conjugate |
| poisson_gamma | Scalar | Source-faithful or adaptation | Scalar | — | 2008 conjugate |
| beta_binomial | Scalar | Source-faithful or adaptation | Scalar | — | 2008 conjugate |
| nig | Scalar/joint | Adaptation | Scalar/joint | PPD / plug-in | Joint trace, NIG prior with kappa>2 |
| eight_schools | Population/focal | Adaptation, population/focal | Population/focal | Population PPD / plug-in | Case2, lognormal tau |
| lupus_logistic | Scalar/joint | Adaptation | Scalar/joint | PPD / plug-in | — |

Clarke and generic curvature Morita use available target-density interfaces:
Clarke requires a proper informative prior and candidate densities, while
generic Morita takes an explicit epsilon_prior. KDE and exact-density variants
are listed in their method pages. Moment-based losses additionally check the
required target moments; candidate propriety is checked by the model.

## Custom model names

The [Stan-only posterior workflow](stan_posterior.md) selects a model through
`model.stan` and discovers parameters from its compiled interface; it needs no
built-in model name or Python adapter.

`user_normal` and `user_regression` in the tutorials are adapter-defined names,
registered through `model.adapter` and `model.stan`. Their adapters define these
names. See [general tutorial](general_tutorial.md) for the adapter interface.
The full [preset catalog](methods/index.md) lists method names and data laws.
