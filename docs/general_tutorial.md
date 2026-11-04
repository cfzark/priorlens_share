# Bring your own model

For posterior computation, start with the [Stan-only Lupus example](stan_posterior.md):
write the model once in Stan and supply JSON observations plus YAML prior/target
settings. Both NUTS and IS use that Stan definition. No Python adapter is needed.

The sections below describe the existing general **EPSS** interface, which
also supplies candidate-data operations and response generation.

A model adapter connects the likelihood, priors, data operations and response
generator to PriorLens. The Stan source supplies the corresponding NUTS model.
Adapters are executable Python modules.

## 1. Prepare data

Supply aligned y, design columns and unique row IDs. Standardize/encode the
design before calculation and retain its transformation metadata. Define the
EPSS unit through row operations; expand frequency counts when the unit is an
individual. Grouped data use a corresponding grouped resampling protocol.

## 2. Custom 1D example

```bash
priorlens validate --spec examples/custom_1d/importance.yaml
priorlens run --spec examples/custom_1d/importance.yaml --output results/custom1
```

The model is Normal(mu,known_sigma) with a Normal prior. The example uses generic
IS, and its analytic posterior serves as a test reference. adapter.py and
model.stan implement the same likelihood/prior parameterization.

## 3. Custom 2D example

```bash
priorlens run --spec examples/custom_2d/importance.yaml --output results/custom2
priorlens risk-summary --spec examples/custom_2d/importance.yaml --output results/custom2_risk
```

This is Gaussian regression with independent Normal coefficient priors.
The adapter implements row subsets/appends, empirical-x response generation
and vectorized density evaluation. Targets are the joint coefficients and
both marginals. Tests compare IS moments with the matrix-form Gaussian posterior.

## 4. NUTS

After installing CmdStan:

```bash
priorlens run --spec examples/custom_1d/nuts.yaml --output results/custom1_nuts
priorlens run --spec examples/custom_2d/nuts.yaml --output results/custom2_nuts
```

These examples use CP at m=0 with one bootstrap path and four chains.
For paired backend comparisons, keep data, priors, target coordinates, paths
and seeds fixed while changing inference settings.

## 5. Adapter interface

| Members | Responsibility |
|---|---|
| name, variables, dimension, unit | Model identity, parameter order and EPSS unit |
| validate_data; validate_prior/validate_target | Shapes, support, metadata and target checks |
| subset(data,indices,auxiliary), append(data,new) | Row operations with aligned IDs/design |
| simulate(theta,count,rng,design) | Fresh responses at a fixed natural-parameter vector |
| loglik(theta,data), logprior(theta,prior) | Vectorized natural-coordinate densities |
| logkernel(z,data,prior) | Unconstrained-coordinate kernel with Jacobian |
| to_unconstrained, from_unconstrained | Parameter-coordinate transforms |
| stan_data, stan_draws, stan_parameters | Stan payload, parameter extraction and diagnostics |
| check_target_moments | Required moment-existence checks |
| check_posterior_admissibility | Candidate-posterior propriety checks |
| sampling_signature or options | Settings affecting pseudo-data generation |
| optimized (optional) | Registered model-specific posterior provider |

The loader uses the Stan declarations for both IS and NUTS adapters; compilation
occurs when the general backend is selected. Put extra source dependencies in
model.extra_files and provide complete Stan include dependencies.

simulate receives one natural-parameter vector in variables order. The selected
method supplies that parameter from a plug-in value or a posterior/prior draw;
the generator draws responses conditional on it.

## 6. Configure and run

Choose targets and transformations, method presets, prior locations, candidate
meanings and explicit budgets. Validate the specification, run it into a new
output directory, then inspect selected.csv and the per-task diagnostics.

R/W uses certified target moments; Jones uses quantiles or weighted samples;
PDF-HDR uses a density in the target coordinates. Clarke additionally uses the
informative prior; Morita specifies an epsilon-information construction.
Lin's registered generators cover the built-ins listed in [models](models.md).

For hierarchical models, collapsed local effects provide a common parameter
space across candidate sizes. Other designs can implement a model-specific
posterior interface.
