# PriorLens

PriorLens measures prior impact through scalar and bivariate posterior targets.
It combines Stan models, posterior computation, pseudo-data sampling and EPSS
methods in a configurable Python workflow. Start with a model written once in
Stan, JSON observations and a YAML configuration.

## Install

Python 3.10 or newer. Run the EPSS workflows on Linux (including WSL on
Windows); task execution and built-in Stan compilation use POSIX file locks
and do not support native Windows. From the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
priorlens --help
```

For development, use `pip install -e .`. For Stan/NUTS:

```bash
python -m pip install '.[stan]'
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

The default tutorials use conjugate calculations, quadrature and importance
sampling. See [installation](docs/installation.md) for batch execution.

## First example: Lupus posterior through the general channel

The example fits the 55-patient Lupus logistic model through Stan, without a
Python model adapter or the specialized Lupus posterior engine:

```text
examples/lupus_stan/
├── model.stan       # Likelihood and prior
├── data.json        # N, processed x and binary y
└── posterior.yaml   # Prior pair, location grid, targets and backend
```

The likelihood is `y[i] ~ Bernoulli(logistic(beta0 + beta1*x[i]))`. The two
coefficients have independent Student-t priors. Both intercept priors use
df=7, location=0 and scale=10. The baseline slope uses location=0 and scale=1000;
the informative slope uses scale=2.5 and locations 1, 3 and 5.

### 1. Write the model: model.stan

```stan
data {
  int<lower=1> N;
  vector[N] x;
  array[N] int<lower=0, upper=1> y;
  vector[2] prior_loc;
  vector<lower=0>[2] prior_scale;
  real<lower=0> prior_df;
}
parameters {
  real beta0;
  real beta1;
}
model {
  beta0 ~ student_t(prior_df, prior_loc[1], prior_scale[1]);
  beta1 ~ student_t(prior_df, prior_loc[2], prior_scale[2]);
  y ~ bernoulli_logit(beta0 + beta1 * x);
}
```

Prior hyperparameters are Stan data inputs, so the same compiled model serves
the baseline and every informative prior location. Parameter names come from
Stan's compiled metadata, rather than a model-name lookup.

### 2. Supply observations: data.json

The [complete JSON](examples/lupus_stan/data.json) contains N=55, all 55
processed x values and all 55 binary outcomes. Each patient is one row. For
example, this valid three-row JSON illustrates the file structure:

```json
{
  "N": 3,
  "x": [-0.331987285356289, 0.32013059659356446, 0.972248478543418],
  "y": [0, 1, 1]
}
```

The shipped example uses the full dataset. Covariates have already been
processed; neither Stan nor the runner re-estimates that transformation.

### 3. Choose priors and targets: posterior.yaml

```yaml
schema_version: 1
workflow: posterior
seed: 20260312
model: {stan: model.stan}
data: data.json

priors:
  baseline: {prior_loc: [0, 0], prior_scale: [10, 1000], prior_df: 7}
  interest: {prior_loc: [0, 3], prior_scale: [10, 2.5], prior_df: 7}
prior_grid:
  - {prior_id: slope_1, prior_loc: [0, 1]}
  - {prior_id: slope_3, prior_loc: [0, 3]}
  - {prior_id: slope_5, prior_loc: [0, 5]}

targets:
  - id: beta_joint
    coordinates: [{variable: beta0}, {variable: beta1}]
  - id: beta0
    coordinates: [{variable: beta0}]
  - id: beta1
    coordinates: [{variable: beta1}]

inference:
  backend: general
  chains: 4
  parallel_chains: 1
  warmup: 1000
  sampling: 1000
  adapt_delta: 0.95
  stan_min_ess: 400
  sobol_power: 15
  is_min_ess: 500
```

Paths are relative to the YAML. Prior keys match the Stan data block exactly;
each grid row replaces the named interest-prior fields. The baseline is fitted
once, and each posterior is shared by all three targets. `general` uses NUTS;
`importance` uses BridgeStan to evaluate the same Stan posterior kernel.

### 4. Run and read the posterior

With CmdStan installed as above:

```bash
priorlens validate --spec examples/lupus_stan/posterior.yaml
priorlens posterior --spec examples/lupus_stan/posterior.yaml --output results/lupus_posterior
```

For Stan-based IS with a mode/curvature mixture-t proposal:

```bash
python -m pip install '.[bridge]'
priorlens posterior --spec examples/lupus_stan/posterior.yaml --backend importance --output results/lupus_posterior_is
```

The IS budget is 2^15 proposal points per posterior. BridgeStan compiles the
Stan density, and the runner fits the proposal in unconstrained coordinates
with the Jacobian included. See [the Stan-only tutorial](docs/stan_posterior.md)
for proposal construction, parameter naming and output details.

Outputs include `summary.csv` (means, SDs and marginal quantiles),
`fit_*/posterior.npz` (natural-coordinate points and weights), target covariance
matrices and backend diagnostics. This example computes posteriors only;
method lists and M-grids belong to the EPSS workflows below.

Python API:

```python
from priorlens import posterior

posterior("examples/lupus_stan/posterior.yaml", "results/lupus_python")
```

## EPSS calculations

```bash
priorlens validate --spec examples/nig/all_methods.yaml
priorlens run --spec examples/nig/all_methods.yaml --output results/nig
```

This runs Reimherr, Wiesenfarth, Jones, CP and both Lin variants through the
same task runner. The YAML specifies data, priors, targets, candidate grids
and computational budgets. Quick examples use one prior location and small
budgets; study configurations provide the larger experiment grids.

To compute R/W mean and median from a shared risk bank:

```bash
priorlens risk-summary --spec examples/nig/quick.yaml --output results/nig_risk
```

Python API:

```python
from priorlens import validate, run

tasks = validate("examples/nig/all_methods.yaml")
result = run("examples/nig/all_methods.yaml", output="results/nig_python")
```

Combined Python demonstration:

```bash
python examples/run_demo.py nig --include-lin --output results/demo
```

## Three real-data tutorials

| Case | Example | Posterior computation |
|---|---|---|
| Midge / NIG | examples/nig/ | Conjugate marginals and joint conditional sampling |
| Eight Schools | examples/eight_schools/ | Collapsed quadrature and conditional Gaussian calculations |
| Lupus | examples/lupus/ | GH for risk; IS18 for Jones/CP/Lin |

```bash
priorlens run --spec examples/eight_schools/all_methods.yaml --output results/schools
priorlens risk-summary --spec examples/lupus/quick_risk.yaml --output results/lupus_rw
priorlens run --spec examples/lupus/matching_with_lin.yaml --output results/lupus_matching
```

See [case settings](docs/cases.md) for data, priors, targets and study grids.

## Plot saved results

```bash
python plotting/plot_realdata.py --output figures/realdata
```

The bundled tables produce three PDF/PNG pairs: NIG five targets, Schools four
targets and Lupus three targets, each with five method columns. Plotting reads
the saved results directly. Lupus simulation tables are also included in data/.

## Documentation

- [Model names and prior parameters](docs/models.md)
- [Preset catalog and method support](docs/methods/index.md)
- [Configuration: data, targets and grids](docs/configuration.md)
- [General model tutorial](docs/general_tutorial.md)
- [Stan-only posterior: YAML, Stan and JSON](docs/stan_posterior.md)
- [Posterior computation](docs/posterior.md) and [pseudo-data sampling](docs/sampling.md)
- [Jones](docs/methods/jones.md), [CP](docs/methods/cp.md), [Lin](docs/methods/lin.md)
- [Reimherr / Wiesenfarth](docs/methods/risk.md), [Clarke](docs/methods/clarke.md), [Morita](docs/methods/morita.md)
- [Method options](docs/options.md), [outputs and diagnostics](docs/diagnostics.md)
- [Complexity](docs/complexity.md), [architecture](docs/architecture.md)
- [Scientific description](description.md), [references](docs/references.md)

## Development checks

```bash
python -m unittest discover -s tests
python tools/check_release.py
```

The public API is in priorlens; numerical implementations are in prioreffect_v2
and prioreffect. source_map.json records source provenance, and
release_manifest.json records the shipped files.
