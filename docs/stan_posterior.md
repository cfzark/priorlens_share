# A posterior from Stan, JSON and YAML

The `priorlens posterior` command evaluates a prior pair and a prior-location
grid on one observed dataset. Its model definition is a self-contained Stan
file. A Python adapter, response generator and EPSS M-grid are unnecessary for
this posterior-only workflow.

The runnable example is [Lupus](../examples/lupus_stan/posterior.yaml); its
[Stan model](../examples/lupus_stan/model.stan) and [JSON](../examples/lupus_stan/data.json)
are shown and explained in the [README](../README.md).

## Configuration

| Field | Meaning |
|---|---|
| schema_version: 1 | Configuration format |
| workflow: posterior | Observed-data posterior computation |
| model.stan | Self-contained Stan file, relative to YAML |
| data | JSON filename or inline mapping of observation/design inputs |
| priors.baseline / interest | Stan data-block hyperparameters; same input keys for both priors |
| prior_grid | Unique prior_id plus complete replacement values for selected interest-prior fields |
| targets | One or two parameter coordinates per target; multiple targets share each fit |
| inference.backend | general for CmdStan NUTS; importance for BridgeStan IS |
| seed | Positive integer below 2^32; stable prior/data identities derive per-fit seeds |

The data mapping and prior mappings have disjoint keys. They are merged into
the exact data payload supplied to Stan. For example, replacing prior_loc with
[0,5] keeps the intercept prior location at zero and sets the slope location
to five. A change to a vector field replaces the whole vector.

## Parameter identification

The runner obtains parameter names from compiled Stan interfaces. It does not
classify the model as logistic, Normal or another built-in. Scalar parameters
are selected by name; vector/matrix entries use Stan notation such as beta[1]
or Sigma[1,2]. YAML variables refer to the parameters block. Coordinate
transformations identity/log/logit/square and explicit scales use the same
target implementation as the EPSS runner.

All model parameters are sampled, including parameters outside the selected
target. Selecting a marginal keeps the corresponding columns of the full
posterior; selecting a two-coordinate target retains their joint draws.

## NUTS

`backend: general` compiles through CmdStanPy and runs NUTS. Specify chains,
warmup and sampling. parallel_chains controls simultaneous chains; the example
uses one at a time. adapt_delta defaults to .95 and max_treedepth to12.

All parameters are checked for R-hat<=1.01, bulk/tail ESS>=stan_min_ess and zero
divergences. The example sets stan_min_ess=400. Raw chain CSVs, parameter
diagnostics and the iteration-by-chain parameter array are retained.

## Importance sampling

Install the bridge extra, then select backend importance. The same Stan file
supplies the unnormalized posterior and parameter transforms through BridgeStan.
There is no Python copy of the Lupus likelihood and no specialized GH dispatch.

1. Optimize the log posterior in Stan's unconstrained coordinates, including
   the transformation Jacobian. initial_unconstrained optionally supplies an
   initial vector; the default is zero in those coordinates.
2. Form covariance Sigma from inverse negative log-density Hessian at the mode.
3. Draw scrambled Sobol proposal points from
   0.75*t5(mode,4Sigma)+0.25*t5(mode,16Sigma), with Student-t shape matrices.
4. Evaluate raw importance weights using the complete mixture density.
5. Transform points back to natural parameters and retain normalized weights.

sobol_power fixes the budget at 2^power points. Unlike the specialized Lupus
proposal, both mixture components here are centered at the mode. Proposal
center, curvature, optimizer status and raw log weights are saved.

The gates are absolute ESS>=is_min_ess (default500), relative ESS>=.02,
maximum weight<=.02, raw-ratio GPD k<.7 and split-mean discrepancy<=.15 SD.
Failed checks save the point cloud and diagnostics, write failure.json and stop
without a completed-run marker. Changing backend is an explicit user choice.

## Output layout

```text
results/lupus_posterior/
├── inputs/                  # Frozen Stan source and observation JSON
├── build/                   # Compiled model reused across prior cells
├── resolved_spec.json
├── provenance.json
├── fit_0000/                # Baseline
│   ├── stan_data.json
│   ├── posterior.npz
│   ├── diagnostics.json
│   ├── target_covariance.json
│   └── done.json
├── fit_0001/                # First informative prior cell
├── ...
├── summary.csv
├── diagnostics.json
└── done.json
```

NUTS also saves a chains/ directory per fit. posterior.npz contains points,
weights and parameter_names; NUTS adds chain_draws, while IS adds unconstrained
points and log_weight_raw. Quantiles use the common weighted-quantile rule.
Joint targets contribute one summary row per coordinate and a covariance
matrix; their marginal rows intentionally repeat the corresponding marginals.

The example makes four fits: one baseline plus three informative priors. Each
has three targets and four summary rows, for sixteen summary rows in total.
Use a fresh output directory for each run. Input and artifact hashes accompany
completion markers.

## Lupus data and scientific settings

The JSON contains the same ordered 55 binary outcomes and processed covariates
as examples/lupus/quick_matching.yaml. Patient frequencies are expanded into
individual rows. The processing rule is x=(x_raw-0.509090909090909)/1.5334650799790488;
it is applied before writing the JSON. Both coefficients remain estimated.

The prior-scale parameter is Student-t scale, not standard deviation. Baseline:
beta0~t7(0,10), beta1~t7(0,1000). Informative: beta0~t7(0,10), beta1~t7(location,2.5).
The short location grid 1,3,5 illustrates configuration and posterior comparison.

## Moving to EPSS

This command supplies observed-data posteriors. The current EPSS run command
uses built-ins or Python adapters with data operations and response generators;
see [the general EPSS tutorial](general_tutorial.md). Adding method names to a
posterior-only YAML is rejected, rather than ignored. The two workflows have
separate configuration schemas and explicit CLI commands.

Technical interfaces: [BridgeStan Python](https://brianward.dev/bridgestan/latest/languages/python.html)
and [CmdStanPy](https://mc-stan.org/cmdstanpy/api.html).
