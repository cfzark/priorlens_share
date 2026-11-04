# Posterior computation

Posterior inference represents the distribution given a dataset and a prior.
Pseudo-data generation is configured separately in [sampling](sampling.md).

For a model written only in Stan, use the [posterior command](stan_posterior.md).
Its general backend uses NUTS; its importance backend uses BridgeStan density
and transformation interfaces. The table below describes the existing EPSS
posterior providers, including their built-in and Python-adapter inputs.

| Backend | Calculation | Interface |
|---|---|---|
| optimized | Registered model-specific calculation | Conjugacy, collapsed Schools, logistic GH |
| importance | Scrambled Sobol mixture-Student-t IS | logkernel and coordinate transforms |
| general | CmdStan NUTS | Stan source and model adapter |
| auto | optimized if registered, otherwise general | Provider dispatch |

## Optimized calculations

NIG updates are analytic. Scalar Student-t/inverse-Gamma marginals supply
CDF/PPF and finite moments. Joint samples draw variance followed by the
conditional Normal mean.

Schools integrates local effects, applies quadrature on log tau, and uses
analytic conditional Gaussian mu calculations. The grid expands its domain
with a fixed node count. Its stopping rule checks tail mass and accepts either
a sufficiently small moment change or at least one domain expansion; the
recorded moment_change supports subsequent refinement.

Logistic GH represents the posterior with weighted points around its mode.
The order schedule checks changes in posterior moments. Region and quantile
accuracy can be studied separately using fixed posterior comparisons.

## Importance sampling

sobol_power and sobol_max_power specify the first and last powers of two.
Set them equal for a fixed budget. The default proposal is

.75*t5(mode,4cov) + .25*t5(mode,16cov),

where cov is the inverse local curvature and the matrices are t shape matrices.
Lupus centers the broad component at the prior location with a diagonal
prior-scale-squared shape. The unconstrained logkernel includes coordinate
Jacobians. Weights use the complete mixture denominator and log-sum-exp
normalization; raw weights are retained.

The numerical gates are ESS>=is_min_ess (default500), relative ESS>=.02,
maximum weight<=.02, raw-ratio GPD-MLE k<.7, split-mean discrepancy<=.15 posterior
SD. Repeated scrambles and fixed-posterior references measure errors in the
quantities used by the EPSS objective.

## NUTS

Defaults: four chains, 1000 warmup and 1000 sampling iterations per chain,
R-hat<=1.01, bulk/tail ESS>=400 and zero divergences. A second attempt uses
adapt_delta=.99 and maximum tree depth15, keeping iteration counts and seed.

## Representation and reuse

Ordinary IS/GH targets retain the complete weighted cloud. NUTS supplies
equal-weight chain draws. Analytic scalar targets expose CDF/PPF functions;
analytic joint targets use posterior_draws. Lin's joint targets use common
resampled natural-coordinate points across targets.

posterior_reuse: importance reweights an anchor posterior in the same parameter
space. If its checks fail, the selected backend refits and records the fallback.
Hierarchical adapters can collapse local effects to keep a common parameter space.

## Inference settings

| Keys | Meaning |
|---|---|
| backend | optimized / importance / general / auto |
| posterior_reuse | none / importance |
| sobol_power, sobol_max_power | First/last IS powers |
| is_min_ess | Absolute importance-ESS threshold |
| reuse_draws | Draw count for sampling an anchor without weighted support |
| gh_nodes | 1D Hermite order schedule; 2D uses its tensor product |
| quadrature_tolerance | GH moment-refinement threshold |
| school_grid_size | Starting log-tau node count, default801 |
| tail_tol | Schools tail-mass control, default1e-10 |
| school_convergence_tol | Schools moment-change threshold, default1e-5 |
| max_expansions | Maximum domain refinements, default8 |
| school_engine | reference / batched |
| school_batch_size, school_cache_size | Batch size and bounded cache capacity |
| school_cache_by_data | Reuse deterministic Schools fits across target seeds |
| school_cache_quantiles | Reuse scalar quantile solutions |
| school_quantile_engine | brentq / vectorized_newton |
| chains, parallel_chains | NUTS total and simultaneous chain counts |
| warmup, sampling | NUTS iterations per chain |
| adapt_delta | Initial NUTS target acceptance |
| stan_min_ess | Minimum monitored bulk/tail ESS |
| cmdstan, stan_cache | Installation and compilation-cache paths |

Use the controls for the selected backend. Schools acceleration settings apply
to optimized/auto with posterior_reuse: none.

[Stan diagnostics](https://mc-stan.org/docs/reference-manual/analysis.html)
provides the definitions of chain diagnostics.
