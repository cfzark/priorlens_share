# Description of the code and experiments

## 1. Purpose and reproducibility scope

This package accompanies `paper/realdata.tex`. It compares effective prior
sample size (EPSS) under Reimherr, Wiesenfarth, Jones, and CP definitions in
three models, each with simulation and real-data examples.

There are two distinct reproducibility levels:

1. **Exact plotted numerical inputs.** Seven compact CSV tables contain every
   displayed EPSS, and the simulation SDs where applicable. Their SHA256 hashes
   are checked before drawing. No original result archive or fitting is needed.
2. **New computations.** The numerical source packages, C++ source, historical
   NIG runners, configurations, and portable wrappers are supplied. The wrappers
   permit small tests and fresh full-precision runs. They are not a byte-for-byte
   reconstruction of the historical distributed execution, and no full-grid
   rerun is certified by a smoke test.

The supplied TeX preserves the current author edits: six figures are active,
and the NIG v=0.25 figure is commented out. All seven figure assets and source
tables remain available; the plotting command redraws all seven.

The source snapshot includes components for additional models/methods because
the three examples share these modules. The manuscript and portable entry
points select only the cases and four displayed methods described here.

## 2. What the four columns mean

For a target transformation T, posterior-mean reference z, and pseudo-data D,
the posterior-MSE loss is

`L_R(D) = trace(Cov(T | D)) + ||E(T | D) - z||^2`,

whereas the squared-mean loss is

`L_W(D) = ||E(T | D) - z||^2`.

Reimherr uses empirical bootstrap pseudo-data and matches informative risk at
n to baseline risk at n+m. Wiesenfarth uses plug-in likelihood pseudo-data
and matches baseline risk at n to informative risk at n-m. R/W are not merely
different labels on the same loss or sample-size direction.

For the current real-data NIG, Schools, and Lupus implementations, z is the
baseline posterior mean of the target, `E_b[T(theta) | y]`; for nonlinear T
this is generally not `T(E_b[theta | y])`. Joint risk losses use their stated
coordinates without automatic whitening. Historical NIG target-specific
settings remain in the retained source and JSON configurations.

Risk aggregation is **mean** for NIG/Schools and **median** for Lupus. The
median is taken over all internal pseudo-data losses before selecting m;
it is neither the posterior median nor the median selected EPSS across
observed datasets. It is a different risk functional, not a generic denoiser.

Jones minimizes Wasserstein discrepancies (sliced distances for joint targets)
within each path, then averages signed selections. The current model-specific
real-data/Schools/Lupus versions draw one informative-posterior parameter per
path, shared across directions; future candidates are non-nested over m.
The historical NIG simulation implementations are supplied without relabeling
them as the current generic source-path implementation.

For reciprocal coverages c1 and c2, the displayed CP loss is

`H1 = (c1 - c2)^2 + (min(c1, c2) - 0.95)^2`.

CP uses bootstrap deletion: retain the first n-m records of a length-n
bootstrap sample. Thus m=0 is already a resampled dataset, not the original
observed sample. Selection occurs separately within each path; exact
directional ties prefer negative EPSS, with the smallest m within a direction.
There is no absolute-adequacy threshold or unmatched gate in these figures.
The figure column is labeled simply **CP**.

## 3. Normal / NIG case

### Simulation

- Likelihood: `X_i | mu, sigma^2 ~ Normal(mu, sigma^2)`, n=20.
- DGP: mu=0, sigma^2=1; 50 datasets, data seed 321+r.
- Informative prior: `mu | sigma^2 ~ Normal(mu0, 0.2 sigma^2)` and
  `sigma^2 ~ IG(kappa, eta)`, with density proportional to
  `(sigma^2)^(-kappa-1) exp(-eta/sigma^2)`.
- Baseline: independent Jeffreys density proportional to `1/sigma^2`.
- Set a=prior mean of sigma^2 and v=prior variance; then
  `kappa=2+a^2/v` and `eta=a(kappa-1)`.
- Grid: mu0=-4:0.5:4 and
  a={0.25,0.35,0.5,0.7,0.85,1,1.15,1.3,1.5,1.75,2,2.5,3,4,5}.
  The two figures use v=0.25 and 20: 255 cells per variance.
- Targets: joint (mu,log sigma^2), joint (mu,sigma^2), and scalar mu,
  log sigma^2, sigma^2, without the real-data rescaling below.
- R/W use 1,000 inner losses; Jones/CP use 500 paths. Candidate grids are
  target-specific and are retained in each configuration and the manuscript.

These are historical computations. In particular, Wiesenfarth uses squared
posterior-mean error for the joint targets but posterior MSE for the scalar
targets. CP uses joint and scalar-scale HDRs and a central Student-t interval
for marginal mu. Those values are retained, not silently recomputed with the
newer central-rectangle implementation. `configs/nig_simulation/` records all
40 combinations of two variances, five targets, and four methods.

### Midge real data

The nine wing lengths (mm) are
`[1.64,1.70,1.72,1.74,1.82,1.82,1.82,1.90,2.08]`.
The informative prior is `mu | sigma^2 ~ Normal(mu0,sigma^2)` and
`sigma^2 ~ IG(0.5,0.005)`; the baseline is unchanged. The 82 prior locations
span 1.3--2.3, with finer spacing near 1.8--2.0 and the observed mean included.
Targets are mu/0.1, log(sigma^2/0.01), and their joint posterior.

R/W use mean aggregation of 1,000 losses, with signed grids -3:40 and -40:5.
Jones uses 0:40 additions; CP uses 0:5 deletions, each with 500 paths.
CP uses central intervals or calibrated central rectangles. Degenerate
bootstrap samples are conditionally redrawn according to the recorded rules.
The sample mean and original prior location 1.9 are reference lines, not truths.

## 4. Lupus logistic regression

The grouped raw predictor levels are [0,0.5,1,1.5,2]. Negative-response counts
are [31,2,2,0,2] and positive counts [5,0,2,6,5]: 55 patients, 18 positives.
The model is

`logit(p_i) = beta0 + beta1 * (x_i - mean(x))/(2*sample_sd(x))`.

The independent informative priors are t7(mu_beta0,10) and
t7(mu_beta1,2.5); the independent baseline priors are t7(0,10) and t7(0,1000).
Here the second argument is the Student-t scale, not its SD. Locations vary
on the 561-cell grid mu_beta0=-5:0.5:3, mu_beta1=-5:0.5:11.

The real-data figure has four targets: joint (beta0,beta1), scalar beta1,
joint (p(0),p(2)), and p(2)-p(0), with probabilities evaluated at raw x.
The simulation keeps the 55 predictors fixed and generates Bernoulli outcomes
at true beta=(-1,3), for 30 datasets. It has three targets: joint coefficients,
beta0, beta1. The simulation DGP uses the supplied stable-seed implementation
and data seed 20260817; do not replace it with an unrelated `seed+r` convention.

R/W use medians of 500 internal losses, with m=-20:20. Jones and CP use 500
paths, with additions or deletions respectively on m=0:20. Jones uses 32
projections for baseline-SD-standardized joint distances and unscaled scalar
Wasserstein distances. CP uses central intervals or jointly calibrated central
rectangles, not a product of separate 95% marginal intervals.

The five-level likelihood is evaluated using the C++ grouped-count engine,
adaptive 21/31/41-node Gauss--Hermite rules, and seeded importance-sampling
fallback. Grouping does not change patient-level bootstrap resampling.
Baseline risk fits are reused across prior locations; median aggregation
requires the complete raw loss bank, not medians of shard means/medians.

The standalone runner recomputes Jones when requested; it needs no historical
Jones cache. It keeps the model-specific seed rule. Shared CP scoring also
computes D0 internally at negligible additional fitting cost, but the portable
selected-result table retains only H1. Numerical convergence is not evidence
that 500 risk draws stabilize every risk-matching argmin.

## 5. Eight Schools

The data are y=[28,8,-3,7,-1,1,18,12] with known
s=[15,10,16,11,9,11,10,18]. The model is

`y_j | theta_j ~ Normal(theta_j,s_j^2)`,
`theta_j | mu,tau ~ Normal(mu,tau^2)`.

One EPSS unit is one school summary, not one pupil. Both priors use
`mu ~ Normal(0,25^2)`. Baseline tau is half-t4(scale 25); informative tau is
half-t4(scale tau0), independent of mu. The 100 scales are
{0.25:0.25:10} union {10.5:0.5:30} union {31:1:50}.

Targets are joint (mu/10,log(tau/10)), joint (mu/10,tau^2/100), scalar
log(tau/10), and scalar tau^2/100. They are hyperparameter targets, not
school-specific marginal effects conditional on fixed hyperparameters.

The simulation uses 50 datasets: theta_j~Normal(8,6^2), then
y_j~Normal(theta_j,s_j^2), with fixed s and RNG seed 321+r.
R/W use mean risk with 1,000 pseudo-datasets; their signed grids are -6:20
and -20:6. Jones uses 500 paths and 0:20 additions; CP uses 500 paths and
0:6 bootstrap deletions. Auxiliary schools draw standard errors from the
observed design. Posterior computation uses analytic conditional Gaussian
mu distributions and adaptive quadrature in log tau, not MCMC.

## 6. Reading and editing the figures

- The CP column label is always `CP`; the exact implementation remains in
  configuration/provenance rather than the column title.
- NIG and Lupus heatmaps use a shared zero-centered [-20,20] EPSS scale.
- Stars denote simulation truths only. The Lupus real-data crosshair marks
  its unpenalized MLE, not a correct-prior point.
- Schools curves average 50 dataset-specific selections. Shading is one
  between-dataset SD, not an SE, CI, or internal numerical MC error.
- Schools prior scale 25 equals the baseline. The simulation x=6 line marks
  the numerical true heterogeneity, not a claim that prior scale 6 is correct.
- The real-data and simulation target sets intentionally differ in NIG and
  Lupus; do not fabricate uncomputed target rows to make layouts match.

## 7. Data, verification, and limitations

`data/manifest.json` records compact-table hashes, aggregation rules, and CP
scope. `paper/V2_figure/realdata/manifest.json` records supplied asset hashes.
`bundle_manifest.json` inventories distributable files with relative paths.
`VALIDATION.md` reports the package checks actually run.

The package does not contain the full raw 30/50-replication fit archives.
The included plotting tables were exported from completed results; their
historical strict audits are not replayed against missing archives here.
Numerical values can be redrawn exactly from these tables, but this is not
independent scientific verification. New computations retain diagnostics and
write new outputs; candidate-endpoint and Monte Carlo stability checks remain
necessary before replacing any publication result.

The real datasets reproduce the grouped/numeric inputs already used by the
project; data-source attribution and publication permissions should be checked
by the authors before a public release. No software license is inferred.
