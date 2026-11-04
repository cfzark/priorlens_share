# Method presets

A preset selects a method definition. model.name selects the likelihood and
prior interface; inference.backend selects posterior computation. These three
fields are independent configuration layers. See [model names](../models.md).

## Catalog

Use the exact identifiers below in `methods[].preset`. For example, the
source-faithful Jones identifier is `jones_source_faithful` (the shorthand
`jones_faithful` is not a registered alias). `id` is an optional result label.

| Preset | Family / data construction |
|---|---|
| reimherr_project_bootstrap_v2 | R: bootstrap, posterior MSE, baseline N+m vs informative N |
| wiesenfarth_project_plugin_v2 | W: plug-in likelihood, squared posterior-mean error, informative N-m vs baseline N |
| jones_source_faithful | Jones source construction for the three classic conjugate scalar models |
| jones_source_adaptation_v2 | Jones source construction extended to other supported models/targets |
| jones_nested_shared_source_v2 | Jones with nested shared-source PPD paths |
| jones_bootstrap_augmentation_v2 | Jones with nested bootstrap additions |
| jones_bootstrap_deletion_v2 | Jones with nested bootstrap deletions |
| jones_subsample_deletion_v2 | Jones with nested without-replacement deletions |
| cp_source_faithful | CP objective on source-factorized paths |
| cp_directional_ppd | CP with direction-specific posterior-predictive sources |
| cp_nested_directional_v2 | CP with nested directional PPD paths |
| cp_nested_baseline_directional_v2 | CP with nested baseline-directional PPD paths |
| cp_bootstrap_augmentation_v2 | CP with nested bootstrap additions |
| cp_bootstrap_deletion_v2 | CP with nested bootstrap deletions |
| cp_subsample_deletion_v2 | CP with nested without-replacement deletions |
| lin_ppd_v1 | Lin with one baseline posterior parameter draw per path |
| lin_plugin_v1 | Lin with baseline posterior means as plug-in parameters |
| clarke_kl_v2 | Clarke full KL |
| clarke_tkl_v2 | Clarke HDR-truncated KL |
| clarke_central_tkl_v2 | Clarke central-region-truncated KL |
| morita_curvature_v2 | Numerical curvature with explicit epsilon prior |
| morita_2008_conjugate_1d_v2 | Analytic scalar conjugate information |
| morita_nig_joint_trace_project_v2 | NIG joint trace-curvature adaptation |
| morita_2012_case2_schools_mc_v2 | Schools Case2 determinant/covariance matching |

## Reading a preset name

- The first word selects the method family: `lin`, `jones`, `cp`, etc.
- `ppd` specifies posterior-predictive generation; `plugin` fixes the
  generating parameter at a posterior-mean estimate.
- `source` specifies the Jones source-style path construction; `faithful`
  preserves its scalar conjugate implementation, while `adaptation` extends
  that construction to other models and targets.
- `nested` means candidate datasets are prefixes of a shared generated vector.
- `bootstrap` samples observed rows with replacement; `subsample` uses them
  without replacement. `augmentation` appends rows; `deletion` retains fewer rows.
- `kl`, `tkl`, `central_tkl` select full, HDR-truncated and central-truncated KL.
- `2008` and `2012_case2` identify the Morita construction. `project` labels
  the project's specified implementation or extension.
- `_v1` and `_v2` are definition identifiers, not accuracy rankings or posterior
  sampler choices. They are part of the preset name.

## Jones and CP: what each path preset changes

Let X contain N observed rows, P=q_pi(theta|X), and B=q_b(theta|X).
In augmentation, the positive comparison is P versus q_b(theta|X+Y_plus);
the negative comparison is B versus q_pi(theta|X+Y_minus).

| Preset(s) | Parameter sources per path | Candidate construction | Default direction tie |
|---|---|---|---|
| jones_source_faithful; jones_source_adaptation_v2; cp_source_faithful | One draw from P shared by both directions | Fresh additions for each m and direction, conditionally independent given that draw | Positive |
| cp_directional_ppd | One draw from P for positive; independent draw from B for negative | Fresh additions for each m and direction | Positive |
| jones_nested_shared_source_v2 | One draw from P shared by both directions | Generate a separate maximum-length vector per direction; take prefixes across m | Negative |
| cp_nested_directional_v2 | One draw from P for positive; independent draw from B for negative | Separate maximum-length vectors; prefixes across m | Negative |
| cp_nested_baseline_directional_v2 | Two independent draws from B, one per direction | Separate maximum-length vectors; prefixes across m | Negative |
| jones_bootstrap_augmentation_v2; cp_bootstrap_augmentation_v2 | Empirical observed-row distribution | One bootstrap addition vector, shared by directions; append its first m rows to X | Negative |
| jones_bootstrap_deletion_v2; cp_bootstrap_deletion_v2 | Empirical observed-row distribution | One length-N bootstrap, shared by directions; retain its first N-m rows | Negative |
| jones_subsample_deletion_v2; cp_subsample_deletion_v2 | Random permutation of observed rows | One permutation, shared by directions; retain its first N-m rows | Negative |

For deletion, positive compares B with q_pi(theta|D_(N-m)); negative compares
P with q_b(theta|D_(N-m)). At m=0 bootstrap deletion uses a full bootstrap
sample, whereas subsample deletion uses all observed rows. Both families select
within each path and then average signed m; within a direction ties select
the smallest m.

`jones_source_faithful` applies to normal_known, poisson_gamma and beta_binomial.
`jones_source_adaptation_v2` uses the same source/path/tie construction for
NIG, Schools, Lupus and compatible adapters. Jones uses scalar W2 or bivariate
sliced W2. See [Jones](jones.md) for distance computation and target geometry.

The CP presets select the path construction; region and loss are separate
components. Their defaults are `region: equal_tail`, `alphas: [0.05]`,
`loss_family: historical`, `gap_weight: 1`, `cap_nominal: false` (H1).
Set `region: central` for scalar intervals or joint calibrated rectangles.
`cp_source_faithful` means CP loss on the source-style paths. See [CP](cp.md)
for PDF-HDR, KDE-HDR, capped D0 and adequacy selection.

## Lin: PPD versus plug-in

Both Lin presets fix B and compare it with q_pi(theta|Y_(1:N-m)). They generate
a fresh length-2N vector per path and use nested prefixes for signed m in
[-N,N]. Thus m=0 uses N fresh observations, not the observed dataset.

- `lin_ppd_v1`: draw one global parameter vector from B per path and generate
  the entire response vector conditionally on it.
- `lin_plugin_v1`: use the baseline posterior means of the natural parameters
  as a fixed generating vector. Response and design generation still varies
  between paths.

Both fix central regions and capped D0, select m per path, and average the
selections. Ties prefer smallest |m|, then negative m. Region, objective and tie
are fixed by these presets; alpha, budgets, targets and the grid are supplied
in the configuration. See [Lin](lin.md) for hierarchical generation and seeds.

## Risk, Clarke and Morita presets

| Preset | Defining defaults and distinction |
|---|---|
| reimherr_project_bootstrap_v2 | Bootstrap data; posterior variance plus squared posterior-mean error; informative risk at N vs baseline risk at N+m |
| wiesenfarth_project_plugin_v2 | Likelihood data at baseline posterior-mean parameters; squared posterior-mean error; baseline risk at N vs informative risk at N-m |
| clarke_kl_v2 | Enumerate observed-row multisets, compare each baseline posterior with the informative prior using full KL |
| clarke_tkl_v2 | Same enumeration; use a baseline-posterior HDR and normalize both distributions on that region before KL |
| clarke_central_tkl_v2 | Same enumeration and normalization, with a central interval or calibrated rectangle |
| morita_curvature_v2 | Numerical log-density curvature with a user-specified epsilon prior |
| morita_2008_conjugate_1d_v2 | Analytic expected-curvature formulas for the three scalar conjugate models; epsilon_c controls prior variance inflation |
| morita_nig_joint_trace_project_v2 | NIG joint trace-curvature formula and canonical variance-inflated epsilon prior; kappa>2 |
| morita_2012_case2_schools_mc_v2 | Match prior precision determinant to expected epsilon-posterior precision determinant; lognormal tau, nested new-school data at fixed prior-mean hyperparameters |

R/W default to baseline posterior target means as the risk reference,
coordinate-wise posterior moments and shared pseudo-data by sample size.
Both accept bootstrap or likelihood sampling and alternative risk components.
The common runner reduces mean risk before selection; `risk-summary` computes
paired mean and median reductions. See [risk](risk.md), [Clarke](clarke.md)
and [Morita](morita.md) for formulas and configurable components.

## Candidate-grid meanings

| Preset family / path | candidates.meaning | Interpretation of grid value m |
|---|---|---|
| Jones/CP augmentation, including source and directional PPD | number_augmented | Append m rows; direction supplies the EPSS sign |
| Jones/CP deletion | number_deleted | Retain N-m rows; direction supplies the EPSS sign |
| Lin | informative_fresh_size_reduction | Generate a fresh informative dataset of size N-m; m itself is signed |
| Reimherr | baseline_size_offset | Baseline pseudo-dataset size N+m |
| Wiesenfarth | informative_size_reduction | Informative pseudo-dataset size N-m |
| Clarke / Morita | pseudo_sample_size | Candidate pseudo-dataset size m |

In mixed-method configurations, set `candidates` inside each method entry
when meanings or values differ. `unit` is observation for NIG/Lupus and school
for the Schools new-unit design. The [option catalog](../options.md) documents
component overrides; resolved task files store the actual settings used.

## Required posterior capabilities

| Family | Posterior / prior representation | Target support |
|---|---|---|
| R/W | Certified coordinate moments | Scalar/joint; mean or median raw-risk reduction |
| Jones | PPFs or samples/weighted points, finite second moments | Scalar W2 / bivariate sliced W2 |
| CP central | Quantiles/CDFs or weighted support | Scalar intervals / joint rectangles |
| CP PDF-HDR | Target density/kernel and posterior support | Targets with an available density in their own coordinates |
| CP KDE-HDR | Nonsingular weighted support | Scalar/joint KDE regions |
| Lin | Built-in predictive generator and posterior provider | Population targets in Normal-known, NIG, Schools, Lupus |
| Clarke | Informative prior and candidate posterior densities | Density-supported targets, including specified focal protocol |
| Morita | Epsilon prior and information construction | Version-specific model/prior/target combinations |

All presets run through `priorlens run`. `risk-summary` additionally reports
paired mean/median R/W results from a shared raw-risk bank.
The [method option catalog](../options.md) describes configurable components.
