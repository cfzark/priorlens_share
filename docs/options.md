# Method option catalog

Options below belong in each methods entry. The [preset catalog](methods/index.md)
identifies the relevant family; [posterior settings](posterior.md) belong under
inference. Production configurations declare their method budgets explicitly.

## Identity and scope

| Keys | Meaning |
|---|---|
| `preset` | Registered method preset |
| `id` | Unique configured-method identifier |
| `name` | Family populated by the preset |
| `targets` | Target IDs evaluated by this method |
| `candidates` | Method-local values, unit and meaning |
| `paths` | Jones/CP/Lin path count |
| `posterior_draws` | Draw count for sampled target representations |
| `path_group` | Shared path-bank identity for compatible generic constructions |
| `focal_protocol` | fixed_observed_anchor_v1 for focal Schools |

## Jones/CP paths and selection

| Keys | Values / meaning |
|---|---|
| `scheme` | source_factorized, directional_factorized, nested_shared_source, nested_directional, nested_baseline_directional, bootstrap_augmentation, bootstrap_deletion, subsample_deletion |
| `tie` | positive / negative direction preference, then smallest m |
| `invalid_path_policy` | error / redraw_whole_path; redraw applies to bootstrap deletion |
| `max_path_redraws` | Maximum whole-path attempts |

## Jones distance

| Keys | Values / meaning |
|---|---|
| `quantile_grid_size` | Analytic-PPF integration points, default2000; cloud calculations use their fixed256 grid |
| `integration_rule` | quantile / empirical_sort (equal-size uniform clouds) |
| `squared` | Square the calculated distance |
| `projections` | Bivariate sliced-W2 direction count, default32 |
| `projection_rule` | random / nig; nig uses deterministic half-step angles |
| `projection_seed` | Random-direction seed |

## CP regions and objectives

| Keys | Values / meaning |
|---|---|
| `region` | equal_tail / central / pdf_hdr / exact_density / kde; a target-ID mapping is also accepted |
| `alphas` | Tail-probability vector, usually [.05] |
| `alpha_weights` | Nonnegative level weights, normalized in the objective |
| `quantile_rule` | weighted_cdf / numpy_linear (uniform weights) |
| `loss_family` | historical / direct |
| `gap_weight` | Nonnegative squared-gap coefficient |
| `cap_nominal` | Cap both coverages at nominal mass before all objective terms |
| `cache_regions` | Reuse prepared regions |
| `delta` | Absolute-adequacy tolerance |
| `delta_artifact` | Setting-matched no-conflict calibration file |
| `secondary` | gap / min_m among adequate candidates |

H1 uses historical, gap_weight=1, cap_nominal=false.
Capped D0 uses direct, gap_weight=0, cap_nominal=true.
See [CP](methods/cp.md) for formulas and region definitions.

## Lin

lin_ppd_v1 and lin_plugin_v1 accept id, targets, candidates, paths,
posterior_draws and alpha. Their fixed components are region=central,
loss_family=direct, gap_weight=0, cap_nominal=true and tie=min_abs_negative.
Choose the generating law through the preset. Candidate meaning is
informative_fresh_size_reduction. See [Lin](methods/lin.md) for seed and grouping rules.

## R/W components

| Keys | Values / meaning |
|---|---|
| `risk` | posterior_mse / squared_mean / posterior_variance |
| `risk_reps` | Risk pseudo-datasets per size |
| `sampling` | bootstrap / likelihood |
| `reference_rule` | baseline_posterior_mean / legacy_plugin |
| `theta_star` | Natural-parameter plug-in vector |
| `target_reference` | Reference vector in target coordinates |
| `risk_moments` | marginal / legacy_joint |
| `risk_coupling` | shared_by_size / independent; different sizes have independent streams |
| `risk_data_seed` | Unsigned64-bit shared-data seed |
| `risk_invalid_policy` | error / redraw_baseline_proper (NIG bootstrap) |
| `risk_max_redraws` | Redraw cap, default1000 |

risk-summary reduces the same raw bank by mean and median before matching.

## Clarke

| Keys | Values / meaning |
|---|---|
| `truncated` | Full KL / common-region truncated KL |
| `region` | hdr / central, optionally by target |
| `alpha` | Region tail probability, default.05; also used by Lin |
| `density_engine` | exact_density / kde |
| `max_candidates` | Exhaustive-enumeration bound |
| `central_tkl_orders` | Increasing integration orders |
| `central_tkl_tolerance` | Integration tolerance, default1e-6 |
| `quantile_rule` | weighted_cdf / numpy_linear for central regions |

The feasible candidate set and exclusions are recorded.

## Morita

| Keys | Values / meaning |
|---|---|
| `epsilon_prior` | Explicit epsilon prior for generic curvature |
| `hessian_step` | Finite-difference step for generic curvature |
| `prior_predictive_reps` | Monte Carlo budget for the applicable information construction |
| `epsilon_c` | Versioned epsilon variance inflation |
| `morita_design` | new_schools_empirical_s for Schools Case2 |
| `moment_tolerance` | Schools covariance numerical tolerance |

Versioned presets construct their epsilon priors from epsilon_c.
See [Morita](methods/morita.md) for model/target definitions.
