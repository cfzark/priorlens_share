# Pseudo-data sampling

The experiment has three repetition levels: outer datasets, internal paths/risk
datasets, and posterior points/draws.

| Law | Parameter source | New data |
|---|---|---|
| bootstrap | Empirical observed distribution | Complete rows with replacement |
| subsample deletion | Observed rows | Nested without-replacement subsets |
| likelihood | Fixed plug-in parameter | Empirical design rows and fresh responses |
| joint PPD | One posterior parameter draw per path | Empirical design rows and conditionally independent responses |

A PPD path shares its global parameter. Integrating this parameter induces
dependence among responses. Schools share mu,tau and have distinct local effects.

## Default constructions

| Method | Construction |
|---|---|
| Reimherr | Bootstrap; baseline N+m vs informative N |
| Wiesenfarth | Likelihood at baseline natural-coordinate posterior means; informative N-m vs baseline N |
| Jones source adaptation | Informative posterior parameter per path; independent additions across sizes/directions conditional on it |
| CP bootstrap deletion | Length-N bootstrap vector with nested N-m prefixes, shared by both directions |
| CP bootstrap augmentation | Observed data plus nested bootstrap additions |
| Lin PPD | Baseline posterior parameter per path; length-2N fresh vector with nested N-m prefixes |
| Lin plug-in | Same nested design with fixed baseline natural-coordinate means |

At m=0, augmentation compares the original observed posteriors. Bootstrap
deletion uses resampled N rows; Lin uses fresh N rows.

R/W sampling is configurable. shared_by_size couples compatible laws across
methods/priors at the same size; distinct sizes use independent streams.
The [preset catalog](methods/index.md) lists the PPD and empirical alternatives.

## Conditional data laws

For NIG/Jeffreys, bootstrap samples may have zero residual variation.
redraw_whole_path conditions matching paths on proper candidate posteriors;
redraw_baseline_proper conditions the risk dataset law. Rejection counts and
retry caps are recorded. Clarke uses the feasible enumeration of proper
baseline posteriors and records excluded candidate counts.

Focal Schools retains the original school ID/y/s exactly once, with distinct
IDs for auxiliary schools. The [model table](models.md) lists focal-method support.
