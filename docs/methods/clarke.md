# Clarke: KL and truncated KL

For each candidate size m, enumerate observed-row multisets with replacement.
Construct the baseline posterior P and compare it with informative prior Q in
the declared target. Select the minimum divergence over datasets and sizes.
The feasible set consists of candidates with proper baseline posteriors;
excluded counts are recorded.

| Preset | Objective |
|---|---|
| clarke_kl_v2 | Full KL |
| clarke_tkl_v2 | HDR-truncated KL |
| clarke_central_tkl_v2 | Central-region-truncated KL |

For truncation, construct a region R from P and normalize both distributions
on that common region:

KL(P|R || Q|R) = integral_R p log(p/q) / P(R) + log(Q(R)/P(R)).

Central uses scalar equal tails or a joint-mass-calibrated rectangle.
density_engine: exact_density uses normalized target densities;
kde uses a weighted Gaussian density estimate.

alpha sets region mass. central_tkl_orders and central_tkl_tolerance specify
integration refinement. max_candidates bounds exhaustive enumeration.
Candidates use meaning: pseudo_sample_size.

The central preset fixes central truncation. Conditional focal-school Clarke
uses the fixed-observed-anchor protocol. Numerical fit failures are recorded
separately from improper-candidate exclusions.

Run the example with the common CLI:
`configs/clarke_central_smoke.yaml`.
