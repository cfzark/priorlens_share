# Cost and method characteristics

Let G denote prior cells, T targets, K sizes, S matching paths, B risk datasets,
L posterior points, D projections and C(n) the cost of a posterior fit.

| Method | Fitting work | Reuse |
|---|---|---|
| R/W | O(G T B K C(n)) for per-target tasks | Shared baseline and size-indexed risk data |
| Jones/CP | O(G T S K C(n)), up to two directions | Shared path banks and posterior caches |
| Lin | O(G S K C(n)) for an all-target worker group | Fixed anchor, common paths, grouped targets and cached outputs |
| Clarke | sum_m binomial(n+m-1,m) candidate comparisons | Sufficient statistics / repeated candidates |
| Morita | Analytic formulas or O(B K) information calculations | Shared predictive prefixes and prior-only calculations |

Lin's grouped fitting removes T from the fit count; region/target work remains
target-specific. Independent task workers recompute their target group, so
run-task-group is the efficient layout for related Lin leaves.

## Posterior costs

NIG updates are constant-time after sufficient statistics; forming statistics
is O(n). Schools quadrature is roughly proportional to node count times school
count, with analytic conditional moments. Logistic GH/IS kernel evaluations
cost roughly O(nL), plus mode fitting. NUTS cost depends on gradient evaluation,
leapfrog steps, warmup and chain efficiency.

Scalar empirical W2 sorting costs O(L log L); sliced W2 repeats projection and
sorting over D directions. Central regions use quantiles and membership
probabilities. PDF-HDR adds density ranking. KDE cross-evaluation may be
quadratic in cloud size. Fixed-anchor region caching reduces repeated work.

## Measured Lupus campaign

At the completed real+25-dataset campaign's full grids/budgets, successful
workers used approximately 7970 CPU-hours for Jones, 4746 for CP and 4059 for
Lin: 307,183,156 per dataset. This measurement covers the specialized workers;
anchor/pilot/merge work, failures and allocated-idle time are separate costs.
Executable recipe and saved-campaign settings are in [cases](cases.md).

## Method characteristics

| Method | Emphasis | Main numerical/design controls |
|---|---|---|
| R/W | Posterior moments and risk | Target coordinates, reference, moment existence, risk budget |
| Jones | Distributional location/spread | Geometry, W2 quantiles, projection count |
| CP | Cross-region probabilities | Region definition, mass accuracy, objective and adequacy rule |
| Lin | One-sided observed anchor and nested signed sizes | Predictive law, grid boundaries, selected coverage |
| Clarke | Prior/posterior KL comparison | Density representation, truncation and enumeration size |
| Morita | Prior information | Epsilon construction, information metric and experimental unit |
