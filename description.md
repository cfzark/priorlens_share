# Scientific description

PriorLens compares a baseline prior b and a prior of interest pi through
Z=T(theta). Targets are scalar marginals and bivariate joint distributions.
Marginalization integrates the remaining model parameters.

The Stan-only posterior entry takes a Stan model, JSON observations and YAML
prior/target settings. It evaluates the baseline and informative prior grid
through NUTS or Stan-based IS. The main README demonstrates this workflow with
the Lupus logistic model. EPSS workflows additionally specify the following
data-generation and matching components.

An EPSS configuration consists of the prior pair, target coordinates,
pseudo-data law, candidate sizes, discrepancy, aggregation and tie rule.

| Family | Compared objects | Aggregation |
|---|---|---|
| Reimherr | Informative risk at N vs baseline risk at N+m | Reduce risk, then select m |
| Wiesenfarth | Baseline risk at N vs informative risk at N-m | Reduce risk, then select m |
| Jones | Observed posterior vs opposite-prior augmented posterior | Select signed m per path, then average |
| CP | Reciprocal posterior-region probabilities | Select signed m per path, then average |
| Lin | Fixed baseline observed posterior vs informative fresh-data posterior | Select signed m per nested path, then average |
| Clarke | Baseline candidate posterior vs informative prior | Minimize over candidate data and sizes |
| Morita | Informative-prior information vs epsilon-posterior information | Construct and match an information curve |

R/W compose bootstrap or plug-in likelihood with posterior-MSE,
squared-mean-error or variance losses. Their mean and median versions reduce
the same raw-risk bank. Jones uses scalar W2 and bivariate sliced W2. CP offers
central, PDF-HDR and KDE-HDR regions. Lin uses central regions with capped D0.

All families are configured with presets and use the same manifest execution
and result format. Three real-data tutorials cover conjugate Normal, hierarchical
Normal and logistic models. Custom 1D/2D adapters illustrate IS and NUTS.

The repository includes quick calculations, study-budget configurations and
saved numerical tables for plotting. [Case settings](docs/cases.md) specify
the computation used by each.
