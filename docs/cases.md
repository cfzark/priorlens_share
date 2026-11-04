# Three real-data examples

Commands below run from the installed repository checkout. Quick configurations
use one prior location, B=4 risk datasets, S=2 paths and 512 target draws where
sampling is used. Study configurations specify the larger grids and budgets.

## NIG: midge wing lengths

Data in mm: 1.64,1.70,1.72,1.74,1.82,1.82,1.82,1.90,2.08.
Each row is one measurement. The likelihood is Normal(mu,sigma²). The informative
prior has mu|sigma²~Normal(mu0,sigma²), sigma²~IG(.5,.005); the baseline density
is 1/sigma² in dmu d(sigma²). The location study has 82 mu0 values in [1.3,2.3].

Five targets: (mu/.1,log(sigma²/.01)), (mu/.1,sigma²/.01), mu/.1,
log(sigma²/.01), sigma²/.01. Scalar posteriors are analytic; joint samples draw
sigma² then mu|sigma².

```bash
priorlens run --spec examples/nig/all_methods.yaml --output results/nig
priorlens risk-summary --spec examples/nig/quick.yaml --output results/nig_risk
priorlens run --spec examples/nig/lin_presets.yaml --output results/nig_lin
```

Quick mu0=1.9; risk grids [-1,0,1], Jones/CP [0,1], Lin [-1,0,1].
Study B1000/S500, 4096 target draws; R -3:40, W -40:5, Jones0:40,
CP0:5 and Lin-9:9. NIG bootstrap sampling uses the declared proper-posterior
conditioning rules.

## Eight Schools

y=[28,8,-3,7,-1,1,18,12], s=[15,10,16,11,9,11,10,18].
Each row is one school summary:

y_j|theta_j~Normal(theta_j,s_j²), theta_j|mu,tau~Normal(mu,tau²).

Both priors use mu~Normal(0,25²). Baseline tau~half-t4(25);
informative tau~half-t4(tau0), with 100 study scales from .25 to50.
Four targets: (mu/10,log(tau/10)), (mu/10,tau²/100), log(tau/10), tau²/100.

```bash
priorlens run --spec examples/eight_schools/all_methods.yaml --output results/schools
priorlens run --spec examples/eight_schools/lin_presets.yaml --output results/schools_lin
```

Quick tau0=6. Study B1000/S500; R -6:20, W -20:6, Jones0:20,
CP0:6 and Lin-8:8. Posterior computation uses collapsed log-tau quadrature
with 801 starting nodes and conditional Gaussian mu. New schools draw empirical
s values and fresh responses. Original s values remain fixed in outer simulations.

The separate Morita example uses a lognormal tau prior:
`configs/morita_schools_smoke.yaml`.

## Lupus logistic regression

The expanded data contain 55 patients and 18 positive responses. Raw x levels
are [0,.5,1,1.5,2], failures [31,2,2,0,2], successes [5,0,2,6,5].
The fixed transformation is x*=(x-mean(x))/(2 sample_sd(x)).

logit(p)=beta0+beta1*x*. Both intercept priors are t7(0,scale10).
Baseline beta1~t7(0,1000); informative beta1~t7(location,2.5).
The study varies location over -5:.5:11. Both coefficients are estimated.
Targets are joint(beta0,beta1), beta0 and beta1.

```bash
priorlens risk-summary --spec examples/lupus/quick_risk.yaml --output results/lupus_rw
priorlens run --spec examples/lupus/matching_with_lin.yaml --output results/lupus_matching
priorlens run --spec examples/lupus/lin_presets.yaml --output results/lupus_lin
```

Quick location=3. R/W use GH; Jones/CP/Lin use IS18. CP joint targets include
central and PDF-HDR; marginals use central intervals. Study B1000/S500:
R/W -20:20, Jones/CP0:20, Lin-20:20. risk-summary computes mean and median
before selecting m from a shared raw-risk bank.

## Saved tables and configurations

`plotting/plot_realdata.py` renders the bundled tables. Their computation and
the executable recipe settings are recorded separately:

| Detail | Saved comparisons | Executable recipes |
|---|---|---|
| Lupus joint Jones geometry | Observed-baseline SD scaling | Explicit target-coordinate scales |
| Lupus Lin joint support | Weighted posterior cloud | Common resampled natural-coordinate draws |
| Lupus IS k policy | k-only warnings retained in the campaign | All standard IS diagnostic gates applied |
| NIG paired coordinates | Historical target-RNG aliases | Target identities in the YAML |
| Execution | Specialized all-target campaign engines | Common manifest runner, including Lin target-group reuse |

Study recipes configure new calculations; plot reproduction reads the saved
values directly. Each table's source and export hashes are in data/manifest.json.

Saved real-data figures use NIG R/W mean+median, Schools mean, Lupus mean+median,
Jones, CP and Lin PPD. Vertical references are the NIG observed mean/prior1.9,
Schools equal-prior scale25 and Lupus slope MLE2.8569763869.

The Lupus simulation tables provide all-available cohorts (R/W23, other25)
and the common23 cohort. Bands represent between-dataset SD. Recorded numerical
warning metadata remains in the source tables.
