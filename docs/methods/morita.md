# Morita information matching

Morita weakens the informative prior to an epsilon prior and matches its
candidate posterior information to the original prior information.

| Preset | Information | Model / target |
|---|---|---|
| morita_2008_conjugate_1d_v2 | Analytic expected curvature | Normal-known, Gamma/Poisson, Beta/Binomial; affine scalar target |
| morita_nig_joint_trace_project_v2 | Joint trace curvature | NIG joint natural/log-variance coordinates |
| morita_curvature_v2 | Numerical log-density curvature | Density-supported target and explicit epsilon_prior |
| morita_2012_case2_schools_mc_v2 | Expected determinant of posterior precision | Schools population targets, lognormal tau, optimized backend |

## Conjugate entries

For variance inflation c=epsilon_c>1, scalar results are:

- Normal: sigma²/s0²*(1-1/c).
- Gamma/Poisson: rate b*(1-1/c).
- Beta/Binomial with L trials per row: (a+b)*(1-1/c)/L.

These entries are analytic. The NIG project entry uses the implemented trace
formula and canonical variance-inflated epsilon construction, requiring kappa>2.
Its definition uses the trace directly, with the project convention for degrees
of freedom.

## Schools Case2

Match det(Cov_pi(Z)^-1) to E_Y[det(Cov_epsilon(Z|Y_m)^-1)].
Generate new school summaries, draw s from the empirical design, and fix the
generating hyperparameters at prior means in the declared coordinates.
Nested prefixes couple candidate m values.

morita_design: new_schools_empirical_s counts new schools.
prior_predictive_reps sets the Monte Carlo budget for this fixed-hypermean law.
The lognormal tau prior supplies the required prior covariance; the target is
a scalar or joint population hyperparameter.

## Selection

Versioned entries interpolate the information curve at a crossing and report
the discrete epss_grid as well. With no crossing, they return the grid argmin
and match_bracketed=false. epsilon_c and numerical budgets are explicit controls.

The hierarchical example is configs/morita_schools_smoke.yaml.
