"""Explicit Morita definitions; historical curvature runner stays unchanged.

See docs/MORITA_VERSIONS.md for source equations and intentional adaptations.
No external baseline, observed responses, or posterior predictive is used.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, logsumexp

from .core import DefinitionError, NumericalFailure, digest, seed_for
from .moments import check_target_moments


VERSIONS = {
    "morita_2008_conjugate_1d_v2",
    "morita_nig_joint_trace_project_v2",
    "morita_2012_case2_schools_mc_v2",
}


def validate_version(model, data, prior, target, method):
    version = method.get("preset")
    if version not in VERSIONS:
        return
    c = method.get("epsilon_c")
    if not np.isscalar(c) or isinstance(c, (bool, str)) or not np.isfinite(c) or c <= 1:
        raise DefinitionError("Versioned Morita requires explicit finite epsilon_c > 1")
    if "epsilon_prior" in method:
        raise DefinitionError("Versioned Morita constructs its epsilon prior; do not also supply epsilon_prior")
    if target.focal_unit_id:
        raise DefinitionError("2012 Case 1/focal Morita is not implemented by the hyperparameter Case 2 entry")
    if version == "morita_2008_conjugate_1d_v2":
        families = {"normal_known": "normal", "poisson_gamma": "gamma", "beta_binomial": "beta"}
        if (families.get(model.name) != prior.get("family") or target.dimension != 1
                or target.coordinates[0].variable != model.variables[0]
                or target.coordinates[0].transform != "identity"):
            raise DefinitionError("2008 conjugate entry requires Normal/Gamma/Beta prior and affine natural scalar target")
    elif version == "morita_nig_joint_trace_project_v2":
        if model.name != "nig" or prior.get("family") != "nig" or target.dimension != 2:
            raise DefinitionError("NIG trace entry requires full NIG prior and joint (mu, sigma2) target")
        coords = {x.variable: x for x in target.coordinates}
        if (set(coords) != {"mu", "sigma2"} or coords["mu"].transform != "identity"
                or coords["sigma2"].transform not in {"identity", "log"}):
            raise DefinitionError("NIG trace supports affine mu and sigma2 or log(sigma2)")
        # Canonical epsilon construction preserves finite natural-coordinate variances.
        if prior["kappa"] <= 2:
            raise DefinitionError("NIG canonical variance-inflated epsilon requires kappa > 2")
    else:
        if model.name != "eight_schools" or prior.get("tau_kind") != "lognormal":
            raise DefinitionError("Case 2 reference currently supports Eight Schools with lognormal tau only")
        for x in target.coordinates:
            if not (x.variable == "mu" and x.transform == "identity" or
                    x.variable == "tau" and x.transform in {"identity", "square", "log"}):
                raise DefinitionError("Case 2 requires scalar/joint hyperparameter targets")
        if method.get("morita_design") != "new_schools_empirical_s":
            raise DefinitionError("Declare morita_design: new_schools_empirical_s; fixed-K within-school data is a different model")
        for key in ("moment_tolerance",):
            value = method.get(key, 1e-4)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise DefinitionError(f"{key} must be finite and positive")
    check_target_moments(model, model.subset(data, []), prior, target, 2)


def epsilon_prior(model, prior, target, c):
    q = dict(prior)
    if model.name == "normal_known":
        q["sd"] *= np.sqrt(c)
    elif model.name == "poisson_gamma":
        q["shape"] /= c
        q["rate"] /= c
    elif model.name == "beta_binomial":
        q["a"] /= c
        q["b"] /= c
    elif model.name == "nig":
        mean_v = prior["eta"] / (prior["kappa"] - 1)
        q["kappa"] = 2 + (prior["kappa"] - 2) / c
        q["eta"] = mean_v * (q["kappa"] - 1)
        q["tau0_sq"] *= c
    elif model.name == "eight_schools":
        # Weaken only the hyperparameter(s) of interest, keeping nuisance prior fixed.
        for x in target.coordinates:
            if x.variable == "mu":
                q["mu_sd"] *= np.sqrt(c)
            else:
                l, v = np.log(prior["tau_scale"]), prior["tau_log_sd"] ** 2
                if x.transform == "log":
                    new_l, new_v = l, c * v
                else:
                    power = 2 if x.transform == "square" else 1
                    new_v = np.log1p(c * np.expm1(power ** 2 * v)) / power ** 2
                    new_l = l + .5 * power * (v - new_v)
                q["tau_scale"], q["tau_log_sd"] = float(np.exp(new_l)), float(np.sqrt(new_v))
    model.validate_prior(q)
    return q


def select_curve(rows, desired, candidate_key, *, log_values=False):
    """Piecewise-linear information interpolation, not a spline of |difference|.

    This numerical choice is explicit: papers allow interpolation, without
    mandating this particular interpolant. No extrapolation past the M-grid.
    """
    ordered = sorted(rows, key=lambda r: r["m"])
    values = np.array([r[candidate_key] for r in ordered], dtype=float)
    if not np.all(np.isfinite(values)) or not np.isfinite(desired):
        raise NumericalFailure("Nonfinite Morita information curve")
    # One common factor preserves both crossings and absolute-difference argmins.
    if log_values:
        shift = max(float(values.max()), desired)
        v, d = np.exp(values - shift), float(np.exp(desired - shift))
        with np.errstate(divide="ignore"):
            ranking_losses = np.maximum(values, desired) + np.log(-np.expm1(-abs(values - desired)))
    else:
        v, d = values, desired
        ranking_losses = abs(v - d)
    losses = abs(v - d)
    ix = min(range(len(ordered)), key=lambda i: (ranking_losses[i], ordered[i]["m"]))
    roots = [float(r["m"]) for r, value in zip(ordered, values) if value == desired]
    for i in range(len(values) - 1):
        left, right = values[i:i + 2]
        if (left < desired < right) or (right < desired < left):
            if not log_values:
                fraction = (desired - left) / (right - left)
            elif left < right:
                fraction = np.exp(desired - right) * (-np.expm1(left - desired)) / (-np.expm1(left - right))
            else:
                fraction = (-np.expm1(desired - left)) / (-np.expm1(right - left))
            roots.append(float(ordered[i]["m"] + fraction * (ordered[i + 1]["m"] - ordered[i]["m"])))
    best_grid = ordered[ix]["m"]
    chosen = min(roots) if roots else float(best_grid)
    for r, loss in zip(ordered, losses):
        r["loss"] = float(loss)
        if log_values:
            r["loss_common_log_scale"] = float(shift)
    return {"status": "matched", "epss": chosen, "epss_grid": best_grid,
            "boundary_rate": float(chosen == ordered[-1]["m"]),
            "match_bracketed": bool(roots), "crossing_count": len(roots),
            "selection": "piecewise_linear_information_smallest_crossing_else_grid_argmin",
            "finite_grid_only": True}


def classic_curve(model, prior, q, target, grid):
    c = target.coordinates[0]
    if model.name == "normal_known":
        desired, start = 1 / prior["sd"] ** 2, 1 / q["sd"] ** 2
        slope = 1 / model.options.get("sigma", 1.) ** 2
    elif model.name == "poisson_gamma":
        mean = prior["shape"] / prior["rate"]
        desired, start = (prior["shape"] - 1) / mean ** 2, (q["shape"] - 1) / mean ** 2
        slope = 1 / mean
    else:
        mean = prior["a"] / (prior["a"] + prior["b"])
        desired = (prior["a"] - 1) / mean ** 2 + (prior["b"] - 1) / (1 - mean) ** 2
        start = (q["a"] - 1) / mean ** 2 + (q["b"] - 1) / (1 - mean) ** 2
        slope = model.options.get("trials", 20) / (mean * (1 - mean))
    desired, start, slope = (float(x * c.scale ** 2) for x in (desired, start, slope))
    rows = [{"m": int(m), "prior_curvature": desired, "candidate_curvature": start + int(m) * slope} for m in grid]
    return desired, rows, (desired - start) / slope


def nig_curve(prior, q, target, grid):
    coords = {x.variable: x for x in target.coordinates}
    a, b, t = prior["kappa"], prior["eta"], prior["tau0_sq"]
    v = b / (a - 1)
    aq, bq, tq = q["kappa"], q["eta"], q["tau0_sq"]
    if coords["sigma2"].transform == "identity":
        desired = np.array([1 / (t * v), (a - 3.5) / v ** 2])
        start = np.array([1 / (tq * v), (aq - 3.5) / v ** 2])
        slope = np.array([1 / v, (.5 + t) / v ** 2])
        point = v
    else:
        point = float(np.log(b) - digamma(a))
        iv = np.exp(-point)
        desired = np.array([1 / t, b]) * iv
        start = np.array([1 / tq, bq]) * iv
        slope = np.array([1, .5 * v * (1 + t)]) * iv
    scales = np.array([coords["mu"].scale ** 2, coords["sigma2"].scale ** 2])
    desired, start, slope = (float(x @ scales) for x in (desired, start, slope))
    rows = [{"m": int(m), "prior_curvature": desired, "candidate_curvature": start + int(m) * slope} for m in grid]
    return desired, rows, (desired - start) / slope, point


def school_prior_moments(prior, target):
    means, variances = [], []
    l, v = np.log(prior["tau_scale"]), prior["tau_log_sd"] ** 2
    for c in target.coordinates:
        if c.variable == "mu":
            mean, var = prior["mu_mean"] / c.scale, (prior["mu_sd"] / c.scale) ** 2
        elif c.transform == "log":
            mean, var = (l - np.log(c.reference)) / c.scale, v / c.scale ** 2
        else:
            p = 2 if c.transform == "square" else 1
            mean = np.exp(p * l + .5 * p ** 2 * v) / c.scale
            var = mean ** 2 * np.expm1(p ** 2 * v)
        means.append(mean)
        variances.append(var)
    return np.array(means), np.diag(variances)


def precision_logdet(covariance):
    if not np.all(np.isfinite(covariance)):
        raise NumericalFailure("Nonfinite Morita covariance")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise NumericalFailure("Morita requires positive definite covariance; no eigenvalue clipping") from error
    return float(-2 * np.log(np.diag(chol)).sum())


def school_grid_moments(post, target):
    """Full covariance, including Cov(mu, T(tau)), without posterior resampling."""
    columns, conditional_variances = [], []
    for c in target.coordinates:
        if c.variable == "mu":
            columns.append(post.mu_cond_mean / c.scale)
            conditional_variances.append(post.mu_cond_var / c.scale ** 2)
        else:
            columns.append(c.forward(np.exp(post.eta)))
            conditional_variances.append(np.zeros(len(post.eta)))
    values = np.column_stack(columns)
    mean = post.masses @ values
    delta = values - mean
    covariance = (delta.T * post.masses) @ delta + np.diag(post.masses @ np.column_stack(conditional_variances))
    precision_logdet(covariance)
    return mean, covariance


def school_checked_moments(post, target, provider, method):
    import copy
    from .quadrature import SchoolPosterior
    from .quadrature import trapezoid_weights
    if not isinstance(post, SchoolPosterior):
        raise DefinitionError("2012 Schools reference requires optimized collapsed_grid; IS/Stan not yet certified for covariance determinants")
    previous = school_grid_moments(post, target)
    tolerance = method.get("moment_tolerance", 1e-4)
    for refinement in range(1, 4):
        # Expand the domain AND halve spacing; simply doubling nodes in the
        # same truncated range cannot establish moment-tail convergence.
        size = 3 * (len(post.eta) - 1) + 1
        width = post.eta[-1] - post.eta[0]
        refined = copy.copy(post)
        refined.eta = np.linspace(post.eta[0] - width / 4, post.eta[-1] + width / 4, size)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            kernel = refined.log_eta_kernel(refined.eta)
            log_mass = kernel + np.log(trapezoid_weights(refined.eta))
            refined.masses = np.exp(log_mass - logsumexp(log_mass))
            refined.mu_cond_mean, refined.mu_cond_var = refined.conditional_mu(refined.eta)
        if not np.all(np.isfinite(refined.masses)) or not np.all(np.isfinite(refined.mu_cond_var)):
            raise NumericalFailure("Morita moment-domain expansion overflow; epsilon/grid requires a different numerical engine")
        post = refined
        current = school_grid_moments(post, target)
        scale = np.sqrt(np.diag(current[1]))
        error = max(float(np.max(abs(current[0] - previous[0]) / scale)),
                    float(np.max(abs(current[1] - previous[1]) / np.outer(scale, scale))),
                    abs(precision_logdet(current[1]) - precision_logdet(previous[1])))
        if error <= tolerance:
            provider.accuracy.append({"engine": "morita_covariance_refinement", "converged": True,
                "grid_size": size, "eta_bounds": [float(post.eta[0]), float(post.eta[-1])],
                "refinement": "expanded_domain_and_halved_spacing", "moment_error": error, "moment_tolerance": tolerance})
            return current
        previous = current
    raise NumericalFailure(f"Morita full covariance did not converge: error={error}, tolerance={tolerance}")


def schools_curve(model, data, prior, q, target, method, grid, provider, seed):
    desired = precision_logdet(school_prior_moments(prior, target)[1])
    # Full hyperparameter plug-in; nuisance tau uses the natural tau mean.
    tau_coordinate = next((c for c in target.coordinates if c.variable == "tau"), None)
    l, v = np.log(prior["tau_scale"]), prior["tau_log_sd"] ** 2
    tau = (np.exp(l) if tau_coordinate is not None and tau_coordinate.transform == "log"
           else np.exp(l + v) if tau_coordinate is not None and tau_coordinate.transform == "square"
           else np.exp(l + v / 2))
    hyper_mean = [prior["mu_mean"], float(tau)]
    reps = method.get("prior_predictive_reps", 16)
    banks = [model.simulate(hyper_mean, int(max(grid)), np.random.default_rng(seed_for(seed, "morita2012_data", b)), data) for b in range(reps)]
    rows = []
    for m in grid:
        logs = []
        if m == 0:
            logs = [precision_logdet(school_prior_moments(q, target)[1])] * reps
        else:
            candidates = [model.subset(bank, np.arange(int(m))) for bank in banks]
            posts = provider.fit_many(candidates, q, [seed_for(seed, "morita2012_fit", int(m), b) for b in range(reps)])
            for candidate, post in zip(candidates, posts):
                check_target_moments(model, candidate, q, target, 2)
                _, covariance = school_checked_moments(post, target, provider, method)
                logs.append(precision_logdet(covariance))
        log_mean = float(logsumexp(logs) - np.log(reps))
        normalized = np.exp(np.array(logs) - log_mean)
        rows.append({"m": int(m), "prior_log_precision_determinant": desired,
                     "candidate_log_mean_precision_determinant": log_mean,
                     "relative_mc_se": float(np.std(normalized, ddof=1) / np.sqrt(reps)) if reps > 1 else None})
    return desired, rows, {"hyperparameter_plugin_natural": hyper_mean,
        "pseudo_data_checksums": [digest(x) for x in banks], "mc_reps": reps,
        "data_coupling": "nested_prefixes_within_replication", "design_s": list(data["s"]),
        "averaging": "mean_of_precision_determinants_not_precision_of_mean_covariance"}


def run_version(model, data, prior, target, method, grid, provider, seed):
    validate_version(model, data, prior, target, method)
    if not len(grid) or any(int(m) != m or m < 0 for m in grid) or len(set(grid)) != len(grid):
        raise DefinitionError("Morita grid must contain distinct nonnegative integers")
    version, c = method["preset"], method["epsilon_c"]
    q = epsilon_prior(model, prior, target, c)
    metadata = {"morita_version": version, "epsilon_prior": q, "epsilon_c": c,
                "external_baseline_used": False, "observed_responses_used": False,
                "epsilon_sensitivity_verified": False, "unit": model.unit}
    if version == "morita_2012_case2_schools_mc_v2":
        desired, rows, audit = schools_curve(model, data, prior, q, target, method, grid, provider, seed)
        result = select_curve(rows, desired, "candidate_log_mean_precision_determinant", log_values=True)
        metadata.update(audit, source="Morita 2012 equations (11), (14)-(16)",
                        design=method["morita_design"], fidelity="Case2_extended_definition_project_new_school_design")
    else:
        if version == "morita_2008_conjugate_1d_v2":
            desired, rows, unrestricted = classic_curve(model, prior, q, target, grid)
            metadata.update(source="Morita 2008 equation (3), Algorithm 1",
                fidelity="2008_scalar_definition_Poisson_is_project_likelihood_application")
        else:
            desired, rows, unrestricted, point = nig_curve(prior, q, target, grid)
            metadata.update(source="Morita 2008 trace-inspired project NIG extension",
                fidelity="project_adaptation_not_full_2008_df_adjusted_reproduction",
                epsilon_coordinates="natural_mu_sigma2", scale_evaluation_point=point,
                degrees_of_freedom_adjustment="not_applied_legacy_project_definition")
        result = select_curve(rows, desired, "candidate_curvature")
        metadata["epss_unrestricted"] = float(unrestricted)
        metadata["engine"] = "analytic_prior_predictive_curvature"
    return {**result, **metadata}, rows, []
