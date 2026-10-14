"""Method families compose inference, paths, target losses and selection."""
from __future__ import annotations

import itertools
from collections import OrderedDict
from math import comb
import numpy as np

from .core import DefinitionError, InadmissiblePosterior, UndefinedMoment, UncertifiedMoment, NumericalFailure, digest, seed_for, seed_scope, sampling_data
from .inference import PosteriorProvider
from .losses import Wasserstein, Coverage, kl, posterior_risk, risk_from_moments
from .paths import build_paths
from .selection import BidirectionalSelector
from .risk_components import RiskSampler, RiskDatasetBank, resolve_reference, sampling_for, validate_risk_settings, validate_risk_moments
from .moments import check_target_moments, marginal_summary


PRESETS = {
    "jones_source_faithful": {"name": "jones", "scheme": "source_factorized", "tie": "positive", "classic_only": True},
    "cp_source_faithful": {"name": "cp", "scheme": "source_factorized", "tie": "positive"},
    "cp_directional_ppd": {"name": "cp", "scheme": "directional_factorized", "tie": "positive"},
    "jones_source_adaptation_v2": {"name": "jones", "scheme": "source_factorized", "tie": "positive"},
    "jones_nested_shared_source_v2": {"name": "jones", "scheme": "nested_shared_source", "tie": "negative"},
    "cp_nested_directional_v2": {"name": "cp", "scheme": "nested_directional", "tie": "negative"},
    "cp_nested_baseline_directional_v2": {"name": "cp", "scheme": "nested_baseline_directional", "tie": "negative"},
    "reimherr_project_bootstrap_v2": {"name": "reimherr", "risk": "posterior_mse", "sampling": "bootstrap", "reference_rule": "baseline_posterior_mean", "risk_moments": "marginal", "risk_coupling": "shared_by_size"},
    "wiesenfarth_project_plugin_v2": {"name": "wiesenfarth", "risk": "squared_mean", "sampling": "likelihood", "reference_rule": "baseline_posterior_mean", "risk_moments": "marginal", "risk_coupling": "shared_by_size"},
    "clarke_kl_v2": {"name": "clarke", "truncated": False},
    "clarke_tkl_v2": {"name": "clarke", "truncated": True},
    "clarke_central_tkl_v2": {"name": "clarke", "truncated": True, "region": "central"},
    "morita_curvature_v2": {"name": "morita"},
    "morita_2008_conjugate_1d_v2": {"name": "morita"},
    "morita_nig_joint_trace_project_v2": {"name": "morita"},
    "morita_2012_case2_schools_mc_v2": {"name": "morita"},
}
for _name in ["cp", "jones"]:
    for _scheme in ["bootstrap_augmentation", "bootstrap_deletion", "subsample_deletion"]:
        PRESETS[f"{_name}_{_scheme}_v2"] = {"name": _name, "scheme": _scheme, "tie": "negative"}


def resolve_method(spec):
    spec = {"preset": spec} if isinstance(spec, str) else dict(spec)
    preset = spec.get("preset")
    if preset not in PRESETS:
        raise DefinitionError(f"Unknown method preset {preset}; available: {sorted(PRESETS)}")
    allowed = {"preset", "id", "name", "scheme", "tie", "paths", "path_group", "posterior_draws",
               "quantile_grid_size", "projections", "projection_rule", "projection_seed", "squared",
               "region", "alphas", "gap_weight", "alpha_weights", "delta", "delta_artifact", "secondary",
               "candidates", "risk", "risk_reps", "theta_star", "target_reference", "truncated", "alpha",
               "density_engine", "max_candidates", "epsilon_prior", "hessian_step", "prior_predictive_reps",
               "integration_rule", "quantile_rule", "loss_family", "cap_nominal", "invalid_path_policy", "max_path_redraws",
               "epsilon_c", "morita_design", "moment_tolerance", "focal_protocol", "targets", "cache_regions",
               "central_tkl_orders", "central_tkl_tolerance", "sampling", "reference_rule", "risk_moments", "risk_coupling", "risk_data_seed",
               "risk_invalid_policy", "risk_max_redraws"}
    if set(spec) - allowed:
        raise DefinitionError(f"Unknown method settings: {set(spec)-allowed}")
    resolved = {**PRESETS[preset], **spec, "id": spec.get("id", preset)}
    if resolved["name"] != PRESETS[preset]["name"]:
        raise DefinitionError("A preset cannot change its method family")
    if "focal_protocol" in resolved and resolved["focal_protocol"] != "fixed_observed_anchor_v1":
        raise DefinitionError("Unknown focal protocol")
    if "cache_regions" in resolved and (resolved["name"] != "cp" or not isinstance(resolved["cache_regions"], bool)):
        raise DefinitionError("cache_regions is a boolean CP optimization")
    from .morita_versions import VERSIONS
    if {"epsilon_c", "morita_design", "moment_tolerance"} & spec.keys() and preset not in VERSIONS:
        raise DefinitionError("Versioned Morita controls require a versioned Morita preset")
    if preset in VERSIONS and {"hessian_step", "epsilon_prior"} & spec.keys():
        raise DefinitionError("Versioned Morita does not use hessian_step or a user epsilon_prior")
    if preset in VERSIONS and preset != "morita_2012_case2_schools_mc_v2" and {"morita_design", "moment_tolerance"} & spec.keys():
        raise DefinitionError("morita_design/moment_tolerance belong to the Schools covariance entry")
    if preset == "jones_source_faithful" and any(resolved[k] != PRESETS[preset][k] for k in ["scheme", "tie"]):
        raise DefinitionError("Use an explicitly named adaptation when changing source-faithful Jones semantics")
    for field in ["paths", "posterior_draws", "quantile_grid_size", "projections", "risk_reps", "max_candidates", "prior_predictive_reps", "max_path_redraws"]:
        if field in resolved and (not isinstance(resolved[field], int) or isinstance(resolved[field], bool) or resolved[field] <= 0):
            raise DefinitionError(f"{field} must be a positive integer")
    if resolved.get("invalid_path_policy", "error") not in {"error", "redraw_whole_path"}:
        raise DefinitionError("Unknown invalid_path_policy")
    if resolved.get("invalid_path_policy") == "redraw_whole_path" and resolved.get("scheme") != "bootstrap_deletion":
        raise DefinitionError("Whole-path redraw is only supported for bootstrap_deletion")
    if resolved["name"] == "clarke":
        validate_clarke_settings(resolved)
    if resolved["name"] in {"reimherr", "wiesenfarth"}:
        validate_risk_settings(resolved)
    elif {"sampling", "reference_rule", "risk_moments", "risk_coupling", "risk_data_seed", "risk_invalid_policy", "risk_max_redraws"} & spec.keys():
        raise DefinitionError("sampling/reference_rule are risk components; Jones/CP use scheme")
    return resolved


def validate_clarke_settings(method):
    regions = method.get("region", "hdr")
    regions = list(regions.values()) if isinstance(regions, dict) else [regions]
    if not regions or any(r not in {"hdr", "central"} for r in regions):
        raise DefinitionError("Clarke truncation region must be hdr or central")
    if "region" in method and not method.get("truncated", False):
        raise DefinitionError("A Clarke region requires truncated: true")
    if method.get("preset") == "clarke_central_tkl_v2" and (not method.get("truncated") or any(r != "central" for r in regions)):
        raise DefinitionError("clarke_central_tkl_v2 must retain central truncation")
    if method.get("truncated", False):
        alpha = method.get("alpha", .05)
        if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise DefinitionError("Clarke alpha must lie in (0,1)")
    if method.get("density_engine", "exact_density") not in {"exact_density", "kde"}:
        raise DefinitionError("Unknown Clarke density engine")
    if "central" in regions:
        from .central_tkl import validate_controls
        validate_controls(method.get("central_tkl_orders", (16, 32, 64, 128, 256, 512)), method.get("central_tkl_tolerance", 1e-6))
        if method.get("quantile_rule", "weighted_cdf") not in {"weighted_cdf", "numpy_linear"}:
            raise DefinitionError("Unknown central TKL quantile rule")
    elif {"central_tkl_orders", "central_tkl_tolerance"} & method.keys():
        raise DefinitionError("central_tkl controls require a central truncation region")


def matching(model, data, info, base, target, method, grid, provider, seed, bank=None):
    from .focal import validate_focal
    focal = validate_focal(model, data, target, method)
    p0, q0 = provider.fit(data, info, seed), provider.fit(data, base, seed)
    if bank is None:
        check = None
        if hasattr(model, "check_posterior_admissibility"):
            def check(current):
                for prior in (info, base):
                    model.check_posterior_admissibility(current, prior)
        bank = build_paths(model, data, p0, q0, grid, method["scheme"], method.get("paths", 8), np.random.default_rng(seed), target.focal_unit_id,
            admissibility_check=check, invalid_path_policy=method.get("invalid_path_policy", "error"), max_path_redraws=method.get("max_path_redraws", 1000))
    if focal:
        for candidate in bank.candidates:
            focal.check(candidate["positive_data"])
            focal.check(candidate["negative_data"])
    compare = (Wasserstein(grid_size=method.get("quantile_grid_size", 2000), projections=method.get("projections", 32), projection_rule=method.get("projection_rule", "random"), squared=method.get("squared", False), seed=method.get("projection_seed", 20260312), integration_rule=method.get("integration_rule", "quantile"))
               if method["name"] == "jones" else Coverage(tuple(method.get("alphas", [.05])), method.get("region", "equal_tail"), method.get("gap_weight", 1.), method.get("alpha_weights"), method.get("quantile_rule", "weighted_cdf"), method.get("loss_family", "historical"), method.get("cap_nominal", False), method.get("cache_regions", False)))
    selector = BidirectionalSelector(method["tie"], method.get("delta"), method.get("secondary", "gap"))
    draws = method.get("posterior_draws", 1024)
    measures = OrderedDict()
    def measure(post):
        if method["name"] == "jones":
            check_target_moments(model, post.data, post.prior, target, 2)
        key = digest([sampling_data(post.data), post.prior, post.engine, target.id])
        if key not in measures:
            measures[key] = post.target(target, draws, seed_for(seed, "target", key))
        elif provider.settings.get("school_engine") == "batched":
            measures.move_to_end(key)
        if provider.settings.get("school_engine") == "batched":
            while len(measures) > provider.settings.get("school_cache_size", 256):
                measures.popitem(last=False)
        return measures[key]
    rows = []
    path_posts, active_path = {}, None
    for candidate in bank.candidates:
        m, path = candidate["m"], candidate["path"]
        deletion = method["scheme"].endswith("deletion")
        if provider.settings.get("school_engine") == "batched" and path != active_path:
            active_path, path_posts = path, {}
            path_candidates = [r for r in bank.candidates if r["path"] == path]
            for direction in ["positive", "negative"]:
                selected_prior = info if (direction == "positive") == deletion else base
                anchor = p0 if selected_prior is info else q0
                posts = provider.fit_many([r[direction + "_data"] for r in path_candidates], selected_prior,
                    [seed_for(seed, "fit", path, direction, r["m"]) for r in path_candidates], anchor)
                path_posts.update({(direction, r["m"]): post for r, post in zip(path_candidates, posts)})
        for direction in ["positive", "negative"]:
            selected_prior = info if (direction == "positive") == deletion else base
            anchor = p0 if selected_prior is info else q0
            current = candidate[direction + "_data"]
            try:
                post = path_posts[(direction, m)] if path_posts else provider.fit(current, selected_prior, seed_for(seed, "fit", path, direction, m), anchor)
                fixed = q0 if selected_prior is info else p0
                components = compare(measure(fixed), measure(post))
            except Exception as error:
                error.context = {"path": path, "m": m, "direction": direction}
                raise
            rows.append({"path": path, "m": m, "direction": direction, **components})
    selected = [{"path": path, **selector([r for r in rows if r["path"] == path])} for path in sorted({r["path"] for r in rows})]
    matched = [r["signed_m"] for r in selected if r["status"] == "matched"]
    result = {"epss": float(np.mean(matched)) if matched else None,
        "epss_meaning": "conditional_on_matched" if method.get("delta") is not None else "mean_pathwise_signed_m",
        "match_rate": len(matched) / len(selected), "unmatched_rate": 1 - len(matched) / len(selected),
        "numerical_failure_rate": 0., "boundary_rate": np.mean([r["boundary_selected"] for r in selected]),
        "path_checksum": bank.checksum, "source_checksum": bank.source_checksum,
        "path_diagnostics": bank.diagnostics,
        "status": "matched" if matched else "unmatched", "n_paths": len(selected)}
    if focal:
        result["focal_design"] = focal.audit()
    return result, rows, selected


def risk_matching(model, data, info, base, target, method, grid, provider, seed, *, raw_risks=None):
    """Mean-risk reference, optionally exporting each full pseudo-data loss bank.

    raw_risks is a caller-owned list, not a scientific setting. It receives
    one independent array per estimate call (fixed first, then grid order).
    Existing scalar-only caches are recomputed if their raw bank is unavailable.
    No RNG calls, posterior loss, averaging or selection rules are changed.
    """
    from .focal import validate_focal
    focal = validate_focal(model, data, target, method)
    rng = np.random.default_rng(seed)
    n = len(data["y"])
    validate_risk_moments(model, data, info, base, target, method, grid)
    theta, reference, reference_audit = resolve_reference(model, data, base, target, method, provider, seed)
    sampling = sampling_for(method)
    policy = method.get("risk_invalid_policy", "error")
    sampler = RiskSampler(model, data, sampling, theta, target.focal_unit_id,
        invalid_policy=policy, admissibility_prior=base if policy != "error" else None,
        max_redraws=method.get("risk_max_redraws", 1000))
    coupling = method.get("risk_coupling", "independent")
    bank = RiskDatasetBank(sampler, method.get("risk_data_seed", seed)) if coupling == "shared_by_size" else None
    moment_mode = method.get("risk_moments", "legacy_joint")
    metric = method.get("risk", "posterior_mse")
    order = 1 if metric == "squared_mean" else 2
    moment_engines = set()
    risk_cache_hits = 0
    B = method.get("risk_reps", 16)
    raw_cache = {}
    def estimate(prior, count):
        nonlocal risk_cache_hits
        values = []
        # Sampling remains sequential in the old RNG order. Posterior engines
        # consume separate seeds, so collecting a batch does not change paths.
        datasets = [bank.draw(count, b) if bank is not None else sampler.draw(count, rng) for b in range(B)]
        fit_seed = method.get("risk_data_seed", seed) if bank is not None else seed
        for b, current in enumerate(datasets):
            try:
                if hasattr(model, "check_posterior_admissibility"):
                    model.check_posterior_admissibility(current, prior)
                check_target_moments(model, current, prior, target, order)
            except Exception as error:
                error.context = {**getattr(error, "context", {}), "sample_size": int(count), "risk_rep": b, "prior": prior}
                raise
        risk_key = None
        if provider.settings.get("school_cache_by_data", False):
            risk_key = digest([prior, [(c.variable, c.transform, c.scale, c.reference) for c in target.coordinates],
                target.focal_unit_id, metric, reference, moment_mode, method.get("posterior_draws", 1024), fit_seed,
                [sampling_data(d) for d in datasets]])
            if risk_key in provider.risk_cache and (raw_risks is None or risk_key in raw_cache):
                cached, engines = provider.risk_cache[risk_key]
                provider.risk_cache.move_to_end(risk_key)
                moment_engines.update(engines)
                risk_cache_hits += 1
                provider.accuracy.append({"engine": "school_risk_estimate_cache", "converged": True,
                    "cache_hit": True, "risk_cache_key": risk_key})
                if raw_risks is not None:
                    raw_risks.append(raw_cache[risk_key].copy())
                return cached
        try:
            posts = provider.fit_many(datasets, prior, [seed_for(fit_seed, "risk", digest(sampling_data(d))) for d in datasets])
        except Exception as error:
            error.context = {**getattr(error, "context", {}), "sample_size": int(count), "prior": prior, "stage": "risk_batch_fit"}
            raise
        for b in range(B):
            current = datasets[b]
            try:
                post = posts[b]
                target_seed = seed_for(fit_seed, "risk_target", b, count)
                if moment_mode == "marginal":
                    mean, variance, audit = marginal_summary(post, target, order, method.get("posterior_draws", 1024), target_seed)
                    moment_engines.update(audit["engines"])
                    value = risk_from_moments(mean, variance, reference, metric)
                else:
                    value = posterior_risk(post.target(target, method.get("posterior_draws", 1024), target_seed), reference, metric)
                    moment_engines.add("legacy_joint_measure")
                values.append(value)
            except Exception as error:
                error.context = {**getattr(error, "context", {}), "sample_size": int(count), "risk_rep": b, "prior": prior}
                raise
        value = float(np.mean(values))
        if raw_risks is not None:
            raw_risks.append(np.asarray(values, dtype=float).copy())
            if risk_key is not None:
                raw_cache[risk_key] = raw_risks[-1].copy()
        if risk_key is not None:
            provider.risk_cache[risk_key] = (value, sorted(moment_engines))
            while len(provider.risk_cache) > provider.settings.get("school_cache_size", 256):
                provider.risk_cache.popitem(last=False)
        return value
    if method["name"] == "reimherr":
        fixed = estimate(info, n)
        values = [estimate(base, n + int(m)) for m in grid]
    else:
        fixed = estimate(base, n)
        values = [estimate(info, n - int(m)) for m in grid]
    errors = np.abs(np.asarray(values) - fixed)
    index = int(np.argmin(errors))
    rows = [{"m": int(m), "reference_risk": fixed, "candidate_risk": value, "loss": float(error)} for m, value, error in zip(grid, values, errors)]
    return {"epss": int(grid[index]), "status": "matched", "boundary_rate": float(index in (0, len(grid)-1)), "risk": metric,
        **({"focal_design": focal.audit()} if focal else {}),
        "risk_components": {"sampling": sampling, "loss": metric, **reference_audit,
            "matching": "baseline_n_plus_m_vs_informative_n" if method["name"] == "reimherr" else "informative_n_minus_m_vs_baseline_n",
            "aggregation": "mean", "coupling": coupling, "risk_moments": moment_mode,
            "risk_cache_hits": risk_cache_hits,
            "moment_engines": sorted(moment_engines), "moment_order_required": order,
            "data_seed": bank.seed if bank is not None else None,
            "data_law_hash": bank.law if bank is not None else None,
            **({"bootstrap_admissibility": sampler.diagnostics()} if policy != "error" else {}),
            "data_checksums": bank.checksums() if bank is not None else None,
            "selection": "discrete_absolute_gap_first_grid_tie"}}, rows, []


def clarke(model, data, info, base, target, method, grid, provider, seed):
    from .focal import validate_focal, PROTOCOL
    focal = validate_focal(model, data, target, method)
    if focal and method.get("focal_protocol") != PROTOCOL:
        raise DefinitionError("Conditional Clarke requires explicit focal_protocol")
    validate_clarke_settings(method)
    n = len(data["y"])
    pool = focal.pool if focal else np.arange(n)
    budget = method.get("max_candidates", 100000)
    expected = sum((comb(len(pool) + int(m) - 1, int(m)) if len(pool) else 0) if m else 1 for m in grid)
    if expected > budget:
        raise DefinitionError(f"Clarke requires {expected} candidates, exceeding explicit budget {budget}")
    empty = focal.anchor if focal else model.subset(data, [])
    p_info = provider.fit(empty, info, seed).target(target, method.get("posterior_draws", 2048), seed)
    rows, candidate_id = [], 0
    totals = {"enumerated": 0, "admissible": 0, "excluded_improper": 0, "numerical_failed": 0}
    for m in grid:
        best, best_id, best_diagnostics = np.inf, None, None
        counts = dict.fromkeys(totals, 0)
        for indices in itertools.combinations_with_replacement(pool, int(m)):
            current_id = candidate_id
            candidate_id += 1  # IDs/seeds retain the FULL enumeration, including exclusions.
            counts["enumerated"] += 1
            current = model.subset(data, indices, auxiliary=True)
            if focal:
                current = focal.attach(current)
            try:
                posterior = provider.fit(current, base, seed_for(seed, "clarke", current_id))
            except InadmissiblePosterior:
                counts["excluded_improper"] += 1
                continue
            except NumericalFailure as error:
                counts["admissible"] += 1
                counts["numerical_failed"] += 1
                error.context = {"m": int(m), "candidate_id": current_id, "counts_at_m": counts}
                raise
            counts["admissible"] += 1
            try:
                measure = posterior.target(target, method.get("posterior_draws", 2048), seed_for(seed, "clarke_target", current_id))
                diagnostics = None
                if method.get("region") == "central":
                    from .central_tkl import central_truncated_kl
                    comparison = central_truncated_kl(measure, p_info, method.get("alpha", .05),
                        method.get("density_engine", "exact_density"), method.get("quantile_rule", "weighted_cdf"),
                        method.get("central_tkl_orders", (16, 32, 64, 128, 256, 512)), method.get("central_tkl_tolerance", 1e-6))
                    value, diagnostics = comparison["loss"], comparison["region_diagnostics"]
                else:
                    value = kl(measure, p_info, method.get("truncated", False), method.get("alpha", .05), method.get("density_engine", "exact_density"))
                if not np.isfinite(value):
                    raise NumericalFailure("Clarke candidate comparison is nonfinite")
            except NumericalFailure as error:
                counts["numerical_failed"] += 1
                error.context = {**getattr(error, "context", {}), "m": int(m), "candidate_id": current_id, "counts_at_m": counts}
                raise
            if value < best:
                best, best_id, best_diagnostics = value, current_id, diagnostics
        rows.append({"m": int(m), "loss": best if best_id is not None else None, "candidate_id": best_id,
                     "status": "admissible" if best_id is not None else "no_admissible_candidate", **counts})
        if best_diagnostics is not None:
            rows[-1]["region_diagnostics"] = best_diagnostics
        for key in totals:
            totals[key] += counts[key]
    feasible = [r for r in rows if r["candidate_id"] is not None]
    audit = {"candidate_count": candidate_id, "candidate_counts": totals,
             "feasible_domain": "baseline_posterior_proper"}
    if focal:
        audit.update(focal_design=focal.audit(), interpretation="conditional_anchor_Clarke_project_adaptation",
                     informative_object="p_info(theta_j | anchor)", candidate_object="p_base(theta_j | anchor, auxiliary_data)")
    if not feasible:
        return {"epss": None, "status": "inadmissible", "error": "No proper baseline candidate posterior", **audit}, rows, []
    best = min(feasible, key=lambda r: (r["loss"], r["m"]))
    if method.get("region") == "central":
        audit.update(truncation_region="central", selected_region_diagnostics=best["region_diagnostics"])
    return {"epss": best["m"], "status": "matched", **audit, "boundary_rate": float(best["m"] == max(grid))}, rows, []


def morita(model, data, info, base, target, method, grid, provider, seed):
    from .morita_versions import VERSIONS, run_version
    if method.get("preset") in VERSIONS:
        return run_version(model, data, info, target, method, grid, provider, seed)
    if "epsilon_prior" not in method:
        raise DefinitionError("Morita requires explicit epsilon_prior; external baseline is not used")
    epsilon = method["epsilon_prior"]
    empty = model.subset(data, [])
    pprior = provider.fit(empty, info, seed)
    check_target_moments(model, empty, info, target, 1)
    p = pprior.target(target, method.get("posterior_draws", 2048), seed)
    if (p.logpdf or p.log_kernel) is None or not np.all(np.isfinite(p.mean)):
        raise DefinitionError("Morita requires a target prior mean and differentiable target density")
    point = p.mean
    h = method.get("hessian_step", 1e-4)
    def curvature(measure):
        if (measure.logpdf or measure.log_kernel) is None:
            raise DefinitionError("Morita target density unavailable")
        lp = measure.logpdf or measure.log_kernel
        result = 0.
        for step in np.eye(target.dimension) * h:
            result -= float((lp((point + step)[None, :])[0] - 2 * lp(point[None, :])[0] + lp((point - step)[None, :])[0]) / h**2)
        return result
    desired = curvature(p)
    rows = []
    for m in grid:
        rng = np.random.default_rng(seed + 1000 + int(m))
        values = []
        for b in range(method.get("prior_predictive_reps", 16)):
            draw = pprior.sample(1, rng)
            theta = [draw[v][0] for v in model.variables]
            current = model.simulate(theta, int(m), rng, data)
            post = provider.fit(current, epsilon, seed_for(seed, "morita", int(m), b))
            values.append(curvature(post.target(target, method.get("posterior_draws", 2048), seed)))
        mean = float(np.mean(values))
        rows.append({"m": int(m), "prior_curvature": desired, "candidate_curvature": mean, "loss": abs(desired - mean)})
    best = min(rows, key=lambda r: (r["loss"], r["m"]))
    return {"epss": best["m"], "status": "matched", "boundary_rate": float(best["m"] == max(grid))}, rows, []


def run_method(model, data, info, base, target, method, grid, settings, seed, *, seed_scheme="sha256_32_v1", provider=None):
    if provider is None:
        provider = PosteriorProvider(model, settings)
    elif provider.model.name != model.name or provider.settings != settings or getattr(provider.model, "options", None) != getattr(model, "options", None):
        raise DefinitionError("Shared provider does not match model/settings")
    provider.accuracy = []  # Keep cached fits, but audit each task separately.
    function = {"jones": matching, "cp": matching, "clarke": clarke, "reimherr": risk_matching,
                "wiesenfarth": risk_matching, "morita": morita}[method["name"]]
    try:
        with seed_scope(seed_scheme):
            result, curves, selected = function(model, data, info, base, target, method, grid, provider, seed)
    except (UndefinedMoment, UncertifiedMoment) as e:
        return {"status": "undefined_moment" if isinstance(e, UndefinedMoment) else "uncertified_moment", "epss": None, "error": str(e), "failure_context": getattr(e, "context", {})}, [], [], provider.accuracy
    except InadmissiblePosterior as e:
        return {"status": "inadmissible", "epss": None, "error": str(e), "failure_context": getattr(e, "context", {})}, [], [], provider.accuracy
    except NumericalFailure as e:
        return {"status": "numerical_failure", "epss": None, "error": str(e), "failure_context": getattr(e, "context", {}), "numerical_failure_rate": 1.}, [], [], provider.accuracy
    except Exception as error:
        error.posterior_accuracy = provider.accuracy
        raise
    return result, curves, selected, provider.accuracy
