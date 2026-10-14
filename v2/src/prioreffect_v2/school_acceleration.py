"""Opt-in Eight Schools batching, preserving the scalar V2 stopping rule.

Ported design: scripts/run_eight_schools_simulation.py batch quadrature.
Unlike an all-rows stopping test, each row stops on exactly its scalar grid.
No legacy runner is imported and no RNG is consumed by this module.
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

from .core import DefinitionError, NumericalFailure, Posterior, weighted_quantile
from .quadrature import SchoolPosterior, trapezoid_weights


def validate_settings(model, settings):
    engine = settings.get("school_engine", "reference")
    if engine not in {"reference", "batched"}:
        raise DefinitionError("school_engine must be reference or batched")
    if settings.get("school_quantile_engine", "brentq") not in {"brentq", "vectorized_newton"}:
        raise DefinitionError("Unknown school_quantile_engine")
    for field, default in (("school_batch_size", 16), ("school_cache_size", 256)):
        value = settings.get(field, default)
        if type(value) is not int or value < 1:
            raise DefinitionError(f"{field} must be a positive integer")
    if not isinstance(settings.get("school_cache_by_data", False), bool):
        raise DefinitionError("school_cache_by_data must be boolean")
    if not isinstance(settings.get("school_cache_quantiles", False), bool):
        raise DefinitionError("school_cache_quantiles must be boolean")
    if engine == "batched" or settings.get("school_cache_by_data", False) or settings.get("school_quantile_engine", "brentq") != "brentq":
        if model.name != "eight_schools" or settings.get("backend", "auto") not in {"auto", "optimized"} or settings.get("posterior_reuse", "none") != "none":
            raise DefinitionError("School acceleration requires Eight Schools optimized/auto without importance reuse")


def batch_posteriors(model, datasets, prior, settings):
    """One prior and equal-sized datasets; bounded chunks are handled by caller."""
    if not datasets:
        return []
    lengths = {len(d["y"]) for d in datasets}
    if len(lengths) != 1:
        raise DefinitionError("School batch requires equal-sized datasets")
    y = np.asarray([d["y"] for d in datasets], dtype=float)
    s = np.asarray([d["s"] for d in datasets], dtype=float)
    size = int(settings.get("school_grid_size", 801))
    tail_tol = settings.get("tail_tol", 1e-10)
    tolerance = settings.get("school_convergence_tol", 1e-5)
    center = np.log(max(1., prior["tau_scale"]))
    lo, hi = min(-12., center - 10), max(np.log(500.), center + 8)
    bounded = prior["tau_kind"] == "bounded_uniform"
    if bounded:
        hi = np.log(prior["tau_scale"])
    previous = {}
    pending = np.arange(len(datasets))
    output = [None] * len(datasets)
    for expansion in range(int(settings.get("max_expansions", 8)) + 1):
        eta = np.linspace(lo, hi, size)
        variance = s[pending, None, :] ** 2 + np.exp(2 * eta)[None, :, None]
        precision = 1 / prior["mu_sd"] ** 2 + (1 / variance).sum(axis=-1)
        var = 1 / precision
        linear = prior["mu_mean"] / prior["mu_sd"] ** 2 + (y[pending, None, :] / variance).sum(axis=-1)
        mean = var * linear
        quadratic = prior["mu_mean"] ** 2 / prior["mu_sd"] ** 2 + (y[pending, None, :] ** 2 / variance).sum(axis=-1)
        constant = -.5 * np.log(2 * np.pi * variance).sum(axis=-1) - .5 * np.log(2 * np.pi * prior["mu_sd"] ** 2)
        theta = np.column_stack([np.full(size, prior["mu_mean"]), np.exp(eta)])
        log_prior = model.logprior(theta, prior) - stats.norm.logpdf(prior["mu_mean"], prior["mu_mean"], prior["mu_sd"]) + eta
        kernel = log_prior + constant - .5 * quadratic + .5 * linear ** 2 / precision + .5 * np.log(2 * np.pi / precision)
        log_norm = logsumexp(kernel + np.log(trapezoid_weights(eta)), axis=1)
        mass = np.exp(kernel + np.log(trapezoid_weights(eta)) - log_norm[:, None])
        remaining = []
        for row, idx in enumerate(pending):
            w, cm, cv = mass[row], mean[row], var[row]
            summary = np.r_[w @ eta, w @ eta ** 2, w @ cm, w @ (cv + cm ** 2), weighted_quantile(eta, [.025, .5, .975], w)]
            edge = max(float(w[:8].sum()), 0 if bounded else float(w[-8:].sum()))
            delta = np.inf if idx not in previous else float(np.max(abs(summary - previous[idx])))
            if edge < tail_tol and (delta < tolerance or expansion >= 1):
                post = SchoolPosterior.__new__(SchoolPosterior)
                Posterior.__init__(post, model, datasets[idx], prior, "collapsed_grid", {})
                post.cache_quantiles = settings.get("school_cache_quantiles", True)
                post.quantile_engine = settings.get("school_quantile_engine", "brentq")
                post.y, post.s = y[idx], s[idx]
                post.install_grid(eta.copy(), w.copy(), kernel[row].copy(), log_norm[row], cm.copy(), cv.copy(), edge, expansion, delta)
                post.accuracy.update(numerical_engine="school_batched_v1", batch_rows=len(datasets))
                output[idx] = post
            else:
                previous[idx] = summary
                remaining.append(idx)
        if not remaining:
            return output
        pending = np.array(remaining, dtype=int)
        width = hi - lo
        lo -= .35 * width
        if not bounded:
            hi += .35 * width
    error = NumericalFailure("School batch has unconverged rows; no failed row is dropped")
    error.context = {"unconverged_batch_rows": pending.tolist()}
    raise error
