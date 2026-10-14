"""Jones-source-faithful path construction for the lupus logistic model.

This is a model-specific two-dimensional adaptation of Jones et al. (2021),
equations (3.3)--(3.4) and Algorithms 1--2.  It deliberately does not claim
that the previously validated one-dimensional conjugate reproduction extends
automatically to logistic regression.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import expit

from .jones_source_paths import select_source_signed
from .lupus_logistic import (
    PosteriorTargetCache,
    append_data,
    fit_logistic_posterior_counts,
    fit_logistic_posteriors_counts_batch,
    grouped_binomial_counts,
    target_values,
    wasserstein_distance_cached,
)


VARIANT = "jones_source_faithful_logistic_2d_adaptation_v1"
PATH_SCHEME = "jones_shared_informative_factorized_ppd_non_nested"


def target_standardizer(baseline_posterior, target):
    values = target_values(baseline_posterior.points, target)
    if np.ndim(values) == 1:
        return None
    mean = np.sum(values * baseline_posterior.weights[:, None], axis=0)
    variance = np.sum(
        baseline_posterior.weights[:, None] * np.square(values - mean), axis=0
    )
    return mean, np.sqrt(np.maximum(variance, 1e-12))


def _draw_independent_candidate(
    x,
    y,
    theta_star,
    r,
    rng,
    levels,
):
    """Draw a fresh candidate dataset conditional on one path latent beta."""
    r = int(r)
    if r == 0:
        return grouped_binomial_counts(x, y, levels=levels), np.empty(0), np.empty(0)
    indices = rng.choice(len(x), size=r, replace=True)
    x_new = np.asarray(x)[indices]
    probability = expit(theta_star[0] + theta_star[1] * x_new)
    y_new = rng.binomial(1, probability).astype(int)
    augmented = append_data(x, y, x_new, y_new)
    return (
        grouped_binomial_counts(*augmented, levels=levels),
        x_new,
        y_new,
    )


def _fit_candidate_batch(
    execution_engine,
    levels,
    totals,
    cases,
    prior,
    seeds,
    full_posterior,
    gh_nodes,
    tolerance,
    sobol_power,
    sobol_max_power,
):
    if not len(totals):
        return []
    if execution_engine == "cpp":
        return fit_logistic_posteriors_counts_batch(
            levels,
            np.asarray(totals, dtype=np.int64),
            np.asarray(cases, dtype=np.int64),
            prior,
            seeds,
            gh_nodes=gh_nodes,
            tolerance=tolerance,
            sobol_power=sobol_power,
            sobol_max_power=sobol_max_power,
            starts=np.tile(full_posterior.mean, (len(totals), 1)),
        )
    if execution_engine != "counts":
        raise ValueError("Logistic Jones source paths require counts or cpp.")
    output = []
    start = full_posterior.mean
    for totals_r, cases_r, seed in zip(totals, cases, seeds):
        posterior = fit_logistic_posterior_counts(
            levels,
            totals_r,
            cases_r,
            prior,
            seed=int(seed),
            gh_nodes=gh_nodes,
            tolerance=tolerance,
            sobol_power=sobol_power,
            start=start,
            mode_engine="scipy",
        )
        output.append(posterior)
        start = posterior.mean
    return output


def run_source_path(
    *,
    x,
    y,
    prior,
    baseline,
    info_full,
    base_full,
    targets,
    m_grid,
    seed,
    projections,
    execution_engine,
    gh_nodes,
    tolerance,
    sobol_power,
    sobol_max_power=None,
):
    """Run one Jones outer path and return signed selections for all targets."""
    m_grid = np.unique(np.asarray(m_grid, dtype=int))
    if len(m_grid) == 0 or np.any(m_grid < 0):
        raise ValueError("m_grid must contain nonnegative candidate additions.")
    rng = np.random.default_rng(int(seed))
    theta_star = info_full.resample(1, rng)[0]
    levels = np.unique(np.asarray(x, dtype=float))
    fitted_m = [int(m) for m in m_grid if int(m) != 0]
    positive_totals = []
    positive_cases = []
    negative_totals = []
    negative_cases = []
    hasher = hashlib.sha256()
    hasher.update(np.asarray(theta_star, dtype="<f8").tobytes())

    for m in fitted_m:
        # Separate calls make the two directions independent conditional on
        # theta_star.  Repeating these calls for every m also makes candidates
        # independent over m rather than nested prefixes.
        positive_grouped, x_positive, y_positive = _draw_independent_candidate(
            x, y, theta_star, m, rng, levels
        )
        negative_grouped, x_negative, y_negative = _draw_independent_candidate(
            x, y, theta_star, m, rng, levels
        )
        positive_totals.append(positive_grouped[1])
        positive_cases.append(positive_grouped[2])
        negative_totals.append(negative_grouped[1])
        negative_cases.append(negative_grouped[2])
        hasher.update(np.asarray([m], dtype="<i8").tobytes())
        for x_new, y_new in (
            (x_positive, y_positive),
            (x_negative, y_negative),
        ):
            hasher.update(np.asarray(x_new, dtype="<f8").tobytes())
            hasher.update(np.asarray(y_new, dtype="<i8").tobytes())

    positive_posteriors = _fit_candidate_batch(
        execution_engine,
        levels,
        positive_totals,
        positive_cases,
        baseline,
        [int(seed) + 10_000 + m for m in fitted_m],
        base_full,
        gh_nodes,
        tolerance,
        sobol_power,
        sobol_max_power,
    )
    negative_posteriors = _fit_candidate_batch(
        execution_engine,
        levels,
        negative_totals,
        negative_cases,
        prior,
        [int(seed) + 20_000 + m for m in fitted_m],
        info_full,
        gh_nodes,
        tolerance,
        sobol_power,
        sobol_max_power,
    )
    positive_by_m = {0: base_full, **dict(zip(fitted_m, positive_posteriors))}
    negative_by_m = {0: info_full, **dict(zip(fitted_m, negative_posteriors))}
    base_cache = PosteriorTargetCache(base_full)
    info_cache = PosteriorTargetCache(info_full)
    positive_caches = {
        m: (base_cache if m == 0 else PosteriorTargetCache(posterior))
        for m, posterior in positive_by_m.items()
    }
    negative_caches = {
        m: (info_cache if m == 0 else PosteriorTargetCache(posterior))
        for m, posterior in negative_by_m.items()
    }
    selections = {}
    details = []
    for target in targets:
        standardizer = target_standardizer(base_full, target)
        positive = []
        negative = []
        for m_raw in m_grid:
            m = int(m_raw)
            positive_value = wasserstein_distance_cached(
                info_full,
                positive_by_m[m],
                target,
                order=2,
                projections=projections,
                seed=int(seed) + 30_000 + m,
                standardizer=standardizer,
                cache_a=info_cache,
                cache_b=positive_caches[m],
            )
            negative_value = wasserstein_distance_cached(
                base_full,
                negative_by_m[m],
                target,
                order=2,
                projections=projections,
                seed=int(seed) + 40_000 + m,
                standardizer=standardizer,
                cache_a=base_cache,
                cache_b=negative_caches[m],
            )
            positive.append(positive_value)
            negative.append(negative_value)
            details.append(
                {
                    "target": target,
                    "m": m,
                    "positive_objective": float(positive_value),
                    "negative_objective": float(negative_value),
                }
            )
        selections[target] = select_source_signed(positive, negative, m_grid)

    accuracy = [
        posterior.accuracy
        for posterior in [*positive_posteriors, *negative_posteriors]
    ]
    return {
        "selection": selections,
        "details": details,
        "accuracy": accuracy,
        "theta_star": np.asarray(theta_star, dtype=float),
        "path_checksum": hasher.hexdigest(),
    }

