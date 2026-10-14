"""Vectorized CP diagnostics for the NIG marginal-mu experiment.

This module implements three posterior-predictive path constructions on a
common observed Normal sample:

``current_ppd_nested``
    The project's current CP construction.  Future observations are drawn
    marginally from the direction-specific posterior predictive and prefixes
    form one nested path over ``m``.

``cp_source_faithful``
    Jones' source factorization with the CP loss.  One joint
    ``(mu, sigma2)`` draw from the informative posterior is shared across
    directions and candidate sample sizes.  Conditional datasets are
    independent across directions and ``m``.

``cp_directional_ppd``
    The semi-source-faithful directional construction.  The positive
    direction draws ``(mu, sigma2)`` from the informative posterior and the
    negative direction draws it from the Jeffreys posterior.  Conditional
    datasets remain nonnested across ``m``.

All comparisons use the exact central-interval cross-coverage objective for
the marginal Student-t posterior of ``mu``.  The two direction-specific
minima are found within each outer draw before the winning signed ``m`` is
recorded.  The Jones-style source variants break exact ties toward the
positive direction; ``current_ppd_nested`` preserves its historical negative
tie rule.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import stats


VARIANTS = (
    "current_ppd_nested",
    "cp_source_faithful",
    "cp_directional_ppd",
)


@dataclass(frozen=True)
class NIGPriorSpec:
    mu0: float
    tau0_sq: float
    kappa: float
    eta: float


def ig_params_from_mean_var(mean: float, variance: float) -> tuple[float, float]:
    """Return IG(shape, scale) from its finite mean and variance."""
    mean = float(mean)
    variance = float(variance)
    if mean <= 0.0 or variance <= 0.0:
        raise ValueError("Inverse-gamma mean and variance must be positive.")
    kappa = 2.0 + mean * mean / variance
    eta = mean * (kappa - 1.0)
    return float(kappa), float(eta)


def stable_cell_seed(
    base_seed: int,
    rep: int,
    prior_loc: float,
    sigma2_mean: float,
    sigma2_variance: float,
) -> int:
    """Build an account- and process-independent 64-bit seed for one cell."""
    key = (
        f"{int(base_seed)}|{int(rep)}|{float(prior_loc):.12g}|"
        f"{float(sigma2_mean):.12g}|{float(sigma2_variance):.12g}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _rng(seed: int, stream: int) -> np.random.Generator:
    seed = int(seed)
    sequence = np.random.SeedSequence(
        [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, int(stream)]
    )
    return np.random.default_rng(sequence)


def marginal_mu_student_params(
    n,
    sums,
    sums_sq,
    prior: NIGPriorSpec | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal Student-t parameters under NIG or Jeffreys (``prior=None``)."""
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    if np.any(n < 2.0):
        raise ValueError("The marginal-mu posterior requires n >= 2.")
    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)

    if prior is None:
        if np.any(sse <= 0.0):
            raise ValueError("Jeffreys posterior requires positive SSE.")
        df = n - 1.0
        loc = ybar
        scale = np.sqrt(sse / (n * (n - 1.0)))
        return df, loc, scale

    lambda0 = 1.0 / float(prior.tau0_sq)
    lambda_n = lambda0 + n
    kappa_n = float(prior.kappa) + 0.5 * n
    eta_n = float(prior.eta) + 0.5 * (
        sse
        + lambda0
        * n
        / lambda_n
        * np.square(ybar - float(prior.mu0))
    )
    loc = (lambda0 * float(prior.mu0) + sums) / lambda_n
    scale = np.sqrt(eta_n / (kappa_n * lambda_n))
    return 2.0 * kappa_n, loc, scale


def joint_posterior_params(
    n: int,
    sums: float,
    sums_sq: float,
    prior: NIGPriorSpec | None,
) -> tuple[float, float, float, float]:
    """Return ``(mu, lambda, IG shape, IG scale)`` for the joint posterior."""
    n = int(n)
    sums = float(sums)
    sums_sq = float(sums_sq)
    if n < 2:
        raise ValueError("Joint posterior sampling requires n >= 2.")
    ybar = sums / n
    sse = max(sums_sq - sums * sums / n, 0.0)
    if prior is None:
        if sse <= 0.0:
            raise ValueError("Jeffreys posterior requires positive SSE.")
        return ybar, float(n), 0.5 * (n - 1.0), 0.5 * sse

    lambda0 = 1.0 / float(prior.tau0_sq)
    lambda_n = lambda0 + n
    mu_n = (lambda0 * float(prior.mu0) + sums) / lambda_n
    shape_n = float(prior.kappa) + 0.5 * n
    scale_n = float(prior.eta) + 0.5 * (
        sse
        + lambda0 * n / lambda_n * (ybar - float(prior.mu0)) ** 2
    )
    return float(mu_n), float(lambda_n), float(shape_n), float(scale_n)


def predictive_student_params(
    n: int,
    sums: float,
    sums_sq: float,
    prior: NIGPriorSpec | None,
) -> tuple[float, float, float]:
    mu, lambda_n, shape, scale = joint_posterior_params(
        n, sums, sums_sq, prior
    )
    return (
        2.0 * shape,
        mu,
        float(np.sqrt(scale / shape * (1.0 + 1.0 / lambda_n))),
    )


def draw_joint_posterior(
    params: tuple[float, float, float, float],
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mu, lambda_n, shape, scale = params
    precision = rng.gamma(shape=shape, scale=1.0 / scale, size=int(size))
    sigma2 = 1.0 / precision
    latent_mu = rng.normal(mu, np.sqrt(sigma2 / lambda_n), size=int(size))
    return latent_mu, sigma2


def student_cp_components(
    p: tuple[np.ndarray, np.ndarray, np.ndarray],
    q: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact CP components for two location-scale Student-t laws."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    p_df, p_loc, p_scale = np.broadcast_arrays(*[np.asarray(x) for x in p])
    q_df, q_loc, q_scale = np.broadcast_arrays(*[np.asarray(x) for x in q])
    p_df, p_loc, p_scale, q_df, q_loc, q_scale = np.broadcast_arrays(
        p_df, p_loc, p_scale, q_df, q_loc, q_scale
    )
    if np.any(p_df <= 0.0) or np.any(q_df <= 0.0):
        raise ValueError("Student-t degrees of freedom must be positive.")
    if np.any(p_scale <= 0.0) or np.any(q_scale <= 0.0):
        raise ValueError("Student-t scales must be positive.")

    lo_p = p_loc + p_scale * stats.t.ppf(alpha / 2.0, p_df)
    hi_p = p_loc + p_scale * stats.t.ppf(1.0 - alpha / 2.0, p_df)
    lo_q = q_loc + q_scale * stats.t.ppf(alpha / 2.0, q_df)
    hi_q = q_loc + q_scale * stats.t.ppf(1.0 - alpha / 2.0, q_df)
    cp1 = stats.t.cdf((hi_q - p_loc) / p_scale, p_df) - stats.t.cdf(
        (lo_q - p_loc) / p_scale, p_df
    )
    cp2 = stats.t.cdf((hi_p - q_loc) / q_scale, q_df) - stats.t.cdf(
        (lo_p - q_loc) / q_scale, q_df
    )
    r1 = np.maximum(cp1, cp2)
    r2 = np.minimum(cp1, cp2)
    gap = np.square(r1 - r2)
    coverage = np.square(r2 - (1.0 - alpha))
    objective = gap + coverage
    return cp1, cp2, r1, r2, objective


def _update_hash(hasher, *arrays) -> None:
    for array in arrays:
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        hasher.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        hasher.update(canonical.tobytes())


def _candidate_components(
    fixed,
    candidate,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, r1, r2, objective = student_cp_components(
        fixed, candidate, alpha=alpha
    )
    gap = np.square(r1 - r2)
    coverage = np.square(r2 - (1.0 - alpha))
    return r1, r2, gap, coverage, objective


def run_cell(
    y,
    prior: NIGPriorSpec,
    variant: str,
    *,
    B: int = 500,
    m_grid=range(21),
    alpha: float = 0.05,
    seed: int = 20260312,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one prior-grid cell and return its summary and draw-level output."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    y = np.asarray(y, dtype=float).ravel()
    B = int(B)
    if len(y) < 2 or B <= 1:
        raise ValueError("Require at least two observations and B > 1.")
    m_grid = np.asarray(tuple(m_grid), dtype=int)
    if len(m_grid) == 0 or m_grid[0] != 0 or np.any(np.diff(m_grid) <= 0):
        raise ValueError("m_grid must be strictly increasing and start at zero.")

    n0 = len(y)
    sum0 = float(np.sum(y))
    sumsq0 = float(np.sum(np.square(y)))
    inf_obs = marginal_mu_student_params(n0, sum0, sumsq0, prior)
    base_obs = marginal_mu_student_params(n0, sum0, sumsq0, None)
    shape = (B, len(m_grid))
    negative = {key: np.empty(shape) for key in ("r1", "r2", "gap", "cov", "obj")}
    positive = {key: np.empty(shape) for key in ("r1", "r2", "gap", "cov", "obj")}
    positive_hasher = hashlib.sha256()
    negative_hasher = hashlib.sha256()

    if variant == "current_ppd_nested":
        max_m = int(m_grid[-1])
        base_pred = predictive_student_params(n0, sum0, sumsq0, None)
        inf_pred = predictive_student_params(n0, sum0, sumsq0, prior)
        base_path = (
            base_pred[1]
            + base_pred[2]
            * _rng(seed, 31).standard_t(base_pred[0], size=(B, max_m))
        )
        inf_path = (
            inf_pred[1]
            + inf_pred[2]
            * _rng(seed, 32).standard_t(inf_pred[0], size=(B, max_m))
        )
        base_csum = np.cumsum(base_path, axis=1)
        base_csum_sq = np.cumsum(np.square(base_path), axis=1)
        inf_csum = np.cumsum(inf_path, axis=1)
        inf_csum_sq = np.cumsum(np.square(inf_path), axis=1)
        _update_hash(negative_hasher, base_path)
        _update_hash(positive_hasher, inf_path)
    else:
        inf_joint = joint_posterior_params(n0, sum0, sumsq0, prior)
        pos_mu, pos_sigma2 = draw_joint_posterior(
            inf_joint, B, _rng(seed, 11)
        )
        if variant == "cp_source_faithful":
            neg_mu, neg_sigma2 = pos_mu, pos_sigma2
        else:
            base_joint = joint_posterior_params(n0, sum0, sumsq0, None)
            neg_mu, neg_sigma2 = draw_joint_posterior(
                base_joint, B, _rng(seed, 12)
            )
        pos_data_rng = _rng(seed, 21)
        neg_data_rng = _rng(seed, 22)
        _update_hash(positive_hasher, pos_mu, pos_sigma2)
        _update_hash(negative_hasher, neg_mu, neg_sigma2)

    for column, m in enumerate(m_grid):
        m = int(m)
        if m == 0:
            add_pos_sum = np.zeros(B)
            add_pos_sumsq = np.zeros(B)
            add_neg_sum = np.zeros(B)
            add_neg_sumsq = np.zeros(B)
        elif variant == "current_ppd_nested":
            idx = m - 1
            add_neg_sum = base_csum[:, idx]
            add_neg_sumsq = base_csum_sq[:, idx]
            add_pos_sum = inf_csum[:, idx]
            add_pos_sumsq = inf_csum_sq[:, idx]
        else:
            # Separate calls for every m deliberately create conditionally
            # independent, nonnested datasets as in Jones' source factorization.
            pos_add = pos_mu[:, None] + np.sqrt(pos_sigma2)[:, None] * (
                pos_data_rng.standard_normal(size=(B, m))
            )
            neg_add = neg_mu[:, None] + np.sqrt(neg_sigma2)[:, None] * (
                neg_data_rng.standard_normal(size=(B, m))
            )
            add_pos_sum = np.sum(pos_add, axis=1)
            add_pos_sumsq = np.sum(np.square(pos_add), axis=1)
            add_neg_sum = np.sum(neg_add, axis=1)
            add_neg_sumsq = np.sum(np.square(neg_add), axis=1)
            _update_hash(positive_hasher, pos_add)
            _update_hash(negative_hasher, neg_add)

        n_aug = n0 + m
        inf_aug_neg = marginal_mu_student_params(
            n_aug, sum0 + add_neg_sum, sumsq0 + add_neg_sumsq, prior
        )
        base_aug_pos = marginal_mu_student_params(
            n_aug, sum0 + add_pos_sum, sumsq0 + add_pos_sumsq, None
        )
        neg_values = _candidate_components(base_obs, inf_aug_neg, alpha)
        pos_values = _candidate_components(inf_obs, base_aug_pos, alpha)
        for key, value in zip(("r1", "r2", "gap", "cov", "obj"), neg_values):
            negative[key][:, column] = value
        for key, value in zip(("r1", "r2", "gap", "cov", "obj"), pos_values):
            positive[key][:, column] = value

    negative_index = np.argmin(negative["obj"], axis=1)
    positive_index = np.argmin(positive["obj"], axis=1)
    rows = np.arange(B)
    negative_min = negative["obj"][rows, negative_index]
    positive_min = positive["obj"][rows, positive_index]
    # Preserve the historical current-CP tie rule, while the two Jones-style
    # source constructions follow the source estimator's positive tie rule.
    positive_wins = (
        positive_min < negative_min
        if variant == "current_ppd_nested"
        else positive_min <= negative_min
    )
    negative_m = m_grid[negative_index]
    positive_m = m_grid[positive_index]
    signed = np.where(positive_wins, positive_m, -negative_m).astype(float)
    selected_obj = np.where(positive_wins, positive_min, negative_min)

    draw = {
        "draw": rows,
        "signed_m": signed,
        "negative_mstar": negative_m,
        "positive_mstar": positive_m,
        "positive_wins": positive_wins.astype(int),
        "negative_min_obj": negative_min,
        "positive_min_obj": positive_min,
        "selected_obj": selected_obj,
    }
    for direction, values, indices in (
        ("negative", negative, negative_index),
        ("positive", positive, positive_index),
    ):
        for key in ("r1", "r2", "gap", "cov"):
            draw[f"{direction}_best_{key}"] = values[key][rows, indices]

    def mean_sd(values) -> tuple[float, float]:
        values = np.asarray(values, dtype=float)
        return float(np.mean(values)), float(np.std(values, ddof=1))

    summary = {
        "epss": float(np.mean(signed)),
        "positive_path_checksum": positive_hasher.hexdigest(),
        "negative_path_checksum": negative_hasher.hexdigest(),
    }
    fields = {
        "forward_mstar": negative_m,
        "backward_mstar": positive_m,
        "abs_mstar": np.abs(signed),
        "forward_min_obj": negative_min,
        "backward_min_obj": positive_min,
        "selected_obj": selected_obj,
        "forward_best_r1": draw["negative_best_r1"],
        "forward_best_r2": draw["negative_best_r2"],
        "forward_best_term_gap": draw["negative_best_gap"],
        "forward_best_term_cov": draw["negative_best_cov"],
        "backward_best_r1": draw["positive_best_r1"],
        "backward_best_r2": draw["positive_best_r2"],
        "backward_best_term_gap": draw["positive_best_gap"],
        "backward_best_term_cov": draw["positive_best_cov"],
    }
    for name, values in fields.items():
        summary[f"mean_{name}"], summary[f"sd_{name}"] = mean_sd(values)
    summary["pr_backward_final"] = float(np.mean(positive_wins))

    numeric = np.concatenate(
        [np.asarray(value, dtype=float).ravel() for value in draw.values()]
    )
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("Nonfinite NIG CP diagnostic output.")
    return summary, draw


def generate_coverage_grid(
    y,
    prior: NIGPriorSpec,
    variant: str,
    *,
    B: int = 500,
    m_grid=range(21),
    alpha_grid=(0.01, 0.05, 0.20, 0.40),
    seed: int = 20260312,
) -> dict:
    """Generate reciprocal-coverage arrays on common pseudo-data paths.

    Returns arrays with shape ``(n_alpha, B, n_m)``.  This lower-level
    interface is intended for objective-function sensitivity studies: many
    losses can be evaluated on exactly the same posterior pairs without
    regenerating pseudo-data or changing Monte Carlo correlation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    y = np.asarray(y, dtype=float).ravel()
    B = int(B)
    m_grid = np.asarray(tuple(m_grid), dtype=int)
    alpha_grid = np.asarray(tuple(alpha_grid), dtype=float)
    if len(y) < 2 or B <= 1:
        raise ValueError("Require at least two observations and B > 1.")
    if len(m_grid) == 0 or m_grid[0] != 0 or np.any(np.diff(m_grid) <= 0):
        raise ValueError("m_grid must be strictly increasing and start at zero.")
    if len(alpha_grid) == 0 or np.any((alpha_grid <= 0.0) | (alpha_grid >= 1.0)):
        raise ValueError("alpha_grid must lie strictly inside (0, 1).")

    n0 = len(y)
    sum0 = float(np.sum(y))
    sumsq0 = float(np.sum(np.square(y)))
    inf_obs = marginal_mu_student_params(n0, sum0, sumsq0, prior)
    base_obs = marginal_mu_student_params(n0, sum0, sumsq0, None)
    array_shape = (len(alpha_grid), B, len(m_grid))
    output = {
        direction: {
            component: np.empty(array_shape, dtype=float)
            for component in ("cp1", "cp2")
        }
        for direction in ("negative", "positive")
    }
    positive_hasher = hashlib.sha256()
    negative_hasher = hashlib.sha256()

    if variant == "current_ppd_nested":
        max_m = int(m_grid[-1])
        base_pred = predictive_student_params(n0, sum0, sumsq0, None)
        inf_pred = predictive_student_params(n0, sum0, sumsq0, prior)
        base_path = (
            base_pred[1]
            + base_pred[2]
            * _rng(seed, 31).standard_t(base_pred[0], size=(B, max_m))
        )
        inf_path = (
            inf_pred[1]
            + inf_pred[2]
            * _rng(seed, 32).standard_t(inf_pred[0], size=(B, max_m))
        )
        base_csum = np.cumsum(base_path, axis=1)
        base_csum_sq = np.cumsum(np.square(base_path), axis=1)
        inf_csum = np.cumsum(inf_path, axis=1)
        inf_csum_sq = np.cumsum(np.square(inf_path), axis=1)
        _update_hash(negative_hasher, base_path)
        _update_hash(positive_hasher, inf_path)
    else:
        inf_joint = joint_posterior_params(n0, sum0, sumsq0, prior)
        pos_mu, pos_sigma2 = draw_joint_posterior(
            inf_joint, B, _rng(seed, 11)
        )
        if variant == "cp_source_faithful":
            neg_mu, neg_sigma2 = pos_mu, pos_sigma2
        else:
            base_joint = joint_posterior_params(n0, sum0, sumsq0, None)
            neg_mu, neg_sigma2 = draw_joint_posterior(
                base_joint, B, _rng(seed, 12)
            )
        pos_data_rng = _rng(seed, 21)
        neg_data_rng = _rng(seed, 22)
        _update_hash(positive_hasher, pos_mu, pos_sigma2)
        _update_hash(negative_hasher, neg_mu, neg_sigma2)

    for column, m_value in enumerate(m_grid):
        m = int(m_value)
        if m == 0:
            add_pos_sum = np.zeros(B)
            add_pos_sumsq = np.zeros(B)
            add_neg_sum = np.zeros(B)
            add_neg_sumsq = np.zeros(B)
        elif variant == "current_ppd_nested":
            idx = m - 1
            add_neg_sum = base_csum[:, idx]
            add_neg_sumsq = base_csum_sq[:, idx]
            add_pos_sum = inf_csum[:, idx]
            add_pos_sumsq = inf_csum_sq[:, idx]
        else:
            pos_add = pos_mu[:, None] + np.sqrt(pos_sigma2)[:, None] * (
                pos_data_rng.standard_normal(size=(B, m))
            )
            neg_add = neg_mu[:, None] + np.sqrt(neg_sigma2)[:, None] * (
                neg_data_rng.standard_normal(size=(B, m))
            )
            add_pos_sum = np.sum(pos_add, axis=1)
            add_pos_sumsq = np.sum(np.square(pos_add), axis=1)
            add_neg_sum = np.sum(neg_add, axis=1)
            add_neg_sumsq = np.sum(np.square(neg_add), axis=1)
            _update_hash(positive_hasher, pos_add)
            _update_hash(negative_hasher, neg_add)

        n_aug = n0 + m
        inf_aug_neg = marginal_mu_student_params(
            n_aug, sum0 + add_neg_sum, sumsq0 + add_neg_sumsq, prior
        )
        base_aug_pos = marginal_mu_student_params(
            n_aug, sum0 + add_pos_sum, sumsq0 + add_pos_sumsq, None
        )
        for alpha_index, alpha in enumerate(alpha_grid):
            neg_cp1, neg_cp2, _, _, _ = student_cp_components(
                base_obs, inf_aug_neg, alpha=float(alpha)
            )
            pos_cp1, pos_cp2, _, _, _ = student_cp_components(
                inf_obs, base_aug_pos, alpha=float(alpha)
            )
            output["negative"]["cp1"][alpha_index, :, column] = neg_cp1
            output["negative"]["cp2"][alpha_index, :, column] = neg_cp2
            output["positive"]["cp1"][alpha_index, :, column] = pos_cp1
            output["positive"]["cp2"][alpha_index, :, column] = pos_cp2

    for direction in ("negative", "positive"):
        for component in ("cp1", "cp2"):
            if not np.all(np.isfinite(output[direction][component])):
                raise RuntimeError("Nonfinite reciprocal coverage grid.")
    output.update(
        {
            "variant": variant,
            "m_grid": m_grid,
            "alpha_grid": alpha_grid,
            "positive_path_checksum": positive_hasher.hexdigest(),
            "negative_path_checksum": negative_hasher.hexdigest(),
        }
    )
    return output


def evaluate_coverage_loss(
    coverage_grid: dict,
    *,
    family: str,
    gap_weight: float = 1.0,
    alpha_values=(0.05,),
    m_max: int | None = None,
    adequacy_threshold: float | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Select signed ``m`` under a loss evaluated on a coverage grid.

    ``family='historical'`` evaluates

    ``gap_weight * (r1-r2)^2 + (r2-(1-alpha))^2``.

    ``family='direct'`` evaluates

    ``(CP1-c)^2 + (CP2-c)^2 + gap_weight * (CP1-CP2)^2``.

    Losses are averaged with equal weights over ``alpha_values``.  Adequacy is
    the maximum absolute nominal-coverage error across both reciprocal
    coverages and all selected alpha values at the winning direction/m.
    """
    if family not in {"historical", "direct"}:
        raise ValueError("family must be 'historical' or 'direct'.")
    gap_weight = float(gap_weight)
    if gap_weight < 0.0:
        raise ValueError("gap_weight must be nonnegative.")
    available_alpha = np.asarray(coverage_grid["alpha_grid"], dtype=float)
    requested_alpha = np.asarray(tuple(alpha_values), dtype=float)
    alpha_indices = []
    for alpha in requested_alpha:
        matches = np.flatnonzero(np.isclose(available_alpha, alpha, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"alpha={alpha} is absent or duplicated in coverage grid.")
        alpha_indices.append(int(matches[0]))
    alpha_indices = np.asarray(alpha_indices, dtype=int)

    full_m_grid = np.asarray(coverage_grid["m_grid"], dtype=int)
    if m_max is None:
        m_mask = np.ones(len(full_m_grid), dtype=bool)
    else:
        m_mask = full_m_grid <= int(m_max)
    m_grid = full_m_grid[m_mask]
    if len(m_grid) == 0 or m_grid[0] != 0:
        raise ValueError("Selected m grid is empty or excludes m=0.")
    nominal = (1.0 - requested_alpha)[:, None, None]

    losses = {}
    selected_components = {}
    for direction in ("negative", "positive"):
        cp1 = coverage_grid[direction]["cp1"][alpha_indices][:, :, m_mask]
        cp2 = coverage_grid[direction]["cp2"][alpha_indices][:, :, m_mask]
        gap = np.square(cp1 - cp2)
        if family == "historical":
            r2 = np.minimum(cp1, cp2)
            alpha_loss = gap_weight * gap + np.square(r2 - nominal)
        else:
            alpha_loss = (
                np.square(cp1 - nominal)
                + np.square(cp2 - nominal)
                + gap_weight * gap
            )
        losses[direction] = np.mean(alpha_loss, axis=0)
        selected_components[direction] = (cp1, cp2)

    B = losses["negative"].shape[0]
    rows = np.arange(B)
    negative_index = np.argmin(losses["negative"], axis=1)
    positive_index = np.argmin(losses["positive"], axis=1)
    direction_indices = {
        "negative": negative_index,
        "positive": positive_index,
    }
    negative_min = losses["negative"][rows, negative_index]
    positive_min = losses["positive"][rows, positive_index]
    positive_wins = (
        positive_min < negative_min
        if coverage_grid["variant"] == "current_ppd_nested"
        else positive_min <= negative_min
    )
    negative_m = m_grid[negative_index]
    positive_m = m_grid[positive_index]
    signed = np.where(positive_wins, positive_m, -negative_m).astype(float)
    selected_loss = np.where(positive_wins, positive_min, negative_min)

    # Retain an exact loss decomposition and reciprocal coverages at each
    # direction's own optimum.  These quantities are used by the diagnosis
    # figures; the two loss components add exactly to the minimized loss.
    nominal_best = nominal[:, :, 0]
    direction_best = {}
    for direction in ("negative", "positive"):
        cp1, cp2 = selected_components[direction]
        index = direction_indices[direction]
        best_cp1 = cp1[:, rows, index]
        best_cp2 = cp2[:, rows, index]
        gap_alpha = np.square(best_cp1 - best_cp2)
        if family == "historical":
            calibration_alpha = np.square(
                np.minimum(best_cp1, best_cp2) - nominal_best
            )
        else:
            calibration_alpha = (
                np.square(best_cp1 - nominal_best)
                + np.square(best_cp2 - nominal_best)
            )
        mean_gap = np.mean(gap_alpha, axis=0)
        weighted_gap = gap_weight * mean_gap
        calibration = np.mean(calibration_alpha, axis=0)
        adequacy = np.maximum(
            np.max(np.abs(best_cp1 - nominal_best), axis=0),
            np.max(np.abs(best_cp2 - nominal_best), axis=0),
        )
        direction_best[direction] = {
            "cp1": best_cp1,
            "cp2": best_cp2,
            "mean_cp1": np.mean(best_cp1, axis=0),
            "mean_cp2": np.mean(best_cp2, axis=0),
            "mean_r1": np.mean(np.maximum(best_cp1, best_cp2), axis=0),
            "mean_r2": np.mean(np.minimum(best_cp1, best_cp2), axis=0),
            "gap": mean_gap,
            "weighted_gap": weighted_gap,
            "calibration": calibration,
            "adequacy": adequacy,
        }
        reconstructed = weighted_gap + calibration
        if not np.allclose(
            reconstructed,
            losses[direction][rows, index],
            rtol=1e-11,
            atol=1e-13,
        ):
            raise RuntimeError("Direction-specific CP loss decomposition failed.")

    # Select diagnostics from the same winning direction as signed m.
    selected_adequacy = np.where(
        positive_wins,
        direction_best["positive"]["adequacy"],
        direction_best["negative"]["adequacy"],
    )
    selected_calibration = np.where(
        positive_wins,
        direction_best["positive"]["calibration"],
        direction_best["negative"]["calibration"],
    )
    selected_weighted_gap = np.where(
        positive_wins,
        direction_best["positive"]["weighted_gap"],
        direction_best["negative"]["weighted_gap"],
    )
    if adequacy_threshold is None:
        matched = np.ones(B, dtype=bool)
    else:
        matched = selected_adequacy <= float(adequacy_threshold)

    conditional_epss = float(np.mean(signed[matched])) if np.any(matched) else np.nan
    draw = {
        "draw": rows,
        "signed_m": signed,
        "negative_mstar": negative_m,
        "positive_mstar": positive_m,
        "positive_wins": positive_wins.astype(int),
        "negative_min_loss": negative_min,
        "positive_min_loss": positive_min,
        "selected_loss": selected_loss,
        "selected_calibration": selected_calibration,
        "selected_weighted_gap": selected_weighted_gap,
        "selected_adequacy": selected_adequacy,
        "matched": matched.astype(int),
    }
    for direction in ("negative", "positive"):
        for component in (
            "mean_cp1",
            "mean_cp2",
            "mean_r1",
            "mean_r2",
            "gap",
            "weighted_gap",
            "calibration",
            "adequacy",
        ):
            draw[f"{direction}_best_{component}"] = direction_best[direction][
                component
            ]
    summary = {
        "epss_raw": float(np.mean(signed)),
        "sd_signed_m": float(np.std(signed, ddof=1)),
        "mean_negative_mstar": float(np.mean(negative_m)),
        "sd_negative_mstar": float(np.std(negative_m, ddof=1)),
        "mean_positive_mstar": float(np.mean(positive_m)),
        "sd_positive_mstar": float(np.std(positive_m, ddof=1)),
        "pr_positive": float(np.mean(positive_wins)),
        "mean_selected_loss": float(np.mean(selected_loss)),
        "q95_selected_loss": float(np.quantile(selected_loss, 0.95)),
        "sd_selected_loss": float(np.std(selected_loss, ddof=1)),
        "mean_selected_calibration": float(np.mean(selected_calibration)),
        "mean_selected_weighted_gap": float(np.mean(selected_weighted_gap)),
        "mean_selected_adequacy": float(np.mean(selected_adequacy)),
        "sd_selected_adequacy": float(np.std(selected_adequacy, ddof=1)),
        "q95_selected_adequacy": float(np.quantile(selected_adequacy, 0.95)),
        "match_rate": float(np.mean(matched)),
        "epss_matched_conditional": conditional_epss,
        "epss_unmatched_as_zero": float(np.mean(np.where(matched, signed, 0.0))),
        "negative_boundary_rate": float(np.mean(negative_m == m_grid[-1])),
        "positive_boundary_rate": float(np.mean(positive_m == m_grid[-1])),
        "selected_boundary_rate": float(np.mean(np.abs(signed) == m_grid[-1])),
        "family": family,
        "gap_weight": gap_weight,
        "alpha_values": requested_alpha.tolist(),
        "m_max": int(m_grid[-1]),
        "adequacy_threshold": (
            None if adequacy_threshold is None else float(adequacy_threshold)
        ),
    }
    for direction in ("negative", "positive"):
        direction_loss = negative_min if direction == "negative" else positive_min
        summary[f"mean_{direction}_min_loss"] = float(np.mean(direction_loss))
        summary[f"sd_{direction}_min_loss"] = float(
            np.std(direction_loss, ddof=1)
        )
        for component in (
            "mean_cp1",
            "mean_cp2",
            "mean_r1",
            "mean_r2",
            "gap",
            "weighted_gap",
            "calibration",
            "adequacy",
        ):
            values = direction_best[direction][component]
            summary[f"mean_{direction}_best_{component}"] = float(np.mean(values))
            summary[f"sd_{direction}_best_{component}"] = float(
                np.std(values, ddof=1)
            )
    return summary, draw


def evaluate_constrained_coverage(
    coverage_grid: dict,
    *,
    delta: float,
    alpha_values=(0.05,),
    secondary_rule: str = "minimum_gap",
    m_max: int | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Select signed ``m`` from an absolute-adequacy feasible set.

    For every outer draw, direction, and candidate ``m``, define

    ``A = max_alpha max(|CP1-(1-alpha)|, |CP2-(1-alpha)|)``.

    Only candidates satisfying ``A <= delta`` are eligible.  The selected
    candidate is then determined lexicographically by either

    * ``secondary_rule='minimum_gap'``: mean squared reciprocal-coverage
      gap, then ``m``; or
    * ``secondary_rule='minimum_m'``: ``m``, then mean squared gap.

    Negative precedes positive only for an otherwise exact tie.  A draw with
    no feasible candidate is returned as unmatched with ``NaN`` signed ``m``;
    it is never silently encoded as zero.
    """
    delta = float(delta)
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite nonnegative number.")
    if secondary_rule not in {"minimum_gap", "minimum_m"}:
        raise ValueError(
            "secondary_rule must be 'minimum_gap' or 'minimum_m'."
        )

    available_alpha = np.asarray(coverage_grid["alpha_grid"], dtype=float)
    requested_alpha = np.asarray(tuple(alpha_values), dtype=float)
    if len(requested_alpha) == 0:
        raise ValueError("alpha_values must be nonempty.")
    alpha_indices = []
    for alpha in requested_alpha:
        matches = np.flatnonzero(np.isclose(available_alpha, alpha, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"alpha={alpha} is absent or duplicated in coverage grid.")
        alpha_indices.append(int(matches[0]))
    alpha_indices = np.asarray(alpha_indices, dtype=int)

    full_m_grid = np.asarray(coverage_grid["m_grid"], dtype=int)
    if m_max is None:
        m_mask = np.ones(len(full_m_grid), dtype=bool)
    else:
        m_mask = full_m_grid <= int(m_max)
    m_grid = full_m_grid[m_mask]
    if len(m_grid) == 0 or m_grid[0] != 0:
        raise ValueError("Selected m grid is empty or excludes m=0.")

    nominal = (1.0 - requested_alpha)[:, None, None]
    candidate = {}
    for direction in ("negative", "positive"):
        cp1 = coverage_grid[direction]["cp1"][alpha_indices][:, :, m_mask]
        cp2 = coverage_grid[direction]["cp2"][alpha_indices][:, :, m_mask]
        adequacy = np.maximum(
            np.max(np.abs(cp1 - nominal), axis=0),
            np.max(np.abs(cp2 - nominal), axis=0),
        )
        gap = np.mean(np.square(cp1 - cp2), axis=0)
        candidate[direction] = {
            "adequacy": adequacy,
            "gap": gap,
            "cp1": cp1,
            "cp2": cp2,
            "feasible": adequacy <= delta,
        }

    B = candidate["negative"]["adequacy"].shape[0]
    n_m = len(m_grid)
    all_adequacy = np.concatenate(
        (candidate["negative"]["adequacy"], candidate["positive"]["adequacy"]),
        axis=1,
    )
    all_gap = np.concatenate(
        (candidate["negative"]["gap"], candidate["positive"]["gap"]),
        axis=1,
    )
    all_feasible = np.concatenate(
        (candidate["negative"]["feasible"], candidate["positive"]["feasible"]),
        axis=1,
    )
    all_m = np.tile(m_grid, 2)
    all_direction = np.repeat(np.asarray((0, 1), dtype=int), n_m)

    matched = np.any(all_feasible, axis=1)
    chosen_flat = np.full(B, -1, dtype=int)
    for draw_index in np.flatnonzero(matched):
        eligible = np.flatnonzero(all_feasible[draw_index])
        if secondary_rule == "minimum_gap":
            order = np.lexsort(
                (
                    all_direction[eligible],
                    all_m[eligible],
                    all_gap[draw_index, eligible],
                )
            )
        else:
            order = np.lexsort(
                (
                    all_direction[eligible],
                    all_gap[draw_index, eligible],
                    all_m[eligible],
                )
            )
        chosen_flat[draw_index] = int(eligible[order[0]])

    selected_direction = np.full(B, -1, dtype=int)
    selected_m = np.full(B, np.nan, dtype=float)
    signed_m = np.full(B, np.nan, dtype=float)
    selected_adequacy = np.full(B, np.nan, dtype=float)
    selected_gap = np.full(B, np.nan, dtype=float)
    selected_cp1_mean = np.full(B, np.nan, dtype=float)
    selected_cp2_mean = np.full(B, np.nan, dtype=float)
    selected_r1_mean = np.full(B, np.nan, dtype=float)
    selected_r2_mean = np.full(B, np.nan, dtype=float)
    for draw_index in np.flatnonzero(matched):
        flat = chosen_flat[draw_index]
        direction_code = int(all_direction[flat])
        column = int(flat % n_m)
        direction = "negative" if direction_code == 0 else "positive"
        cp1 = candidate[direction]["cp1"][:, draw_index, column]
        cp2 = candidate[direction]["cp2"][:, draw_index, column]
        m_value = float(m_grid[column])
        selected_direction[draw_index] = direction_code
        selected_m[draw_index] = m_value
        signed_m[draw_index] = m_value if direction_code == 1 else -m_value
        selected_adequacy[draw_index] = candidate[direction]["adequacy"][
            draw_index, column
        ]
        selected_gap[draw_index] = candidate[direction]["gap"][draw_index, column]
        selected_cp1_mean[draw_index] = float(np.mean(cp1))
        selected_cp2_mean[draw_index] = float(np.mean(cp2))
        selected_r1_mean[draw_index] = float(np.mean(np.maximum(cp1, cp2)))
        selected_r2_mean[draw_index] = float(np.mean(np.minimum(cp1, cp2)))

    minimum_adequacy = np.min(all_adequacy, axis=1)
    negative_has_feasible = np.any(candidate["negative"]["feasible"], axis=1)
    positive_has_feasible = np.any(candidate["positive"]["feasible"], axis=1)
    both_have_feasible = negative_has_feasible & positive_has_feasible
    positive_selected = matched & (selected_direction == 1)
    n_matched = int(np.sum(matched))

    def matched_mean(values):
        return float(np.mean(values[matched])) if n_matched else np.nan

    def matched_sd(values):
        return (
            float(np.std(values[matched], ddof=1)) if n_matched > 1 else np.nan
        )

    draw = {
        "draw": np.arange(B),
        "matched": matched.astype(int),
        "signed_m": signed_m,
        "selected_m": selected_m,
        "selected_direction": selected_direction,
        "positive_selected": positive_selected.astype(int),
        "selected_adequacy": selected_adequacy,
        "selected_gap": selected_gap,
        "selected_cp1_mean": selected_cp1_mean,
        "selected_cp2_mean": selected_cp2_mean,
        "selected_r1_mean": selected_r1_mean,
        "selected_r2_mean": selected_r2_mean,
        "minimum_adequacy": minimum_adequacy,
        "negative_has_feasible": negative_has_feasible.astype(int),
        "positive_has_feasible": positive_has_feasible.astype(int),
        "both_have_feasible": both_have_feasible.astype(int),
        "n_feasible_candidates": np.sum(all_feasible, axis=1),
    }
    summary = {
        "epss_matched_conditional": matched_mean(signed_m),
        "sd_signed_m_matched": matched_sd(signed_m),
        "mean_abs_m_matched": matched_mean(np.abs(signed_m)),
        "mean_selected_m_matched": matched_mean(selected_m),
        "match_rate": float(np.mean(matched)),
        "n_matched": n_matched,
        "n_draws": int(B),
        "pr_positive_conditional": (
            float(np.mean(positive_selected[matched])) if n_matched else np.nan
        ),
        "negative_feasible_rate": float(np.mean(negative_has_feasible)),
        "positive_feasible_rate": float(np.mean(positive_has_feasible)),
        "both_directions_feasible_rate": float(np.mean(both_have_feasible)),
        "mean_n_feasible_candidates": float(np.mean(np.sum(all_feasible, axis=1))),
        "mean_selected_adequacy_matched": matched_mean(selected_adequacy),
        "mean_selected_gap_matched": matched_mean(selected_gap),
        "mean_minimum_adequacy": float(np.mean(minimum_adequacy)),
        "q95_minimum_adequacy": float(np.quantile(minimum_adequacy, 0.95)),
        "selected_boundary_rate_conditional": (
            float(np.mean(selected_m[matched] == m_grid[-1])) if n_matched else np.nan
        ),
        "delta": delta,
        "secondary_rule": secondary_rule,
        "alpha_values": requested_alpha.tolist(),
        "m_max": int(m_grid[-1]),
        "unmatched_semantics": "missing_not_zero",
    }
    return summary, draw
