import argparse
import json
import sys
import time
from itertools import chain, combinations, combinations_with_replacement, islice
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "function"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prioreffect.joint_nig import (  # noqa: E402
    augmented_sufficient_stats,
    fixed_unit_directions,
    joint_hdr_cp_batch_parameterized,
    nig_posterior_params_from_stats,
    sample_nig_coordinates,
    sliced_wasserstein2_batch,
)
from prioreffect.nig_truncated_kl import (  # noqa: E402
    build_q_hdr_probability_table,
    canonical_logspace_hdr_moments,
    truncated_kl_logspace_batched,
)
from prioreffect.sampling import (  # noqa: E402
    SAMPLING_METHODS,
    empirical_bootstrap_augmentation_path,
    empirical_deletion_path,
    is_deletion_sampling,
    is_empirical_augmentation,
    is_exploratory_sampling,
    select_signed_direction,
)


METHOD_ALIASES = {
    "jones": "jones_joint2d",
    "jones_joint2d": "jones_joint2d",
    "cp": "cp_joint2d",
    "cp_joint2d": "cp_joint2d",
    "reimherr": "reimherr_joint2d",
    "reimherr_joint2d": "reimherr_joint2d",
    "wiesenfarth": "wiesenfarth_joint2d",
    "wiesenfarth_joint2d": "wiesenfarth_joint2d",
    "clarke": "clarke_nig2d_kl",
    "clarke_kl": "clarke_nig2d_kl",
    "clarke_nig2d_kl": "clarke_nig2d_kl",
    "clarke_tkl": "clarke_nig2d_truncated_kl",
    "clarke_truncated_kl": "clarke_nig2d_truncated_kl",
    "clarke_nig2d_truncated_kl": "clarke_nig2d_truncated_kl",
}


def default_mu_grid():
    return np.linspace(-4.0, 4.0, 17)


def default_sigma2_mean_grid():
    return np.asarray(
        [
            0.25,
            0.35,
            0.50,
            0.70,
            0.85,
            1.00,
            1.15,
            1.30,
            1.50,
            1.75,
            2.00,
            2.50,
            3.00,
            4.00,
            5.00,
        ],
        dtype=float,
    )


def parse_float_grid(value, default):
    if value is None:
        return np.asarray(default, dtype=float)
    pieces = []
    for part in str(value).replace(",", " ").split():
        pieces.append(float(part))
    if not pieces:
        raise ValueError("Empty numeric grid.")
    return np.asarray(pieces, dtype=float)


def parse_int_grid(value, default):
    if value is None:
        return np.asarray(list(default), dtype=int)
    pieces = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            pieces.extend(range(int(a), int(b) + 1))
        else:
            pieces.append(int(part))
    if not pieces:
        raise ValueError("Empty integer grid.")
    return np.asarray(sorted(set(pieces)), dtype=int)


def ig_params_from_mean_var(mean, var):
    mean = np.asarray(mean, dtype=float)
    var = float(var)
    if np.any(mean <= 0.0):
        raise ValueError("sigma2 prior means must be positive.")
    if var <= 0.0:
        raise ValueError("sigma2 prior variance must be positive.")
    kappa = 2.0 + np.square(mean) / var
    eta = mean * (kappa - 1.0)
    return kappa, eta


def make_prior_grid(args):
    mu_values = parse_float_grid(args.prior_values, default_mu_grid())
    sigma2_means = parse_float_grid(args.sigma2_mean_values, default_sigma2_mean_grid())
    n_grid = len(mu_values) * len(sigma2_means)
    if n_grid > args.max_grid_points:
        raise ValueError(
            f"Requested {n_grid} grid points, above --max-grid-points={args.max_grid_points}."
        )
    rows = []
    grid_id = 0
    for mu0 in mu_values:
        for sigma2_mean in sigma2_means:
            kappa, eta = ig_params_from_mean_var(sigma2_mean, args.sigma2_prior_var)
            rows.append(
                {
                    "grid_id": int(grid_id),
                    "prior_loc": float(mu0),
                    "prior_value": float(mu0),
                    "sigma2_prior_mean": float(sigma2_mean),
                    "sigma2_prior_var": float(args.sigma2_prior_var),
                    "kappa": float(kappa),
                    "eta": float(eta),
                    "tau0_sq": float(args.tau0_sq),
                }
            )
            grid_id += 1
    return pd.DataFrame(rows)


def grid_seed_offset(spec):
    loc_code = int(round((float(spec["prior_loc"]) + 50.0) * 10_000.0))
    sig_code = int(round(float(spec["sigma2_prior_mean"]) * 10_000.0))
    return 1009 * loc_code + 9176 * sig_code + 7919 * int(spec["grid_id"])


def posterior_params(y, prior_type, spec, args):
    y = np.asarray(y, dtype=float)
    return nig_posterior_params_from_stats(
        len(y),
        np.sum(y, dtype=float),
        np.sum(np.square(y), dtype=float),
        prior_type=prior_type,
        prior_loc=float(spec["prior_loc"]),
        tau0_sq=args.tau0_sq,
        kappa=float(spec["kappa"]),
        eta=float(spec["eta"]),
    )


def posterior_path(y, future, extra_grid, prior_type, spec, args):
    n_path, sums, sums_sq = augmented_sufficient_stats(y, future, extra_grid)
    return nig_posterior_params_from_stats(
        n_path,
        sums,
        sums_sq,
        prior_type=prior_type,
        prior_loc=float(spec["prior_loc"]),
        tau0_sq=args.tau0_sq,
        kappa=float(spec["kappa"]),
        eta=float(spec["eta"]),
    )


def posterior_predictive_draws(params, size, rng):
    params = {key: float(np.asarray(value)) for key, value in params.items()}
    df = 2.0 * params["kappa"]
    scale = np.sqrt(
        params["eta"] / params["kappa"] * (1.0 + 1.0 / params["lambda"])
    )
    return stats.t.rvs(
        df=df,
        loc=params["mu"],
        scale=scale,
        size=int(size),
        random_state=rng,
    )


def joint_sw2(target, path, args, rng, directions):
    target_z, _, _ = sample_nig_coordinates(
        target,
        args.joint_draws,
        rng,
        parameterization=args.parameterization,
        mu_scale=args.mu_scale,
        sigma2_scale=args.sigma2_scale,
        log_sigma2_scale=args.log_sigma2_scale,
    )
    path_z, _, _ = sample_nig_coordinates(
        path,
        args.joint_draws,
        rng,
        parameterization=args.parameterization,
        mu_scale=args.mu_scale,
        sigma2_scale=args.sigma2_scale,
        log_sigma2_scale=args.log_sigma2_scale,
    )
    return sliced_wasserstein2_batch(target_z, path_z, directions)


def deletion_sufficient_stats(y, m_grid, args, rng):
    grid, samples = empirical_deletion_path(
        y,
        m_grid,
        rng,
        args.sampling_method,
        min_retained=args.min_retained,
        require_positive_variance=True,
    )
    n_path = np.asarray([len(samples[int(m)]) for m in grid], dtype=int)
    sums = np.asarray(
        [np.sum(samples[int(m)], dtype=float) for m in grid], dtype=float
    )
    sums_sq = np.asarray(
        [np.sum(np.square(samples[int(m)]), dtype=float) for m in grid],
        dtype=float,
    )
    return grid, n_path, sums, sums_sq


def deletion_posterior_path(
    n_path, sums, sums_sq, prior_type, spec, args
):
    return nig_posterior_params_from_stats(
        n_path,
        sums,
        sums_sq,
        prior_type=prior_type,
        prior_loc=float(spec["prior_loc"]),
        tau0_sq=args.tau0_sq,
        kappa=float(spec["kappa"]),
        eta=float(spec["eta"]),
    )


def run_jones_one(y, spec, args, rng):
    n = len(y)
    deletion = is_deletion_sampling(args.sampling_method)
    empirical_augmentation = is_empirical_augmentation(
        args.sampling_method
    )
    absolute_grid = np.asarray(args.jones_m_grid, dtype=int)
    if deletion:
        deletion_grid = absolute_grid
    else:
        deletion_grid = None
    extra_grid = absolute_grid - n
    max_extra = (
        int(np.max(extra_grid))
        if not deletion and not empirical_augmentation
        else None
    )
    directions = fixed_unit_directions(args.n_projections)

    inf0 = posterior_params(y, "informative", spec, args)
    base0 = posterior_params(y, args.baseline, spec, args)

    opess = np.empty(args.outer_reps, dtype=float)
    for b in range(args.outer_reps):
        if deletion:
            grid, n_path, sums, sums_sq = deletion_sufficient_stats(
                y, deletion_grid, args, rng
            )
            base_path = deletion_posterior_path(
                n_path, sums, sums_sq, args.baseline, spec, args
            )
            inf_path = deletion_posterior_path(
                n_path, sums, sums_sq, "informative", spec, args
            )
            negative = joint_sw2(inf0, base_path, args, rng, directions)
            positive = joint_sw2(base0, inf_path, args, rng, directions)
            selected = select_signed_direction(negative, positive, grid)
            opess[b] = selected["signed_m"]
        elif empirical_augmentation:
            grid, addition_path = empirical_bootstrap_augmentation_path(
                y, absolute_grid, rng
            )
            future = addition_path[int(np.max(grid))]
            base_path = posterior_path(
                y, future, grid, args.baseline, spec, args
            )
            inf_path = posterior_path(
                y, future, grid, "informative", spec, args
            )
            negative = joint_sw2(
                base0, inf_path, args, rng, directions
            )
            positive = joint_sw2(
                inf0, base_path, args, rng, directions
            )
            selected = select_signed_direction(
                negative, positive, grid
            )
            opess[b] = selected["signed_m"]
        else:
            future_d = posterior_predictive_draws(inf0, max_extra, rng)
            future_dsharp = posterior_predictive_draws(inf0, max_extra, rng)
            base_path = posterior_path(
                y, future_d, extra_grid, args.baseline, spec, args
            )
            inf_path = posterior_path(
                y, future_dsharp, extra_grid, "informative", spec, args
            )
            d = joint_sw2(inf0, base_path, args, rng, directions)
            dsharp = joint_sw2(base0, inf_path, args, rng, directions)
            i = int(np.argmin(d))
            j = int(np.argmin(dsharp))
            opess[b] = (
                extra_grid[i] if d[i] <= dsharp[j] else -extra_grid[j]
            )

    return {
        "ess": float(np.mean(opess)),
        "outer_sd_ess": float(np.std(opess, ddof=1)) if len(opess) > 1 else 0.0,
    }


def run_cp_one(y, spec, args, rng):
    deletion = is_deletion_sampling(args.sampling_method)
    empirical_augmentation = is_empirical_augmentation(
        args.sampling_method
    )
    extra_grid = np.asarray(args.cp_m_grid, dtype=int)
    max_extra = (
        int(np.max(extra_grid))
        if not deletion and not empirical_augmentation
        else None
    )
    inf0 = posterior_params(y, "informative", spec, args)
    base0 = posterior_params(y, args.baseline, spec, args)

    signed = np.empty(args.outer_reps, dtype=float)
    for b in range(args.outer_reps):
        if deletion:
            grid, n_path, sums, sums_sq = deletion_sufficient_stats(
                y, extra_grid, args, rng
            )
            base_path = deletion_posterior_path(
                n_path, sums, sums_sq, args.baseline, spec, args
            )
            inf_path = deletion_posterior_path(
                n_path, sums, sums_sq, "informative", spec, args
            )
            _, _, _, _, negative = joint_hdr_cp_batch_parameterized(
                inf0,
                base_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            _, _, _, _, positive = joint_hdr_cp_batch_parameterized(
                base0,
                inf_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            selected = select_signed_direction(negative, positive, grid)
            signed[b] = selected["signed_m"]
        elif empirical_augmentation:
            grid, addition_path = empirical_bootstrap_augmentation_path(
                y, extra_grid, rng
            )
            future = addition_path[int(np.max(grid))]
            base_path = posterior_path(
                y, future, grid, args.baseline, spec, args
            )
            inf_path = posterior_path(
                y, future, grid, "informative", spec, args
            )
            _, _, _, _, negative = joint_hdr_cp_batch_parameterized(
                base0,
                inf_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            _, _, _, _, positive = joint_hdr_cp_batch_parameterized(
                inf0,
                base_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            selected = select_signed_direction(
                negative, positive, grid
            )
            signed[b] = selected["signed_m"]
        else:
            future_base = posterior_predictive_draws(base0, max_extra, rng)
            future_inf = posterior_predictive_draws(inf0, max_extra, rng)
            inf_path = posterior_path(
                y, future_base, extra_grid, "informative", spec, args
            )
            base_path = posterior_path(
                y, future_inf, extra_grid, args.baseline, spec, args
            )
            _, _, _, _, obj_f = joint_hdr_cp_batch_parameterized(
                base0,
                inf_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            _, _, _, _, obj_b = joint_hdr_cp_batch_parameterized(
                inf0,
                base_path,
                args.alpha,
                args.joint_draws,
                rng,
                parameterization=args.parameterization,
            )
            idx = int(np.argmin(np.minimum(obj_f, obj_b)))
            sign = 1 if obj_f[idx] > obj_b[idx] else -1
            signed[b] = sign * extra_grid[idx]

    return {
        "ess": float(np.mean(signed)),
        "outer_sd_ess": float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0,
    }


def stats_from_y(y):
    y = np.asarray(y, dtype=float).ravel()
    return len(y), float(np.sum(y)), float(np.sum(np.square(y)))


def posterior_mean_vector_grid_from_stats(n, sums, sums_sq, grid_df, prior_type, args):
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    if np.any(n <= 0):
        raise ValueError("n must be positive.")
    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)
    prior_type = str(prior_type).lower()

    if prior_type == "informative":
        loc = grid_df["prior_loc"].to_numpy(dtype=float)
        kappa = grid_df["kappa"].to_numpy(dtype=float)
        eta = grid_df["eta"].to_numpy(dtype=float)
        lambda0 = 1.0 / float(args.tau0_sq)
        lambda_n = lambda0 + n[..., None]
        n2 = n[..., None]
        ybar2 = ybar[..., None]
        sse2 = sse[..., None]
        mu_n = (lambda0 * loc[None, :] + sums[..., None]) / lambda_n
        kappa_n = kappa[None, :] + 0.5 * n2
        eta_n = eta[None, :] + 0.5 * (
            sse2 + (lambda0 * n2 / lambda_n) * np.square(ybar2 - loc[None, :])
        )
        if args.parameterization == "mu_log_sigma2":
            second_n = np.log(np.maximum(eta_n, 1e-300)) - special.digamma(kappa_n)
        elif args.parameterization == "mu_sigma2":
            second_n = eta_n / (kappa_n - 1.0)
        else:
            raise ValueError("parameterization must be mu_log_sigma2 or mu_sigma2.")
        return mu_n, second_n

    if prior_type == "jeffreys":
        if np.any(n < 2):
            raise ValueError("Jeffreys posterior requires n >= 2.")
        kappa_n = 0.5 * (n - 1.0)
        eta_n = 0.5 * sse
        if np.any(eta_n <= 0):
            raise ValueError("Jeffreys posterior requires positive residual sum of squares.")
        if args.parameterization == "mu_log_sigma2":
            second_n = np.log(eta_n) - special.digamma(kappa_n)
        elif args.parameterization == "mu_sigma2":
            if np.any(kappa_n <= 1.0):
                raise ValueError(
                    "Jeffreys posterior mean of sigma^2 requires effective sample size > 3."
                )
            second_n = eta_n / (kappa_n - 1.0)
        else:
            raise ValueError("parameterization must be mu_log_sigma2 or mu_sigma2.")
        return ybar, second_n

    raise ValueError("This heatmap runner supports baseline='jeffreys' only.")


def posterior_coordinate_variances_grid_from_stats(
    n, sums, sums_sq, grid_df, prior_type, args
):
    """Return marginal posterior variances in the selected joint coordinates."""
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    if np.any(n <= 0):
        raise ValueError("n must be positive.")
    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)
    prior_type = str(prior_type).lower()

    if prior_type == "informative":
        loc = grid_df["prior_loc"].to_numpy(dtype=float)
        kappa = grid_df["kappa"].to_numpy(dtype=float)
        eta = grid_df["eta"].to_numpy(dtype=float)
        lambda0 = 1.0 / float(args.tau0_sq)
        n2 = n[..., None]
        lambda_n = lambda0 + n2
        kappa_n = kappa[None, :] + 0.5 * n2
        eta_n = eta[None, :] + 0.5 * (
            sse[..., None]
            + (lambda0 * n2 / lambda_n)
            * np.square(ybar[..., None] - loc[None, :])
        )
    elif prior_type == "jeffreys":
        if np.any(n < 2):
            raise ValueError("Jeffreys posterior requires n >= 2.")
        lambda_n = n
        kappa_n = 0.5 * (n - 1.0)
        eta_n = 0.5 * sse
        if np.any(eta_n <= 0.0):
            raise ValueError("Jeffreys posterior requires positive residual sum of squares.")
    else:
        raise ValueError("prior_type must be informative or jeffreys.")

    if np.any(kappa_n <= 1.0):
        raise ValueError("The marginal posterior variance of mu requires kappa_n > 1.")
    var_mu = eta_n / (lambda_n * (kappa_n - 1.0))

    if args.parameterization == "mu_log_sigma2":
        var_second = special.polygamma(1, kappa_n)
    elif args.parameterization == "mu_sigma2":
        if np.any(kappa_n <= 2.0):
            raise ValueError("The posterior variance of sigma^2 requires kappa_n > 2.")
        var_second = np.square(eta_n) / (
            np.square(kappa_n - 1.0) * (kappa_n - 2.0)
        )
    else:
        raise ValueError("parameterization must be mu_log_sigma2 or mu_sigma2.")
    return var_mu, var_second


def target_from_observed(y, args):
    n, sums, sums_sq = stats_from_y(y)
    dummy = pd.DataFrame(
        [{"prior_loc": 0.0, "kappa": 3.0, "eta": 2.0, "tau0_sq": args.tau0_sq}]
    )
    mu_hat, second_hat = posterior_mean_vector_grid_from_stats(
        n, sums, sums_sq, dummy, "jeffreys", args
    )
    return float(mu_hat), float(second_hat)


def bootstrap_stats(y, s, B, rng, require_positive_sse=False):
    y = np.asarray(y, dtype=float).ravel()
    s = int(s)
    B = int(B)
    if s <= 0:
        raise ValueError("Pseudo sample size must be positive.")
    sums = np.empty(B, dtype=float)
    sums_sq = np.empty(B, dtype=float)
    remaining = np.arange(B)
    attempts = 0
    while len(remaining) > 0:
        attempts += 1
        idx = rng.integers(0, len(y), size=(len(remaining), s))
        vals = y[idx]
        local_sum = vals.sum(axis=1, dtype=float)
        local_sq = np.square(vals).sum(axis=1, dtype=float)
        if require_positive_sse:
            local_sse = local_sq - local_sum * local_sum / float(s)
            ok = local_sse > 1e-12
        else:
            ok = np.ones(len(remaining), dtype=bool)
        sums[remaining[ok]] = local_sum[ok]
        sums_sq[remaining[ok]] = local_sq[ok]
        remaining = remaining[~ok]
        if attempts > 1000 and len(remaining) > 0:
            vals = y[
                rng.integers(0, len(y), size=(len(remaining), s))
            ] + rng.normal(0.0, 1e-10, size=(len(remaining), s))
            sums[remaining] = vals.sum(axis=1, dtype=float)
            sums_sq[remaining] = np.square(vals).sum(axis=1, dtype=float)
            break
    return np.full(B, s, dtype=float), sums, sums_sq


def normal_likelihood_stats(mu, sigma2, s, B, rng):
    s = int(s)
    vals = rng.normal(loc=float(mu), scale=np.sqrt(float(sigma2)), size=(int(B), s))
    return (
        np.full(int(B), s, dtype=float),
        vals.sum(axis=1, dtype=float),
        np.square(vals).sum(axis=1, dtype=float),
    )


def loss_for_stats(n, sums, sums_sq, prior_type, grid_df, target, args):
    mu_hat, second_hat = posterior_mean_vector_grid_from_stats(
        n, sums, sums_sq, grid_df, prior_type, args
    )
    target_mu, target_second = target
    loss = np.square(mu_hat - target_mu) + np.square(second_hat - target_second)
    if args.risk_metric == "posterior_mse":
        var_mu, var_second = posterior_coordinate_variances_grid_from_stats(
            n, sums, sums_sq, grid_df, prior_type, args
        )
        loss = loss + var_mu + var_second
    if prior_type == "informative":
        return np.mean(loss, axis=0)
    return float(np.mean(loss))


def run_reimherr_rep(rep, grid_df, args):
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    target = target_from_observed(y, args)
    rng = np.random.default_rng(args.base_seed + 1_000_003 * int(rep) + 101)

    n_stats, sums, sums_sq = bootstrap_stats(
        y, args.n, args.B, rng, require_positive_sse=False
    )
    loss_prior = loss_for_stats(
        n_stats, sums, sums_sq, "informative", grid_df, target, args
    )

    m_grid = np.asarray(args.reimherr_m_grid, dtype=int)
    if args.risk_metric == "posterior_mse":
        min_size = 6 if args.parameterization == "mu_sigma2" else 4
    else:
        min_size = 4 if args.parameterization == "mu_sigma2" else 2
    m_grid = m_grid[args.n + m_grid >= min_size]
    baseline_path = []
    for m in m_grid:
        s = int(args.n + m)
        n_b, sum_b, sq_b = bootstrap_stats(
            y, s, args.B, rng, require_positive_sse=True
        )
        baseline_path.append(
            loss_for_stats(n_b, sum_b, sq_b, "jeffreys", grid_df.iloc[:1], target, args)
        )
    baseline_path = np.asarray(baseline_path, dtype=float)

    rows = []
    for j, spec in grid_df.iterrows():
        diff = np.abs(baseline_path - loss_prior[j])
        idx = int(np.argmin(diff))
        rows.append(base_output_row(y, spec, rep, args, ess=float(m_grid[idx])))
    return rows


def run_wiesenfarth_rep(rep, grid_df, args):
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    target = target_from_observed(y, args)
    target_mu, target_second = target
    target_sigma2 = (
        float(np.exp(target_second))
        if args.parameterization == "mu_log_sigma2"
        else float(target_second)
    )
    rng = np.random.default_rng(args.base_seed + 1_000_003 * int(rep) + 211)

    n_b, sum_b, sq_b = normal_likelihood_stats(
        target_mu, target_sigma2, args.n, args.B, rng
    )
    loss_baseline = loss_for_stats(
        n_b, sum_b, sq_b, "jeffreys", grid_df.iloc[:1], target, args
    )

    m_grid = np.asarray(args.wiesenfarth_m_grid, dtype=int)
    m_grid = m_grid[args.n - m_grid >= 1]
    loss_prior_path = np.empty((len(m_grid), len(grid_df)), dtype=float)
    for i, m in enumerate(m_grid):
        s = int(args.n - m)
        n_p, sum_p, sq_p = normal_likelihood_stats(
            target_mu, target_sigma2, s, args.B, rng
        )
        loss_prior_path[i] = loss_for_stats(
            n_p, sum_p, sq_p, "informative", grid_df, target, args
        )

    rows = []
    for j, spec in grid_df.iterrows():
        diff = np.abs(loss_prior_path[:, j] - loss_baseline)
        idx = int(np.argmin(diff))
        rows.append(base_output_row(y, spec, rep, args, ess=float(m_grid[idx])))
    return rows


def candidate_count(n, k, subset_mode):
    if subset_mode == "with_replacement":
        return comb(n + k - 1, k)
    if k > n:
        return 0
    return comb(n, k)


def iter_index_chunks(n, k, subset_mode, chunk_size):
    iterator = (
        combinations_with_replacement(range(n), k)
        if subset_mode == "with_replacement"
        else combinations(range(n), k)
    )
    while True:
        chunk = list(islice(iterator, int(chunk_size)))
        if not chunk:
            break
        flat = np.fromiter(chain.from_iterable(chunk), dtype=np.int16, count=len(chunk) * k)
        yield flat.reshape(len(chunk), k)


def subset_sufficient_stats(y, idx):
    vals = y[idx]
    sums = vals.sum(axis=1, dtype=float)
    sums_sq = np.square(vals).sum(axis=1, dtype=float)
    k = idx.shape[1]
    ybar = sums / k
    sse = np.maximum(sums_sq - sums * sums / k, 0.0)
    return ybar, sse


def all_candidate_ranges(y, k, args):
    y = np.asarray(y, dtype=float)
    if args.subset_mode == "with_replacement":
        mu_min = float(np.min(y))
        mu_max = float(np.max(y))
        data_range = mu_max - mu_min
        sse_max = 0.25 * k * data_range * data_range
        unique_y = np.unique(y)
        positive_gaps = np.diff(unique_y)
        positive_gaps = positive_gaps[positive_gaps > 0.0]
        if len(positive_gaps) == 0:
            return np.inf, -np.inf, mu_min, mu_max
        min_sse = ((k - 1.0) / k) * float(np.min(positive_gaps)) ** 2
        return 0.5 * min_sse, 0.5 * sse_max, mu_min, mu_max

    eta_min = np.inf
    eta_max = -np.inf
    mu_min = np.inf
    mu_max = -np.inf
    for idx in iter_index_chunks(args.n, k, args.subset_mode, args.chunk_size):
        ybar, sse = subset_sufficient_stats(y, idx)
        valid = sse > 0.0
        if np.any(valid):
            eta_p = 0.5 * sse[valid]
            eta_min = min(eta_min, float(np.min(eta_p)))
            eta_max = max(eta_max, float(np.max(eta_p)))
            mu_min = min(mu_min, float(np.min(ybar[valid])))
            mu_max = max(mu_max, float(np.max(ybar[valid])))
    return eta_min, eta_max, mu_min, mu_max


def jeffreys_subset_posterior(k, ybar, sse):
    valid = sse > 0.0
    return ybar, 1.0 / float(k), 0.5 * (k - 1.0), 0.5 * sse, valid


def kl_nig_to_prior_grid(mu1, tau1_sq, kappa1, eta1, grid_df):
    mu1 = np.asarray(mu1, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)
    prior_locs = grid_df["prior_loc"].to_numpy(dtype=float)
    kappa2 = grid_df["kappa"].to_numpy(dtype=float)
    eta2 = grid_df["eta"].to_numpy(dtype=float)
    tau2_sq = grid_df["tau0_sq"].to_numpy(dtype=float)

    kl_sigma2 = (
        special.gammaln(kappa2)[None, :]
        - special.gammaln(kappa1)
        + kappa2[None, :] * (np.log(eta1)[:, None] - np.log(eta2)[None, :])
        + (kappa1 - kappa2[None, :]) * special.digamma(kappa1)
        + kappa1 * (eta2[None, :] - eta1[:, None]) / eta1[:, None]
    )

    e_inv_sigma2 = kappa1 / eta1
    delta2 = np.square(mu1[:, None] - prior_locs[None, :])
    kl_mu = 0.5 * (
        np.log(tau2_sq[None, :] / tau1_sq)
        + tau1_sq / tau2_sq[None, :]
        - 1.0
        + (delta2 / tau2_sq[None, :]) * e_inv_sigma2[:, None]
    )
    return np.maximum(kl_sigma2 + kl_mu, 0.0)


def run_clarke_rep(rep, grid_df, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    y = rng.normal(loc=args.true_mu, scale=args.true_sigma, size=args.n)
    n_prior = len(grid_df)
    best_d = np.full(n_prior, np.inf, dtype=float)
    best_k = np.full(n_prior, -1, dtype=int)

    for k in args.k_grid:
        k = int(k)
        n_candidates = candidate_count(args.n, k, args.subset_mode)
        if n_candidates <= 0:
            continue
        if n_candidates > args.max_candidates_per_k:
            raise RuntimeError(
                f"k={k} has {n_candidates} candidates under {args.subset_mode}, "
                f"above --max-candidates-per-k={args.max_candidates_per_k}."
            )

        k_best_d = np.full(n_prior, np.inf, dtype=float)
        for idx in iter_index_chunks(args.n, k, args.subset_mode, args.chunk_size):
            ybar, sse = subset_sufficient_stats(y, idx)
            mu1, tau1_sq, kappa1, eta1, valid = jeffreys_subset_posterior(k, ybar, sse)
            if not np.any(valid):
                continue
            dist = kl_nig_to_prior_grid(
                mu1=mu1[valid],
                tau1_sq=tau1_sq,
                kappa1=kappa1,
                eta1=eta1[valid],
                grid_df=grid_df,
            )
            chunk_mins = np.min(dist, axis=0)
            improved = chunk_mins < k_best_d
            k_best_d[improved] = chunk_mins[improved]

        improved_global = k_best_d < best_d
        best_d[improved_global] = k_best_d[improved_global]
        best_k[improved_global] = k

    rows = []
    for j, spec in grid_df.iterrows():
        row = base_output_row(y, spec, rep, args, ess=float(best_k[j]))
        row["d_min"] = float(best_d[j])
        row["distance"] = "nig_kl"
        rows.append(row)
    return rows


def run_clarke_truncated_rep(rep, grid_df, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    y = rng.normal(loc=args.true_mu, scale=args.true_sigma, size=args.n)
    n_prior = len(grid_df)
    best_d = np.full(n_prior, np.inf, dtype=float)
    best_k = np.full(n_prior, -1, dtype=int)
    lambda_q = 1.0 / float(args.tau0_sq)
    log_t_power = 0.5 if args.parameterization == "mu_log_sigma2" else 1.5

    grouped = [
        (indices.to_numpy(dtype=int), group.copy())
        for _, group in grid_df.groupby("sigma2_prior_mean", sort=False)
        for indices in [group.index]
    ]

    for k in args.k_grid:
        k = int(k)
        n_candidates = candidate_count(args.n, k, args.subset_mode)
        if n_candidates <= 0:
            continue
        if n_candidates > args.max_candidates_per_k:
            raise RuntimeError(
                f"k={k} has {n_candidates} candidates under {args.subset_mode}, "
                f"above --max-candidates-per-k={args.max_candidates_per_k}."
            )

        eta_min, eta_max, mu_min, mu_max = all_candidate_ranges(y, k, args)
        if not np.isfinite(eta_min):
            continue

        lambda_p = float(k)
        kappa_p = 0.5 * (k - 1.0)
        hdr = canonical_logspace_hdr_moments(
            kappa_p, args.alpha, args.hdr_sample_power, log_t_power
        )
        k_best_d = np.full(n_prior, np.inf, dtype=float)

        group_tables = {}
        for group_index, group in grouped:
            prior_locs = group["prior_loc"].to_numpy(dtype=float)
            delta_max = max(
                abs(mu_min - float(np.max(prior_locs))),
                abs(mu_max - float(np.min(prior_locs))),
            )
            log_eta_bounds = (
                np.log(eta_min) - float(args.interpolation_pad),
                np.log(eta_max) + float(args.interpolation_pad),
            )
            group_tables[tuple(group_index.tolist())] = build_q_hdr_probability_table(
                lambda_p=lambda_p,
                kappa_p=kappa_p,
                lambda_q=lambda_q,
                kappa_q=float(group["kappa"].iloc[0]),
                eta_q=float(group["eta"].iloc[0]),
                hdr_threshold=hdr["threshold"],
                log_eta_bounds=log_eta_bounds,
                delta_max=delta_max * (1.0 + float(args.interpolation_pad)) + 1e-8,
                eta_grid_size=args.eta_grid_size,
                delta_grid_size=args.delta_grid_size,
                quadrature_size=args.quadrature_size,
                log_t_power=log_t_power,
            )

        for idx in iter_index_chunks(args.n, k, args.subset_mode, args.chunk_size):
            ybar, sse = subset_sufficient_stats(y, idx)
            valid = sse > 0.0
            if not np.any(valid):
                continue
            mu_p = ybar[valid]
            eta_p = 0.5 * sse[valid]
            for group_index, group in grouped:
                key = tuple(group_index.tolist())
                dist = truncated_kl_logspace_batched(
                    mu_p=mu_p,
                    lambda_p=lambda_p,
                    kappa_p=kappa_p,
                    eta_p=eta_p,
                    mu_q=group["prior_loc"].to_numpy(dtype=float),
                    lambda_q=lambda_q,
                    kappa_q=float(group["kappa"].iloc[0]),
                    eta_q=float(group["eta"].iloc[0]),
                    hdr_moments=hdr,
                    probability_table=group_tables[key],
                    eps=args.eps,
                )
                chunk_mins = np.min(dist, axis=0)
                improved = chunk_mins < k_best_d[group_index]
                if np.any(improved):
                    k_best_d[group_index[improved]] = chunk_mins[improved]

        improved_global = k_best_d < best_d
        best_d[improved_global] = k_best_d[improved_global]
        best_k[improved_global] = k

    rows = []
    for j, spec in grid_df.iterrows():
        row = base_output_row(y, spec, rep, args, ess=float(best_k[j]))
        row["d_min"] = float(best_d[j])
        row["distance"] = "truncated_kl"
        rows.append(row)
    return rows


def base_output_row(y, spec, rep, args, ess):
    return {
        "model": "normal_unknown",
        "method": args.method,
        "baseline": args.baseline,
        "rep": int(rep),
        "n": int(args.n),
        "grid_id": int(spec["grid_id"]),
        "prior_value": float(spec["prior_loc"]),
        "prior_loc": float(spec["prior_loc"]),
        "sigma2_prior_mean": float(spec["sigma2_prior_mean"]),
        "sigma2_prior_var": float(spec["sigma2_prior_var"]),
        "tau0_sq": float(spec["tau0_sq"]),
        "kappa": float(spec["kappa"]),
        "eta": float(spec["eta"]),
        "parameterization": args.parameterization,
        "sample_mean": float(np.mean(y)),
        "sample_var": float(np.var(y, ddof=1)) if len(y) > 1 else 0.0,
        "ess": float(ess),
    }


def run_jones_cp_task(rep, spec_dict, args):
    spec = pd.Series(spec_dict)
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    seed = int(args.base_seed + 1_000_003 * int(rep) + grid_seed_offset(spec))
    rng = np.random.default_rng(seed)

    if args.method == "jones_joint2d":
        out = run_jones_one(y, spec, args, rng)
    elif args.method == "cp_joint2d":
        out = run_cp_one(y, spec, args, rng)
    else:
        raise ValueError("run_jones_cp_task only handles Jones and CP.")

    row = base_output_row(y, spec, rep, args, ess=out["ess"])
    row["outer_sd_ess"] = out["outer_sd_ess"]
    row["sampling_method"] = args.sampling_method
    row["m_grid_semantics"] = (
        "number_deleted"
        if is_deletion_sampling(args.sampling_method)
        else (
            "number_augmented"
            if is_empirical_augmentation(args.sampling_method)
            or args.method == "cp_joint2d"
            else "total_sample_size"
        )
    )
    return row


def summarize(df):
    return (
        df.groupby(
            [
                "model",
                "method",
                "baseline",
                "prior_loc",
                "sigma2_prior_mean",
                "sigma2_prior_var",
                "tau0_sq",
                "kappa",
                "eta",
            ]
        )["ess"]
        .agg(mean_ess="mean", sd_ess="std", median_ess="median", n_rep="count")
        .reset_index()
        .sort_values(["sigma2_prior_mean", "prior_loc"])
    )


def save_outputs(df, summary, grid_df, args, runtime_seconds):
    method_dir = args.method
    if args.method == "reimherr_joint2d" and args.risk_metric == "posterior_mse":
        method_dir = "reimherr_joint2d_posterior_mse"
    method_path = Path(method_dir)
    if (
        args.method in {"jones_joint2d", "cp_joint2d"}
        and is_exploratory_sampling(args.sampling_method)
    ):
        method_path = method_path / args.sampling_method
    out_dir = Path(args.output_root) / method_path / args.parameterization
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    grid_df.to_csv(out_dir / "prior_grid.csv", index=False)
    runtime = {
        "method": args.method,
        "runtime_seconds": float(runtime_seconds),
        "runtime_minutes": float(runtime_seconds / 60.0),
        "n_rows": int(len(df)),
        "seconds_per_row": float(runtime_seconds / len(df)) if len(df) else None,
    }
    config = {
        "model": "normal_unknown",
        "method": args.method,
        "baseline": args.baseline,
        "parameterization": args.parameterization,
        "mu_scale": float(args.mu_scale),
        "sigma2_scale": float(args.sigma2_scale),
        "log_sigma2_scale": float(args.log_sigma2_scale),
        "R": int(args.R),
        "rep_start": int(args.rep_start),
        "rep_end": int(args.rep_end),
        "n": int(args.n),
        "true_mu": float(args.true_mu),
        "true_sigma": float(args.true_sigma),
        "tau0_sq": float(args.tau0_sq),
        "sigma2_prior_var": float(args.sigma2_prior_var),
        "n_grid": int(len(grid_df)),
        "max_grid_points": int(args.max_grid_points),
        "mu_grid": sorted(grid_df["prior_loc"].unique().tolist()),
        "sigma2_mean_grid": sorted(grid_df["sigma2_prior_mean"].unique().tolist()),
        "data_seed": int(args.data_seed),
        "base_seed": int(args.base_seed),
        "sampling_method": (
            args.sampling_method
            if args.method in {"jones_joint2d", "cp_joint2d"}
            else None
        ),
        "m_grid_semantics": (
            "number_deleted"
            if is_deletion_sampling(args.sampling_method)
            else (
                "number_augmented"
                if is_empirical_augmentation(args.sampling_method)
                or args.method == "cp_joint2d"
                else "total_sample_size"
            )
        ),
        "min_retained": int(args.min_retained),
        "negative_direction": (
            (
                "pi(.|X_n) vs pi_b(.|Y_{n-m})"
                if is_deletion_sampling(args.sampling_method)
                else "pi_b(.|X_n) vs pi(.|X_n union A_m)"
            )
            if args.method in {"jones_joint2d", "cp_joint2d"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "positive_direction": (
            (
                "pi(.|Y_{n-m}) vs pi_b(.|X_n)"
                if is_deletion_sampling(args.sampling_method)
                else "pi(.|X_n) vs pi_b(.|X_n union A_m)"
            )
            if args.method in {"jones_joint2d", "cp_joint2d"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "outer_selection_rule": (
            "minimize_each_direction_then_compare_then_average_signed_m"
            if args.method in {"jones_joint2d", "cp_joint2d"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "runtime": runtime,
    }
    if args.method == "jones_joint2d":
        config.update(
            {
                "outer_reps": int(args.outer_reps),
                "joint_draws": int(args.joint_draws),
                "n_projections": int(args.n_projections),
                "jones_m_grid": args.jones_m_grid.tolist(),
                "distance": "sliced_wasserstein2",
            }
        )
    elif args.method == "cp_joint2d":
        config.update(
            {
                "outer_reps": int(args.outer_reps),
                "joint_draws": int(args.joint_draws),
                "alpha": float(args.alpha),
                "cp_m_grid": args.cp_m_grid.tolist(),
                "distance": "joint_hdr_cp",
            }
        )
    elif args.method == "reimherr_joint2d":
        config.update(
            {
                "B": int(args.B),
                "variant": "bootstrap",
                "risk_metric": args.risk_metric,
                "posterior_uncertainty_included": args.risk_metric == "posterior_mse",
                "reimherr_m_grid": args.reimherr_m_grid.tolist(),
            }
        )
    elif args.method == "wiesenfarth_joint2d":
        config.update(
            {
                "B": int(args.B),
                "variant": "plugin_likelihood",
                "risk_metric": "trace_mse",
                "wiesenfarth_m_grid": args.wiesenfarth_m_grid.tolist(),
            }
        )
    elif args.method == "clarke_nig2d_kl":
        config.update(
            {
                "distance": "nig_kl",
                "search": args.subset_mode,
                "k_grid": args.k_grid.tolist(),
                "chunk_size": int(args.chunk_size),
                "max_candidates_per_k": int(args.max_candidates_per_k),
            }
        )
    elif args.method == "clarke_nig2d_truncated_kl":
        config.update(
            {
                "distance": "truncated_kl",
                "parameterization": args.parameterization,
                "hdr_definition": f"highest_density_region_in_{args.parameterization}",
                "implementation": "canonical_hdr_moments_plus_qmass_2d_interpolation_by_sigma2_mean",
                "search": args.subset_mode,
                "alpha": float(args.alpha),
                "log_t_power": 0.5 if args.parameterization == "mu_log_sigma2" else 1.5,
                "hdr_sample_power": int(args.hdr_sample_power),
                "eta_grid_size": int(args.eta_grid_size),
                "delta_grid_size": int(args.delta_grid_size),
                "quadrature_size": int(args.quadrature_size),
                "interpolation_pad": float(args.interpolation_pad),
                "eps": float(args.eps),
                "k_grid": args.k_grid.tolist(),
                "chunk_size": int(args.chunk_size),
                "max_candidates_per_k": int(args.max_candidates_per_k),
            }
        )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "NIG 2D EPSS heatmap over informative prior mean of mu and "
            "informative prior mean of sigma^2, with fixed Var(sigma^2)."
        )
    )
    parser.add_argument("--method", required=True, choices=sorted(METHOD_ALIASES))
    parser.add_argument("--output-root", default="results/NIG_sigma2_mean_heatmap")
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--prior-values", default=None)
    parser.add_argument("--sigma2-mean-values", default=None)
    parser.add_argument("--sigma2-prior-var", type=float, default=2.0)
    parser.add_argument("--max-grid-points", type=int, default=300)
    parser.add_argument("--baseline", choices=["jeffreys"], default="jeffreys")
    parser.add_argument("--true-mu", type=float, default=0.0)
    parser.add_argument("--true-sigma", type=float, default=1.0)
    parser.add_argument("--tau0-sq", type=float, default=0.2)
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument(
        "--risk-metric",
        choices=["squared_mean", "posterior_mse"],
        default="squared_mean",
        help=(
            "Reimherr joint loss. posterior_mse adds the analytic posterior "
            "covariance trace to squared posterior-mean error."
        ),
    )
    parser.add_argument("--outer-reps", type=int, default=500)
    parser.add_argument(
        "--sampling-method",
        choices=SAMPLING_METHODS,
        default="ppd_augmentation",
    )
    parser.add_argument("--min-retained", type=int, default=3)
    parser.add_argument("--joint-draws", type=int, default=512)
    parser.add_argument("--n-projections", type=int, default=16)
    parser.add_argument(
        "--parameterization",
        choices=["mu_log_sigma2", "mu_sigma2"],
        default="mu_log_sigma2",
    )
    parser.add_argument("--mu-scale", type=float, default=1.0)
    parser.add_argument("--sigma2-scale", type=float, default=1.0)
    parser.add_argument("--log-sigma2-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--hdr-sample-power", type=int, default=18)
    parser.add_argument("--eta-grid-size", type=int, default=80)
    parser.add_argument("--delta-grid-size", type=int, default=181)
    parser.add_argument("--quadrature-size", type=int, default=128)
    parser.add_argument("--interpolation-pad", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--jones-m-grid", default=None)
    parser.add_argument("--cp-m-grid", default=None)
    parser.add_argument("--reimherr-m-grid", default=None)
    parser.add_argument("--wiesenfarth-m-grid", default=None)
    parser.add_argument("--k-grid", default=None)
    parser.add_argument(
        "--subset-mode",
        choices=["with_replacement", "without_replacement"],
        default="with_replacement",
    )
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--max-candidates-per-k", type=int, default=10_000_000)
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--base-seed", type=int, default=20260312)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)
    args = parser.parse_args()
    args.method = METHOD_ALIASES[args.method]
    args.rep_end = args.R if args.rep_end is None else args.rep_end
    if is_deletion_sampling(args.sampling_method):
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(0, args.n - args.min_retained + 1)
        )
        args.jones_m_grid = args.jones_m_grid[
            (args.jones_m_grid >= 0)
            & (args.n - args.jones_m_grid >= args.min_retained)
        ]
    elif is_empirical_augmentation(args.sampling_method):
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(0, args.n + 1)
        )
        args.jones_m_grid = args.jones_m_grid[
            args.jones_m_grid >= 0
        ]
    else:
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(args.n, 2 * args.n + 1)
        )
        args.jones_m_grid = args.jones_m_grid[
            args.jones_m_grid >= args.n
        ]
    cp_default_grid = (
        range(0, args.n + 1)
        if is_empirical_augmentation(args.sampling_method)
        else range(0, args.n + 1)
    )
    args.cp_m_grid = parse_int_grid(args.cp_m_grid, cp_default_grid)
    args.cp_m_grid = args.cp_m_grid[args.cp_m_grid >= 0]
    if is_deletion_sampling(args.sampling_method):
        args.cp_m_grid = args.cp_m_grid[
            args.n - args.cp_m_grid >= args.min_retained
        ]
    args.reimherr_m_grid = parse_int_grid(
        args.reimherr_m_grid, range(-(args.n - 1), args.n + 1)
    )
    if args.risk_metric == "posterior_mse":
        reimherr_min_size = 6 if args.parameterization == "mu_sigma2" else 4
    else:
        reimherr_min_size = 4 if args.parameterization == "mu_sigma2" else 2
    args.reimherr_m_grid = args.reimherr_m_grid[
        args.n + args.reimherr_m_grid >= reimherr_min_size
    ]
    args.wiesenfarth_m_grid = parse_int_grid(
        args.wiesenfarth_m_grid, range(-args.n, args.n)
    )
    args.wiesenfarth_m_grid = args.wiesenfarth_m_grid[args.n - args.wiesenfarth_m_grid >= 1]
    args.k_grid = parse_int_grid(args.k_grid, range(2, 10))
    args.k_grid = args.k_grid[(args.k_grid >= 2) & (args.k_grid <= args.n)]
    if len(args.k_grid) == 0:
        raise ValueError("k-grid has no valid values.")
    if args.outer_reps <= 0 or args.joint_draws <= 1:
        raise ValueError("outer-reps must be positive and joint-draws must exceed one.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if args.mu_scale <= 0 or args.sigma2_scale <= 0 or args.log_sigma2_scale <= 0:
        raise ValueError("coordinate scales must be positive.")
    return args


def main():
    args = parse_args()
    grid_df = make_prior_grid(args)
    start = time.perf_counter()

    if args.method in {"jones_joint2d", "cp_joint2d"}:
        tasks = [
            (rep, spec, args)
            for rep in range(args.rep_start, args.rep_end)
            for spec in grid_df.to_dict("records")
        ]
        rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_jones_cp_task)(*task) for task in tasks
        )
    elif args.method == "reimherr_joint2d":
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_reimherr_rep)(rep, grid_df, args)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    elif args.method == "wiesenfarth_joint2d":
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_wiesenfarth_rep)(rep, grid_df, args)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    elif args.method == "clarke_nig2d_kl":
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_clarke_rep)(rep, grid_df, args)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    elif args.method == "clarke_nig2d_truncated_kl":
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_clarke_truncated_rep)(rep, grid_df, args)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    df = pd.DataFrame(rows)
    summary = summarize(df)
    runtime_seconds = time.perf_counter() - start
    out_dir = save_outputs(df, summary, grid_df, args, runtime_seconds)
    print(
        f"{args.method}: {len(df)} rows, {len(grid_df)} grid points, "
        f"{runtime_seconds:.2f}s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
