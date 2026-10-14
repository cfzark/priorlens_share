import argparse
import copy
import json
import sys
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage, special, stats
from scipy.integrate import quad
from scipy.special import beta as beta_function


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "function", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_normal_unknown_clarke_exhaustive_fast as clarke_fast  # noqa: E402
import run_normal_unknown_methods as marginal_methods  # noqa: E402
import run_rw_sensitivity as rw_methods  # noqa: E402
from prioreffect.sampling import (  # noqa: E402
    SAMPLING_METHODS,
    empirical_bootstrap_augmentation_matrix,
    empirical_deletion_matrix,
    is_deletion_sampling,
    is_empirical_augmentation,
    is_exploratory_sampling,
)


METHOD_ALIASES = {
    "jones": "jones",
    "cp": "cp",
    "reimherr": "reimherr",
    "wiesenfarth": "wiesenfarth",
    "clarke_kl": "clarke_kl",
    "clarke_tkl": "clarke_tkl",
    "clarke_truncated_kl": "clarke_tkl",
    "morita": "morita",
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
    values = [float(item) for item in str(value).replace(",", " ").split()]
    if not values:
        raise ValueError("Empty numeric grid.")
    return np.asarray(values, dtype=float)


def parse_int_grid(value, default):
    if value is None:
        return np.asarray(list(default), dtype=int)
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            lower, upper = item.split(":", 1)
            values.extend(range(int(lower), int(upper) + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("Empty integer grid.")
    return np.asarray(sorted(set(values)), dtype=int)


def ig_params_from_mean_var(mean, var):
    mean = float(mean)
    var = float(var)
    if mean <= 0.0 or var <= 0.0:
        raise ValueError("IG mean and variance must be positive.")
    kappa = 2.0 + mean * mean / var
    eta = mean * (kappa - 1.0)
    return float(kappa), float(eta)


def make_prior_grid(args):
    mu_values = parse_float_grid(args.prior_values, default_mu_grid())
    sigma2_means = parse_float_grid(
        args.sigma2_mean_values, default_sigma2_mean_grid()
    )
    if len(mu_values) * len(sigma2_means) > args.max_grid_points:
        raise ValueError(
            f"Requested {len(mu_values) * len(sigma2_means)} grid points, "
            f"above --max-grid-points={args.max_grid_points}."
        )
    rows = []
    for grid_id, (mu0, sigma2_mean) in enumerate(
        (mu0, sigma2_mean)
        for sigma2_mean in sigma2_means
        for mu0 in mu_values
    ):
        kappa, eta = ig_params_from_mean_var(
            sigma2_mean, args.sigma2_prior_var
        )
        rows.append(
            {
                "grid_id": int(grid_id),
                "prior_loc": float(mu0),
                "prior_value": float(mu0),
                "sigma2_prior_mean": float(sigma2_mean),
                "sigma2_prior_var": float(args.sigma2_prior_var),
                "tau0_sq": float(args.tau0_sq),
                "kappa": kappa,
                "eta": eta,
            }
        )
    return pd.DataFrame(rows)


def add_grid_metadata(row, spec, args, ess):
    row = dict(row)
    row.update(
        {
            "target": "marginal_mu",
            "grid_id": int(spec["grid_id"]),
            "prior_loc": float(spec["prior_loc"]),
            "prior_value": float(spec["prior_loc"]),
            "sigma2_prior_mean": float(spec["sigma2_prior_mean"]),
            "sigma2_prior_var": float(spec["sigma2_prior_var"]),
            "tau0_sq": float(args.tau0_sq),
            "kappa": float(spec["kappa"]),
            "eta": float(spec["eta"]),
            "ess": float(ess),
        }
    )
    return row


def marginal_mu_student_params(n, sums, sums_sq, spec, prior_type, args):
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)
    if prior_type == "jeffreys":
        if np.any(n < 2.0) or np.any(sse <= 0.0):
            raise ValueError("Jeffreys marginal-mu posterior requires n >= 2 and positive SSE.")
        df = n - 1.0
        loc = ybar
        scale = np.sqrt(sse / (n * (n - 1.0)))
        return df, loc, scale
    if prior_type == "informative":
        lambda0 = 1.0 / float(args.tau0_sq)
        lambda_n = lambda0 + n
        kappa_n = float(spec["kappa"]) + 0.5 * n
        eta_n = float(spec["eta"]) + 0.5 * (
            sse
            + lambda0 * n / lambda_n
            * np.square(ybar - float(spec["prior_loc"]))
        )
        loc = (lambda0 * float(spec["prior_loc"]) + sums) / lambda_n
        return 2.0 * kappa_n, loc, np.sqrt(eta_n / (kappa_n * lambda_n))
    raise ValueError("prior_type must be informative or jeffreys.")


def informative_predictive_params(y, spec, args):
    n = len(y)
    sums = float(np.sum(y))
    sums_sq = float(np.sum(np.square(y)))
    df, loc, _ = marginal_mu_student_params(
        n, sums, sums_sq, spec, "informative", args
    )
    lambda_n = 1.0 / float(args.tau0_sq) + n
    ybar = sums / n
    sse = max(sums_sq - sums * sums / n, 0.0)
    kappa_n = float(spec["kappa"]) + 0.5 * n
    eta_n = float(spec["eta"]) + 0.5 * (
        sse
        + (1.0 / float(args.tau0_sq)) * n / lambda_n
        * (ybar - float(spec["prior_loc"])) ** 2
    )
    scale = np.sqrt(eta_n / kappa_n * (1.0 + 1.0 / lambda_n))
    return float(df), float(loc), float(scale)


def trapz_moments_for_student_pair(df1, df2, grid_size, eps=1e-6):
    u = np.linspace(float(eps), 1.0 - float(eps), int(grid_size))
    w = clarke_fast.trapz_weights(u)
    a = stats.t.ppf(u, df=float(df1))
    b = stats.t.ppf(u, df=float(df2))
    return (
        float(np.sum(w)),
        float(np.sum(w * a)),
        float(np.sum(w * b)),
        float(np.sum(w * a * a)),
        float(np.sum(w * b * b)),
        float(np.sum(w * a * b)),
    )


def student_w2_vectorized(loc1, scale1, df1, loc2, scale2, df2, grid_size):
    loc1, scale1, loc2, scale2 = np.broadcast_arrays(
        np.asarray(loc1, dtype=float),
        np.asarray(scale1, dtype=float),
        np.asarray(loc2, dtype=float),
        np.asarray(scale2, dtype=float),
    )
    w0, a1, b1, a2, b2, ab = trapz_moments_for_student_pair(
        df1, df2, grid_size
    )
    delta = loc1 - loc2
    value = (
        w0 * np.square(delta)
        + 2.0 * delta * (scale1 * a1 - scale2 * b1)
        + np.square(scale1) * a2
        + np.square(scale2) * b2
        - 2.0 * scale1 * scale2 * ab
    )
    return np.sqrt(np.maximum(value, 0.0))


def run_jones_task_vectorized(rep, spec_dict, args):
    spec = pd.Series(spec_dict)
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    rng = np.random.default_rng(
        args.base_seed
        + 1_000_000 * int(rep)
        + int(round((float(spec["prior_loc"]) + 10.0) * 10_000))
    )
    deletion = is_deletion_sampling(args.sampling_method)
    empirical_augmentation = is_empirical_augmentation(
        args.sampling_method
    )
    total_grid = np.asarray(args.jones_m_grid, dtype=int)
    extra_grid = (
        total_grid - int(args.n)
        if not deletion and not empirical_augmentation
        else None
    )
    n0 = len(y)
    sum0 = float(np.sum(y))
    sq0 = float(np.sum(np.square(y)))
    df_inf0, loc_inf0, scale_inf0 = marginal_mu_student_params(
        n0, sum0, sq0, spec, "informative", args
    )
    df_base0, loc_base0, scale_base0 = marginal_mu_student_params(
        n0, sum0, sq0, spec, "jeffreys", args
    )
    pred_df, pred_loc, pred_scale = informative_predictive_params(y, spec, args)
    d = np.empty((args.S, len(total_grid)), dtype=float)
    dsharp = np.empty_like(d)

    if deletion:
        sequences = empirical_deletion_matrix(
            y,
            args.S,
            rng,
            args.sampling_method,
            min_retained=int(args.n - np.max(total_grid)),
            require_positive_variance=True,
        )
        prefix_sum = np.concatenate(
            [np.zeros((args.S, 1)), np.cumsum(sequences, axis=1)], axis=1
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.S, 1)),
                np.cumsum(np.square(sequences), axis=1),
            ],
            axis=1,
        )
        loop_values = [
            (int(args.n - removed), int(removed))
            for removed in total_grid
        ]
    elif empirical_augmentation:
        sequences = empirical_bootstrap_augmentation_matrix(
            y,
            args.S,
            int(np.max(total_grid)),
            rng,
        )
        prefix_sum = np.concatenate(
            [np.zeros((args.S, 1)), np.cumsum(sequences, axis=1)],
            axis=1,
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.S, 1)),
                np.cumsum(np.square(sequences), axis=1),
            ],
            axis=1,
        )
        loop_values = [
            (int(args.n + added), int(added))
            for added in total_grid
        ]
    else:
        loop_values = list(zip(total_grid, extra_grid))

    for j, (total, extra) in enumerate(loop_values):
        if deletion:
            sums_reduced = prefix_sum[:, int(total)]
            sq_reduced = prefix_sq[:, int(total)]
            df_base, loc_base, scale_base = marginal_mu_student_params(
                total,
                sums_reduced,
                sq_reduced,
                spec,
                "jeffreys",
                args,
            )
            df_inf, loc_inf, scale_inf = marginal_mu_student_params(
                total,
                sums_reduced,
                sq_reduced,
                spec,
                "informative",
                args,
            )
        elif empirical_augmentation:
            sums_augmented = sum0 + prefix_sum[:, int(extra)]
            sq_augmented = sq0 + prefix_sq[:, int(extra)]
            df_base, loc_base, scale_base = marginal_mu_student_params(
                total,
                sums_augmented,
                sq_augmented,
                spec,
                "jeffreys",
                args,
            )
            df_inf, loc_inf, scale_inf = marginal_mu_student_params(
                total,
                sums_augmented,
                sq_augmented,
                spec,
                "informative",
                args,
            )
        elif extra == 0:
            df_base, loc_base, scale_base = df_base0, loc_base0, scale_base0
            df_inf, loc_inf, scale_inf = df_inf0, loc_inf0, scale_inf0
        else:
            future_d = stats.t.rvs(
                df=pred_df,
                loc=pred_loc,
                scale=pred_scale,
                size=(args.S, int(extra)),
                random_state=rng,
            )
            future_dsharp = stats.t.rvs(
                df=pred_df,
                loc=pred_loc,
                scale=pred_scale,
                size=(args.S, int(extra)),
                random_state=rng,
            )
            df_base, loc_base, scale_base = marginal_mu_student_params(
                total,
                sum0 + future_d.sum(axis=1),
                sq0 + np.square(future_d).sum(axis=1),
                spec,
                "jeffreys",
                args,
            )
            df_inf, loc_inf, scale_inf = marginal_mu_student_params(
                total,
                sum0 + future_dsharp.sum(axis=1),
                sq0 + np.square(future_dsharp).sum(axis=1),
                spec,
                "informative",
                args,
            )
        if deletion:
            d[:, j] = student_w2_vectorized(
                loc_inf0,
                scale_inf0,
                df_inf0,
                loc_base,
                scale_base,
                float(np.asarray(df_base).flat[0]),
                args.wasserstein_grid_size,
            )
            dsharp[:, j] = student_w2_vectorized(
                loc_inf,
                scale_inf,
                float(np.asarray(df_inf).flat[0]),
                loc_base0,
                scale_base0,
                df_base0,
                args.wasserstein_grid_size,
            )
        elif empirical_augmentation:
            d[:, j] = student_w2_vectorized(
                loc_base0,
                scale_base0,
                df_base0,
                loc_inf,
                scale_inf,
                float(np.asarray(df_inf).flat[0]),
                args.wasserstein_grid_size,
            )
            dsharp[:, j] = student_w2_vectorized(
                loc_inf0,
                scale_inf0,
                df_inf0,
                loc_base,
                scale_base,
                float(np.asarray(df_base).flat[0]),
                args.wasserstein_grid_size,
            )
        else:
            d[:, j] = student_w2_vectorized(
                loc_base,
                scale_base,
                float(np.asarray(df_base).flat[0]),
                loc_inf0,
                scale_inf0,
                df_inf0,
                args.wasserstein_grid_size,
            )
            dsharp[:, j] = student_w2_vectorized(
                loc_base0,
                scale_base0,
                df_base0,
                loc_inf,
                scale_inf,
                float(np.asarray(df_inf).flat[0]),
                args.wasserstein_grid_size,
            )

    i = np.argmin(d, axis=1)
    j = np.argmin(dsharp, axis=1)
    rows = np.arange(args.S)
    if deletion:
        signed = np.where(
            d[rows, i] <= dsharp[rows, j],
            -total_grid[i],
            total_grid[j],
        )
    elif empirical_augmentation:
        signed = np.where(
            d[rows, i] <= dsharp[rows, j],
            -total_grid[i],
            total_grid[j],
        )
    else:
        signed = np.where(
            d[rows, i] <= dsharp[rows, j],
            extra_grid[i],
            -extra_grid[j],
        )
    row = {
        "model": "normal_unknown",
        "method": "jones",
        "variant": "vectorized_student_w2",
        "sampling_method": args.sampling_method,
        "m_grid_semantics": (
            "number_deleted"
            if deletion
            else (
                "number_augmented"
                if empirical_augmentation
                else "total_sample_size"
            )
        ),
        "baseline": "jeffreys",
        "rep": int(rep),
        "n": int(args.n),
        "sample_mean": float(np.mean(y)),
        "sample_var": float(np.var(y, ddof=1)),
        "mopess": float(np.mean(signed)),
        "outer_sd_ess": float(np.std(signed, ddof=1)),
        "best_m_D": int(total_grid[int(np.argmin(np.mean(d, axis=0)))]),
        "best_m_Dsharp": int(
            total_grid[int(np.argmin(np.mean(dsharp, axis=0)))]
        ),
    }
    return add_grid_metadata(row, spec, args, float(np.mean(signed)))


def run_jones_cp_task(rep, spec_dict, args):
    if args.method == "jones":
        return run_jones_task_vectorized(rep, spec_dict, args)
    spec = pd.Series(spec_dict)
    local = copy.copy(args)
    local.kappa = float(spec["kappa"])
    local.eta = float(spec["eta"])
    local.method = args.method
    local.distance = "wasserstein"
    row = marginal_methods.run_one(float(spec["prior_loc"]), int(rep), local)
    ess = float(row["epss"])
    keep = {
        "outer_sd_ess": float(row.get("sd_signed_mstar", np.nan)),
        "mean_forward_mstar": row.get("mean_forward_mstar", np.nan),
        "mean_backward_mstar": row.get("mean_backward_mstar", np.nan),
    }
    compact = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (list, tuple, dict, np.ndarray))
    }
    compact.update(keep)
    return add_grid_metadata(compact, spec, args, ess)


def run_rw_task(rep, spec_dict, args):
    spec = pd.Series(spec_dict)
    local = copy.copy(args)
    local.normal_unknown_kappa = float(spec["kappa"])
    local.normal_unknown_eta = float(spec["eta"])
    method = args.method
    variant = "bootstrap" if method == "reimherr" else "likelihood_minus_m"
    row = rw_methods.run_one_task(
        "normal_unknown",
        method,
        variant,
        args.risk_metric,
        float(spec["prior_loc"]),
        int(rep),
        local,
    )
    return add_grid_metadata(row, spec, args, float(row["ecss"]))


def clarke_distance_setup(args):
    if (
        args.clarke_distance_engine == "direct"
        and args.direct_quadrature_rule == "gauss_legendre"
    ):
        nodes, weights = np.polynomial.legendre.leggauss(
            int(args.direct_quadrature_size)
        )
        if args.method == "clarke_tkl":
            lower = args.alpha / 2.0
            upper = 1.0 - args.alpha / 2.0
        else:
            lower = 0.0
            upper = 1.0
        u = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        weights = 0.5 * (upper - lower) * weights
        return u, weights

    if args.method == "clarke_tkl":
        u = np.linspace(
            args.alpha / 2.0,
            1.0 - args.alpha / 2.0,
            args.grid_size,
        )
    else:
        u = np.linspace(args.eps, 1.0 - args.eps, args.grid_size)
    return u, clarke_fast.trapz_weights(u)


def _student_standard_logpdf(x, df):
    return (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - 0.5 * (df + 1.0) * np.log1p(np.square(x) / df)
    )


def build_student_distance_lookup(
    df_post, df_prior, delta_bounds, log_rho_bounds, args
):
    delta_axis = np.linspace(
        float(delta_bounds[0]),
        float(delta_bounds[1]),
        int(args.interp_delta_grid_size),
    )
    log_rho_axis = np.linspace(
        float(log_rho_bounds[0]),
        float(log_rho_bounds[1]),
        int(args.interp_scale_grid_size),
    )
    nodes, quad_weights = np.polynomial.legendre.leggauss(
        int(args.interp_quadrature_size)
    )
    if args.method == "clarke_tkl":
        u_lo, u_hi = args.alpha / 2.0, 1.0 - args.alpha / 2.0
    else:
        u_lo, u_hi = args.eps, 1.0 - args.eps
    u = 0.5 * (u_hi - u_lo) * nodes + 0.5 * (u_hi + u_lo)
    weights = 0.5 * (u_hi - u_lo) * quad_weights
    mass = float(np.sum(weights))
    weights = weights / mass
    t_post = stats.t.ppf(u, df=float(df_post))
    e_logp_standard = float(
        np.sum(weights * _student_standard_logpdf(t_post, float(df_post)))
    )
    logq_const = float(_student_standard_logpdf(0.0, float(df_prior)))
    q_coef = 0.5 * (float(df_prior) + 1.0)
    table = np.empty((len(delta_axis), len(log_rho_axis)), dtype=np.float64)
    if args.method == "clarke_tkl":
        t_lower = stats.t.ppf(args.alpha / 2.0, df=float(df_post))
        t_upper = stats.t.ppf(1.0 - args.alpha / 2.0, df=float(df_post))

    for j, log_rho in enumerate(log_rho_axis):
        rho = float(np.exp(log_rho))
        standardized_q = delta_axis[:, None] + rho * t_post[None, :]
        e_logq = logq_const - q_coef * np.sum(
            weights[None, :]
            * np.log1p(np.square(standardized_q) / float(df_prior)),
            axis=1,
        )
        distance = e_logp_standard - log_rho - e_logq
        if args.method == "clarke_tkl":
            q_mass = (
                stats.t.cdf(delta_axis + rho * t_upper, df=float(df_prior))
                - stats.t.cdf(delta_axis + rho * t_lower, df=float(df_prior))
            )
            distance = distance + np.log(
                np.maximum(q_mass, 1e-300) / (1.0 - args.alpha)
            )
        table[:, j] = np.maximum(distance, 0.0)
    return {
        "delta_axis": delta_axis,
        "log_rho_axis": log_rho_axis,
        "distance": table,
    }


def clarke_lookup_domains(grid_df, args):
    ymin = np.inf
    ymax = -np.inf
    min_spacing = np.inf
    max_range = 0.0
    for rep in range(args.rep_start, args.rep_end):
        rng = np.random.default_rng(args.data_seed + int(rep))
        y = np.sort(rng.normal(args.true_mu, args.true_sigma, size=args.n))
        ymin = min(ymin, float(y[0]))
        ymax = max(ymax, float(y[-1]))
        spacing = np.diff(y)
        spacing = spacing[spacing > 0.0]
        if len(spacing):
            min_spacing = min(min_spacing, float(np.min(spacing)))
        max_range = max(max_range, float(y[-1] - y[0]))
    if not np.isfinite(min_spacing):
        min_spacing = 1e-10

    domains = {}
    for sigma2_mean, group in grid_df.groupby("sigma2_prior_mean", sort=False):
        kappa = float(group["kappa"].iloc[0])
        eta = float(group["eta"].iloc[0])
        prior_scale = np.sqrt(eta * args.tau0_sq / kappa)
        prior_locs = group["prior_loc"].to_numpy(dtype=float)
        for k in args.k_grid:
            k = int(k)
            delta_lo = (ymin - float(np.max(prior_locs))) / prior_scale
            delta_hi = (ymax - float(np.min(prior_locs))) / prior_scale
            delta_pad = max(0.5, args.interp_domain_pad * (delta_hi - delta_lo))
            rho_lo = max(min_spacing / (k * prior_scale), 1e-12)
            rho_hi = max(
                max_range / (2.0 * np.sqrt(max(k - 1.0, 1.0)) * prior_scale),
                rho_lo * 1.01,
            )
            log_lo = np.log(rho_lo) - args.interp_domain_pad
            log_hi = np.log(rho_hi) + args.interp_domain_pad
            domains[(k, float(sigma2_mean))] = (
                (delta_lo - delta_pad, delta_hi + delta_pad),
                (log_lo, log_hi),
            )
    return domains


def build_clarke_lookup_tables(grid_df, args):
    domains = clarke_lookup_domains(grid_df, args)
    tables = {}
    for sigma2_mean, group in grid_df.groupby("sigma2_prior_mean", sort=False):
        df_prior = 2.0 * float(group["kappa"].iloc[0])
        for k in args.k_grid:
            key = (int(k), float(sigma2_mean))
            delta_bounds, log_rho_bounds = domains[key]
            tables[key] = build_student_distance_lookup(
                df_post=float(k - 1),
                df_prior=df_prior,
                delta_bounds=delta_bounds,
                log_rho_bounds=log_rho_bounds,
                args=args,
            )
    return tables


def distance_matrix_from_lookup(loc, scale, prior_locs, prior_scale, lookup):
    delta = (
        np.asarray(loc, dtype=float)[:, None]
        - np.asarray(prior_locs, dtype=float)[None, :]
    ) / float(prior_scale)
    log_rho = np.log(np.asarray(scale, dtype=float) / float(prior_scale))
    log_rho = np.broadcast_to(log_rho[:, None], delta.shape)
    delta_axis = lookup["delta_axis"]
    log_rho_axis = lookup["log_rho_axis"]
    delta_index = (delta - delta_axis[0]) / (delta_axis[1] - delta_axis[0])
    rho_index = (log_rho - log_rho_axis[0]) / (
        log_rho_axis[1] - log_rho_axis[0]
    )
    values = ndimage.map_coordinates(
        lookup["distance"],
        [delta_index.ravel(), rho_index.ravel()],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return values.reshape(delta.shape)


def run_clarke_rep(rep, grid_df, args, lookup_tables=None):
    rng = np.random.default_rng(args.data_seed + int(rep))
    y = rng.normal(args.true_mu, args.true_sigma, size=args.n)
    u, weights = clarke_distance_setup(args)
    n_grid = len(grid_df)
    best_distance = np.full(n_grid, np.inf, dtype=float)
    best_k = np.full(n_grid, -1, dtype=int)
    groups = [
        (indices.to_numpy(dtype=int), group.copy())
        for _, group in grid_df.groupby("sigma2_prior_mean", sort=False)
        for indices in [group.index]
    ]

    for k in args.k_grid:
        k = int(k)
        n_candidates = clarke_fast.candidate_count(
            args.n, k, args.subset_mode
        )
        if n_candidates > args.max_candidates_per_k:
            raise RuntimeError(
                f"k={k} has {n_candidates} candidates, above "
                f"--max-candidates-per-k={args.max_candidates_per_k}."
            )
        k_best = np.full(n_grid, np.inf, dtype=float)
        for idx in clarke_fast.iter_index_chunks(
            args.n, k, args.subset_mode, args.chunk_size
        ):
            ybar, sse = clarke_fast.subset_sufficient_stats(y, idx)
            df_post, loc, scale, valid = clarke_fast.baseline_student_params(
                k,
                ybar,
                sse,
                baseline="jeffreys",
                kappa=3.0,
                eta=2.0,
            )
            if not np.any(valid):
                continue
            loc = loc[valid]
            scale = scale[valid]
            for group_index, group in groups:
                if args.clarke_distance_engine == "interpolation":
                    prior_scale = np.sqrt(
                        float(group["eta"].iloc[0])
                        * args.tau0_sq
                        / float(group["kappa"].iloc[0])
                    )
                    distance = distance_matrix_from_lookup(
                        loc,
                        scale,
                        group["prior_loc"].to_numpy(dtype=float),
                        prior_scale,
                        lookup_tables[(k, float(group["sigma2_prior_mean"].iloc[0]))],
                    )
                else:
                    local = copy.copy(args)
                    local.kappa = float(group["kappa"].iloc[0])
                    local.eta = float(group["eta"].iloc[0])
                    local.distance = (
                        "truncated_kl" if args.method == "clarke_tkl" else "kl"
                    )
                    distance = clarke_fast.distance_matrix(
                        loc,
                        scale,
                        df_post,
                        group["prior_loc"].to_numpy(dtype=float),
                        local,
                        u,
                        weights,
                    )
                chunk_min = np.min(distance, axis=0)
                improved = chunk_min < k_best[group_index]
                if np.any(improved):
                    k_best[group_index[improved]] = chunk_min[improved]
        improved = k_best < best_distance
        best_distance[improved] = k_best[improved]
        best_k[improved] = k

    rows = []
    for j, spec in grid_df.iterrows():
        row = {
            "model": "normal_unknown",
            "method": "clarke_exhaustive",
            "distance": "truncated_kl"
            if args.method == "clarke_tkl"
            else "kl",
            "baseline": "jeffreys",
            "rep": int(rep),
            "n": int(args.n),
            "sample_mean": float(np.mean(y)),
            "sample_var": float(np.var(y, ddof=1)),
            "d_min": float(best_distance[j]),
            "subset_mode": args.subset_mode,
        }
        rows.append(add_grid_metadata(row, spec, args, best_k[j]))
    return rows


def student_t_curvature_at(df, loc, scale_sq, theta):
    delta_sq = np.square(float(theta) - np.asarray(loc, dtype=float))
    core = float(df) * np.asarray(scale_sq, dtype=float)
    return (float(df) + 1.0) * (core - delta_sq) / np.square(
        core + delta_sq
    )


def expected_jeffreys_marginal_mu_curvature(
    m, kappa, eta, tau0_sq
):
    """Prior-predictive expected curvature of the Jeffreys posterior.

    Writing U=S/sigma^2 ~ chi-square(m-1), Z^2 ~ chi-square(1), and
    T=U/(U+Z^2), the expectation reduces to a one-dimensional beta
    integral.  The expectation is not finite at m=2.
    """
    m = int(m)
    if m <= 2:
        return np.nan

    alpha = 0.5 * (m - 1.0)
    beta = 0.5
    predictive_mean_variance = float(tau0_sq) + 1.0 / float(m)

    def beta_integrand(t):
        within = t / float(m)
        between = predictive_mean_variance * (1.0 - t)
        return (within - between) / np.square(within + between)

    angular_expectation = quad(
        beta_integrand,
        0.0,
        1.0,
        weight="alg",
        wvar=(alpha - 1.0, beta - 1.0),
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )[0] / beta_function(alpha, beta)

    return (
        float(m)
        * (float(kappa) / float(eta))
        * angular_expectation
        / float(m - 2)
    )


def run_morita(grid_df, args):
    rows = []
    m_grid = np.asarray(args.morita_m_grid, dtype=int)
    for _, group in grid_df.groupby("sigma2_prior_mean", sort=False):
        first = group.iloc[0]
        kappa = float(first["kappa"])
        eta = float(first["eta"])
        prior_curvature = (2.0 * kappa + 1.0) / (
            2.0 * eta * float(args.tau0_sq)
        )
        baseline_curvature = np.asarray(
            [
                expected_jeffreys_marginal_mu_curvature(
                    m=m,
                    kappa=kappa,
                    eta=eta,
                    tau0_sq=args.tau0_sq,
                )
                for m in m_grid
            ],
            dtype=float,
        )
        delta = np.abs(baseline_curvature - prior_curvature)
        if not np.any(np.isfinite(delta)):
            raise ValueError(
                "Morita marginal-mu curvature requires at least one m >= 3."
            )
        best_index = int(np.nanargmin(delta))
        ess = float(m_grid[best_index])
        for _, spec in group.iterrows():
            row = {
                "model": "normal_unknown",
                "method": "morita",
                "baseline": "jeffreys",
                "rep": -1,
                "n": int(args.n),
                "source": "prior_predictive_beta_quadrature_curvature",
                "prior_curvature": float(prior_curvature),
                "matched_baseline_curvature": float(
                    baseline_curvature[best_index]
                ),
                "min_abs_curvature_difference": float(delta[best_index]),
            }
            rows.append(add_grid_metadata(row, spec, args, ess))
    return rows


def summarize(df):
    group_cols = [
        "model",
        "method",
        "baseline",
        "target",
        "prior_loc",
        "sigma2_prior_mean",
        "sigma2_prior_var",
        "tau0_sq",
        "kappa",
        "eta",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)["ess"]
        .agg(
            mean_ess="mean",
            sd_ess="std",
            median_ess="median",
            n_rep="count",
        )
        .reset_index()
        .sort_values(["sigma2_prior_mean", "prior_loc"])
    )
    summary["sd_ess"] = summary["sd_ess"].fillna(0.0)
    return summary


def output_parts(args):
    if args.method == "jones":
        variant = "wasserstein"
        if is_exploratory_sampling(args.sampling_method):
            variant = f"{variant}/{args.sampling_method}"
        return "jones", variant
    if args.method == "cp":
        variant = "coverage"
        if is_exploratory_sampling(args.sampling_method):
            variant = f"{variant}/{args.sampling_method}"
        return "cp", variant
    if args.method == "reimherr":
        return "reimherr", f"bootstrap/{args.risk_metric}"
    if args.method == "wiesenfarth":
        return "wiesenfarth", f"likelihood_minus_m/{args.risk_metric}"
    if args.method == "clarke_kl":
        return "clarke_exhaustive", "kl"
    if args.method == "clarke_tkl":
        return "clarke_exhaustive", "truncated_kl"
    return "morita", "marginal_curvature"


def save_outputs(df, summary, grid_df, args, runtime_seconds):
    method_dir, variant_dir = output_parts(args)
    out_dir = Path(args.output_root) / "jeffreys" / method_dir / variant_dir
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
        "target": "marginal_mu",
        "method": args.method,
        "baseline": "jeffreys",
        "R": int(args.R),
        "rep_start": int(args.rep_start),
        "rep_end": int(args.rep_end),
        "n": int(args.n),
        "true_mu": float(args.true_mu),
        "true_sigma": float(args.true_sigma),
        "tau0_sq": float(args.tau0_sq),
        "sigma2_prior_var": float(args.sigma2_prior_var),
        "mu_grid": sorted(grid_df["prior_loc"].unique().tolist()),
        "sigma2_mean_grid": sorted(
            grid_df["sigma2_prior_mean"].unique().tolist()
        ),
        "n_grid": int(len(grid_df)),
        "data_seed": int(args.data_seed),
        "base_seed": int(args.base_seed),
        "sampling_method": (
            args.sampling_method if args.method in {"jones", "cp"} else None
        ),
        "m_grid_semantics": (
            "number_deleted"
            if is_deletion_sampling(args.sampling_method)
            else (
                "number_augmented"
                if is_empirical_augmentation(args.sampling_method)
                or args.method != "jones"
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
            if args.method in {"jones", "cp"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "positive_direction": (
            (
                "pi(.|Y_{n-m}) vs pi_b(.|X_n)"
                if is_deletion_sampling(args.sampling_method)
                else "pi(.|X_n) vs pi_b(.|X_n union A_m)"
            )
            if args.method in {"jones", "cp"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "outer_selection_rule": (
            "minimize_each_direction_then_compare_then_average_signed_m"
            if args.method in {"jones", "cp"}
            and is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "risk_metric": args.risk_metric
        if args.method in {"reimherr", "wiesenfarth"}
        else None,
        "B": int(args.B),
        "S": int(args.S),
        "B_fit": int(args.B_fit),
        "alpha": float(args.alpha),
        "jones_m_grid": args.jones_m_grid.tolist(),
        "jones_implementation": "vectorized_student_t_wasserstein2"
        if args.method == "jones"
        else None,
        "cp_m_grid": args.cp_m_grid.tolist(),
        "reimherr_m_grid": args.reimherr_m_grid.tolist(),
        "wiesenfarth_m_grid": args.wiesenfarth_m_grid.tolist(),
        "k_grid": args.k_grid.tolist(),
        "subset_mode": args.subset_mode,
        "clarke_search": "exhaustive_with_replacement"
        if args.method in {"clarke_kl", "clarke_tkl"}
        else None,
        "clarke_distance_engine": args.clarke_distance_engine
        if args.method in {"clarke_kl", "clarke_tkl"}
        else None,
        "direct_quadrature_rule": args.direct_quadrature_rule
        if args.method in {"clarke_kl", "clarke_tkl"}
        and args.clarke_distance_engine == "direct"
        else None,
        "direct_quadrature_size": int(args.direct_quadrature_size)
        if args.method in {"clarke_kl", "clarke_tkl"}
        and args.clarke_distance_engine == "direct"
        else None,
        "grid_size": int(args.grid_size),
        "interp_delta_grid_size": int(args.interp_delta_grid_size),
        "interp_scale_grid_size": int(args.interp_scale_grid_size),
        "interp_quadrature_size": int(args.interp_quadrature_size),
        "chunk_size": int(args.chunk_size),
        "morita_m_grid": args.morita_m_grid.tolist(),
        "morita_integration": "beta_quadrature",
        "morita_T": int(args.morita_T),
        "runtime": runtime,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2)
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Marginal-mu NIG EPSS heatmap over mu0 and E[sigma^2], "
            "at fixed Var(sigma^2)."
        )
    )
    parser.add_argument("--method", choices=sorted(METHOD_ALIASES), required=True)
    parser.add_argument(
        "--output-root",
        default="results/NIG_marginal_sigma2_mean_heatmap/sigma2var_2",
    )
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--prior-values", default=None)
    parser.add_argument("--sigma2-mean-values", default=None)
    parser.add_argument("--sigma2-prior-var", type=float, default=2.0)
    parser.add_argument("--max-grid-points", type=int, default=300)
    parser.add_argument("--true-mu", type=float, default=0.0)
    parser.add_argument("--true-sigma", type=float, default=1.0)
    parser.add_argument("--tau0-sq", type=float, default=0.2)
    parser.add_argument("--risk-metric", choices=["posterior_mse", "squared_mean"], default="posterior_mse")
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--S", type=int, default=500)
    parser.add_argument(
        "--sampling-method",
        choices=SAMPLING_METHODS,
        default="ppd_augmentation",
    )
    parser.add_argument("--min-retained", type=int, default=3)
    parser.add_argument("--B-fit", type=int, default=500)
    parser.add_argument("--mc-B", type=int, default=2000)
    parser.add_argument("--wasserstein-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--wasserstein-grid-size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--jones-m-grid", default=None)
    parser.add_argument("--cp-m-grid", default=None)
    parser.add_argument("--reimherr-m-grid", default=None)
    parser.add_argument("--wiesenfarth-m-grid", default=None)
    parser.add_argument("--k-grid", default=None)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--clarke-distance-engine",
        choices=["direct", "interpolation"],
        default="interpolation",
    )
    parser.add_argument(
        "--direct-quadrature-rule",
        choices=["gauss_legendre", "trapezoid"],
        default="gauss_legendre",
    )
    parser.add_argument("--direct-quadrature-size", type=int, default=128)
    parser.add_argument("--interp-delta-grid-size", type=int, default=401)
    parser.add_argument("--interp-scale-grid-size", type=int, default=161)
    parser.add_argument("--interp-quadrature-size", type=int, default=64)
    parser.add_argument("--interp-domain-pad", type=float, default=0.05)
    parser.add_argument("--subset-mode", choices=["with_replacement"], default="with_replacement")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--max-candidates-per-k", type=int, default=10_000_000)
    parser.add_argument("--morita-m-grid", default=None)
    parser.add_argument(
        "--morita-T",
        type=int,
        default=20000,
        help="Legacy compatibility option; deterministic beta quadrature is used.",
    )
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--base-seed", type=int, default=20260312)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--interpolate", action="store_true", default=True)
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    args = parser.parse_args()
    args.method = METHOD_ALIASES[args.method]
    args.baseline = "jeffreys"
    args.rep_end = args.R if args.rep_end is None else args.rep_end
    if is_deletion_sampling(args.sampling_method):
        default_deletion = range(0, args.n - args.min_retained + 1)
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, default_deletion
        )
        args.cp_m_grid = parse_int_grid(args.cp_m_grid, default_deletion)
        args.jones_m_grid = args.jones_m_grid[
            args.n - args.jones_m_grid >= args.min_retained
        ]
        args.cp_m_grid = args.cp_m_grid[
            args.n - args.cp_m_grid >= args.min_retained
        ]
    elif is_empirical_augmentation(args.sampling_method):
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(0, args.n + 1)
        )
        args.cp_m_grid = parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    else:
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(args.n, 2 * args.n + 1)
        )
        args.cp_m_grid = parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    args.reimherr_m_grid = parse_int_grid(args.reimherr_m_grid, range(-(args.n - 1), args.n + 1))
    reimherr_min_size = 4 if args.risk_metric == "posterior_mse" else 2
    args.reimherr_m_grid = args.reimherr_m_grid[
        args.n + args.reimherr_m_grid >= reimherr_min_size
    ]
    args.wiesenfarth_m_grid = parse_int_grid(args.wiesenfarth_m_grid, range(-args.n, args.n))
    args.wiesenfarth_m_grid = args.wiesenfarth_m_grid[args.n - args.wiesenfarth_m_grid >= 2]
    args.k_grid = parse_int_grid(args.k_grid, range(2, 10))
    args.k_grid = args.k_grid[(args.k_grid >= 2) & (args.k_grid <= args.n)]
    args.morita_m_grid = parse_int_grid(args.morita_m_grid, range(2, 21))

    # Attributes expected by the established marginal Jones/CP and R/W runners.
    args.alpha_grid = np.asarray([0.01, 0.05, 0.10, 0.20, 0.30])
    args.multi_alpha_label = "multi_alpha"
    args.fast_student_kl = True
    args.kl_grid_size = 400
    args.clarke_k_max = int(np.max(args.k_grid))
    args.clarke_k_grid = args.k_grid.copy()
    args.theta_star_mode = "baseline_posterior_mean"
    args.normal_unknown_true_mu = args.true_mu
    args.normal_unknown_true_sigma = args.true_sigma
    args.normal_unknown_tau0_sq = args.tau0_sq
    args.normal_sigma = 1.0
    args.normal_true_mu = 0.0
    args.normal_prior_sd = np.sqrt(0.2)
    args.poisson_true_lambda = 3.0
    args.poisson_prior_strength = 4.0
    args.beta_trials = 20
    args.beta_true_p = 0.3
    args.beta_prior_strength = 80.0
    return args


def main():
    args = parse_args()
    grid_df = make_prior_grid(args)
    start = time.perf_counter()
    if args.method in {"jones", "cp"}:
        rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_jones_cp_task)(rep, spec, args)
            for rep in range(args.rep_start, args.rep_end)
            for spec in grid_df.to_dict("records")
        )
    elif args.method in {"reimherr", "wiesenfarth"}:
        rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_rw_task)(rep, spec, args)
            for rep in range(args.rep_start, args.rep_end)
            for spec in grid_df.to_dict("records")
        )
    elif args.method in {"clarke_kl", "clarke_tkl"}:
        lookup_tables = (
            build_clarke_lookup_tables(grid_df, args)
            if args.clarke_distance_engine == "interpolation"
            else None
        )
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(run_clarke_rep)(rep, grid_df, args, lookup_tables)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    else:
        rows = run_morita(grid_df, args)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    out_dir = save_outputs(
        df,
        summary,
        grid_df,
        args,
        time.perf_counter() - start,
    )
    print(
        f"{args.method}: {len(df)} rows, {len(grid_df)} grid points -> {out_dir}"
    )


if __name__ == "__main__":
    main()
