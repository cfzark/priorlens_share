import argparse
import json
import sys
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special, stats


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "function", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_nig_marginal_sigma2_mean_heatmap as grid_tools  # noqa: E402
import run_normal_unknown_clarke_exhaustive_fast as clarke_fast  # noqa: E402
from prioreffect.marginal_sigma2 import (  # noqa: E402
    hdr_cross_coverage,
    inverse_gamma_coordinate_moments,
    inverse_gamma_kl,
    inverse_gamma_wasserstein2,
    standardized_hdr_interval,
    truncated_inverse_gamma_kl,
)
from prioreffect.sampling import (  # noqa: E402
    SAMPLING_METHODS,
    empirical_bootstrap_augmentation_matrix,
    empirical_deletion_matrix,
    is_deletion_sampling,
    is_empirical_augmentation,
    is_exploratory_sampling,
    validate_deletion_grid,
)


METHOD_ALIASES = {
    "jones": "jones",
    "cp": "cp",
    "reimherr": "reimherr",
    "wiesenfarth": "wiesenfarth",
    "morita": "morita",
    "clarke_kl": "clarke_kl",
    "clarke_tkl": "clarke_tkl",
    "clarke_truncated_kl": "clarke_tkl",
}


def stats_from_y(y):
    y = np.asarray(y, dtype=float)
    return len(y), float(np.sum(y)), float(np.sum(np.square(y)))


def posterior_sigma2_from_stats(n, sums, sums_sq, prior_type, spec, args):
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)
    if prior_type == "jeffreys":
        if np.any(n < 2) or np.any(sse <= 0.0):
            raise ValueError("Jeffreys posterior requires n >= 2 and positive SSE.")
        return 0.5 * (n - 1.0), 0.5 * sse
    if prior_type == "informative":
        lambda0 = 1.0 / float(args.tau0_sq)
        shape = float(spec["kappa"]) + 0.5 * n
        scale = float(spec["eta"]) + 0.5 * (
            sse
            + lambda0
            * n
            / (lambda0 + n)
            * np.square(ybar - float(spec["prior_loc"]))
        )
        return shape, scale
    raise ValueError("prior_type must be informative or jeffreys.")


def full_posterior_params(y, prior_type, spec, args):
    n, sums, sums_sq = stats_from_y(y)
    ybar = sums / n
    if prior_type == "jeffreys":
        shape, scale = posterior_sigma2_from_stats(
            n, sums, sums_sq, prior_type, spec, args
        )
        return {
            "mu": ybar,
            "lambda": float(n),
            "shape": float(shape),
            "scale": float(scale),
        }
    lambda0 = 1.0 / float(args.tau0_sq)
    shape, scale = posterior_sigma2_from_stats(
        n, sums, sums_sq, prior_type, spec, args
    )
    return {
        "mu": float((lambda0 * float(spec["prior_loc"]) + sums) / (lambda0 + n)),
        "lambda": float(lambda0 + n),
        "shape": float(shape),
        "scale": float(scale),
    }


def posterior_predictive_matrix(params, outer_reps, max_extra, rng):
    if max_extra <= 0:
        return np.empty((int(outer_reps), 0), dtype=float)
    df = 2.0 * float(params["shape"])
    scale = np.sqrt(
        float(params["scale"])
        / float(params["shape"])
        * (1.0 + 1.0 / float(params["lambda"]))
    )
    return stats.t.rvs(
        df=df,
        loc=float(params["mu"]),
        scale=scale,
        size=(int(outer_reps), int(max_extra)),
        random_state=rng,
    )


def augmented_path(y, future, extra_grid, prior_type, spec, args):
    y = np.asarray(y, dtype=float)
    future = np.asarray(future, dtype=float)
    extra_grid = np.asarray(extra_grid, dtype=int)
    outer_reps = future.shape[0]
    prefix_sum = np.concatenate(
        [np.zeros((outer_reps, 1)), np.cumsum(future, axis=1)], axis=1
    )
    prefix_sq = np.concatenate(
        [np.zeros((outer_reps, 1)), np.cumsum(np.square(future), axis=1)],
        axis=1,
    )
    n0, sum0, sq0 = stats_from_y(y)
    n_path = n0 + extra_grid[None, :]
    sums = sum0 + prefix_sum[:, extra_grid]
    sums_sq = sq0 + prefix_sq[:, extra_grid]
    return posterior_sigma2_from_stats(
        n_path, sums, sums_sq, prior_type, spec, args
    )


def grid_seed_offset(spec):
    loc_code = int(round((float(spec["prior_loc"]) + 50.0) * 10_000.0))
    scale_code = int(round(float(spec["sigma2_prior_mean"]) * 10_000.0))
    return 1009 * loc_code + 9176 * scale_code


def base_row(y, spec, rep, args, ess):
    return {
        "model": "normal_unknown",
        "target": "marginal_sigma2",
        "coordinate": args.coordinate,
        "method": args.method,
        "sampling_method": (
            args.sampling_method if args.method in {"jones", "cp"} else None
        ),
        "m_grid_semantics": (
            "number_deleted"
            if args.method in {"jones", "cp"}
            and is_deletion_sampling(args.sampling_method)
            else (
                "number_augmented"
                if args.method == "cp"
                or (
                    args.method == "jones"
                    and is_empirical_augmentation(args.sampling_method)
                )
                else (
                    "total_sample_size"
                    if args.method == "jones"
                    else None
                )
            )
        ),
        "baseline": "jeffreys",
        "rep": int(rep),
        "n": int(args.n),
        "grid_id": int(spec["grid_id"]),
        "prior_loc": float(spec["prior_loc"]),
        "prior_value": float(spec["prior_loc"]),
        "sigma2_prior_mean": float(spec["sigma2_prior_mean"]),
        "sigma2_prior_var": float(spec["sigma2_prior_var"]),
        "tau0_sq": float(args.tau0_sq),
        "kappa": float(spec["kappa"]),
        "eta": float(spec["eta"]),
        "sample_mean": float(np.mean(y)) if len(y) else np.nan,
        "sample_var": float(np.var(y, ddof=1)) if len(y) > 1 else np.nan,
        "ess": float(ess),
    }


def run_jones_task(rep, spec_dict, args):
    spec = pd.Series(spec_dict)
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    rng = np.random.default_rng(
        args.base_seed + 1_000_003 * int(rep) + grid_seed_offset(spec)
    )
    deletion = is_deletion_sampling(args.sampling_method)
    empirical_augmentation = is_empirical_augmentation(
        args.sampling_method
    )
    candidate_grid = np.asarray(args.jones_m_grid, dtype=int)
    extra_grid = (
        candidate_grid - args.n
        if not deletion and not empirical_augmentation
        else None
    )
    max_extra = (
        int(np.max(extra_grid))
        if not deletion and not empirical_augmentation
        else None
    )
    inf0 = full_posterior_params(y, "informative", spec, args)
    base0 = full_posterior_params(y, "jeffreys", spec, args)
    if deletion:
        retained_grid = args.n - candidate_grid
        sequences = empirical_deletion_matrix(
            y,
            args.outer_reps,
            rng,
            args.sampling_method,
            min_retained=int(np.min(retained_grid)),
            require_positive_variance=True,
        )
        prefix_sum = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(sequences, axis=1),
            ],
            axis=1,
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(np.square(sequences), axis=1),
            ],
            axis=1,
        )
        sums = prefix_sum[:, retained_grid]
        sums_sq = prefix_sq[:, retained_grid]
        n_path = retained_grid[None, :]
        shape_base_path, scale_base_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "jeffreys", spec, args
        )
        shape_inf_path, scale_inf_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "informative", spec, args
        )
    elif empirical_augmentation:
        additions = empirical_bootstrap_augmentation_matrix(
            y,
            args.outer_reps,
            int(np.max(candidate_grid)),
            rng,
        )
        prefix_sum = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(additions, axis=1),
            ],
            axis=1,
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(np.square(additions), axis=1),
            ],
            axis=1,
        )
        n0, sum0, sq0 = stats_from_y(y)
        n_path = n0 + candidate_grid[None, :]
        sums = sum0 + prefix_sum[:, candidate_grid]
        sums_sq = sq0 + prefix_sq[:, candidate_grid]
        shape_base_path, scale_base_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "jeffreys", spec, args
        )
        shape_inf_path, scale_inf_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "informative", spec, args
        )
    else:
        future_d = posterior_predictive_matrix(
            inf0, args.outer_reps, max_extra, rng
        )
        future_dsharp = posterior_predictive_matrix(
            inf0, args.outer_reps, max_extra, rng
        )
        shape_base_path, scale_base_path = augmented_path(
            y, future_d, extra_grid, "jeffreys", spec, args
        )
        shape_inf_path, scale_inf_path = augmented_path(
            y, future_dsharp, extra_grid, "informative", spec, args
        )
    d = np.empty_like(scale_base_path)
    dsharp = np.empty_like(scale_inf_path)
    for j in range(len(candidate_grid)):
        if deletion:
            d[:, j] = inverse_gamma_wasserstein2(
                inf0["shape"],
                inf0["scale"],
                float(np.asarray(shape_base_path)[0, j]),
                scale_base_path[:, j],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
            dsharp[:, j] = inverse_gamma_wasserstein2(
                float(np.asarray(shape_inf_path)[0, j]),
                scale_inf_path[:, j],
                base0["shape"],
                base0["scale"],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
        elif empirical_augmentation:
            d[:, j] = inverse_gamma_wasserstein2(
                base0["shape"],
                base0["scale"],
                float(np.asarray(shape_inf_path)[0, j]),
                scale_inf_path[:, j],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
            dsharp[:, j] = inverse_gamma_wasserstein2(
                inf0["shape"],
                inf0["scale"],
                float(np.asarray(shape_base_path)[0, j]),
                scale_base_path[:, j],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
        else:
            d[:, j] = inverse_gamma_wasserstein2(
                inf0["shape"],
                inf0["scale"],
                float(np.asarray(shape_base_path)[0, j]),
                scale_base_path[:, j],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
            dsharp[:, j] = inverse_gamma_wasserstein2(
                base0["shape"],
                base0["scale"],
                float(np.asarray(shape_inf_path)[0, j]),
                scale_inf_path[:, j],
                coordinate=args.coordinate,
                grid_size=args.distance_grid_size,
            )
    i = np.argmin(d, axis=1)
    j = np.argmin(dsharp, axis=1)
    row_index = np.arange(args.outer_reps)
    if deletion:
        signed = np.where(
            d[row_index, i] <= dsharp[row_index, j],
            -candidate_grid[i],
            candidate_grid[j],
        )
    elif empirical_augmentation:
        signed = np.where(
            d[row_index, i] <= dsharp[row_index, j],
            -candidate_grid[i],
            candidate_grid[j],
        )
    else:
        signed = np.where(
            d[row_index, i] <= dsharp[row_index, j],
            extra_grid[i],
            -extra_grid[j],
        )
    row = base_row(y, spec, rep, args, np.mean(signed))
    row["outer_sd_ess"] = float(np.std(signed, ddof=1))
    return row


def cp_objective(cp1, cp2, alpha):
    r1 = np.maximum(cp1, cp2)
    r2 = np.minimum(cp1, cp2)
    return np.square(r1 - r2) + np.square(r2 - (1.0 - alpha))


def run_cp_task(rep, spec_dict, args):
    spec = pd.Series(spec_dict)
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    rng = np.random.default_rng(
        args.base_seed + 1_000_003 * int(rep) + grid_seed_offset(spec) + 31
    )
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
    inf0 = full_posterior_params(y, "informative", spec, args)
    base0 = full_posterior_params(y, "jeffreys", spec, args)
    if deletion:
        retained_grid = args.n - extra_grid
        sequences = empirical_deletion_matrix(
            y,
            args.outer_reps,
            rng,
            args.sampling_method,
            min_retained=int(np.min(retained_grid)),
            require_positive_variance=True,
        )
        prefix_sum = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(sequences, axis=1),
            ],
            axis=1,
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(np.square(sequences), axis=1),
            ],
            axis=1,
        )
        sums = prefix_sum[:, retained_grid]
        sums_sq = prefix_sq[:, retained_grid]
        n_path = retained_grid[None, :]
        shape_inf_path, scale_inf_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "informative", spec, args
        )
        shape_base_path, scale_base_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "jeffreys", spec, args
        )
    elif empirical_augmentation:
        additions = empirical_bootstrap_augmentation_matrix(
            y,
            args.outer_reps,
            int(np.max(extra_grid)),
            rng,
        )
        prefix_sum = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(additions, axis=1),
            ],
            axis=1,
        )
        prefix_sq = np.concatenate(
            [
                np.zeros((args.outer_reps, 1)),
                np.cumsum(np.square(additions), axis=1),
            ],
            axis=1,
        )
        n0, sum0, sq0 = stats_from_y(y)
        n_path = n0 + extra_grid[None, :]
        sums = sum0 + prefix_sum[:, extra_grid]
        sums_sq = sq0 + prefix_sq[:, extra_grid]
        shape_inf_path, scale_inf_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "informative", spec, args
        )
        shape_base_path, scale_base_path = posterior_sigma2_from_stats(
            n_path, sums, sums_sq, "jeffreys", spec, args
        )
    else:
        future_base = posterior_predictive_matrix(
            base0, args.outer_reps, max_extra, rng
        )
        future_inf = posterior_predictive_matrix(
            inf0, args.outer_reps, max_extra, rng
        )
        shape_inf_path, scale_inf_path = augmented_path(
            y, future_base, extra_grid, "informative", spec, args
        )
        shape_base_path, scale_base_path = augmented_path(
            y, future_inf, extra_grid, "jeffreys", spec, args
        )
    obj_f = np.empty_like(scale_inf_path)
    obj_b = np.empty_like(scale_base_path)
    hdr_inf0 = standardized_hdr_interval(
        inf0["shape"], args.alpha, args.coordinate
    )
    hdr_base0 = standardized_hdr_interval(
        base0["shape"], args.alpha, args.coordinate
    )
    for j in range(len(extra_grid)):
        inf_shape = float(np.asarray(shape_inf_path)[0, j])
        base_shape = float(np.asarray(shape_base_path)[0, j])
        hdr_inf_path = standardized_hdr_interval(
            inf_shape, args.alpha, args.coordinate
        )
        hdr_base_path = standardized_hdr_interval(
            base_shape, args.alpha, args.coordinate
        )
        if deletion:
            cp1 = hdr_cross_coverage(
                inf0["shape"],
                inf0["scale"],
                base_shape,
                scale_base_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base_path,
            )
            cp2 = hdr_cross_coverage(
                base_shape,
                scale_base_path[:, j],
                inf0["shape"],
                inf0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf0,
            )
            obj_b[:, j] = cp_objective(cp1, cp2, args.alpha)

            cp1 = hdr_cross_coverage(
                inf_shape,
                scale_inf_path[:, j],
                base0["shape"],
                base0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base0,
            )
            cp2 = hdr_cross_coverage(
                base0["shape"],
                base0["scale"],
                inf_shape,
                scale_inf_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf_path,
            )
            obj_f[:, j] = cp_objective(cp1, cp2, args.alpha)
        elif empirical_augmentation:
            cp1 = hdr_cross_coverage(
                base0["shape"],
                base0["scale"],
                inf_shape,
                scale_inf_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf_path,
            )
            cp2 = hdr_cross_coverage(
                inf_shape,
                scale_inf_path[:, j],
                base0["shape"],
                base0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base0,
            )
            obj_b[:, j] = cp_objective(cp1, cp2, args.alpha)

            cp1 = hdr_cross_coverage(
                inf0["shape"],
                inf0["scale"],
                base_shape,
                scale_base_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base_path,
            )
            cp2 = hdr_cross_coverage(
                base_shape,
                scale_base_path[:, j],
                inf0["shape"],
                inf0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf0,
            )
            obj_f[:, j] = cp_objective(cp1, cp2, args.alpha)
        else:
            cp1 = hdr_cross_coverage(
                base0["shape"],
                base0["scale"],
                inf_shape,
                scale_inf_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf_path,
            )
            cp2 = hdr_cross_coverage(
                inf_shape,
                scale_inf_path[:, j],
                base0["shape"],
                base0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base0,
            )
            obj_f[:, j] = cp_objective(cp1, cp2, args.alpha)

            cp1 = hdr_cross_coverage(
                inf0["shape"],
                inf0["scale"],
                base_shape,
                scale_base_path[:, j],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_base_path,
            )
            cp2 = hdr_cross_coverage(
                base_shape,
                scale_base_path[:, j],
                inf0["shape"],
                inf0["scale"],
                args.alpha,
                args.coordinate,
                q_hdr=hdr_inf0,
            )
            obj_b[:, j] = cp_objective(cp1, cp2, args.alpha)

    row_index = np.arange(args.outer_reps)
    if deletion or empirical_augmentation:
        best_negative = np.argmin(obj_b, axis=1)
        best_positive = np.argmin(obj_f, axis=1)
        signed = np.where(
            obj_b[row_index, best_negative]
            <= obj_f[row_index, best_positive],
            -extra_grid[best_negative],
            extra_grid[best_positive],
        )
    else:
        selected = np.argmin(np.minimum(obj_f, obj_b), axis=1)
        sign = np.where(
            obj_f[row_index, selected] > obj_b[row_index, selected],
            1.0,
            -1.0,
        )
        signed = sign * extra_grid[selected]
    row = base_row(y, spec, rep, args, np.mean(signed))
    row["outer_sd_ess"] = float(np.std(signed, ddof=1))
    row["cp_region"] = f"exact_hdr_in_{args.coordinate}"
    return row


def bootstrap_stats(y, sample_size, B, rng, require_positive_sse=False):
    y = np.asarray(y, dtype=float)
    sample_size = int(sample_size)
    values = y[rng.integers(0, len(y), size=(int(B), sample_size))]
    sums = values.sum(axis=1)
    sums_sq = np.square(values).sum(axis=1)
    if require_positive_sse:
        sse = sums_sq - sums * sums / sample_size
        invalid = sse <= 1e-12
        while np.any(invalid):
            replacement = y[
                rng.integers(0, len(y), size=(int(np.sum(invalid)), sample_size))
            ]
            sums[invalid] = replacement.sum(axis=1)
            sums_sq[invalid] = np.square(replacement).sum(axis=1)
            sse = sums_sq - sums * sums / sample_size
            invalid = sse <= 1e-12
    return np.full(int(B), sample_size), sums, sums_sq


def normal_stats(mu, sigma2, sample_size, B, rng):
    values = rng.normal(
        float(mu), np.sqrt(float(sigma2)), size=(int(B), int(sample_size))
    )
    return (
        np.full(int(B), int(sample_size)),
        values.sum(axis=1),
        np.square(values).sum(axis=1),
    )


def coordinate_risk(shape, scale, target, coordinate, risk_metric):
    mean, var = inverse_gamma_coordinate_moments(shape, scale, coordinate)
    risk = np.square(mean - float(target))
    if risk_metric == "posterior_mse":
        risk = risk + var
    return risk


def risk_grid(n, sums, sums_sq, grid_df, prior_type, target, args):
    if prior_type == "jeffreys":
        dummy = grid_df.iloc[0]
        shape, scale = posterior_sigma2_from_stats(
            n, sums, sums_sq, prior_type, dummy, args
        )
        return float(
            np.mean(
                coordinate_risk(
                    shape, scale, target, args.coordinate, args.risk_metric
                )
            )
        )
    losses = np.empty(len(grid_df), dtype=float)
    for j, spec in grid_df.iterrows():
        shape, scale = posterior_sigma2_from_stats(
            n, sums, sums_sq, prior_type, spec, args
        )
        losses[j] = np.mean(
            coordinate_risk(
                shape, scale, target, args.coordinate, args.risk_metric
            )
        )
    return losses


def baseline_target(y, coordinate, args):
    n, sums, sums_sq = stats_from_y(y)
    dummy = {"prior_loc": 0.0, "kappa": 3.0, "eta": 2.0}
    shape, scale = posterior_sigma2_from_stats(
        n, sums, sums_sq, "jeffreys", dummy, args
    )
    return float(inverse_gamma_coordinate_moments(shape, scale, coordinate)[0])


def run_reimherr_rep(rep, grid_df, args):
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    target = baseline_target(y, args.coordinate, args)
    rng = np.random.default_rng(args.base_seed + 1_000_003 * int(rep) + 101)
    n_p, sum_p, sq_p = bootstrap_stats(y, args.n, args.B, rng)
    prior_risk = risk_grid(
        n_p, sum_p, sq_p, grid_df, "informative", target, args
    )
    baseline_path = []
    valid_m = []
    min_size = 6 if args.coordinate == "sigma2" else 2
    for m in args.reimherr_m_grid:
        sample_size = args.n + int(m)
        if sample_size < min_size:
            continue
        n_b, sum_b, sq_b = bootstrap_stats(
            y, sample_size, args.B, rng, require_positive_sse=True
        )
        baseline_path.append(
            risk_grid(n_b, sum_b, sq_b, grid_df, "jeffreys", target, args)
        )
        valid_m.append(int(m))
    baseline_path = np.asarray(baseline_path)
    rows = []
    for j, spec in grid_df.iterrows():
        index = int(np.argmin(np.abs(baseline_path - prior_risk[j])))
        rows.append(base_row(y, spec, rep, args, valid_m[index]))
    return rows


def run_wiesenfarth_rep(rep, grid_df, args):
    rng_data = np.random.default_rng(args.data_seed + int(rep))
    y = rng_data.normal(args.true_mu, args.true_sigma, size=args.n)
    target = baseline_target(y, args.coordinate, args)
    target_sigma2 = np.exp(target) if args.coordinate == "log_sigma2" else target
    target_mu = float(np.mean(y))
    rng = np.random.default_rng(args.base_seed + 1_000_003 * int(rep) + 211)
    n_b, sum_b, sq_b = normal_stats(
        target_mu, target_sigma2, args.n, args.B, rng
    )
    baseline_risk = risk_grid(
        n_b, sum_b, sq_b, grid_df, "jeffreys", target, args
    )
    path = np.empty((len(args.wiesenfarth_m_grid), len(grid_df)))
    for i, m in enumerate(args.wiesenfarth_m_grid):
        sample_size = args.n - int(m)
        n_p, sum_p, sq_p = normal_stats(
            target_mu, target_sigma2, sample_size, args.B, rng
        )
        path[i] = risk_grid(
            n_p, sum_p, sq_p, grid_df, "informative", target, args
        )
    rows = []
    for j, spec in grid_df.iterrows():
        index = int(np.argmin(np.abs(path[:, j] - baseline_risk)))
        rows.append(
            base_row(y, spec, rep, args, args.wiesenfarth_m_grid[index])
        )
    return rows


def run_clarke_rep(rep, grid_df, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    y = rng.normal(args.true_mu, args.true_sigma, size=args.n)
    unique = (
        grid_df.drop_duplicates("sigma2_prior_mean")
        .sort_values("sigma2_prior_mean")
        .reset_index(drop=True)
    )
    q_shape = unique["kappa"].to_numpy(dtype=float)
    q_scale = unique["eta"].to_numpy(dtype=float)
    best_distance = np.full(len(unique), np.inf)
    best_k = np.full(len(unique), -1, dtype=int)
    for k in args.k_grid:
        k = int(k)
        n_candidates = clarke_fast.candidate_count(
            args.n, k, "with_replacement"
        )
        if n_candidates > args.max_candidates_per_k:
            raise RuntimeError(
                f"k={k} has {n_candidates} candidates, above the configured limit."
            )
        k_best = np.full(len(unique), np.inf)
        shape_p = 0.5 * (k - 1.0)
        p_hdr = None
        if args.method == "clarke_tkl":
            p_hdr = standardized_hdr_interval(
                shape_p, args.alpha, args.coordinate
            )
        for idx in clarke_fast.iter_index_chunks(
            args.n, k, "with_replacement", args.chunk_size
        ):
            _, sse = clarke_fast.subset_sufficient_stats(y, idx)
            scale_p = 0.5 * sse
            valid = scale_p > 0.0
            if not np.any(valid):
                continue
            scale_p = scale_p[valid]
            if args.method == "clarke_kl":
                distance = inverse_gamma_kl(
                    shape_p,
                    scale_p[:, None],
                    q_shape[None, :],
                    q_scale[None, :],
                )
            else:
                distance = truncated_inverse_gamma_kl(
                    shape_p,
                    scale_p,
                    q_shape,
                    q_scale,
                    alpha=args.alpha,
                    coordinate=args.coordinate,
                    quadrature_size=args.truncated_quadrature_size,
                    p_hdr=p_hdr,
                )
            k_best = np.minimum(k_best, np.min(distance, axis=0))
        improved = k_best < best_distance
        best_distance[improved] = k_best[improved]
        best_k[improved] = k
    lookup = {
        float(row["sigma2_prior_mean"]): (best_k[i], best_distance[i])
        for i, row in unique.iterrows()
    }
    rows = []
    for _, spec in grid_df.iterrows():
        ess, distance = lookup[float(spec["sigma2_prior_mean"])]
        row = base_row(y, spec, rep, args, ess)
        row["d_min"] = float(distance)
        row["distance"] = "kl" if args.method == "clarke_kl" else "truncated_kl"
        row["subset_mode"] = "with_replacement"
        rows.append(row)
    return rows


def run_morita(grid_df, args):
    rows = []
    m_grid = np.asarray(args.morita_m_grid, dtype=int)
    for _, spec in grid_df.iterrows():
        kappa = float(spec["kappa"])
        mean_sigma2 = float(spec["sigma2_prior_mean"])
        if args.coordinate == "sigma2":
            unrestricted = 2.0 * kappa - 3.0
            prior_curvature = (kappa - 3.0) / np.square(mean_sigma2)
            baseline_curve = (m_grid - 3.0) / (
                2.0 * np.square(mean_sigma2)
            )
        else:
            unrestricted = 2.0 * kappa - 1.0
            prior_curvature = float(np.exp(special.digamma(kappa)))
            baseline_curve = (
                (m_grid - 1.0)
                * prior_curvature
                / (2.0 * (kappa - 1.0))
            )
        index = int(np.argmin(np.abs(baseline_curve - prior_curvature)))
        row = base_row(np.asarray([]), spec, -1, args, m_grid[index])
        row.update(
            {
                "source": "closed_form_prior_predictive_curvature",
                "ess_unrestricted": float(unrestricted),
                "prior_curvature": float(prior_curvature),
                "matched_baseline_curvature": float(baseline_curve[index]),
                "boundary_hit": bool(index in {0, len(m_grid) - 1}),
            }
        )
        rows.append(row)
    return rows


def summarize(df):
    columns = [
        "model",
        "method",
        "baseline",
        "target",
        "coordinate",
        "prior_loc",
        "sigma2_prior_mean",
        "sigma2_prior_var",
        "tau0_sq",
        "kappa",
        "eta",
    ]
    summary = (
        df.groupby(columns, dropna=False)["ess"]
        .agg(mean_ess="mean", sd_ess="std", median_ess="median", n_rep="count")
        .reset_index()
        .sort_values(["sigma2_prior_mean", "prior_loc"])
    )
    summary["sd_ess"] = summary["sd_ess"].fillna(0.0)
    return summary


def output_path(args):
    if args.method == "clarke_kl":
        return Path(args.output_root) / "jeffreys/clarke_exhaustive/kl/invariant"
    if args.method == "clarke_tkl":
        return Path(args.output_root) / f"jeffreys/clarke_exhaustive/truncated_kl/{args.coordinate}"
    if args.method == "reimherr":
        return Path(args.output_root) / f"jeffreys/reimherr/bootstrap/{args.risk_metric}/{args.coordinate}"
    if args.method == "wiesenfarth":
        return Path(args.output_root) / f"jeffreys/wiesenfarth/likelihood_minus_m/{args.risk_metric}/{args.coordinate}"
    if args.method == "jones":
        path = Path(args.output_root) / "jeffreys/jones/wasserstein"
        if is_exploratory_sampling(args.sampling_method):
            path = path / args.sampling_method
        return path / args.coordinate
    if args.method == "cp":
        path = Path(args.output_root) / "jeffreys/cp/hdr"
        if is_exploratory_sampling(args.sampling_method):
            path = path / args.sampling_method
        return path / args.coordinate
    return Path(args.output_root) / f"jeffreys/morita/curvature/{args.coordinate}"


def save_outputs(df, summary, grid_df, args, runtime_seconds):
    out_dir = output_path(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    grid_df.to_csv(out_dir / "prior_grid.csv", index=False)
    runtime = {
        "method": args.method,
        "sampling_method": (
            args.sampling_method if args.method in {"jones", "cp"} else None
        ),
        "m_grid_semantics": (
            "number_deleted"
            if args.method in {"jones", "cp"}
            and is_deletion_sampling(args.sampling_method)
            else (
                "number_augmented"
                if args.method == "cp"
                or (
                    args.method == "jones"
                    and is_empirical_augmentation(args.sampling_method)
                )
                else (
                    "total_sample_size"
                    if args.method == "jones"
                    else None
                )
            )
        ),
        "min_retained": (
            int(args.min_retained)
            if args.method in {"jones", "cp"}
            and is_deletion_sampling(args.sampling_method)
            else None
        ),
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
        "runtime_seconds": float(runtime_seconds),
        "runtime_minutes": float(runtime_seconds / 60.0),
        "n_rows": int(len(df)),
    }
    config = {
        "model": "normal_unknown",
        "target": "marginal_sigma2",
        "coordinate": "invariant" if args.method == "clarke_kl" else args.coordinate,
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
        "sigma2_mean_grid": sorted(grid_df["sigma2_prior_mean"].unique().tolist()),
        "risk_metric": args.risk_metric if args.method in {"reimherr", "wiesenfarth"} else None,
        "cp_region": f"exact_hdr_in_{args.coordinate}" if args.method == "cp" else None,
        "clarke_search": "exhaustive_with_replacement" if args.method.startswith("clarke") else None,
        "B": int(args.B),
        "outer_reps": int(args.outer_reps),
        "alpha": float(args.alpha),
        "jones_m_grid": args.jones_m_grid.tolist(),
        "cp_m_grid": args.cp_m_grid.tolist(),
        "reimherr_m_grid": args.reimherr_m_grid.tolist(),
        "wiesenfarth_m_grid": args.wiesenfarth_m_grid.tolist(),
        "k_grid": args.k_grid.tolist(),
        "morita_m_grid": args.morita_m_grid.tolist(),
        "data_seed": int(args.data_seed),
        "base_seed": int(args.base_seed),
        "runtime": runtime,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2)
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="NIG marginal-sigma2 EPSS heatmap.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_ALIASES))
    parser.add_argument("--coordinate", choices=["sigma2", "log_sigma2"], default="sigma2")
    parser.add_argument("--output-root", default="results/NIG_sigma2_marginal_heatmap/sigma2var_2")
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
    parser.add_argument("--outer-reps", type=int, default=500)
    parser.add_argument(
        "--sampling-method",
        choices=SAMPLING_METHODS,
        default="ppd_augmentation",
    )
    parser.add_argument("--min-retained", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--distance-grid-size", type=int, default=256)
    parser.add_argument("--truncated-quadrature-size", type=int, default=512)
    parser.add_argument("--jones-m-grid", default=None)
    parser.add_argument("--cp-m-grid", default=None)
    parser.add_argument("--reimherr-m-grid", default=None)
    parser.add_argument("--wiesenfarth-m-grid", default=None)
    parser.add_argument("--k-grid", default=None)
    parser.add_argument("--morita-m-grid", default=None)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--max-candidates-per-k", type=int, default=10_000_000)
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--base-seed", type=int, default=20260312)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)
    args = parser.parse_args()
    args.method = METHOD_ALIASES[args.method]
    args.rep_end = args.R if args.rep_end is None else args.rep_end
    if is_deletion_sampling(args.sampling_method):
        default_deletion_grid = range(
            0, args.n - int(args.min_retained) + 1
        )
        args.jones_m_grid = validate_deletion_grid(
            grid_tools.parse_int_grid(
                args.jones_m_grid, default_deletion_grid
            ),
            args.n,
            args.min_retained,
        )
        args.cp_m_grid = validate_deletion_grid(
            grid_tools.parse_int_grid(
                args.cp_m_grid, default_deletion_grid
            ),
            args.n,
            args.min_retained,
        )
    elif is_empirical_augmentation(args.sampling_method):
        args.jones_m_grid = grid_tools.parse_int_grid(
            args.jones_m_grid, range(0, args.n + 1)
        )
        args.cp_m_grid = grid_tools.parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    else:
        args.jones_m_grid = grid_tools.parse_int_grid(
            args.jones_m_grid, range(args.n, 2 * args.n + 1)
        )
        args.cp_m_grid = grid_tools.parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    args.reimherr_m_grid = grid_tools.parse_int_grid(args.reimherr_m_grid, range(-(args.n - 1), args.n + 1))
    args.wiesenfarth_m_grid = grid_tools.parse_int_grid(args.wiesenfarth_m_grid, range(-args.n, args.n))
    args.wiesenfarth_m_grid = args.wiesenfarth_m_grid[args.n - args.wiesenfarth_m_grid >= 1]
    args.k_grid = grid_tools.parse_int_grid(args.k_grid, range(2, 10))
    args.k_grid = args.k_grid[(args.k_grid >= 2) & (args.k_grid <= args.n)]
    args.morita_m_grid = grid_tools.parse_int_grid(args.morita_m_grid, range(2, 21))
    if args.method == "clarke_kl":
        args.coordinate = "sigma2"
    return args


def main():
    args = parse_args()
    grid_df = grid_tools.make_prior_grid(args)
    start = time.perf_counter()
    if args.method in {"jones", "cp"}:
        worker = run_jones_task if args.method == "jones" else run_cp_task
        rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(worker)(rep, spec, args)
            for rep in range(args.rep_start, args.rep_end)
            for spec in grid_df.to_dict("records")
        )
    elif args.method in {"reimherr", "wiesenfarth", "clarke_kl", "clarke_tkl"}:
        worker = {
            "reimherr": run_reimherr_rep,
            "wiesenfarth": run_wiesenfarth_rep,
            "clarke_kl": run_clarke_rep,
            "clarke_tkl": run_clarke_rep,
        }[args.method]
        outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
            delayed(worker)(rep, grid_df, args)
            for rep in range(args.rep_start, args.rep_end)
        )
        rows = list(chain.from_iterable(outputs))
    else:
        rows = run_morita(grid_df, args)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    out_dir = save_outputs(
        df, summary, grid_df, args, time.perf_counter() - start
    )
    print(f"{args.method}/{args.coordinate}: {len(df)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
