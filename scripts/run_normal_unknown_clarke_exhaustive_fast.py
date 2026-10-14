import argparse
import sys
from itertools import chain, combinations, combinations_with_replacement, islice
from math import comb, lgamma
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "function"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prioreffect import (  # noqa: E402
    add_runtime_metadata,
    default_normal_prior_loc_grid,
    save_experiment_outputs,
    start_timer,
)


def parse_int_grid(value, default):
    if value is None:
        return np.asarray(default, dtype=int)
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
    return np.asarray(sorted(set(pieces)), dtype=int)


def candidate_count(n, k, subset_mode):
    if subset_mode == "with_replacement":
        return comb(n + k - 1, k)
    if k > n:
        return 0
    return comb(n, k)


def iter_index_chunks(n, k, subset_mode, chunk_size):
    if subset_mode == "with_replacement":
        iterator = combinations_with_replacement(range(n), k)
    else:
        iterator = combinations(range(n), k)

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


def baseline_student_params(k, ybar, sse, baseline, kappa, eta):
    if baseline == "flat":
        shape = kappa + 0.5 * (k - 1)
        scale = eta + 0.5 * sse
        df = 2.0 * kappa + k - 1.0
        scale_sq = (scale / shape) / k
        valid = scale_sq > 0
        return df, ybar, np.sqrt(scale_sq), valid

    if baseline == "jeffreys":
        if k < 2:
            valid = np.zeros_like(ybar, dtype=bool)
            return float(k - 1), ybar, np.ones_like(ybar), valid
        df = float(k - 1)
        valid = sse > 0
        scale_sq = np.empty_like(ybar, dtype=float)
        scale_sq[valid] = (sse[valid] / (k - 1.0)) / k
        scale_sq[~valid] = 1.0
        return df, ybar, np.sqrt(scale_sq), valid

    raise ValueError("baseline must be flat or jeffreys.")


def trapz_weights(u):
    w = np.empty_like(u, dtype=float)
    if len(u) == 1:
        w[0] = 1.0
        return w
    dx = float(u[1] - u[0])
    w.fill(dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def w2_student_to_prior_vectorized(loc, scale, df_post, prior_locs, df_prior, prior_scale, u, w):
    a = stats.t.ppf(u, df=df_post)
    b = stats.t.ppf(u, df=df_prior)
    A1 = float(np.sum(w * a))
    B1 = float(np.sum(w * b))
    A2 = float(np.sum(w * a * a))
    B2 = float(np.sum(w * b * b))
    AB = float(np.sum(w * a * b))
    W0 = float(np.sum(w))

    delta = loc[:, None] - prior_locs[None, :]
    scale_col = scale[:, None]
    val_sq = (
        W0 * delta * delta
        + 2.0 * delta * (scale_col * A1 - prior_scale * B1)
        + scale_col * scale_col * A2
        - 2.0 * scale_col * prior_scale * AB
        + prior_scale * prior_scale * B2
    )
    return np.sqrt(np.maximum(val_sq, 0.0))


def w1_student_to_prior_batched(loc, scale, df_post, prior_locs, df_prior, prior_scale, u, w):
    a = stats.t.ppf(u, df=df_post)
    b = stats.t.ppf(u, df=df_prior)
    out = np.empty((len(loc), len(prior_locs)), dtype=float)
    for j, prior_loc in enumerate(prior_locs):
        diff = (
            loc[:, None]
            + scale[:, None] * a[None, :]
            - prior_loc
            - prior_scale * b[None, :]
        )
        out[:, j] = np.sum(w[None, :] * np.abs(diff), axis=1)
    return out


def kl_student_to_prior_batched(loc, scale, df_post, prior_locs, df_prior, prior_scale, u, w):
    a = stats.t.ppf(u, df=df_post)
    logp_std = student_t_standard_logpdf(a, df_post)
    e_logp_std = float(np.sum(w * logp_std))
    logq_const = student_t_standard_logpdf_const(df_prior) - np.log(prior_scale)
    q_coef = 0.5 * (df_prior + 1.0)
    out = np.empty((len(loc), len(prior_locs)), dtype=float)
    for j, prior_loc in enumerate(prior_locs):
        zq = (loc[:, None] + scale[:, None] * a[None, :] - prior_loc) / prior_scale
        e_logq = logq_const - q_coef * np.sum(w[None, :] * np.log1p((zq * zq) / df_prior), axis=1)
        e_logp = e_logp_std - np.log(scale)
        out[:, j] = np.maximum(e_logp - e_logq, 0.0)
    return out


def truncated_kl_student_to_prior_batched(loc, scale, df_post, prior_locs, df_prior, prior_scale, u, w, alpha):
    # u is already restricted to [alpha/2, 1-alpha/2], so the weighted
    # expectation approximates E_P[log p - log q | X in I_alpha(P)].
    a = stats.t.ppf(u, df=df_post)
    logp_std = student_t_standard_logpdf(a, df_post)
    e_logp_std = float(np.sum(w * logp_std) / max(1.0 - alpha, 1e-12))
    logq_const = student_t_standard_logpdf_const(df_prior) - np.log(prior_scale)
    q_coef = 0.5 * (df_prior + 1.0)
    out = np.empty((len(loc), len(prior_locs)), dtype=float)
    lower = loc + scale * stats.t.ppf(alpha / 2.0, df=df_post)
    upper = loc + scale * stats.t.ppf(1.0 - alpha / 2.0, df=df_post)
    for j, prior_loc in enumerate(prior_locs):
        zq = (loc[:, None] + scale[:, None] * a[None, :] - prior_loc) / prior_scale
        e_logq = logq_const - q_coef * (
            np.sum(w[None, :] * np.log1p((zq * zq) / df_prior), axis=1)
            / max(1.0 - alpha, 1e-12)
        )
        e_logp = e_logp_std - np.log(scale)
        q_interval = (
            stats.t.cdf((upper - prior_loc) / prior_scale, df=df_prior)
            - stats.t.cdf((lower - prior_loc) / prior_scale, df=df_prior)
        )
        q_interval = np.maximum(q_interval, 1e-12)
        out[:, j] = np.maximum(e_logp - e_logq + np.log(q_interval / max(1.0 - alpha, 1e-12)), 0.0)
    return out


def student_t_standard_logpdf_const(df):
    return float(lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - 0.5 * np.log(df * np.pi))


def student_t_standard_logpdf(x, df):
    return student_t_standard_logpdf_const(df) - 0.5 * (df + 1.0) * np.log1p((x * x) / df)


def distance_matrix(loc, scale, df_post, prior_locs, args, u, w):
    df_prior = 2.0 * args.kappa
    prior_scale = float(np.sqrt(args.eta * args.tau0_sq / args.kappa))

    if args.distance == "wasserstein":
        if args.wasserstein_order == 2:
            return w2_student_to_prior_vectorized(
                loc=loc,
                scale=scale,
                df_post=df_post,
                prior_locs=prior_locs,
                df_prior=df_prior,
                prior_scale=prior_scale,
                u=u,
                w=w,
            )
        return w1_student_to_prior_batched(
            loc=loc,
            scale=scale,
            df_post=df_post,
            prior_locs=prior_locs,
            df_prior=df_prior,
            prior_scale=prior_scale,
            u=u,
            w=w,
        )

    if args.distance == "kl":
        return kl_student_to_prior_batched(
            loc=loc,
            scale=scale,
            df_post=df_post,
            prior_locs=prior_locs,
            df_prior=df_prior,
            prior_scale=prior_scale,
            u=u,
            w=w,
        )

    if args.distance == "truncated_kl":
        return truncated_kl_student_to_prior_batched(
            loc=loc,
            scale=scale,
            df_post=df_post,
            prior_locs=prior_locs,
            df_prior=df_prior,
            prior_scale=prior_scale,
            u=u,
            w=w,
            alpha=args.alpha,
        )

    raise ValueError("distance must be wasserstein, kl, or truncated_kl.")


def exhaustive_rep(rep, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    y = rng.normal(loc=args.true_mu, scale=args.true_sigma, size=args.n)
    prior_locs = np.asarray(args.prior_values, dtype=float)
    n_prior = len(prior_locs)

    if args.distance == "truncated_kl":
        u = np.linspace(args.alpha / 2.0, 1.0 - args.alpha / 2.0, args.grid_size)
    else:
        u = np.linspace(args.eps, 1.0 - args.eps, args.grid_size)
    w = trapz_weights(u)

    best_d = np.full(n_prior, np.inf, dtype=float)
    best_k = np.full(n_prior, -1, dtype=int)
    best_ybar = np.full(n_prior, np.nan, dtype=float)
    best_sse = np.full(n_prior, np.nan, dtype=float)
    best_subset = [None for _ in range(n_prior)]
    path_rows = []

    for k in args.k_grid:
        n_candidates = candidate_count(args.n, int(k), args.subset_mode)
        if n_candidates <= 0:
            continue
        if n_candidates > args.max_candidates_per_k:
            raise RuntimeError(
                f"k={k} has {n_candidates} candidates under {args.subset_mode}, "
                f"above --max-candidates-per-k={args.max_candidates_per_k}. "
                "Use a smaller k-grid or increase the limit deliberately."
            )

        k_best_d = np.full(n_prior, np.inf, dtype=float)
        k_best_ybar = np.full(n_prior, np.nan, dtype=float)
        k_best_sse = np.full(n_prior, np.nan, dtype=float)
        k_best_subset = [None for _ in range(n_prior)]

        for idx in iter_index_chunks(args.n, int(k), args.subset_mode, args.chunk_size):
            ybar, sse = subset_sufficient_stats(y, idx)
            df_post, loc, scale, valid = baseline_student_params(
                int(k),
                ybar,
                sse,
                baseline=args.baseline,
                kappa=args.kappa,
                eta=args.eta,
            )
            if not np.any(valid):
                continue
            loc = loc[valid]
            scale = scale[valid]
            idx_valid = idx[valid]
            sse_valid = sse[valid]

            dist = distance_matrix(loc, scale, df_post, prior_locs, args, u, w)
            chunk_argmins = np.argmin(dist, axis=0)
            chunk_mins = dist[chunk_argmins, np.arange(n_prior)]
            improved = chunk_mins < k_best_d
            for j in np.flatnonzero(improved):
                row_i = int(chunk_argmins[j])
                k_best_d[j] = float(chunk_mins[j])
                k_best_ybar[j] = float(loc[row_i])
                k_best_sse[j] = float(sse_valid[row_i])
                k_best_subset[j] = idx_valid[row_i].astype(int).tolist()

        for j, prior_loc in enumerate(prior_locs):
            path_rows.append(
                {
                    "model": "normal_unknown",
                    "method": "clarke_exhaustive",
                    "baseline": args.baseline,
                    "distance": args.distance,
                    "prior_value": float(prior_loc),
                    "prior_loc": float(prior_loc),
                    "rep": int(rep),
                    "n": int(args.n),
                    "tau0_sq": float(args.tau0_sq),
                    "kappa": float(args.kappa),
                    "eta": float(args.eta),
                    "k": int(k),
                    "d_star": float(k_best_d[j]),
                    "n_candidates": int(n_candidates),
                    "best_ybar": float(k_best_ybar[j]),
                    "best_sse": float(k_best_sse[j]),
                    "best_subset_idx": k_best_subset[j],
                    "best_subset_values": None if k_best_subset[j] is None else y[k_best_subset[j]].tolist(),
                }
            )

        improved_global = k_best_d < best_d
        for j in np.flatnonzero(improved_global):
            best_d[j] = k_best_d[j]
            best_k[j] = int(k)
            best_ybar[j] = k_best_ybar[j]
            best_sse[j] = k_best_sse[j]
            best_subset[j] = k_best_subset[j]

    rows = []
    for j, prior_loc in enumerate(prior_locs):
        rows.append(
            {
                "model": "normal_unknown",
                "method": "clarke_exhaustive",
                "prior_loc": float(prior_loc),
                "prior_value": float(prior_loc),
                "rep": int(rep),
                "n": int(args.n),
                "sample_mean": float(np.mean(y)),
                "sample_var": float(np.var(y, ddof=1)) if len(y) > 1 else 0.0,
                "baseline": args.baseline,
                "tau0_sq": float(args.tau0_sq),
                "kappa": float(args.kappa),
                "eta": float(args.eta),
                "distance": args.distance,
                "ess": int(best_k[j]),
                "d_min": float(best_d[j]),
                "best_ybar": float(best_ybar[j]),
                "best_sse": float(best_sse[j]),
                "selected_subset_idx": best_subset[j],
                "selected_subset_values": None if best_subset[j] is None else y[best_subset[j]].tolist(),
                "subset_mode": args.subset_mode,
                "search": "all_combinations_with_replacement"
                if args.subset_mode == "with_replacement"
                else "all_combinations_without_replacement",
            }
        )

    return rows, path_rows


def summarize(df):
    return (
        df.groupby(["model", "method", "baseline", "prior_value"])["ess"]
        .agg(mean_ess="mean", median_ess="median", sd_ess="std", n_rep="count")
        .reset_index()
        .sort_values(["baseline", "prior_value"])
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast exhaustive Clarke search for Gaussian mean with unknown variance."
    )
    parser.add_argument("--output-root", default="results/normal_unknown")
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--prior-values",
        type=float,
        nargs="+",
        default=None,
        help="Prior location grid. Defaults to the same dense normal grid used by Jones/CP.",
    )
    parser.add_argument("--baseline", choices=["flat", "jeffreys"], default="flat")
    parser.add_argument("--true-mu", type=float, default=0.0)
    parser.add_argument("--true-sigma", type=float, default=1.0)
    parser.add_argument("--tau0-sq", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--distance", choices=["kl", "truncated_kl", "wasserstein"], default="kl")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--wasserstein-order", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Quantile grid size. KL uses an entropy-split deterministic quadrature.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--k-grid", default=None)
    parser.add_argument("--subset-mode", choices=["with_replacement", "without_replacement"], default="with_replacement")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--max-candidates-per-k", type=int, default=10_000_000)
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--save-json", action="store_true", help="Also save results.json and summary.json.")
    parser.add_argument("--save-pickle", action="store_true", help="Also save results.pkl and summary.pkl.")
    parser.add_argument("--save-clarke-path", action="store_true", help="Save full Clarke path over k.")
    args = parser.parse_args()
    args.rep_end = args.rep_end if args.rep_end is not None else args.R

    if args.k_grid is None:
        default = range(2, 10) if args.baseline == "jeffreys" else range(1, 10)
    else:
        default = None
    args.k_grid = parse_int_grid(args.k_grid, default)
    args.k_grid = args.k_grid[(args.k_grid >= 1) & (args.k_grid <= args.n)]
    if args.baseline == "jeffreys":
        args.k_grid = args.k_grid[args.k_grid >= 2]
    if len(args.k_grid) == 0:
        raise ValueError("k-grid has no valid values.")
    if args.prior_values is None:
        args.prior_values = default_normal_prior_loc_grid()
    else:
        args.prior_values = np.asarray(args.prior_values, dtype=float)
    return args


def main():
    args = parse_args()
    start = start_timer()
    reps = list(range(args.rep_start, args.rep_end))
    outputs = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(exhaustive_rep)(rep, args) for rep in reps
    )
    rows = list(chain.from_iterable(item[0] for item in outputs))
    path_rows = list(chain.from_iterable(item[1] for item in outputs))
    df = pd.DataFrame(rows)
    path_df = pd.DataFrame(path_rows)
    summary = summarize(df)
    config = {
        "model": "normal_unknown",
        "method": "clarke_exhaustive",
        "R": int(args.R),
        "rep_start": int(args.rep_start),
        "rep_end": int(args.rep_end),
        "n": int(args.n),
        "prior_grid": args.prior_values.tolist(),
        "baseline": args.baseline,
        "true_mu": float(args.true_mu),
        "true_sigma": float(args.true_sigma),
        "tau0_sq": float(args.tau0_sq),
        "kappa": float(args.kappa),
        "eta": float(args.eta),
        "distance": args.distance,
        "kl_implementation": "entropy_split_quantile_grid" if args.distance == "kl" else None,
        "truncated_kl_implementation": "central_interval_entropy_split_quantile_grid" if args.distance == "truncated_kl" else None,
        "alpha": float(args.alpha),
        "wasserstein_order": int(args.wasserstein_order),
        "grid_size": int(args.grid_size),
        "eps": float(args.eps),
        "k_grid": args.k_grid.tolist(),
        "subset_mode": args.subset_mode,
        "chunk_size": int(args.chunk_size),
        "max_candidates_per_k": int(args.max_candidates_per_k),
        "n_tasks": int(len(reps)),
        "n_rows": int(len(df)),
    }
    config, summary, runtime = add_runtime_metadata(
        df=df,
        summary=summary,
        config=config,
        start_time=start,
        variant=f"normal_unknown_clarke_exhaustive_{args.distance}",
    )
    out_dir = Path(args.output_root) / "clarke_exhaustive" / args.distance
    save_experiment_outputs(
        df=df,
        summary=summary,
        out_dir=out_dir,
        config=config,
        grid_filename="prior_grid.csv",
        grid_values=args.prior_values,
        save_json=args.save_json,
        save_pickle=args.save_pickle,
    )
    if args.save_clarke_path:
        path_df.to_csv(out_dir / "clarke_path.csv", index=False)
    print(
        f"normal_unknown/clarke_exhaustive/{args.distance}: "
        f"{len(df)} rows, {len(path_df)} path rows in {runtime['runtime_seconds']:.2f}s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
