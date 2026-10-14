import argparse
import sys
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
    FlatMuInverseGammaPrior,
    JeffreysPrior,
    KLDivergence,
    NormalInverseGammaDist,
    NormalMeanUnknownVarModel,
    StudentTDist,
    TruncatedKLDivergence,
    WassersteinDistance,
    add_runtime_metadata,
    default_normal_prior_loc_grid,
    save_experiment_outputs,
    start_timer,
)
from prioreffect.cp import (  # noqa: E402
    obj_func_exact,
    obj_func_exact_multi_alpha,
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


def build_distance(args):
    if args.distance == "kl":
        return KLDivergence(B=args.mc_B)
    if args.distance == "truncated_kl":
        return TruncatedKLDivergence(alpha=args.alpha, B=args.mc_B)
    if args.distance == "wasserstein":
        return WassersteinDistance(order=args.wasserstein_order, gamma_grid_size=args.wasserstein_grid_size)
    raise ValueError("distance must be kl, truncated_kl, or wasserstein.")


def fast_student_t_kl(P, Q, grid_size=400):
    if not isinstance(P, StudentTDist) or not isinstance(Q, StudentTDist):
        return None
    u = np.linspace(1.0 / (grid_size + 1.0), grid_size / (grid_size + 1.0), int(grid_size))
    x = stats.t.ppf(u, df=P.df, loc=P.loc, scale=P.scale)
    return float(max(0.0, np.mean(P.logpdf(x) - Q.logpdf(x))))


def compute_distance(P, Q, distance, args, rng):
    if args.distance == "kl" and args.fast_student_kl:
        val = fast_student_t_kl(P, Q, grid_size=args.kl_grid_size)
        if val is not None:
            return val
    return distance.compute(P, Q, rng=rng)


def build_model_priors(prior_loc, args):
    model = NormalMeanUnknownVarModel()
    prior_inf = NormalInverseGammaDist(
        mu0=float(prior_loc),
        tau0_sq=args.tau0_sq,
        kappa=args.kappa,
        eta=args.eta,
    )
    if args.baseline == "flat":
        prior_base = FlatMuInverseGammaPrior(kappa=args.kappa, eta=args.eta)
    elif args.baseline == "jeffreys":
        prior_base = JeffreysPrior()
    else:
        raise ValueError("baseline must be flat or jeffreys.")
    prior_inf_mu = model.posterior([], prior_inf)
    return model, prior_inf, prior_inf_mu, prior_base


def simulate_y(rep, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    return rng.normal(loc=args.true_mu, scale=args.true_sigma, size=args.n)


def posterior_predictive_sample(model, y, prior, size, rng):
    pred = model.posterior_predictive_dist(y, prior)
    return pred.sample(size=size, rng=rng)


def run_jones_dataset(y, model, prior_inf, prior_base, distance, args, rng):
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    deletion = is_deletion_sampling(args.sampling_method)
    empirical_augmentation = is_empirical_augmentation(
        args.sampling_method
    )
    m_grid = np.asarray(args.jones_m_grid, dtype=int)
    if deletion:
        m_grid = np.unique(
            m_grid[
                (m_grid >= 0)
                & (n - m_grid >= args.min_retained)
            ]
        )
    elif empirical_augmentation:
        m_grid = np.unique(m_grid[m_grid >= 0])
    else:
        m_grid = np.unique(m_grid[m_grid >= n])
    if len(m_grid) == 0:
        raise ValueError("Jones m_grid has no valid values.")

    q_n = model.posterior(y, prior_inf)
    q_n_b = model.posterior(y, prior_base)
    opess_draws = []
    d_path_mean = np.zeros(len(m_grid), dtype=float)
    dsharp_path_mean = np.zeros(len(m_grid), dtype=float)

    for _ in range(args.S):
        d_vals = np.zeros(len(m_grid), dtype=float)
        dsharp_vals = np.zeros(len(m_grid), dtype=float)
        if deletion:
            _, path = empirical_deletion_path(
                y,
                m_grid,
                rng,
                args.sampling_method,
                min_retained=args.min_retained,
                require_positive_variance=True,
            )
        elif empirical_augmentation:
            _, path = empirical_bootstrap_augmentation_path(
                y, m_grid, rng
            )
        for j, m in enumerate(m_grid):
            if deletion:
                reduced = path[int(m)]
                q_m_b = model.posterior(reduced, prior_base)
                q_m = model.posterior(reduced, prior_inf)
                d_vals[j] = compute_distance(
                    q_n, q_m_b, distance, args, rng
                )
                dsharp_vals[j] = compute_distance(
                    q_m, q_n_b, distance, args, rng
                )
            elif empirical_augmentation:
                additions = path[int(m)]
                augmented = (
                    y
                    if int(m) == 0
                    else np.concatenate([y, additions])
                )
                q_m_b = model.posterior(augmented, prior_base)
                q_m = model.posterior(augmented, prior_inf)
                d_vals[j] = compute_distance(
                    q_n_b, q_m, distance, args, rng
                )
                dsharp_vals[j] = compute_distance(
                    q_n, q_m_b, distance, args, rng
                )
            else:
                r = int(m - n)
                if r == 0:
                    q_m_b = q_n_b
                    q_m = q_n
                else:
                    future_b = posterior_predictive_sample(
                        model, y, prior_inf, r, rng
                    )
                    future_i = posterior_predictive_sample(
                        model, y, prior_inf, r, rng
                    )
                    q_m_b = model.posterior(
                        np.concatenate([y, future_b]), prior_base
                    )
                    q_m = model.posterior(
                        np.concatenate([y, future_i]), prior_inf
                    )
                d_vals[j] = compute_distance(
                    q_m_b, q_n, distance, args, rng
                )
                dsharp_vals[j] = compute_distance(
                    q_n_b, q_m, distance, args, rng
                )
            d_path_mean[j] += d_vals[j]
            dsharp_path_mean[j] += dsharp_vals[j]

        if deletion or empirical_augmentation:
            selected = select_signed_direction(
                d_vals, dsharp_vals, m_grid
            )
            opess_draws.append(selected["signed_m"])
        else:
            idx_d = int(np.nanargmin(d_vals))
            idx_dsharp = int(np.nanargmin(dsharp_vals))
            if d_vals[idx_d] <= dsharp_vals[idx_dsharp]:
                opess_draws.append(int(m_grid[idx_d] - n))
            else:
                opess_draws.append(int(n - m_grid[idx_dsharp]))

    opess_draws = np.asarray(opess_draws, dtype=float)
    d_path_mean /= args.S
    dsharp_path_mean /= args.S
    return {
        "mopess": float(np.mean(opess_draws)),
        "sd_opess": float(np.std(opess_draws, ddof=1)) if len(opess_draws) > 1 else 0.0,
        "median_opess": float(np.median(opess_draws)),
        "q05_opess": float(np.quantile(opess_draws, 0.05)),
        "q95_opess": float(np.quantile(opess_draws, 0.95)),
        "best_m_D": int(m_grid[int(np.argmin(d_path_mean))]),
        "best_m_Dsharp": int(m_grid[int(np.argmin(dsharp_path_mean))]),
        "opess_draws": opess_draws.tolist(),
        "mean_D_path": d_path_mean.tolist(),
        "mean_Dsharp_path": dsharp_path_mean.tolist(),
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
    }


def cp_pair_objective(post_a, post_b, args):
    if args.method == "multi_alpha_cp":
        out = obj_func_exact_multi_alpha(
            post_a,
            post_b,
            alpha_grid=args.alpha_grid,
        )
        return (
            out["mean_cp1"],
            out["mean_cp2"],
            out["mean_r1"],
            out["mean_r2"],
            out["obj"],
        )
    return obj_func_exact(post_a, post_b, alpha=args.alpha)


def run_cp_deletion_dataset(y, model, prior_inf, prior_base, args, rng):
    y = np.asarray(y, dtype=float).ravel()
    m_grid = np.asarray(args.cp_m_grid, dtype=int)
    m_grid = np.unique(
        m_grid[
            (m_grid >= 0)
            & (len(y) - m_grid >= args.min_retained)
        ]
    )
    if len(m_grid) == 0:
        raise ValueError("CP deletion grid has no valid values.")

    post_inf_full = model.posterior(y, prior_inf)
    post_base_full = model.posterior(y, prior_base)
    if args.method == "ppd_cp":
        target_negative = model.posterior_predictive_dist(y, prior_inf)
        target_positive = model.posterior_predictive_dist(y, prior_base)
    else:
        target_negative = post_inf_full
        target_positive = post_base_full

    signed = np.empty(args.B_fit, dtype=float)
    negative_best = np.empty(args.B_fit, dtype=float)
    positive_best = np.empty(args.B_fit, dtype=float)
    diag_negative = np.zeros((len(m_grid), 5), dtype=float)
    diag_positive = np.zeros((len(m_grid), 5), dtype=float)

    for b in range(args.B_fit):
        _, path = empirical_deletion_path(
            y,
            m_grid,
            rng,
            args.sampling_method,
            min_retained=args.min_retained,
            require_positive_variance=True,
        )
        obj_negative = np.empty(len(m_grid), dtype=float)
        obj_positive = np.empty(len(m_grid), dtype=float)
        for i, m in enumerate(m_grid):
            reduced = path[int(m)]
            if args.method == "ppd_cp":
                reduced_base = model.posterior_predictive_dist(
                    reduced, prior_base
                )
                reduced_inf = model.posterior_predictive_dist(
                    reduced, prior_inf
                )
            else:
                reduced_base = model.posterior(reduced, prior_base)
                reduced_inf = model.posterior(reduced, prior_inf)
            negative = cp_pair_objective(
                target_negative, reduced_base, args
            )
            positive = cp_pair_objective(
                reduced_inf, target_positive, args
            )
            obj_negative[i] = negative[4]
            obj_positive[i] = positive[4]
            diag_negative[i] += np.asarray(negative, dtype=float)
            diag_positive[i] += np.asarray(positive, dtype=float)

        selected = select_signed_direction(
            obj_negative, obj_positive, m_grid
        )
        signed[b] = selected["signed_m"]
        negative_best[b] = selected["negative_m"]
        positive_best[b] = selected["positive_m"]

    diag_rows = []
    names = ("cp1", "cp2", "r1", "r2", "obj")
    for i, m in enumerate(m_grid):
        row = {"m": int(m)}
        for j, name in enumerate(names):
            row[f"mean_{name}_forward"] = float(
                diag_negative[i, j] / args.B_fit
            )
            row[f"mean_{name}_backward"] = float(
                diag_positive[i, j] / args.B_fit
            )
        row["mean_min_obj"] = min(
            row["mean_obj_forward"], row["mean_obj_backward"]
        )
        row["mean_signal"] = (
            1.0
            if row["mean_obj_forward"] > row["mean_obj_backward"]
            else -1.0
        )
        diag_rows.append(row)

    return {
        "epss": float(np.mean(signed)),
        "mean_signed_mstar": float(np.mean(signed)),
        "sd_signed_mstar": (
            float(np.std(signed, ddof=1)) if args.B_fit > 1 else 0.0
        ),
        "mean_forward_mstar": float(np.mean(negative_best)),
        "sd_forward_mstar": (
            float(np.std(negative_best, ddof=1))
            if args.B_fit > 1
            else 0.0
        ),
        "mean_backward_mstar": float(np.mean(positive_best)),
        "sd_backward_mstar": (
            float(np.std(positive_best, ddof=1))
            if args.B_fit > 1
            else 0.0
        ),
        "signed_mstars": signed.tolist(),
        "forward_mstars": negative_best.tolist(),
        "backward_mstars": positive_best.tolist(),
        "cp_grid_diagnostics": diag_rows,
        "sampling_method": args.sampling_method,
        "m_grid_semantics": "number_deleted",
        "forward_direction": "negative",
        "backward_direction": "positive",
    }


def run_cp_bootstrap_augmentation_dataset(
    y, model, prior_inf, prior_base, args, rng
):
    y = np.asarray(y, dtype=float).ravel()
    m_grid = np.unique(
        np.asarray(args.cp_m_grid, dtype=int)
    )
    m_grid = m_grid[m_grid >= 0]
    if len(m_grid) == 0:
        raise ValueError("CP augmentation grid has no valid values.")

    post_inf_full = model.posterior(y, prior_inf)
    post_base_full = model.posterior(y, prior_base)
    if args.method == "ppd_cp":
        target_negative = model.posterior_predictive_dist(y, prior_base)
        target_positive = model.posterior_predictive_dist(y, prior_inf)
    else:
        target_negative = post_base_full
        target_positive = post_inf_full

    signed = np.empty(args.B_fit, dtype=float)
    negative_best = np.empty(args.B_fit, dtype=float)
    positive_best = np.empty(args.B_fit, dtype=float)
    diag_negative = np.zeros((len(m_grid), 5), dtype=float)
    diag_positive = np.zeros((len(m_grid), 5), dtype=float)

    for b in range(args.B_fit):
        _, path = empirical_bootstrap_augmentation_path(
            y, m_grid, rng
        )
        obj_negative = np.empty(len(m_grid), dtype=float)
        obj_positive = np.empty(len(m_grid), dtype=float)
        for i, m in enumerate(m_grid):
            additions = path[int(m)]
            augmented = (
                y
                if int(m) == 0
                else np.concatenate([y, additions])
            )
            if args.method == "ppd_cp":
                augmented_inf = model.posterior_predictive_dist(
                    augmented, prior_inf
                )
                augmented_base = model.posterior_predictive_dist(
                    augmented, prior_base
                )
            else:
                augmented_inf = model.posterior(augmented, prior_inf)
                augmented_base = model.posterior(augmented, prior_base)
            negative = cp_pair_objective(
                target_negative, augmented_inf, args
            )
            positive = cp_pair_objective(
                target_positive, augmented_base, args
            )
            obj_negative[i] = negative[4]
            obj_positive[i] = positive[4]
            diag_negative[i] += np.asarray(negative, dtype=float)
            diag_positive[i] += np.asarray(positive, dtype=float)

        selected = select_signed_direction(
            obj_negative, obj_positive, m_grid
        )
        signed[b] = selected["signed_m"]
        negative_best[b] = selected["negative_m"]
        positive_best[b] = selected["positive_m"]

    diag_rows = []
    names = ("cp1", "cp2", "r1", "r2", "obj")
    for i, m in enumerate(m_grid):
        row = {"m": int(m)}
        for j, name in enumerate(names):
            row[f"mean_{name}_forward"] = float(
                diag_negative[i, j] / args.B_fit
            )
            row[f"mean_{name}_backward"] = float(
                diag_positive[i, j] / args.B_fit
            )
        row["mean_min_obj"] = min(
            row["mean_obj_forward"], row["mean_obj_backward"]
        )
        row["mean_signal"] = (
            1.0
            if row["mean_obj_forward"] > row["mean_obj_backward"]
            else -1.0
        )
        diag_rows.append(row)

    return {
        "epss": float(np.mean(signed)),
        "mean_signed_mstar": float(np.mean(signed)),
        "sd_signed_mstar": (
            float(np.std(signed, ddof=1)) if args.B_fit > 1 else 0.0
        ),
        "mean_forward_mstar": float(np.mean(negative_best)),
        "sd_forward_mstar": (
            float(np.std(negative_best, ddof=1))
            if args.B_fit > 1
            else 0.0
        ),
        "mean_backward_mstar": float(np.mean(positive_best)),
        "sd_backward_mstar": (
            float(np.std(positive_best, ddof=1))
            if args.B_fit > 1
            else 0.0
        ),
        "signed_mstars": signed.tolist(),
        "forward_mstars": negative_best.tolist(),
        "backward_mstars": positive_best.tolist(),
        "cp_grid_diagnostics": diag_rows,
        "sampling_method": args.sampling_method,
        "m_grid_semantics": "number_augmented",
        "forward_direction": "negative",
        "backward_direction": "positive",
    }


def run_cp_dataset(y, model, prior_inf, prior_base, args, rng):
    if is_deletion_sampling(args.sampling_method):
        return run_cp_deletion_dataset(
            y, model, prior_inf, prior_base, args, rng
        )
    if is_empirical_augmentation(args.sampling_method):
        return run_cp_bootstrap_augmentation_dataset(
            y, model, prior_inf, prior_base, args, rng
        )
    y = np.asarray(y, dtype=float).ravel()
    m_grid = np.asarray(args.cp_m_grid, dtype=int)
    m_grid = np.unique(m_grid[m_grid >= 0])
    if len(m_grid) == 0:
        raise ValueError("CP m_grid must contain nonnegative values.")

    post_base0 = model.posterior(y, prior_base)
    post_inf0 = model.posterior(y, prior_inf)
    ppd_base0 = model.posterior_predictive_dist(y, prior_base)
    ppd_inf0 = model.posterior_predictive_dist(y, prior_inf)
    M = int(np.max(m_grid))
    signed_mstars = np.empty(args.B_fit, dtype=float)
    forward_mstars = np.empty(args.B_fit, dtype=float)
    backward_mstars = np.empty(args.B_fit, dtype=float)
    diag = {
        "cp1_forward": np.zeros(len(m_grid), dtype=float),
        "cp2_forward": np.zeros(len(m_grid), dtype=float),
        "r1_forward": np.zeros(len(m_grid), dtype=float),
        "r2_forward": np.zeros(len(m_grid), dtype=float),
        "obj_forward": np.zeros(len(m_grid), dtype=float),
        "cp1_backward": np.zeros(len(m_grid), dtype=float),
        "cp2_backward": np.zeros(len(m_grid), dtype=float),
        "r1_backward": np.zeros(len(m_grid), dtype=float),
        "r2_backward": np.zeros(len(m_grid), dtype=float),
        "obj_backward": np.zeros(len(m_grid), dtype=float),
        "min_obj": np.zeros(len(m_grid), dtype=float),
        "signal": np.zeros(len(m_grid), dtype=float),
    }

    for b in range(args.B_fit):
        y_new_base = posterior_predictive_sample(model, y, prior_base, M, rng) if M > 0 else np.array([])
        y_new_inf = posterior_predictive_sample(model, y, prior_inf, M, rng) if M > 0 else np.array([])
        obj1 = np.zeros(len(m_grid), dtype=float)
        obj2 = np.zeros(len(m_grid), dtype=float)

        for i, m in enumerate(m_grid):
            y_aug_1 = y if m == 0 else np.concatenate([y, y_new_base[:m]])
            post_inf_m = model.posterior(y_aug_1, prior_inf)
            if args.method == "ppd_cp":
                ppd_inf_m = model.posterior_predictive_dist(y_aug_1, prior_inf)
                cp1_f, cp2_f, r1_f, r2_f, obj1[i] = obj_func_exact(ppd_base0, ppd_inf_m, alpha=args.alpha)
            elif args.method == "multi_alpha_cp":
                out_f = obj_func_exact_multi_alpha(
                    post_base0,
                    post_inf_m,
                    alpha_grid=args.alpha_grid,
                )
                cp1_f, cp2_f, r1_f, r2_f, obj1[i] = (
                    out_f["mean_cp1"],
                    out_f["mean_cp2"],
                    out_f["mean_r1"],
                    out_f["mean_r2"],
                    out_f["obj"],
                )
            else:
                cp1_f, cp2_f, r1_f, r2_f, obj1[i] = obj_func_exact(
                    post_base0,
                    post_inf_m,
                    alpha=args.alpha,
                )

            y_aug_2 = y if m == 0 else np.concatenate([y, y_new_inf[:m]])
            post_base_m = model.posterior(y_aug_2, prior_base)
            if args.method == "ppd_cp":
                ppd_base_m = model.posterior_predictive_dist(y_aug_2, prior_base)
                cp1_b, cp2_b, r1_b, r2_b, obj2[i] = obj_func_exact(ppd_inf0, ppd_base_m, alpha=args.alpha)
            elif args.method == "multi_alpha_cp":
                out_b = obj_func_exact_multi_alpha(
                    post_inf0,
                    post_base_m,
                    alpha_grid=args.alpha_grid,
                )
                cp1_b, cp2_b, r1_b, r2_b, obj2[i] = (
                    out_b["mean_cp1"],
                    out_b["mean_cp2"],
                    out_b["mean_r1"],
                    out_b["mean_r2"],
                    out_b["obj"],
                )
            else:
                cp1_b, cp2_b, r1_b, r2_b, obj2[i] = obj_func_exact(
                    post_inf0,
                    post_base_m,
                    alpha=args.alpha,
                )
            signal = 1.0 if obj1[i] > obj2[i] else -1.0
            diag["cp1_forward"][i] += cp1_f
            diag["cp2_forward"][i] += cp2_f
            diag["r1_forward"][i] += r1_f
            diag["r2_forward"][i] += r2_f
            diag["obj_forward"][i] += obj1[i]
            diag["cp1_backward"][i] += cp1_b
            diag["cp2_backward"][i] += cp2_b
            diag["r1_backward"][i] += r1_b
            diag["r2_backward"][i] += r2_b
            diag["obj_backward"][i] += obj2[i]
            diag["min_obj"][i] += min(obj1[i], obj2[i])
            diag["signal"][i] += signal

        m_star_idx = int(np.argmin(np.minimum(obj1, obj2)))
        forward_mstars[b] = int(m_grid[int(np.argmin(obj1))])
        backward_mstars[b] = int(m_grid[int(np.argmin(obj2))])
        sign = 1 if obj1[m_star_idx] > obj2[m_star_idx] else -1
        signed_mstars[b] = int(m_grid[m_star_idx]) * sign

    diag_rows = []
    for i, m in enumerate(m_grid):
        row = {"m": int(m)}
        for key, value in diag.items():
            row[f"mean_{key}"] = float(value[i] / args.B_fit)
        diag_rows.append(row)

    return {
        "epss": float(np.mean(signed_mstars)),
        "mean_signed_mstar": float(np.mean(signed_mstars)),
        "sd_signed_mstar": float(np.std(signed_mstars, ddof=1)) if args.B_fit > 1 else 0.0,
        "mean_forward_mstar": float(np.mean(forward_mstars)),
        "sd_forward_mstar": float(np.std(forward_mstars, ddof=1)) if args.B_fit > 1 else 0.0,
        "mean_backward_mstar": float(np.mean(backward_mstars)),
        "sd_backward_mstar": float(np.std(backward_mstars, ddof=1)) if args.B_fit > 1 else 0.0,
        "signed_mstars": signed_mstars.tolist(),
        "forward_mstars": forward_mstars.tolist(),
        "backward_mstars": backward_mstars.tolist(),
        "cp_grid_diagnostics": diag_rows,
        "sampling_method": args.sampling_method,
        "m_grid_semantics": "number_augmented",
    }


def run_clarke_dataset(y, model, prior_inf_mu, prior_base, distance, args, rng):
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    k_grid = np.asarray(args.clarke_k_grid, dtype=int)
    k_grid = np.unique(k_grid[(k_grid >= 1) & (k_grid <= args.clarke_k_max)])
    if len(k_grid) == 0:
        raise ValueError("Clarke k_grid has no valid positive values.")

    d_star = np.full(len(k_grid), np.inf, dtype=float)
    best_subset_values = []
    for i, k in enumerate(k_grid):
        n_candidates = args.clarke_candidates
        best_vals = None
        for _ in range(n_candidates):
            idx = rng.integers(0, n, size=int(k))
            y_subset = y[idx]
            try:
                post_base = model.posterior(y_subset, prior_base)
                d_val = compute_distance(post_base, prior_inf_mu, distance, args, rng)
            except Exception:
                continue
            if np.isfinite(d_val) and d_val < d_star[i]:
                d_star[i] = float(d_val)
                best_vals = y_subset.tolist()
        best_subset_values.append(best_vals)

    if not np.any(np.isfinite(d_star)):
        raise RuntimeError("No finite Clarke distances. Check baseline/k_grid.")
    k_hat = int(k_grid[int(np.nanargmin(d_star))])
    return {
        "ess": float(k_hat),
        "k_hat": k_hat,
        "k_grid": k_grid.tolist(),
        "d_star": d_star.tolist(),
        "best_subset_values": best_subset_values,
        "clarke_search": "bootstrap_candidates_not_all_subsets",
        "clarke_candidates_per_k": int(args.clarke_candidates),
    }


def run_one(prior_loc, rep, args):
    model, prior_inf, prior_inf_mu, prior_base = build_model_priors(prior_loc, args)
    y = simulate_y(rep, args)
    rng = np.random.default_rng(
        args.base_seed + 1000000 * int(rep) + int(round((float(prior_loc) + 10.0) * 10000))
    )
    distance = build_distance(args)

    base = {
        "model": "normal_unknown",
        "method": args.method,
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
    }

    if args.method == "jones":
        out = run_jones_dataset(y, model, prior_inf, prior_base, distance, args, rng)
        return {**base, **out}
    if args.method == "cp":
        out = run_cp_dataset(y, model, prior_inf, prior_base, args, rng)
        return {**base, **out}
    if args.method == "multi_alpha_cp":
        out = run_cp_dataset(y, model, prior_inf, prior_base, args, rng)
        return {**base, **out}
    if args.method == "ppd_cp":
        out = run_cp_dataset(y, model, prior_inf, prior_base, args, rng)
        return {**base, **out}
    if args.method == "clarke":
        out = run_clarke_dataset(y, model, prior_inf_mu, prior_base, distance, args, rng)
        return {**base, **out}
    raise ValueError("method must be jones, cp, or clarke.")


def summarize(df, method):
    value_col = {"jones": "mopess", "cp": "epss", "multi_alpha_cp": "epss", "ppd_cp": "epss", "clarke": "ess"}[method]
    return (
        df.groupby(["model", "method", "prior_value"])[value_col]
        .agg(mean="mean", median="median", sd="std", n_rep="count")
        .reset_index()
        .rename(columns={"mean": f"mean_{value_col}", "median": f"median_{value_col}", "sd": f"sd_{value_col}"})
        .sort_values("prior_value")
    )


def parse_int_grid(value, default):
    if value is None:
        return np.asarray(default, dtype=int)
    pieces = []
    for part in value.split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":", 1)
            pieces.extend(range(int(a), int(b) + 1))
        elif part:
            pieces.append(int(part))
    return np.asarray(sorted(set(pieces)), dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Jones/CP/Fast-Clarke for Gaussian mean unknown variance.")
    parser.add_argument("--method", choices=["jones", "cp", "multi_alpha_cp", "ppd_cp", "clarke"], required=True)
    parser.add_argument("--output-root", default="results/normal_unknown")
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--prior-values", type=float, nargs="+", default=None)
    parser.add_argument("--baseline", choices=["flat", "jeffreys"], default="flat")
    parser.add_argument("--true-mu", type=float, default=0.0)
    parser.add_argument("--true-sigma", type=float, default=1.0)
    parser.add_argument("--tau0-sq", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--distance", choices=["kl", "truncated_kl", "wasserstein"], default="wasserstein")
    parser.add_argument("--wasserstein-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--wasserstein-grid-size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--multi-alpha-label", default="multi_alpha")
    parser.add_argument("--mc-B", type=int, default=2000)
    parser.add_argument("--fast-student-kl", action="store_true", default=True)
    parser.add_argument("--no-fast-student-kl", dest="fast_student_kl", action="store_false")
    parser.add_argument("--kl-grid-size", type=int, default=400)
    parser.add_argument("--S", type=int, default=500, help="Jones Monte Carlo repetitions.")
    parser.add_argument("--B-fit", type=int, default=500, help="CP Monte Carlo repetitions.")
    parser.add_argument(
        "--sampling-method",
        choices=SAMPLING_METHODS,
        default="ppd_augmentation",
    )
    parser.add_argument("--min-retained", type=int, default=3)
    parser.add_argument("--clarke-candidates", type=int, default=250)
    parser.add_argument("--clarke-k-max", type=int, default=20)
    parser.add_argument("--jones-m-grid", default=None, help="Absolute m grid, e.g. 20:40")
    parser.add_argument("--cp-m-grid", default=None, help="Nonnegative m grid, e.g. 0:20")
    parser.add_argument("--clarke-k-grid", default=None, help="Positive k grid, e.g. 1:20")
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--base-seed", type=int, default=20260312)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--save-json", action="store_true", help="Also save results.json and summary.json.")
    parser.add_argument("--save-pickle", action="store_true", help="Also save results.pkl and summary.pkl.")
    parser.add_argument("--save-cp-diagnostics", action="store_true", help="Save full CP m-grid diagnostics.")
    args = parser.parse_args()
    if args.method == "clarke" and "--distance" not in sys.argv:
        args.distance = "kl"
    args.rep_end = args.rep_end if args.rep_end is not None else args.R
    args.alpha_grid = np.asarray(args.alpha_grid, dtype=float)
    args.multi_alpha_label = str(args.multi_alpha_label).strip() or "multi_alpha"
    if args.prior_values is None:
        args.prior_values = default_normal_prior_loc_grid()
    elif is_empirical_augmentation(args.sampling_method):
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(0, args.n + 1)
        )
        args.cp_m_grid = parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    else:
        args.prior_values = np.asarray(args.prior_values, dtype=float)
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
    else:
        args.jones_m_grid = parse_int_grid(
            args.jones_m_grid, range(args.n, 2 * args.n + 1)
        )
        args.cp_m_grid = parse_int_grid(
            args.cp_m_grid, range(0, args.n + 1)
        )
    args.clarke_k_grid = parse_int_grid(args.clarke_k_grid, range(1, args.n + 1))
    return args


def main():
    args = parse_args()
    tasks = [
        (float(prior_loc), rep, args)
        for rep in range(args.rep_start, args.rep_end)
        for prior_loc in np.asarray(args.prior_values, dtype=float)
    ]
    start = start_timer()
    rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(run_one)(*task) for task in tasks
    )
    df = pd.DataFrame(rows)
    cp_grid_diagnostics = None
    if args.method in {"cp", "multi_alpha_cp", "ppd_cp"} and "cp_grid_diagnostics" in df.columns:
        diag_rows = []
        for _, row in df.iterrows():
            for diag in row["cp_grid_diagnostics"]:
                diag_rows.append(
                    {
                        "model": row["model"],
                        "method": row["method"],
                        "baseline": row["baseline"],
                        "distance": row["distance"],
                        "prior_value": row["prior_value"],
                        "prior_loc": row["prior_loc"],
                        "rep": row["rep"],
                        "n": row["n"],
                        "tau0_sq": row["tau0_sq"],
                        "kappa": row["kappa"],
                        "eta": row["eta"],
                        **diag,
                    }
                )
        cp_grid_diagnostics = pd.DataFrame(diag_rows)
        df = df.drop(columns=["cp_grid_diagnostics"])
    summary = summarize(df, args.method)
    config = {
        "model": "normal_unknown",
        "method": args.method,
        "R": int(args.R),
        "rep_start": int(args.rep_start),
        "rep_end": int(args.rep_end),
        "n": int(args.n),
        "prior_grid": np.asarray(args.prior_values, dtype=float).tolist(),
        "baseline": args.baseline,
        "true_mu": float(args.true_mu),
        "true_sigma": float(args.true_sigma),
        "tau0_sq": float(args.tau0_sq),
        "kappa": float(args.kappa),
        "eta": float(args.eta),
        "distance": args.distance,
        "variant": args.multi_alpha_label if args.method == "multi_alpha_cp" else ("predictive" if args.method == "ppd_cp" else args.distance),
        "sampling_method": args.sampling_method,
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
            if is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "positive_direction": (
            (
                "pi(.|Y_{n-m}) vs pi_b(.|X_n)"
                if is_deletion_sampling(args.sampling_method)
                else "pi(.|X_n) vs pi_b(.|X_n union A_m)"
            )
            if is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "outer_selection_rule": (
            "minimize_each_direction_then_compare_then_average_signed_m"
            if is_exploratory_sampling(args.sampling_method)
            else None
        ),
        "alpha_grid": args.alpha_grid.tolist() if args.method == "multi_alpha_cp" else None,
        "fast_student_kl": bool(args.fast_student_kl),
        "kl_grid_size": int(args.kl_grid_size),
        "jones_m_grid": args.jones_m_grid.tolist(),
        "cp_m_grid": args.cp_m_grid.tolist(),
        "clarke_k_grid": args.clarke_k_grid.tolist(),
        "clarke_candidates": int(args.clarke_candidates),
        "cp_region": (
            "exact_student_t_equal_tailed_equals_hdr"
            if args.method in {"cp", "multi_alpha_cp"}
            else None
        ),
        "n_tasks": int(len(tasks)),
    }
    config, summary, runtime = add_runtime_metadata(
        df=df,
        summary=summary,
        config=config,
        start_time=start,
        variant=f"normal_unknown_{args.method}_{args.multi_alpha_label if args.method == 'multi_alpha_cp' else ('predictive' if args.method == 'ppd_cp' else args.distance)}",
    )
    out_variant = args.multi_alpha_label if args.method == "multi_alpha_cp" else ("predictive" if args.method == "ppd_cp" else args.distance)
    variant_path = (
        Path(out_variant) / args.sampling_method
        if is_exploratory_sampling(args.sampling_method)
        and args.method in {"jones", "cp", "multi_alpha_cp", "ppd_cp"}
        else Path(out_variant)
    )
    out_dir = Path(args.output_root) / args.method / variant_path
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
    if cp_grid_diagnostics is not None and args.save_cp_diagnostics:
        cp_grid_diagnostics.to_csv(out_dir / "cp_grid_diagnostics.csv", index=False)
    print(f"normal_unknown/{args.method}/{out_variant}: {len(df)} rows in {runtime['runtime_seconds']:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()
