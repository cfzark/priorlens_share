import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma


DEFAULT_MU_GRID = np.linspace(-4.0, 4.0, 17)
DEFAULT_SIGMA2_MEAN_GRID = np.asarray(
    [0.25, 0.35, 0.50, 0.70, 0.85, 1.00, 1.15, 1.30,
     1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00],
    dtype=float,
)
DEFAULT_SIGMA2_VAR_GRID = np.asarray([0.25, 2.0, 4.0, 10.0, 20.0])
PARAMETERIZATIONS = ("mu_log_sigma2", "mu_sigma2")


def parse_float_grid(value, default):
    if value is None:
        return np.asarray(default, dtype=float)
    values = [float(piece) for piece in str(value).replace(",", " ").split()]
    if not values:
        raise ValueError("Empty numeric grid.")
    return np.asarray(values, dtype=float)


def sigma_var_tag(value):
    return f"{float(value):g}".replace(".", "p")


def ig_params_from_mean_var(mean, variance):
    mean = np.asarray(mean, dtype=float)
    variance = float(variance)
    if np.any(mean <= 0.0) or variance <= 0.0:
        raise ValueError("IG means and variance must be positive.")
    kappa = 2.0 + np.square(mean) / variance
    eta = mean * (kappa - 1.0)
    return kappa, eta


def morita_components(
    sigma2_mean,
    sigma2_var,
    tau0_sq,
    parameterization,
    epsilon_c,
):
    """Analytic joint Morita trace information under the prior predictive.

    A single canonical epsilon prior is constructed in (mu, sigma2), then
    transformed when the trace is evaluated in (mu, log sigma2).
    """
    kappa, eta = ig_params_from_mean_var(sigma2_mean, sigma2_var)
    kappa = float(kappa)
    eta = float(eta)
    sigma2_mean = float(sigma2_mean)
    lambda0 = 1.0 / float(tau0_sq)
    lambda_eps = lambda0 / float(epsilon_c)
    kappa_eps = 2.0 + (kappa - 2.0) / float(epsilon_c)
    eta_eps = sigma2_mean * (kappa_eps - 1.0)

    if parameterization == "mu_sigma2":
        prior_mu = lambda0 / sigma2_mean
        prior_scale = (kappa - 3.5) / sigma2_mean**2
        epsilon_mu = lambda_eps / sigma2_mean
        epsilon_scale = (kappa_eps - 3.5) / sigma2_mean**2
        slope_mu = 1.0 / sigma2_mean
        slope_scale = (0.5 + tau0_sq) / sigma2_mean**2
        evaluation_scale = sigma2_mean
    elif parameterization == "mu_log_sigma2":
        mean_log_sigma2 = np.log(eta) - digamma(kappa)
        epsilon_mean_log_sigma2 = np.log(eta_eps) - digamma(kappa_eps)
        inverse_evaluation_scale = float(np.exp(-mean_log_sigma2))

        prior_mu = lambda0 * inverse_evaluation_scale
        prior_scale = eta * inverse_evaluation_scale
        epsilon_mu = lambda_eps * inverse_evaluation_scale
        epsilon_scale = eta_eps * inverse_evaluation_scale
        slope_mu = inverse_evaluation_scale
        slope_scale = (
            0.5
            * sigma2_mean
            * (1.0 + tau0_sq)
            * inverse_evaluation_scale
        )
        evaluation_scale = mean_log_sigma2
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    d_prior = prior_mu + prior_scale
    d_epsilon_0 = epsilon_mu + epsilon_scale
    slope = slope_mu + slope_scale
    continuous_ess = (d_prior - d_epsilon_0) / slope

    return {
        "kappa": kappa,
        "eta": eta,
        "kappa_epsilon": float(kappa_eps),
        "eta_epsilon": float(eta_eps),
        "epsilon_mean_log_sigma2": float(
            np.log(eta_eps) - digamma(kappa_eps)
        ),
        "lambda0": float(lambda0),
        "lambda_epsilon": float(lambda_eps),
        "evaluation_scale": float(evaluation_scale),
        "D_prior_mu": float(prior_mu),
        "D_prior_scale": float(prior_scale),
        "D_prior_trace": float(d_prior),
        "D_epsilon_0_mu": float(epsilon_mu),
        "D_epsilon_0_scale": float(epsilon_scale),
        "D_epsilon_0_trace": float(d_epsilon_0),
        "Dq_slope_mu": float(slope_mu),
        "Dq_slope_scale": float(slope_scale),
        "Dq_slope_trace": float(slope),
        "ess_unrestricted": float(continuous_ess),
    }


def build_rows(args):
    mu_grid = parse_float_grid(args.prior_values, DEFAULT_MU_GRID)
    sigma2_mean_grid = parse_float_grid(
        args.sigma2_mean_values, DEFAULT_SIGMA2_MEAN_GRID
    )
    sigma2_var_grid = parse_float_grid(
        args.sigma2_prior_vars, DEFAULT_SIGMA2_VAR_GRID
    )
    parameterizations = (
        PARAMETERIZATIONS
        if args.parameterization == "all"
        else (args.parameterization,)
    )

    rows = []
    for sigma2_var in sigma2_var_grid:
        for parameterization in parameterizations:
            for sigma2_mean in sigma2_mean_grid:
                components = morita_components(
                    sigma2_mean=sigma2_mean,
                    sigma2_var=sigma2_var,
                    tau0_sq=args.tau0_sq,
                    parameterization=parameterization,
                    epsilon_c=args.epsilon_c,
                )
                unrestricted = components["ess_unrestricted"]
                clipped = float(np.clip(unrestricted, args.m_min, args.m_max))
                integer_ess = int(
                    args.m_grid[np.argmin(np.abs(args.m_grid - unrestricted))]
                )
                boundary_hit = bool(
                    unrestricted <= args.m_min or unrestricted >= args.m_max
                )
                for prior_loc in mu_grid:
                    rows.append(
                        {
                            "model": "normal_unknown",
                            "method": "morita_joint2d",
                            "baseline": "epsilon",
                            "source": "analytic_prior_predictive_trace_curvature",
                            "parameterization": parameterization,
                            "prior_loc": float(prior_loc),
                            "prior_value": float(prior_loc),
                            "sigma2_prior_mean": float(sigma2_mean),
                            "sigma2_prior_var": float(sigma2_var),
                            "tau0_sq": float(args.tau0_sq),
                            "nominal_mu_ess": float(1.0 / args.tau0_sq),
                            "epsilon_c": float(args.epsilon_c),
                            "ess": clipped,
                            "ess_continuous": clipped,
                            "ess_unrestricted": unrestricted,
                            "ess_integer": integer_ess,
                            "boundary_hit": boundary_hit,
                            **components,
                        }
                    )
    return pd.DataFrame(rows)


def save_leaf(group, output_root, args, runtime_seconds):
    sigma2_var = float(group["sigma2_prior_var"].iloc[0])
    parameterization = str(group["parameterization"].iloc[0])
    out_dir = (
        Path(output_root)
        / f"sigma2var_{sigma_var_tag(sigma2_var)}"
        / "epsilon"
        / "morita_joint2d"
        / parameterization
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = group.copy().sort_values(["sigma2_prior_mean", "prior_loc"])
    summary = results.copy()
    summary["mean_ess"] = summary["ess_continuous"]
    summary["sd_ess"] = 0.0
    summary["median_ess"] = summary["ess_continuous"]
    summary["n_rep"] = 0
    prior_grid = summary[
        [
            "prior_loc",
            "sigma2_prior_mean",
            "sigma2_prior_var",
            "tau0_sq",
            "kappa",
            "eta",
        ]
    ].copy()

    results.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    prior_grid.to_csv(out_dir / "prior_grid.csv", index=False)

    config = {
        "model": "normal_unknown",
        "method": "morita_joint2d",
        "baseline": "epsilon",
        "parameterization": parameterization,
        "distance": "absolute_trace_information_difference",
        "expectation": "analytic_informative_prior_predictive",
        "tau0_sq": float(args.tau0_sq),
        "nominal_mu_ess": float(1.0 / args.tau0_sq),
        "sigma2_prior_var": sigma2_var,
        "epsilon_c": float(args.epsilon_c),
        "m_grid": args.m_grid.tolist(),
        "interpolation": "exact_linear_objective",
        "reported_ess": "continuous_minimizer_clipped_to_m_grid_bounds",
        "translation_invariant_in_mu0": True,
        "epsilon_prior_note": (
            "One canonical NIG epsilon prior is constructed in (mu, sigma2) "
            "with both variances multiplied by epsilon_c. The log-scale "
            "analysis uses the exact transformation of that same prior."
        ),
    }
    runtime = {
        "runtime_seconds_total": float(runtime_seconds),
        "analytic": True,
        "n_rows_in_leaf": int(len(results)),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2)
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analytic Morita joint-NIG heatmaps in two parameterizations."
    )
    parser.add_argument(
        "--output-root", default="results/NIG_sigma2_mean_heatmap"
    )
    parser.add_argument("--tau0-sq", type=float, default=0.2)
    parser.add_argument("--sigma2-prior-vars", default=None)
    parser.add_argument("--prior-values", default=None)
    parser.add_argument("--sigma2-mean-values", default=None)
    parser.add_argument(
        "--parameterization",
        choices=["all", *PARAMETERIZATIONS],
        default="all",
    )
    parser.add_argument("--epsilon-c", type=float, default=10_000.0)
    parser.add_argument("--m-min", type=int, default=0)
    parser.add_argument("--m-max", type=int, default=20)
    args = parser.parse_args()
    if args.tau0_sq <= 0.0:
        parser.error("--tau0-sq must be positive.")
    if args.epsilon_c <= 1.0:
        parser.error("--epsilon-c must exceed 1.")
    if args.m_max < args.m_min:
        parser.error("--m-max must be at least --m-min.")
    args.m_grid = np.arange(args.m_min, args.m_max + 1, dtype=int)
    return args


def main():
    args = parse_args()
    start = time.perf_counter()
    rows = build_rows(args)
    runtime_seconds = time.perf_counter() - start
    output_dirs = []
    for _, group in rows.groupby(
        ["sigma2_prior_var", "parameterization"], sort=True
    ):
        output_dirs.append(
            save_leaf(group, args.output_root, args, runtime_seconds)
        )

    combined_path = Path(args.output_root) / "morita_joint2d_all_summary.csv"
    combined = rows.copy()
    combined["mean_ess"] = combined["ess_continuous"]
    combined["sd_ess"] = 0.0
    combined.to_csv(combined_path, index=False)

    print("Morita joint-NIG analytic heatmaps complete")
    print(f"  tau0_sq: {args.tau0_sq:g}")
    print(f"  nominal mu ESS: {1.0 / args.tau0_sq:g}")
    print(f"  rows: {len(rows)}")
    print(f"  combined: {combined_path}")
    for out_dir in output_dirs:
        print(f"  leaf: {out_dir}")


if __name__ == "__main__":
    main()
