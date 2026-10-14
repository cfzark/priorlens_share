import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "function"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prioreffect import (  # noqa: E402
    BetaDist,
    BinomialModel,
    FlatPrior,
    GammaDist,
    JeffreysPrior,
    FlatMuInverseGammaPrior,
    NormalDist,
    NormalInverseGammaDist,
    NormalMeanKnownVarModel,
    NormalMeanUnknownVarModel,
    PoissonModel,
    ReimherrRunner1D,
    WiesenfarthCalderazzoECSSRunner1D,
    add_runtime_metadata,
    default_beta_prior_mean_grid,
    default_normal_prior_loc_grid,
    default_poisson_prior_lambda_grid,
    make_baseline_prior,
    reimherr_quick_ratio,
    save_experiment_outputs,
    start_timer,
)


DEFAULT_RISK_METRICS = ("squared_mean", "posterior_mse", "posterior_variance")
REIMHERR_VARIANTS = ("bootstrap", "likelihood", "quick")


def prior_grid_for_model(model_name, grid_name):
    if grid_name != "default":
        raise ValueError("Only --prior-grid default is currently supported.")
    if model_name == "normal":
        return default_normal_prior_loc_grid()
    if model_name == "normal_unknown":
        return default_normal_prior_loc_grid()
    if model_name == "poisson_gamma":
        return default_poisson_prior_lambda_grid()
    if model_name == "beta_binomial":
        return default_beta_prior_mean_grid()
    raise ValueError(f"Unknown model: {model_name}")


def model_and_prior(model_name, prior_loc, args):
    if model_name == "normal":
        model = NormalMeanKnownVarModel(sigma=args.normal_sigma)
        prior = NormalDist(prior_loc, args.normal_prior_sd)
        return model, prior

    if model_name == "normal_unknown":
        model = NormalMeanUnknownVarModel()
        prior = NormalInverseGammaDist(
            mu0=prior_loc,
            tau0_sq=args.normal_unknown_tau0_sq,
            kappa=args.normal_unknown_kappa,
            eta=args.normal_unknown_eta,
        )
        return model, prior

    if model_name == "poisson_gamma":
        model = PoissonModel()
        alpha = float(prior_loc) * float(args.poisson_prior_strength)
        beta = float(args.poisson_prior_strength)
        prior = GammaDist(alpha=alpha, beta=beta)
        return model, prior

    if model_name == "beta_binomial":
        model = BinomialModel(trials=args.beta_trials)
        alpha = float(prior_loc) * float(args.beta_prior_strength)
        beta = (1.0 - float(prior_loc)) * float(args.beta_prior_strength)
        prior = BetaDist(alpha=alpha, beta=beta)
        return model, prior

    raise ValueError(f"Unknown model: {model_name}")


def simulate_dataset(model_name, rep, args):
    rng = np.random.default_rng(args.data_seed + int(rep))
    if model_name == "normal":
        return rng.normal(loc=args.normal_true_mu, scale=args.normal_sigma, size=args.n)
    if model_name == "normal_unknown":
        return rng.normal(
            loc=args.normal_unknown_true_mu,
            scale=args.normal_unknown_true_sigma,
            size=args.n,
        )
    if model_name == "poisson_gamma":
        return rng.poisson(lam=args.poisson_true_lambda, size=args.n).astype(float)
    if model_name == "beta_binomial":
        return rng.binomial(n=args.beta_trials, p=args.beta_true_p, size=args.n).astype(float)
    raise ValueError(f"Unknown model: {model_name}")


def truth_for_model(model_name, args):
    if model_name == "normal":
        return float(args.normal_true_mu)
    if model_name == "normal_unknown":
        return float(args.normal_unknown_true_mu)
    if model_name == "poisson_gamma":
        return float(args.poisson_true_lambda)
    if model_name == "beta_binomial":
        return float(args.beta_true_p)
    raise ValueError(f"Unknown model: {model_name}")


def prior_axis_name(model_name):
    if model_name == "normal":
        return "prior_loc"
    if model_name == "normal_unknown":
        return "prior_loc"
    if model_name == "poisson_gamma":
        return "prior_lambda"
    if model_name == "beta_binomial":
        return "prior_mean"
    raise ValueError(f"Unknown model: {model_name}")


def prior_moments(prior):
    return float(prior.mean()), float(prior.var())


def baseline_prior_for_model(model_name, args):
    if model_name == "normal_unknown":
        if args.baseline == "flat":
            return FlatMuInverseGammaPrior(
                kappa=args.normal_unknown_kappa,
                eta=args.normal_unknown_eta,
            )
        if args.baseline == "jeffreys":
            return JeffreysPrior()
        raise ValueError("baseline must be flat or jeffreys.")
    return make_baseline_prior(args.baseline)


def marginal_mu_prior(model, prior):
    return model.posterior([], prior)


def data_sampler_for_model(model_name, args):
    if model_name != "normal_unknown":
        return None

    def _sampler(theta_star, r, rng):
        return rng.normal(
            loc=float(theta_star),
            scale=args.normal_unknown_true_sigma,
            size=int(r),
        )

    return _sampler


def nondegenerate_bootstrap_sampler(y_obs):
    y_obs = np.asarray(y_obs, dtype=float).ravel()

    def _sampler(theta_star, r, rng):
        r = int(r)
        for _ in range(1000):
            idx = rng.integers(low=0, high=len(y_obs), size=r)
            yb = y_obs[idx]
            if r < 2 or np.sum((yb - np.mean(yb)) ** 2) > 0:
                return yb
        noise = rng.normal(loc=0.0, scale=1e-10, size=r)
        return yb + noise

    return _sampler


def theta_star_for(model, y, baseline_prior, model_name, args):
    if args.theta_star_mode == "truth":
        return truth_for_model(model_name, args)
    if args.theta_star_mode == "baseline_posterior_mean":
        return float(model.posterior(y, baseline_prior).mean())
    raise ValueError("theta_star_mode must be truth or baseline_posterior_mean.")


def reimherr_quick_scalar(model, y, prior, baseline_prior, risk_metric):
    post_b = model.posterior(y, baseline_prior)
    if isinstance(prior, NormalInverseGammaDist):
        prior_mu = marginal_mu_prior(model, prior)
        prior_mean, prior_var = prior_moments(prior_mu)
    else:
        prior_mean, prior_var = prior_moments(prior)
    post_mean = float(post_b.mean())
    post_var = float(post_b.var())
    r = post_var / prior_var

    key = str(risk_metric).lower().replace("-", "_")
    if key in {"posterior_variance", "variance", "uncertainty"}:
        M = len(y) * r
        return {
            "ecss": float(M),
            "quick_R": float(r),
            "quick_r": float(r),
            "quick_delta2": 0.0,
        }

    delta2 = (post_mean - prior_mean) ** 2 / prior_var
    ratio = reimherr_quick_ratio(r=r, delta2=delta2, risk_metric=risk_metric)
    return {
        "ecss": float(len(y) * ratio),
        "quick_R": float(ratio),
        "quick_r": float(r),
        "quick_delta2": float(delta2),
    }


def run_one_task(model_name, method, variant, risk_metric, prior_loc, rep, args):
    baseline_prior = baseline_prior_for_model(model_name, args)
    model, prior = model_and_prior(model_name, prior_loc, args)
    y = simulate_dataset(model_name, rep, args)
    theta_star = theta_star_for(model, y, baseline_prior, model_name, args)

    axis = prior_axis_name(model_name)
    base = {
        "model": model_name,
        "method": method,
        "variant": variant,
        "risk_metric": risk_metric,
        "rep": int(rep),
        axis: float(prior_loc),
        "prior_value": float(prior_loc),
        "baseline": args.baseline,
        "theta_star": float(theta_star),
        "sample_mean": float(np.mean(y)),
        "sample_sum": float(np.sum(y)),
        "n": int(args.n),
    }

    if method == "reimherr" and variant == "quick":
        out = reimherr_quick_scalar(model, y, prior, baseline_prior, risk_metric)
        return {
            **base,
            "ecss": float(out["ecss"]),
            "ecss_discrete": np.nan,
            "quick_R": out["quick_R"],
            "quick_r": out["quick_r"],
            "quick_delta2": out["quick_delta2"],
            "sampler": "quick",
            "interpolate": False,
        }

    rng = np.random.default_rng(
        args.base_seed
        + 1000000 * int(rep)
        + int(round(10000 * float(prior_loc)))
        + (17 if method == "wiesenfarth" else 0)
    )

    if method == "reimherr":
        sampler = variant
        data_sampler = None
        if model_name == "normal_unknown" and variant == "likelihood":
            sampler = "custom"
            data_sampler = data_sampler_for_model(model_name, args)
        if model_name == "normal_unknown" and args.baseline == "jeffreys" and variant == "bootstrap":
            sampler = "custom"
            data_sampler = nondegenerate_bootstrap_sampler(y)
        m_grid = np.asarray(args.reimherr_m_grid, dtype=int)
        if model_name == "normal_unknown" and args.baseline == "jeffreys":
            m_grid = m_grid[len(y) + m_grid >= 2]
        runner = ReimherrRunner1D(
            model=model,
            prior_p=prior,
            prior_b=baseline_prior,
            theta_star=theta_star,
            y_obs=y,
            B=args.B,
            risk_metric=risk_metric,
            sampler=sampler,
            data_sampler=data_sampler,
        )
        out = runner.run(
            n=len(y),
            m_grid=m_grid,
            rng=rng,
            interpolate=args.interpolate,
        )
        return {
            **base,
            "ecss": float(out["ecss_float"] if args.interpolate else out["ecss"]),
            "ecss_discrete": int(out["ecss"]),
            "loss_prior": float(out["loss_prior"]),
            "sampler": out["sampler"],
            "interpolate": bool(out["interpolate"]),
        }

    if method == "wiesenfarth":
        runner = WiesenfarthCalderazzoECSSRunner1D(
            model=model,
            prior_p=prior,
            prior_b=baseline_prior,
            theta_star=theta_star,
            B=args.B,
            risk_metric=risk_metric,
            data_sampler=data_sampler_for_model(model_name, args),
        )
        out = runner.run(
            n=len(y),
            m_grid=np.asarray(args.wiesenfarth_m_grid, dtype=int),
            rng=rng,
        )
        return {
            **base,
            "ecss": float(out["ecss"]),
            "ecss_discrete": int(out["ecss"]),
            "loss_baseline": float(out["loss_baseline"]),
            "sampler": "likelihood",
            "interpolate": False,
        }

    raise ValueError(f"Unsupported method/variant: {method}/{variant}")


def summarize(df):
    return (
        df.groupby(["model", "method", "variant", "risk_metric", "prior_value"])["ecss"]
        .agg(mean_ecss="mean", sd_ecss="std", median_ecss="median", n_rep="count")
        .reset_index()
        .sort_values(["model", "method", "variant", "risk_metric", "prior_value"])
    )


def expanded_methods(args):
    methods = ["reimherr", "wiesenfarth"] if args.method == "all" else [args.method]
    risk_metrics = list(DEFAULT_RISK_METRICS) if args.risk_metric == "all" else [args.risk_metric]
    tasks = []
    for method in methods:
        if method == "reimherr":
            variants = list(REIMHERR_VARIANTS) if args.reimherr_variant == "all" else [args.reimherr_variant]
        else:
            variants = ["likelihood_minus_m"]
        for variant in variants:
            for risk_metric in risk_metrics:
                if variant == "quick" and risk_metric == "l1":
                    continue
                tasks.append((method, variant, risk_metric))
    return tasks


def run_combo(model_name, method, variant, risk_metric, args):
    if args.prior_values is not None:
        prior_grid = np.asarray(args.prior_values, dtype=float)
    else:
        prior_grid = prior_grid_for_model(model_name, args.prior_grid)
    rep_end = args.rep_end if args.rep_end is not None else args.R
    reps = range(args.rep_start, rep_end)
    tasks = [
        (model_name, method, variant, risk_metric, float(prior_loc), rep, args)
        for rep in reps
        for prior_loc in prior_grid
    ]

    start = start_timer()
    rows = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(run_one_task)(*task) for task in tasks
    )
    df = pd.DataFrame(rows)
    summary = summarize(df)

    config = {
        "model": model_name,
        "method": method,
        "variant": variant,
        "risk_metric": risk_metric,
        "R": int(args.R),
        "rep_start": int(args.rep_start),
        "rep_end": int(rep_end),
        "n": int(args.n),
        "B": int(args.B),
        "baseline": args.baseline,
        "theta_star_mode": args.theta_star_mode,
        "interpolate": bool(args.interpolate),
        "prior_grid": prior_grid.tolist(),
        "reimherr_m_grid": list(map(int, args.reimherr_m_grid)),
        "wiesenfarth_m_grid": list(map(int, args.wiesenfarth_m_grid)),
        "normal_sigma": float(args.normal_sigma),
        "normal_true_mu": float(args.normal_true_mu),
        "normal_prior_sd": float(args.normal_prior_sd),
        "normal_unknown_true_mu": float(args.normal_unknown_true_mu),
        "normal_unknown_true_sigma": float(args.normal_unknown_true_sigma),
        "normal_unknown_tau0_sq": float(args.normal_unknown_tau0_sq),
        "normal_unknown_kappa": float(args.normal_unknown_kappa),
        "normal_unknown_eta": float(args.normal_unknown_eta),
        "poisson_true_lambda": float(args.poisson_true_lambda),
        "poisson_prior_strength": float(args.poisson_prior_strength),
        "beta_trials": int(args.beta_trials),
        "beta_true_p": float(args.beta_true_p),
        "beta_prior_strength": float(args.beta_prior_strength),
        "data_seed": int(args.data_seed),
        "base_seed": int(args.base_seed),
        "n_tasks": int(len(tasks)),
    }
    config, summary, runtime = add_runtime_metadata(
        df=df,
        summary=summary,
        config=config,
        start_time=start,
        variant=f"{method}_{variant}_{risk_metric}",
    )

    out_dir = (
        Path(args.output_root)
        / model_name
        / method
        / str(variant)
        / str(risk_metric)
    )
    save_experiment_outputs(
        df=df,
        summary=summary,
        out_dir=out_dir,
        config=config,
        grid_filename="prior_grid.csv",
        grid_values=prior_grid,
    )
    print(
        f"{model_name}/{method}/{variant}/{risk_metric}: "
        f"{len(df)} rows in {runtime['runtime_seconds']:.2f}s -> {out_dir}"
    )
    return df, summary


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
    parser = argparse.ArgumentParser(
        description="Run Reimherr/Wiesenfarth sensitivity experiments across models and D choices."
    )
    parser.add_argument("--model", choices=["normal", "normal_unknown", "poisson_gamma", "beta_binomial", "all"], default="all")
    parser.add_argument("--method", choices=["reimherr", "wiesenfarth", "all"], default="all")
    parser.add_argument("--reimherr-variant", choices=["bootstrap", "likelihood", "quick", "all"], default="all")
    parser.add_argument(
        "--risk-metric",
        choices=["squared_mean", "posterior_mse", "posterior_variance", "l1", "all"],
        default="all",
    )
    parser.add_argument("--prior-grid", choices=["default"], default="default")
    parser.add_argument(
        "--prior-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit prior grid values; overrides --prior-grid.",
    )
    parser.add_argument("--output-root", default="results/rw_sensitivity")

    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--baseline", choices=["flat", "jeffreys"], default="flat")
    parser.add_argument("--theta-star-mode", choices=["baseline_posterior_mean", "truth"], default="baseline_posterior_mean")
    parser.add_argument("--interpolate", action="store_true", default=True)
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    parser.add_argument("--reimherr-m-grid", default=None, help="Comma/range grid, e.g. -19:30")
    parser.add_argument("--wiesenfarth-m-grid", default=None, help="Comma/range grid, e.g. 0:19")
    parser.add_argument("--data-seed", type=int, default=321)
    parser.add_argument("--base-seed", type=int, default=20260312)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=10)

    parser.add_argument("--normal-true-mu", type=float, default=0.0)
    parser.add_argument("--normal-sigma", type=float, default=1.0)
    parser.add_argument("--normal-prior-sd", type=float, default=np.sqrt(0.2))

    parser.add_argument("--normal-unknown-true-mu", type=float, default=0.0)
    parser.add_argument("--normal-unknown-true-sigma", type=float, default=1.0)
    parser.add_argument("--normal-unknown-tau0-sq", type=float, default=0.2)
    parser.add_argument("--normal-unknown-kappa", type=float, default=3.0)
    parser.add_argument("--normal-unknown-eta", type=float, default=2.0)

    parser.add_argument("--poisson-true-lambda", type=float, default=3.0)
    parser.add_argument("--poisson-prior-strength", type=float, default=4.0)

    parser.add_argument("--beta-trials", type=int, default=20)
    parser.add_argument("--beta-true-p", type=float, default=0.3)
    parser.add_argument("--beta-prior-strength", type=float, default=80.0)
    args = parser.parse_args()

    args.reimherr_m_grid = parse_int_grid(args.reimherr_m_grid, range(-(args.n - 1), args.n + 1))
    args.wiesenfarth_m_grid = parse_int_grid(args.wiesenfarth_m_grid, range(-args.n, args.n))
    return args


def main():
    args = parse_args()
    models = ["normal", "normal_unknown", "poisson_gamma", "beta_binomial"] if args.model == "all" else [args.model]
    combos = expanded_methods(args)
    for model_name in models:
        for method, variant, risk_metric in combos:
            run_combo(model_name, method, variant, risk_metric, args)


if __name__ == "__main__":
    main()
