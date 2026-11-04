import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from .distributions import BetaDist, FlatPrior, JeffreysPrior
    from .models import BinomialModel
    from .Jones import JonesObservedPriorRunner1D
    from .cp import obj_func_exact
    from .reimherr import ReimherrRunner1D
except ImportError:  # pragma: no cover
    from distributions import BetaDist, FlatPrior, JeffreysPrior
    from models import BinomialModel
    from Jones import JonesObservedPriorRunner1D
    from cp import obj_func_exact
    from reimherr import ReimherrRunner1D


def beta_prior_from_mean_strength(prior_mean, prior_strength):
    prior_mean = float(prior_mean)
    prior_strength = float(prior_strength)
    if not (0 < prior_mean < 1):
        raise ValueError("Beta prior mean must be in (0, 1).")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive.")
    return BetaDist(
        alpha=prior_mean * prior_strength,
        beta=(1.0 - prior_mean) * prior_strength,
    )


class CoverageProbabilityExactBetaBinomialRunner:
    """
    Exact CP-EPSS runner for Beta-Binomial posteriors.
    """

    def __init__(self, model, prior_p, prior_b, m_grid, alpha=0.05, B_fit=50):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.m_grid = np.asarray(m_grid, dtype=int)
        self.alpha = float(alpha)
        self.B_fit = int(B_fit)

        if np.any(self.m_grid < 0):
            raise ValueError("m_grid must be nonnegative.")
        if self.B_fit <= 0:
            raise ValueError("B_fit must be positive.")

    def _posterior(self, y, prior):
        post = self.model.posterior(y, prior)
        if not isinstance(post, BetaDist):
            raise ValueError("Beta-Binomial exact CP runner expects BetaDist posteriors.")
        return post

    def _one_fit(self, y, rng):
        y = np.asarray(y, dtype=float).ravel()
        post_base0 = self._posterior(y, self.prior_b)
        post_inf0 = self._posterior(y, self.prior_p)

        M = int(np.max(self.m_grid))
        y_new_all_1 = self.model.posterior_predictive(post_base0, M, rng=rng)
        y_new_all_2 = self.model.posterior_predictive(post_inf0, M, rng=rng)

        obj1 = np.zeros(len(self.m_grid), dtype=float)
        obj2 = np.zeros(len(self.m_grid), dtype=float)

        for i, m in enumerate(self.m_grid):
            y_aug_1 = y.copy() if m == 0 else np.concatenate([y, y_new_all_1[:m]])
            post_inf_m = self._posterior(y_aug_1, self.prior_p)
            _, _, _, _, obj1[i] = obj_func_exact(post_base0, post_inf_m, alpha=self.alpha)

            y_aug_2 = y.copy() if m == 0 else np.concatenate([y, y_new_all_2[:m]])
            post_base_m = self._posterior(y_aug_2, self.prior_b)
            _, _, _, _, obj2[i] = obj_func_exact(post_inf0, post_base_m, alpha=self.alpha)

        m_star_idx = int(np.argmin(np.minimum(obj1, obj2)))
        sign = 1 if obj1[m_star_idx] > obj2[m_star_idx] else -1
        return int(self.m_grid[m_star_idx]) * sign, obj1, obj2

    def run_dataset(self, y, rng):
        signed_mstars = np.empty(self.B_fit, dtype=float)
        for b in range(self.B_fit):
            m_star, _, _ = self._one_fit(y, rng=rng)
            signed_mstars[b] = m_star
        return {
            "epss_dataset": float(np.mean(signed_mstars)),
            "signed_mstars": signed_mstars,
        }


def _baseline_name(prior):
    return prior.__class__.__name__


def _simulate_beta_binomial_dataset(n, trials, true_p, rng):
    return rng.binomial(n=trials, p=true_p, size=n).astype(float)


def _reimherr_one(prior_mean, rep, n, trials, true_p, prior_strength,
                  baseline_prior, m_grid, B, data_seed, base_seed,
                  theta_star_mode, risk_metric, sampler, interpolate):
    rng_data = np.random.default_rng(data_seed + int(rep))
    y = _simulate_beta_binomial_dataset(n, trials, true_p, rng_data)

    model = BinomialModel(trials=trials)
    if theta_star_mode == "baseline_posterior_mean":
        theta_star = float(model.posterior(y, baseline_prior).mean())
    elif theta_star_mode == "truth":
        theta_star = float(true_p)
    else:
        raise ValueError("theta_star_mode must be 'baseline_posterior_mean' or 'truth'.")

    rng_fit = np.random.default_rng(base_seed + 10000 * (int(rep) + 1))
    prior_p = beta_prior_from_mean_strength(prior_mean, prior_strength)
    out = ReimherrRunner1D(
        model=model,
        prior_p=prior_p,
        prior_b=baseline_prior,
        theta_star=theta_star,
        y_obs=y,
        B=B,
        risk_metric=risk_metric,
        sampler=sampler,
    ).run(n=len(y), m_grid=m_grid, rng=rng_fit, interpolate=interpolate)

    return {
        "rep": int(rep),
        "prior_mean": float(prior_mean),
        "alpha0": float(prior_p.alpha),
        "beta0": float(prior_p.beta),
        "ecss": int(out["ecss"]),
        "ecss_float": float(out["ecss_float"]),
        "loss_prior": float(out["loss_prior"]),
        "m_grid": out["m_grid"].tolist(),
        "loss_baseline": out["loss_baseline"].tolist(),
        "abs_diff": out["abs_diff"].tolist(),
        "theta_star": float(theta_star),
        "sample": y.tolist(),
        "sample_mean": float(np.mean(y)),
        "sample_sum": float(np.sum(y)),
        "model_type": "beta_binomial",
        "baseline_type": _baseline_name(baseline_prior),
        "trials": int(trials),
        "true_p": float(true_p),
        "risk_metric": out["risk_metric"],
        "sampler": out["sampler"],
        "interpolate": bool(out["interpolate"]),
    }


def run_reimherr_beta_binomial_grid(
    prior_means,
    R=50,
    rep_start=0,
    rep_end=None,
    n=20,
    trials=20,
    true_p=0.3,
    prior_strength=80.0,
    baseline_prior=None,
    m_grid=None,
    B=2000,
    data_seed=321,
    base_seed=20260312,
    theta_star_mode="baseline_posterior_mean",
    risk_metric=None,
    sampler="bootstrap",
    interpolate=False,
    n_jobs=-1,
    verbose=10,
):
    baseline_prior = FlatPrior() if baseline_prior is None else baseline_prior
    prior_means = np.asarray(prior_means, dtype=float)
    m_grid = np.arange(-(n - 1), 31) if m_grid is None else np.asarray(m_grid, dtype=int)
    rep_end = R if rep_end is None else rep_end
    tasks = [(pm, rep) for rep in range(rep_start, rep_end) for pm in prior_means]

    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_reimherr_one)(
            prior_mean=pm,
            rep=rep,
            n=n,
            trials=trials,
            true_p=true_p,
            prior_strength=prior_strength,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            B=B,
            data_seed=data_seed,
            base_seed=base_seed,
            theta_star_mode=theta_star_mode,
            risk_metric=risk_metric,
            sampler=sampler,
            interpolate=interpolate,
        )
        for pm, rep in tasks
    )
    df = pd.DataFrame(rows)
    ecss_col = "ecss_float" if interpolate else "ecss"
    summary = (
        df.groupby("prior_mean")[ecss_col]
        .agg(mean_ecss="mean", sd_ecss="std", median_ecss="median", n_rep="count")
        .reset_index()
        .sort_values("prior_mean")
    )
    return df, summary


def _jones_one(prior_mean, rep, n, trials, true_p, prior_strength,
               baseline_prior, m_grid, S_opess, data_seed, distance_type,
               wasserstein_order, gamma_method, gamma_grid_size, alpha, mc_B):
    rng = np.random.default_rng(data_seed + int(rep))
    y = _simulate_beta_binomial_dataset(n, trials, true_p, rng)
    model = BinomialModel(trials=trials)
    prior_inf = beta_prior_from_mean_strength(prior_mean, prior_strength)

    out = JonesObservedPriorRunner1D(
        model=model,
        prior_inf=prior_inf,
        prior_base=baseline_prior,
        distance_type=distance_type,
        wasserstein_order=wasserstein_order,
        alpha=alpha,
        mc_B=mc_B,
    ).run(y=y, m_grid=m_grid, S=S_opess, rng=rng)

    return {
        "rep": int(rep),
        "prior_mean": float(prior_mean),
        "alpha0": float(prior_inf.alpha),
        "beta0": float(prior_inf.beta),
        "sample": y.tolist(),
        "sample_mean": float(np.mean(y)),
        "sample_sum": float(np.sum(y)),
        "mopess": float(out["mopess"]),
        "sd_opess": float(out["sd_opess"]),
        "median_opess": float(out["median_opess"]),
        "q05_opess": float(out["q05_opess"]),
        "q95_opess": float(out["q95_opess"]),
        "best_m_D": int(out["best_m_D"]),
        "best_m_Dsharp": int(out["best_m_Dsharp"]),
        "m_grid": out["m_grid"],
        "mean_D_path": out["mean_D_path"],
        "mean_Dsharp_path": out["mean_Dsharp_path"],
        "opess_draws": out["opess_draws"],
        "distance_type": distance_type,
        "wasserstein_order": int(wasserstein_order) if distance_type == "wasserstein" else None,
        "gamma_method": gamma_method if distance_type == "wasserstein" else None,
        "gamma_grid_size": int(gamma_grid_size) if distance_type == "wasserstein" else None,
        "model_type": "beta_binomial",
        "baseline_type": _baseline_name(baseline_prior),
        "trials": int(trials),
        "true_p": float(true_p),
        "S_opess": int(S_opess),
    }


def run_jones_beta_binomial_grid(
    prior_means,
    R=50,
    rep_start=0,
    rep_end=None,
    n=20,
    trials=20,
    true_p=0.3,
    prior_strength=80.0,
    baseline_prior=None,
    m_grid=None,
    S_opess=500,
    data_seed=321,
    n_jobs=-1,
    distance_type="wasserstein",
    wasserstein_order=2,
    gamma_method="grid",
    gamma_grid_size=2000,
    alpha=0.05,
    mc_B=2000,
    verbose=10,
):
    baseline_prior = FlatPrior() if baseline_prior is None else baseline_prior
    prior_means = np.asarray(prior_means, dtype=float)
    m_grid = list(range(n, 51)) if m_grid is None else list(np.asarray(m_grid, dtype=int))
    rep_end = R if rep_end is None else rep_end
    tasks = [(pm, rep) for rep in range(rep_start, rep_end) for pm in prior_means]

    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_jones_one)(
            prior_mean=pm,
            rep=rep,
            n=n,
            trials=trials,
            true_p=true_p,
            prior_strength=prior_strength,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            S_opess=S_opess,
            data_seed=data_seed,
            distance_type=distance_type,
            wasserstein_order=wasserstein_order,
            gamma_method=gamma_method,
            gamma_grid_size=gamma_grid_size,
            alpha=alpha,
            mc_B=mc_B,
        )
        for pm, rep in tasks
    )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("prior_mean")["mopess"]
        .agg(mean_mopess="mean", sd_mopess="std", median_mopess="median")
        .reset_index()
        .sort_values("prior_mean")
    )
    return df, summary


def _cp_one(prior_mean, rep, n, trials, true_p, prior_strength,
            baseline_prior, m_grid, alpha, B_fit, data_seed, fit_seed_base):
    model = BinomialModel(trials=trials)
    prior_p = beta_prior_from_mean_strength(prior_mean, prior_strength)
    rng_data = np.random.default_rng(data_seed + int(rep))
    y = _simulate_beta_binomial_dataset(n, trials, true_p, rng_data)
    rng_fit = np.random.default_rng(fit_seed_base + 10000 * (int(rep) + 1))

    out = CoverageProbabilityExactBetaBinomialRunner(
        model=model,
        prior_p=prior_p,
        prior_b=baseline_prior,
        m_grid=m_grid,
        alpha=alpha,
        B_fit=B_fit,
    ).run_dataset(y, rng=rng_fit)

    return {
        "rep": int(rep),
        "prior_mean": float(prior_mean),
        "alpha0": float(prior_p.alpha),
        "beta0": float(prior_p.beta),
        "epss": float(out["epss_dataset"]),
        "mean_signed_mstar": float(np.mean(out["signed_mstars"])),
        "sd_signed_mstar": float(np.std(out["signed_mstars"], ddof=1)) if B_fit > 1 else 0.0,
        "model_type": "beta_binomial",
        "baseline_type": _baseline_name(baseline_prior),
        "sample": y.tolist(),
        "sample_mean": float(np.mean(y)),
        "sample_sum": float(np.sum(y)),
        "signed_mstars": out["signed_mstars"].tolist(),
        "trials": int(trials),
        "true_p": float(true_p),
    }


def run_exact_cp_beta_binomial_grid(
    prior_means,
    R=50,
    rep_start=0,
    rep_end=None,
    n=20,
    trials=20,
    true_p=0.3,
    prior_strength=80.0,
    baseline_prior=None,
    m_grid=None,
    alpha=0.05,
    B_fit=500,
    data_seed=321,
    fit_seed_base=20260312,
    n_jobs=-1,
    verbose=10,
):
    baseline_prior = FlatPrior() if baseline_prior is None else baseline_prior
    prior_means = np.asarray(prior_means, dtype=float)
    m_grid = np.arange(0, 31) if m_grid is None else np.asarray(m_grid, dtype=int)
    rep_end = R if rep_end is None else rep_end
    tasks = [(pm, rep) for rep in range(rep_start, rep_end) for pm in prior_means]

    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_cp_one)(
            prior_mean=pm,
            rep=rep,
            n=n,
            trials=trials,
            true_p=true_p,
            prior_strength=prior_strength,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            alpha=alpha,
            B_fit=B_fit,
            data_seed=data_seed,
            fit_seed_base=fit_seed_base,
        )
        for pm, rep in tasks
    )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("prior_mean")["epss"]
        .agg(mean_epss="mean", median_epss="median", sd_epss="std")
        .reset_index()
        .sort_values("prior_mean")
    )
    return df, summary
