import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from .distributions import *
    from .models import *
    from .risk_metrics import SquaredPosteriorMeanError, make_risk_metric
except ImportError:  # pragma: no cover
    from distributions import *
    from models import *
    from risk_metrics import SquaredPosteriorMeanError, make_risk_metric


class ReimherrRunner1D:
    """
    Reimherr-style ECSS runner using posterior risk matching.

    By default this preserves the original local behavior: bootstrap-resampled
    pseudo datasets and squared posterior-mean error. Set sampler="likelihood"
    to simulate pseudo data from model.sample_given_theta(theta_star, r).
    """

    def __init__(
        self,
        model,
        prior_p,
        prior_b,
        theta_star,
        y_obs,
        B=2000,
        risk_metric=None,
        sampler="bootstrap",
        data_sampler=None,
    ):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.theta_star = float(theta_star)
        self.y_obs = np.asarray(y_obs, dtype=float).ravel()
        self.B = int(B)
        self.risk_metric = self._coerce_risk_metric(risk_metric)
        self.sampler = str(sampler).lower()
        self.data_sampler = data_sampler

        if self.y_obs.ndim != 1:
            raise ValueError("y_obs must be a 1D array.")
        if len(self.y_obs) == 0:
            raise ValueError("y_obs must be non-empty.")
        if self.B <= 0:
            raise ValueError("B must be positive.")
        if self.sampler not in {"bootstrap", "likelihood", "custom"}:
            raise ValueError("sampler must be one of: bootstrap, likelihood, custom.")
        if self.sampler == "custom" and self.data_sampler is None:
            raise ValueError("data_sampler is required when sampler='custom'.")

    @staticmethod
    def _coerce_risk_metric(risk_metric):
        if risk_metric is None:
            return SquaredPosteriorMeanError()
        if isinstance(risk_metric, str):
            return make_risk_metric(risk_metric)
        return risk_metric

    def _posterior_mean(self, y, prior):
        return float(self.model.posterior(y, prior).mean())

    def _simulate_dataset(self, r, rng):
        if r <= 0:
            raise ValueError("Pseudo sample size r must be positive.")
        if self.sampler == "bootstrap":
            idx = rng.integers(low=0, high=len(self.y_obs), size=r)
            return self.y_obs[idx]
        if self.sampler == "likelihood":
            if not hasattr(self.model, "sample_given_theta"):
                raise ValueError("model must implement sample_given_theta for likelihood sampling.")
            return np.asarray(
                self.model.sample_given_theta(self.theta_star, r, rng=rng),
                dtype=float,
            ).ravel()
        return np.asarray(
            self.data_sampler(theta_star=self.theta_star, r=r, rng=rng),
            dtype=float,
        ).ravel()

    def estimate_risk(self, prior, r, rng):
        vals = np.empty(self.B, dtype=float)
        for b in range(self.B):
            yb = self._simulate_dataset(r=r, rng=rng)
            posterior = self.model.posterior(yb, prior)
            vals[b] = self.risk_metric.compute(
                posterior=posterior,
                theta_star=self.theta_star,
                rng=rng,
            )
        return float(np.mean(vals))

    def estimate_mse(self, prior, r, rng):
        return self.estimate_risk(prior=prior, r=r, rng=rng)

    @staticmethod
    def _interpolate_ecss(m_grid, loss_baseline, target_loss, discrete_ecss):
        m_grid = np.asarray(m_grid, dtype=float)
        loss_baseline = np.asarray(loss_baseline, dtype=float)
        diff = loss_baseline - float(target_loss)

        roots = []
        for i in range(len(m_grid) - 1):
            d0, d1 = diff[i], diff[i + 1]
            if not (np.isfinite(d0) and np.isfinite(d1)):
                continue
            if d0 == 0:
                roots.append(m_grid[i])
            elif d0 * d1 < 0 and loss_baseline[i + 1] != loss_baseline[i]:
                frac = (target_loss - loss_baseline[i]) / (
                    loss_baseline[i + 1] - loss_baseline[i]
                )
                roots.append(m_grid[i] + frac * (m_grid[i + 1] - m_grid[i]))
        if len(roots) == 0:
            return float(discrete_ecss)
        return float(min(roots, key=lambda x: abs(x - discrete_ecss)))

    def run(self, n, m_grid, rng, interpolate=False):
        m_grid = np.asarray(m_grid, dtype=int)
        m_grid = m_grid[n + m_grid >= 1]
        if len(m_grid) == 0:
            raise ValueError("No valid m in m_grid such that n + m >= 1.")

        loss_prior = self.estimate_risk(prior=self.prior_p, r=n, rng=rng)
        loss_baseline = np.empty(len(m_grid), dtype=float)

        for i, m in enumerate(m_grid):
            loss_baseline[i] = self.estimate_risk(
                prior=self.prior_b,
                r=int(n + m),
                rng=rng,
            )

        abs_diff = np.abs(loss_prior - loss_baseline)
        discrete_ecss = int(m_grid[np.argmin(abs_diff)])
        ecss_float = (
            self._interpolate_ecss(m_grid, loss_baseline, loss_prior, discrete_ecss)
            if interpolate
            else float(discrete_ecss)
        )
        return {
            "n": int(n),
            "m_grid": m_grid,
            "loss_prior": float(loss_prior),
            "loss_baseline": loss_baseline,
            "abs_diff": abs_diff,
            "ecss": discrete_ecss,
            "ecss_float": float(ecss_float),
            "risk_metric": repr(self.risk_metric),
            "sampler": self.sampler,
            "interpolate": bool(interpolate),
        }


def reimherr_quick_ratio(r, delta2, risk_metric="posterior_mse"):
    """
    Quick/asymptotic Reimherr estimate R(k)=M(k)/k.

    Parameters
    ----------
    r : float
        Nominal prior-to-likelihood information ratio, e.g. posterior baseline
        variance divided by prior variance in the scalar normal approximation.
    delta2 : float
        Standardized prior-data discrepancy, e.g.
        (posterior_baseline_mean - prior_mean)^2 / prior_variance.
    risk_metric : {"posterior_mse", "squared_mean"}
        "posterior_mse" includes posterior uncertainty and posterior-mean error;
        "squared_mean" matches the older local implementation that uses only
        squared posterior-mean error.
    """
    r = float(r)
    delta2 = float(delta2)
    key = str(risk_metric).lower().replace("-", "_")
    if r < 0:
        raise ValueError("r must be nonnegative.")
    if delta2 < 0:
        raise ValueError("delta2 must be nonnegative.")

    if key in {"posterior_mse", "expected_squared", "posterior_expected_squared"}:
        denom = 2.0 + r + r * delta2
        return float(2.0 * (1.0 + r) ** 2 / denom - 1.0)

    if key in {"squared_mean", "squared_mean_error", "mse_mean", "l2"}:
        denom = 1.0 + r * delta2
        return float((1.0 + r) ** 2 / denom - 1.0)

    raise ValueError("risk_metric must be 'posterior_mse' or 'squared_mean'.")


def reimherr_quick_m(k, r, delta2, risk_metric="posterior_mse"):
    """Return M(k)=k*R(k) from reimherr_quick_ratio."""
    return float(k) * reimherr_quick_ratio(
        r=r,
        delta2=delta2,
        risk_metric=risk_metric,
    )


def reimherr_quick_normal_known(
    y,
    prior,
    baseline_prior,
    model,
    risk_metric="posterior_mse",
):
    """
    Convenience quick estimate for scalar Normal-known-variance analyses.

    The approximation uses the prior mean/variance and the baseline posterior
    mean/variance from all observed data.
    """
    if not isinstance(prior, NormalDist):
        raise ValueError("prior must be NormalDist.")
    post_b = model.posterior(y, baseline_prior)
    r = float(post_b.var() / prior.var())
    delta2 = float((post_b.mean() - prior.mean()) ** 2 / prior.var())
    n = len(np.asarray(y, dtype=float).ravel())
    ratio = reimherr_quick_ratio(r=r, delta2=delta2, risk_metric=risk_metric)
    return {
        "n": int(n),
        "r": float(r),
        "delta2": float(delta2),
        "R": float(ratio),
        "M": float(n * ratio),
        "risk_metric": str(risk_metric),
    }


def compute_one_reimherr_normal_known(
    mu0,
    r,
    y,
    model,
    prior_q0,
    m_grid,
    B=2000,
    prior_sd=np.sqrt(0.2),
    theta_star=0.0,
    base_seed=321,
    risk_metric=None,
    sampler="bootstrap",
    interpolate=False,
):
    rng = np.random.default_rng(base_seed + int(r))
    runner = ReimherrRunner1D(
        model=model,
        prior_p=NormalDist(mu0, prior_sd),
        prior_b=prior_q0,
        theta_star=theta_star,
        y_obs=y,
        B=B,
        risk_metric=risk_metric,
        sampler=sampler,
    )
    out = runner.run(n=len(y), m_grid=m_grid, rng=rng, interpolate=interpolate)
    return {
        "rep": int(r),
        "prior_loc": float(mu0),
        "ecss": int(out["ecss"]),
        "ecss_float": float(out["ecss_float"]),
        "loss_prior": float(out["loss_prior"]),
        "m_grid": out["m_grid"].tolist(),
        "loss_baseline": out["loss_baseline"].tolist(),
        "abs_diff": out["abs_diff"].tolist(),
        "theta_star": float(theta_star),
        "sample": np.asarray(y, dtype=float).tolist(),
        "model_type": "normal_known",
        "risk_metric": out["risk_metric"],
        "sampler": out["sampler"],
        "interpolate": bool(out["interpolate"]),
    }


def run_reimherr_normal_known_grid(
    prior_means,
    R=100,
    rep_start=0,
    rep_end=None,
    n=20,
    true_mu=0.0,
    sigma=np.sqrt(3.0),
    prior_sd=np.sqrt(0.2),
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
    prior_means = np.asarray(prior_means, dtype=float)
    baseline_prior = FlatPrior() if baseline_prior is None else baseline_prior
    model = NormalMeanKnownVarModel(sigma=sigma)
    m_grid = np.arange(-(n - 1), 31) if m_grid is None else np.asarray(m_grid, dtype=int)
    if rep_end is None:
        rep_end = R

    tasks = []
    for r in range(rep_start, rep_end):
        rng = np.random.default_rng(data_seed + r)
        y = rng.normal(loc=true_mu, scale=sigma, size=n)
        if theta_star_mode == "baseline_posterior_mean":
            theta_star = float(model.posterior(y, baseline_prior).mean())
        elif theta_star_mode == "truth":
            theta_star = float(true_mu)
        else:
            raise ValueError("theta_star_mode must be 'baseline_posterior_mean' or 'truth'.")
        for mu0 in prior_means:
            tasks.append((mu0, y, theta_star, r))

    results_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_one_reimherr_normal_known)(
            mu0=mu0,
            r=r,
            y=y,
            model=model,
            prior_q0=baseline_prior,
            m_grid=m_grid,
            B=B,
            prior_sd=prior_sd,
            theta_star=theta_star,
            base_seed=base_seed,
            risk_metric=risk_metric,
            sampler=sampler,
            interpolate=interpolate,
        )
        for mu0, y, theta_star, r in tasks
    )
    results_df = pd.DataFrame(results_list)
    ecss_col = "ecss_float" if interpolate else "ecss"
    summary_df = (
        results_df.groupby("prior_loc")[ecss_col]
        .agg(mean_ecss="mean", sd_ecss="std", n_rep="count")
        .reset_index()
        .sort_values("prior_loc")
    )
    return results_df, summary_df


def compute_one_reimherr_poisson(
    lambda0,
    r,
    y,
    model,
    prior_q0,
    m_grid,
    B=2000,
    prior_strength=4.0,
    theta_star=1.0,
    base_seed=20260312,
    risk_metric=None,
    sampler="bootstrap",
    interpolate=False,
):
    rng = np.random.default_rng(base_seed + 10000 * (int(r) + 1))
    alpha0 = float(lambda0) * float(prior_strength)
    beta0 = float(prior_strength)
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("Poisson informative prior requires alpha0 > 0 and beta0 > 0.")

    runner = ReimherrRunner1D(
        model=model,
        prior_p=GammaDist(alpha0, beta0),
        prior_b=prior_q0,
        theta_star=theta_star,
        y_obs=y,
        B=B,
        risk_metric=risk_metric,
        sampler=sampler,
    )
    out = runner.run(n=len(y), m_grid=m_grid, rng=rng, interpolate=interpolate)
    return {
        "rep": int(r),
        "prior_loc": float(lambda0),
        "ecss": int(out["ecss"]),
        "ecss_float": float(out["ecss_float"]),
        "loss_prior": float(out["loss_prior"]),
        "m_grid": out["m_grid"].tolist(),
        "loss_baseline": out["loss_baseline"].tolist(),
        "abs_diff": out["abs_diff"].tolist(),
        "theta_star": float(theta_star),
        "sample": np.asarray(y, dtype=float).tolist(),
        "model_type": "poisson",
        "risk_metric": out["risk_metric"],
        "sampler": out["sampler"],
        "interpolate": bool(out["interpolate"]),
    }


def run_reimherr_poisson_grid(
    prior_means,
    R=100,
    rep_start=0,
    rep_end=None,
    n=20,
    true_lambda=3.0,
    prior_strength=4.0,
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
    prior_means = np.asarray(prior_means, dtype=float)
    baseline_prior = FlatPrior() if baseline_prior is None else baseline_prior
    model = PoissonModel()
    m_grid = np.arange(-(n - 1), 31) if m_grid is None else np.asarray(m_grid, dtype=int)
    if rep_end is None:
        rep_end = R

    tasks = []
    for r in range(rep_start, rep_end):
        rng = np.random.default_rng(data_seed + r)
        y = rng.poisson(lam=true_lambda, size=n).astype(float)
        if theta_star_mode == "baseline_posterior_mean":
            theta_star = float(model.posterior(y, baseline_prior).mean())
        elif theta_star_mode == "truth":
            theta_star = float(true_lambda)
        else:
            raise ValueError("theta_star_mode must be 'baseline_posterior_mean' or 'truth'.")
        for lam0 in prior_means:
            tasks.append((lam0, y, theta_star, r))

    results_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_one_reimherr_poisson)(
            lambda0=lam0,
            r=r,
            y=y,
            model=model,
            prior_q0=baseline_prior,
            m_grid=m_grid,
            B=B,
            prior_strength=prior_strength,
            theta_star=theta_star,
            base_seed=base_seed,
            risk_metric=risk_metric,
            sampler=sampler,
            interpolate=interpolate,
        )
        for lam0, y, theta_star, r in tasks
    )
    results_df = pd.DataFrame(results_list)
    ecss_col = "ecss_float" if interpolate else "ecss"
    summary_df = (
        results_df.groupby("prior_loc")[ecss_col]
        .agg(mean_ecss="mean", sd_ecss="std", n_rep="count")
        .reset_index()
        .sort_values("prior_loc")
    )
    return results_df, summary_df
