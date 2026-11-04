import os
import json
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from .distributions import *
    from .models import *
    from .metrics import *
    from .augmentation import *
    from .sampling import (
        empirical_bootstrap_augmentation_path,
        empirical_deletion_path,
        is_deletion_sampling,
        is_empirical_augmentation,
        select_signed_direction,
        validate_sampling_method,
    )
    from .utils import *
except ImportError:  # pragma: no cover - keeps old notebook-style imports working
    from distributions import *
    from models import *
    from metrics import *
    from augmentation import *
    from sampling import (
        empirical_bootstrap_augmentation_path,
        empirical_deletion_path,
        is_deletion_sampling,
        is_empirical_augmentation,
        select_signed_direction,
        validate_sampling_method,
    )
    from utils import *


class JonesObservedPriorRunner1D:
    """
    Jones-style OPESS / mOPESS runner for one-dimensional posterior objects.

    This is the Gaussian known-variance implementation from the notebook. It
    also works for other one-dimensional models when posterior_predictive and
    posterior are available and the selected metric supports the posterior
    distribution type.
    """

    def __init__(
        self,
        model,
        prior_inf,
        prior_base,
        distance_type="wasserstein",
        wasserstein_order=2,
        alpha=0.05,
        mc_B=2000,
        sampling_method="ppd_augmentation",
        min_retained=1,
    ):
        self.model = model
        self.prior_inf = prior_inf
        self.prior_base = prior_base
        self.distance_type = distance_type
        self.wasserstein_order = wasserstein_order
        self.alpha = alpha
        self.mc_B = mc_B
        self.sampling_method = validate_sampling_method(sampling_method)
        self.min_retained = int(min_retained)

        if distance_type == "kl":
            self.distance = KLDivergence(B=mc_B)
        elif distance_type == "truncated_kl":
            self.distance = TruncatedKLDivergence(alpha=alpha, B=mc_B)
        elif distance_type == "wasserstein":
            self.distance = WassersteinDistance(order=wasserstein_order)
        else:
            raise ValueError(
                "distance_type must be one of: "
                "'kl', 'truncated_kl', 'wasserstein'."
            )

    def _compute_distance(self, P, Q, rng):
        return self.distance.compute(P, Q, rng=rng)

    def run(self, y, m_grid, S=1000, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        if is_deletion_sampling(self.sampling_method):
            return self._run_deletion(y=y, m_grid=m_grid, S=S, rng=rng)
        if is_empirical_augmentation(self.sampling_method):
            return self._run_bootstrap_augmentation(
                y=y, m_grid=m_grid, S=S, rng=rng
            )

        m_grid = np.asarray(m_grid, dtype=int)
        m_grid = np.unique(m_grid[m_grid >= n])

        if len(m_grid) == 0:
            raise ValueError("m_grid must contain at least one value >= n.")

        q_n = self.model.posterior(y, self.prior_inf)
        q_n_b = self.model.posterior(y, self.prior_base)

        opess_draws = []
        d_path_mean = np.zeros(len(m_grid), dtype=float)
        dsharp_path_mean = np.zeros(len(m_grid), dtype=float)

        for _ in range(S):
            d_vals = []
            dsharp_vals = []

            for j, m in enumerate(m_grid):
                r = int(m - n)

                if r == 0:
                    q_m_b = q_n_b
                    q_m = q_n
                else:
                    future_for_base_side = self.model.posterior_predictive(
                        q_n, size=r, rng=rng
                    )
                    future_for_inf_side = self.model.posterior_predictive(
                        q_n, size=r, rng=rng
                    )

                    x_m = np.concatenate([y, future_for_base_side])
                    x_tilde_m = np.concatenate([y, future_for_inf_side])

                    q_m_b = self.model.posterior(x_m, self.prior_base)
                    q_m = self.model.posterior(x_tilde_m, self.prior_inf)

                d_m = self._compute_distance(q_m_b, q_n, rng=rng)
                dsharp_m = self._compute_distance(q_n_b, q_m, rng=rng)

                d_vals.append(d_m)
                dsharp_vals.append(dsharp_m)

                d_path_mean[j] += d_m
                dsharp_path_mean[j] += dsharp_m

            d_vals = np.asarray(d_vals, dtype=float)
            dsharp_vals = np.asarray(dsharp_vals, dtype=float)

            idx_d = int(np.nanargmin(d_vals))
            idx_dsharp = int(np.nanargmin(dsharp_vals))

            if d_vals[idx_d] <= dsharp_vals[idx_dsharp]:
                opess = int(m_grid[idx_d] - n)
            else:
                opess = int(n - m_grid[idx_dsharp])

            opess_draws.append(opess)

        opess_draws = np.asarray(opess_draws, dtype=float)
        d_path_mean /= S
        dsharp_path_mean /= S

        return {
            "n": int(n),
            "m_grid": m_grid.tolist(),
            "opess_draws": opess_draws.tolist(),
            "mopess": float(np.mean(opess_draws)),
            "sd_opess": float(np.std(opess_draws, ddof=1)) if len(opess_draws) > 1 else 0.0,
            "median_opess": float(np.median(opess_draws)),
            "q05_opess": float(np.quantile(opess_draws, 0.05)),
            "q95_opess": float(np.quantile(opess_draws, 0.95)),
            "mean_D_path": d_path_mean.tolist(),
            "mean_Dsharp_path": dsharp_path_mean.tolist(),
            "best_m_D": int(m_grid[int(np.argmin(d_path_mean))]),
            "best_m_Dsharp": int(m_grid[int(np.argmin(dsharp_path_mean))]),
            "sampling_method": self.sampling_method,
            "m_grid_semantics": "total_sample_size",
        }

    def _run_deletion(self, y, m_grid, S, rng):
        """Run paired empirical deletion paths.

        The negative direction compares the informative full-data posterior
        with a baseline posterior based on n-m observations. The positive
        direction compares the informative n-m posterior with the baseline
        full-data posterior.
        """
        y = np.asarray(y, dtype=float).ravel()
        q_inf_full = self.model.posterior(y, self.prior_inf)
        q_base_full = self.model.posterior(y, self.prior_base)

        signed = np.empty(int(S), dtype=float)
        negative_m = np.empty(int(S), dtype=float)
        positive_m = np.empty(int(S), dtype=float)
        negative_sum = None
        positive_sum = None

        for outer in range(int(S)):
            grid, path = empirical_deletion_path(
                y,
                m_grid,
                rng,
                self.sampling_method,
                min_retained=self.min_retained,
            )
            negative = np.empty(len(grid), dtype=float)
            positive = np.empty(len(grid), dtype=float)
            for j, m in enumerate(grid):
                reduced = path[int(m)]
                q_base_reduced = self.model.posterior(reduced, self.prior_base)
                q_inf_reduced = self.model.posterior(reduced, self.prior_inf)
                negative[j] = self._compute_distance(
                    q_inf_full, q_base_reduced, rng=rng
                )
                positive[j] = self._compute_distance(
                    q_inf_reduced, q_base_full, rng=rng
                )

            selected = select_signed_direction(negative, positive, grid)
            signed[outer] = selected["signed_m"]
            negative_m[outer] = selected["negative_m"]
            positive_m[outer] = selected["positive_m"]
            if negative_sum is None:
                negative_sum = np.zeros_like(negative)
                positive_sum = np.zeros_like(positive)
            negative_sum += negative
            positive_sum += positive

        negative_mean = negative_sum / int(S)
        positive_mean = positive_sum / int(S)
        return {
            "n": int(len(y)),
            "m_grid": grid.tolist(),
            "opess_draws": signed.tolist(),
            "mopess": float(np.mean(signed)),
            "sd_opess": float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0,
            "median_opess": float(np.median(signed)),
            "q05_opess": float(np.quantile(signed, 0.05)),
            "q95_opess": float(np.quantile(signed, 0.95)),
            "mean_negative_path": negative_mean.tolist(),
            "mean_positive_path": positive_mean.tolist(),
            "best_m_negative_mean_path": int(grid[int(np.argmin(negative_mean))]),
            "best_m_positive_mean_path": int(grid[int(np.argmin(positive_mean))]),
            "mean_negative_mstar": float(np.mean(negative_m)),
            "mean_positive_mstar": float(np.mean(positive_m)),
            "sampling_method": self.sampling_method,
            "m_grid_semantics": "number_deleted",
            "negative_direction": "pi(.|X_n) vs pi_b(.|Y_{n-m})",
            "positive_direction": "pi(.|Y_{n-m}) vs pi_b(.|X_n)",
        }

    def _run_bootstrap_augmentation(self, y, m_grid, S, rng):
        """Run paired empirical-bootstrap augmentation paths."""
        y = np.asarray(y, dtype=float).ravel()
        q_inf_full = self.model.posterior(y, self.prior_inf)
        q_base_full = self.model.posterior(y, self.prior_base)

        signed = np.empty(int(S), dtype=float)
        negative_m = np.empty(int(S), dtype=float)
        positive_m = np.empty(int(S), dtype=float)
        negative_sum = None
        positive_sum = None

        for outer in range(int(S)):
            grid, path = empirical_bootstrap_augmentation_path(
                y, m_grid, rng
            )
            negative = np.empty(len(grid), dtype=float)
            positive = np.empty(len(grid), dtype=float)
            for j, m in enumerate(grid):
                additions = path[int(m)]
                augmented = (
                    y
                    if int(m) == 0
                    else np.concatenate([y, additions])
                )
                q_base_augmented = self.model.posterior(
                    augmented, self.prior_base
                )
                q_inf_augmented = self.model.posterior(
                    augmented, self.prior_inf
                )
                negative[j] = self._compute_distance(
                    q_base_full, q_inf_augmented, rng=rng
                )
                positive[j] = self._compute_distance(
                    q_inf_full, q_base_augmented, rng=rng
                )

            selected = select_signed_direction(negative, positive, grid)
            signed[outer] = selected["signed_m"]
            negative_m[outer] = selected["negative_m"]
            positive_m[outer] = selected["positive_m"]
            if negative_sum is None:
                negative_sum = np.zeros_like(negative)
                positive_sum = np.zeros_like(positive)
            negative_sum += negative
            positive_sum += positive

        negative_mean = negative_sum / int(S)
        positive_mean = positive_sum / int(S)
        return {
            "n": int(len(y)),
            "m_grid": grid.tolist(),
            "opess_draws": signed.tolist(),
            "mopess": float(np.mean(signed)),
            "sd_opess": (
                float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0
            ),
            "median_opess": float(np.median(signed)),
            "q05_opess": float(np.quantile(signed, 0.05)),
            "q95_opess": float(np.quantile(signed, 0.95)),
            "mean_negative_path": negative_mean.tolist(),
            "mean_positive_path": positive_mean.tolist(),
            "best_m_negative_mean_path": int(
                grid[int(np.argmin(negative_mean))]
            ),
            "best_m_positive_mean_path": int(
                grid[int(np.argmin(positive_mean))]
            ),
            "mean_negative_mstar": float(np.mean(negative_m)),
            "mean_positive_mstar": float(np.mean(positive_m)),
            "sampling_method": self.sampling_method,
            "m_grid_semantics": "number_augmented",
            "negative_direction": "pi_b(.|X_n) vs pi(.|X_n union A_m)",
            "positive_direction": "pi(.|X_n) vs pi_b(.|X_n union A_m)",
        }


class JonesObservedPriorRunnerPoisson:
    """
    Jones-style OPESS / mOPESS runner for Poisson-Gamma model.

    Model:
        X_i | lambda ~ Pois(lambda)

    Priors:
        informative prior: Gamma(alpha0, beta0)
        baseline prior   : FlatPrior(), i.e. q(lambda) proportional to 1

    Posterior:
        informative:
            lambda | y ~ Gamma(alpha0 + sum(y), beta0 + n)

        flat baseline:
            lambda | y ~ Gamma(sum(y) + 1, n)

    Distance choices:
        - "kl"
        - "truncated_kl"
        - "wasserstein"
    """

    def __init__(
        self,
        model,
        prior_inf,
        prior_base,
        distance_type="wasserstein",
        wasserstein_order=2,
        gamma_method="grid",
        gamma_grid_size=2000,
        alpha=0.05,
        mc_B=2000,
        sampling_method="ppd_augmentation",
        min_retained=1,
    ):
        self.model = model
        self.prior_inf = prior_inf
        self.prior_base = prior_base
        self.distance_type = distance_type
        self.wasserstein_order = wasserstein_order
        self.gamma_method = gamma_method
        self.gamma_grid_size = gamma_grid_size
        self.alpha = alpha
        self.mc_B = mc_B
        self.sampling_method = validate_sampling_method(sampling_method)
        self.min_retained = int(min_retained)

        if distance_type == "kl":
            self.distance = KLDivergence(B=mc_B)
        elif distance_type == "truncated_kl":
            self.distance = TruncatedKLDivergence(alpha=alpha, B=mc_B)
        elif distance_type == "wasserstein":
            self.distance = WassersteinDistance(
                order=wasserstein_order,
                gamma_method=gamma_method,
                gamma_grid_size=gamma_grid_size,
            )
        else:
            raise ValueError(
                "distance_type must be one of: "
                "'kl', 'truncated_kl', 'wasserstein'."
            )

    def _compute_distance(self, P, Q, rng):
        return self.distance.compute(P, Q, rng=rng)

    def run(self, y, m_grid, S=1000, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        if is_deletion_sampling(
            self.sampling_method
        ) or is_empirical_augmentation(self.sampling_method):
            generic = JonesObservedPriorRunner1D(
                model=self.model,
                prior_inf=self.prior_inf,
                prior_base=self.prior_base,
                distance_type=self.distance_type,
                wasserstein_order=self.wasserstein_order,
                alpha=self.alpha,
                mc_B=self.mc_B,
                sampling_method=self.sampling_method,
                min_retained=self.min_retained,
            )
            generic.distance = self.distance
            return generic.run(y=y, m_grid=m_grid, S=S, rng=rng)

        m_grid = np.asarray(m_grid, dtype=int)
        m_grid = np.unique(m_grid[m_grid >= n])

        if len(m_grid) == 0:
            raise ValueError("m_grid must contain at least one value >= n.")

        # observed-data posteriors
        q_n = self.model.posterior(y, self.prior_inf)
        q_n_b = self.model.posterior(y, self.prior_base)

        opess_draws = []
        d_path_mean = np.zeros(len(m_grid), dtype=float)
        dsharp_path_mean = np.zeros(len(m_grid), dtype=float)

        for s in range(S):
            d_vals = []
            dsharp_vals = []

            for j, m in enumerate(m_grid):
                r = int(m - n)

                if r == 0:
                    q_m_b = q_n_b
                    q_m = q_n
                else:
                    # Jones: future samples from informative posterior predictive
                    future_for_base_side = self.model.posterior_predictive(
                        q_n, size=r, rng=rng
                    )
                    future_for_inf_side = self.model.posterior_predictive(
                        q_n, size=r, rng=rng
                    )

                    x_m = np.concatenate([y, future_for_base_side])
                    x_tilde_m = np.concatenate([y, future_for_inf_side])

                    q_m_b = self.model.posterior(x_m, self.prior_base)
                    q_m = self.model.posterior(x_tilde_m, self.prior_inf)

                d_m = self._compute_distance(q_m_b, q_n, rng=rng)
                dsharp_m = self._compute_distance(q_n_b, q_m, rng=rng)

                d_vals.append(d_m)
                dsharp_vals.append(dsharp_m)

                d_path_mean[j] += d_m
                dsharp_path_mean[j] += dsharp_m

            d_vals = np.asarray(d_vals, dtype=float)
            dsharp_vals = np.asarray(dsharp_vals, dtype=float)

            idx_d = int(np.nanargmin(d_vals))
            idx_dsharp = int(np.nanargmin(dsharp_vals))

            min_d = d_vals[idx_d]
            min_dsharp = dsharp_vals[idx_dsharp]

            if min_d <= min_dsharp:
                opess = int(m_grid[idx_d] - n)
            else:
                opess = int(n - m_grid[idx_dsharp])

            opess_draws.append(opess)

        opess_draws = np.asarray(opess_draws, dtype=float)
        d_path_mean /= S
        dsharp_path_mean /= S

        return {
            "n": int(n),
            "m_grid": m_grid.tolist(),
            "opess_draws": opess_draws.tolist(),
            "mopess": float(np.mean(opess_draws)),
            "sd_opess": float(np.std(opess_draws, ddof=1)) if len(opess_draws) > 1 else 0.0,
            "median_opess": float(np.median(opess_draws)),
            "q05_opess": float(np.quantile(opess_draws, 0.05)),
            "q95_opess": float(np.quantile(opess_draws, 0.95)),
            "mean_D_path": d_path_mean.tolist(),
            "mean_Dsharp_path": dsharp_path_mean.tolist(),
            "best_m_D": int(m_grid[int(np.argmin(d_path_mean))]),
            "best_m_Dsharp": int(m_grid[int(np.argmin(dsharp_path_mean))]),
            "sampling_method": self.sampling_method,
            "m_grid_semantics": "total_sample_size",
        }


def simulate_one_dataset(n, true_mu, true_sigma, rng):
    return rng.normal(loc=true_mu, scale=true_sigma, size=n).astype(float)


def compute_one_result(
    prior_loc,
    rep,
    *,
    n=20,
    true_mu=0.0,
    true_sigma=np.sqrt(3.0),
    sigma=np.sqrt(3.0),
    prior_sd=np.sqrt(0.2),
    m_grid=None,
    S_opess=1000,
    base_seed=321,
    distance_type="wasserstein",
    wasserstein_order=2,
    alpha=0.05,
    mc_B=2000,
):
    if m_grid is None:
        m_grid = list(range(n, 51))

    rng = np.random.default_rng(base_seed + rep)
    y = simulate_one_dataset(n=n, true_mu=true_mu, true_sigma=true_sigma, rng=rng)

    model = NormalMeanKnownVarModel(sigma=sigma)
    prior_inf = NormalDist(prior_loc, prior_sd)
    prior_base = FlatPrior()

    runner = JonesObservedPriorRunner1D(
        model=model,
        prior_inf=prior_inf,
        prior_base=prior_base,
        distance_type=distance_type,
        wasserstein_order=wasserstein_order,
        alpha=alpha,
        mc_B=mc_B,
    )

    out = runner.run(y=y, m_grid=m_grid, S=S_opess, rng=rng)

    return {
        "rep": int(rep),
        "prior_loc": float(prior_loc),
        "sample": y.tolist(),
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
        "model_type": "normal_known",
        "baseline_type": "flat_mu",
        "sigma": float(sigma),
        "true_mu": float(true_mu),
        "true_sigma": float(true_sigma),
        "prior_sd": float(prior_sd),
        "S_opess": int(S_opess),
    }


def run_jones_experiment(
    *,
    R=20,
    prior_locs=(0.0, 1.0, 3.0, 5.0),
    n=20,
    true_mu=0.0,
    true_sigma=np.sqrt(3.0),
    sigma=np.sqrt(3.0),
    prior_sd=np.sqrt(0.2),
    m_grid=None,
    S_opess=1000,
    base_seed=321,
    n_jobs=-1,
    distance_type="wasserstein",
    wasserstein_order=2,
    alpha=0.05,
    mc_B=2000,
    verbose=10,
):
    if m_grid is None:
        m_grid = list(range(n, 51))

    tasks = [(prior_loc, rep) for rep in range(R) for prior_loc in prior_locs]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_one_result)(
            prior_loc,
            rep,
            n=n,
            true_mu=true_mu,
            true_sigma=true_sigma,
            sigma=sigma,
            prior_sd=prior_sd,
            m_grid=m_grid,
            S_opess=S_opess,
            base_seed=base_seed,
            distance_type=distance_type,
            wasserstein_order=wasserstein_order,
            alpha=alpha,
            mc_B=mc_B,
        )
        for prior_loc, rep in tasks
    )

    df = pd.DataFrame(results)
    summary = (
        df.groupby("prior_loc")["mopess"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={
            "mean": "mean_mopess",
            "std": "sd_mopess",
            "median": "median_mopess"
        })
    )
    return df, summary


def simulate_one_poisson_dataset(n, true_lambda, rng):
    return rng.poisson(lam=true_lambda, size=n).astype(float)


def compute_one_poisson_jones_result(
    prior_mean,
    rep,
    *,
    n=20,
    true_lambda=3.0,
    beta0=4.0,
    baseline_prior=None,
    m_grid=None,
    S_opess=1000,
    base_seed=321,
    distance_type="wasserstein",
    wasserstein_order=2,
    gamma_method="grid",
    gamma_grid_size=2000,
    alpha=0.05,
    mc_B=2000,
):
    if m_grid is None:
        m_grid = list(range(n, n + 31))

    rng = np.random.default_rng(base_seed + rep)

    y = simulate_one_poisson_dataset(
        n=n,
        true_lambda=true_lambda,
        rng=rng
    )

    alpha0 = beta0 * prior_mean
    if alpha0 <= 0:
        raise ValueError("Need alpha0 = beta0 * prior_mean > 0.")

    model = PoissonModel()
    prior_inf = GammaDist(alpha=alpha0, beta=beta0)
    prior_base = FlatPrior() if baseline_prior is None else baseline_prior

    runner = JonesObservedPriorRunnerPoisson(
        model=model,
        prior_inf=prior_inf,
        prior_base=prior_base,
        distance_type=distance_type,
        wasserstein_order=wasserstein_order,
        gamma_method=gamma_method,
        gamma_grid_size=gamma_grid_size,
        alpha=alpha,
        mc_B=mc_B,
    )

    out = runner.run(
        y=y,
        m_grid=m_grid,
        S=S_opess,
        rng=rng,
    )

    return {
        "rep": int(rep),
        "prior_mean": float(prior_mean),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
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
        "model_type": "poisson",
        "baseline_type": prior_base.__class__.__name__,
        "true_lambda": float(true_lambda),
        "S_opess": int(S_opess),
    }

def run_poisson_jones_experiment(
    *,
    R=20,
    rep_start=0,
    rep_end=None,
    prior_means=(1.0, 2.0, 3.0, 4.0, 5.0),
    n=20,
    true_lambda=3.0,
    beta0=4.0,
    baseline_prior=None,
    m_grid=None,
    S_opess=1000,
    base_seed=321,
    n_jobs=-1,
    distance_type="wasserstein",
    wasserstein_order=2,
    gamma_method="grid",
    gamma_grid_size=2000,
    alpha=0.05,
    mc_B=2000,
    verbose=10,
):
    if m_grid is None:
        m_grid = list(range(n, n + 31))

    if rep_end is None:
        rep_end = R
    if baseline_prior is None:
        baseline_prior = FlatPrior()

    rep_ids = list(range(rep_start, rep_end))

    tasks = [(prior_mean, rep) for rep in rep_ids for prior_mean in prior_means]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_one_poisson_jones_result)(
            prior_mean,
            rep,
            n=n,
            true_lambda=true_lambda,
            beta0=beta0,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            S_opess=S_opess,
            base_seed=base_seed,
            distance_type=distance_type,
            wasserstein_order=wasserstein_order,
            gamma_method=gamma_method,
            gamma_grid_size=gamma_grid_size,
            alpha=alpha,
            mc_B=mc_B,
        )
        for prior_mean, rep in tasks
    )

    df = pd.DataFrame(results)

    summary = (
        df.groupby("prior_mean")["mopess"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={
            "mean": "mean_mopess",
            "std": "sd_mopess",
            "median": "median_mopess"
        })
    )

    return df, summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Jones-style mOPESS simulation for Poisson-Gamma model."
    )

    # ----------------------------
    # Replication control
    # ----------------------------
    parser.add_argument("--R", type=int, default=50,
                        help="Total number of replications in the full experiment.")
    parser.add_argument("--rep_start", type=int, default=0,
                        help="Start replicate index, inclusive.")
    parser.add_argument("--rep_end", type=int, default=None,
                        help="End replicate index, exclusive. If None, use R.")

    parser.add_argument("--base_seed", type=int, default=321,
                        help="Base seed. Replicate rep uses seed base_seed + rep.")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs.")
    parser.add_argument("--save_dir", type=str, default="results_jones_poisson",
                        help="Directory to save results.")

    # ----------------------------
    # Data generation
    # ----------------------------
    parser.add_argument("--n", type=int, default=20,
                        help="Observed sample size.")
    parser.add_argument("--true_lambda", type=float, default=3.0,
                        help="True Poisson rate.")

    # ----------------------------
    # Prior setting
    # ----------------------------
    parser.add_argument("--beta0", type=float, default=4.0,
                        help="Gamma prior rate beta0.")

    # ----------------------------
    # Jones OPESS setting
    # ----------------------------
    parser.add_argument("--m_grid", type=int, nargs="+",
                        default=list(range(20, 51)),
                        help="Candidate total sample sizes m. Must be >= n.")
    parser.add_argument("--S_opess", type=int, default=500,
                        help="Monte Carlo size for OPESS averaging.")

    # ----------------------------
    # Distance metric
    # ----------------------------
    parser.add_argument("--distance_type", type=str, default="wasserstein",
                        choices=["wasserstein", "kl", "truncated_kl"],
                        help="Posterior distance metric.")
    parser.add_argument("--wasserstein_order", type=int, default=2,
                        help="Wasserstein order. Usually 2 for Jones.")
    parser.add_argument("--gamma_method", type=str, default="grid",
                        choices=["grid", "quad"],
                        help="Method for Gamma-Gamma Wasserstein distance.")
    parser.add_argument("--gamma_grid_size", type=int, default=1000,
                        help="Grid size for Gamma-Gamma Wasserstein approximation.")

    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Truncation level for truncated KL.")
    parser.add_argument("--mc_B", type=int, default=2000,
                        help="Monte Carlo size for generic distance computation.")
    parser.add_argument("--verbose", type=int, default=10,
                        help="Verbosity for joblib.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rep_end is None:
        args.rep_end = args.R

    os.makedirs(args.save_dir, exist_ok=True)


    prior_lambda_grid = np.concatenate([
        np.linspace(0.5, 2.0, 31)[:-1],
        np.linspace(2.0, 4.0, 101)[:-1],
        np.linspace(4.0, 8.0, 41)
    ])
    
    df, summary = run_poisson_jones_experiment(
        R=args.R,
        rep_start=args.rep_start,
        rep_end=args.rep_end,
        prior_means=prior_lambda_grid,
        n=args.n,
        true_lambda=args.true_lambda,
        beta0=args.beta0,
        m_grid=args.m_grid,
        S_opess=args.S_opess,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
        distance_type=args.distance_type,
        wasserstein_order=args.wasserstein_order,
        gamma_method=args.gamma_method,
        gamma_grid_size=args.gamma_grid_size,
        alpha=args.alpha,
        mc_B=args.mc_B,
        verbose=args.verbose,
    )

    tag = (
        f"{args.distance_type}"
        f"_rep_{args.rep_start}_{args.rep_end}"
        f"_seed_{args.base_seed}"
    )

    out_json = os.path.join(args.save_dir, f"all_results_{tag}.json")
    out_csv = os.path.join(args.save_dir, f"summary_{tag}.csv")

    df.to_json(out_json, orient="records")
    summary.to_csv(out_csv, index=False)

    print(f"Saved row-level results to {out_json}")
    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()
