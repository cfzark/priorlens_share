import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from .distributions import *
    from .models import *
    from .sampling import (
        empirical_bootstrap_augmentation_path,
        empirical_deletion_path,
        is_deletion_sampling,
        is_empirical_augmentation,
        select_signed_direction,
        validate_sampling_method,
    )
    from .utils import obj_func
except ImportError:  # pragma: no cover
    from distributions import *
    from models import *
    from sampling import (
        empirical_bootstrap_augmentation_path,
        empirical_deletion_path,
        is_deletion_sampling,
        is_empirical_augmentation,
        select_signed_direction,
        validate_sampling_method,
    )
    from utils import obj_func


class CoverageProbabilityEPSSRunner1D:
    """
    Coverage-probability EPSS runner for 1D posterior objects.
    """

    def __init__(
        self,
        model,
        prior_p,
        prior_b,
        m_grid,
        alpha=0.05,
        B_fit=50,
        sampling_method="ppd_augmentation",
        min_retained=1,
    ):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.m_grid = np.asarray(m_grid, dtype=int)
        self.alpha = float(alpha)
        self.B_fit = int(B_fit)
        self.sampling_method = validate_sampling_method(sampling_method)
        self.min_retained = int(min_retained)

        if np.any(self.m_grid < 0):
            raise ValueError("m_grid must be nonnegative.")
        if self.B_fit <= 0:
            raise ValueError("B_fit must be positive.")

    def _posterior_mean_var(self, y, prior):
        post = self.model.posterior(y, prior)
        return float(post.mean()), float(post.var()), post

    def _one_fit(self, y, rng):
        y = np.asarray(y, dtype=float).ravel()

        m_base0, v_base0, post_base0 = self._posterior_mean_var(y, self.prior_b)
        m_inf0, v_inf0, post_inf0 = self._posterior_mean_var(y, self.prior_p)

        if is_deletion_sampling(self.sampling_method):
            grid, path = empirical_deletion_path(
                y,
                self.m_grid,
                rng,
                self.sampling_method,
                min_retained=self.min_retained,
            )
            obj_negative = np.empty(len(grid), dtype=float)
            obj_positive = np.empty(len(grid), dtype=float)
            for i, m in enumerate(grid):
                reduced = path[int(m)]
                _, _, post_base_reduced = self._posterior_mean_var(
                    reduced, self.prior_b
                )
                _, _, post_inf_reduced = self._posterior_mean_var(
                    reduced, self.prior_p
                )
                _, _, _, _, obj_negative[i], _, _ = obj_func(
                    m_inf0,
                    v_inf0,
                    float(post_base_reduced.mean()),
                    float(post_base_reduced.var()),
                    alpha=self.alpha,
                )
                _, _, _, _, obj_positive[i], _, _ = obj_func(
                    float(post_inf_reduced.mean()),
                    float(post_inf_reduced.var()),
                    m_base0,
                    v_base0,
                    alpha=self.alpha,
                )
            selected = select_signed_direction(
                obj_negative, obj_positive, grid
            )
            return (
                int(selected["signed_m"]),
                obj_negative,
                obj_positive,
            )
        if is_empirical_augmentation(self.sampling_method):
            grid, path = empirical_bootstrap_augmentation_path(
                y, self.m_grid, rng
            )
            obj_negative = np.empty(len(grid), dtype=float)
            obj_positive = np.empty(len(grid), dtype=float)
            for i, m in enumerate(grid):
                additions = path[int(m)]
                augmented = (
                    y
                    if int(m) == 0
                    else np.concatenate([y, additions])
                )
                _, _, post_base_augmented = self._posterior_mean_var(
                    augmented, self.prior_b
                )
                _, _, post_inf_augmented = self._posterior_mean_var(
                    augmented, self.prior_p
                )
                _, _, _, _, obj_negative[i], _, _ = obj_func(
                    m_base0,
                    v_base0,
                    float(post_inf_augmented.mean()),
                    float(post_inf_augmented.var()),
                    alpha=self.alpha,
                )
                _, _, _, _, obj_positive[i], _, _ = obj_func(
                    m_inf0,
                    v_inf0,
                    float(post_base_augmented.mean()),
                    float(post_base_augmented.var()),
                    alpha=self.alpha,
                )
            selected = select_signed_direction(
                obj_negative, obj_positive, grid
            )
            return (
                int(selected["signed_m"]),
                obj_negative,
                obj_positive,
            )

        M = int(np.max(self.m_grid))
        y_new_all_1 = self.model.posterior_predictive(post_base0, M, rng=rng)
        y_new_all_2 = self.model.posterior_predictive(post_inf0, M, rng=rng)

        obj1 = np.zeros(len(self.m_grid), dtype=float)
        obj2 = np.zeros(len(self.m_grid), dtype=float)

        for i, m in enumerate(self.m_grid):
            y_aug_1 = y.copy() if m == 0 else np.concatenate([y, y_new_all_1[:m]])
            post_inf_m = self.model.posterior(y_aug_1, self.prior_p)
            _, _, _, _, obj_forward, _, _ = obj_func(
                m_base0, v_base0, float(post_inf_m.mean()), float(post_inf_m.var()),
                alpha=self.alpha
            )
            obj1[i] = obj_forward

            y_aug_2 = y.copy() if m == 0 else np.concatenate([y, y_new_all_2[:m]])
            post_base_m = self.model.posterior(y_aug_2, self.prior_b)
            _, _, _, _, obj_backward, _, _ = obj_func(
                m_inf0, v_inf0, float(post_base_m.mean()), float(post_base_m.var()),
                alpha=self.alpha
            )
            obj2[i] = obj_backward

        obj_min = np.minimum(obj1, obj2)
        m_star_idx = int(np.argmin(obj_min))
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
            "sampling_method": self.sampling_method,
            "m_grid_semantics": (
                "number_deleted"
                if is_deletion_sampling(self.sampling_method)
                else "number_augmented"
            ),
        }


def _cp_normal_known_one_task(
    mu0,
    rep,
    n,
    true_mu,
    sigma,
    prior_sd,
    baseline_prior,
    m_grid,
    alpha,
    B_fit,
    data_seed,
    fit_seed_base,
):
    rng_rep_data = np.random.default_rng(data_seed + rep)
    y = rng_rep_data.normal(loc=true_mu, scale=sigma, size=n)
    rng_fit = np.random.default_rng(
        fit_seed_base + 10000 * (rep + 1) + int(round((mu0 + 10) * 1000))
    )

    model = NormalMeanKnownVarModel(sigma=sigma)
    prior_p = NormalDist(mu0, prior_sd)
    runner = CoverageProbabilityEPSSRunner1D(
        model=model,
        prior_p=prior_p,
        prior_b=baseline_prior,
        m_grid=m_grid,
        alpha=alpha,
        B_fit=B_fit,
    )
    out = runner.run_dataset(y, rng=rng_fit)

    return {
        "rep": int(rep),
        "prior_loc": float(mu0),
        "epss": float(out["epss_dataset"]),
        "mean_signed_mstar": float(np.mean(out["signed_mstars"])),
        "sd_signed_mstar": float(np.std(out["signed_mstars"], ddof=1)) if B_fit > 1 else 0.0,
        "model_type": "normal_known_cp",
        "sample": y.tolist(),
    }


def run_cp_epss_normal_known_grid_parallel(
    mu0_grid,
    R=50,
    rep_start=0,
    rep_end=None,
    n=20,
    true_mu=0.0,
    sigma=1.0,
    prior_sd=np.sqrt(0.2),
    baseline_prior=None,
    m_grid=None,
    alpha=0.05,
    B_fit=50,
    data_seed=321,
    fit_seed_base=20260312,
    n_jobs=-1,
    verbose=10,
):
    if baseline_prior is None:
        baseline_prior = FlatPrior()
    if m_grid is None:
        m_grid = np.arange(0, 31)
    if rep_end is None:
        rep_end = R

    mu0_grid = np.asarray(mu0_grid, dtype=float)
    tasks = [(mu0, rep) for rep in range(rep_start, rep_end) for mu0 in mu0_grid]
    results_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_cp_normal_known_one_task)(
            mu0=mu0,
            rep=rep,
            n=n,
            true_mu=true_mu,
            sigma=sigma,
            prior_sd=prior_sd,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            alpha=alpha,
            B_fit=B_fit,
            data_seed=data_seed,
            fit_seed_base=fit_seed_base,
        )
        for mu0, rep in tasks
    )

    results_df = pd.DataFrame(results_list)
    summary_df = (
        results_df.groupby("prior_loc")["epss"]
        .agg(mean_epss="mean", median_epss="median", sd_epss="std")
        .reset_index()
        .sort_values("prior_loc")
    )
    return results_df, summary_df


def cp_exact(P, Q, alpha=0.05):
    a, b = Q.interval(alpha=alpha)
    return float(P.prob_interval(a, b))


def obj_func_exact(P, Q, alpha=0.05):
    CP1 = cp_exact(P, Q, alpha=alpha)
    CP2 = cp_exact(Q, P, alpha=alpha)
    r1 = max(CP1, CP2)
    r2 = min(CP1, CP2)
    obj = (r1 - r2) ** 2 + (r2 - (1 - alpha)) ** 2
    return float(CP1), float(CP2), float(r1), float(r2), float(obj)


def _normalize_alpha_grid(alpha_grid, weights=None):
    alpha_grid = np.asarray(alpha_grid, dtype=float).ravel()
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must contain at least one alpha.")
    if np.any((alpha_grid <= 0.0) | (alpha_grid >= 1.0)):
        raise ValueError("All alpha values must lie in (0, 1).")
    if weights is None:
        weights = np.ones(len(alpha_grid), dtype=float) / len(alpha_grid)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if len(weights) != len(alpha_grid):
            raise ValueError("weights must have the same length as alpha_grid.")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("weights must have positive sum.")
        weights = weights / total
    return alpha_grid, weights


def obj_func_multi_alpha_from_components(cp_pairs, alpha_grid, weights=None):
    """
    Multi-alpha CP objective from pairs (CP1_alpha, CP2_alpha).

    The objective averages the original CP loss over alpha values:
        (r1_alpha - r2_alpha)^2 + (r2_alpha - (1-alpha))^2.
    """
    alpha_grid, weights = _normalize_alpha_grid(alpha_grid, weights)
    cp_pairs = np.asarray(cp_pairs, dtype=float)
    if cp_pairs.shape != (len(alpha_grid), 2):
        raise ValueError("cp_pairs must have shape (len(alpha_grid), 2).")
    cp1 = cp_pairs[:, 0]
    cp2 = cp_pairs[:, 1]
    r1 = np.maximum(cp1, cp2)
    r2 = np.minimum(cp1, cp2)
    losses = (r1 - r2) ** 2 + (r2 - (1.0 - alpha_grid)) ** 2
    obj = float(np.sum(weights * losses))
    return {
        "cp1": cp1,
        "cp2": cp2,
        "r1": r1,
        "r2": r2,
        "losses": losses,
        "obj": obj,
        "mean_cp1": float(np.sum(weights * cp1)),
        "mean_cp2": float(np.sum(weights * cp2)),
        "mean_r1": float(np.sum(weights * r1)),
        "mean_r2": float(np.sum(weights * r2)),
        "term_gap": float(np.sum(weights * (r1 - r2) ** 2)),
        "term_cov": float(np.sum(weights * (r2 - (1.0 - alpha_grid)) ** 2)),
    }


def obj_func_normal_multi_alpha(mA, vA, mB, vB, alpha_grid, weights=None):
    alpha_grid, weights = _normalize_alpha_grid(alpha_grid, weights)
    cp_pairs = []
    for alpha in alpha_grid:
        CP1, CP2, _, _, _, _, _ = obj_func(mA, vA, mB, vB, alpha=float(alpha))
        cp_pairs.append((CP1, CP2))
    return obj_func_multi_alpha_from_components(cp_pairs, alpha_grid, weights)


def obj_func_exact_multi_alpha(P, Q, alpha_grid, weights=None):
    alpha_grid, weights = _normalize_alpha_grid(alpha_grid, weights)
    cp_pairs = []
    for alpha in alpha_grid:
        cp_pairs.append((cp_exact(P, Q, alpha=float(alpha)), cp_exact(Q, P, alpha=float(alpha))))
    return obj_func_multi_alpha_from_components(cp_pairs, alpha_grid, weights)


class MultiAlphaCoverageProbabilityEPSSRunner1D:
    """
    Multi-alpha coverage-probability EPSS runner.

    This is a stabilized variant of CP. It keeps the same forward/backward
    simulation scheme as CoverageProbabilityEPSSRunner1D, but replaces the
    single-alpha objective by a weighted average over alpha_grid.
    """

    def __init__(self, model, prior_p, prior_b, m_grid, alpha_grid=None, weights=None, B_fit=50):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.m_grid = np.asarray(m_grid, dtype=int)
        self.alpha_grid, self.weights = _normalize_alpha_grid(
            [0.01, 0.05, 0.10, 0.20, 0.30] if alpha_grid is None else alpha_grid,
            weights=weights,
        )
        self.B_fit = int(B_fit)
        if np.any(self.m_grid < 0):
            raise ValueError("m_grid must be nonnegative.")
        if self.B_fit <= 0:
            raise ValueError("B_fit must be positive.")

    def _posterior(self, y, prior):
        return self.model.posterior(y, prior)

    def _objective(self, P, Q):
        if hasattr(P, "interval") and hasattr(P, "prob_interval") and hasattr(Q, "interval") and hasattr(Q, "prob_interval"):
            return obj_func_exact_multi_alpha(P, Q, self.alpha_grid, self.weights)
        return obj_func_normal_multi_alpha(
            float(P.mean()),
            float(P.var()),
            float(Q.mean()),
            float(Q.var()),
            self.alpha_grid,
            self.weights,
        )

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
            obj1[i] = self._objective(post_base0, post_inf_m)["obj"]

            y_aug_2 = y.copy() if m == 0 else np.concatenate([y, y_new_all_2[:m]])
            post_base_m = self._posterior(y_aug_2, self.prior_b)
            obj2[i] = self._objective(post_inf0, post_base_m)["obj"]

        idx = int(np.argmin(np.minimum(obj1, obj2)))
        sign = 1 if obj1[idx] > obj2[idx] else -1
        return int(self.m_grid[idx]) * sign, obj1, obj2

    def run_dataset(self, y, rng):
        signed_mstars = np.empty(self.B_fit, dtype=float)
        for b in range(self.B_fit):
            m_star, _, _ = self._one_fit(y, rng=rng)
            signed_mstars[b] = m_star
        return {
            "epss_dataset": float(np.mean(signed_mstars)),
            "signed_mstars": signed_mstars,
        }


class PosteriorPredictiveCoverageProbabilityEPSSRunner1D:
    """
    Posterior-predictive CP runner.

    predictive_fn must return a univariate distribution object with interval()
    and prob_interval() methods. Its signature is
        predictive_fn(model, y, prior, posterior)
    so both simple conjugate models and hierarchical models can be supported.
    """

    def __init__(self, model, prior_p, prior_b, m_grid, predictive_fn, alpha=0.05, B_fit=50):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.m_grid = np.asarray(m_grid, dtype=int)
        self.predictive_fn = predictive_fn
        self.alpha = float(alpha)
        self.B_fit = int(B_fit)
        if np.any(self.m_grid < 0):
            raise ValueError("m_grid must be nonnegative.")
        if self.B_fit <= 0:
            raise ValueError("B_fit must be positive.")

    def _posterior(self, y, prior):
        return self.model.posterior(y, prior)

    def _predictive(self, y, prior, posterior):
        return self.predictive_fn(self.model, y, prior, posterior)

    def _one_fit(self, y, rng):
        y = np.asarray(y, dtype=float).ravel()
        post_base0 = self._posterior(y, self.prior_b)
        post_inf0 = self._posterior(y, self.prior_p)
        ppd_base0 = self._predictive(y, self.prior_b, post_base0)
        ppd_inf0 = self._predictive(y, self.prior_p, post_inf0)
        M = int(np.max(self.m_grid))
        y_new_all_1 = self.model.posterior_predictive(post_base0, M, rng=rng)
        y_new_all_2 = self.model.posterior_predictive(post_inf0, M, rng=rng)
        obj1 = np.zeros(len(self.m_grid), dtype=float)
        obj2 = np.zeros(len(self.m_grid), dtype=float)

        for i, m in enumerate(self.m_grid):
            y_aug_1 = y.copy() if m == 0 else np.concatenate([y, y_new_all_1[:m]])
            post_inf_m = self._posterior(y_aug_1, self.prior_p)
            ppd_inf_m = self._predictive(y_aug_1, self.prior_p, post_inf_m)
            obj1[i] = obj_func_exact(ppd_base0, ppd_inf_m, alpha=self.alpha)[4]

            y_aug_2 = y.copy() if m == 0 else np.concatenate([y, y_new_all_2[:m]])
            post_base_m = self._posterior(y_aug_2, self.prior_b)
            ppd_base_m = self._predictive(y_aug_2, self.prior_b, post_base_m)
            obj2[i] = obj_func_exact(ppd_inf0, ppd_base_m, alpha=self.alpha)[4]

        idx = int(np.argmin(np.minimum(obj1, obj2)))
        sign = 1 if obj1[idx] > obj2[idx] else -1
        return int(self.m_grid[idx]) * sign, obj1, obj2

    def run_dataset(self, y, rng):
        signed_mstars = np.empty(self.B_fit, dtype=float)
        for b in range(self.B_fit):
            m_star, _, _ = self._one_fit(y, rng=rng)
            signed_mstars[b] = m_star
        return {
            "epss_dataset": float(np.mean(signed_mstars)),
            "signed_mstars": signed_mstars,
        }


class CoverageProbabilityExactPoissonGammaRunner:
    """
    Exact coverage-probability EPSS runner for Poisson-Gamma posteriors.
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
        if not isinstance(post, GammaDist):
            raise ValueError("Poisson-Gamma exact CP runner expects GammaDist posteriors.")
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
            "signed_mstars": signed_mstars
        }


def _exact_cp_poisson_gamma_one_task(
    lam0,
    rep,
    n,
    true_lambda,
    prior_strength,
    baseline_prior,
    m_grid,
    alpha,
    B_fit,
    data_seed,
    fit_seed_base,
):
    model = PoissonModel()
    alpha0 = float(lam0) * float(prior_strength)
    beta0 = float(prior_strength)
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError("Informative Gamma prior requires positive alpha and beta.")

    prior_p = GammaDist(alpha0, beta0)
    rng_rep_data = np.random.default_rng(data_seed + int(rep))
    y = rng_rep_data.poisson(lam=true_lambda, size=n).astype(float)
    rng_fit = np.random.default_rng(fit_seed_base + 10000 * (int(rep) + 1))

    runner = CoverageProbabilityExactPoissonGammaRunner(
        model=model,
        prior_p=prior_p,
        prior_b=baseline_prior,
        m_grid=m_grid,
        alpha=alpha,
        B_fit=B_fit,
    )
    out = runner.run_dataset(y, rng=rng_fit)

    return {
        "rep": int(rep),
        "prior_loc": float(lam0),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "epss": float(out["epss_dataset"]),
        "mean_signed_mstar": float(np.mean(out["signed_mstars"])),
        "sd_signed_mstar": float(np.std(out["signed_mstars"], ddof=1)) if B_fit > 1 else 0.0,
        "model_type": "poisson_exact_cp",
        "sample": y.tolist(),
        "signed_mstars": out["signed_mstars"].tolist(),
    }


def run_exact_cp_poisson_gamma_grid(
    lambda0_grid,
    R=50,
    rep_start=0,
    rep_end=None,
    n=20,
    true_lambda=3.0,
    prior_strength=4.0,
    baseline_prior=None,
    m_grid=None,
    alpha=0.05,
    B_fit=50,
    data_seed=321,
    fit_seed_base=20260312,
    n_jobs=1,
    verbose=10,
):
    if baseline_prior is None:
        baseline_prior = FlatPrior()
    if m_grid is None:
        m_grid = np.arange(0, 31)
    if rep_end is None:
        rep_end = R

    lambda0_grid = np.asarray(lambda0_grid, dtype=float)
    tasks = [(lam0, rep) for lam0 in lambda0_grid for rep in range(rep_start, rep_end)]
    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_exact_cp_poisson_gamma_one_task)(
            lam0=lam0,
            rep=rep,
            n=n,
            true_lambda=true_lambda,
            prior_strength=prior_strength,
            baseline_prior=baseline_prior,
            m_grid=m_grid,
            alpha=alpha,
            B_fit=B_fit,
            data_seed=data_seed,
            fit_seed_base=fit_seed_base,
        )
        for lam0, rep in tasks
    )

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df.groupby("prior_loc")["epss"]
        .agg(mean_epss="mean", median_epss="median", sd_epss="std")
        .reset_index()
        .sort_values("prior_loc")
    )
    return results_df, summary_df
