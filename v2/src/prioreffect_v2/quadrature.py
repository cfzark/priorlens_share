"""Deterministic reduced quadrature and weighted posterior representations."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .core import DefinitionError, NumericalFailure, Measure, Posterior, weighted_quantile


def trapezoid_weights(x):
    dx = np.diff(x)
    return np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]


class SchoolPosterior(Posterior):
    """p(eta|y) quadrature, p(mu|eta,y) Gaussian, theta_j marginal mixtures."""
    def __init__(self, model, data, prior, settings):
        super().__init__(model, data, prior, "collapsed_grid", {})
        self.cache_quantiles = settings.get("school_cache_quantiles", settings.get("school_engine") == "batched")
        self.quantile_engine = settings.get("school_quantile_engine", "brentq")
        self.y, self.s = np.asarray(data["y"]), np.asarray(data["s"])
        size = int(settings.get("school_grid_size", 801))
        tail_tol = settings.get("tail_tol", 1e-10)
        tolerance = settings.get("school_convergence_tol", 1e-5)
        center = np.log(max(1., prior["tau_scale"]))
        lo, hi = min(-12., center - 10), max(np.log(500.), center + 8)
        bounded = prior["tau_kind"] == "bounded_uniform"
        if bounded:
            hi = np.log(prior["tau_scale"])
        previous, converged = None, False
        for expansion in range(int(settings.get("max_expansions", 8)) + 1):
            eta = np.linspace(lo, hi, size)
            kernel = self.log_eta_kernel(eta)
            log_norm = logsumexp(kernel + np.log(trapezoid_weights(eta)))
            mass = np.exp(kernel + np.log(trapezoid_weights(eta)) - log_norm)
            mean, var = self.conditional_mu(eta)
            summary = np.r_[mass @ eta, mass @ (eta ** 2), mass @ mean, mass @ (var + mean ** 2), weighted_quantile(eta, [.025, .5, .975], mass)]
            edge = max(float(mass[:8].sum()), 0 if bounded else float(mass[-8:].sum()))
            delta = np.inf if previous is None else float(np.max(np.abs(summary - previous)))
            # Legacy grid's stopping convention retained for regression; delta is exposed.
            if edge < tail_tol and (delta < tolerance or expansion >= 1):
                converged = True
                break
            previous = summary
            width = hi - lo
            lo -= .35 * width
            if not bounded:
                hi += .35 * width
        if not converged:
            raise NumericalFailure(f"School grid tails not controlled: {edge}")
        self.install_grid(eta, mass, kernel, log_norm, mean, var, edge, expansion, delta)

    def install_grid(self, eta, mass, kernel, log_norm, mean, var, edge, expansion, delta):
        """Shared installation for scalar and per-row-equivalent batched grids."""
        size = len(eta)
        self.eta, self.masses, self.log_norm = eta, mass / mass.sum(), float(log_norm)
        self.log_eta_density = kernel - log_norm
        self.mu_cond_mean, self.mu_cond_var = mean, var
        self.density = np.exp(self.log_eta_density - np.max(self.log_eta_density))
        areas = .5 * (self.density[:-1] + self.density[1:]) * np.diff(eta)
        self.interval_prob = areas / areas.sum()
        self.interval_cdf = np.r_[0., np.cumsum(self.interval_prob)]
        self.interval_cdf[-1] = 1.
        self.accuracy = {"engine": self.engine, "converged": True, "grid_size": size,
                         "tail_mass": edge, "n_expansions": expansion, "moment_change": delta,
                         "stopping_rule": "legacy_tail_and_one_expansion", "log_normalizer": self.log_norm}

    def conditional_mu(self, eta):
        variance = self.s ** 2 + np.exp(2 * np.asarray(eta))[..., None]
        precision = 1 / self.prior["mu_sd"] ** 2 + (1 / variance).sum(axis=-1)
        var = 1 / precision
        mean = var * (self.prior["mu_mean"] / self.prior["mu_sd"] ** 2 + (self.y / variance).sum(axis=-1))
        return mean, var

    def log_eta_kernel(self, eta):
        eta = np.asarray(eta)
        variance = self.s ** 2 + np.exp(2 * eta)[..., None]
        precision = 1 / self.prior["mu_sd"] ** 2 + (1 / variance).sum(axis=-1)
        linear = self.prior["mu_mean"] / self.prior["mu_sd"] ** 2 + (self.y / variance).sum(axis=-1)
        quadratic = self.prior["mu_mean"] ** 2 / self.prior["mu_sd"] ** 2 + (self.y ** 2 / variance).sum(axis=-1)
        constant = -.5 * np.log(2 * np.pi * variance).sum(axis=-1) - .5 * np.log(2 * np.pi * self.prior["mu_sd"] ** 2)
        theta = np.column_stack([np.full(eta.size, self.prior["mu_mean"]), np.exp(eta.ravel())])
        log_eta_prior = self.model.logprior(theta, self.prior).reshape(eta.shape) - stats.norm.logpdf(self.prior["mu_mean"], self.prior["mu_mean"], self.prior["mu_sd"]) + eta
        return log_eta_prior + constant - .5 * quadratic + .5 * linear ** 2 / precision + .5 * np.log(2 * np.pi / precision)

    def eta_ppf(self, u):
        u = np.asarray(u)
        clipped = np.clip(u, 0, np.nextafter(1., 0.))
        ix = np.clip(np.searchsorted(self.interval_cdf, clipped, side="right") - 1, 0, len(self.eta) - 2)
        local = np.clip((clipped - self.interval_cdf[ix]) / np.maximum(self.interval_prob[ix], 1e-300), 0, 1)
        a, b = self.density[ix], self.density[ix + 1]
        denominator = a + np.sqrt(np.maximum(a ** 2 + (b - a) * local * (a + b), 0))
        fraction = np.divide(local * (a + b), denominator, out=local.copy(), where=denominator > 0)
        linear = np.abs(b - a) <= 1e-12 * np.maximum(a + b, 1e-300)
        fraction = np.where(linear, local, fraction)
        return self.eta[ix] + np.clip(fraction, 0, 1) * np.diff(self.eta)[ix]

    def sample(self, size, rng):
        eta = self.eta_ppf(rng.random(size))
        mean, var = self.conditional_mu(eta)
        return {"mu": rng.normal(mean, np.sqrt(var)), "tau": np.exp(eta)}

    def target(self, target, size=1024, seed=123):
        self.model.validate_target(target, self.data)
        if target.dimension == 2:
            if {c.variable for c in target.coordinates} != {"mu", "tau"}:
                raise DefinitionError("School joint target supports mu and tau")
            draws = self.sample(size, np.random.default_rng(seed))
            def lp(z):
                z = np.atleast_2d(z)
                raw = {c.variable: c.inverse(z[:, i]) for i, c in enumerate(target.coordinates)}
                theta = np.column_stack([raw["mu"], raw["tau"]])
                return self.model.logprior(theta, self.prior) + self.model.loglik(theta, self.data) - self.log_norm - sum(c.log_jacobian(raw[c.variable]) for c in target.coordinates)
            return Measure(target.values(draws), np.ones(size), logpdf=lp)
        c = target.coordinates[0]
        if c.variable == "tau":
            raw = np.exp(self.eta)
            values = c.forward(raw)
            mean = np.array([self.masses @ values])
            var = np.array([[self.masses @ ((values - mean[0]) ** 2)]])
            def lp(z):
                x = c.inverse(np.asarray(z).reshape(-1))
                return self.log_eta_kernel(np.log(x)) - self.log_norm - np.log(x) - c.log_jacobian(x)
            return Measure(values, self.masses, lp, lambda u: c.forward(np.exp(self.eta_ppf(u))), mean=mean, covariance=var)
        if c.transform != "identity":
            raise DefinitionError("School mu/theta marginals currently require affine coordinates")
        cm, cv = self.mu_cond_mean, self.mu_cond_var
        if c.variable.startswith("theta["):
            if not target.focal_unit_id:
                raise DefinitionError("School-specific target requires focal_unit_id")
            try:
                j = self.data["unit_ids"].index(target.focal_unit_id)
            except ValueError as e:
                raise DefinitionError("Focal original school is absent") from e
            v = np.exp(2 * self.eta)
            w = self.s[j] ** 2 / (self.s[j] ** 2 + v)
            cm, cv = w * cm + (1 - w) * self.y[j], v * w + w ** 2 * cv
        elif c.variable != "mu":
            raise DefinitionError(f"Unknown school target {c.variable}")
        cm, cv = cm / c.scale, cv / c.scale ** 2
        mean = float(self.masses @ cm)
        var = float(self.masses @ (cv + (cm - mean) ** 2))
        def lp(z):
            z = np.asarray(z).reshape(-1)
            return logsumexp(np.log(self.masses + 1e-300)[None, :] + stats.norm.logpdf(z[:, None], cm, np.sqrt(cv)), axis=1)
        def cdf(z):
            if getattr(self, "quantile_engine", "brentq") == "vectorized_newton":
                from scipy.special import ndtr
                return np.sum(self.masses * ndtr((np.asarray(z)[..., None] - cm) / np.sqrt(cv)), axis=-1)
            return np.sum(self.masses * stats.norm.cdf(np.asarray(z)[..., None], cm, np.sqrt(cv)), axis=-1)
        quantile_cache = {}
        def ppf(p):
            from scipy.optimize import brentq
            probabilities = np.atleast_1d(np.asarray(p, dtype=float))
            key = probabilities.tobytes()
            if getattr(self, "cache_quantiles", False) and key in quantile_cache:
                return quantile_cache[key].copy()
            if getattr(self, "quantile_engine", "brentq") == "vectorized_newton":
                values = mixture_quantiles(probabilities, cm, cv, self.masses)
            else:
                values = np.asarray([brentq(lambda v: float(cdf(v)) - u, float(np.min(cm - 12 * np.sqrt(cv))), float(np.max(cm + 12 * np.sqrt(cv)))) for u in probabilities])
            if getattr(self, "cache_quantiles", False):
                if len(quantile_cache) >= 4:
                    quantile_cache.pop(next(iter(quantile_cache)))
                quantile_cache[key] = values.copy()
            return values
        # Deterministic mixture integration support, with analytic moments and density.
        nodes, weight = hermgauss(15)
        values = (cm[:, None] + np.sqrt(2 * cv[:, None]) * nodes).ravel()
        mass = (self.masses[:, None] * weight / np.sqrt(np.pi)).ravel()
        return Measure(values, mass, lp, ppf, cdf, np.array([mean]), np.array([[var]]))


def mixture_quantiles(probabilities, means, variances, weights):
    """Safeguarded vector Newton roots of the SAME Gaussian-mixture CDF.

    Bounded chunks; no interpolation, support thinning, or RNG changes.
    Unresolved roots fall back to the reference Brent solver.
    """
    from scipy.special import ndtr, ndtri
    from scipy.optimize import brentq
    sd = np.sqrt(variances)
    left, right = float(np.min(means - 12 * sd)), float(np.max(means + 12 * sd))
    mean = weights @ means
    sigma = np.sqrt(weights @ (variances + (means - mean) ** 2))
    output = np.empty(len(probabilities))
    for start in range(0, len(probabilities), 128):
        p = probabilities[start:start + 128]
        if np.any((p <= 0) | (p >= 1)):
            raise DefinitionError("Mixture quantiles require probabilities strictly between zero and one")
        lo, hi = np.full(len(p), left), np.full(len(p), right)
        x = np.clip(mean + sigma * ndtri(p), left, right)
        active = np.arange(len(p))
        for _ in range(60):
            if not len(active):
                break
            z = (x[active, None] - means) / sd
            residual = np.sum(ndtr(z) * weights, axis=1) - p[active]
            density = np.sum(np.exp(-.5 * z ** 2) * (weights / (sd * np.sqrt(2 * np.pi))), axis=1)
            step = np.divide(residual, density, out=np.full(len(active), np.inf), where=density > 0)
            done = np.abs(step) <= 2e-12 * np.maximum(1., np.abs(x[active]))
            lower = residual < 0
            lo[active[lower]] = x[active[lower]]
            hi[active[~lower]] = x[active[~lower]]
            proposed = x[active] - step
            valid = np.isfinite(proposed) & (proposed > lo[active]) & (proposed < hi[active])
            update = np.where(valid, proposed, (lo[active] + hi[active]) / 2)
            x[active[~done]] = update[~done]
            active = active[~done]
        for i in active:
            x[i] = brentq(lambda v: float(np.sum(weights * ndtr((v-means)/sd))) - p[i], left, right)
        output[start:start + len(p)] = x
    return output


class WeightedPosterior(Posterior):
    def __init__(self, model, data, prior, points, weights, accuracy, log_normalizer=None):
        super().__init__(model, data, prior, accuracy["engine"], accuracy)
        self.points = np.asarray(points)
        self.weights = np.asarray(weights) / np.sum(weights)
        self.log_normalizer = log_normalizer

    def sample(self, size, rng):
        indices = rng.choice(len(self.points), size=size, p=self.weights)
        return {v: self.points[indices, i] for i, v in enumerate(self.model.variables)}

    def target(self, target, size=1024, seed=123):
        if hasattr(self.model, "validate_target"):
            self.model.validate_target(target, self.data)
        draws = {v: self.points[:, i] for i, v in enumerate(self.model.variables)}
        if self.model.name == "lupus_logistic" and any(c.variable not in self.model.variables for c in target.coordinates):
            center, divisor = self.data["x_center"], self.data["x_divisor"]
            for x in [0., .5, 1., 1.5, 2.]:
                draws[f"p_x_{x:g}"] = expit(draws["beta0"] + draws["beta1"] * ((x - center) / divisor))
            draws["risk_difference"] = draws["p_x_2"] - draws["p_x_0"]
        if self.model.name == "eight_schools" and any(c.variable.startswith("theta[") for c in target.coordinates):
            if target.dimension != 1 or not target.focal_unit_id:
                raise DefinitionError("Focal school marginal requires one coordinate and focal ID")
            c = target.coordinates[0]
            if c.transform != "identity":
                raise DefinitionError("Focal marginal requires affine coordinate")
            j = self.data["unit_ids"].index(target.focal_unit_id)
            t2 = draws["tau"] ** 2
            s2 = self.data["s"][j] ** 2
            var = 1 / (1 / t2 + 1 / s2)
            mean = var * (draws["mu"] / t2 + self.data["y"][j] / s2) / c.scale
            var /= c.scale ** 2
            nodes, weights = hermgauss(9)
            values = (mean[:, None] + np.sqrt(2 * var[:, None]) * nodes).ravel()
            mass = (self.weights[:, None] * weights / np.sqrt(np.pi)).ravel()
            def lp(z):
                z = np.asarray(z).reshape(-1)
                return np.array([logsumexp(np.log(self.weights + 1e-300) + stats.norm.logpdf(v, mean, np.sqrt(var))) for v in z])
            return Measure(values, mass, logpdf=lp)
        values = target.values(draws)
        lp = kernel = None
        if set(c.variable for c in target.coordinates) == set(self.model.variables):
            def kernel(z):
                z = np.atleast_2d(z)
                raw = {c.variable: c.inverse(z[:, i]) for i, c in enumerate(target.coordinates)}
                theta = np.column_stack([raw[v] for v in self.model.variables])
                return self.model.loglik(theta, self.data) + self.model.logprior(theta, self.prior) - sum(c.log_jacobian(raw[c.variable]) for c in target.coordinates)
            if self.log_normalizer is not None:
                lp = lambda z: kernel(z) - self.log_normalizer
        return Measure(values, self.weights, logpdf=lp, log_kernel=kernel)


def smoothed_quantile(values, weights):
    mean = np.sum(values * weights)
    sd = np.sqrt(max(np.sum(weights * (values - mean) ** 2), 1e-14))
    points = mean + np.sqrt(1 - .4 ** 2) * (values - mean)
    bandwidth = max(.4 * sd, 1e-8)
    result = []
    for p in [.025, .5, .975]:
        lo, hi = float(points.min() - 8 * bandwidth), float(points.max() + 8 * bandwidth)
        for _ in range(45):
            mid = (lo + hi) / 2
            if np.sum(weights * stats.norm.cdf((mid - points) / bandwidth)) < p:
                lo = mid
            else:
                hi = mid
        result.append((lo + hi) / 2)
    return np.array(result)


def logistic_mode(model, data, prior):
    x, y = np.asarray(data["x"]), np.asarray(data["y"])
    X = np.column_stack([np.ones(len(x)), x])
    loc, scale = np.asarray(prior["loc"]), np.asarray(prior["scale"])
    df = prior.get("df", 7.)
    def gradient(beta):
        delta = beta - loc
        return X.T @ (y - expit(X @ beta)) - (df + 1) * delta / (df * scale ** 2 + delta ** 2)
    def objective(beta):
        return -float(model.loglik(beta, data)[0] + model.logprior(beta, prior)[0])
    fit = minimize(objective, loc.copy(), jac=lambda b: -gradient(b), method="BFGS")
    if not fit.success:
        fit = minimize(objective, loc.copy(), method="Nelder-Mead")
    p = expit(X @ fit.x)
    delta2 = (fit.x - loc) ** 2
    h = (X.T * (p * (1 - p))) @ X + np.diag((df + 1) * (df * scale ** 2 - delta2) / (df * scale ** 2 + delta2) ** 2)
    eigen, vectors = np.linalg.eigh((h + h.T) / 2)
    covariance = (vectors / np.maximum(eigen, 1e-7)) @ vectors.T
    if not fit.success:
        raise NumericalFailure(f"Logistic mode failed: {fit.message}")
    return fit.x, covariance


def logistic_posterior(model, data, prior, settings, seed):
    mode, covariance = logistic_mode(model, data, prior)
    previous, log_previous = None, None
    error = np.inf
    for order in settings.get("gh_nodes", [21, 31, 41]):
        roots, raw = hermgauss(order)
        a, b = np.meshgrid(roots, roots, indexing="ij")
        wa, wb = np.meshgrid(raw, raw, indexing="ij")
        points = mode + (np.column_stack([a.ravel(), b.ravel()]) * np.sqrt(2)) @ np.linalg.cholesky(covariance).T
        log_terms = np.log((wa * wb).ravel()) - np.log(np.pi) + model.logkernel(points, data, prior) - stats.multivariate_normal.logpdf(points, mode, covariance)
        log_norm = float(logsumexp(log_terms))
        mass = np.exp(log_terms - log_norm)
        mean = np.sum(points * mass[:, None], axis=0)
        delta = points - mean
        cov = (delta.T * mass) @ delta
        quantiles = np.column_stack([smoothed_quantile(points[:, j], mass) for j in range(2)]).reshape(-1)
        summary = np.r_[mean, cov.reshape(-1), quantiles]
        ess = float(1 / np.sum(mass ** 2))
        if previous is not None:
            error = max(float(np.max(np.abs(summary - previous) / np.maximum(1., np.abs(previous)))), abs(log_norm - log_previous) / max(1., abs(log_previous)))
            if error <= settings.get("quadrature_tolerance", 1e-3) and ess >= 25 and mass.max() <= .075:
                return WeightedPosterior(model, data, prior, points, mass,
                    {"engine": "gauss_hermite", "converged": True, "nodes_per_axis": order,
                     "n_points": len(points), "importance_ess": ess, "max_weight": float(mass.max()),
                     "max_relative_change": error, "fallback_used": False}, log_norm)
        previous, log_previous = summary, log_norm
    from .inference import importance_posterior
    post = importance_posterior(model, data, prior, settings, seed, mode=mode, covariance=covariance)
    post.accuracy.update(fallback_used=True, fallback_reason="quadrature_not_converged", last_gh_error=error)
    return post
