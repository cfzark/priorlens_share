"""Posterior comparisons, CP objectives and posterior risks."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from scipy import stats, integrate
from scipy.special import digamma, gammaln

from .core import DefinitionError, NumericalFailure, weighted_quantile


@dataclass(frozen=True)
class Wasserstein:
    order: int = 2
    grid_size: int = 2000
    projections: int = 32
    projection_rule: str = "random"
    squared: bool = False
    seed: int = 20260312
    integration_rule: str = "quantile"

    def __call__(self, p, q):
        if self.integration_rule not in {"quantile", "empirical_sort"}:
            raise DefinitionError("Unknown Wasserstein integration rule")
        empirical = self.integration_rule == "empirical_sort"
        if empirical and (len(p.values) != len(q.values) or not np.allclose(p.weights, 1/len(p.values)) or not np.allclose(q.weights, 1/len(q.values))):
            raise DefinitionError("empirical_sort requires equal-size, uniformly weighted posterior samples")
        if self.order not in (1, 2):
            raise DefinitionError("Wasserstein order must be 1 or 2")
        if not np.all(np.isfinite(p.covariance)) or not np.all(np.isfinite(q.covariance)):
            raise NumericalFailure("Requested Wasserstein comparison lacks verified finite moments")
        if p.values.shape[1] == 1:
            if empirical:
                value = float(np.mean(np.abs(np.sort(p.values[:,0])-np.sort(q.values[:,0]))**self.order)**(1/self.order))
            elif p.family == q.family == "norm" and self.order == 2:
                value = float(np.sqrt((p.mean[0] - q.mean[0]) ** 2 + (np.sqrt(p.covariance[0, 0]) - np.sqrt(q.covariance[0, 0])) ** 2))
            elif p.ppf and q.ppf:
                u = np.linspace(1e-6, 1 - 1e-6, self.grid_size)
                value = float(np.trapezoid(np.abs(p.quantile(u) - q.quantile(u)) ** self.order, u) ** (1 / self.order))
            else:
                u = (np.arange(256) + .5) / 256
                value = float(np.mean(np.abs(p.quantile(u) - q.quantile(u)) ** self.order) ** (1 / self.order))
        else:
            if self.projection_rule == "random":
                directions = np.random.default_rng(self.seed).normal(size=(self.projections, 2))
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            else:
                offset = .5 if self.projection_rule == "nig" else 0.
                angles = (np.arange(self.projections) + offset) * np.pi / self.projections
                directions = np.column_stack([np.cos(angles), np.sin(angles)])
            u = (np.arange(256) + .5) / 256
            values = []
            for d in directions:
                a = np.sort(p.values @ d) if empirical else weighted_quantile(p.values @ d, u, p.weights)
                b = np.sort(q.values @ d) if empirical else weighted_quantile(q.values @ d, u, q.weights)
                values.append(np.mean(np.abs(a - b) ** self.order))
            value = float(np.mean(values) ** (1 / self.order))
        return {"loss": value ** 2 if self.squared else value}


def density(measure, engine, normalized=False):
    if engine == "exact_density":
        function = measure.logpdf if normalized else (measure.logpdf or measure.log_kernel)
        if function is None:
            raise DefinitionError("Target marginal/joint normalized density is unavailable; choose a declared KDE variant")
        return function
    if engine != "kde":
        raise DefinitionError(f"Unknown density engine {engine}")
    kde = stats.gaussian_kde(measure.values.T, weights=measure.weights)
    return lambda x: kde.logpdf(np.asarray(x).reshape(-1, measure.values.shape[1]).T)


def coverage_terms(cp1, cp2, nominal=.95, family="historical", gap_weight=1., cap_nominal=False):
    """Elementwise loss; cap BOTH coverages before ALL terms when requested.

    Raw coverages remain available for diagnosis and absolute adequacy.
    """
    if family not in {"historical", "direct"}:
        raise DefinitionError("Unknown CP loss family")
    if not np.isfinite(gap_weight) or gap_weight < 0:
        raise DefinitionError("CP gap_weight must be finite and nonnegative")
    a, b = np.asarray(cp1), np.asarray(cp2)
    if not isinstance(cap_nominal, (bool, np.bool_)):
        raise DefinitionError("cap_nominal must be boolean")
    if cap_nominal:
        a, b = np.minimum(a, nominal), np.minimum(b, nominal)
    gap = (a - b) ** 2
    coverage = ((np.minimum(a, b) - nominal) ** 2 if family == "historical"
                else (a - nominal) ** 2 + (b - nominal) ** 2)
    return {"loss": gap_weight * gap + coverage, "gap": gap,
            "term_gap": gap_weight * gap, "term_coverage": coverage,
            "cp1_used": a, "cp2_used": b}


@dataclass(frozen=True)
class Coverage:
    alphas: tuple = (.05,)
    region: str = "equal_tail"
    gap_weight: float = 1.
    alpha_weights: tuple | None = None
    quantile_rule: str = "weighted_cdf"
    loss_family: str = "historical"
    cap_nominal: bool = False
    cache_prepared: bool = False
    _prepared: dict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)

    def __call__(self, p, q):
        if self.region not in {"equal_tail", "central", "kde", "exact_density", "pdf_hdr"}:
            raise DefinitionError("Unknown CP region")
        if p.values.shape[1] != q.values.shape[1]:
            raise DefinitionError("CP posterior dimensions differ")
        if self.quantile_rule not in {"weighted_cdf", "numpy_linear"}:
            raise DefinitionError("Unknown coverage quantile rule")
        if self.region == "pdf_hdr" and self.quantile_rule != "weighted_cdf":
            raise DefinitionError("pdf_hdr uses weighted density ranks, not numpy_linear interpolation")
        def quantile(values, probabilities, weights):
            if self.quantile_rule == "numpy_linear":
                if not np.allclose(weights, 1/len(weights)):
                    raise DefinitionError("numpy_linear requires uniformly weighted empirical samples")
                return np.quantile(values, probabilities)
            return weighted_quantile(values, probabilities, weights)
        cp1, cp2, region_diagnostics = [], [], []
        def prepare(measure):
            key = id(measure)
            if self.cache_prepared and key in self._prepared:
                self._prepared.move_to_end(key)
                return self._prepared[key][1]
            result = []
            lp = density(measure, self.region) if self.region in {"kde", "exact_density"} else None
            rank = (lambda z: np.exp(lp(z))) if self.region == "kde" else lp
            for alpha in self.alphas:
                if not 0 < alpha < 1:
                    raise DefinitionError("alpha must lie in (0,1)")
                if self.region == "central":
                    from .regions import central_region
                    result.append(central_region(measure, alpha, self.quantile_rule))
                elif self.region == "pdf_hdr":
                    from .density_hdr import density_hdr_region
                    result.append(density_hdr_region(measure, alpha))
                elif self.region == "equal_tail":
                    if measure.values.shape[1] != 1:
                        raise DefinitionError("Equal-tail CP is scalar")
                    probabilities = [alpha / 2, 1 - alpha / 2]
                    result.append(measure.quantile(probabilities) if measure.ppf or self.quantile_rule == "weighted_cdf" else quantile(measure.values[:, 0], probabilities, measure.weights))
                else:
                    result.append((rank, quantile(rank(measure.values), [alpha], measure.weights)[0]))
            if self.cache_prepared:
                # Holding the object prevents id reuse. Bound memory regardless
                # of the number of candidates; observed posteriors stay hot.
                self._prepared[key] = (measure, result)
                while len(self._prepared) > 16:
                    self._prepared.popitem(last=False)
            return result
        pp, qq = prepare(p), prepare(q)
        for i, alpha in enumerate(self.alphas):
            if not 0 < alpha < 1:
                raise DefinitionError("alpha must lie in (0,1)")
            if self.region in {"central", "pdf_hdr"}:
                rp, rq = pp[i], qq[i]
                cp1.append(rq.mass(p))
                cp2.append(rp.mass(q))
                region_diagnostics.append({"alpha": float(alpha), "p": rp.diagnostics(), "q": rq.diagnostics()})
            elif self.region == "equal_tail":
                if p.values.shape[1] != 1:
                    raise DefinitionError("Equal-tail CP is scalar; choose an explicit joint HDR engine")
                pa, pb = pp[i]
                qa, qb = qq[i]
                cp1.append(p.interval_mass(qa, qb))
                cp2.append(q.interval_mass(pa, pb))
            else:
                # Historical KDE interpolates density, while exact HDR uses log
                # density; interpolation and nonlinear transforms do not commute.
                (rankp, cutp), (rankq, cutq) = pp[i], qq[i]
                cp1.append(float(p.weights @ (rankq(p.values) >= cutq)))
                cp2.append(float(q.weights @ (rankp(q.values) >= cutp)))
        cp1, cp2 = np.asarray(cp1), np.asarray(cp2)
        gaps = (cp1 - cp2) ** 2
        nominal = 1 - np.asarray(self.alphas)
        terms = coverage_terms(cp1, cp2, nominal, self.loss_family, self.gap_weight, self.cap_nominal)
        w = np.ones(len(cp1)) if self.alpha_weights is None else np.asarray(self.alpha_weights)
        if w.shape != cp1.shape or np.any(w < 0) or w.sum() <= 0:
            raise DefinitionError("Invalid alpha weights")
        w = w / w.sum()
        result = {"loss": float(w @ terms["loss"]),
                "cp1": cp1.tolist(), "cp2": cp2.tolist(), "gap": float(w @ gaps),
                "cp1_used": terms["cp1_used"].tolist(), "cp2_used": terms["cp2_used"].tolist(),
                "term_gap": float(w @ terms["term_gap"]), "term_coverage": float(w @ terms["term_coverage"]),
                "adequacy": float(np.max(np.maximum(np.abs(cp1 - nominal), np.abs(cp2 - nominal))))}
        if self.region in {"central", "pdf_hdr"}:
            result["region_diagnostics"] = region_diagnostics
        return result


def kl(p, q, truncated=False, alpha=.05, density_engine="exact_density"):
    if not truncated and p.family == q.family and p.family in {"norm", "gamma", "beta"}:
        a, b = p.parameters["rv"], q.parameters["rv"]
        if p.family == "norm":
            return float(np.log(np.sqrt(q.covariance[0, 0] / p.covariance[0, 0])) + (p.covariance[0, 0] + (p.mean[0] - q.mean[0]) ** 2) / (2 * q.covariance[0, 0]) - .5)
        if p.parameters["coordinate"] != q.parameters["coordinate"]:
            raise DefinitionError("Exact KL requires common target coordinates")
        if p.family == "gamma":
            a1, a2 = a.args[0], b.args[0]
            b1, b2 = 1 / a.kwds.get("scale", 1), 1 / b.kwds.get("scale", 1)
            return float(a2 * np.log(b1 / b2) - gammaln(a1) + gammaln(a2) + (a1 - a2) * digamma(a1) + a1 * (b2 / b1 - 1))
        a1, b1 = a.args
        a2, b2 = b.args
        return float(gammaln(a2) + gammaln(b2) - gammaln(a2 + b2) - gammaln(a1) - gammaln(b1) + gammaln(a1 + b1) + (a1 - a2)*digamma(a1) + (b1-b2)*digamma(b1) - (a1+b1-a2-b2)*digamma(a1+b1))
    if not truncated and p.family == q.family == "nig":
        a, b = p.parameters, q.parameters
        value = gammaln(b["kappa"]) - gammaln(a["kappa"]) + b["kappa"] * np.log(a["eta"] / b["eta"]) + (a["kappa"] - b["kappa"]) * digamma(a["kappa"]) + a["kappa"] * (b["eta"] - a["eta"]) / a["eta"]
        value += .5 * (np.log(a["lambda"] / b["lambda"]) + b["lambda"] / a["lambda"] - 1 + (a["mu"] - b["mu"]) ** 2 * b["lambda"] * a["kappa"] / a["eta"])
        return float(value)
    # HDR membership only needs a kernel. KL (including conditional/TKL) needs
    # normalized densities because pa/qa below are normalized probability masses.
    lp, lq = density(p, density_engine, normalized=True), density(q, density_engine, normalized=True)
    if not truncated and p.values.shape[1] == 1 and p.ppf and q.ppf:
        # Integrate normalized scalar densities, including their tail mass.
        rv = p.parameters.get("rv")
        c = p.parameters.get("coordinate")
        if rv is not None and c is not None:
            lo, hi = rv.support()
            def f(x):
                z = np.array([[c.forward(x)]])
                return float(np.exp(rv.logpdf(x)) * (lp(z)[0] - lq(z)[0]))
            return float(integrate.quad(f, lo, hi, epsabs=1e-9, epsrel=1e-7, limit=200)[0])
    diff = lp(p.values) - lq(p.values)
    if not truncated:
        return float(p.weights @ diff)
    cutoff = weighted_quantile(lp(p.values), [alpha], p.weights)[0]
    inside = lp(p.values) >= cutoff
    pa = float(p.weights @ inside)
    qa = float(q.weights @ (lp(q.values) >= cutoff))
    if min(pa, qa) <= 0:
        raise NumericalFailure("Truncated KL region mass was not resolved")
    return float(p.weights @ (inside * diff) / pa + np.log(qa / pa))


def posterior_risk(p, reference, metric="posterior_mse"):
    return risk_from_moments(p.mean, np.diag(p.covariance), reference, metric)


def risk_from_moments(mean, variances, reference, metric="posterior_mse"):
    """Euclidean risk needs marginal variances, not a joint covariance matrix."""
    mean, variances = np.asarray(mean), np.asarray(variances)
    reference = np.asarray(reference).reshape(-1)
    if reference.shape != mean.shape or variances.shape != mean.shape:
        raise DefinitionError("Risk reference must be in the complete target coordinate system")
    bias2 = float(np.sum((mean - reference) ** 2))
    if metric == "squared_mean":
        value = bias2
    elif metric == "posterior_variance":
        value = float(np.sum(variances))
    elif metric == "posterior_mse":
        value = bias2 + float(np.sum(variances))
    else:
        raise DefinitionError(f"Unknown risk metric {metric}")
    if not np.isfinite(value):
        raise NumericalFailure("Posterior risk is infinite or unresolved")
    return value
