"""Numerical building blocks for the lupus logistic EPSS experiment.

The implementation is deliberately independent of the conjugate-model code.
It represents the observed data as grouped Bernoulli cells, but empirical
resampling always operates on expanded patient-level ``(x, y)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import lgamma

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import chi2, gaussian_kde, norm, qmc, t


BASE_SEED = 20260312
RAW_X_LEVELS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
CONTROL_COUNTS = np.asarray([31, 2, 2, 0, 2], dtype=int)
CASE_COUNTS = np.asarray([5, 0, 2, 6, 5], dtype=int)
LUPUS_SUMMARY_ENGINE = "batched_q6_v1"


def lupus_standardized_levels():
    """Return the five standardized predictor levels in canonical order."""
    data = lupus_patient_data()
    return (RAW_X_LEVELS - data["x_mean"]) / (2.0 * data["x_sd"])


def grouped_binomial_counts(x_std, y, levels=None):
    """Compress patient rows to binomial totals/cases at the five levels.

    All lupus pseudo-data generators preserve the empirical predictor support,
    so these counts are sufficient for the logistic likelihood.  Rejecting an
    off-grid predictor is intentional: silently rounding it would change the
    likelihood.
    """
    x_std, y = _validate_xy(x_std, y)
    levels = (
        lupus_standardized_levels()
        if levels is None
        else np.asarray(levels, dtype=float).reshape(-1)
    )
    totals = np.zeros(len(levels), dtype=np.int64)
    cases = np.zeros(len(levels), dtype=np.int64)
    for value, outcome in zip(x_std, y):
        distances = np.abs(levels - value)
        index = int(np.argmin(distances))
        if distances[index] > 1e-12:
            raise ValueError(
                f"Predictor value {value!r} is not on the lupus five-level support."
            )
        totals[index] += 1
        cases[index] += int(outcome)
    return levels, totals, cases


def _validate_grouped_counts(levels, totals, cases):
    levels = np.asarray(levels, dtype=float).reshape(-1)
    totals = np.asarray(totals, dtype=np.int64).reshape(-1)
    cases = np.asarray(cases, dtype=np.int64).reshape(-1)
    if not (len(levels) == len(totals) == len(cases)):
        raise ValueError("levels, totals, and cases must have the same length.")
    if np.any(totals < 0) or np.any(cases < 0) or np.any(cases > totals):
        raise ValueError("Grouped binomial counts must satisfy 0 <= cases <= totals.")
    return levels, totals, cases


def lupus_patient_data():
    """Return the reconstructed 55-patient lupus data."""
    x = np.repeat(RAW_X_LEVELS, CONTROL_COUNTS + CASE_COUNTS)
    y_parts = []
    for n0, n1 in zip(CONTROL_COUNTS, CASE_COUNTS):
        y_parts.append(np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    y = np.concatenate(y_parts)
    x_mean = float(np.mean(x))
    x_sd = float(np.std(x, ddof=1))
    x_std = (x - x_mean) / (2.0 * x_sd)
    return {
        "x": x,
        "y": y,
        "x_std": x_std,
        "x_mean": x_mean,
        "x_sd": x_sd,
    }


def lupus_grouped_cells():
    """Return only the eight observed ``(x, y)`` types and their counts."""
    data = lupus_patient_data()
    cells = []
    for x_raw, n0, n1 in zip(RAW_X_LEVELS, CONTROL_COUNTS, CASE_COUNTS):
        x_std = (x_raw - data["x_mean"]) / (2.0 * data["x_sd"])
        if n0:
            cells.append((float(x_raw), float(x_std), 0, int(n0)))
        if n1:
            cells.append((float(x_raw), float(x_std), 1, int(n1)))
    return cells


def lupus_all_model_cells():
    """Return all ten possible ``(x, y)`` types on the lupus design support."""
    data = lupus_patient_data()
    cells = []
    for x_raw in RAW_X_LEVELS:
        x_std = (x_raw - data["x_mean"]) / (2.0 * data["x_sd"])
        for outcome in (0, 1):
            cells.append((float(x_raw), float(x_std), outcome, 0))
    return cells


LUPUS_MLE_BETA0 = -0.8839755753332119
LUPUS_MLE_BETA1 = 2.8569763872486007
LUPUS_MLE_SE_BETA0 = 0.36324046
LUPUS_MLE_SE_BETA1 = 0.74494720


def prior_grid(prior_design):
    """Construct the exact symmetric grids specified for the experiment."""
    design = str(prior_design)
    if design == "reference":
        rows = [(0.0, 0.0, s) for s in (2.5, 5.0, 10.0)]
    elif design == "location":
        beta0 = np.arange(-3.0, 3.0 + 0.25, 0.5)
        beta1 = np.arange(-5.0, 5.0 + 0.25, 0.5)
        rows = [(a, b, 2.5) for a in beta0 for b in beta1]
    elif design == "location_upper":
        # Independent real-data boundary sensitivity.  The completed location
        # root already contains beta1=5, so start at 5.5 and extend the joined
        # grid to 11 without recomputing the anchor under different grid-index
        # seeds.  Joined with beta1=-5:5, the range is symmetric about 3,
        # close to both the lupus MLE and diffuse-baseline posterior slope.
        beta0 = np.arange(-3.0, 3.0 + 0.25, 0.5)
        beta1 = np.arange(5.5, 11.0 + 0.25, 0.5)
        rows = [(a, b, 2.5) for a in beta0 for b in beta1]
    elif design == "location_mle_slice":
        # Two exact one-dimensional cuts through the unpenalized lupus MLE.
        # Integer offsets avoid floating-point drift at the shared center.
        # The ranges span about +/-1.6 likelihood standard errors while the
        # fine steps are about 0.14 standard errors in either coordinate.
        slice_rows = []
        for step in range(-12, 13):
            offset = 0.05 * step
            slice_rows.append(
                {
                    "mu_beta0": LUPUS_MLE_BETA0 + offset,
                    "mu_beta1": LUPUS_MLE_BETA1,
                    "s_beta1": 2.5,
                    "slice_axis": "center" if step == 0 else "beta0",
                    "slice_offset": offset,
                }
            )
        for step in range(-12, 13):
            if step == 0:
                continue
            offset = 0.1 * step
            slice_rows.append(
                {
                    "mu_beta0": LUPUS_MLE_BETA0,
                    "mu_beta1": LUPUS_MLE_BETA1 + offset,
                    "s_beta1": 2.5,
                    "slice_axis": "beta1",
                    "slice_offset": offset,
                }
            )
        return [
            {"grid_index": index, **row}
            for index, row in enumerate(slice_rows)
        ]
    elif design == "location_strength":
        beta1 = np.arange(-5.0, 5.0 + 0.25, 0.5)
        rows = [(0.0, b, s) for b in beta1 for s in (0.5, 1.0, 2.5, 5.0, 10.0, 20.0)]
    else:
        raise ValueError(
            "prior_design must be reference, location, location_upper, "
            "location_mle_slice, or location_strength."
        )
    return [
        {
            "grid_index": i,
            "mu_beta0": float(a),
            "mu_beta1": float(b),
            "s_beta1": float(s),
        }
        for i, (a, b, s) in enumerate(rows)
    ]


@dataclass(frozen=True)
class IndependentTPrior:
    """Independent Student-t prior on standardized logistic coefficients."""

    loc: np.ndarray
    scale: np.ndarray
    df: float = 7.0
    label: str = "informative"

    def __post_init__(self):
        object.__setattr__(self, "loc", np.asarray(self.loc, dtype=float))
        object.__setattr__(self, "scale", np.asarray(self.scale, dtype=float))
        if self.loc.shape != (2,) or self.scale.shape != (2,):
            raise ValueError("loc and scale must be length-two vectors.")
        if np.any(self.scale <= 0) or self.df <= 0:
            raise ValueError("Student-t scales and degrees of freedom must be positive.")

    @property
    def has_finite_mean(self):
        return self.df > 1.0

    @property
    def has_finite_variance(self):
        return self.df > 2.0

    def logpdf(self, beta):
        beta = np.asarray(beta, dtype=float)
        return np.sum(
            t.logpdf(beta, df=self.df, loc=self.loc, scale=self.scale),
            axis=-1,
        )

    def gradient(self, beta):
        beta = np.asarray(beta, dtype=float)
        z = beta - self.loc
        return -(self.df + 1.0) * z / (
            self.df * np.square(self.scale) + np.square(z)
        )

    def negative_hessian_diag(self, beta):
        beta = np.asarray(beta, dtype=float)
        z2 = np.square(beta - self.loc)
        a = self.df * np.square(self.scale)
        return (self.df + 1.0) * (a - z2) / np.square(a + z2)

    def covariance(self):
        if not self.has_finite_variance:
            raise ValueError("This Student-t prior has no finite covariance.")
        return np.diag(np.square(self.scale) * self.df / (self.df - 2.0))

    def sample(self, size, rng):
        return self.loc + self.scale * rng.standard_t(self.df, size=(int(size), 2))


def make_prior(mu_beta0, mu_beta1, slope_scale, family="t7", label="informative"):
    family = str(family).lower()
    if family == "t7":
        df = 7.0
    elif family == "cauchy":
        df = 1.0
    else:
        raise ValueError("prior_family must be t7 or cauchy.")
    return IndependentTPrior(
        loc=np.asarray([mu_beta0, mu_beta1]),
        scale=np.asarray([10.0, slope_scale]),
        df=df,
        label=label,
    )


def make_baseline(scale=1000.0, family="t7"):
    return make_prior(0.0, 0.0, float(scale), family=family, label="baseline")


def epsilon_prior(prior, variance_multiplier=1000.0):
    """Student-t epsilon prior preserving location and inflating scales."""
    return IndependentTPrior(
        loc=prior.loc.copy(),
        scale=prior.scale * np.sqrt(float(variance_multiplier)),
        df=prior.df,
        label="epsilon",
    )


def _validate_xy(x_std, y):
    x_std = np.asarray(x_std, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x_std) != len(y):
        raise ValueError("x_std and y must have the same length.")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("y must be binary.")
    return x_std, y


def logistic_loglik(beta, x_std, y):
    x_std, y = _validate_xy(x_std, y)
    beta = np.asarray(beta, dtype=float)
    eta = beta[..., 0, None] + beta[..., 1, None] * x_std
    return np.sum(y * eta - np.logaddexp(0.0, eta), axis=-1)


def logistic_gradient_hessian(beta, x_std, y):
    x_std, y = _validate_xy(x_std, y)
    eta = beta[0] + beta[1] * x_std
    p = expit(eta)
    z = np.column_stack([np.ones_like(x_std), x_std])
    gradient = z.T @ (y - p)
    hessian = -(z.T * (p * (1.0 - p))) @ z
    return gradient, hessian


def logistic_loglik_counts(beta, levels, totals, cases):
    """Logistic log likelihood evaluated from five grouped binomial cells."""
    levels, totals, cases = _validate_grouped_counts(levels, totals, cases)
    beta = np.asarray(beta, dtype=float)
    eta = beta[..., 0, None] + beta[..., 1, None] * levels
    return np.sum(cases * eta - totals * np.logaddexp(0.0, eta), axis=-1)


def logistic_gradient_hessian_counts(beta, levels, totals, cases):
    """Gradient/Hessian of the grouped logistic log likelihood."""
    levels, totals, cases = _validate_grouped_counts(levels, totals, cases)
    beta = np.asarray(beta, dtype=float).reshape(2)
    eta = beta[0] + beta[1] * levels
    probability = expit(eta)
    residual = cases - totals * probability
    weight = totals * probability * (1.0 - probability)
    gradient = np.asarray(
        [np.sum(residual), np.sum(residual * levels)], dtype=float
    )
    hessian = -np.asarray(
        [
            [np.sum(weight), np.sum(weight * levels)],
            [np.sum(weight * levels), np.sum(weight * np.square(levels))],
        ],
        dtype=float,
    )
    return gradient, hessian


def log_posterior(beta, x_std, y, prior):
    return logistic_loglik(beta, x_std, y) + prior.logpdf(beta)


def log_posterior_counts(beta, levels, totals, cases, prior):
    return logistic_loglik_counts(beta, levels, totals, cases) + prior.logpdf(beta)


def posterior_mode_covariance(x_std, y, prior, start=None):
    x_std, y = _validate_xy(x_std, y)
    start = prior.loc.copy() if start is None else np.asarray(start, dtype=float)

    def objective(beta):
        return -float(log_posterior(beta, x_std, y, prior))

    def jac(beta):
        grad, _ = logistic_gradient_hessian(beta, x_std, y)
        return -(grad + prior.gradient(beta))

    fit = minimize(objective, start, jac=jac, method="BFGS")
    if not fit.success:
        fit = minimize(objective, start, method="Nelder-Mead")
    mode = np.asarray(fit.x, dtype=float)
    _, ll_hess = logistic_gradient_hessian(mode, x_std, y)
    neg_hess = -ll_hess + np.diag(prior.negative_hessian_diag(mode))
    eigval, eigvec = np.linalg.eigh(0.5 * (neg_hess + neg_hess.T))
    eigval = np.maximum(eigval, 1e-7)
    covariance = (eigvec / eigval) @ eigvec.T
    return mode, covariance, bool(fit.success), str(fit.message)


def _load_lupus_fast_backend():
    """Import the optional compiled backend without making it a dependency."""
    try:
        from . import _lupus_fast
    except ImportError:
        return None
    return _lupus_fast


def lupus_fast_backend_available():
    return _load_lupus_fast_backend() is not None


def posterior_mode_covariance_counts(
    levels,
    totals,
    cases,
    prior,
    start=None,
    mode_engine="scipy",
):
    """Posterior mode/covariance using grouped sufficient statistics.

    ``mode_engine='cpp'`` uses the optional batched Newton kernel.  A failed
    Newton solve falls back to the SciPy reference optimizer, so enabling the
    compiled backend cannot silently return a failed mode.
    """
    levels, totals, cases = _validate_grouped_counts(levels, totals, cases)
    start = prior.loc.copy() if start is None else np.asarray(start, dtype=float)
    engine = str(mode_engine).lower()
    if engine not in {"scipy", "cpp", "auto"}:
        raise ValueError("mode_engine must be scipy, cpp, or auto.")
    backend = _load_lupus_fast_backend() if engine in {"cpp", "auto"} else None
    if engine == "cpp" and backend is None:
        raise RuntimeError(
            "The C++ lupus backend is not built. Run scripts/build_lupus_fast_backend.sh."
        )
    if backend is not None:
        output = backend.batch_newton_modes(
            levels,
            totals.reshape(1, -1),
            cases.reshape(1, -1),
            prior.loc.reshape(1, 2),
            prior.scale.reshape(1, 2),
            np.asarray([prior.df], dtype=float),
            start.reshape(1, 2),
            60,
            1e-10,
        )
        mode = np.asarray(output["modes"][0], dtype=float)
        cpp_ok = bool(output["converged"][0]) and np.all(np.isfinite(mode))
        if cpp_ok:
            _, ll_hess = logistic_gradient_hessian_counts(
                mode, levels, totals, cases
            )
            neg_hess = -ll_hess + np.diag(prior.negative_hessian_diag(mode))
            eigval, eigvec = np.linalg.eigh(0.5 * (neg_hess + neg_hess.T))
            eigval = np.maximum(eigval, 1e-7)
            covariance = (eigvec / eigval) @ eigvec.T
            return (
                mode,
                covariance,
                True,
                f"cpp_newton:{int(output['iterations'][0])}",
            )

    def objective(beta):
        return -float(log_posterior_counts(beta, levels, totals, cases, prior))

    def jac(beta):
        grad, _ = logistic_gradient_hessian_counts(
            beta, levels, totals, cases
        )
        return -(grad + prior.gradient(beta))

    fit = minimize(objective, start, jac=jac, method="BFGS")
    if not fit.success:
        fit = minimize(objective, start, method="Nelder-Mead")
    mode = np.asarray(fit.x, dtype=float)
    _, ll_hess = logistic_gradient_hessian_counts(mode, levels, totals, cases)
    neg_hess = -ll_hess + np.diag(prior.negative_hessian_diag(mode))
    eigval, eigvec = np.linalg.eigh(0.5 * (neg_hess + neg_hess.T))
    eigval = np.maximum(eigval, 1e-7)
    covariance = (eigvec / eigval) @ eigvec.T
    message = str(fit.message)
    if backend is not None:
        message = "cpp_newton_fallback:" + message
    return mode, covariance, bool(fit.success), message


def _normal_logpdf(points, mean, covariance):
    points = np.atleast_2d(points)
    delta = points - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Proposal covariance must be positive definite.")
    solved = np.linalg.solve(covariance, delta.T).T
    return -0.5 * (
        2.0 * np.log(2.0 * np.pi)
        + logdet
        + np.sum(delta * solved, axis=1)
    )


def _student_logpdf_2d(points, mean, covariance, df=5.0):
    points = np.atleast_2d(points)
    delta = points - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Proposal covariance must be positive definite.")
    q = np.sum(delta * np.linalg.solve(covariance, delta.T).T, axis=1)
    return (
        lgamma((df + 2.0) / 2.0)
        - lgamma(df / 2.0)
        - np.log(df * np.pi)
        - 0.5 * logdet
        - 0.5 * (df + 2.0) * np.log1p(q / df)
    )


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def _smoothed_weighted_quantile(values, quantiles, weights):
    """Quantiles of a lightly Gaussian-smoothed weighted empirical CDF.

    Gauss-Hermite nodes are excellent for smooth moments but raw empirical
    quantiles jump when the node order changes.  Smoothing only the numerical
    convergence diagnostic avoids treating that quadrature artifact as a
    posterior-accuracy failure.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mean = float(np.sum(values * weights))
    variance = float(np.sum(weights * np.square(values - mean)))
    # A moment-preserving Gaussian smooth makes the CDF diagnostic stable
    # across Hermite orders: shrink the node spread by sqrt(1-h^2), then add
    # N(0, h^2 Var) noise.  The resulting mixture retains the original mean
    # and variance instead of artificially widening the posterior.
    smooth_fraction = 0.4
    sd = np.sqrt(max(variance, 1e-14))
    smoothed_values = mean + np.sqrt(1.0 - smooth_fraction**2) * (values - mean)
    bandwidth = max(smooth_fraction * sd, 1e-8)
    lower = float(np.min(smoothed_values) - 8.0 * bandwidth)
    upper = float(np.max(smoothed_values) + 8.0 * bandwidth)
    answer = []
    for probability in np.asarray(quantiles, dtype=float):
        lo, hi = lower, upper
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            cdf = float(
                np.sum(weights * norm.cdf((mid - smoothed_values) / bandwidth))
            )
            if cdf < probability:
                lo = mid
            else:
                hi = mid
        answer.append(0.5 * (lo + hi))
    return np.asarray(answer)


def _summary_vector(points, weights):
    mean = np.sum(points * weights[:, None], axis=0)
    delta = points - mean
    covariance = (delta.T * weights) @ delta
    q = np.column_stack(
        [
            _smoothed_weighted_quantile(
                points[:, j], [0.025, 0.5, 0.975], weights
            )
            for j in range(2)
        ]
    ).reshape(-1)
    return np.r_[mean, covariance.reshape(-1), q]


def _smoothed_quantile_components(values, weights):
    """Exactly the reference diagnostic's moment-preserving Gaussian smooth."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mean = float(np.sum(values * weights))
    variance = float(np.sum(weights * np.square(values - mean)))
    sd = np.sqrt(max(variance, 1e-14))
    smooth = mean + np.sqrt(1.0 - 0.4**2) * (values - mean)
    bandwidth = max(0.4 * sd, 1e-8)
    return (smooth, bandwidth, float(np.min(smooth) - 8.0 * bandwidth),
            float(np.max(smooth) + 8.0 * bandwidth))


def _summary_vector_q6(points, weights):
    """Six simultaneous bisections; unchanged float64 reductions and ordering."""
    mean = np.sum(points * weights[:, None], axis=0)
    delta = points - mean
    covariance = (delta.T * weights) @ delta
    prepared = [_smoothed_quantile_components(points[:, j], weights) for j in range(2)]
    smooth = np.repeat(np.stack([p[0] for p in prepared]), 3, axis=0)
    bandwidth = np.repeat([p[1] for p in prepared], 3)
    lower = np.repeat([p[2] for p in prepared], 3)
    upper = np.repeat([p[3] for p in prepared], 3)
    probabilities = np.tile([0.025, 0.5, 0.975], 2)
    work = np.empty_like(smooth, order="C")
    for _ in range(45):
        mid = 0.5 * (lower + upper)
        np.subtract(mid[:, None], smooth, out=work)
        np.divide(work, bandwidth[:, None], out=work)
        ndtr(work, out=work)
        np.multiply(work, weights[None, :], out=work)
        low = np.sum(work, axis=1) < probabilities
        lower = np.where(low, mid, lower)
        upper = np.where(low, upper, mid)
    q = (0.5 * (lower + upper)).reshape(2, 3).T.reshape(-1)
    return np.r_[mean, covariance.reshape(-1), q]


def _summary_vectors_batch(points, weights, block_size=32):
    """Bound quantile scratch space; keep each posterior's reductions intact.

    This is only the integration convergence diagnostic, not posterior draws
    or CP credible regions. Singleton batches retain the cheaper q6 path.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if points.ndim != 3 or points.shape[2] != 2 or weights.shape != points.shape[:2]:
        raise ValueError("Expected (B,N,2) points and (B,N) weights")
    if len(points) == 1:
        return _summary_vector_q6(points[0], weights[0])[None, :]
    output = np.empty((len(points), 12))
    probabilities = np.array([0.025, 0.5, 0.975])
    for start in range(0, len(points), block_size):
        stop = min(start + block_size, len(points))
        size, nodes = stop - start, points.shape[1]
        smooth = np.empty((size, 2, 1, nodes))
        bandwidth = np.empty((size, 2, 1))
        lower, upper = np.empty((size, 2, 3)), np.empty((size, 2, 3))
        for b, (p, w) in enumerate(zip(points[start:stop], weights[start:stop])):
            mean = np.sum(p * w[:, None], axis=0)
            delta = p - mean
            covariance = (delta.T * w) @ delta
            output[start + b, :6] = np.r_[mean, covariance.reshape(-1)]
            for j in range(2):
                s, h, lo, hi = _smoothed_quantile_components(p[:, j], w)
                smooth[b, j, 0] = s
                bandwidth[b, j, 0] = h
                lower[b, j], upper[b, j] = lo, hi
        work = np.empty((size, 2, 3, nodes))
        for _ in range(45):
            mid = 0.5 * (lower + upper)
            np.subtract(mid[..., None], smooth, out=work)
            np.divide(work, bandwidth[..., None], out=work)
            ndtr(work, out=work)
            np.multiply(work, weights[start:stop, None, None, :], out=work)
            low = np.sum(work, axis=-1) < probabilities
            lower = np.where(low, mid, lower)
            upper = np.where(low, upper, mid)
        output[start:stop, 6:] = (0.5 * (lower + upper)).transpose(0, 2, 1).reshape(size, 6)
    return output


@dataclass
class WeightedPosterior:
    points: np.ndarray
    weights: np.ndarray
    log_normalizer: float
    prior: IndependentTPrior
    x_std: np.ndarray
    y: np.ndarray
    accuracy: dict
    grouped_counts: tuple | None = None

    @property
    def mean(self):
        return np.sum(self.points * self.weights[:, None], axis=0)

    @property
    def covariance(self):
        delta = self.points - self.mean
        return (delta.T * self.weights) @ delta

    def beta_logpdf(self, beta):
        if self.grouped_counts is None:
            log_likelihood = logistic_loglik(beta, self.x_std, self.y)
        else:
            log_likelihood = logistic_loglik_counts(
                beta, *self.grouped_counts
            )
        return log_likelihood + self.prior.logpdf(beta) - self.log_normalizer

    def resample(self, size, rng):
        index = rng.choice(len(self.points), size=int(size), p=self.weights)
        return self.points[index]


def _gauss_hermite_posterior(x_std, y, prior, nodes, mode, covariance):
    roots, raw_weights = hermgauss(int(nodes))
    z1, z2 = np.meshgrid(roots, roots, indexing="ij")
    w1, w2 = np.meshgrid(raw_weights, raw_weights, indexing="ij")
    z = np.column_stack([z1.ravel(), z2.ravel()]) * np.sqrt(2.0)
    chol = np.linalg.cholesky(covariance)
    points = mode + z @ chol.T
    log_quadrature = np.log((w1 * w2).ravel()) - np.log(np.pi)
    log_proposal = _normal_logpdf(points, mode, covariance)
    log_unnormalized = log_posterior(points, x_std, y, prior)
    log_terms = log_quadrature + log_unnormalized - log_proposal
    log_norm = float(logsumexp(log_terms))
    weights = np.exp(log_terms - log_norm)
    ess = float(1.0 / np.sum(np.square(weights)))
    return points, weights, log_norm, ess


def _gauss_hermite_posterior_counts(
    levels, totals, cases, prior, nodes, mode, covariance
):
    roots, raw_weights = hermgauss(int(nodes))
    z1, z2 = np.meshgrid(roots, roots, indexing="ij")
    w1, w2 = np.meshgrid(raw_weights, raw_weights, indexing="ij")
    z = np.column_stack([z1.ravel(), z2.ravel()]) * np.sqrt(2.0)
    chol = np.linalg.cholesky(covariance)
    points = mode + z @ chol.T
    log_quadrature = np.log((w1 * w2).ravel()) - np.log(np.pi)
    log_proposal = _normal_logpdf(points, mode, covariance)
    log_unnormalized = log_posterior_counts(
        points, levels, totals, cases, prior
    )
    log_terms = log_quadrature + log_unnormalized - log_proposal
    log_norm = float(logsumexp(log_terms))
    weights = np.exp(log_terms - log_norm)
    ess = float(1.0 / np.sum(np.square(weights)))
    return points, weights, log_norm, ess


def _sobol_mixture_posterior(x_std, y, prior, mode, covariance, seed, power=15):
    n = 2 ** int(power)
    engine = qmc.Sobol(d=4, scramble=True, seed=int(seed))
    u = np.clip(engine.random_base2(int(power)), 1e-12, 1.0 - 1e-12)
    component = (u[:, 0] >= 0.75).astype(int)
    centers = np.vstack([mode, prior.loc])
    covariances = np.stack(
        [4.0 * covariance, np.diag(np.maximum(np.square(prior.scale), 1e-6))]
    )
    df = 5.0
    normal_draws = norm.ppf(u[:, 1:3])
    scale_draws = np.sqrt(chi2.ppf(u[:, 3], df=df) / df)
    z = normal_draws / scale_draws[:, None]
    points = np.empty((n, 2), dtype=float)
    for k in (0, 1):
        selected = component == k
        if not np.any(selected):
            continue
        points[selected] = centers[k] + z[selected] @ np.linalg.cholesky(covariances[k]).T
    log_q0 = _student_logpdf_2d(points, centers[0], covariances[0], df=df)
    log_q1 = _student_logpdf_2d(points, centers[1], covariances[1], df=df)
    log_proposal = logsumexp(
        np.column_stack([np.log(0.75) + log_q0, np.log(0.25) + log_q1]),
        axis=1,
    )
    log_terms = log_posterior(points, x_std, y, prior) - log_proposal
    log_norm = float(logsumexp(log_terms) - np.log(n))
    weights = np.exp(log_terms - logsumexp(log_terms))
    ess = float(1.0 / np.sum(np.square(weights)))
    return points, weights, log_norm, ess


def _sobol_mixture_posterior_counts(
    levels, totals, cases, prior, mode, covariance, seed, power=15
):
    n = 2 ** int(power)
    engine = qmc.Sobol(d=4, scramble=True, seed=int(seed))
    u = np.clip(engine.random_base2(int(power)), 1e-12, 1.0 - 1e-12)
    component = (u[:, 0] >= 0.75).astype(int)
    centers = np.vstack([mode, prior.loc])
    covariances = np.stack(
        [4.0 * covariance, np.diag(np.maximum(np.square(prior.scale), 1e-6))]
    )
    df = 5.0
    normal_draws = norm.ppf(u[:, 1:3])
    scale_draws = np.sqrt(chi2.ppf(u[:, 3], df=df) / df)
    z = normal_draws / scale_draws[:, None]
    points = np.empty((n, 2), dtype=float)
    for k in (0, 1):
        selected = component == k
        if not np.any(selected):
            continue
        points[selected] = (
            centers[k]
            + z[selected] @ np.linalg.cholesky(covariances[k]).T
        )
    log_q0 = _student_logpdf_2d(points, centers[0], covariances[0], df=df)
    log_q1 = _student_logpdf_2d(points, centers[1], covariances[1], df=df)
    log_proposal = logsumexp(
        np.column_stack([np.log(0.75) + log_q0, np.log(0.25) + log_q1]),
        axis=1,
    )
    log_terms = (
        log_posterior_counts(points, levels, totals, cases, prior)
        - log_proposal
    )
    log_norm = float(logsumexp(log_terms) - np.log(n))
    weights = np.exp(log_terms - logsumexp(log_terms))
    ess = float(1.0 / np.sum(np.square(weights)))
    return points, weights, log_norm, ess


def fit_logistic_posterior(
    x_std,
    y,
    prior,
    seed=BASE_SEED,
    gh_nodes=(21, 31, 41),
    tolerance=1e-3,
    sobol_power=15,
):
    """Fit a two-dimensional logistic posterior by adaptive quadrature.

    Gauss-Hermite rules are importance-corrected for the local Gaussian
    proposal.  If consecutive rules do not stabilize posterior moments and
    marginal quantiles, a seeded Sobol Student-t mixture is used.
    """
    x_std, y = _validate_xy(x_std, y)
    mode, covariance, mode_ok, mode_message = posterior_mode_covariance(
        x_std, y, prior
    )
    previous = None
    previous_log_norm = None
    last = None
    relative_error = np.inf
    normalizer_error = np.inf
    for nodes in gh_nodes:
        points, weights, log_norm, ess = _gauss_hermite_posterior(
            x_std, y, prior, nodes, mode, covariance
        )
        summary = _summary_vector(points, weights)
        if previous is not None:
            scale = np.maximum(1.0, np.abs(previous))
            summary_error = float(np.max(np.abs(summary - previous) / scale))
            normalizer_error = float(
                abs(log_norm - previous_log_norm)
                / max(1.0, abs(previous_log_norm))
            )
            relative_error = max(summary_error, normalizer_error)
            tail_adequate = ess >= 25.0 and float(np.max(weights)) <= 0.075
            if relative_error <= tolerance and tail_adequate:
                accuracy = {
                    "engine": "gauss_hermite",
                    "fallback_used": False,
                    "nodes_per_axis": int(nodes),
                    "n_points": int(len(points)),
                    "converged": True,
                    "max_relative_change": relative_error,
                    "normalizer_relative_change": normalizer_error,
                    "importance_ess": ess,
                    "max_weight": float(np.max(weights)),
                    "mode_converged": mode_ok,
                    "mode_message": mode_message,
                }
                return WeightedPosterior(
                    points, weights, log_norm, prior, x_std, y, accuracy
                )
        previous = summary
        previous_log_norm = log_norm
        last = points, weights, log_norm, ess, nodes

    points, weights, log_norm, ess = _sobol_mixture_posterior(
        x_std, y, prior, mode, covariance, seed=seed, power=sobol_power
    )
    accuracy = {
        "engine": "sobol_mixture_importance",
        "fallback_used": True,
        "nodes_per_axis": 0,
        "n_points": int(len(points)),
        "converged": bool(ess >= 500),
        "max_relative_change": relative_error,
        "normalizer_relative_change": normalizer_error,
        "importance_ess": ess,
        "max_weight": float(np.max(weights)),
        "mode_converged": mode_ok,
        "mode_message": mode_message,
        "last_gh_nodes": int(last[-1]),
    }
    return WeightedPosterior(points, weights, log_norm, prior, x_std, y, accuracy)


def fit_logistic_posterior_counts(
    levels,
    totals,
    cases,
    prior,
    seed=BASE_SEED,
    gh_nodes=(21, 31, 41),
    tolerance=1e-3,
    sobol_power=15,
    start=None,
    mode_engine="scipy",
):
    """Fit the same posterior from five sufficient-statistic cells.

    This is an opt-in fast path.  It retains the adaptive GH sequence and the
    seeded Sobol fallback of :func:`fit_logistic_posterior`; only likelihood
    evaluation and (optionally) mode finding use grouped/compiled kernels.
    """
    levels, totals, cases = _validate_grouped_counts(levels, totals, cases)
    mode, covariance, mode_ok, mode_message = posterior_mode_covariance_counts(
        levels,
        totals,
        cases,
        prior,
        start=start,
        mode_engine=mode_engine,
    )
    previous = None
    previous_log_norm = None
    last = None
    relative_error = np.inf
    normalizer_error = np.inf
    grouped = (levels.copy(), totals.copy(), cases.copy())
    for nodes in gh_nodes:
        points, weights, log_norm, ess = _gauss_hermite_posterior_counts(
            levels, totals, cases, prior, nodes, mode, covariance
        )
        summary = _summary_vector(points, weights)
        if previous is not None:
            scale = np.maximum(1.0, np.abs(previous))
            summary_error = float(np.max(np.abs(summary - previous) / scale))
            normalizer_error = float(
                abs(log_norm - previous_log_norm)
                / max(1.0, abs(previous_log_norm))
            )
            relative_error = max(summary_error, normalizer_error)
            tail_adequate = ess >= 25.0 and float(np.max(weights)) <= 0.075
            if relative_error <= tolerance and tail_adequate:
                accuracy = {
                    "engine": "gauss_hermite_counts",
                    "fallback_used": False,
                    "nodes_per_axis": int(nodes),
                    "n_points": int(len(points)),
                    "converged": True,
                    "max_relative_change": relative_error,
                    "normalizer_relative_change": normalizer_error,
                    "importance_ess": ess,
                    "max_weight": float(np.max(weights)),
                    "mode_converged": mode_ok,
                    "mode_message": mode_message,
                }
                return WeightedPosterior(
                    points,
                    weights,
                    log_norm,
                    prior,
                    np.empty(0),
                    np.empty(0, dtype=int),
                    accuracy,
                    grouped_counts=grouped,
                )
        previous = summary
        previous_log_norm = log_norm
        last = points, weights, log_norm, ess, nodes

    points, weights, log_norm, ess = _sobol_mixture_posterior_counts(
        levels,
        totals,
        cases,
        prior,
        mode,
        covariance,
        seed=seed,
        power=sobol_power,
    )
    accuracy = {
        "engine": "sobol_mixture_importance_counts",
        "fallback_used": True,
        "nodes_per_axis": 0,
        "n_points": int(len(points)),
        "converged": bool(ess >= 500),
        "max_relative_change": relative_error,
        "normalizer_relative_change": normalizer_error,
        "importance_ess": ess,
        "max_weight": float(np.max(weights)),
        "mode_converged": mode_ok,
        "mode_message": mode_message,
        "last_gh_nodes": int(last[-1]),
    }
    return WeightedPosterior(
        points,
        weights,
        log_norm,
        prior,
        np.empty(0),
        np.empty(0, dtype=int),
        accuracy,
        grouped_counts=grouped,
    )


def fit_logistic_posteriors_counts_batch(
    levels,
    totals,
    cases,
    priors,
    seeds,
    gh_nodes=(21, 31, 41),
    tolerance=1e-3,
    sobol_power=15,
    sobol_max_power=None,
    starts=None,
    summary_engine=LUPUS_SUMMARY_ENGINE,
):
    """Fit a batch of grouped posteriors with one Python/C++ boundary.

    The adaptive GH decision remains posterior-specific. Batched q6 summaries
    use blocks of 32 (q6 alone for singletons); ``summary_engine="reference"``
    retains the original diagnostic for regression checks. Only batched mode
    finding and batched log-posterior evaluation are compiled; unresolved
    modes or quadrature rules fall back to the same SciPy/Sobol reference
    paths used by the scalar count engine.
    """
    if summary_engine not in ("reference", LUPUS_SUMMARY_ENGINE):
        raise ValueError("Unknown lupus summary engine: " + str(summary_engine))
    sobol_power = int(sobol_power)
    sobol_max_power = (
        sobol_power if sobol_max_power is None else int(sobol_max_power)
    )
    if sobol_max_power < sobol_power:
        raise ValueError("sobol_max_power must be at least sobol_power.")
    backend = _load_lupus_fast_backend()
    if backend is None:
        raise RuntimeError(
            "The C++ lupus backend is not built. Run scripts/build_lupus_fast_backend.sh."
        )
    levels = np.asarray(levels, dtype=float).reshape(-1)
    totals = np.asarray(totals, dtype=np.int64)
    cases = np.asarray(cases, dtype=np.int64)
    if totals.ndim != 2 or cases.shape != totals.shape:
        raise ValueError("totals and cases must have matching (batch, levels) shapes.")
    if totals.shape[1] != len(levels):
        raise ValueError("Count columns must match the predictor levels.")
    batch = len(totals)
    if isinstance(priors, IndependentTPrior):
        priors = [priors] * batch
    else:
        priors = list(priors)
    if len(priors) != batch:
        raise ValueError("priors must contain one prior per grouped dataset.")
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if len(seeds) != batch:
        raise ValueError("seeds must contain one seed per grouped dataset.")
    locations = np.asarray([prior.loc for prior in priors], dtype=float)
    scales = np.asarray([prior.scale for prior in priors], dtype=float)
    dfs = np.asarray([prior.df for prior in priors], dtype=float)
    if starts is None:
        starts = locations.copy()
    else:
        starts = np.asarray(starts, dtype=float).reshape(batch, 2)
    mode_output = backend.batch_newton_modes(
        levels,
        totals,
        cases,
        locations,
        scales,
        dfs,
        starts,
        60,
        1e-10,
    )
    modes = np.asarray(mode_output["modes"], dtype=float)
    mode_ok = np.asarray(mode_output["converged"], dtype=bool)
    iterations = np.asarray(mode_output["iterations"], dtype=int)
    covariance = np.empty((batch, 2, 2), dtype=float)
    for index in range(batch):
        _, ll_hessian = logistic_gradient_hessian_counts(
            modes[index], levels, totals[index], cases[index]
        )
        negative_hessian = -ll_hessian + np.diag(
            priors[index].negative_hessian_diag(modes[index])
        )
        eigenvalue, eigenvector = np.linalg.eigh(
            0.5 * (negative_hessian + negative_hessian.T)
        )
        eigenvalue = np.maximum(eigenvalue, 1e-7)
        covariance[index] = (eigenvector / eigenvalue) @ eigenvector.T

    results = [None] * batch
    unresolved = np.flatnonzero(mode_ok & np.all(np.isfinite(modes), axis=1))
    failed_modes = np.flatnonzero(~(mode_ok & np.all(np.isfinite(modes), axis=1)))
    for index in failed_modes:
        results[index] = fit_logistic_posterior_counts(
            levels,
            totals[index],
            cases[index],
            priors[index],
            seed=int(seeds[index]),
            gh_nodes=gh_nodes,
            tolerance=tolerance,
            sobol_power=sobol_power,
            start=starts[index],
            mode_engine="scipy",
        )

    previous_summary = {}
    previous_log_normalizer = {}
    relative_error = {int(index): np.inf for index in unresolved}
    normalizer_error = {int(index): np.inf for index in unresolved}
    for nodes in gh_nodes:
        if not len(unresolved):
            break
        roots, raw_weights = hermgauss(int(nodes))
        z1, z2 = np.meshgrid(roots, roots, indexing="ij")
        w1, w2 = np.meshgrid(raw_weights, raw_weights, indexing="ij")
        z = np.column_stack([z1.ravel(), z2.ravel()]) * np.sqrt(2.0)
        log_quadrature = np.log((w1 * w2).ravel()) - np.log(np.pi)
        active_covariance = covariance[unresolved]
        cholesky = np.linalg.cholesky(active_covariance)
        points = modes[unresolved, None, :] + np.einsum(
            "nj,bkj->bnk", z, cholesky
        )
        log_unnormalized = backend.batch_count_logposterior(
            points,
            levels,
            totals[unresolved],
            cases[unresolved],
            locations[unresolved],
            scales[unresolved],
            dfs[unresolved],
        )
        _, log_determinant = np.linalg.slogdet(active_covariance)
        log_proposal = (
            -np.log(2.0 * np.pi)
            - 0.5 * log_determinant[:, None]
            - 0.5 * np.sum(np.square(z), axis=1)[None, :]
        )
        log_terms = (
            log_quadrature[None, :] + log_unnormalized - log_proposal
        )
        log_normalizer = logsumexp(log_terms, axis=1)
        weights = np.exp(log_terms - log_normalizer[:, None])
        summaries = (
            _summary_vectors_batch(points, weights)
            if summary_engine == LUPUS_SUMMARY_ENGINE
            else np.asarray([_summary_vector(p, w) for p, w in zip(points, weights)])
        )
        still_unresolved = []
        for local_index, global_index_raw in enumerate(unresolved):
            global_index = int(global_index_raw)
            summary = summaries[local_index]
            converged = False
            if global_index in previous_summary:
                scale = np.maximum(1.0, np.abs(previous_summary[global_index]))
                summary_error = float(
                    np.max(
                        np.abs(summary - previous_summary[global_index]) / scale
                    )
                )
                norm_error = float(
                    abs(
                        log_normalizer[local_index]
                        - previous_log_normalizer[global_index]
                    )
                    / max(1.0, abs(previous_log_normalizer[global_index]))
                )
                relative_error[global_index] = max(summary_error, norm_error)
                normalizer_error[global_index] = norm_error
                effective_size = float(
                    1.0 / np.sum(np.square(weights[local_index]))
                )
                tail_adequate = (
                    effective_size >= 25.0
                    and float(np.max(weights[local_index])) <= 0.075
                )
                converged = (
                    relative_error[global_index] <= tolerance and tail_adequate
                )
                if converged:
                    accuracy = {
                        "engine": "gauss_hermite_counts_cpp_batch",
                        "fallback_used": False,
                        "nodes_per_axis": int(nodes),
                        "n_points": int(points.shape[1]),
                        "converged": True,
                        "max_relative_change": relative_error[global_index],
                        "normalizer_relative_change": norm_error,
                        "importance_ess": effective_size,
                        "max_weight": float(np.max(weights[local_index])),
                        "mode_converged": True,
                        "mode_message": (
                            f"cpp_batch_newton:{int(iterations[global_index])}"
                        ),
                    }
                    grouped = (
                        levels.copy(),
                        totals[global_index].copy(),
                        cases[global_index].copy(),
                    )
                    results[global_index] = WeightedPosterior(
                        points[local_index].copy(),
                        weights[local_index].copy(),
                        float(log_normalizer[local_index]),
                        priors[global_index],
                        np.empty(0),
                        np.empty(0, dtype=int),
                        accuracy,
                        grouped_counts=grouped,
                    )
            if not converged:
                previous_summary[global_index] = summary
                previous_log_normalizer[global_index] = float(
                    log_normalizer[local_index]
                )
                still_unresolved.append(global_index)
        unresolved = np.asarray(still_unresolved, dtype=int)

    for index_raw in unresolved:
        index = int(index_raw)
        power_used = sobol_power
        points, weights, log_normalizer, effective_size = (
            _sobol_mixture_posterior_counts(
                levels,
                totals[index],
                cases[index],
                priors[index],
                modes[index],
                covariance[index],
                seed=int(seeds[index]),
                power=sobol_power,
            )
        )
        # Refine only the exceptional posterior that misses the ESS gate.
        # Applying the largest Sobol rule to every fallback would needlessly
        # multiply an exhaustive Clarke search by 2**(max_power-power).
        if effective_size < 500 and sobol_max_power > sobol_power:
            power_used = sobol_max_power
            points, weights, log_normalizer, effective_size = (
                _sobol_mixture_posterior_counts(
                    levels,
                    totals[index],
                    cases[index],
                    priors[index],
                    modes[index],
                    covariance[index],
                    seed=int(seeds[index]),
                    power=power_used,
                )
            )
        accuracy = {
            "engine": "sobol_mixture_importance_counts_cpp_batch",
            "fallback_used": True,
            "nodes_per_axis": 0,
            "n_points": int(len(points)),
            "converged": bool(effective_size >= 500),
            "max_relative_change": relative_error[index],
            "normalizer_relative_change": normalizer_error[index],
            "importance_ess": effective_size,
            "max_weight": float(np.max(weights)),
            "mode_converged": True,
            "mode_message": f"cpp_batch_newton:{int(iterations[index])}",
            "last_gh_nodes": int(tuple(gh_nodes)[-1]),
            "sobol_power_used": int(power_used),
        }
        grouped = (levels.copy(), totals[index].copy(), cases[index].copy())
        results[index] = WeightedPosterior(
            points,
            weights,
            log_normalizer,
            priors[index],
            np.empty(0),
            np.empty(0, dtype=int),
            accuracy,
            grouped_counts=grouped,
        )
    if any(result is None for result in results):
        raise RuntimeError("Internal error: an adaptive batch posterior is missing.")
    return results


def standardized_x(raw_x):
    data = lupus_patient_data()
    return (np.asarray(raw_x, dtype=float) - data["x_mean"]) / (2.0 * data["x_sd"])


def target_values(beta, target):
    """Transform standardized coefficients to a named EPSS target."""
    beta = np.atleast_2d(np.asarray(beta, dtype=float))
    target = str(target)
    if target == "beta_joint":
        return beta
    if target == "beta0":
        return beta[:, 0]
    if target == "beta1":
        return beta[:, 1]
    p_low = expit(beta[:, 0] + beta[:, 1] * standardized_x(0.0))
    p_high = expit(beta[:, 0] + beta[:, 1] * standardized_x(2.0))
    if target == "prediction_joint":
        return np.column_stack([p_low, p_high])
    if target == "risk_difference":
        return p_high - p_low
    if target.startswith("p_x_"):
        raw = float(target.removeprefix("p_x_").replace("p", ".").replace("m", "-"))
        return expit(beta[:, 0] + beta[:, 1] * standardized_x(raw))
    raise ValueError(f"Unknown target: {target}")


def target_dimension(target):
    return 2 if target in {"beta_joint", "prediction_joint"} else 1


class PosteriorTargetCache:
    """Cache target transforms, quantiles, and joint KDE HDR objects."""

    def __init__(self, posterior):
        self.posterior = posterior
        self._values = {}
        self._quantiles = {}
        self._kde_hdr = {}

    def values(self, target):
        if target not in self._values:
            self._values[target] = target_values(self.posterior.points, target)
        return self._values[target]

    def quantiles(self, target, probabilities):
        probabilities = tuple(float(value) for value in probabilities)
        key = (target, probabilities)
        if key not in self._quantiles:
            self._quantiles[key] = weighted_quantile(
                self.values(target),
                probabilities,
                self.posterior.weights,
            )
        return self._quantiles[key]

    def kde_hdr(self, target, alpha):
        """Return the historical weighted KDE and its own HDR threshold."""
        if target_dimension(target) != 2:
            raise ValueError("KDE HDR caching is defined only for joint targets.")
        key = (target, float(alpha))
        if key not in self._kde_hdr:
            values = self.values(target)
            kde = _weighted_kde(values, self.posterior.weights)
            own_density = kde(values.T)
            threshold = weighted_quantile(
                own_density, [alpha], self.posterior.weights
            )[0]
            self._kde_hdr[key] = (kde, float(threshold))
        return self._kde_hdr[key]


def clarke_scalar_coordinate(beta, target):
    """Stable one-to-one coordinate used for scalar Clarke KL quadrature."""
    values = np.asarray(target_values(beta, target), dtype=float)
    if target == "beta1":
        return np.arcsinh(values / 5.0)
    if target == "risk_difference":
        transformed = np.arctanh(np.clip(values, -1.0 + 1e-12, 1.0 - 1e-12))
        return np.arcsinh(transformed / 3.0)
    if target.startswith("p_x_"):
        logits = np.log(
            np.clip(values, 1e-12, 1.0 - 1e-12)
            / np.clip(1.0 - values, 1e-12, 1.0)
        )
        return np.arcsinh(logits / 5.0)
    raise ValueError(f"Target {target!r} is not scalar.")


def clarke_scalar_grid(size=1201):
    """Common transformed grid covering diffuse baseline-prior tails."""
    return np.linspace(-18.0, 18.0, int(size))


def beta1_prior_density_on_clarke_grid(prior, grid):
    """Exact transformed beta1 density for an independent Student-t prior."""
    grid = np.asarray(grid, dtype=float)
    beta1 = 5.0 * np.sinh(grid)
    jacobian = 5.0 * np.cosh(grid)
    density = np.exp(
        t.logpdf(
            beta1,
            df=prior.df,
            loc=prior.loc[1],
            scale=prior.scale[1],
        )
    ) * jacobian
    dx = float(grid[1] - grid[0])
    density = np.maximum(density, 1e-300)
    density /= np.sum(density) * dx
    return density


def weighted_density_grid(values, weights, grid):
    """Fast normalized density estimate on an equally spaced grid."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float)
    dx = float(grid[1] - grid[0])
    edges = np.r_[grid - 0.5 * dx, grid[-1] + 0.5 * dx]
    mass, _ = np.histogram(values, bins=edges, weights=weights)
    captured = float(np.sum(mass))
    if captured <= 0.0:
        raise ValueError("Density grid captured no probability mass.")
    mass = mass / captured
    mean = float(np.sum(values * weights) / np.sum(weights))
    variance = float(
        np.sum(weights * np.square(values - mean)) / np.sum(weights)
    )
    effective_n = float(np.square(np.sum(weights)) / np.sum(np.square(weights)))
    bandwidth = (
        0.35
        * np.sqrt(max(variance, 1e-12))
        * effective_n ** (-0.2)
    )
    bandwidth = max(0.65 * dx, bandwidth)
    offsets = np.arange(-(len(grid) - 1), len(grid), dtype=float) * dx
    kernel = t.pdf(offsets / bandwidth, df=5.0) / bandwidth
    density = fftconvolve(mass, kernel, mode="same")
    density = np.maximum(density, 1e-300)
    density /= np.sum(density) * dx
    return density, captured


def scalar_kl_to_many_prior_grids(
    posterior,
    target,
    grid,
    prior_log_density_matrix,
):
    """Evaluate one scalar posterior against many fixed prior marginals."""
    values = clarke_scalar_coordinate(posterior.points, target)
    p_density, captured = weighted_density_grid(
        values, posterior.weights, grid
    )
    dx = float(grid[1] - grid[0])
    p_mass = p_density * dx
    entropy_part = float(np.sum(p_mass * np.log(p_density)))
    cross_entropy = prior_log_density_matrix @ p_mass
    return entropy_part - cross_entropy, captured


def weighted_gaussian_kde_regular_grid(
    values,
    weights,
    grid_size=4097,
    tail_bandwidths=8.0,
):
    """Approximate SciPy's weighted 1D Gaussian KDE on a regular grid.

    The bandwidth is Scott's weighted rule used by :class:`gaussian_kde`.
    Linear binning followed by an FFT Gaussian convolution makes repeated
    density evaluation cheap while retaining the original target coordinate.
    The returned tuple is ``(density, lower, step, bandwidth)``.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != len(weights) or not len(values):
        raise ValueError("values and weights must be nonempty and have equal length.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("KDE inputs must be finite.")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("KDE weights must have positive total mass.")
    weights = weights / total_weight
    weight_square_sum = float(np.sum(np.square(weights)))
    effective_n = 1.0 / weight_square_sum
    mean = float(np.sum(weights * values))
    variance_numerator = float(
        np.sum(weights * np.square(values - mean))
    )
    variance = variance_numerator / max(1.0 - weight_square_sum, 1e-12)
    bandwidth = float(
        np.sqrt(max(variance, 1e-14)) * effective_n ** (-0.2)
    )
    bandwidth = max(bandwidth, 1e-10 * max(1.0, abs(mean)))
    grid_size = int(grid_size)
    if grid_size < 257:
        raise ValueError("grid_size must be at least 257.")
    lower = float(np.min(values) - float(tail_bandwidths) * bandwidth)
    upper = float(np.max(values) + float(tail_bandwidths) * bandwidth)
    if not upper > lower:
        upper = lower + max(16.0 * bandwidth, 1e-8)
    step = (upper - lower) / float(grid_size - 1)

    position = np.clip((values - lower) / step, 0.0, grid_size - 1.0)
    left = np.floor(position).astype(np.int64)
    fraction = position - left
    right = np.minimum(left + 1, grid_size - 1)
    mass = np.bincount(
        left,
        weights=weights * (1.0 - fraction),
        minlength=grid_size,
    ).astype(float)
    mass += np.bincount(
        right,
        weights=weights * fraction,
        minlength=grid_size,
    )
    offsets = np.arange(-(grid_size - 1), grid_size, dtype=float) * step
    kernel = np.exp(-0.5 * np.square(offsets / bandwidth))
    kernel /= np.sqrt(2.0 * np.pi) * bandwidth
    density = fftconvolve(mass, kernel, mode="same")
    density = np.maximum(density, 1e-300)
    density /= float(np.sum(density) * step)
    return density, lower, step, bandwidth


def interpolate_regular_density(values, density, lower, step):
    """Linearly interpolate one or many row-specific regular density grids."""
    values = np.asarray(values, dtype=float)
    density = np.asarray(density, dtype=float)
    lower = np.asarray(lower, dtype=float)
    step = np.asarray(step, dtype=float)
    if density.ndim == 1:
        position = (values - float(lower)) / float(step)
        left = np.floor(position).astype(np.int64)
        valid = (left >= 0) & (left < len(density) - 1)
        clipped = np.clip(left, 0, len(density) - 2)
        fraction = position - left
        answer = (
            density[clipped] * (1.0 - fraction)
            + density[clipped + 1] * fraction
        )
        return np.where(valid, np.maximum(answer, 1e-300), 1e-300)
    if density.ndim != 2:
        raise ValueError("density must have one or two dimensions.")
    if values.ndim == 1:
        values = np.broadcast_to(values[None, :], (len(density), len(values)))
    if values.ndim != 2 or values.shape[0] != density.shape[0]:
        raise ValueError("row-specific values must match density rows.")
    lower = lower.reshape(-1, 1)
    step = step.reshape(-1, 1)
    position = (values - lower) / step
    left = np.floor(position).astype(np.int64)
    valid = (left >= 0) & (left < density.shape[1] - 1)
    clipped = np.clip(left, 0, density.shape[1] - 2)
    fraction = position - left
    left_density = np.take_along_axis(density, clipped, axis=1)
    right_density = np.take_along_axis(density, clipped + 1, axis=1)
    answer = left_density * (1.0 - fraction) + right_density * fraction
    return np.where(valid, np.maximum(answer, 1e-300), 1e-300)


def scalar_tkl_to_many_prior_kde_grids(
    posterior,
    target,
    prior_density_grids,
    prior_grid_lower,
    prior_grid_step,
    prior_target_samples,
    alpha=0.05,
    grid_size=4097,
):
    """Evaluate scalar raw-coordinate HDR-TKL against many cached priors.

    This is a gridded acceleration of :func:`target_kl_to_prior`: it retains
    the raw target coordinate, Gaussian KDE bandwidth rule, P-HDR truncation,
    fixed prior samples, and normalization correction.  Only KDE evaluation
    is replaced by linear-binned FFT grids and interpolation.
    """
    p_values = np.asarray(target_values(posterior.points, target), dtype=float)
    p_density_grid, p_lower, p_step, _ = weighted_gaussian_kde_regular_grid(
        p_values,
        posterior.weights,
        grid_size=grid_size,
    )
    p_density_at_p = interpolate_regular_density(
        p_values, p_density_grid, p_lower, p_step
    )
    log_p = np.log(p_density_at_p)
    threshold = float(weighted_quantile(log_p, [alpha], posterior.weights)[0])
    inside = log_p >= threshold
    p_mass = float(np.sum(posterior.weights[inside]))

    prior_density_at_p = interpolate_regular_density(
        p_values,
        prior_density_grids,
        prior_grid_lower,
        prior_grid_step,
    )
    pointwise = (
        np.sum(
            posterior.weights[inside][None, :]
            * (
                log_p[inside][None, :]
                - np.log(prior_density_at_p[:, inside])
            ),
            axis=1,
        )
        / p_mass
    )
    p_density_at_q = interpolate_regular_density(
        prior_target_samples,
        p_density_grid,
        p_lower,
        p_step,
    )
    q_mass = np.mean(np.log(p_density_at_q) >= threshold, axis=1)
    prior_draws = int(prior_target_samples.shape[1])
    q_mass = np.maximum(q_mass, 0.5 / float(prior_draws))
    values = pointwise + np.log(q_mass) - np.log(p_mass)
    diagnostics = {
        "distance_mc_draws": prior_draws,
        "hdr_p_mass": p_mass,
        "hdr_q_mass": q_mass,
        "distance_engine": "cached_raw_gaussian_kde_grid",
    }
    return np.asarray(values, dtype=float), diagnostics


def target_log_jacobian(beta, target):
    """Log absolute Jacobian from beta to a two-dimensional target."""
    beta = np.atleast_2d(np.asarray(beta, dtype=float))
    if target == "beta_joint":
        return np.zeros(len(beta), dtype=float)
    if target != "prediction_joint":
        raise ValueError("A two-dimensional Jacobian is only defined for joint targets.")
    x_low = float(standardized_x(0.0))
    x_high = float(standardized_x(2.0))
    p_low = expit(beta[:, 0] + beta[:, 1] * x_low)
    p_high = expit(beta[:, 0] + beta[:, 1] * x_high)
    return (
        np.log(np.maximum(p_low * (1.0 - p_low), 1e-300))
        + np.log(np.maximum(p_high * (1.0 - p_high), 1e-300))
        + np.log(abs(x_high - x_low))
    )


def posterior_target_moments(posterior, target, standardizer=None):
    values = target_values(posterior.points, target)
    if values.ndim == 1:
        mean = float(np.sum(values * posterior.weights))
        variance = float(np.sum(posterior.weights * np.square(values - mean)))
        return np.asarray([mean]), np.asarray([[variance]])
    if standardizer is not None:
        values = (values - standardizer[0]) / standardizer[1]
    mean = np.sum(values * posterior.weights[:, None], axis=0)
    delta = values - mean
    covariance = (delta.T * posterior.weights) @ delta
    return mean, covariance


def posterior_mse(posterior, target, theta_star, standardizer=None):
    mean, covariance = posterior_target_moments(
        posterior, target, standardizer=standardizer
    )
    theta_star = np.atleast_1d(theta_star)
    if standardizer is not None:
        theta_star = (theta_star - standardizer[0]) / standardizer[1]
    return float(np.trace(covariance) + np.sum(np.square(mean - theta_star)))


def posterior_mse_many(posterior, targets, theta_stars, standardizers):
    """Compute all requested posterior MSEs with one target-transform cache."""
    cache = PosteriorTargetCache(posterior)
    output = {}
    for target in targets:
        values = cache.values(target)
        standardizer = standardizers[target]
        if standardizer is not None:
            values = (values - standardizer[0]) / standardizer[1]
        if np.ndim(values) == 1:
            mean = np.asarray([np.sum(values * posterior.weights)])
            variance = float(
                np.sum(posterior.weights * np.square(values - mean[0]))
            )
            covariance = np.asarray([[variance]])
        else:
            mean = np.sum(values * posterior.weights[:, None], axis=0)
            delta = values - mean
            covariance = (delta.T * posterior.weights) @ delta
        theta_star = np.atleast_1d(theta_stars[target])
        if standardizer is not None:
            theta_star = (theta_star - standardizer[0]) / standardizer[1]
        output[target] = float(
            np.trace(covariance) + np.sum(np.square(mean - theta_star))
        )
    return output


def wasserstein_distance(
    post_a,
    post_b,
    target,
    order=2,
    projections=32,
    seed=BASE_SEED,
    standardizer=None,
):
    a = target_values(post_a.points, target)
    b = target_values(post_b.points, target)
    if standardizer is not None:
        a = (a - standardizer[0]) / standardizer[1]
        b = (b - standardizer[0]) / standardizer[1]
    quantiles = (np.arange(256) + 0.5) / 256.0
    if np.ndim(a) == 1:
        aq = weighted_quantile(a, quantiles, post_a.weights)
        bq = weighted_quantile(b, quantiles, post_b.weights)
        return float(np.mean(np.abs(aq - bq) ** order) ** (1.0 / order))
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(projections), a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = []
    for direction in directions:
        aq = weighted_quantile(a @ direction, quantiles, post_a.weights)
        bq = weighted_quantile(b @ direction, quantiles, post_b.weights)
        distances.append(np.mean(np.abs(aq - bq) ** order))
    return float(np.mean(distances) ** (1.0 / order))


def wasserstein_distance_cached(
    post_a,
    post_b,
    target,
    order=2,
    projections=32,
    seed=BASE_SEED,
    standardizer=None,
    cache_a=None,
    cache_b=None,
):
    """Wasserstein objective reusing posterior target transforms/quantiles."""
    cache_a = PosteriorTargetCache(post_a) if cache_a is None else cache_a
    cache_b = PosteriorTargetCache(post_b) if cache_b is None else cache_b
    a = cache_a.values(target)
    b = cache_b.values(target)
    if standardizer is not None:
        a = (a - standardizer[0]) / standardizer[1]
        b = (b - standardizer[0]) / standardizer[1]
    quantiles = (np.arange(256) + 0.5) / 256.0
    if np.ndim(a) == 1:
        if standardizer is None:
            aq = cache_a.quantiles(target, quantiles)
            bq = cache_b.quantiles(target, quantiles)
        else:
            aq = weighted_quantile(a, quantiles, post_a.weights)
            bq = weighted_quantile(b, quantiles, post_b.weights)
        return float(np.mean(np.abs(aq - bq) ** order) ** (1.0 / order))
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(projections), a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = []
    for direction in directions:
        aq = weighted_quantile(a @ direction, quantiles, post_a.weights)
        bq = weighted_quantile(b @ direction, quantiles, post_b.weights)
        distances.append(np.mean(np.abs(aq - bq) ** order))
    return float(np.mean(distances) ** (1.0 / order))


def _weighted_kde(values, weights):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    else:
        values = values.T
    try:
        return gaussian_kde(values, weights=weights)
    except np.linalg.LinAlgError:
        jitter = np.linspace(-1e-8, 1e-8, values.shape[1])
        return gaussian_kde(values + jitter, weights=weights)


def cp_objective(post_a, post_b, target, alpha=0.05):
    """Symmetric cross-coverage objective used by the project CP method."""
    a = target_values(post_a.points, target)
    b = target_values(post_b.points, target)
    if target_dimension(target) == 1:
        b_interval = weighted_quantile(b, [alpha / 2.0, 1.0 - alpha / 2.0], post_b.weights)
        a_interval = weighted_quantile(a, [alpha / 2.0, 1.0 - alpha / 2.0], post_a.weights)
        cp_ab = float(np.sum(post_a.weights * ((a >= b_interval[0]) & (a <= b_interval[1]))))
        cp_ba = float(np.sum(post_b.weights * ((b >= a_interval[0]) & (b <= a_interval[1]))))
    else:
        kde_a = _weighted_kde(a, post_a.weights)
        kde_b = _weighted_kde(b, post_b.weights)
        density_a = kde_a(a.T)
        density_b = kde_b(b.T)
        threshold_a = weighted_quantile(density_a, [alpha], post_a.weights)[0]
        threshold_b = weighted_quantile(density_b, [alpha], post_b.weights)[0]
        cp_ab = float(np.sum(post_a.weights * (kde_b(a.T) >= threshold_b)))
        cp_ba = float(np.sum(post_b.weights * (kde_a(b.T) >= threshold_a)))
    r1 = max(cp_ab, cp_ba)
    r2 = min(cp_ab, cp_ba)
    objective = (r1 - r2) ** 2 + (r2 - (1.0 - alpha)) ** 2
    return {
        "objective": float(objective),
        "cp_ab": cp_ab,
        "cp_ba": cp_ba,
        "r1": float(r1),
        "r2": float(r2),
    }


def cp_objective_cached(
    post_a,
    post_b,
    target,
    alpha=0.05,
    cache_a=None,
    cache_b=None,
    joint_engine="kde",
):
    """CP objective with cached transforms and joint HDR objects.

    ``exact_density`` is available only for the two one-to-one joint targets.
    It evaluates their normalized posterior density directly rather than
    using the historical Gaussian KDE.  The ``kde`` engine retains that
    historical estimator while caching its KDE and own-HDR threshold.
    """
    cache_a = PosteriorTargetCache(post_a) if cache_a is None else cache_a
    cache_b = PosteriorTargetCache(post_b) if cache_b is None else cache_b
    if target_dimension(target) == 1:
        a = cache_a.values(target)
        b = cache_b.values(target)
        probabilities = [alpha / 2.0, 1.0 - alpha / 2.0]
        b_interval = cache_b.quantiles(target, probabilities)
        a_interval = cache_a.quantiles(target, probabilities)
        cp_ab = float(
            np.sum(
                post_a.weights
                * ((a >= b_interval[0]) & (a <= b_interval[1]))
            )
        )
        cp_ba = float(
            np.sum(
                post_b.weights
                * ((b >= a_interval[0]) & (b <= a_interval[1]))
            )
        )
    elif joint_engine == "exact_density":
        if target not in {"beta_joint", "prediction_joint"}:
            raise ValueError(f"No exact-density CP engine for target {target!r}.")
        own_log_a = (
            post_a.beta_logpdf(post_a.points)
            - target_log_jacobian(post_a.points, target)
        )
        own_log_b = (
            post_b.beta_logpdf(post_b.points)
            - target_log_jacobian(post_b.points, target)
        )
        threshold_a = weighted_quantile(
            own_log_a, [alpha], post_a.weights
        )[0]
        threshold_b = weighted_quantile(
            own_log_b, [alpha], post_b.weights
        )[0]
        log_b_at_a = (
            post_b.beta_logpdf(post_a.points)
            - target_log_jacobian(post_a.points, target)
        )
        log_a_at_b = (
            post_a.beta_logpdf(post_b.points)
            - target_log_jacobian(post_b.points, target)
        )
        cp_ab = float(np.sum(post_a.weights * (log_b_at_a >= threshold_b)))
        cp_ba = float(np.sum(post_b.weights * (log_a_at_b >= threshold_a)))
    elif joint_engine == "kde":
        a = cache_a.values(target)
        b = cache_b.values(target)
        kde_a, threshold_a = cache_a.kde_hdr(target, alpha)
        kde_b, threshold_b = cache_b.kde_hdr(target, alpha)
        cp_ab = float(
            np.sum(post_a.weights * (kde_b(a.T) >= threshold_b))
        )
        cp_ba = float(
            np.sum(post_b.weights * (kde_a(b.T) >= threshold_a))
        )
    else:
        raise ValueError("joint_engine must be kde or exact_density.")
    r1 = max(cp_ab, cp_ba)
    r2 = min(cp_ab, cp_ba)
    objective = (r1 - r2) ** 2 + (r2 - (1.0 - alpha)) ** 2
    return {
        "objective": float(objective),
        "cp_ab": cp_ab,
        "cp_ba": cp_ba,
        "r1": float(r1),
        "r2": float(r2),
        "joint_engine": joint_engine if target_dimension(target) == 2 else "equal_tail_cached",
    }


def posterior_kl(post_p, log_q):
    log_p = post_p.beta_logpdf(post_p.points)
    return float(np.sum(post_p.weights * (log_p - log_q(post_p.points))))


def posterior_truncated_kl(post_p, log_q, alpha=0.05):
    log_p = post_p.beta_logpdf(post_p.points)
    threshold = weighted_quantile(log_p, [alpha], post_p.weights)[0]
    inside = log_p >= threshold
    mass = float(np.sum(post_p.weights[inside]))
    return float(
        np.sum(post_p.weights[inside] * (log_p[inside] - log_q(post_p.points[inside])))
        / mass
    )


def prior_kl_from_posterior(post_p, prior_q, truncated=False, alpha=0.05):
    if truncated:
        return posterior_truncated_kl(post_p, prior_q.logpdf, alpha=alpha)
    return posterior_kl(post_p, prior_q.logpdf)


def target_kl_to_prior(
    posterior,
    prior,
    target,
    seed=BASE_SEED,
    prior_draws=8192,
    truncated=False,
    alpha=0.05,
    return_diagnostics=False,
    prior_samples=None,
):
    """Numerical KL from a posterior target marginal to its prior marginal.

    Joint coefficient and joint prediction targets are one-to-one transforms,
    so the exact coefficient-space KL is used.  Scalar marginals use weighted
    KDEs and a fixed seeded prior sample.
    """
    rng = np.random.default_rng(int(seed))
    if prior_samples is not None:
        prior_beta = np.asarray(prior_samples, dtype=float)
        prior_draws = len(prior_beta)
    else:
        prior_beta = None
    if target_dimension(target) == 2:
        if not truncated:
            value = prior_kl_from_posterior(posterior, prior, truncated=False)
            diagnostics = {
                "distance_mc_draws": 0,
                "hdr_p_mass": np.nan,
                "hdr_q_mass": np.nan,
            }
            return (value, diagnostics) if return_diagnostics else value
        log_jacobian_p = target_log_jacobian(posterior.points, target)
        log_p = posterior.beta_logpdf(posterior.points) - log_jacobian_p
        threshold = weighted_quantile(log_p, [alpha], posterior.weights)[0]
        inside = log_p >= threshold
        p_mass = float(np.sum(posterior.weights[inside]))
        if prior_beta is None:
            prior_beta = prior.sample(prior_draws, rng)
        prior_log_p = (
            posterior.beta_logpdf(prior_beta)
            - target_log_jacobian(prior_beta, target)
        )
        q_mass = float(np.mean(prior_log_p >= threshold))
        q_mass = max(q_mass, 0.5 / float(prior_draws))
        pointwise_ratio = (
            posterior.beta_logpdf(posterior.points[inside])
            - prior.logpdf(posterior.points[inside])
        )
        value = float(
            np.sum(posterior.weights[inside] * pointwise_ratio) / p_mass
            + np.log(q_mass)
            - np.log(p_mass)
        )
        diagnostics = {
            "distance_mc_draws": int(prior_draws),
            "hdr_p_mass": p_mass,
            "hdr_q_mass": q_mass,
        }
        return (value, diagnostics) if return_diagnostics else value

    p_values = target_values(posterior.points, target)
    if prior_beta is None:
        prior_beta = prior.sample(prior_draws, rng)
    q_values = target_values(prior_beta, target)
    kde_p = _weighted_kde(p_values, posterior.weights)
    uniform_q = np.full(len(q_values), 1.0 / len(q_values))
    kde_q = _weighted_kde(q_values, uniform_q)
    log_p = np.log(np.maximum(kde_p(p_values), 1e-300))
    log_q = np.log(np.maximum(kde_q(p_values), 1e-300))
    if truncated:
        threshold = weighted_quantile(log_p, [alpha], posterior.weights)[0]
        inside = log_p >= threshold
        p_mass = float(np.sum(posterior.weights[inside]))
        q_mass = float(np.mean(np.log(np.maximum(kde_p(q_values), 1e-300)) >= threshold))
        q_mass = max(q_mass, 0.5 / float(prior_draws))
        value = float(
            np.sum(posterior.weights[inside] * (log_p[inside] - log_q[inside]))
            / p_mass
            + np.log(q_mass)
            - np.log(p_mass)
        )
        diagnostics = {
            "distance_mc_draws": int(prior_draws),
            "hdr_p_mass": p_mass,
            "hdr_q_mass": q_mass,
        }
        return (value, diagnostics) if return_diagnostics else value
    value = float(np.sum(posterior.weights * (log_p - log_q)))
    diagnostics = {
        "distance_mc_draws": int(prior_draws),
        "hdr_p_mass": np.nan,
        "hdr_q_mass": np.nan,
    }
    return (value, diagnostics) if return_diagnostics else value


def enumerate_type_count_vectors(m, n_types=8):
    """Yield every with-replacement count vector of total size ``m``."""
    for indices in combinations_with_replacement(range(int(n_types)), int(m)):
        counts = np.bincount(indices, minlength=int(n_types))
        yield counts


def expand_type_counts(type_counts, cells=None):
    cells = lupus_grouped_cells() if cells is None else list(cells)
    if len(type_counts) != len(cells):
        raise ValueError(
            f"type_counts has length {len(type_counts)}, expected {len(cells)}."
        )
    x = []
    y = []
    for count, (_, x_std, outcome, _) in zip(type_counts, cells):
        if count:
            x.extend([x_std] * int(count))
            y.extend([outcome] * int(count))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int)


def sample_posterior_predictive(posterior, m, rng):
    """Draw a patient-level future path using empirical predictor frequencies."""
    m = int(m)
    if m == 0:
        return np.empty(0), np.empty(0, dtype=int)
    data = lupus_patient_data()
    index = rng.choice(len(data["x_std"]), size=m, replace=True)
    x_new = data["x_std"][index]
    beta = posterior.resample(m, rng)
    y_new = rng.binomial(1, expit(beta[:, 0] + beta[:, 1] * x_new))
    return x_new, y_new


def plugin_predictive_sample(beta_star, m, rng):
    m = int(m)
    data = lupus_patient_data()
    index = rng.choice(len(data["x_std"]), size=m, replace=True)
    x_new = data["x_std"][index]
    probability = expit(beta_star[0] + beta_star[1] * x_new)
    return x_new, rng.binomial(1, probability)


def append_data(x, y, x_new, y_new):
    return np.r_[x, x_new], np.r_[y, y_new]


def resample_patient_pairs(x, y, size, rng, replace=True):
    index = rng.choice(len(y), size=int(size), replace=replace)
    return np.asarray(x)[index], np.asarray(y)[index]


def fisher_information_at(beta, x_std):
    x_std = np.asarray(x_std, dtype=float)
    z = np.column_stack([np.ones_like(x_std), x_std])
    p = expit(beta[0] + beta[1] * x_std)
    return (z.T * (p * (1.0 - p))) @ z / len(x_std)


def target_information(precision, beta, target):
    """Reduce a coefficient precision matrix to a scalar target information."""
    precision = np.asarray(precision, dtype=float)
    covariance = np.linalg.inv(precision)
    if target == "beta_joint":
        return float(np.trace(precision))
    if target == "beta1":
        return float(1.0 / covariance[1, 1])
    x_low = float(standardized_x(0.0))
    x_high = float(standardized_x(2.0))
    p_low = float(expit(beta[0] + beta[1] * x_low))
    p_high = float(expit(beta[0] + beta[1] * x_high))
    g_low = p_low * (1.0 - p_low) * np.asarray([1.0, x_low])
    g_high = p_high * (1.0 - p_high) * np.asarray([1.0, x_high])
    if target == "prediction_joint":
        jacobian = np.vstack([g_low, g_high])
        transformed_cov = jacobian @ covariance @ jacobian.T
        return float(np.trace(np.linalg.inv(transformed_cov)))
    if target == "risk_difference":
        gradient = g_high - g_low
    elif target.startswith("p_x_"):
        raw = float(target.removeprefix("p_x_").replace("p", ".").replace("m", "-"))
        x_value = float(standardized_x(raw))
        probability = float(expit(beta[0] + beta[1] * x_value))
        gradient = probability * (1.0 - probability) * np.asarray([1.0, x_value])
    else:
        raise ValueError(f"Unknown target: {target}")
    return float(1.0 / (gradient @ covariance @ gradient))
