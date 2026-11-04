from abc import ABC, abstractmethod
import numpy as np


class PosteriorRiskMetric(ABC):
    """
    Risk/loss D(posterior, theta_star) used by Reimherr-style PSS methods.

    This is intentionally separate from metrics.DistanceMetric, which compares
    two posterior distributions. Reimherr's D is a posterior loss relative to
    the data-generating parameter.
    """

    @abstractmethod
    def compute(self, posterior, theta_star, rng=None):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def _posterior_mean(posterior):
    return float(posterior.mean())


def _posterior_var(posterior):
    return float(posterior.var())


def _posterior_samples(posterior, size, rng):
    draws = posterior.sample(size=size, rng=rng)
    if isinstance(draws, tuple):
        draws = draws[0]
    return np.asarray(draws, dtype=float).ravel()


class SquaredPosteriorMeanError(PosteriorRiskMetric):
    """
    D = (E[theta | y] - theta_star)^2.

    This is the original loss used by the current local implementations.
    """

    def compute(self, posterior, theta_star, rng=None):
        err = _posterior_mean(posterior) - float(theta_star)
        return float(err * err)


class AbsolutePosteriorMeanError(PosteriorRiskMetric):
    """D = |E[theta | y] - theta_star|."""

    def compute(self, posterior, theta_star, rng=None):
        return float(abs(_posterior_mean(posterior) - float(theta_star)))


class PosteriorExpectedSquaredError(PosteriorRiskMetric):
    """
    D = E[(theta - theta_star)^2 | y].

    For scalar posteriors this is Var(theta | y) + bias^2.
    """

    def __init__(self, B=5000):
        self.B = int(B)

    def __repr__(self):
        return f"PosteriorExpectedSquaredError(B={self.B})"

    def compute(self, posterior, theta_star, rng=None):
        theta_star = float(theta_star)
        try:
            bias = _posterior_mean(posterior) - theta_star
            return float(_posterior_var(posterior) + bias * bias)
        except Exception:
            rng = np.random.default_rng() if rng is None else rng
            draws = _posterior_samples(posterior, self.B, rng)
            return float(np.mean((draws - theta_star) ** 2))


class PosteriorExpectedAbsoluteError(PosteriorRiskMetric):
    """D = E[|theta - theta_star| | y], estimated by Monte Carlo."""

    def __init__(self, B=5000):
        self.B = int(B)

    def __repr__(self):
        return f"PosteriorExpectedAbsoluteError(B={self.B})"

    def compute(self, posterior, theta_star, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        draws = _posterior_samples(posterior, self.B, rng)
        return float(np.mean(np.abs(draws - float(theta_star))))


class PosteriorVarianceRisk(PosteriorRiskMetric):
    """D = Var(theta | y), an uncertainty-only risk."""

    def compute(self, posterior, theta_star, rng=None):
        return _posterior_var(posterior)


def make_risk_metric(name="squared_mean", **kwargs):
    key = str(name).lower().replace("-", "_")
    if key in {"squared_mean", "squared_mean_error", "mse_mean", "l2"}:
        return SquaredPosteriorMeanError()
    if key in {"absolute_mean", "absolute_mean_error", "mae_mean", "l1"}:
        return AbsolutePosteriorMeanError()
    if key in {"posterior_mse", "expected_squared", "posterior_expected_squared"}:
        return PosteriorExpectedSquaredError(**kwargs)
    if key in {"posterior_mae", "expected_absolute", "posterior_expected_absolute"}:
        return PosteriorExpectedAbsoluteError(**kwargs)
    if key in {"posterior_variance", "variance", "uncertainty"}:
        return PosteriorVarianceRisk()
    raise ValueError(f"Unknown risk metric: {name}")
