"""Density-free central credible regions in the declared target coordinates.

A 2D region uses a common marginal tail probability t and calibrates
P(Q1(t) <= Z1 <= Q1(1-t), Q2(t) <= Z2 <= Q2(1-t)) >= 1-alpha
on the weighted joint posterior support. It is NOT the product of two
marginal (1-alpha) intervals, nor a minimum-volume or rotation-invariant HDR.
"""
from dataclasses import dataclass
import numpy as np

from .core import DefinitionError


@dataclass(frozen=True)
class CentralRegion:
    lower: np.ndarray
    upper: np.ndarray
    tail_probability: float
    nominal_mass: float
    own_mass: float

    def mass(self, measure):
        if measure.values.shape[1] != len(self.lower):
            raise DefinitionError("Region and posterior target dimensions differ")
        if len(self.lower) == 1:
            return float(measure.interval_mass(self.lower[0], self.upper[0]))
        inside = np.all((measure.values >= self.lower) & (measure.values <= self.upper), axis=1)
        return float(measure.weights @ inside)

    def diagnostics(self):
        return {"algorithm": "central_equal_tail_rectangle_v1", "lower": self.lower.tolist(),
                "upper": self.upper.tolist(), "marginal_tail_probability": self.tail_probability,
                "nominal_mass": self.nominal_mass, "own_mass": self.own_mass,
                "mass_error": self.own_mass - self.nominal_mass,
                "mass_evaluation": "scalar_cdf_if_available_else_weighted_support" if len(self.lower) == 1 else "weighted_joint_support"}


def central_region(measure, alpha=.05, quantile_rule="weighted_cdf"):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise DefinitionError("alpha must lie in (0,1)")
    if quantile_rule not in {"weighted_cdf", "numpy_linear"}:
        raise DefinitionError("Unknown central-region quantile rule")
    values, weights = measure.values, measure.weights
    dimension = values.shape[1]
    if dimension not in (1, 2):
        raise DefinitionError("Central regions support scalar and 2D targets")
    if quantile_rule == "numpy_linear" and not np.allclose(weights, 1 / len(weights)):
        raise DefinitionError("numpy_linear requires uniformly weighted empirical samples")
    if dimension == 1:
        probs = [alpha / 2, 1 - alpha / 2]
        a, b = (measure.quantile(probs) if measure.ppf or quantile_rule == "weighted_cdf"
                else np.quantile(values[:, 0], probs))
        return CentralRegion(np.array([a]), np.array([b]), alpha / 2, 1 - alpha,
                             float(measure.interval_mass(a, b)))

    # Zero-weight support points cannot define credible-region endpoints.
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    orders = [np.argsort(values[:, j]) for j in range(dimension)]
    sorted_values = [values[ix, j] for j, ix in enumerate(orders)]
    cdfs = [np.cumsum(weights[ix]) / weights.sum() for ix in orders]

    def bounds(t):
        if quantile_rule == "numpy_linear":
            limits = [np.quantile(v, [t, 1 - t]) for v in sorted_values]
        else:
            limits = [np.interp([t, 1 - t], cdf, v) for cdf, v in zip(cdfs, sorted_values)]
        return np.asarray(limits)[:, 0], np.asarray(limits)[:, 1]

    def evaluate(t):
        lower, upper = bounds(t)
        mass = float(weights @ np.all((values >= lower) & (values <= upper), axis=1))
        return CentralRegion(lower, upper, t, 1 - alpha, mass)

    # Retain the last feasible rectangle at a discrete jump. Degenerate/tied
    # supports can overshoot nominal mass substantially; report, never conceal.
    lower_t, upper_t = 0., .5
    best = evaluate(lower_t)
    for _ in range(52):
        t = (lower_t + upper_t) / 2
        candidate = evaluate(t)
        if candidate.own_mass >= 1 - alpha:
            lower_t, best = t, candidate
        else:
            upper_t = t
    return best
