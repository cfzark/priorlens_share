"""PDF-ranked HDRs with weighted posterior mass, never ranked IS weights.

The density is evaluated in the declared target coordinates. Its unknown
normalizing constant cancels between each region's threshold and membership
test. Finite weighted support approximates both calibration and coverage;
this is not an exact continuous-mass integration certificate.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import DefinitionError, NumericalFailure


def _scores(log_density, points):
    scores = np.atleast_1d(np.asarray(log_density(points), dtype=float))
    if scores.shape != (len(points),) or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise NumericalFailure("HDR density must return one valid log density per point")
    # Minus infinity is legitimate outside the density's support.
    return scores


@dataclass(frozen=True)
class DensityHDR:
    log_density: Callable
    dimension: int
    log_cutoff: float
    nominal_mass: float
    own_mass: float
    strict_mass: float
    boundary_mass: float
    max_atom_weight: float
    density_source: str

    def mass(self, measure):
        if measure.values.shape[1] != self.dimension:
            raise DefinitionError("Region and posterior target dimensions differ")
        positive = measure.weights > 0
        inside = _scores(self.log_density, measure.values[positive]) >= self.log_cutoff
        return float(measure.weights[positive] @ inside)

    def diagnostics(self):
        return {
            "algorithm": "pdf_ranked_hdr_v1", "dimension": self.dimension,
            "log_cutoff": self.log_cutoff, "density_source": self.density_source,
            "cutoff_rule": "descending_density_weighted_mass_no_interpolation",
            "boundary_rule": "include_all_equal_density_points",
            "nominal_mass": self.nominal_mass, "own_mass": self.own_mass,
            "mass_error": self.own_mass - self.nominal_mass,
            "strict_interior_mass": self.strict_mass, "boundary_mass": self.boundary_mass,
            "max_atom_weight": self.max_atom_weight,
            "mass_evaluation": "weighted_posterior_support",
        }


def density_hdr_region(measure, alpha=.05):
    """Highest-density superlevel set with weighted own mass >= 1-alpha.

    Sort actual target density values descending and include the first level
    whose cumulative posterior mass reaches the nominal value. Include all
    density ties, so excess mass is bounded by the boundary tie group (not
    necessarily by one atom). Log evaluation avoids PDF underflow and has
    exactly the same ranking as the PDF; there is no log/PDF interpolation.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise DefinitionError("alpha must lie in (0,1)")
    dimension = measure.values.shape[1]
    if dimension not in (1, 2):
        raise DefinitionError("PDF HDR supports scalar and 2D targets")
    log_density = measure.logpdf if measure.logpdf is not None else measure.log_kernel
    if log_density is None:
        raise DefinitionError("pdf_hdr requires a target logpdf or log_kernel; "
                              "a joint density is not a marginal density; no KDE fallback")
    positive = measure.weights > 0
    weights = measure.weights[positive]
    scores = _scores(log_density, measure.values[positive])
    if not np.all(np.isfinite(scores)):
        raise NumericalFailure("Positive posterior weight at zero density in HDR calibration")
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, 1 - alpha, side="left")), len(order) - 1)
    cutoff = float(scores[order[index]])
    inside, boundary = scores >= cutoff, scores == cutoff
    return DensityHDR(log_density, dimension, cutoff, 1 - alpha,
                      float(weights @ inside), float(weights @ (scores > cutoff)),
                      float(weights @ boundary), float(weights.max()),
                      "normalized_logpdf" if measure.logpdf is not None else "unnormalized_log_kernel")
