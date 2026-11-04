"""Public facade; numerical kernels retain their versioned namespace."""
from prioreffect_v2.core import TargetSpec, Coordinate, Measure, DefinitionError, NumericalFailure
from prioreffect_v2.models import BuiltinModel
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.losses import Wasserstein, Coverage
from .api import validate, run
from .stan_posterior import posterior, validate_posterior

__version__ = "0.1.0"
__all__ = ["TargetSpec", "Coordinate", "Measure", "DefinitionError", "NumericalFailure",
           "BuiltinModel", "PosteriorProvider", "Wasserstein", "Coverage", "validate", "run",
           "posterior", "validate_posterior"]
