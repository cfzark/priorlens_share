"""
Unified entry point for prior-effect method classes.

Notebooks should import method runners from here instead of redefining classes
inside notebook cells. The statistical implementations remain in their
method-specific modules; this file only centralizes the public interface.
"""

try:
    from .Clarke import ClarkePriorSizeRunner1D
    from .Jones import JonesObservedPriorRunner1D, JonesObservedPriorRunnerPoisson
    from .cp import (
        CoverageProbabilityEPSSRunner1D,
        CoverageProbabilityExactPoissonGammaRunner,
    )
    from .beta_binomial import CoverageProbabilityExactBetaBinomialRunner
    from .morita import MoritaESSRunner1D
    from .reimherr import ReimherrRunner1D
    from .wiesenfarth import WiesenfarthCalderazzoECSSRunner1D
    from .wim import NormalNormalWIMRunner1D
except ImportError:  # pragma: no cover
    from Clarke import ClarkePriorSizeRunner1D
    from Jones import JonesObservedPriorRunner1D, JonesObservedPriorRunnerPoisson
    from cp import CoverageProbabilityEPSSRunner1D, CoverageProbabilityExactPoissonGammaRunner
    from beta_binomial import CoverageProbabilityExactBetaBinomialRunner
    from morita import MoritaESSRunner1D
    from reimherr import ReimherrRunner1D
    from wiesenfarth import WiesenfarthCalderazzoECSSRunner1D
    from wim import NormalNormalWIMRunner1D


METHOD_RUNNERS = {
    "clarke": ClarkePriorSizeRunner1D,
    "jones": JonesObservedPriorRunner1D,
    "jones_poisson": JonesObservedPriorRunnerPoisson,
    "cp": CoverageProbabilityEPSSRunner1D,
    "cp_poisson_gamma": CoverageProbabilityExactPoissonGammaRunner,
    "cp_beta_binomial": CoverageProbabilityExactBetaBinomialRunner,
    "morita": MoritaESSRunner1D,
    "reimherr": ReimherrRunner1D,
    "wiesenfarth": WiesenfarthCalderazzoECSSRunner1D,
    "wim": NormalNormalWIMRunner1D,
}


def get_method_runner(name):
    key = str(name).lower()
    try:
        return METHOD_RUNNERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(METHOD_RUNNERS))
        raise ValueError(f"Unknown method runner '{name}'. Available: {available}") from exc


def available_method_runners():
    return tuple(sorted(METHOD_RUNNERS))
