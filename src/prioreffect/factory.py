"""
Small factory layer for experiments.

The goal is to make notebooks and batch scripts specify names and parameters,
then receive the right model, prior, metric, or runner object without repeating
construction boilerplate.
"""

try:
    from .distributions import *
    from .models import *
    from .metrics import *
    from .Clarke import ClarkePriorSizeRunner1D
    from .Jones import JonesObservedPriorRunner1D, JonesObservedPriorRunnerPoisson
    from .cp import CoverageProbabilityEPSSRunner1D, CoverageProbabilityExactPoissonGammaRunner
    from .cp import MultiAlphaCoverageProbabilityEPSSRunner1D, PosteriorPredictiveCoverageProbabilityEPSSRunner1D
    from .morita import MoritaESSRunner1D
    from .reimherr import ReimherrRunner1D
    from .wiesenfarth import WiesenfarthCalderazzoECSSRunner1D
    from .wim import NormalNormalWIMRunner1D
    from .beta_binomial import CoverageProbabilityExactBetaBinomialRunner
except ImportError:  # pragma: no cover
    from distributions import *
    from models import *
    from metrics import *
    from Clarke import ClarkePriorSizeRunner1D
    from Jones import JonesObservedPriorRunner1D, JonesObservedPriorRunnerPoisson
    from cp import CoverageProbabilityEPSSRunner1D, CoverageProbabilityExactPoissonGammaRunner
    from cp import MultiAlphaCoverageProbabilityEPSSRunner1D, PosteriorPredictiveCoverageProbabilityEPSSRunner1D
    from morita import MoritaESSRunner1D
    from reimherr import ReimherrRunner1D
    from wiesenfarth import WiesenfarthCalderazzoECSSRunner1D
    from wim import NormalNormalWIMRunner1D
    from beta_binomial import CoverageProbabilityExactBetaBinomialRunner


def make_model(name, **kwargs):
    key = name.lower()
    if key in {"normal_known", "normal_mean_known_var"}:
        return NormalMeanKnownVarModel(sigma=kwargs.pop("sigma"))
    if key in {"t_location", "student_t_location", "t_location_is"}:
        return TLocationModelIS(**kwargs)
    if key in {"normal_unknown", "normal_mean_unknown_var"}:
        return NormalMeanUnknownVarModel()
    if key in {"poisson", "poisson_gamma"}:
        return PoissonModel()
    if key in {"binomial", "beta_binomial"}:
        return BinomialModel(trials=kwargs.pop("trials"))
    raise ValueError(f"Unknown model name: {name}")


def make_prior(name, **kwargs):
    key = name.lower()
    if key in {"normal", "normal_mu"}:
        return NormalDist(kwargs.pop("mu"), kwargs.pop("sd"))
    if key in {"student_t", "t"}:
        return StudentTDist(kwargs.pop("df"), kwargs.pop("loc", 0.0), kwargs.pop("scale", 1.0))
    if key in {"gamma", "gamma_rate"}:
        return GammaDist(kwargs.pop("alpha"), kwargs.pop("beta"))
    if key == "beta":
        return BetaDist(kwargs.pop("alpha"), kwargs.pop("beta"))
    if key in {"inverse_gamma", "invgamma"}:
        return InverseGammaDist(kwargs.pop("shape"), kwargs.pop("scale"))
    if key in {"nig", "normal_inverse_gamma"}:
        return NormalInverseGammaDist(
            mu0=kwargs.pop("mu0"),
            tau0_sq=kwargs.pop("tau0_sq"),
            kappa=kwargs.pop("kappa"),
            eta=kwargs.pop("eta"),
        )
    if key in {"flat", "flat_mu", "flat_lambda"}:
        return FlatPrior()
    if key in {"jeffreys", "jeffreys_mu", "jeffreys_lambda"}:
        return JeffreysPrior()
    if key in {"flat_mu_ig", "flat_mu_inverse_gamma"}:
        return FlatMuInverseGammaPrior(kappa=kwargs.pop("kappa"), eta=kwargs.pop("eta"))
    raise ValueError(f"Unknown prior name: {name}")


def make_metric(name, **kwargs):
    key = name.lower()
    if key in {"kl", "kld", "kl_divergence"}:
        return KLDivergence(**kwargs)
    if key in {"truncated_kl", "tkl"}:
        return TruncatedKLDivergence(**kwargs)
    if key in {"wasserstein", "w"}:
        return WassersteinDistance(**kwargs)
    if key in {"w1_normal", "wasserstein1_normal"}:
        return Wasserstein1Normal()
    if key in {"mean_difference", "mean_diff"}:
        return MeanDifference()
    if key in {"variance_difference", "var_diff"}:
        return VarianceDifference()
    if key in {"objective", "cp_objective"}:
        return ObjectiveFunctionMetric(**kwargs)
    raise ValueError(f"Unknown metric name: {name}")


def make_runner(name, **kwargs):
    key = name.lower()
    if key in {"clarke", "clarke_prior_size"}:
        return ClarkePriorSizeRunner1D(**kwargs)
    if key in {"jones", "jones_normal", "opess"}:
        return JonesObservedPriorRunner1D(**kwargs)
    if key in {"jones_poisson", "opess_poisson"}:
        return JonesObservedPriorRunnerPoisson(**kwargs)
    if key in {"cp", "coverage_probability"}:
        return CoverageProbabilityEPSSRunner1D(**kwargs)
    if key in {"multi_alpha_cp", "multi_alpha_coverage_probability"}:
        return MultiAlphaCoverageProbabilityEPSSRunner1D(**kwargs)
    if key in {"ppd_cp", "posterior_predictive_cp", "posterior_predictive_coverage_probability"}:
        return PosteriorPredictiveCoverageProbabilityEPSSRunner1D(**kwargs)
    if key in {"cp_poisson", "coverage_probability_poisson"}:
        return CoverageProbabilityExactPoissonGammaRunner(**kwargs)
    if key in {"cp_beta_binomial", "coverage_probability_beta_binomial"}:
        return CoverageProbabilityExactBetaBinomialRunner(**kwargs)
    if key == "morita":
        return MoritaESSRunner1D(**kwargs)
    if key == "reimherr":
        return ReimherrRunner1D(**kwargs)
    if key in {"wiesenfarth", "wiesenfarth_calderazzo", "ecss"}:
        return WiesenfarthCalderazzoECSSRunner1D(**kwargs)
    if key == "wim":
        return NormalNormalWIMRunner1D(**kwargs)
    raise ValueError(f"Unknown runner name: {name}")
