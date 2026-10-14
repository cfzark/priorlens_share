"""Risk reference and pseudo-data components, independent of matching direction.

These are project risk estimators, not claims of source-identical reproductions.
Unresolved low-level calls retain the old ``legacy_plugin`` reference rule;
public presets explicitly select ``baseline_posterior_mean``.
"""
from dataclasses import dataclass, field

import numpy as np

from .core import Coordinate, TargetSpec, DefinitionError, InadmissiblePosterior, UncertifiedMoment, NumericalFailure, seed_for, digest, sampling_data
from .moments import check_target_moments, marginal_summary, BUILTINS


SAMPLING = {"bootstrap", "likelihood"}
REFERENCE_RULES = {"baseline_posterior_mean", "legacy_plugin"}
RISK_METRICS = {"posterior_mse", "squared_mean", "posterior_variance"}


def sampling_for(method):
    return method.get("sampling", "bootstrap" if method["name"] == "reimherr" else "likelihood")


def validate_risk_settings(method, model=None, target=None):
    policy = method.get("risk_invalid_policy", "error")
    if policy not in {"error", "redraw_baseline_proper"}:
        raise DefinitionError("Unknown risk_invalid_policy")
    if policy == "redraw_baseline_proper":
        if sampling_for(method) != "bootstrap" or (model is not None and model.name != "nig"):
            raise DefinitionError("Conditional proper-baseline resampling is currently NIG bootstrap only")
    if type(method.get("risk_max_redraws", 1000)) is not int or method.get("risk_max_redraws", 1000) <= 0:
        raise DefinitionError("risk_max_redraws must be a positive integer")
    if method.get("risk_coupling", "independent") not in {"independent", "shared_by_size"}:
        raise DefinitionError("risk_coupling must be independent or shared_by_size; nested sizes are not enabled")
    if method.get("risk_moments", "legacy_joint") not in {"marginal", "legacy_joint"}:
        raise DefinitionError("risk_moments must be marginal or legacy_joint")
    if "risk_data_seed" in method and (type(method["risk_data_seed"]) is not int or not 0 <= method["risk_data_seed"] < 2**64):
        raise DefinitionError("risk_data_seed must be an unsigned 64-bit integer")
    if sampling_for(method) not in SAMPLING:
        raise DefinitionError("Risk sampling must be bootstrap or likelihood (not posterior predictive)")
    if method.get("reference_rule", "legacy_plugin") not in REFERENCE_RULES:
        raise DefinitionError("Unknown risk reference_rule")
    if method.get("risk", "posterior_mse") not in RISK_METRICS:
        raise DefinitionError("Unknown risk metric")
    for key, dimension in (("theta_star", getattr(model, "dimension", None)),
                           ("target_reference", getattr(target, "dimension", None))):
        if key in method:
            value = np.asarray(method[key], dtype=float)
            if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)) or (dimension is not None and len(value) != dimension):
                raise DefinitionError(f"{key} must be a finite vector with the declared dimension")
    if model is not None:
        if "theta_star" in method:
            validate_theta(model, method["theta_star"])
        needs_theta = sampling_for(method) == "likelihood" or "target_reference" not in method
        if method.get("reference_rule", "legacy_plugin") == "legacy_plugin" and needs_theta and model.dimension > 1 and "theta_star" not in method:
            raise DefinitionError("legacy_plugin requires explicit theta_star for multivariate models")
    if target is not None and target.focal_unit_id:
        from .focal import PROTOCOL
        if model is None or model.name != "eight_schools" or method.get("focal_protocol") != PROTOCOL:
            raise DefinitionError("Focal-school risk requires explicit focal_protocol: fixed_observed_anchor_v1")


def validate_theta(model, theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (model.dimension,) or not np.all(np.isfinite(theta)):
        raise DefinitionError("theta_star must be a finite vector in model.variables order")
    invalid = ((model.name == "nig" and theta[-1] <= 0)
               or (model.name in {"poisson_gamma", "eight_schools"} and theta[-1] < 0)
               or (model.name == "beta_binomial" and not 0 <= theta[0] <= 1))
    if invalid:
        raise DefinitionError("theta_star is outside the model's natural parameter domain")
    if hasattr(model, "validate_risk_theta"):
        model.validate_risk_theta(theta)
    return theta


def resolve_reference(model, data, base, target, method, provider, seed):
    """Resolve E_b[T(theta)|y] separately from the likelihood plug-in parameter.

    Means are obtained from scalar marginal targets, preserving analytic/grid
    moments when available instead of introducing joint resampling error.
    No posterior-predictive draw is used here.
    """
    validate_risk_settings(method, model, target)
    rule = method.get("reference_rule", "legacy_plugin")
    draws = method.get("posterior_draws", 1024)
    posterior = None
    reference_seed = method.get("risk_data_seed", seed) if method.get("risk_coupling") == "shared_by_size" else seed

    def marginal_means(coordinates):
        nonlocal posterior
        for coordinate in coordinates:
            focal = target.focal_unit_id if coordinate.variable.startswith("theta[") else None
            check_target_moments(model, data, base, TargetSpec("reference", (coordinate,), focal), 1)
        if posterior is None:
            posterior = provider.fit(data, base, seed_for(reference_seed, "risk_reference_fit"))
        means = []
        for coordinate in coordinates:
            scalar = TargetSpec("risk_reference_" + coordinate.variable, (coordinate,), target.focal_unit_id if coordinate.variable.startswith("theta[") else None)
            target_seed = seed_for(reference_seed, "risk_reference_target", coordinate.variable)
            if method.get("risk_moments", "legacy_joint") == "marginal":
                mean, _, _ = marginal_summary(posterior, scalar, 1, draws, target_seed)
                means.append(float(mean[0]))
            else:
                measure = posterior.target(scalar, draws, target_seed)
                means.append(float(measure.mean[0]))
        if not np.all(np.isfinite(means)):
            raise NumericalFailure("Baseline posterior risk reference has no resolved finite mean")
        return np.asarray(means)

    theta = method.get("theta_star")
    theta_source = "explicit" if theta is not None else None
    need_theta = sampling_for(method) == "likelihood" or (rule == "legacy_plugin" and "target_reference" not in method)
    if theta is None and need_theta:
        if rule == "baseline_posterior_mean":
            theta = marginal_means([Coordinate(v) for v in model.variables])
            theta_source = "baseline_natural_parameter_mean"
        else:
            if model.dimension != 1:
                raise DefinitionError("legacy_plugin requires explicit multivariate theta_star")
            center = np.mean(data["y"])
            if model.name == "beta_binomial":
                center /= model.options.get("trials", 20)
            theta, theta_source = [center], "legacy_observed_mle"
    if theta is not None:
        theta = validate_theta(model, theta)
    if "target_reference" in method:
        reference, reference_source = np.asarray(method["target_reference"], dtype=float), "explicit"
    elif rule == "baseline_posterior_mean":
        reference = marginal_means(target.coordinates)
        reference_source = "baseline_target_posterior_mean"
    else:
        try:
            reference = target.values({v: np.array([theta[i]]) for i, v in enumerate(model.variables)})[0]
        except KeyError as error:
            raise DefinitionError("Derived target with legacy_plugin requires explicit target_reference") from error
        reference_source = "transform_of_plugin"
    if not np.all(np.isfinite(reference)):
        raise NumericalFailure("Risk target reference is nonfinite")
    return theta, reference, {"reference_rule": rule, "target_reference": reference.tolist(),
        "target_reference_source": reference_source, "theta_star": None if theta is None else theta.tolist(),
        "theta_star_source": theta_source, "theta_used_for_sampling": sampling_for(method) == "likelihood"}


@dataclass
class RiskSampler:
    model: object
    data: dict
    sampling: str
    theta: object = None
    focal_unit_id: str | None = None
    invalid_policy: str = "error"
    admissibility_prior: dict | None = None
    max_redraws: int = 1000
    counts: dict = field(default_factory=dict)

    def draw(self, count, rng):
        if self.invalid_policy == "error":
            return self._draw(count, rng)
        if (self.invalid_policy != "redraw_baseline_proper" or self.model.name != "nig"
                or self.sampling != "bootstrap" or self.admissibility_prior is None):
            raise DefinitionError("Invalid conditional bootstrap contract")
        if count < 2 or np.ptp(self.data["y"]) == 0:
            raise InadmissiblePosterior("No proper Jeffreys bootstrap candidate is attainable")
        audit = self.counts.setdefault(str(int(count)), {"accepted": 0, "rejected": 0})
        for attempt in range(self.max_redraws + 1):
            data = self._draw(count, rng)
            try:
                self.model.check_posterior_admissibility(data, self.admissibility_prior)
            except InadmissiblePosterior:
                audit["rejected"] += 1
                if attempt == self.max_redraws:
                    raise InadmissiblePosterior("Conditional bootstrap redraw limit exceeded")
            else:
                audit["accepted"] += 1
                return data

    def diagnostics(self):
        return {"policy": self.invalid_policy,
                "data_law": "conditional_on_baseline_posterior_proper" if self.invalid_policy != "error" else "unconditional",
                "admissibility_prior": self.admissibility_prior,
                "counts_by_size": self.counts, "max_redraws": self.max_redraws}

    def _draw(self, count, rng):
        if self.focal_unit_id:
            from .focal import FocalAnchor
            anchor = FocalAnchor(self.model, self.data, self.focal_unit_id)
            if count < 1:
                raise DefinitionError("Focal risk must retain its one anchor")
            if self.sampling == "bootstrap":
                if count > 1 and not len(anchor.pool):
                    raise DefinitionError("Cannot bootstrap auxiliaries from an empty pool")
                indices = rng.choice(anchor.pool, size=count - 1, replace=True)
                auxiliary = self.model.subset(self.data, indices, auxiliary=True)
            elif self.sampling == "likelihood" and self.theta is not None:
                # Resampling s values is design sampling, not duplication of y_j.
                auxiliary = self.model.simulate(self.theta, count - 1, rng, self.data)
            else:
                raise DefinitionError("Invalid focal risk sampler")
            return anchor.attach(auxiliary)
        if self.sampling == "bootstrap":
            indices = rng.integers(0, len(self.data["y"]), size=count)
            return self.model.subset(self.data, indices, auxiliary=True)
        if self.sampling == "likelihood" and self.theta is not None:
            return self.model.simulate(self.theta, count, rng, self.data)
        raise DefinitionError("Invalid risk sampler or missing likelihood theta_star")


class RiskDatasetBank:
    """Same law/size/replication -> same complete pseudo-dataset, no size nesting.

    No prior, method, target, candidate-grid position or B enters the draw key.
    Increasing B or refining a grid preserves all existing datasets. Planning
    supplies a data seed independent of prior-grid cells and method identifiers.
    """
    def __init__(self, sampler, seed):
        self.sampler, self.seed, self.cache = sampler, seed, {}
        signature = getattr(sampler.model, "sampling_signature", None)
        options = signature() if callable(signature) else getattr(sampler.model, "options", None)
        if options is None:
            raise DefinitionError("Shared risk data requires model.options or sampling_signature() covering all simulation settings")
        self.law = digest({"model": sampler.model.name, "options": options,
            "data": sampling_data(sampler.data), "sampling": sampler.sampling,
            "theta": sampler.theta if sampler.sampling == "likelihood" else None,
            **({"focal_unit_id": sampler.focal_unit_id, "protocol": "fixed_observed_anchor_v1"} if sampler.focal_unit_id else {}),
            **({"invalid_policy": sampler.invalid_policy, "admissibility_prior": sampler.admissibility_prior,
                "max_redraws": sampler.max_redraws} if sampler.invalid_policy != "error" else {})})

    def draw(self, count, b):
        key = (int(count), int(b))
        if key not in self.cache:
            rng = np.random.default_rng(seed_for(self.seed, "risk_shared_by_size_v1", self.law, *key))
            self.cache[key] = self.sampler.draw(count, rng)
        return self.cache[key]

    def checksums(self):
        return {str(count): digest([sampling_data(self.cache[(count, b)]) for c, b in sorted(self.cache) if c == count])
                for count in sorted({c for c, b in self.cache})}


def validate_risk_moments(model, data, info, base, target, method, grid):
    """Reject impossible built-in risk moment requirements before sampling/fitting."""
    if model.name not in BUILTINS:
        if not callable(getattr(model, "check_target_moments", None)):
            raise UncertifiedMoment("External risk adapter must provide check_target_moments")
        return  # The adapter is checked against each real dataset at runtime.
    if method.get("reference_rule", "legacy_plugin") == "baseline_posterior_mean":
        if "target_reference" not in method:
            check_target_moments(model, data, base, target, 1)
        if sampling_for(method) == "likelihood" and "theta_star" not in method:
            for variable in model.variables:
                check_target_moments(model, data, base, TargetSpec("plugin_reference", (Coordinate(variable),)), 1)
    order = 1 if method.get("risk", "posterior_mse") == "squared_mean" else 2
    n = len(data["y"])
    fixed, candidate = (info, base) if method["name"] == "reimherr" else (base, info)
    check_target_moments(model, data, fixed, target, order)
    for m in grid:
        count = n + int(m) if method["name"] == "reimherr" else n - int(m)
        if target.focal_unit_id and count < 1:
            raise DefinitionError("Focal risk cannot remove the anchor")
        try:
            check_target_moments(model, data, candidate, target, order, sample_size=count)
        except DefinitionError as error:
            error.context = {**getattr(error, "context", {}), "m": int(m), "stage": "risk_candidate_moment_preflight"}
            raise
