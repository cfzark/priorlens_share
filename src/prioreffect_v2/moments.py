"""Loss-specific moment existence and deterministic marginal risk summaries.

Checks are independent of inference: an IS/MCMC cloud cannot certify its own
tail moments. No checks are imposed on CP merely because Measure has moments.
"""
import numpy as np
from scipy.special import digamma, polygamma

from .core import (Coordinate, TargetSpec, DefinitionError, UndefinedMoment,
                   UncertifiedMoment, NumericalFailure)


BUILTINS = {"normal_known", "poisson_gamma", "beta_binomial", "nig", "eight_schools", "lupus_logistic"}


def check_target_moments(model, data, prior, target, order, *, sample_size=None):
    """Certify finite absolute coordinate moments (order 1 or 2).

    sample_size is for built-in static planning only; actual candidate propriety
    is checked separately at fit time. External adapters must provide a hook.
    """
    if order not in (1, 2):
        raise DefinitionError("Moment order must be 1 or 2")
    if model.name not in BUILTINS:
        hook = getattr(model, "check_target_moments", None)
        if hook is None:
            raise UncertifiedMoment("External adapter must implement check_target_moments(data, prior, target, order)")
        audit = hook(data, prior, target, order)
        if not isinstance(audit, dict) or audit.get("certified") is not True:
            raise UncertifiedMoment("Adapter did not certify the requested target moments")
        return audit
    n = len(data["y"]) if sample_size is None else int(sample_size)
    rows = []
    for c in target.coordinates:
        if model.name not in {"eight_schools", "lupus_logistic"} and c.variable not in model.variables:
            raise DefinitionError("Moment target is not a declared model variable")
        threshold, power = None, order
        rule = "finite_all_orders"
        if model.name in {"normal_known", "nig"} and c.variable == "mu":
            if c.transform != "identity":
                raise DefinitionError("Normal-location moment target requires identity (affine) coordinates")
            if model.name == "nig":
                a = nig_shape(prior, n)
                threshold, rule = 2*a, "Student_t_absolute_order_less_than_df"
        elif model.name == "nig":
            if c.variable != "sigma2" or c.transform not in {"identity", "square", "log"}:
                raise DefinitionError("Unsupported NIG moment coordinate")
            a = nig_shape(prior, n)
            if c.transform != "log":
                threshold = a
                power = order * (2 if c.transform == "square" else 1)
                rule = "inverse_gamma_power_less_than_shape"
        elif model.name == "eight_schools":
            if c.variable == "tau":
                if c.transform not in {"identity", "square", "log"}:
                    raise DefinitionError("Unsupported tau moment coordinate")
                if c.transform != "log" and prior["tau_kind"] in {"half_t", "half_cauchy"}:
                    df = 1 if prior["tau_kind"] == "half_cauchy" else prior["tau_df"]
                    threshold, power = n + df, order * (2 if c.transform == "square" else 1)
                    rule = "proper_normal_mu_prior_tau_tail_power_less_than_n_plus_df"
            elif (c.variable != "mu" and not c.variable.startswith("theta[")) or c.transform != "identity":
                raise DefinitionError("Unsupported Schools moment coordinate")
        elif model.name == "lupus_logistic":
            probability = c.variable.startswith("p_x_")
            bounded = probability and c.transform in {"identity", "square"} or c.variable == "risk_difference" and c.transform == "identity"
            if not bounded:
                if not (c.variable in model.variables and c.transform == "identity" or probability and c.transform in {"log", "logit"}):
                    raise DefinitionError("Unsupported logistic moment coordinate")
                # Bounded likelihood + proper prior certifies these moments.
                # A low-df prior does NOT prove posterior moment divergence:
                # the likelihood may improve tails; require a separate proof.
                if prior.get("df", 7) <= order:
                    raise UncertifiedMoment("Low-df logistic prior: posterior tail proof required; finite grid moments are not a certificate")
                rule = "bounded_likelihood_and_finite_prior_moment"
        elif model.name == "poisson_gamma" and c.transform not in {"identity", "square", "log"}:
            raise DefinitionError("Unsupported positive-rate moment coordinate")
        elif model.name == "beta_binomial" and c.transform not in {"identity", "square", "log", "logit"}:
            raise DefinitionError("Unsupported probability moment coordinate")
        row = {"variable": c.variable, "transform": c.transform, "order": order,
               "effective_power": power, "strict_upper_bound": threshold, "rule": rule}
        if threshold is not None and not power < threshold:
            error = UndefinedMoment(f"{c.variable}/{c.transform}: required power {power} must be < {threshold}; candidate size={n}")
            error.context = {"moment_requirement": row, "sample_size": n, "prior_family": prior["family"]}
            raise error
        rows.append(row)
    return {"certified": True, "sample_size": n, "requirements": rows}


def nig_shape(prior, n):
    if prior["family"] == "nig":
        return prior["kappa"] + n/2
    return (n-1)/2 + (prior["kappa"] if prior["family"] == "flat_mu_ig" else 0)


def marginal_summary(post, target, order, draws=1024, seed=123):
    """Return means and diagonal variances, NEVER a purported full covariance.

    Analytic conjugate marginals and collapsed-grid Schools eliminate posterior
    resampling noise. Weighted IS/MCMC measures retain their inference error.
    For order=1, unused variances are zero and are explicitly labelled as such.
    """
    if hasattr(post.model, "validate_target"):
        post.model.validate_target(target, post.data)
    audit = check_target_moments(post.model, post.data, post.prior, target, order)
    means, variances, engines = [], [], []
    for c in target.coordinates:
        if post.engine == "analytic" and post.model.name in {"normal_known", "poisson_gamma", "beta_binomial", "nig"}:
            mean, variance = conjugate_moments(post, c, order)
            engine = "analytic_marginals"
        elif post.engine == "collapsed_grid" and post.model.name == "eight_schools":
            mean, variance = school_moments(post, c, target.focal_unit_id, order)
            engine = "collapsed_grid_marginals"
        else:
            scalar = TargetSpec("risk_moment_"+c.variable, (c,), target.focal_unit_id)
            measure = post.target(scalar, draws, seed)
            mean = float(measure.mean[0])
            variance = float(measure.covariance[0, 0]) if order == 2 else 0.
            engine = ("collapsed_grid_marginals" if post.engine == "collapsed_grid" else
                      "weighted_marginals" if hasattr(post, "weights") else "sampled_marginals")
        if not np.isfinite(mean) or not np.isfinite(variance) or variance < -1e-10:
            raise NumericalFailure("Certified finite risk moments were not numerically resolved")
        means.append(mean)
        variances.append(max(0., variance))
        engines.append(engine)
    return np.asarray(means), np.asarray(variances), {**audit, "engines": engines, "variance_used": order == 2}


def school_moments(post, c, focal_unit_id, order):
    """Integrate Gaussian conditional moments directly over the tau grid."""
    if c.variable == "tau":
        values = c.forward(np.exp(post.eta))
        mean = float(post.masses @ values)
        variance = float(post.masses @ ((values-mean)**2)) if order == 2 else 0.
        return mean, variance
    cm, cv = post.mu_cond_mean, post.mu_cond_var
    if c.variable.startswith("theta["):
        j = post.data["unit_ids"].index(focal_unit_id)
        v = np.exp(2*post.eta)
        w = post.s[j]**2/(post.s[j]**2+v)
        cm, cv = w*cm+(1-w)*post.y[j], v*w+w**2*cv
    cm, cv = cm/c.scale, cv/c.scale**2
    mean = float(post.masses @ cm)
    variance = float(post.masses @ (cv+(cm-mean)**2)) if order == 2 else 0.
    return mean, variance


def conjugate_moments(post, c, order):
    """Closed-form mean and (when requested) variance after supported transforms."""
    from scipy import stats
    if post.model.name == "nig":
        p = post.params
        rv = (stats.invgamma(p["kappa"], scale=p["eta"]) if c.variable == "sigma2" else
              stats.t(2*p["kappa"], loc=p["mu"], scale=np.sqrt(p["eta"]/(p["kappa"]*p["lambda"]))))
    else:
        rv = post.rv
    if c.transform == "identity":
        mean, variance = rv.mean(), rv.var() if order == 2 else 0.
    elif c.transform == "square":
        mean = rv.moment(2)
        variance = rv.moment(4)-mean**2 if order == 2 else 0.
    elif c.transform == "logit":
        a, b = rv.args[:2]
        mean = digamma(a)-digamma(b)
        variance = polygamma(1, a)+polygamma(1, b) if order == 2 else 0.
    elif c.transform == "log":
        if rv.dist.name == "invgamma":
            a = rv.args[0]
            mean = np.log(rv.kwds["scale"])-digamma(a)
            variance = polygamma(1, a) if order == 2 else 0.
        elif rv.dist.name == "gamma":
            a = rv.args[0]
            mean = digamma(a)+np.log(rv.kwds["scale"])
            variance = polygamma(1, a) if order == 2 else 0.
        elif rv.dist.name == "beta":
            a, b = rv.args[:2]
            mean = digamma(a)-digamma(a+b)
            variance = polygamma(1, a)-polygamma(1, a+b) if order == 2 else 0.
        else:
            raise DefinitionError("Unsupported analytic log moment")
        mean -= np.log(c.reference)
    else:
        raise DefinitionError("Unsupported analytic risk transform")
    return float(mean/c.scale), float(variance/c.scale**2)
