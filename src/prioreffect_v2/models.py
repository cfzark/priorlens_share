"""Built-in model contracts and conjugate posteriors (no v1 imports)."""
from __future__ import annotations

import numpy as np
import re
from scipy import stats
from scipy.special import expit, logsumexp, digamma, polygamma

from .core import DefinitionError, InadmissiblePosterior, NumericalFailure, Measure, Posterior


SCHOOLS_Y = [28., 8., -3., 7., -1., 1., 18., 12.]
SCHOOLS_S = [15., 10., 16., 11., 9., 11., 10., 18.]


def lupus_data():
    levels = np.arange(5) / 2
    controls, cases = [31, 2, 2, 0, 2], [5, 0, 2, 6, 5]
    x = np.repeat(levels, np.array(controls) + cases)
    y = np.concatenate([np.r_[np.zeros(a), np.ones(b)] for a, b in zip(controls, cases)])
    return {"y": y.tolist(), "x": ((x - x.mean()) / (2 * x.std(ddof=1))).tolist(),
            "x_center": float(x.mean()), "x_divisor": float(2 * x.std(ddof=1)),
            "unit_ids": [f"patient_{i+1}" for i in range(len(y))]}


class BuiltinModel:
    def __init__(self, name, **options):
        if name not in {"normal_known", "poisson_gamma", "beta_binomial", "nig", "eight_schools", "lupus_logistic"}:
            raise DefinitionError(f"Unknown built-in model {name}")
        self.name, self.options = name, options
        self.dimension = 1 if name in {"normal_known", "poisson_gamma", "beta_binomial"} else 2
        self.variables = {"normal_known": ["mu"], "poisson_gamma": ["lambda"], "beta_binomial": ["p"],
                          "nig": ["mu", "sigma2"], "eight_schools": ["mu", "tau"],
                          "lupus_logistic": ["beta0", "beta1"]}[name]
        self.unit = "school" if name == "eight_schools" else "observation"

    def validate_prior(self, prior):
        allowed = {"normal_known": {"normal", "flat", "jeffreys"}, "poisson_gamma": {"gamma", "flat", "jeffreys"},
                   "beta_binomial": {"beta", "flat", "jeffreys"}, "nig": {"nig", "flat_mu_ig", "jeffreys"},
                   "eight_schools": {"schools"}, "lupus_logistic": {"student_t"}}
        family = prior.get("family")
        if family not in allowed[self.name]:
            raise DefinitionError(f"Prior {family} is not supported for {self.name}")
        fields = {"normal": ["mean", "sd"], "gamma": ["shape", "rate"], "beta": ["a", "b"],
                  "nig": ["mu0", "tau0_sq", "kappa", "eta"], "flat_mu_ig": ["kappa", "eta"],
                  "schools": ["mu_mean", "mu_sd", "tau_kind", "tau_scale"], "student_t": ["loc", "scale", "df"]}.get(family, [])
        if any(k not in prior for k in fields):
            raise DefinitionError(f"{family} requires {fields}")
        for k, v in prior.items():
            if k in {"family", "tau_kind"}: continue
            if not np.all(np.isfinite(np.asarray(v, dtype=float))):
                raise DefinitionError(f"Prior field {k} must be finite")
            if k in {"sd", "shape", "rate", "a", "b", "tau0_sq", "kappa", "eta", "mu_sd", "tau_scale", "tau_df", "tau_log_sd", "scale", "df"} and np.any(np.asarray(v) <= 0):
                raise DefinitionError(f"Prior field {k} must be positive")
        if family == "student_t" and (len(prior["loc"]) != 2 or len(prior["scale"]) != 2):
            raise DefinitionError("Logistic prior vectors must have length two")
        if family == "schools":
            if prior["tau_kind"] not in {"half_t", "half_cauchy", "lognormal", "bounded_uniform"}:
                raise DefinitionError("Unknown tau prior")
            for kind, field in [("half_t", "tau_df"), ("lognormal", "tau_log_sd")]:
                if prior["tau_kind"] == kind and field not in prior:
                    raise DefinitionError(f"{kind} requires {field}")
        return dict(prior)

    def validate_data(self, data):
        data = dict(data)
        y = np.asarray(data["y"], dtype=float)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise DefinitionError("y must be a finite vector")
        for name in (["s"] if self.name == "eight_schools" else ["x"] if self.name == "lupus_logistic" else []):
            a = np.asarray(data[name], dtype=float)
            if a.shape != y.shape or not np.all(np.isfinite(a)):
                raise DefinitionError(f"{name} must align with y")
        if self.name == "eight_schools" and np.any(np.asarray(data["s"]) <= 0):
            raise DefinitionError("School standard errors must be positive")
        if self.name in {"poisson_gamma", "beta_binomial", "lupus_logistic"}:
            if np.any(y < 0) or np.any(y != np.floor(y)):
                raise DefinitionError("Discrete outcomes must be nonnegative integers")
            upper = self.options.get("trials", 20) if self.name == "beta_binomial" else 1 if self.name == "lupus_logistic" else np.inf
            if np.any(y > upper):
                raise DefinitionError("Outcome outside support")
        ids = data.get("unit_ids", [f"{self.unit}_{i+1}" for i in range(len(y))])
        if len(ids) != len(y) or len(set(ids)) != len(ids):
            raise DefinitionError("unit_ids must be unique and align with data")
        data["unit_ids"] = list(ids)
        data["y"] = y.tolist()
        if self.name == "eight_schools":
            # Never re-number theta[k] when a candidate deletes/reorders schools.
            data.setdefault("original_unit_ids", list(ids))
        if self.name == "lupus_logistic" and ({"x_center", "x_divisor"} & data.keys()):
            self.validate_prediction_scale(data)
        return data

    @staticmethod
    def validate_prediction_scale(data):
        if not {"x_center", "x_divisor"} <= data.keys():
            raise DefinitionError("Raw-x predictions require explicit x_center and x_divisor (use 0 and 1 for unscaled x)")
        if not all(np.isscalar(data[k]) and np.isfinite(data[k]) for k in ("x_center", "x_divisor")) or data["x_divisor"] <= 0:
            raise DefinitionError("x_center must be finite and x_divisor finite and positive")

    def validate_target(self, target, data):
        if self.name == "lupus_logistic":
            predictions = {"p_x_0", "p_x_0.5", "p_x_1", "p_x_1.5", "p_x_2", "risk_difference"}
            if any(c.variable not in set(self.variables) | predictions for c in target.coordinates):
                raise DefinitionError("Unknown logistic target")
            if any(c.variable in predictions for c in target.coordinates):
                self.validate_prediction_scale(data)
        if self.name != "eight_schools":
            return
        original = data.get("original_unit_ids", data["unit_ids"])
        for c in target.coordinates:
            if c.variable in self.variables:
                continue
            match = re.fullmatch(r"theta\[([1-9][0-9]*)\]", c.variable)
            if not match or int(match[1]) > len(original):
                raise DefinitionError("School target must be theta[k], with a valid 1-based original-school index")
            if target.dimension != 1 or target.focal_unit_id != original[int(match[1]) - 1]:
                raise DefinitionError("theta[k] and focal_unit_id must identify the same original school")
            if data["unit_ids"].count(target.focal_unit_id) != 1:
                raise DefinitionError("Focal original school must occur exactly once")

    def check_posterior_admissibility(self, data, prior):
        """Mathematical feasible domain; deliberately independent of the backend."""
        n = len(data["y"])
        family = prior["family"]
        if self.name == "nig" and family != "nig":
            if n == 0 or (family == "jeffreys" and (n < 2 or np.ptp(data["y"]) == 0)):
                raise InadmissiblePosterior("Improper NIG candidate posterior")
        if self.name in {"normal_known", "poisson_gamma"} and n == 0 and family in {"flat", "jeffreys"}:
            raise InadmissiblePosterior("Improper empty-data baseline posterior")

    def subset(self, data, indices, auxiliary=False):
        ix = np.asarray(indices, dtype=int)
        result = dict(data)
        for name in ["y", "s", "x", "unit_ids"]:
            if name in data:
                result[name] = np.asarray(data[name])[ix].tolist()
        if auxiliary:
            result["unit_ids"] = [f"aux_{i}" for i in range(len(ix))]
        return result

    def append(self, data, additions):
        result = dict(data)
        for name in ["y", "s", "x"]:
            if name in data:
                result[name] = list(data[name]) + list(additions[name])
        result["unit_ids"] = list(data["unit_ids"]) + [f"aux_{i}" for i in range(len(additions["y"]))]
        return result

    def simulate(self, theta, count, rng, design):
        count = int(count)
        if count < 0:
            raise DefinitionError("Negative pseudo sample size")
        out = {"unit_ids": [f"aux_{i}" for i in range(count)]}
        if self.name == "normal_known":
            y = rng.normal(theta[0], self.options.get("sigma", 1.), size=count)
        elif self.name == "poisson_gamma":
            y = rng.poisson(theta[0], size=count)
        elif self.name == "beta_binomial":
            y = rng.binomial(self.options.get("trials", 20), theta[0], size=count)
        elif self.name == "nig":
            y = rng.normal(theta[0], np.sqrt(theta[1]), size=count)
        elif self.name == "eight_schools":
            s = rng.choice(design["s"], size=count, replace=True)
            latent = rng.normal(theta[0], theta[1], size=count)
            y = rng.normal(latent, s)
            out["s"] = s.tolist()
        else:
            x = rng.choice(design["x"], size=count, replace=True)
            y = rng.binomial(1, expit(theta[0] + theta[1] * x))
            out["x"] = x.tolist()
            for key in ["x_center", "x_divisor"]:
                if key in design:
                    out[key] = design[key]
        out["y"] = y.tolist()
        return out

    def to_unconstrained(self, theta):
        z = np.array(theta, dtype=float, copy=True)
        if self.name in {"poisson_gamma", "nig", "eight_schools"}:
            z[..., -1] = np.log(z[..., -1])
        elif self.name == "beta_binomial":
            z[..., 0] = np.log(z[..., 0]) - np.log1p(-z[..., 0])
        return z

    def from_unconstrained(self, z):
        theta = np.array(z, dtype=float, copy=True)
        if self.name in {"poisson_gamma", "nig", "eight_schools"}:
            theta[..., -1] = np.exp(theta[..., -1])
        elif self.name == "beta_binomial":
            theta[..., 0] = expit(theta[..., 0])
        return theta

    def loglik(self, theta, data):
        theta = np.atleast_2d(theta)
        y = np.asarray(data["y"])
        if self.name in {"normal_known", "nig"}:
            sd = self.options.get("sigma", 1.) if self.name == "normal_known" else np.sqrt(theta[:, 1, None])
            return stats.norm.logpdf(y, theta[:, 0, None], sd).sum(axis=1)
        if self.name == "poisson_gamma":
            return stats.poisson.logpmf(y, theta[:, 0, None]).sum(axis=1)
        if self.name == "beta_binomial":
            return stats.binom.logpmf(y, self.options.get("trials", 20), theta[:, 0, None]).sum(axis=1)
        if self.name == "eight_schools":
            sd = np.sqrt(np.asarray(data["s"]) ** 2 + theta[:, 1, None] ** 2)
            return stats.norm.logpdf(y, theta[:, 0, None], sd).sum(axis=1)
        eta = theta[:, :1] + theta[:, 1, None] * np.asarray(data["x"])
        return (y * eta - np.logaddexp(0, eta)).sum(axis=1)

    def logprior(self, theta, prior):
        theta = np.atleast_2d(theta)
        family = prior["family"]
        if self.dimension == 1:
            x = theta[:, 0]
            if family == "flat":
                return np.zeros(len(x))
            if family == "jeffreys":
                return -0.5 * np.log(x) if self.name == "poisson_gamma" else -0.5 * (np.log(x) + np.log1p(-x)) if self.name == "beta_binomial" else np.zeros(len(x))
            return scalar_prior(prior).logpdf(x)
        if self.name == "nig":
            mu, v = theta.T
            if family == "jeffreys":
                return -np.log(v)
            lp = stats.invgamma.logpdf(v, prior["kappa"], scale=prior["eta"])
            return lp if family == "flat_mu_ig" else lp + stats.norm.logpdf(mu, prior["mu0"], np.sqrt(prior["tau0_sq"] * v))
        if self.name == "eight_schools":
            mu, tau = theta.T
            lp = stats.norm.logpdf(mu, prior["mu_mean"], prior["mu_sd"])
            kind, scale = prior["tau_kind"], prior["tau_scale"]
            if kind == "lognormal":
                lt = stats.lognorm.logpdf(tau, prior["tau_log_sd"], scale=scale)
            elif kind == "bounded_uniform":
                lt = np.where((tau > 0) & (tau <= scale), -np.log(scale), -np.inf)
            else:
                lt = np.log(2) + stats.t.logpdf(tau / scale, 1 if kind == "half_cauchy" else prior["tau_df"]) - np.log(scale)
            return lp + lt
        return stats.t.logpdf(theta, df=prior.get("df", 7), loc=prior["loc"], scale=prior["scale"]).sum(axis=1)

    def logkernel(self, z, data, prior):
        z = np.atleast_2d(z)
        theta = self.from_unconstrained(z)
        jac = np.zeros(len(z))
        if self.name in {"poisson_gamma", "nig", "eight_schools"}:
            jac = z[:, -1]
        elif self.name == "beta_binomial":
            jac = -np.logaddexp(0, -z[:, 0]) - np.logaddexp(0, z[:, 0])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = self.loglik(theta, data) + self.logprior(theta, prior) + jac
        return np.where(np.isfinite(value), value, -np.inf)

    def optimized(self, data, prior, settings=None, seed=123):
        from .quadrature import SchoolPosterior, logistic_posterior
        if self.dimension == 1 or self.name == "nig":
            return ConjugatePosterior(self, data, prior)
        if self.name == "eight_schools":
            return SchoolPosterior(self, data, prior, settings or {})
        return logistic_posterior(self, data, prior, settings or {}, seed)

    def default_targets(self):
        from .core import Coordinate, TargetSpec
        return [TargetSpec(v, (Coordinate(v),)) for v in self.variables]


def scalar_prior(p):
    family = p["family"]
    if family == "normal":
        return stats.norm(p["mean"], p["sd"])
    if family == "gamma":
        return stats.gamma(p["shape"], scale=1 / p["rate"])
    if family == "beta":
        return stats.beta(p["a"], p["b"])
    raise DefinitionError(f"No proper scalar prior distribution for {family}")


class ConjugatePosterior(Posterior):
    def __init__(self, model, data, prior):
        model.check_posterior_admissibility(data, prior)
        super().__init__(model, data, prior, "analytic", {"converged": True, "engine": "analytic"})
        y = np.asarray(data["y"], dtype=float)
        n, total = len(y), float(y.sum())
        name, family = model.name, prior["family"]
        if name == "normal_known":
            if family not in {"normal", "flat", "jeffreys"}:
                raise DefinitionError("Normal conjugacy requires normal/flat/Jeffreys prior")
            prec0 = 0. if family in {"flat", "jeffreys"} else 1 / prior["sd"] ** 2
            precision = prec0 + n / model.options.get("sigma", 1.) ** 2
            if precision <= 0:
                raise InadmissiblePosterior("Improper empty-data posterior")
            mean = ((prior.get("mean", 0) * prec0) + total / model.options.get("sigma", 1.) ** 2) / precision
            # Preserve the historical operation order for bitwise path RNG replay.
            if family in {"flat", "jeffreys"}:
                self.rv = stats.norm(float(np.mean(y)), model.options.get("sigma", 1.) / np.sqrt(n))
            else:
                mean = (1 / precision) * (n * (float(np.mean(y)) if n else 0.) / model.options.get("sigma", 1.) ** 2 + prior["mean"] / prior["sd"] ** 2)
                self.rv = stats.norm(mean, np.sqrt(1 / precision)) if n else stats.norm(prior["mean"], prior["sd"])
        elif name == "poisson_gamma":
            if family not in {"gamma", "flat", "jeffreys"}:
                raise DefinitionError("Poisson conjugacy requires gamma/flat/Jeffreys prior")
            a = prior.get("shape", 0.5 if family == "jeffreys" else 1.) + total
            b = prior.get("rate", 0.) + n
            if a <= 0 or b <= 0:
                raise InadmissiblePosterior("Improper gamma posterior")
            self.rv = stats.gamma(a, scale=1 / b)
        elif name == "beta_binomial":
            if family not in {"beta", "flat", "jeffreys"}:
                raise DefinitionError("Binomial conjugacy requires beta/flat/Jeffreys prior")
            base = .5 if family == "jeffreys" else 1.
            self.rv = stats.beta(prior.get("a", base) + total, prior.get("b", base) + n * model.options.get("trials", 20) - total)
        else:
            if family not in {"nig", "flat_mu_ig", "jeffreys"}:
                raise DefinitionError("NIG conjugacy requires nig/flat_mu_ig/Jeffreys prior")
            if n == 0 and family != "nig":
                raise InadmissiblePosterior("Improper empty-data NIG baseline")
            ybar = total / n if n else 0.
            sse = max(float(np.sum(y ** 2)) - total ** 2 / n, 0.) if n else 0.
            # Preserve ordinary legacy arithmetic; repair catastrophic cancellation
            # rather than misclassifying a proper, nonconstant sample as improper.
            if n and sse == 0 and np.ptp(y) > 0:
                sse = float(np.sum((y - ybar) ** 2))
            if family == "nig":
                l0 = 1 / prior["tau0_sq"]
                lam = l0 + n
                mu = (l0 * prior["mu0"] + total) / lam
                a = prior["kappa"] + n / 2
                b = prior["eta"] + .5 * (sse + l0 * n / lam * (ybar - prior["mu0"]) ** 2)
            else:
                lam, mu = n, ybar
                a = (n - 1) / 2 + (prior["kappa"] if family == "flat_mu_ig" else 0)
                b = sse / 2 + (prior["eta"] if family == "flat_mu_ig" else 0)
            if min(a, b, lam) <= 0:
                raise NumericalFailure("NIG posterior parameters could not be represented as positive numbers")
            self.params = {"mu": mu, "lambda": lam, "kappa": a, "eta": b}

    def sample(self, size, rng):
        if self.model.dimension == 1:
            return {self.model.variables[0]: self.rv.rvs(size=size, random_state=rng)}
        p = self.params
        v = 1 / rng.gamma(p["kappa"], scale=1 / p["eta"], size=size)
        mu = rng.normal(p["mu"], np.sqrt(v / p["lambda"]), size=size)
        return {"mu": mu, "sigma2": v}

    def target(self, target, size=1024, seed=123):
        if target.dimension == 1:
            c = target.coordinates[0]
            if self.model.dimension == 1:
                rv = self.rv
            else:
                p = self.params
                rv = stats.invgamma(p["kappa"], scale=p["eta"]) if c.variable == "sigma2" else stats.t(2 * p["kappa"], loc=p["mu"], scale=np.sqrt(p["eta"] / (p["kappa"] * p["lambda"])))
            raw = rv.ppf((np.arange(size) + .5) / size)
            mean = cov = None
            if c.transform == "identity":
                mean, cov = np.array([rv.mean() / c.scale]), np.array([[rv.var() / c.scale ** 2]])
            elif self.model.name == "nig" and c.variable == "sigma2" and c.transform == "log":
                mean = np.array([(np.log(p["eta"]) - digamma(p["kappa"]) - np.log(c.reference)) / c.scale])
                cov = np.array([[polygamma(1, p["kappa"]) / c.scale ** 2]])
            def lp(z):
                x = c.inverse(np.asarray(z).reshape(-1))
                return rv.logpdf(x) - c.log_jacobian(x)
            return Measure(c.forward(raw), np.ones(size), lp,
                           lambda u: c.forward(rv.ppf(u)), lambda z: rv.cdf(c.inverse(z)), mean, cov,
                           rv.dist.name if c.transform == "identity" else None,
                           {"rv": rv, "coordinate": c})
        if self.model.name != "nig" or {c.variable for c in target.coordinates} != {"mu", "sigma2"}:
            raise DefinitionError("Unsupported conjugate joint target")
        p = self.params
        def lp(z):
            z = np.atleast_2d(z)
            raw = {c.variable: c.inverse(z[:, i]) for i, c in enumerate(target.coordinates)}
            value = stats.invgamma.logpdf(raw["sigma2"], p["kappa"], scale=p["eta"]) + stats.norm.logpdf(raw["mu"], p["mu"], np.sqrt(raw["sigma2"] / p["lambda"]))
            return value - sum(c.log_jacobian(raw[c.variable]) for c in target.coordinates)
        draws = self.sample(size, np.random.default_rng(seed))
        return Measure(target.values(draws), np.ones(size), logpdf=lp, family="nig", parameters=p)
