"""Minimal complete registration; no optimized posterior or legacy imports.

Stan determines the posterior sampler. This adapter supplies the scientific
meaning Stan cannot infer: units, pseudo-data law, priors, and target variables.
All log densities below are with respect to the declared natural parameter mu.
"""
import numpy as np
from scipy.stats import norm
from prioreffect_v2.core import DefinitionError


class UserNormal:
    name = "user_normal"
    variables = ["mu"]
    dimension = 1
    unit = "observation"
    stan_parameters = ["mu"]

    def __init__(self, sigma=1.):
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise DefinitionError("sigma must be positive")

    def validate_data(self, data):
        y = np.asarray(data["y"], dtype=float)
        ids = data.get("unit_ids", [f"observation_{i+1}" for i in range(len(y))])
        if y.ndim != 1 or not np.all(np.isfinite(y)) or len(ids) != len(y) or len(set(ids)) != len(ids):
            raise DefinitionError("Expected finite y and aligned unique unit IDs")
        return {"y": y.tolist(), "unit_ids": list(ids)}

    def validate_prior(self, prior):
        if prior.get("family") != "normal" or not np.isfinite(prior["mean"]) or not np.isfinite(prior["sd"]) or prior["sd"] <= 0:
            raise DefinitionError("This registered Stan model supports proper normal priors")

    def subset(self, data, indices, auxiliary=False):
        ix = np.asarray(indices, dtype=int)
        return {"y": np.asarray(data["y"])[ix].tolist(),
                "unit_ids": [f"aux_{j}" for j in range(len(ix))] if auxiliary else np.asarray(data["unit_ids"])[ix].tolist()}

    def append(self, data, additions):
        return {"y": data["y"] + additions["y"],
                "unit_ids": data["unit_ids"] + [f"aux_{i}" for i in range(len(additions["y"]))]}

    def simulate(self, theta, count, rng, design):
        return {"y": rng.normal(theta[0], self.sigma, size=count).tolist(),
                "unit_ids": [f"aux_{i}" for i in range(count)]}

    def check_target_moments(self, data, prior, target, order):
        self.validate_prior(prior)
        if order not in (1, 2) or any(c.variable != "mu" or c.transform != "identity" for c in target.coordinates):
            raise DefinitionError("Only affine normal-location moments are certified")
        return {"certified": True, "rule": "proper_normal_prior_and_normal_likelihood", "order": order}

    def sampling_signature(self):
        return {"sigma": self.sigma}

    def to_unconstrained(self, theta):
        return np.asarray(theta, dtype=float)

    def from_unconstrained(self, z):
        return np.asarray(z, dtype=float)

    def loglik(self, theta, data):
        return norm.logpdf(data["y"], np.atleast_2d(theta)[:, :1], self.sigma).sum(axis=1)

    def logprior(self, theta, prior):
        return norm.logpdf(np.atleast_2d(theta)[:, 0], prior["mean"], prior["sd"])

    def logkernel(self, z, data, prior):
        # Identity unconstraining transform: log |d mu / d z| = 0.
        return self.loglik(z, data) + self.logprior(z, prior)

    def stan_data(self, data, prior):
        return {"N": len(data["y"]), "y": data["y"], "sigma": self.sigma,
                "prior_mean": prior["mean"], "prior_sd": prior["sd"]}

    def stan_draws(self, fit):
        return fit.stan_variable("mu")[:, None]


def create_model(registration):
    return UserNormal(**registration.get("options", {}))
