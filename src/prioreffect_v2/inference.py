"""Posterior inference is independent of pseudo-data path generation."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

from .core import DefinitionError, NumericalFailure, digest, seed_for
from .quadrature import WeightedPosterior


def mode_covariance(model, data, prior):
    if model.name == "lupus_logistic":
        from .quadrature import logistic_mode
        return logistic_mode(model, data, prior)
    start = np.zeros(model.dimension)
    if model.name == "eight_schools":
        start = np.array([prior["mu_mean"], np.log(prior["tau_scale"])])
    objective = lambda z: -float(model.logkernel(np.atleast_2d(z), data, prior)[0])
    fit = minimize(objective, start, method="BFGS")
    if not np.isfinite(fit.fun):
        raise NumericalFailure("IS proposal mode is nonfinite")
    center = fit.x
    d, h = len(center), 1e-3
    H = np.empty((d, d))
    eye = np.eye(d) * h
    for i in range(d):
        for j in range(d):
            H[i, j] = (objective(center + eye[i] + eye[j]) - objective(center + eye[i] - eye[j]) - objective(center - eye[i] + eye[j]) + objective(center - eye[i] - eye[j])) / (4 * h * h)
    eig, vec = np.linalg.eigh((H + H.T) / 2)
    return center, (vec / np.maximum(eig, 1e-6)) @ vec.T


def weight_diagnostics(log_weights, points, min_ess=500, min_relative_ess=.02, max_weight=.02, max_k=.7, anchor_efficiency=1.):
    log_weights = np.asarray(log_weights)
    if np.any(np.isnan(log_weights)) or not np.any(np.isfinite(log_weights)) or np.any(np.isposinf(log_weights)):
        raise NumericalFailure("Invalid importance ratios or unsupported target")
    weights = np.exp(log_weights - logsumexp(log_weights))
    raw_ess = float(1 / np.sum(weights ** 2))
    ess = raw_ess * min(float(anchor_efficiency), 1.)
    positive = np.sort(weights[weights > 0])
    tail_n = min(len(positive) // 5, int(3 * np.sqrt(len(positive))))
    if len(positive) > 1 and np.ptp(positive) < 1e-12 * positive[-1]:
        k = -np.inf
    elif tail_n >= 20:
        threshold = positive[-tail_n - 1]
        k = float(stats.genpareto.fit(positive[-tail_n:] / threshold - 1, floc=0)[0]) if threshold > 0 else np.inf
    else:
        k = np.inf
    mean = weights @ points
    sd = np.sqrt(np.maximum(weights @ (points - mean) ** 2, 1e-12))
    half = len(weights) // 2
    if weights[:half].sum() == 0 or weights[half:].sum() == 0:
        split = np.inf
    else:
        a = weights[:half] @ points[:half] / weights[:half].sum()
        b = weights[half:] @ points[half:] / weights[half:].sum()
        split = float(np.max(np.abs(a - b) / sd))
    passed = ess >= min_ess and ess / len(weights) >= min_relative_ess and weights.max() <= max_weight and k < max_k and split <= .15
    return weights, {"converged": bool(passed), "importance_ess": ess, "raw_weight_ess": raw_ess,
        "relative_ess": ess / len(weights), "max_weight": float(weights.max()), "pareto_k": k,
        "tail_diagnostic": "raw_ratio_GPD_MLE_not_PSIS_smoothing", "split_mean_sd_error": split,
        "anchor_efficiency": anchor_efficiency, "n_points": len(weights)}


def importance_posterior(model, data, prior, settings, seed, mode=None, covariance=None):
    if mode is None:
        mode, covariance = mode_covariance(model, data, prior)
    d = model.dimension
    centers = np.stack([mode, mode])
    covs = np.stack([4 * covariance, 16 * covariance])
    if model.name == "lupus_logistic":
        centers[1] = prior["loc"]
        covs[1] = np.diag(np.maximum(np.square(prior["scale"]), 1e-6))
    for power in range(int(settings.get("sobol_power", 13)), int(settings.get("sobol_max_power", 16)) + 1):
        u = np.clip(qmc.Sobol(d=d + 2, scramble=True, seed=int(seed)).random_base2(power), 1e-12, 1 - 1e-12)
        component = (u[:, 0] >= .75).astype(int)
        z = stats.norm.ppf(u[:, 1:d + 1]) / np.sqrt(stats.chi2.ppf(u[:, -1], df=5)[:, None] / 5)
        proposal = np.empty_like(z)
        for k in [0, 1]:
            ix = component == k
            proposal[ix] = centers[k] + z[ix] @ np.linalg.cholesky(covs[k]).T
        logq = logsumexp(np.column_stack([np.log(w) + stats.multivariate_t.logpdf(proposal, centers[k], covs[k], df=5) for k, w in enumerate([.75, .25])]), axis=1)
        lw = model.logkernel(proposal, data, prior) - logq
        theta = model.from_unconstrained(proposal)
        mass, accuracy = weight_diagnostics(lw, proposal, min_ess=settings.get("is_min_ess", 500))
        accuracy.update(engine="sobol_mixture_importance", sobol_power=power, seed=int(seed), proposal="0.75*t5(mode,4cov)+0.25*t5(wide)", fallback_used=False)
        post = WeightedPosterior(model, data, prior, theta, mass, accuracy, float(logsumexp(lw) - np.log(len(lw))))
        if accuracy["converged"]:
            return post
    raise NumericalFailure(f"IS accuracy gates failed at power={power}: {accuracy}")


def reweight_posterior(anchor, data, prior, settings, seed):
    """Fixed-dimensional posterior ratios; no resampling of outer paths."""
    model = anchor.model
    if isinstance(anchor, WeightedPosterior):
        theta = anchor.points
        base_weights = anchor.weights
    else:
        draws = anchor.sample(settings.get("reuse_draws", 8192), np.random.default_rng(seed))
        theta = np.column_stack([draws[v] for v in model.variables])
        base_weights = np.full(len(theta), 1 / len(theta))
    lr = model.loglik(theta, data) - model.loglik(theta, anchor.data) + model.logprior(theta, prior) - model.logprior(theta, anchor.prior)
    efficiency = anchor.accuracy.get("mcmc_efficiency", 1.)
    mass, accuracy = weight_diagnostics(np.log(base_weights) + lr, model.to_unconstrained(theta), min_ess=settings.get("is_min_ess", 500), anchor_efficiency=efficiency)
    accuracy.update(engine="posterior_importance_reuse", proposal_id=digest([anchor.data, anchor.prior, anchor.engine]), seed=int(seed), fallback_used=False)
    if not accuracy["converged"]:
        raise NumericalFailure(f"Anchor reuse failed: {accuracy}")
    # An anchor's unknown normalizer remains unknown after reweighting.
    logz = getattr(anchor, "log_normalizer", None)
    if logz is not None:
        logz += float(logsumexp(np.log(base_weights) + lr))
    return WeightedPosterior(model, data, prior, theta, mass, accuracy, logz)


def builtin_stan_data(model, data, prior):
    kind = ["normal_known", "poisson_gamma", "beta_binomial", "nig", "eight_schools", "lupus_logistic"].index(model.name) + 1
    families = {"normal": 1, "gamma": 1, "beta": 1, "nig": 1, "schools": 1, "student_t": 1, "flat": 2, "flat_mu_ig": 2, "jeffreys": 3}
    return {"kind": kind, "D": model.dimension, "N": len(data["y"]), "y": data["y"],
        "x": data.get("x", [0.] * len(data["y"])), "s": data.get("s", [1.] * len(data["y"])),
        "sigma": model.options.get("sigma", 1.), "trials": model.options.get("trials", 20),
        "family": families[prior["family"]], "mu0": prior.get("mean", prior.get("mu0", prior.get("mu_mean", 0.))),
        "prior_sd": prior.get("sd", prior.get("mu_sd", 1.)), "a": prior.get("shape", prior.get("a", prior.get("kappa", 1.))),
        "b": prior.get("rate", prior.get("b", prior.get("eta", 1.))), "tau0_sq": prior.get("tau0_sq", 1.),
        "tau_kind": {"half_t": 1, "lognormal": 2, "half_cauchy": 3, "bounded_uniform": 4}.get(prior.get("tau_kind"), 1),
        "tau_scale": prior.get("tau_scale", 1.), "tau_df": prior.get("tau_df", 4.), "tau_log_sd": prior.get("tau_log_sd", 1.),
        "beta_loc": prior.get("loc", [0., 0.]), "beta_scale": prior.get("scale", [1., 1.]), "beta_df": prior.get("df", 7.)}


def stan_posterior(model, data, prior, settings, seed):
    try:
        import cmdstanpy
    except ImportError as e:
        raise DefinitionError("Install prioreffect-v2[stan] for general backend") from e
    root = Path(__file__).resolve().parent
    cache = Path(settings.get("stan_cache", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "priorlens"))
    stan_path = settings.get("cmdstan") or os.environ.get("CMDSTAN")
    if not stan_path:
        candidates = sorted(cache.glob("cmdstan-*/bin/stanc"))
        if candidates:
            stan_path = str(candidates[-1].parents[1])
    if stan_path:
        cmdstanpy.set_cmdstan_path(stan_path)
    source = Path(getattr(model, "stan_file", root / "stan" / "builtin.stan"))
    code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    build = cache / "models" / digest([code_hash, cmdstanpy.cmdstan_path(), cmdstanpy.__version__])[:20]
    build.mkdir(parents=True, exist_ok=True)
    target_source = build / source.name
    # Lock compilation for shared Slurm filesystems.
    import fcntl
    with (build / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target_source.exists():
            shutil.copyfile(source, target_source)
        stan = cmdstanpy.CmdStanModel(stan_file=str(target_source))
    payload = model.stan_data(data, prior) if hasattr(model, "stan_data") else builtin_stan_data(model, data, prior)
    for attempt in range(2):
        fit = stan.sample(data=payload, seed=int(seed) % (2**31 - 1) + 1,
            chains=settings.get("chains", 4), parallel_chains=settings.get("parallel_chains", 1),
            iter_warmup=settings.get("warmup", 1000), iter_sampling=settings.get("sampling", 1000),
            adapt_delta=.99 if attempt else settings.get("adapt_delta", .95), max_treedepth=15 if attempt else 12,
            show_progress=False, show_console=False)
        summary = fit.summary()
        parameters = getattr(model, "stan_parameters", ["z"])
        names = [idx for idx in summary.index if any(idx == p or idx.startswith(p + "[") for p in parameters)]
        if not names:
            raise DefinitionError("Adapter stan_parameters were not found in fit")
        sub = summary.loc[names]
        rhat = float(sub["R_hat"].max())
        bulk = float(sub["ESS_bulk"].min())
        tail = float(sub["ESS_tail"].min())
        divergence = int(fit.method_variables()["divergent__"].sum())
        valid = np.all(np.isfinite(sub[["R_hat", "ESS_bulk", "ESS_tail"]])) and rhat <= 1.01 and min(bulk, tail) >= settings.get("stan_min_ess", 400) and divergence == 0
        if valid:
            theta = model.stan_draws(fit) if hasattr(model, "stan_draws") else fit.stan_variable("theta")
            theta = np.asarray(theta).reshape(-1, model.dimension)
            return WeightedPosterior(model, data, prior, theta, np.ones(len(theta)),
                {"engine": "cmdstan_nuts", "converged": True, "rhat": rhat, "ess_bulk": bulk,
                 "ess_tail": tail, "divergences": divergence, "mcmc_efficiency": min(1., min(bulk, tail) / len(theta)),
                 "attempt": attempt + 1, "stan_source_hash": code_hash, "cmdstan_path": cmdstanpy.cmdstan_path()})
    raise NumericalFailure(f"Stan diagnostic gates failed: Rhat={rhat}, ESS={bulk}/{tail}, divergences={divergence}")


class PosteriorProvider:
    def __init__(self, model, settings=None):
        self.model, self.settings = model, dict(settings or {})
        from .school_acceleration import validate_settings
        validate_settings(model, self.settings)
        self.cache, self.accuracy, self.risk_cache = OrderedDict(), [], OrderedDict()

    def key(self, data, prior, seed, anchor=None):
        if self.settings.get("school_cache_by_data", False):
            # This is allowed only for deterministic optimized Schools fits.
            # Target sampling retains its own original seed downstream.
            return digest([data, prior, self.settings])
        return digest([data, prior, self.settings, seed, digest([anchor.data, anchor.prior, anchor.engine]) if anchor else None])

    def remember(self, key, post):
        self.cache[key] = post
        if self.settings.get("school_engine") == "batched" or self.settings.get("school_cache_by_data", False):
            self.cache.move_to_end(key)
            while len(self.cache) > self.settings.get("school_cache_size", 256):
                self.cache.popitem(last=False)

    def fit_many(self, datasets, prior, seeds, anchor=None):
        if len(datasets) != len(seeds):
            raise DefinitionError("fit_many needs one seed per dataset")
        if self.settings.get("school_engine", "reference") != "batched":
            return [self.fit(d, prior, seed, anchor) for d, seed in zip(datasets, seeds)]
        from .school_acceleration import batch_posteriors
        groups, output = {}, [None] * len(datasets)
        for i, (data, seed) in enumerate(zip(datasets, seeds)):
            self.model.check_posterior_admissibility(data, prior)
            key = self.key(data, prior, seed, anchor)
            if key in self.cache:
                output[i] = self.cache[key]
                self.cache.move_to_end(key)
                self.accuracy.append({"posterior_key": key, **output[i].accuracy, "cache_hit": True})
            else:
                groups.setdefault(len(data["y"]), {}).setdefault(key, []).append(i)
        chunk = self.settings.get("school_batch_size", 16)
        for entries in groups.values():
            items = list(entries.items())
            for start in range(0, len(items), chunk):
                block = items[start:start + chunk]
                try:
                    posts = batch_posteriors(self.model, [datasets[ix[0]] for _, ix in block], prior, self.settings)
                except NumericalFailure as error:
                    error.context = {**getattr(error, "context", {}), "input_rows": [ix[0] for _, ix in block]}
                    raise
                for (key, indices), post in zip(block, posts):
                    self.remember(key, post)
                    self.accuracy.append({"posterior_key": key, **post.accuracy, "cache_hit": False})
                    for i in indices:
                        output[i] = post
        return output

    def fit(self, data, prior, seed=123, anchor=None):
        if hasattr(self.model, "check_posterior_admissibility"):
            self.model.check_posterior_admissibility(data, prior)
        key = self.key(data, prior, seed, anchor)
        if key in self.cache:
            if self.settings.get("school_cache_by_data", False) or self.settings.get("school_engine") == "batched":
                self.cache.move_to_end(key)
                self.accuracy.append({"posterior_key": key, **self.cache[key].accuracy, "cache_hit": True})
            return self.cache[key]
        backend = self.settings.get("backend", "auto")
        backend = ("optimized" if callable(getattr(self.model, "optimized", None)) else "general") if backend == "auto" else backend
        fallback = None
        if anchor is not None and self.settings.get("posterior_reuse", "none") == "importance":
            try:
                post = reweight_posterior(anchor, data, prior, self.settings, seed)
            except NumericalFailure as e:
                fallback = str(e)
            else:
                self.remember(key, post)
                self.accuracy.append({"posterior_key": key, **post.accuracy})
                return post
        if backend == "general":
            post = stan_posterior(self.model, data, prior, self.settings, seed)
        elif backend == "importance":
            post = importance_posterior(self.model, data, prior, self.settings, seed)
        elif backend == "optimized" and callable(getattr(self.model, "optimized", None)):
            post = self.model.optimized(data, prior, self.settings, seed)
        else:
            raise DefinitionError(f"Unsupported backend {backend}")
        if fallback:
            post.accuracy.update(fallback_used=True, fallback_reason=fallback)
        self.remember(key, post)
        self.accuracy.append({"posterior_key": key, **post.accuracy})
        return post


def load_adapter(registration):
    """Trusted local Python adapters are executed once at model registration."""
    path = Path(registration["adapter"]).resolve()
    spec = importlib.util.spec_from_file_location("pe_user_" + digest(str(path))[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.create_model(registration)
    for name in ["name", "variables", "dimension", "unit", "validate_data", "subset", "append", "simulate", "loglik", "logprior", "logkernel", "to_unconstrained", "from_unconstrained", "stan_data", "stan_draws", "stan_parameters"]:
        if not hasattr(model, name):
            raise DefinitionError(f"Adapter lacks {name}")
    model.stan_file = str(Path(registration["stan"]).resolve())
    return model
