"""Lin: fixed observed baseline versus nested, freshly generated informative data.

Public components deliberately do not use the bidirectional CP path/selector.
Changing the informative sample size does not change the observed baseline.
No existing method, seed stream, or posterior backend is modified here.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .core import Coordinate, TargetSpec, Measure, DefinitionError, NumericalFailure, seed_for, digest
from .moments import marginal_summary
from .regions import CentralRegion, central_region


VARIANTS = ("lin_ppd", "lin_plugin")


def seed(*parts):
    return seed_for(20260312, "lin_nested_v1", *parts, scheme="sha256_64_v2")


def signed_grid(n, values=None):
    a = np.asarray(list(range(-n, n+1)) if values is None else values)
    if (a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or
            np.any(a != np.floor(a)) or np.any(np.diff(a) <= 0) or
            np.any(a < -n) or np.any(a > n)):
        raise DefinitionError("Lin requires sorted unique integer m in [-N,N]")
    return a.astype(int)


def select(coverages, grid, alpha=.05):
    """Arrays (..., m, target, reciprocal direction); exact ties only."""
    c = np.asarray(coverages, float)
    if c.shape[-3] != len(grid) or c.shape[-1] != 2 or not np.all(np.isfinite(c)):
        raise NumericalFailure("Invalid Lin coverage bank")
    if np.any(c < -1e-12) or np.any(c > 1+1e-12):
        raise NumericalFailure("Coverage outside [0,1]")
    loss = np.sum((np.minimum(c, 1-alpha)-(1-alpha))**2, axis=-1)
    # Reorder m before argmin: |m| first, negative before positive.
    preference = np.lexsort((grid, np.abs(grid)))
    index = preference[np.argmin(loss[..., preference, :], axis=-2)]
    best = np.take_along_axis(loss, index[..., None, :], axis=-2)[..., 0, :]
    ordered = np.sort(loss, axis=-2)
    selected_c = np.take_along_axis(c, index[..., None, :, None], axis=-3)[..., 0, :, :]
    return dict(signed_m=np.asarray(grid)[index], loss=best,
        tie_count=np.sum(loss == best[..., None, :], axis=-2),
        runner_up_gap=ordered[..., 1, :]-best if len(grid)>1 else np.zeros_like(best),
        selected_coverage=selected_c, boundary=(index == 0) | (index == len(grid)-1))


def central_cached_mass(measure, alpha=.05):
    """Exact reference bisection, caching repeated rectangle memberships.

    Bounds/iterations/sums match regions.central_region. Binary searches detect
    when changed floating bounds include exactly the same support points.
    Especially useful for large Sobol fallbacks; no tolerance-based new ties.
    """
    if measure.values.shape[1] == 1:
        return central_region(measure, alpha)
    positive = measure.weights > 0
    v, w = measure.values[positive], measure.weights[positive]
    orders = [np.argsort(v[:, j]) for j in range(2)]
    sv = [v[ix, j] for j, ix in enumerate(orders)]
    cdf = [np.cumsum(w[ix])/w.sum() for ix in orders]
    cache = {}
    def evaluate(t):
        limits = np.asarray([np.interp([t, 1-t], f, x) for f, x in zip(cdf, sv)])
        lo, hi = limits[:, 0], limits[:, 1]
        key = tuple((int(np.searchsorted(sv[j], lo[j], side="left")),
                     int(np.searchsorted(sv[j], hi[j], side="right"))) for j in range(2))
        if key not in cache:
            cache[key] = float(w @ np.all((v >= lo) & (v <= hi), axis=1))
        return CentralRegion(lo, hi, t, 1-alpha, cache[key])
    low, high = 0., .5
    best = evaluate(low)
    for _ in range(52):
        mid = (low+high)/2
        candidate = evaluate(mid)
        if candidate.own_mass >= 1-alpha:
            low, best = mid, candidate
        else:
            high = mid
    return best


def prepare_measures(measures, alpha=.05, accelerated=True):
    region = central_cached_mass if accelerated else central_region
    return [(m, region(m, alpha)) for m in measures]


def reciprocal(p, q):
    if len(p) != len(q):
        raise DefinitionError("Lin target mismatch")
    return np.asarray([[qr.mass(pm), pr.mass(qm)] for (pm, pr), (qm, qr) in zip(p, q)])


def natural_means(post):
    """E of NATURAL model parameters, not inverse-transformed target means."""
    if hasattr(post, "points") and not hasattr(post, "model"):
        return dict(zip(("beta0", "beta1"), map(float, post.mean)))
    target = TargetSpec("lin_plugin_natural", tuple(Coordinate(v) for v in post.model.variables))
    means, _, _ = marginal_summary(post, target, 1)
    return dict(zip(post.model.variables, map(float, means)))


def draw_parameter(post, rng):
    if hasattr(post, "points") and not hasattr(post, "model"):
        p = post.points[rng.choice(len(post.weights), p=post.weights)]
        return dict(zip(("beta0", "beta1"), map(float, p)))
    return {v: float(a[0]) for v, a in post.sample(1, rng).items()}


def generate_path(model, data, theta, path_id, replication=-1):
    """One vector of 2N NEW units. Identical design/noise seeds across variants.

    Normal noise and Bernoulli uniforms are explicit, making paired simulation
    transparent. New schools integrate local effects, not shared local theta.
    """
    n = len(data["y"])
    rng = np.random.default_rng(seed(model.name, replication, "noise", path_id))
    design_rng = np.random.default_rng(seed(model.name, replication, "design", path_id))
    k = 2*n
    if model.name == "lupus_logistic":
        x = np.asarray(data["x"])[design_rng.integers(n, size=k)]
        y = (rng.random(k) < expit(theta["beta0"]+theta["beta1"]*x)).astype(int)
        return dict(y=y.tolist(), x=x.tolist(), x_center=data["x_center"], x_divisor=data["x_divisor"],
                    unit_ids=[f"lin_{path_id}_{i}" for i in range(k)])
    if model.name == "eight_schools":
        s = np.asarray(data["s"])[design_rng.integers(n, size=k)]
        y = theta["mu"]+np.sqrt(theta["tau"]**2+s*s)*rng.standard_normal(k)
        return dict(y=y.tolist(), s=s.tolist(), unit_ids=[f"lin_{path_id}_{i}" for i in range(k)])
    if model.name in ("nig", "normal_known"):
        sd = np.sqrt(theta["sigma2"]) if model.name == "nig" else model.options.get("sigma", 1.)
        return dict(y=(theta["mu"]+sd*rng.standard_normal(k)).tolist())
    raise DefinitionError("Lin design currently supports normal-known, NIG, Schools and Lupus")


def path_bank(model, data, baseline_post, start, end, replication=-1):
    if not 0 <= start < end:
        raise DefinitionError("Invalid Lin path range")
    plugin = natural_means(baseline_post)
    out = {v: [] for v in VARIANTS}
    theta = []
    for b in range(start, end):
        ppd = draw_parameter(baseline_post, np.random.default_rng(seed(model.name, replication, "theta", b)))
        theta.append(ppd)
        for variant, value in zip(VARIANTS, (ppd, plugin)):
            out[variant].append(generate_path(model, data, value, b, replication))
    return out, dict(plugin=plugin, ppd_parameters=theta, checksum=digest(out),
                     start=start, end=end, design="iid empirical design entries; fresh responses")


def prefix(model, data, k):
    return model.subset(data, np.arange(k), auxiliary=True)


def target_measures(post, targets, draws, draw_seed):
    """Same joint natural draws for all targets; analytic scalar marginals kept."""
    if hasattr(post, "points") and not hasattr(post, "model"):
        raw = dict(beta0=post.points[:, 0], beta1=post.points[:, 1])
        return [Measure(t.values(raw), post.weights) for t in targets]
    joint_draws = post.sample(draws, np.random.default_rng(draw_seed)) if any(t.dimension == 2 for t in targets) else None
    # Conjugate scalar coverage uses exact CDF/PPF; thousands of unused inverse
    # CDF support points would only waste CPU. The one support point is NEVER
    # used for its region or reciprocal mass (both have analytic CDFs).
    scalar_draws = 1 if post.engine == "analytic" else draws
    return [Measure(t.values(joint_draws), np.ones(draws)) if t.dimension == 2
            else post.target(t, scalar_draws, draw_seed) for t in targets]


def evaluate_bank(model, data, prior, baseline_prepared, bank, targets, grid, provider,
                  draws=4096, replication=-1, start=0, alpha=.05, accelerated=True,
                  variants=VARIANTS):
    """Reference/general built-in route, with Schools fit_many acceleration.

    Operates on small path shards. All targets share every fit; prior-only
    candidate and its regions are constructed once, shared by both variants.
    Baseline fits/regions and the PathBank are supplied by the caller/cache.
    """
    n = len(data["y"])
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise DefinitionError("Lin variants must be a nonempty unique subset of VARIANTS")
    grid = signed_grid(n, grid)
    empty = prefix(model, next(iter(bank.values()))[0], 0)
    zero = provider.fit(empty, prior, seed(model.name, "empty", digest(prior))) if n in grid else None
    prepared_zero = prepare_measures(target_measures(zero, targets, draws,
        seed(model.name, "target_empty", digest(prior))), alpha, accelerated) if zero else None
    result = np.empty((len(variants), len(bank[variants[0]]), len(grid), len(targets), 2))
    for v, variant in enumerate(variants):
        for j, m in enumerate(grid):
            if m == n:
                result[v, :, j] = reciprocal(baseline_prepared, prepared_zero)
                continue
            ds = [prefix(model, full, n-int(m)) for full in bank[variant]]
            fit_seeds = [seed(model.name, replication, "fit", digest(d), digest(prior)) for d in ds]
            posts = provider.fit_many(ds, prior, fit_seeds) if hasattr(provider, "fit_many") else [provider.fit(d, prior, s) for d, s in zip(ds, fit_seeds)]
            for b, post in enumerate(posts):
                # Common posterior Monte Carlo across priors and variants; no
                # sharing between baseline and candidate posterior samples.
                pm = target_measures(post, targets, draws, seed(model.name, replication, "target", start+b, int(m)))
                result[v, b, j] = reciprocal(baseline_prepared, prepare_measures(pm, alpha, accelerated))
    return result


def grouped_prefixes(bank, levels):
    """Vectorized sufficient statistics for ALL nested Lupus candidate sizes."""
    out = {}
    for variant in VARIANTS:
        x = np.asarray([d["x"] for d in bank[variant]])
        y = np.asarray([d["y"] for d in bank[variant]], dtype=np.int64)
        ix = np.searchsorted(levels, x)
        if not np.array_equal(np.asarray(levels)[ix], x):
            raise DefinitionError("Predictors must retain the original standardized levels")
        indicators = ix[..., None] == np.arange(len(levels))
        zero = np.zeros((len(x), 1, len(levels)), dtype=np.int64)
        out[variant] = (np.concatenate((zero, np.cumsum(indicators, axis=1)), axis=1),
                        np.concatenate((zero, np.cumsum(indicators*y[..., None], axis=1)), axis=1))
    return out


def evaluate_lupus(data, prior, baseline_prepared, bank, targets, grid, engine,
                   replication=-1, alpha=.05, accelerated=True, block_size=32):
    """Count batching + identical-likelihood memoization, all targets at once.

    Counts (not order) determine fit seeds, so duplicate likelihoods really are
    the same numerical posterior. Deduplication covers both sampling variants
    and all path prefixes within the shard, including the pure prior.
    """
    from prioreffect.lupus_logistic import fit_logistic_posteriors_counts_batch
    grid = signed_grid(len(data["y"]), grid)
    levels = np.unique(data["x"])
    counts = grouped_prefixes(bank, levels)
    result = np.empty((2, len(bank[VARIANTS[0]]), len(grid), len(targets), 2))
    locations, statistics = {}, {}
    for v, variant in enumerate(VARIANTS):
        total, success = counts[variant]
        for b in range(len(total)):
            for j, m in enumerate(grid):
                k = len(data["y"])-int(m)
                a, c = total[b, k], success[b, k]
                key = (a.tobytes(), c.tobytes())
                statistics[key] = (a, c)
                locations.setdefault(key, []).append((v, b, j))
    keys = list(locations)
    for offset in range(0, len(keys), block_size):
        group = keys[offset:offset+block_size]
        a = np.asarray([statistics[key][0] for key in group])
        c = np.asarray([statistics[key][1] for key in group])
        seeds = [seed("lupus_logistic", replication, "count_fit", aa.tolist(), cc.tolist(),
                      np.asarray(prior.loc).tolist(), np.asarray(prior.scale).tolist()) % 2**63 for aa, cc in zip(a, c)]
        posts = fit_logistic_posteriors_counts_batch(levels, a, c, prior, seeds, **engine.settings)
        engine.audit([p.accuracy for p in posts])
        for key, post in zip(group, posts):
            q = prepare_measures(target_measures(post, targets, 0, 0), alpha, accelerated)
            coverage = reciprocal(baseline_prepared, q)
            for index in locations[key]:
                result[index] = coverage
    return result, dict(unique_likelihoods=len(keys), candidates=int(np.prod(result.shape[:3])),
                        accuracy=engine.accuracy)
