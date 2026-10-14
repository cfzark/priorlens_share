"""Batched exact NIG marginal-mu CP; v2 paths and loss are the definitions.

Only finite-grid/path averages are approximated. Student-t quantiles and CDFs
are evaluated directly, without posterior Monte Carlo, KDE, IS or MCMC.
This is an opt-in experiment accelerator, not a replacement of legacy code.
"""
from time import perf_counter
import numpy as np
from scipy.special import stdtr, stdtrit

from .core import DefinitionError, NumericalFailure, TargetSpec, Coordinate
from .models import BuiltinModel
from .paths import build_paths, PathBank, nig_nested_statistics
from .losses import Coverage, coverage_terms
from .selection import BidirectionalSelector

LOSSES = ("H1", "D0_cap", "D1_cap")
LOSS_OPTIONS = ({"family": "historical", "gap_weight": 1., "cap_nominal": False},
                {"family": "direct", "gap_weight": 0., "cap_nominal": True},
                {"family": "direct", "gap_weight": 1., "cap_nominal": True})


def nig_parameters(n, total, sumsq, prior):
    """Vectorized conjugate sufficient-statistic update, checked against v2 objects."""
    n, total, sumsq = np.broadcast_arrays(np.asarray(n, float), total, sumsq)
    if np.any(n <= 0):
        raise DefinitionError("This experiment requires nonempty data")
    mean = total / n
    sse = np.maximum(sumsq - total ** 2 / n, 0.)
    if prior["family"] == "nig":
        l0 = 1 / prior["tau0_sq"]
        lam = l0 + n
        mu = (l0 * prior["mu0"] + total) / lam
        a = prior["kappa"] + n / 2
        b = prior["eta"] + .5 * (sse + l0 * n / lam * (mean - prior["mu0"]) ** 2)
    elif prior["family"] == "jeffreys":
        lam, mu, a, b = n, mean, (n - 1) / 2, sse / 2
    else:
        raise DefinitionError("NIG CP accelerator supports NIG / independent Jeffreys")
    if any(np.any(~np.isfinite(v)) for v in (mu, lam, a, b)) or np.any(np.minimum(np.minimum(lam, a), b) <= 0):
        raise NumericalFailure("Improper or nonfinite NIG posterior")
    return {"mu": mu, "lambda": lam, "kappa": a, "eta": b}


def data_parameters(y, prior):
    y = np.asarray(y, float)
    return nig_parameters(len(y), y.sum(), (y*y).sum(), prior)


def student_coverage(p, q, alpha=.05):
    """P(C_Q), Q(C_P) for exact equal-tail marginal-mu intervals."""
    dp, dq = 2*p["kappa"], 2*q["kappa"]
    sp = np.sqrt(p["eta"] / (p["kappa"] * p["lambda"]))
    sq = np.sqrt(q["eta"] / (q["kappa"] * q["lambda"]))
    def quantile(df, probability):
        # Degrees of freedom depend on m, not on the 500 path data values.
        # Evaluate the same special function once per distinct df; no interpolation.
        unique, inverse = np.unique(df, return_inverse=True)
        return stdtrit(unique, probability)[inverse].reshape(np.shape(df))
    plo = p["mu"] + sp * quantile(dp, alpha/2)
    phi = p["mu"] + sp * quantile(dp, 1-alpha/2)
    qlo = q["mu"] + sq * quantile(dq, alpha/2)
    qhi = q["mu"] + sq * quantile(dq, 1-alpha/2)
    a = stdtr(dp, (qhi-p["mu"])/sp) - stdtr(dp, (qlo-p["mu"])/sp)
    b = stdtr(dq, (phi-q["mu"])/sq) - stdtr(dq, (plo-q["mu"])/sq)
    return np.stack(np.broadcast_arrays(a, b), axis=-1)


def experiment_bank(model, data, p, q, grid, scheme, paths, rng):
    """Use v2 PathBank; explicitly condition bootstrap deletion on proper paths.

    Rejection is only for a zero-variance smallest retained sample, matching
    the historical NIG convention. Entire paths are redrawn, never candidates.
    Existing general PathBank defaults are unchanged.
    """
    bank = build_paths(model, data, p, q, grid, scheme, paths, rng)
    rejected = 0
    if scheme == "bootstrap_deletion":
        width = len(grid)
        for s in range(paths):
            while np.var(bank.candidates[(s+1)*width-1]["negative_data"]["y"]) <= 1e-14:
                rejected += 1
                if rejected > 10000:
                    raise NumericalFailure("Too many degenerate bootstrap paths")
                replacement = build_paths(model, data, p, q, grid, scheme, 1, rng)
                for r in replacement.candidates:
                    r["path"] = s
                bank.candidates[s*width:(s+1)*width] = replacement.candidates
        if rejected:
            from .core import digest
            bank = PathBank(bank.candidates, digest(bank.candidates), bank.source_checksum)
    return bank, rejected


def coverage_grid(data, prior, grid, scheme, paths, seed, bank=None, compact=True):
    model = BuiltinModel("nig")
    data = model.validate_data(dict(data))
    base = {"family": "jeffreys"}
    p, q = model.optimized(data, prior), model.optimized(data, base)
    start = perf_counter()
    rejected = 0
    sufficient = None
    if bank is None and compact:
        sufficient, rejected = nig_nested_statistics(model,data,p,q,grid,scheme,paths,np.random.default_rng(seed))
    elif bank is None:
        bank, rejected = experiment_bank(model, data, p, q, grid, scheme, paths, np.random.default_rng(seed))
    path_seconds = perf_counter() - start
    start = perf_counter()
    observed = [data_parameters(data["y"], prior), data_parameters(data["y"], base)]
    out = []
    for di, direction in enumerate(("negative", "positive")):
        if sufficient is not None:
            n, total, sumsq = sufficient[di]
        else:
            ys = [np.asarray(r[direction+"_data"]["y"], float) for r in bank.candidates]
            n = np.array([len(y) for y in ys])
            total = np.array([y.sum() for y in ys])
            sumsq = np.array([(y*y).sum() for y in ys])
        deletion = scheme.endswith("deletion")
        info_changes = (direction == "positive") == deletion
        candidate = nig_parameters(n, total, sumsq, prior if info_changes else base)
        fixed = observed[1 if info_changes else 0]
        out.append(student_coverage(fixed, candidate).reshape(paths, len(grid), 2))
    cp = np.stack(out, axis=1)  # path, direction(-,+), m, reciprocal coverage
    if not np.all(np.isfinite(cp)) or np.any(cp < 0) or np.any(cp > 1):
        raise NumericalFailure("Invalid coverage grid")
    return cp, bank, {"path_seconds": path_seconds, "coverage_seconds": perf_counter()-start,
                       "bootstrap_redraws": rejected, "posterior_updates": 2+2*paths*len(grid)}


def generic_coverage_grid(data, prior, grid, scheme, paths, bank):
    """Slow object-based reference using the unmodified v2 posterior interface."""
    model = BuiltinModel("nig")
    base = {"family": "jeffreys"}
    p, q = model.optimized(data, prior), model.optimized(data, base)
    target = TargetSpec("mu", (Coordinate("mu"),))
    p0, q0 = p.target(target, 8), q.target(target, 8)
    result = np.empty((paths, 2, len(grid), 2))
    for r in bank.candidates:
        for d, direction in enumerate(("negative", "positive")):
            info_changes = (direction == "positive") == scheme.endswith("deletion")
            post = model.optimized(r[direction+"_data"], prior if info_changes else base).target(target, 8)
            terms = Coverage()(q0 if info_changes else p0, post)
            result[r["path"], d, list(grid).index(r["m"])] = [terms["cp1"][0], terms["cp2"][0]]
    return result


def summarize(cp, grid):
    """All nonlinear loss/minimization precede Monte Carlo averaging.

    Returns compact per-replication summaries and candidate means. No path
    arrays are persisted by the production workflow.
    """
    grid = np.asarray(grid)
    B = len(cp)
    ar = np.arange(B)
    results, curves, timings = [], [], []
    for name, options in zip(LOSSES, LOSS_OPTIONS):
        start = perf_counter()
        terms = coverage_terms(cp[..., 0], cp[..., 1], **options)
        loss = terms["loss"]
        ix = np.argmin(loss, axis=2)
        minima = loss[ar[:, None], np.arange(2), ix]
        positive = minima[:, 1] < minima[:, 0]  # exact ties negative
        direction = positive.astype(int)
        signed = grid[ix[ar, direction]] * np.where(positive, 1, -1)
        selected = loss[ar, direction, ix[ar, direction]]
        row = {"loss_id": name, "epss": signed.mean(), "internal_sd": signed.std(ddof=1) if B>1 else 0.,
               "mean_abs_m": np.abs(signed).mean(), "pr_positive": positive.mean(),
               "pr_zero": (signed==0).mean(), "boundary_rate": (np.abs(signed)==max(grid)).mean(),
               "selected_loss": selected.mean()}
        for d, label in enumerate(("negative", "positive")):
            pick = ix[:, d]
            raw = cp[ar, d, pick]
            row[label+"_mstar"] = grid[pick].mean()
            row[label+"_min_loss"] = minima[:, d].mean()
            row[label+"_boundary"] = (grid[pick]==max(grid)).mean()
            row[label+"_r1_raw"] = np.max(raw, axis=1).mean()
            row[label+"_r2_raw"] = np.min(raw, axis=1).mean()
            for field in ("term_gap", "term_coverage", "cp1_used", "cp2_used"):
                row[label+"_"+field] = terms[field][ar, d, pick].mean()
        results.append(row)
        # Average of losses, NOT loss of average coverages.
        curves.append({k: terms[k].mean(axis=0) for k in ("loss", "term_gap", "term_coverage", "cp1_used", "cp2_used")})
        timings.append(perf_counter()-start)
    return results, curves, timings
