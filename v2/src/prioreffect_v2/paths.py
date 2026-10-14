"""Pseudo-data generation. Posterior inference and losses never consume this RNG."""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from .core import DefinitionError, InadmissiblePosterior, digest, sampling_data

SCHEMES = {"source_factorized", "directional_factorized", "nested_shared_source", "nested_directional",
           "bootstrap_augmentation", "bootstrap_deletion", "subsample_deletion",
           "nested_baseline_directional"}


@dataclass
class PathBank:
    candidates: list
    checksum: str
    source_checksum: str
    diagnostics: dict = field(default_factory=dict)


def build_paths(model, data, post_info, post_base, grid, scheme, repetitions, rng, focal_unit_id=None,
                *, admissibility_check=None, invalid_path_policy="error", max_path_redraws=1000):
    """Optional, explicit conditioning on proper posteriors for the ENTIRE path.

    The default raises on an inadmissible path when a check is supplied. The
    low-level unchecked API is retained for legacy accelerator replay only.
    Redraws occur after the initial bank, in path order; no individual m is redrawn.
    """
    if invalid_path_policy not in {"error", "redraw_whole_path"}:
        raise DefinitionError("invalid_path_policy must be error or redraw_whole_path")
    if invalid_path_policy == "redraw_whole_path" and (scheme != "bootstrap_deletion" or admissibility_check is None):
        raise DefinitionError("Whole-path redraw requires bootstrap_deletion and an admissibility check")
    bank = _build_paths_raw(model, data, post_info, post_base, grid, scheme, repetitions, rng, focal_unit_id)
    rejected, reasons = 0, {}
    if admissibility_check is not None:
        grouped = [[r for r in bank.candidates if r["path"] == path] for path in range(repetitions)]
        for path, rows in enumerate(grouped):
            attempts = 0
            while True:
                try:
                    for row in rows:
                        for direction in ("positive", "negative"):
                            admissibility_check(row[direction + "_data"])
                    break
                except InadmissiblePosterior as error:
                    if invalid_path_policy == "error":
                        error.context = {"path": path, "m": row["m"], "invalid_path_policy": invalid_path_policy}
                        raise
                    rejected += 1
                    attempts += 1
                    reasons[str(error)] = reasons.get(str(error), 0) + 1
                    if attempts > max_path_redraws:
                        raise InadmissiblePosterior(f"Whole-path redraw limit reached at path {path}; {rejected} rejected proposals") from error
                    replacement = _build_paths_raw(model, data, post_info, post_base, grid, scheme, 1, rng, focal_unit_id)
                    rows = [{**r, "path": path} for r in replacement.candidates]
            grouped[path] = rows
        if rejected:
            bank.candidates = [row for rows in grouped for row in rows]
            bank.checksum = digest(bank.candidates)
    bank.diagnostics = {"invalid_path_policy": invalid_path_policy,
        "admissibility_checked": admissibility_check is not None, "rejected_paths": rejected,
        "rejection_reasons": reasons, "max_path_redraws": max_path_redraws,
        "path_law": "conditional_on_all_candidate_posteriors_proper" if invalid_path_policy == "redraw_whole_path" else "unconditional"}
    return bank


def _build_paths_raw(model, data, post_info, post_base, grid, scheme, repetitions, rng, focal_unit_id=None):
    if scheme not in SCHEMES:
        raise DefinitionError(f"Unknown pseudo-data scheme {scheme}")
    n, maximum = len(data["y"]), max(grid)
    candidates, sources = [], []
    legacy_hash = hashlib.sha256()
    for path in range(repetitions):
        empirical = scheme.startswith("bootstrap") or scheme == "subsample_deletion"
        theta1 = theta2 = None
        if not empirical:
            source = post_base if scheme == "nested_baseline_directional" else post_info
            draw1 = source.sample(1, rng)
            theta1 = [float(draw1[v][0]) for v in model.variables]
            if scheme in {"directional_factorized", "nested_directional", "nested_baseline_directional"}:
                draw2 = post_base.sample(1, rng)
                theta2 = [float(draw2[v][0]) for v in model.variables]
            else:
                theta2 = theta1
            sources.append([theta1, theta2])
            if model.dimension == 1:
                legacy_hash.update(np.asarray([path], dtype="<i8").tobytes())
                legacy_hash.update(np.asarray([theta1[0], theta2[0]], dtype="<f8").tobytes())
            if scheme.startswith("nested"):
                positive = model.simulate(theta1, maximum, rng, data)
                negative = model.simulate(theta2, maximum, rng, data)
        elif scheme == "bootstrap_augmentation":
            # The focal original unit is not duplicated as an auxiliary school.
            pool = np.arange(n)
            if focal_unit_id:
                pool = pool[pool != data["unit_ids"].index(focal_unit_id)]
            index = rng.choice(pool, size=maximum, replace=True)
        else:
            pool = np.arange(n)
            if focal_unit_id:
                focal = data["unit_ids"].index(focal_unit_id)
                pool = pool[pool != focal]
                sequence = rng.permutation(pool) if scheme == "subsample_deletion" else rng.choice(pool, size=n - 1, replace=True)
                sequence = np.r_[focal, sequence]
            else:
                sequence = rng.permutation(pool) if scheme == "subsample_deletion" else rng.choice(pool, size=n, replace=True)
        for m in grid:
            if scheme.endswith("deletion"):
                reduced = model.subset(data, sequence[:n - m])
                if focal_unit_id:
                    reduced["unit_ids"] = [focal_unit_id] + [f"aux_{j}" for j in range(len(reduced["y"]) - 1)]
                else:
                    reduced["unit_ids"] = [f"aux_{j}" for j in range(len(reduced["y"]))]
                p_data = q_data = reduced
            elif scheme == "bootstrap_augmentation":
                added = model.subset(data, index[:m], auxiliary=True)
                p_data = q_data = model.append(data, added)
            else:
                if scheme.startswith("nested"):
                    ap = model.subset(positive, np.arange(m), auxiliary=True)
                    an = model.subset(negative, np.arange(m), auxiliary=True)
                else:
                    ap = model.simulate(theta1, m, rng, data)
                    an = model.simulate(theta2, m, rng, data)
                p_data, q_data = model.append(data, ap), model.append(data, an)
                if model.dimension == 1:
                    legacy_hash.update(np.asarray([m], dtype="<i8").tobytes())
                    for additions in [ap, an]:
                        values = np.asarray(additions["y"], dtype="<f8")
                        legacy_hash.update(np.asarray([len(values)], dtype="<i8").tobytes())
                        legacy_hash.update(values.tobytes())
            candidates.append({"path": path, "m": int(m), "positive_data": p_data, "negative_data": q_data})
    hash_rows = [{**r, "positive_data": sampling_data(r["positive_data"]), "negative_data": sampling_data(r["negative_data"])} for r in candidates]
    checksum = legacy_hash.hexdigest() if model.dimension == 1 and not empirical else digest(hash_rows)
    return PathBank(candidates, checksum, digest(sources))


def nig_nested_statistics(model, data, post_info, post_base, grid, scheme, repetitions, rng):
    """Compact NIG implementation with the SAME RNG calls as build_paths.

    Only this explicit experiment fast path conditions bootstrap deletion on
    a proper smallest retained sample. No full candidate dictionaries/hashes
    are built. Shared rows are reduced with numpy.sum, like ConjugatePosterior.
    """
    supported = {"nested_directional", "nested_baseline_directional", "bootstrap_augmentation",
                 "bootstrap_deletion", "subsample_deletion"}
    if model.name != "nig" or scheme not in supported:
        raise DefinitionError("Unsupported compact NIG path")
    y = np.asarray(data["y"], float)
    n, maximum = len(y), max(grid)
    deletion = scheme.endswith("deletion")
    pool = np.arange(n)
    positive, negative = [], []
    for _ in range(repetitions):
        if scheme.startswith("nested"):
            source = post_info if scheme == "nested_directional" else post_base
            p, q = source.sample(1, rng), post_base.sample(1, rng)
            a = model.simulate([p[v][0] for v in model.variables], maximum, rng, data)["y"]
            b = model.simulate([q[v][0] for v in model.variables], maximum, rng, data)["y"]
            positive.append(np.r_[y, a]); negative.append(np.r_[y, b])
        elif scheme == "bootstrap_augmentation":
            ix = rng.choice(pool, size=maximum, replace=True)
            positive.append(np.r_[y,y[ix]]); negative.append(positive[-1])
        else:
            ix = rng.permutation(pool) if scheme == "subsample_deletion" else rng.choice(pool,size=n,replace=True)
            positive.append(y[ix]); negative.append(positive[-1])
    rejected = 0
    if scheme == "bootstrap_deletion":
        for s in range(repetitions):
            while np.var(positive[s][:n-maximum]) <= 1e-14:
                rejected += 1
                if rejected > 10000:
                    raise DefinitionError("Too many degenerate bootstrap paths")
                ix = rng.choice(pool,size=n,replace=True)
                positive[s] = y[ix];negative[s] = positive[s]
    result = []
    for sequence in (negative,positive):
        values = np.asarray(sequence)
        counts = np.array([n-m if deletion else n+m for m in grid])
        sums = np.stack([values[:,:count].sum(axis=1) for count in counts],axis=1)
        squares = np.stack([(values[:,:count]**2).sum(axis=1) for count in counts],axis=1)
        result.append((counts[None,:],sums,squares))
    return result, rejected
