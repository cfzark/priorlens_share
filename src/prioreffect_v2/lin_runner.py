"""Lin adapter for the common task/manifest workflow; numerical kernels in lin.py."""
from collections import OrderedDict
import copy
import numpy as np
from .core import Coordinate, TargetSpec, DefinitionError, digest
from .moments import check_target_moments
from . import lin


def validate_design(model, data, prior, baseline, targets):
    if model.name not in {"normal_known", "nig", "eight_schools", "lupus_logistic"}:
        raise DefinitionError("Lin supports normal_known, nig, eight_schools and lupus_logistic")
    if any(t.focal_unit_id for t in targets):
        raise DefinitionError("Lin targets are population parameters, not focal-school effects")
    if model.name == "lupus_logistic":
        model.validate_prediction_scale(data)
    # Existing path_bank constructs both predictive variants and needs these means.
    for variable in model.variables:
        check_target_moments(model, data, baseline, TargetSpec("lin_natural", (Coordinate(variable),)), 1)


def remember(cache, key, value, capacity):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)
    return value


class AuditedProvider:
    """Record the cached fits actually consumed by the current Lin calculation."""
    def __init__(self, provider):
        self.provider = provider

    def fit(self, data, prior, seed):
        return self.fit_many([data], prior, [seed])[0]

    def fit_many(self, datasets, prior, seeds):
        begin = len(self.provider.accuracy)
        posts = self.provider.fit_many(datasets, prior, seeds)
        recorded = {a.get("posterior_key") for a in self.provider.accuracy[begin:]}
        for data, seed, post in zip(datasets, seeds, posts):
            key = self.provider.key(data, prior, seed)
            if key not in recorded:
                self.provider.accuracy.append({"posterior_key": key, **post.accuracy, "cache_hit": True})
                recorded.add(key)
        return posts


def matching(model, data, info, base, target, method, grid, provider, seed,
             *, replication=0, targets=None):
    targets = [TargetSpec.parse(t) if isinstance(t, dict) else t for t in targets] if targets else [target]
    if target not in targets or len({t.id for t in targets}) != len(targets):
        raise DefinitionError("Lin shared target group must contain the current target exactly once")
    validate_design(model, data, info, base, targets)
    grid = lin.signed_grid(len(data["y"]), grid)
    paths, draws = method.get("paths", 8), method.get("posterior_draws", 1024)
    alpha, variant = method.get("alpha", .05), method["variant"]
    if not hasattr(provider, "lin_contexts"):
        provider.lin_contexts, provider.lin_results = OrderedDict(), OrderedDict()
    target_key = [dict(id=t.id, coordinates=[vars(c) for c in t.coordinates], focal_unit_id=t.focal_unit_id) for t in targets]
    context_key = digest([model.name, getattr(model, "options", {}), data, base, provider.settings,
                          target_key, paths, draws, alpha, replication])
    result_key = digest([context_key, info, grid, variant])
    audited = AuditedProvider(provider)
    if result_key in provider.lin_results:
        provider.lin_results.move_to_end(result_key)
        result_map, accuracy = provider.lin_results[result_key]
        provider.accuracy.extend([{**a, "cache_hit": True, "lin_group_reuse": True} for a in accuracy])
        return copy.deepcopy(result_map[target.id])
    if context_key in provider.lin_contexts:
        provider.lin_contexts.move_to_end(context_key)
        prepared, bank, audit, accuracy = provider.lin_contexts[context_key]
        provider.accuracy.extend([{**a, "cache_hit": True} for a in accuracy])
    else:
        begin = len(provider.accuracy)
        baseline = audited.fit(data, base, lin.seed(model.name, replication, "baseline"))
        prepared = lin.prepare_measures(lin.target_measures(baseline, targets, draws,
            lin.seed(model.name, replication, "baseline_target")), alpha)
        bank, audit = lin.path_bank(model, data, baseline, 0, paths, replication)
        accuracy = copy.deepcopy(provider.accuracy[begin:])
        remember(provider.lin_contexts, context_key, (prepared, bank, audit, accuracy), 2)
    c = lin.evaluate_bank(model, data, info, prepared, bank, targets, grid, audited,
                          draws=draws, replication=replication, alpha=alpha, variants=(variant,))
    chosen = lin.select(c, grid, alpha)
    result_map = {}
    loss = np.sum((np.minimum(c[0], 1-alpha)-(1-alpha))**2, axis=-1)
    for j, t in enumerate(targets):
        selected = []
        for b in range(paths):
            selected.append(dict(path=b, status="matched", signed_m=int(chosen["signed_m"][0,b,j]),
                selected_loss=float(chosen["loss"][0,b,j]),
                selected_coverage=chosen["selected_coverage"][0,b,j].tolist(),
                tie_count=int(chosen["tie_count"][0,b,j]), runner_up_gap=float(chosen["runner_up_gap"][0,b,j]),
                boundary_selected=bool(chosen["boundary"][0,b,j])))
        rows = [dict(path=b, m=int(m), cp1=float(c[0,b,k,j,0]), cp2=float(c[0,b,k,j,1]),
                     loss=float(loss[b,k,j])) for b in range(paths) for k,m in enumerate(grid)]
        result = dict(status="matched", epss=float(chosen["signed_m"][0,:,j].mean()),
            epss_meaning="mean_pathwise_signed_m", match_rate=1., unmatched_rate=0., numerical_failure_rate=0.,
            boundary_rate=float(chosen["boundary"][0,:,j].mean()), n_paths=paths,
            mean_cp1=float(chosen["selected_coverage"][0,:,j,0].mean()),
            mean_cp2=float(chosen["selected_coverage"][0,:,j,1].mean()),
            tie_rate=float((chosen["tie_count"][0,:,j]>1).mean()),
            mean_runner_up_gap=float(chosen["runner_up_gap"][0,:,j].mean()),
            variant=variant, lin_seed_rule="lin_nested_v1", path_checksum=audit["checksum"], path_diagnostics=audit)
        result_map[t.id] = (result, rows, selected)
    remember(provider.lin_results, result_key, (result_map, copy.deepcopy(provider.accuracy)), 2)
    return copy.deepcopy(result_map[target.id])
