"""Bounded path-major CP baseline reuse for the V2 lupus experiment.

V2 already shares empirical paths across cells. Reusing the corresponding
seeded baseline posterior/central regions therefore changes neither marginal
laws NOR cross-cell coupling. The reference matcher remains the only selector.
Jones, KDE/HDR choices and pseudo-data seeds are deliberately untouched.
"""
from collections import Counter, defaultdict
import numpy as np

from .core import DefinitionError, digest
from . import lupus_experiment as le


class CPWorkspace:
    def __init__(self):
        self.fixed = {}
        self.path = None
        self.candidate_data_hash = None
        self.regions = None
        self.full_hits = self.candidate_hits = 0

    def full(self, key, compute):
        if key not in self.fixed:
            self.fixed[key] = compute()
        else:
            self.full_hits += 1
        return self.fixed[key]

    def candidates(self, path, data_hash, compute):
        if path != self.path:
            self.regions = None  # release the previous path before fitting
            self.path, self.candidate_data_hash = path, data_hash
            self.regions = compute()
        else:
            if data_hash != self.candidate_data_hash:
                raise DefinitionError('Shared CP path has inconsistent pseudo-data')
            self.candidate_hits += 1
        return self.regions


def merge_accuracy(a, b):
    for field in ('n_fits','fallback_count','nonconverged_count','mode_failure_count'):
        a[field] += b[field]
    a['max_weight'] = max(a['max_weight'],b['max_weight'])
    values = [x for x in (a['min_importance_ess'],b['min_importance_ess']) if x is not None]
    a['min_importance_ess'] = min(values) if values else None
    a['engines'] = dict(Counter(a['engines'])+Counter(b['engines']))


def matching_group(spec, tasks):
    """Return one ordinary result per cell; hold only one baseline path bank."""
    if not tasks:
        raise DefinitionError('Empty CP cell group')
    first=tasks[0]
    if first['family'] not in le.FAMILIES[3:]:
        raise DefinitionError('Shared baseline grouping supports central CP only')
    identities={(t['rep'],t['family'],t['start'],t['end']) for t in tasks}
    if len(identities)!=1 or len({t['cell']['grid_id'] for t in tasks})!=len(tasks):
        raise DefinitionError('CP group must have one replication/family/path range and unique cells')
    if first['start']>=first['end']:
        raise DefinitionError('Empty CP path range')
    workspace=CPWorkspace(); results=[]; curves=[]
    for _ in tasks:
        results.append(dict(kind='matching',family=first['family'],count=first['end']-first['start'],
            selected=[],curves=[],accuracy=le.Engine(spec['numerics']).accuracy,path_checksums=[]))
        curves.append({})
    # Preserve each cell's original path order and floating-point sum order.
    for b in range(first['start'],first['end']):
        for task,result,diagnostic in zip(tasks,results,curves):
            piece=le.matching_shard(spec,{**task,'start':b,'end':b+1},workspace=workspace)
            result['selected'].extend(piece['selected'])
            result['path_checksums'].extend(piece['path_checksums'])
            merge_accuracy(result['accuracy'],piece['accuracy'])
            for row in piece['curves']:
                key=tuple(row[k] for k in ('target','variant','m','direction'))
                if key not in diagnostic:
                    diagnostic[key]=dict(row)
                else:
                    d=diagnostic[key]
                    for name in ('count','loss_sum','cp1_sum','cp2_sum'):d[name]+=row[name]
                    for name in ('own_mass_p_max','own_mass_q_max'):d[name]=max(d[name],row[name])
    for result,diagnostic in zip(results,curves):
        result['curves']=list(diagnostic.values())
        result['execution']=dict(engine='shared_baseline_v1',cell_group_size=len(tasks),
            shared_fit_accounting='actual fits charged once to computing cell; sum all cells for group cost')
    return results


def scheduling_groups(tasks, cell_block_size=8):
    if type(cell_block_size) is not int or cell_block_size<1:
        raise DefinitionError('cell_block_size must be a positive integer')
    groups=[]; banks=defaultdict(list)
    for i,t in enumerate(tasks):
        if t['family'] in le.FAMILIES[3:]:
            banks[(t['rep'],t['family'],t['start'],t['end'])].append(i)
        else:
            groups.append([i])
    # Similar-cost families together; task order never changes path identities.
    groups.sort(key=lambda g:(tasks[g[0]]['family'],tasks[g[0]]['rep'],tasks[g[0]]['start'],tasks[g[0]]['cell']['grid_id']))
    for bank in banks.values():
        for start in range(0,len(bank),cell_block_size):
            groups.append(bank[start:start+cell_block_size])
    if sorted(i for group in groups for i in group)!=list(range(len(tasks))):
        raise DefinitionError('Scheduling groups do not partition tasks')
    return groups
