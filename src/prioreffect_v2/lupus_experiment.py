"""All-target lupus accelerator: current V2 risk/CP components, retained Jones.

The historical numerical posterior engine is an explicit accelerator, not a
replacement definition. Source Jones uses the exact saved simulation runner.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np

from .core import Measure, DefinitionError, NumericalFailure, digest, seed_for, seed_scope
from .models import BuiltinModel, lupus_data
from .losses import risk_from_moments, coverage_terms
from .regions import central_region
from .selection import BidirectionalSelector
from .risk_components import RiskSampler, RiskDatasetBank
from prioreffect.lupus_logistic import (make_prior, make_baseline, target_values,
    grouped_binomial_counts, fit_logistic_posteriors_counts_batch)
from prioreffect.lupus_jones_source_paths import run_source_path
from prioreffect.lupus_logistic_simulation import (simulate_replication, stable_seed,
    EXPERIMENT_ID, replication_checksum)

TARGETS = ['beta_joint', 'beta1', 'prediction_joint', 'risk_difference',
           'p_x_0', 'p_x_0p5', 'p_x_1', 'p_x_1p5', 'p_x_2']
COEFFICIENT_TARGETS = ['beta_joint', 'beta0', 'beta1']
FAMILIES = ['reimherr', 'wiesenfarth', 'jones', 'cp_bootstrap_augmentation', 'cp_bootstrap_deletion']
METHODS = FAMILIES[:3] + [f'{f}_{loss}' for f in FAMILIES[3:] for loss in ('H1', 'D0_cap')]


def method_names(aggregation='mean'):
    if aggregation not in ('mean', 'median', 'both'):
        raise DefinitionError('rw aggregation must be mean, median or both')
    risks = FAMILIES[:2] if aggregation == 'mean' else [f'{f}_median' for f in FAMILIES[:2]]
    if aggregation == 'both':
        risks = [name for f in FAMILIES[:2] for name in (f, f+'_median')]
    return risks + METHODS[2:]


def seed(*keys):
    return seed_for(20260312, 'lupus_v2_location_20260928', *keys, scheme='sha256_64_v2')


def dataset(rep):
    data = lupus_data()
    if rep >= 0:
        data['y'] = simulate_replication(rep)['y'].tolist()
    return data


def cell_grid():
    # Keep ALL historical q*13+i keys; new intercepts get unused appended IDs.
    out = []
    for q in range(0, 65, 2):
        for i, b0 in enumerate(np.arange(-5., 3.01, .5)):
            old = b0 >= -3
            gid = q * 13 + int(round((b0 + 3) * 2)) if old else 845 + q * 4 + i
            out.append(dict(grid_id=gid, q=q, beta0=float(b0), beta1=-5.+q*.25, legacy_jones=bool(old)))
    return out


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.accuracy = dict(n_fits=0, fallback_count=0, nonconverged_count=0,
            mode_failure_count=0, min_importance_ess=None, max_weight=0., engines={})

    def audit(self, records):
        for a in records:
            s = self.accuracy
            s['n_fits'] += 1
            s['fallback_count'] += bool(a.get('fallback_used', False))
            s['nonconverged_count'] += not a.get('converged', False)
            s['mode_failure_count'] += not a.get('mode_converged', False)
            s['max_weight'] = max(s['max_weight'], float(a.get('max_weight', 0)))
            ess = a.get('importance_ess')
            if ess is not None and np.isfinite(ess):
                s['min_importance_ess'] = float(ess) if s['min_importance_ess'] is None else min(s['min_importance_ess'],float(ess))
            name = a.get('engine', 'unknown')
            s['engines'][name] = s['engines'].get(name, 0) + 1
        if self.accuracy['nonconverged_count'] or self.accuracy['mode_failure_count']:
            raise NumericalFailure(f'Uncertified lupus posterior: {self.accuracy}')

    def fit(self, datasets, prior, seeds):
        levels = np.unique(lupus_data()['x'])
        counts = [grouped_binomial_counts(d['x'],d['y'],levels=levels) for d in datasets]
        posts = fit_logistic_posteriors_counts_batch(levels,
            np.array([c[1] for c in counts]),np.array([c[2] for c in counts]),prior,
            [int(s) % (2**63) for s in seeds],**self.settings)
        self.audit([p.accuracy for p in posts])
        return posts


def selected_targets(spec, task=None):
    targets=spec.get('targets',TARGETS)
    if not isinstance(targets,list) or not targets or any(t not in TARGETS+['beta0'] for t in targets) or len(set(targets))!=len(targets):
        raise DefinitionError('Invalid lupus target list')
    if task is not None and 'targets' in task:
        subset=task['targets']
        if task['family']!='jones' or not isinstance(subset,list) or not subset or len(set(subset))!=len(subset) or any(t not in targets for t in subset):
            raise DefinitionError('Only Jones missing-target tasks may restrict targets')
        return subset
    return targets


def measures(post,targets=TARGETS):
    return {t: Measure(target_values(post.points,t),post.weights) for t in targets}


def risk_vector(post, references, metric):
    return [risk_from_moments(p.mean,np.diag(p.covariance),references[t],metric)
            for t,p in measures(post,list(references)).items()]


def risk_bank(data, base_full, family, rep):
    sampling = 'bootstrap' if family == 'reimherr' else 'likelihood'
    sampler = RiskSampler(BuiltinModel('lupus_logistic'),data,sampling,base_full.mean)
    return RiskDatasetBank(sampler,seed('risk_data',rep))


def baseline_risk_cache(spec, task):
    """Shared baseline fits; common data law is independent of prior location."""
    engine = Engine(spec['numerics'])
    data = dataset(task['rep']); base = make_baseline(1000.)
    full = engine.fit([data],base,[seed('baseline_full',task['rep'])])[0]
    refs = {t:p.mean for t,p in measures(full,selected_targets(spec)).items()}
    family = task['family']; grid = spec['risk_grid']; n = len(data['y'])
    sizes = [n+m for m in grid] if family == 'reimherr' else [n]
    metric = 'posterior_mse' if family == 'reimherr' else 'squared_mean'
    values, checks = [], []
    with seed_scope('sha256_64_v2'):
        bank = risk_bank(data,full,family,task['rep'])
        for b in range(task['start'],task['end']):
            ds = [bank.draw(k,b) for k in sizes]
            checks.append(digest(ds))
            posts = engine.fit(ds,base,[seed('risk_fit',task['rep'],digest(d)) for d in ds])
            values.append([risk_vector(p,refs,metric) for p in posts])
    return dict(values=values, sizes=sizes, references=refs, beta_star=full.mean,
                data_checksums=checks, data_checksum=digest(data), accuracy=engine.accuracy)


def risk_shard(spec, task, cache):
    data = dataset(task['rep']); cell = task['cell']; family = task['family']
    engine = Engine(spec['numerics']); base = make_baseline(1000.)
    full = engine.fit([data],base,[seed('baseline_full',task['rep'])])[0]
    refs = {t:np.asarray(v) for t,v in cache['references'].items()}
    if set(refs)!=set(selected_targets(spec)):
        raise NumericalFailure('Baseline risk cache target mismatch')
    # JSON canonicalization sorts dictionary keys; array axes follow the spec.
    refs = {t:refs[t] for t in selected_targets(spec)}
    if cache['data_checksum'] != digest(data) or not np.array_equal(full.mean,cache['beta_star']):
        raise NumericalFailure('Baseline risk cache reference/data mismatch')
    prior = make_prior(cell['beta0'],cell['beta1'],2.5)
    n = len(data['y']); grid = spec['risk_grid']
    sizes = [n] if family == 'reimherr' else [n-m for m in grid]
    metric = 'posterior_mse' if family == 'reimherr' else 'squared_mean'
    values, checks = [], []
    with seed_scope('sha256_64_v2'):
        bank = risk_bank(data,full,family,task['rep'])
        for b in range(task['start'],task['end']):
            ds = [bank.draw(k,b) for k in sizes]
            checks.append(digest(ds))
            posts = engine.fit(ds,prior,[seed('risk_fit',task['rep'],digest(d)) for d in ds])
            values.append([risk_vector(p,refs,metric) for p in posts])
    info = np.asarray(values); baseline = np.asarray(cache['values'])
    fixed, candidate = (info[:,0,:],baseline) if family == 'reimherr' else (baseline[:,0,:],info)
    result = dict(kind='risk', family=family, count=len(values),
        fixed_sum=fixed.sum(axis=0),candidate_sum=candidate.sum(axis=0),
        fixed_sumsq=(fixed**2).sum(axis=0),candidate_sumsq=(candidate**2).sum(axis=0),
        accuracy=engine.accuracy, data_checksums=checks, metric=metric,
        reference=refs, coordinate_rule='raw declared target; no automatic whitening')
    if spec.get('rw', {}).get('aggregation', 'mean') in ('median', 'both'):
        # Only informative risks are stored per cell. Baseline samples remain
        # in their shared cache; never estimate a median from shard summaries.
        result['informative_samples'] = info
    return result


def cp_candidates(data, grid, scheme, rng):
    model = BuiltinModel('lupus_logistic'); n = len(data['y'])
    ix = rng.choice(n,size=n if scheme.endswith('deletion') else max(grid),replace=True)
    return [model.subset(data,ix[:n-m],auxiliary=True) if scheme.endswith('deletion')
            else model.append(data,model.subset(data,ix[:m],auxiliary=True)) for m in grid]


def prepare_regions(post,targets=TARGETS):
    return {t:(p,central_region(p,.05)) for t,p in measures(post,targets).items()}


def cp_rows(p, q, m, direction):
    """Use the exact V2 central region and loss components for both variants."""
    out = []
    if set(p)!=set(q):raise DefinitionError('CP posterior target mismatch')
    for t in p:
        pm,pr = p[t]; qm,qr = q[t]
        a,b = qr.mass(pm),pr.mass(qm)
        for loss in ('H1','D0_cap'):
            terms = coverage_terms(a,b,.95,'historical' if loss=='H1' else 'direct',
                                   1. if loss=='H1' else 0.,loss=='D0_cap')
            out.append(dict(target=t,variant=loss,m=m,direction=direction,loss=float(terms['loss']),
                cp1=a,cp2=b,gap=(a-b)**2,adequacy=max(abs(a-.95),abs(b-.95)),
                own_mass_p=pr.own_mass,own_mass_q=qr.own_mass))
    return out


def matching_shard(spec, task, workspace=None):
    targets=selected_targets(spec,task)
    data = dataset(task['rep']); cell = task['cell']; family = task['family']
    engine = Engine(spec['numerics']); prior = make_prior(cell['beta0'],cell['beta1'],2.5)
    base = make_baseline(1000.); grid = spec['matching_grid']
    if workspace is not None and family == 'jones':
        raise DefinitionError('Jones source paths cannot share CP baseline work')
    def prepared(prior, fit_seed):
        post = engine.fit([data],prior,[fit_seed])[0]
        return post, prepare_regions(post,targets)
    if workspace is None:
        info = engine.fit([data],prior,[seed('info_full',task['rep'],cell['grid_id'])])[0]
        full = engine.fit([data],base,[seed('baseline_full',task['rep'])])[0]
    else:
        info,info_regions = workspace.full(('info',cell['grid_id']),
            lambda: prepared(prior,seed('info_full',task['rep'],cell['grid_id'])))
        full,base_regions = workspace.full(('base',),
            lambda: prepared(base,seed('baseline_full',task['rep'])))
    selected, checks, diagnostic = [], [], {}
    if family != 'jones' and workspace is None:
        info_regions,base_regions = prepare_regions(info,targets),prepare_regions(full,targets)
    for b in range(task['start'],task['end']):
        if family == 'jones':
            # Same algorithm and historical seed family for a simulation extension.
            s = (stable_seed(20260312,EXPERIMENT_ID,'jones_source_path',task['rep'],cell['grid_id'],b)
                 if task['rep'] >= 0 else seed('jones_source_path','real',cell['grid_id'],b) >> 1)
            result = run_source_path(x=np.asarray(data['x']),y=np.asarray(data['y']),prior=prior,
                baseline=base,info_full=info,base_full=full,targets=targets,m_grid=grid,seed=s,
                projections=32,execution_engine='cpp',**spec['numerics'])
            engine.audit(result['accuracy']); checks.append(result['path_checksum'])
            for t,r in result['selection'].items():
                selected.append(dict(method='jones',target=t,path=b,signed_m=r['signed_m'],
                    direction=r['selected_direction'],boundary=abs(r['signed_m'])==max(grid)))
        else:
            ds = cp_candidates(data,grid,family,np.random.default_rng(seed('cp_paths',task['rep'],family,b)))
            checks.append(digest(ds))
            # In bootstrap deletion m=0 is a BOOTSTRAP SAMPLE, not observed data.
            active = list(range(len(grid))) if family.endswith('deletion') else [i for i,m in enumerate(grid) if m]
            ps = engine.fit([ds[i] for i in active],prior,[seed('cp_fit',task['rep'],family,b,grid[i],'info') for i in active])
            pr = {i:prepare_regions(p,targets) for i,p in zip(active,ps)}
            def baseline_candidates():
                qs = engine.fit([ds[i] for i in active],base,[seed('cp_fit',task['rep'],family,b,grid[i],'base') for i in active])
                return {i:prepare_regions(p,targets) for i,p in zip(active,qs)}
            qr = (baseline_candidates() if workspace is None else
                  workspace.candidates(b,checks[-1],baseline_candidates))
            rows = []
            for i,m in enumerate(grid):
                pi,qi = pr.get(i,info_regions),qr.get(i,base_regions)
                if family.endswith('deletion'):
                    rows += cp_rows(info_regions,qi,m,'negative') + cp_rows(pi,base_regions,m,'positive')
                else:
                    rows += cp_rows(base_regions,pi,m,'negative') + cp_rows(info_regions,qi,m,'positive')
            for t in targets:
                for variant in ('H1','D0_cap'):
                    rr = [r for r in rows if r['target']==t and r['variant']==variant]
                    s = BidirectionalSelector(tie='negative')(rr)
                    losses = sorted(r['loss'] for r in rr)
                    selected.append(dict(method=f'{family}_{variant}',target=t,path=b,signed_m=s['signed_m'],
                        direction=s['selected_direction'],boundary=s['boundary_selected'],
                        exact_tie=sum(v==losses[0] for v in losses)>1,runner_up_gap=losses[1]-losses[0]))
            for r in rows:
                k = (r['target'],r['variant'],r['m'],r['direction'])
                if k not in diagnostic:
                    diagnostic[k] = dict(target=k[0],variant=k[1],m=k[2],direction=k[3],count=0,
                        loss_sum=0.,cp1_sum=0.,cp2_sum=0.,own_mass_p_max=0.,own_mass_q_max=0.)
                d = diagnostic[k]; d['count'] += 1
                for name in ('loss','cp1','cp2'): d[name+'_sum'] += r[name]
                for name in ('own_mass_p','own_mass_q'): d[name+'_max'] = max(d[name+'_max'],r[name])
    return dict(kind='matching',family=family,count=task['end']-task['start'],selected=selected,
        curves=list(diagnostic.values()),accuracy=engine.accuracy,path_checksums=checks)


def summarize_parts(spec, tasks_and_parts):
    """Aggregate risks BEFORE argmin; matching selects per path BEFORE averaging."""
    tasks_and_parts = sorted(tasks_and_parts,key=lambda a:a[0]['start'])
    end=0
    for task,part in tasks_and_parts:
        if task['start'] != end or part['count'] != task['end']-task['start']:
            raise NumericalFailure('Missing/overlapping replication shard')
        end=task['end']
    if end != spec['repetitions']: raise NumericalFailure('Incomplete Monte Carlo bank')
    task = tasks_and_parts[0][0]; family=task['family']; rows=[]
    targets=selected_targets(spec,task)
    if any(selected_targets(spec,t)!=targets for t,_ in tasks_and_parts):
        raise NumericalFailure('Inconsistent target axes across shards')
    identity = dict(rep=task['rep'],**task['cell'])
    parts=[p for _,p in tasks_and_parts]
    if family in ('reimherr','wiesenfarth'):
        fixed=sum(np.asarray(p['fixed_sum']) for p in parts)/end
        candidate=sum(np.asarray(p['candidate_sum']) for p in parts)/end
        aggregation = spec.get('rw', {}).get('aggregation', 'mean')
        method_names(aggregation)  # Reject unsupported settings, never silently default.
        variants = [('mean',fixed,candidate)] if aggregation != 'median' else []
        if aggregation in ('median','both'):
            info = np.concatenate([p['informative_samples'] for p in parts])
            baseline = np.concatenate([p['baseline_samples'] for p in parts])
            expected_info=(end,1 if family=='reimherr' else len(spec['risk_grid']),len(targets))
            expected_base=(end,len(spec['risk_grid']) if family=='reimherr' else 1,len(targets))
            if info.shape!=expected_info or baseline.shape!=expected_base or not np.isfinite(info).all() or not np.isfinite(baseline).all():
                raise NumericalFailure('Invalid full-bank raw risks for median aggregation')
            f,c=(info[:,0,:],baseline) if family=='reimherr' else (baseline[:,0,:],info)
            variants.append(('median',np.median(f,axis=0),np.median(c,axis=0)))
        for name,f,c in variants:
            loss=np.abs(c-f)
            for j,t in enumerate(targets):
                best=int(np.argmin(loss[:,j])); m=spec['risk_grid'][best]
                rows.append(dict(**identity,method=family if name=='mean' else family+'_median',target=t,epss=m,objective=loss[best,j],
                    boundary_rate=float(best in (0,len(spec['risk_grid'])-1)),aggregation=name,repetitions=end))
    else:
        values=[r for p in parts for r in p['selected']]
        for method in sorted({r['method'] for r in values}):
            for t in targets:
                rr=[r for r in values if r['method']==method and r['target']==t]
                if sorted(r['path'] for r in rr) != list(range(end)): raise NumericalFailure('Invalid path coverage')
                x=np.array([r['signed_m'] for r in rr])
                rows.append(dict(**identity,method=method,target=t,epss=float(x.mean()),
                    epss_internal_sd=float(x.std(ddof=1)),pr_negative=float(np.mean(x<0)),
                    boundary_rate=float(np.mean([r['boundary'] for r in rr])),repetitions=end,
                    tie_rate=float(np.mean([r.get('exact_tie',False) for r in rr])) if method!='jones' else None,
                    mean_runner_up_gap=float(np.mean([r['runner_up_gap'] for r in rr])) if method!='jones' else None))
    return rows
