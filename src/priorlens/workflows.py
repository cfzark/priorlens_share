"""Explicit small-run wrappers for existing Lin and paired raw-risk kernels.

These are not historical distributed campaign replays. Failure output is
preserved; no incomplete selection is published as a completed experiment.
"""
from pathlib import Path
import json
import copy
import numpy as np
import pandas as pd
import yaml
from prioreffect_v2.core import DefinitionError, TargetSpec, digest, write_json, seed_scope
from prioreffect_v2.experiments import load_spec, resolved_tasks, model_for, runtime_fingerprint
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.methods import risk_matching
from prioreffect_v2 import lin


def fresh(output):
    path = Path(output).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def certify(out, files, **metadata):
    write_json(out/'done.json', dict(status='pass', precision_certified=False, **metadata,
               files={f: digest((out/f).read_text()) for f in files}))
    return dict(status='pass', output=str(out), **metadata)


def run_risk_summary(spec_path, output):
    """Mean and median BEFORE argmin, sharing the exact raw risk bank."""
    spec = load_spec(spec_path)
    tasks = [t for t in resolved_tasks(spec) if t['method']['name'] in {'reimherr', 'wiesenfarth'}]
    if not tasks:
        raise DefinitionError('No R/W tasks in specification')
    out = fresh(output)
    write_json(out/'resolved.json', {'spec': spec, 'runtime': runtime_fingerprint(spec), 'tasks': tasks})
    results, banks = [], []
    try:
        model = model_for(spec)
        for task in tasks:
            raw = []
            with seed_scope(task['seed_scheme']):
                mean, curves, _ = risk_matching(model, task['data'], task['prior'], task['baseline'],
                    TargetSpec.parse(task['target']), task['method'], np.array(task['candidates']['values']),
                    PosteriorProvider(model, task['inference']), task['seed'], raw_risks=raw)
            a = np.asarray(raw)
            grid = np.asarray(task['candidates']['values'])
            med = np.median(a, axis=1)
            ix = int(np.argmin(np.abs(med[1:]-med[0])))
            identity = dict(prior_id=task['prior_id'], target=task['target']['id'], method=task['method']['id'])
            results.extend([{**identity, 'aggregation': 'mean', **mean},
                            {**identity, 'aggregation': 'median', 'epss': int(grid[ix]),
                             'boundary_rate': float(ix in (0,len(grid)-1)), 'status': 'matched'}])
            banks.append(dict(identity=identity, grid=grid.tolist(), raw=a.tolist(), curves=curves,
                              risk_components=mean['risk_components']))
        pd.DataFrame(results).to_csv(out/'results.csv', index=False)
        write_json(out/'raw_risks.json', banks)
        return certify(out, ['resolved.json','results.csv','raw_risks.json'], rows=len(results))
    except Exception as e:
        write_json(out/'failure.json', {'type': type(e).__name__, 'error': str(e)})
        raise


def _run_lin_reference(spec_path, output):
    """Pre-preset wrapper retained as a paired regression oracle."""
    path = Path(spec_path).resolve()
    settings = yaml.safe_load(path.read_text())
    allowed = {'schema_version','experiment','paths','posterior_draws','grid','variants','alpha','replication_id'}
    if not isinstance(settings,dict) or set(settings)-allowed or settings.get('schema_version') != 1:
        raise DefinitionError('Invalid Lin wrapper specification')
    for k in ('experiment','paths','posterior_draws','grid','variants','alpha'):
        if k not in settings:
            raise DefinitionError('Lin requires explicit '+k)
    if any(not isinstance(settings[k],int) or settings[k]<1 for k in ('paths','posterior_draws')):
        raise DefinitionError('Positive integer Lin paths and posterior_draws required')
    if not 0 < settings['alpha'] < 1:
        raise DefinitionError('Lin alpha must lie in (0,1)')
    spec = load_spec(path.parent/settings['experiment'])
    tasks = resolved_tasks(spec)  # Validate model/target/prior definitions first.
    model = model_for(spec)
    if model.name not in {'normal_known','nig','eight_schools','lupus_logistic'}:
        raise DefinitionError('External-adapter Lin generation is not yet implemented')
    if any(t.get('focal_unit_id') for t in spec['targets']):
        raise DefinitionError('Focal Lin is not implemented')
    data = model.validate_data(spec['data'])
    targets = [TargetSpec.parse(t) for t in spec['targets']]
    grid = lin.signed_grid(len(data['y']),settings['grid'])
    variants = tuple(settings['variants'])
    if not variants or len(set(variants)) != len(variants) or set(variants)-set(lin.VARIANTS):
        raise DefinitionError('Invalid Lin variants')
    priors = {t['prior_id']: t['prior'] for t in tasks}
    provider = PosteriorProvider(model, spec['inference'])
    rep = settings.get('replication_id', -1)
    out = fresh(output)
    write_json(out/'resolved.json', dict(settings=settings,spec=spec,runtime=runtime_fingerprint(spec),
        seed_rule='lin_nested_v1 fixed base 20260312; replication_id and path IDs',
        scope='new tutorial execution, not historical campaign replay'))
    try:
        base = provider.fit(data,spec['priors']['baseline'],lin.seed(model.name,rep,'baseline'))
        bp = lin.prepare_measures(lin.target_measures(base,targets,settings['posterior_draws'],lin.seed(model.name,rep,'baseline_target')),settings['alpha'])
        bank, audit = lin.path_bank(model,data,base,0,settings['paths'],rep)
        rows, curves = [], []
        for prior_id, prior in priors.items():
            c = lin.evaluate_bank(model,data,prior,bp,bank,targets,grid,provider,
                draws=settings['posterior_draws'],replication=rep,alpha=settings['alpha'],variants=variants)
            chosen = lin.select(c,grid,settings['alpha'])
            for v,variant in enumerate(variants):
                for j,target in enumerate(targets):
                    values = chosen['signed_m'][v,:,j]
                    rows.append(dict(prior_id=prior_id,method=variant,target=target.id,
                        epss=float(values.mean()),paths=len(values),boundary_rate=float(chosen['boundary'][v,:,j].mean()),
                        mean_cp1=float(chosen['selected_coverage'][v,:,j,0].mean()),
                        mean_cp2=float(chosen['selected_coverage'][v,:,j,1].mean())))
            curves.append(dict(prior_id=prior_id,coverages=c,selections=chosen))
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
        write_json(out/'curves.json',curves)
        write_json(out/'paths.json',audit)
        write_json(out/'posterior_accuracy.json',provider.accuracy)
        return certify(out,['resolved.json','results.csv','curves.json','paths.json','posterior_accuracy.json'],rows=len(rows))
    except Exception as e:
        write_json(out/'failure.json',{'type':type(e).__name__,'error':str(e)})
        raise


def run_lin(spec_path, output):
    """Compatibility command: translate old Lin YAML into the standard runner."""
    from .api import run, run_settings
    path = Path(spec_path).resolve()
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise DefinitionError('Lin specification must be a mapping')
    if 'experiment' not in settings:
        return run(path, output)
    allowed = {'schema_version','experiment','paths','posterior_draws','grid','variants','alpha','replication_id'}
    required = allowed - {'replication_id'}
    if set(settings)-allowed or required-set(settings) or settings.get('schema_version') != 1:
        raise DefinitionError('Invalid Lin compatibility specification')
    variants = settings['variants']
    if not isinstance(variants,list) or not variants or any(v not in lin.VARIANTS for v in variants) or len(set(variants))!=len(variants):
        raise DefinitionError('Invalid Lin variants')
    spec = load_spec(path.parent/settings['experiment'])
    spec['replication_id'] = settings.get('replication_id',-1)
    spec['methods'] = [dict(preset=v+'_v1',id=v,paths=settings['paths'],posterior_draws=settings['posterior_draws'],alpha=settings['alpha']) for v in variants]
    spec['candidates'] = dict(meaning='informative_fresh_size_reduction',unit=model_for(spec).unit,values=settings['grid'])
    return run_settings(spec,output)
