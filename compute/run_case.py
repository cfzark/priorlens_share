"""Portable sequential entry points; smoke sizes are intentionally not paper sizes."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
sys.path[:0]=[str(ROOT/'v2/src'),str(ROOT/'function')]
import numpy as np
import pandas as pd
import yaml
from prioreffect_v2.core import native, write_json


def generic(args, out):
    from prioreffect_v2.experiments import load_spec, prepare, run_task_group, merge
    school=args.case.startswith('eight_school')
    name='eight_school_real' if school else 'midge_real'
    spec=json.loads((ROOT/'configs'/f'{name}.json').read_text())
    spec['output_root']=str(out/'run')
    if args.case=='eight_school_simulation':
        if not 0<=args.rep<50: raise ValueError('Schools replication must be 0..49')
        rng=np.random.default_rng(321+args.rep)
        s=np.array([15,10,16,11,9,11,10,18])
        theta=rng.normal(8,6,8)
        spec.update(data={'y':rng.normal(theta,s).tolist(),'s':s.tolist()},replication_id=args.rep)
    if args.cell is not None:
        spec['prior_grid']=[spec['prior_grid'][args.cell]]
    elif args.size=='smoke':
        key,value=('tau_scale',6.) if school else ('mu0',1.9)
        spec['prior_grid']=[p for p in spec['prior_grid'] if p[key]==value]
    if args.family:
        spec['methods']=[m for m in spec['methods'] if m['id'].split('_')[0]==args.family]
    if args.size=='smoke':
        spec['run_mode']='smoke'
        for m in spec['methods']:
            for field,value in [('risk_reps',4),('paths',2),('posterior_draws',256),
                                ('quantile_grid_size',128),('projections',8)]:
                if field in m:m[field]=value
            m['candidates']['values']=[-1,0,1] if m['id'] in ('reimherr','wiesenfarth') else [0,1]
    out.mkdir(parents=True)
    # The generic loader is YAML-based; serialize floats with YAML's explicit
    # exponent syntax rather than feeding it ambiguous JSON exponent tokens.
    path=out/'spec.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False))
    prepared=prepare(load_spec(path))
    run_task_group(prepared['manifest'],list(range(prepared['n_pending'])))
    result=merge(spec['output_root'])
    write_json(out/'validation.json',{'status':'pass','case':args.case,'size':args.size,
        'meaning':'New execution; not certification of historical paper equality','merge':result})
    print(json.dumps(native(result),indent=2))


def lupus(args,out):
    from prioreffect_v2.lupus_experiment import baseline_risk_cache, risk_shard, matching_shard, summarize_parts
    mode='simulation' if args.case.endswith('simulation') else 'real'
    spec=json.loads((ROOT/'configs'/f'lupus_{mode}.json').read_text())
    rep=args.rep if mode=='simulation' else -1
    if mode=='simulation' and not 0<=rep<30:raise ValueError('Lupus replication must be 0..29')
    cells=spec['cells']
    if args.cell is not None:cells=[cells[args.cell]]
    elif args.size=='smoke':cells=[c for c in cells if (c['beta0'],c['beta1'])==(-1,3)]
    if args.size=='smoke':
        spec.update(repetitions=2,risk_grid=[-1,0,1],matching_grid=[0,1])
    out.mkdir(parents=True); (out/'parts').mkdir()
    spec['cells']=cells
    write_json(out/'spec.json',dict(spec,rep=rep,size=args.size))
    families=['reimherr','wiesenfarth','jones','cp_bootstrap_deletion']
    if args.family:families=[f for f in families if f.split('_')[0]==args.family]
    caches={};rows=[]
    for family in families:
        task=dict(rep=rep,family=family,start=0,end=spec['repetitions'])
        if family in ('reimherr','wiesenfarth'):
            caches[family]=baseline_risk_cache(spec,task)
            write_json(out/'parts'/f'{family}_baseline.json',caches[family])
        for cell in cells:
            task=dict(task,cell=cell)
            if family in caches:
                part=risk_shard(spec,task,caches[family])
                part['baseline_samples']=caches[family]['values']
            else:
                part=matching_shard(spec,task)
            # Preserve diagnostics and full loss banks, not just final selected EPSS.
            write_json(out/'parts'/f'{family}_{cell["grid_id"]}.json',dict(task=task,result=part))
            selected=summarize_parts(spec,[(task,part)])
            rows.extend(r for r in selected if not r['method'].endswith('D0_cap'))
            print(f'{family}: cell {cell["grid_id"]} complete',flush=True)
    frame=pd.DataFrame(rows)
    expected=len(cells)*len(families)*len(spec['targets'])
    if len(frame)!=expected or frame.duplicated(['rep','grid_id','method','target']).any() or not np.isfinite(frame.epss).all():
        raise ValueError('Incomplete/invalid selected results')
    frame.to_csv(out/'results.csv',index=False)
    write_json(out/'validation.json',dict(status='pass',rows=len(frame),case=args.case,size=args.size,
        meaning='New computation using supplied science modules; not a historical-equality certificate'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=['midge_real','eight_school_real','eight_school_simulation','lupus_real','lupus_simulation'])
    p.add_argument('--size',choices=['smoke','full'],default='smoke')
    p.add_argument('--rep',type=int,default=0,help='One observed-data simulation replication')
    p.add_argument('--cell',type=int,help='Prior-grid row index; default all in full mode, one in smoke mode')
    p.add_argument('--family',choices=['reimherr','wiesenfarth','jones','cp'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Output exists; use a fresh directory. Failed outputs are preserved for inspection.')
    if a.cell is not None and a.cell<0:p.error('--cell must be nonnegative')
    (lupus if a.case.startswith('lupus') else generic)(a,out)


if __name__=='__main__':main()
