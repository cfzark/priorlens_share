"""Run one retained NIG simulation setting with its original model-specific runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def command_for(record, output, full=False, rep=None):
    c=record['configuration'];source=record['source_id'];method=c['method']
    joint='joint2d' in method
    filename=('run_nig2d_sigma2_mean_heatmap.py' if joint else
              'run_nig_sigma2_marginal_heatmap.py' if c.get('target')=='marginal_sigma2' else
              'run_nig_marginal_sigma2_mean_heatmap.py')
    options={'method':method,'output-root':str(output),'R':c['R'] if full else 1,
        'rep-start':0 if rep is None else rep,'rep-end':c['R'] if full and rep is None else (0 if rep is None else rep)+1,
        'n':c['n'],'prior-values':','.join(map(str,c['mu_grid'])) if full else '0',
        'sigma2-mean-values':','.join(map(str,c['sigma2_mean_grid'])) if full else '1',
        'sigma2-prior-var':c['sigma2_prior_var'],'tau0-sq':c['tau0_sq'],
        'true-mu':c['true_mu'],'true-sigma':c['true_sigma'],
        'data-seed':c['data_seed'],'base-seed':c['base_seed'],'n-jobs':1,'verbose':0,
        'B':c.get('B',1000) if full else 4,'alpha':c.get('alpha',.05)}
    if joint:
        options.update(parameterization=c['parameterization'],
            **{'joint-draws':c.get('joint_draws',512) if full else 64,
               'n-projections':c.get('n_projections',16) if full else 4})
        for key in ('mu_scale','sigma2_scale','log_sigma2_scale'):
            if key in c:options[key.replace('_','-')]=c[key]
    elif c.get('target')=='marginal_sigma2':
        options['coordinate']=c['coordinate']
    else:
        options.update(S=c.get('S',500) if full else 2)
    if joint or c.get('target')=='marginal_sigma2':
        options['outer-reps']=c.get('outer_reps',500) if full else 2
    metric=c.get('risk_metric')
    if metric:options['risk-metric']='squared_mean' if metric=='trace_mse' else metric
    if method.startswith('cp'):
        options['sampling-method']='bootstrap_deletion'
        options['min-retained']=3
    for family in ('reimherr','wiesenfarth','jones','cp'):
        if not method.startswith(family):continue
        values=c[family+'_m_grid'] if full else ([20,21] if family=='jones' else [-1,0,1] if family in ('reimherr','wiesenfarth') else [0,1])
        options[family+'-m-grid']=','.join(map(str,values))
    cmd=[sys.executable,str(ROOT/'scripts'/filename)]+[f'--{k}={v}' for k,v in options.items()]
    if not joint and c.get('target')=='marginal_mu':cmd.append('--no-interpolate')
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting',type=Path,required=True,help='configs/nig_simulation/setting_XX.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full',action='store_true',help='Use saved paper grids and repetitions; otherwise tiny smoke')
    p.add_argument('--rep',type=int,help='Compute only this observed-data replication')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Use a fresh output directory')
    record=json.loads(a.setting.read_text())
    if a.rep is not None and not 0<=a.rep<record['configuration']['R']:p.error('Invalid replication')
    cmd=command_for(record,out/'run',a.full,a.rep)
    if a.dry_run:
        print(json.dumps({'source_id':record['source_id'],'command':cmd},indent=2));return
    out.mkdir(parents=True)
    (out/'request.json').write_text(json.dumps({'source_id':record['source_id'],'full':a.full,'command':cmd},indent=2)+'\n')
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env.setdefault(key,'1')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    print('Runner completed; inspect generated config.json, results.csv and summary.csv.')


if __name__=='__main__':main()
