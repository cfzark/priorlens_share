"""Redraw all seven manuscript figures using only the bundled compact tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layouts import heatmaps,line_atlas,require

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    a=p.parse_args();out=a.output.resolve()
    require(not out.exists(),'Use a new output directory')
    c=json.loads((ROOT/'data/manifest.json').read_text())
    tables={}
    for name,h in c['tables'].items():
        path=ROOT/'data/plot_tables'/name
        require(hashlib.sha256(path.read_bytes()).hexdigest()==h,f'Changed source table: {name}')
        tables[name]=pd.read_csv(path)
    for case in ['NIG','lupus','eight_school']:(out/case).mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    nt=['joint_log','joint_direct','marginal_mu','marginal_log','marginal_direct']
    nl=[r'Joint $(\mu,\log\sigma^2)$',r'Joint $(\mu,\sigma^2)$',r'Marginal $\mu$',r'Marginal $\log\sigma^2$',r'Marginal $\sigma^2$']
    for tag,var in [('0p25',.25),('20',20)]:
        stem=f'simulation_var_{tag}_5x4'
        heatmaps(tables['NIG/'+stem+'.csv'],nt,nl,out/'NIG',stem,
            rf'NIG simulation: $R=50$, $\mathrm{{Var}}_\pi(\sigma^2)={var:g}$ (R/W mean risk)',
            'prior_loc','sigma2_prior_mean',truth=(0,1),nig=True)
    line_atlas(tables['NIG/real_3x4.csv'],['mu','log_sigma2','joint_log'],
        [r'Marginal $\mu/0.1$',r'Marginal $\log(\sigma^2/0.01)$',r'Joint $(\mu/0.1,\log(\sigma^2/0.01))$'],
        out/'NIG','real_3x4','mu0','Midge wing lengths: real data (R/W mean risk)',limit=15)
    heatmaps(tables['lupus/real_4x4.csv'],['beta_joint','beta1','prediction_joint','risk_difference'],
        [r'Joint $(\beta_0,\beta_1)$',r'Marginal $\beta_1$',r'Joint $(p(0),p(2))$',r'$p(2)-p(0)$'],
        out/'lupus','real_4x4','Lupus real data (R/W median risk)','beta0','beta1',
        mle=(-.8839755751063929,2.856976386886537))
    heatmaps(tables['lupus/simulation_3x4.csv'],['beta_joint','beta0','beta1'],
        [r'Joint $(\beta_0,\beta_1)$',r'Marginal $\beta_0$',r'Marginal $\beta_1$'],
        out/'lupus','simulation_3x4',r'Lupus simulation: $R=30$, true $\beta=(-1,3)$ (R/W median risk)',
        'beta0','beta1',truth=(-1,3))
    targets=['joint_log','joint_variance','log_tau','tau2']
    labels=[r'Joint $(\mu/10,\log(\tau/10))$',r'Joint $(\mu/10,\tau^2/100)$',r'Marginal $\log(\tau/10)$',r'Marginal $\tau^2/100$']
    real=tables['eight_school/real_4x4.csv'];sim=tables['eight_school/simulation_4x4.csv']
    limit=5*np.ceil(max((sim.epss-sim.sd).abs().max(),(sim.epss+sim.sd).abs().max(),real.epss.abs().max())/5)
    line_atlas(real,targets,labels,out/'eight_school','real_4x4','tau_scale','Eight Schools real data (R/W mean risk)',schools=True,limit=limit)
    line_atlas(sim,targets,labels,out/'eight_school','simulation_4x4','tau_scale',
        r'Eight Schools simulation: $R=50$, true $\mu=8$, $\tau=6$ (R/W mean risk)',schools=True,simulation=True,limit=limit)
    manifest=dict(status='pass',input_tables=c['tables'],outputs={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS: seven PDF/PNG pairs in',out)


if __name__=='__main__':main()
