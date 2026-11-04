"""Portable five-method plots from certified compact tables; never fits a posterior."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METHODS=['Reimherr','Wiesenfarth','Jones','CP','Lin']
COLORS=['#0072B2','#D55E00','#CC79A7','#AA4499','#56B4E9']
ALIASES={'reimherr_mean':'reimherr','wiesenfarth_mean':'wiesenfarth',
         'jones_source':'jones','cp_bootstrap_deletion_H1':'cp','cp_central_H1':'cp'}
TARGETS={
 'nig':['joint_log','joint_variance','mu','log_sigma2','sigma2'],
 'eight_schools':['joint_log','joint_variance','log_tau','tau2'],
 'lupus':['beta_joint','beta0','beta1']}
TITLES={
 'nig':[r'$(\mu/0.1,\log(\sigma^2/0.01))$',r'$(\mu/0.1,\sigma^2/0.01)$',r'$\mu/0.1$',r'$\log(\sigma^2/0.01)$',r'$\sigma^2/0.01$'],
 'eight_schools':[r'$(\mu/10,\log(\tau/10))$',r'$(\mu/10,\tau^2/100)$',r'$\log(\tau/10)$',r'$\tau^2/100$'],
 'lupus':[r'Joint $(\beta_0,\beta_1)$',r'Intercept $\beta_0$',r'Slope $\beta_1$']}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load(case):
    name={'nig':'nig_real.csv','eight_schools':'eight_schools_real.csv','lupus':'lupus_realdata.csv'}[case]
    meta=json.loads((ROOT/'data/manifest.json').read_text())['tables'][name]
    if sha(ROOT/'data'/name)!=meta['sha256']:raise ValueError('Changed input table: '+name)
    d=pd.read_csv(ROOT/'data'/name).rename(columns={'beta1':'location'})
    d['method']=d.method.replace(ALIASES)
    if d.duplicated(['method','target','location']).any() or not np.isfinite(d.epss).all():
        raise ValueError('Invalid plotting matrix')
    return d,dict(table=name,sha256=meta['sha256'])

def plot(case,out):
    d,source=load(case)
    targets=TARGETS[case];n=len(targets)
    fig,axes=plt.subplots(n,5,figsize=(18,3*n),sharex=True,sharey=True,squeeze=False)
    variants=[['reimherr','reimherr_median'],['wiesenfarth','wiesenfarth_median'],
              ['jones'],['cp','cp_pdf_hdr_H1'],['lin_ppd']]
    expected={'nig':82,'eight_schools':100,'lupus':33}[case]
    limits={'nig':(-30,15),'eight_schools':(-20,20),'lupus':(-30,30)}[case]
    if not d.epss.between(*limits).all():raise ValueError('Curve would be clipped')
    for j,family in enumerate(variants):
        for i,target in enumerate(targets):
            ax=axes[i,j];ax.axhline(0,color='.5',lw=.7)
            for v,method in enumerate(family):
                a=d[(d.method==method)&(d.target==target)].sort_values('location')
                if a.empty and (method.endswith('_median') and case=='eight_schools' or method=='cp_pdf_hdr_H1' and (case!='lupus' or target!='beta_joint')):continue
                if len(a)!=expected:raise ValueError(f'Missing cells: {case}/{method}/{target}')
                label=('Median risk' if v else 'Mean risk') if j<2 else ('PDF-HDR' if v else 'Central') if j==3 else METHODS[j]
                ax.plot(a.location,a.epss,color='#009E73' if method=='cp_pdf_hdr_H1' else COLORS[j],
                        ls='--' if v else '-',lw=1.8,label=label)
            if case=='nig':
                ax.axvline(1.8044444444444445,color='.5',ls='--',lw=.8)
                ax.axvline(1.9,color='black',ls=':',lw=.8)
            elif case=='eight_schools':ax.axvline(25,color='.5',ls=':',lw=.8)
            else:ax.axvline(2.856976386886537,color='#B44A4A',ls='--',lw=.8)
            ax.set_ylim(*limits);ax.grid(axis='y',alpha=.12)
            ax.spines[['top','right']].set_visible(False)
            if i==0:ax.set_title(METHODS[j])
            if j==0:ax.set_ylabel(TITLES[case][i])
            if i==0 and (j<2 or j==3):ax.legend(fontsize=8,frameon=False)
    fig.supxlabel({'nig':r'Prior location $\mu_0$','eight_schools':r'Prior scale $\tau_0$','lupus':r'Prior location for $\beta_1$'}[case])
    fig.supylabel('Signed EPSS',x=.006)
    fig.tight_layout(rect=(.025,.025,1,1))
    for ext in ['pdf','png']:fig.savefig(out/f'{case}_real.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    d.to_csv(out/f'{case}_values.csv',index=False)
    return dict(case=case,rows=len(d),source=source,ylim=limits)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    results=[plot(case,args.output) for case in TARGETS]
    manifest=dict(status='pass',precision_certified=False,cases=results,
        script_sha256=sha(Path(__file__)),scope='Saved numeric curves, no raw-leaf precision revalidation',
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
