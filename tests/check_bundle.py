"""Offline integrity/coverage checks; no fitting and no network access."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cert=json.loads((ROOT/'data/manifest.json').read_text())
    assert cert['status']=='pass' and cert['figure_labels']==['Reimherr','Wiesenfarth','Jones','CP']
    assert cert['risk_aggregation']==dict(NIG='mean',lupus='median',eight_school='mean')
    counts={'NIG/simulation_var_0p25_5x4.csv':5100,'NIG/simulation_var_20_5x4.csv':5100,
        'NIG/real_3x4.csv':984,'lupus/real_4x4.csv':8976,'lupus/simulation_3x4.csv':6732,
        'eight_school/real_4x4.csv':1600,'eight_school/simulation_4x4.csv':1600}
    assert set(counts)==set(cert['tables'])
    for name,h in cert['tables'].items():
        path=ROOT/'data/plot_tables'/name
        assert sha(path)==h,name
        f=pd.read_csv(path)
        assert len(f)==counts[name] and np.isfinite(f.epss).all(),name
        assert set(f.method)=={'reimherr','wiesenfarth','jones','cp'},name
        keys=['target','method']+(['prior_loc','sigma2_prior_mean'] if 'simulation_var' in name else
            ['grid_id'] if name.startswith('lupus') else ['prior_id'])
        assert not f.duplicated(keys).any(),name
        if name=='lupus/simulation_3x4.csv':assert (f.reps==30).all()
        if name=='eight_school/simulation_4x4.csv':assert (f.reps==50).all()
    assets=ROOT/'paper/V2_figure/realdata'
    ac=json.loads((assets/'manifest.json').read_text())
    for name,h in ac['outputs'].items():assert sha(assets/name)==h,name
    tex=(ROOT/'paper/realdata.tex').read_text()
    active_tex=re.sub(r'(?<!\\)%[^\n]*','',tex)
    refs=re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}',active_tex)
    assert len(refs)==6 and all((assets/n).is_file() for n in refs)
    assert not re.search(r'\\(?:input|include|bibliography)\s*\{',tex)
    settings=list((ROOT/'configs/nig_simulation').glob('setting_*.json'))
    assert len(settings)==40
    sys.path.insert(0,str(ROOT/'compute'))
    from run_legacy_nig import command_for
    for path in settings:
        record=json.loads(path.read_text());c=record['configuration']
        assert c['R']==50 and c['baseline']=='jeffreys' and c['sigma2_prior_var'] in (.25,20)
        command=command_for(record,Path('unused'),full=True,rep=0)
        assert Path(command[1]).is_file()
    inventory=ROOT/'bundle_manifest.json'
    if inventory.exists():
        for name,h in json.loads(inventory.read_text())['files'].items():assert sha(ROOT/name)==h,name
    print('PASS: seven tables, seven figure pairs, six active manuscript figures, 40 NIG settings, and bundle hashes')


if __name__=='__main__':main()
