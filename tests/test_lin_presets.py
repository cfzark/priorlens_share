from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from priorlens.api import run_settings
from priorlens.workflows import _run_lin_reference
from prioreffect_v2.core import DefinitionError, NumericalFailure, TargetSpec
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task, run_task_group, merge, model_for
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.methods import resolve_method, run_method
from prioreffect_v2 import lin
from prioreffect_v2.methods import PRESETS

ROOT=Path(__file__).resolve().parents[1]


def specification(case='nig'):
    filename='quick_matching.yaml' if case=='lupus' else 'quick.yaml'
    spec=load_spec(ROOT/'examples'/case/filename)
    spec['replication_id']=-1
    spec['methods']=[dict(preset=v+'_v1',id=v,paths=2,posterior_draws=64,alpha=.05) for v in lin.VARIANTS]
    spec['candidates']=dict(meaning='informative_fresh_size_reduction',unit=model_for(spec).unit,values=[-1,0,1])
    if case=='lupus':spec['inference']=dict(backend='optimized',posterior_reuse='none')
    return spec


class LinPresets(unittest.TestCase):
    def test_model_and_preset_documentation(self):
        catalog=(ROOT/'docs/methods/index.md').read_text()
        for name in PRESETS:self.assertIn(name,catalog)
        models=(ROOT/'docs/models.md').read_text()
        for name in ['normal_known','poisson_gamma','beta_binomial','nig','eight_schools','lupus_logistic']:
            self.assertIn('| '+name+' |',models)

    def test_presets_and_component_validation(self):
        for v in lin.VARIANTS:
            m=resolve_method(v+'_v1');self.assertEqual(m['variant'],v)
            self.assertEqual(m['name'],'lin')
        for key,value in [('region','pdf_hdr'),('loss_family','historical'),('tie','negative'),
                          ('gap_weight',1),('cap_nominal',False),('paths',True),('alpha',float('nan')),('sampling','bootstrap')]:
            with self.subTest(key=key),self.assertRaises(DefinitionError):resolve_method(dict(preset='lin_ppd_v1',**{key:value}))

    def test_grid_and_support(self):
        spec=specification();n=len(spec['data']['y'])
        for values in [[-n-1,0],[0,n+1],[1,0],[0,.5]]:
            bad=copy.deepcopy(spec);bad['candidates']['values']=values
            with self.assertRaises(DefinitionError):resolved_tasks(bad)
        bad=copy.deepcopy(spec);bad['candidates']['meaning']='number_deleted'
        with self.assertRaises(DefinitionError):resolved_tasks(bad)
        bad=copy.deepcopy(spec);bad['targets'][0]['focal_unit_id']='school_1'
        with self.assertRaises(DefinitionError):resolved_tasks(bad)
        bad=copy.deepcopy(spec);bad['priors']['interest']={'family':'jeffreys'};bad['prior_grid']=[{'prior_id':'x'}];bad['candidates']['values']=[n]
        with self.assertRaises(DefinitionError):resolved_tasks(bad)
        bad=load_spec(ROOT/'examples/custom_1d/importance.yaml');bad['methods']=spec['methods'];bad['candidates']=spec['candidates']
        with self.assertRaises(DefinitionError):resolved_tasks(bad)

    def test_production_budgets_and_fixed_seed(self):
        spec=specification();spec['run_mode']='production'
        a=resolved_tasks(spec);spec['seed']=19;b=resolved_tasks(spec)
        self.assertEqual(a,b)
        del spec['methods'][0]['paths']
        with self.assertRaises(DefinitionError):resolved_tasks(spec)

    def test_all_three_cases_exact_reference_curves_and_selections(self):
        for case in ['nig','eight_schools','lupus']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp);spec=specification(case)
                # The old wrapper resolves model/targets through a non-Lin experiment.
                reference=copy.deepcopy(spec)
                reference['methods']=[dict(preset='cp_bootstrap_deletion_v2',paths=2,posterior_draws=64,region='central',invalid_path_policy='error')]
                reference['candidates']=dict(meaning='number_deleted',unit=spec['candidates']['unit'],values=[0,1])
                (tmp/'reference.yaml').write_text(yaml.safe_dump(reference))
                (tmp/'lin.yaml').write_text(yaml.safe_dump(dict(schema_version=1,experiment='reference.yaml',paths=2,posterior_draws=64,
                    grid=[-1,0,1],variants=list(lin.VARIANTS),alpha=.05,replication_id=-1)))
                _run_lin_reference(tmp/'lin.yaml',tmp/'old')
                old=json.loads((tmp/'old/curves.json').read_text())[0]
                run_settings(spec,tmp/'new')
                table=pd.read_csv(tmp/'new/selected.csv');cov=np.asarray(old['coverages'])
                selected=old['selections'];audit=json.loads((tmp/'old/paths.json').read_text())
                for row in table.itertuples():
                    v=lin.VARIANTS.index(row.method_id);j=[t['id'] for t in spec['targets']].index(row.target_id)
                    leaf=tmp/'new/leaves'/row.task_key
                    curves=json.loads((leaf/'curves.json').read_text());paths=json.loads((leaf/'paths.json').read_text())
                    for r in curves:
                        k=spec['candidates']['values'].index(r['m'])
                        np.testing.assert_array_equal([r['cp1'],r['cp2']],cov[v,r['path'],k,j])
                    for r in paths:
                        b=r['path'];self.assertEqual(r['signed_m'],selected['signed_m'][v][b][j])
                        self.assertEqual(r['tie_count'],selected['tie_count'][v][b][j])
                        self.assertEqual(r['selected_loss'],selected['loss'][v][b][j])
                    self.assertEqual(row.path_checksum,audit['checksum'])
                    self.assertEqual(row.epss,np.mean(np.asarray(selected['signed_m'])[v,:,j]))

    def test_shared_targets_fit_once_and_identity_keying(self):
        spec=specification();tasks=resolved_tasks(spec);model=model_for(spec);provider=PosteriorProvider(model,spec['inference'])
        def invoke(t):return run_method(model,t['data'],t['prior'],t['baseline'],TargetSpec.parse(t['target']),t['method'],t['candidates']['values'],t['inference'],t['seed'],provider=provider,replication_id=t['replication_id'],shared_targets=t['shared_targets'])
        with patch.object(model,'optimized',wraps=model.optimized) as fit:
            invoke(tasks[0]);count=fit.call_count
            for t in tasks[1:len(spec['targets'])]:invoke(t)
            self.assertEqual(fit.call_count,count)
            other=copy.deepcopy(tasks[0]);other['prior']['mu0']+=.3;invoke(other)
            self.assertGreater(fit.call_count,count)

    def test_grouped_versus_individual_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);spec=specification();spec['targets']=spec['targets'][:2];spec['methods']=spec['methods'][:1]
            spec['output_root']=str(tmp/'split');plan=prepare(spec)
            first=run_task(plan['manifest'],0);done=tmp/'split/leaves'/first['task_key']/'done.json';original=done.read_bytes()
            with patch('prioreffect_v2.lin.evaluate_bank',side_effect=NumericalFailure('injected failure')):
                with self.assertRaises(RuntimeError):run_task(plan['manifest'],1)
            task=json.loads((tmp/'split/tasks'/ (pd.read_csv(plan['manifest'],sep='\t').iloc[1].task_key+'.json')).read_text())
            self.assertFalse((tmp/'split/leaves'/task['task_key']/'done.json').exists())
            recovery=prepare(spec,resume=True);self.assertEqual(recovery['n_pending'],1)
            run_task_group(recovery['manifest'],[0]);self.assertEqual(merge(tmp/'split')['completed_tasks'],2)
            self.assertEqual(original,done.read_bytes())
            run_settings(spec,tmp/'grouped')
            for key in pd.read_csv(tmp/'split/manifest.tsv',sep='\t').task_key:
                for name in ['curves.json','paths.json']:
                    self.assertEqual((tmp/'split/leaves'/key/name).read_bytes(),(tmp/'grouped/leaves'/key/name).read_bytes())

    def test_zero_data_candidate_and_ties(self):
        spec=specification();n=len(spec['data']['y']);spec['candidates']['values']=[-n,0,n]
        spec['targets']=[spec['targets'][0]];spec['methods']=spec['methods'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_settings(spec,Path(tmp)/'out')['completed_tasks'],1)
        c=np.ones((1,1,4,1,2));grid=np.array([-2,-1,1,2])
        self.assertEqual(lin.select(c,grid)['signed_m'].item(),-1)


if __name__=='__main__':unittest.main()
