from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from priorlens import validate,run,TargetSpec
from priorlens.workflows import run_risk_summary,run_lin
from prioreffect_v2.experiments import load_spec,model_for
from prioreffect_v2.inference import PosteriorProvider

ROOT=Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_all_generic_examples_validate(self):
        for p in sorted((ROOT/'examples').glob('*/*.yaml')):
            if p.name in {'lin.yaml','lin_study.yaml'}:continue
            import yaml
            if yaml.safe_load(p.read_text()).get('workflow') == 'posterior':
                from priorlens import validate_posterior
                self.assertEqual(validate_posterior(p)['status'], 'valid')
                continue
            with self.subTest(path=str(p.relative_to(ROOT))):self.assertGreater(len(validate(p)),0)

    def test_custom_is_against_gaussian_oracle(self):
        spec=load_spec(ROOT/'examples/custom_2d/importance.yaml')
        model=model_for(spec);data=model.validate_data(spec['data']);prior=spec['priors']['interest']
        post=PosteriorProvider(model,spec['inference']).fit(data,prior,1469)
        x=np.column_stack([np.ones(len(data['y'])),data['x']]);s=model.sigma
        prec=np.diag(1/np.asarray(prior['sd'])**2)
        cov=np.linalg.inv(prec+x.T@x/s**2)
        mean=cov@(prec@prior['loc']+x.T@data['y']/s**2)
        target=TargetSpec.parse(spec['targets'][0]);measure=post.target(target,512,123)
        self.assertLess(np.max(np.abs(measure.mean-mean)/np.sqrt(np.diag(cov))),.003)
        self.assertLess(np.max(np.abs(measure.covariance-cov)/np.max(np.abs(cov))),.01)

    def test_risk_reductions_before_argmin_and_reference_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=ROOT/'examples/nig/quick.yaml'
            base=Path(tmp);out=base/'paired';run_risk_summary(spec,out)
            rows=pd.read_csv(out/'results.csv');banks=json.loads((out/'raw_risks.json').read_text())
            for b in banks:
                a=np.asarray(b['raw']);grid=np.asarray(b['grid'])
                for label,reduce in [('mean',np.mean),('median',np.median)]:
                    r=reduce(a,axis=1);expected=grid[np.argmin(abs(r[1:]-r[0]))]
                    row=rows[(rows.method==b['identity']['method'])&(rows.target==b['identity']['target'])&(rows.aggregation==label)]
                    self.assertEqual(float(row.epss.iloc[0]),expected)
            result=run(spec,base/'generic');self.assertEqual(result['status'],'pass')
            leaves=list((base/'generic').glob('**/result.json'))
            matched=0
            for leaf in leaves:
                value=json.loads(leaf.read_text())
                # Generic leaf identities include method and target mappings.
                if value.get('method_id') in ('reimherr','wiesenfarth'):
                    row=rows[(rows.method==value['method_id'])&(rows.target==value['target_id'])&(rows.aggregation=='mean')]
                    self.assertEqual(float(row.epss.iloc[0]),value['epss']);matched+=1
            self.assertEqual(matched,10)

    def test_lin_both_variants_and_fresh_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'lin';r=run_lin(ROOT/'examples/nig/lin.yaml',out)
            self.assertEqual(r['completed_tasks'],10)
            rows=pd.read_csv(out/'selected.csv');self.assertTrue(rows.epss.between(-1,1).all())
            self.assertEqual(set(rows.method_id),{'lin_ppd','lin_plugin'})
            with self.assertRaises(ValueError):run_lin(ROOT/'examples/nig/lin.yaml',out)

    def test_portable_source_and_table_hashes(self):
        for name,meta in json.loads((ROOT/'source_map.json').read_text())['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),meta['release_sha256'])
        for name,meta in json.loads((ROOT/'data/manifest.json').read_text())['tables'].items():
            self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),meta['sha256'])

    def test_option_catalog_complete(self):
        spec=importlib.util.spec_from_file_location('release_check',ROOT/'tools/check_release.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.check(check_manifest=False)['status'],'pass')


if __name__=='__main__':unittest.main()
