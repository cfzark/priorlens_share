from pathlib import Path
import copy
import hashlib
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from priorlens.stan_posterior import (load_posterior_spec, validate_posterior, indexed_name,
    importance_draws, target_summaries, posterior, _nuts_draws)
from prioreffect_v2.core import DefinitionError, NumericalFailure, TargetSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT/'examples/lupus_stan/posterior.yaml'


class GaussianBridge:
    """A known correlated Gaussian in unconstrained coordinates."""
    covariance = np.array([[.7, .2], [.2, 1.1]])
    center = np.array([.4, -.2])

    def __init__(self, *args, **kwargs):
        self.precision = np.linalg.inv(self.covariance)

    def param_unc_num(self): return 2
    def param_names(self, **kwargs): return ['beta0', 'beta1']

    def log_density(self, z, propto=True, jacobian=True):
        assert jacobian
        delta = z-self.center
        return -.5*delta@self.precision@delta

    def log_density_gradient(self, z, **kwargs):
        return self.log_density(z, **kwargs), -self.precision@(z-self.center)

    def log_density_hessian(self, z, **kwargs):
        lp, g = self.log_density_gradient(z, **kwargs)
        return lp, g, -self.precision

    def param_constrain(self, z, **kwargs): return z


class PositiveBridge(GaussianBridge):
    """Second natural coordinate is exp(z[1]); the kernel includes Jacobian."""
    def param_constrain(self, z, **kwargs): return np.array([z[0], np.exp(z[1])])


class StanPosteriorTests(unittest.TestCase):
    def _write_spec(self, folder, change=None):
        spec = yaml.safe_load(EXAMPLE.read_text())
        spec['model']['stan'] = str(EXAMPLE.parent/'model.stan')
        spec['data'] = str(EXAMPLE.parent/'data.json')
        spec['inference'].update(backend='importance', sobol_power=11, is_min_ess=100)
        if change: change(spec)
        path = folder/'config.yaml'
        path.write_text(yaml.safe_dump(spec))
        return path

    def test_lupus_inputs_match_existing_example(self):
        spec = load_posterior_spec(EXAMPLE)
        old = yaml.safe_load((ROOT/'examples/lupus/quick_matching.yaml').read_text())
        self.assertEqual(spec['data']['N'], 55)
        np.testing.assert_array_equal(spec['data']['x'], old['data']['x'])
        np.testing.assert_array_equal(spec['data']['y'], old['data']['y'])
        self.assertTrue(all(type(y) is int for y in spec['data']['y']))
        result = validate_posterior(EXAMPLE)
        self.assertEqual((result['posterior_fits'], result['targets_per_fit']), (4, 3))
        for name in ('baseline', 'interest'):
            actual, prior = spec['priors'][name], old['priors'][name]
            self.assertEqual(actual['prior_loc'], prior['loc'])
            self.assertEqual(actual['prior_scale'], prior['scale'])
            self.assertEqual(actual['prior_df'], prior['df'])

    def test_schema_rejects_ambiguous_inputs(self):
        changes = [lambda s:s.update(methods=['lin_ppd_v1']),
                   lambda s:s['priors']['baseline'].update(N=55),
                   lambda s:s['prior_grid'].append(s['prior_grid'][0]),
                   lambda s:s['prior_grid'][0].update(prior_sclae=[1, 1]),
                   lambda s:s['inference'].update(sobol_power=True),
                   lambda s:s['inference'].update(adapt_delta=1),
                   lambda s:s.update(seed=2**32),
                   lambda s:s['targets'].append(s['targets'][0])]
        with tempfile.TemporaryDirectory() as d:
            for change in changes:
                with self.subTest(change=change), self.assertRaises((DefinitionError, ValueError)):
                    load_posterior_spec(self._write_spec(Path(d), change))

    def test_parameter_names_and_target_coordinates(self):
        self.assertEqual(indexed_name('beta.2'), 'beta[2]')
        self.assertEqual(indexed_name('Sigma.1.2'), 'Sigma[1,2]')
        target = TargetSpec.parse(dict(id='joint', coordinates=[dict(variable='mu'),
                                     dict(variable='sigma', transform='log')]))
        values = np.array([[0., np.exp(1)], [2., np.exp(3)]])
        rows, cov = target_summaries(values, np.array([.5, .5]), ['mu', 'sigma'], [target], 'p')
        np.testing.assert_allclose([r['mean'] for r in rows], [1, 2])
        np.testing.assert_allclose(cov['joint'], np.ones((2, 2)))
        with self.assertRaises(DefinitionError):
            target_summaries(values, np.array([.5, .5]), ['wrong', 'sigma'], [target], 'p')

    def test_generic_is_gaussian_and_positive_transform(self):
        for model in (GaussianBridge(), PositiveBridge()):
            points, weights, diag, extra = importance_draws(model, {'sobol_power': 15}, 123)
            self.assertTrue(diag['converged'], diag)
            z = extra['unconstrained']
            mean = weights@z
            cov = ((z-mean).T*weights)@(z-mean)
            np.testing.assert_allclose(mean, model.center, atol=.002)
            np.testing.assert_allclose(cov, model.covariance, atol=.005)
            if isinstance(model, PositiveBridge):
                np.testing.assert_allclose(points[:, 1], np.exp(z[:, 1]))
            np.testing.assert_allclose(diag['proposal']['local_covariance'], model.covariance)

    def test_invalid_curvature_rejected(self):
        model = GaussianBridge()
        model.log_density_hessian = lambda z, **kwargs: (0., np.zeros(2), np.eye(2))
        with self.assertRaises(NumericalFailure):
            importance_draws(model, {'sobol_power': 10}, 123)

    def test_workflow_prior_payloads_and_fit_sharing(self):
        payloads = []
        def construct(library, payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            return GaussianBridge()
        fake = types.SimpleNamespace(StanModel=construct, compile_model=lambda p:p+'.so')
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', bridgestan=fake), \
                patch('priorlens.stan_posterior.importlib.metadata.version', return_value='test'):
            root = Path(d); spec = self._write_spec(root)
            report = posterior(spec, root/'out')
            self.assertEqual(report['posterior_fits'], 4)
            self.assertEqual(report['summary_rows'], 16)
            self.assertEqual(len(payloads), 4)  # Not twelve target-wise fits.
            self.assertEqual([p['prior_loc'] for p in payloads], [[0, 0], [0, 1], [0, 3], [0, 5]])
            self.assertEqual([p['prior_scale'] for p in payloads], [[10, 1000]]+[[10, 2.5]]*3)
            for p in payloads:
                self.assertEqual(len(p['y']), 55)
            out = root/'out'
            done = json.loads((out/'done.json').read_text())
            for name, h in done['artifacts'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), h)
            rows = pd.read_csv(out/'summary.csv')
            self.assertEqual(len(rows), 16)
            with self.assertRaises(FileExistsError): posterior(spec, out)
            # Removing a target preserves the same full posterior and seed.
            changed = self._write_spec(root, lambda s:s.update(targets=s['targets'][1:]))
            posterior(changed, root/'subset')
            for index in range(4):
                with np.load(out/f'fit_{index:04d}/posterior.npz') as a, \
                     np.load(root/f'subset/fit_{index:04d}/posterior.npz') as b:
                    np.testing.assert_array_equal(a['points'], b['points'])
                    np.testing.assert_array_equal(a['weights'], b['weights'])

    def test_failed_gate_saves_draws_without_done(self):
        fake = types.SimpleNamespace(StanModel=GaussianBridge, compile_model=lambda p:p+'.so')
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', bridgestan=fake), \
                patch('priorlens.stan_posterior.importlib.metadata.version', return_value='test'):
            root = Path(d)
            spec = self._write_spec(root, lambda s:s['inference'].update(is_min_ess=1000000))
            with self.assertRaises(NumericalFailure): posterior(spec, root/'out')
            self.assertTrue((root/'out/fit_0000/posterior.npz').exists())
            self.assertTrue((root/'out/fit_0000/diagnostics.json').exists())
            self.assertTrue((root/'out/failure.json').exists())
            self.assertFalse((root/'out/done.json').exists())
            self.assertFalse((root/'out/fit_0000/done.json').exists())

    def test_nuts_metadata_extraction_and_all_parameter_diagnostics(self):
        class Fit:
            column_names = ('lp__', 'beta0', 'beta1', 'prediction')
            def draws(self): return np.arange(32.).reshape(4, 2, 4)
            def summary(self):
                return pd.DataFrame({'R_hat':[1., 1.], 'ESS_bulk':[800, 700],
                                     'ESS_tail':[600, 500]}, index=['beta0', 'beta1'])
            def method_variables(self):
                return {'divergent__':np.zeros((4, 2)), 'treedepth__':np.ones((4, 2))}
        stan = types.SimpleNamespace(src_info=lambda:{'parameters':{'beta0':{}, 'beta1':{}}},
                                     sample=lambda **kwargs:Fit())
        target = TargetSpec.parse(dict(id='b0', coordinates=[dict(variable='beta0')]))
        with tempfile.TemporaryDirectory() as d:
            p,w,names,diag,extra = _nuts_draws(stan, {}, dict(chains=2,warmup=10,sampling=4),
                                               123, Path(d)/'chains', [target])
            self.assertEqual(names, ['beta0', 'beta1'])
            self.assertEqual(p.shape, (8, 2))
            self.assertEqual(extra['chain_draws'].shape, (4, 2, 2))
            self.assertEqual(diag['ess_tail'], 500)
            self.assertTrue(diag['converged'])
            np.testing.assert_array_equal(p[:4], Fit().draws()[:, 0, 1:3])

    @unittest.skipUnless(os.getenv('PRIORLENS_STAN_TEST_LIBRARY'), 'Set compiled library for Stan integration test')
    def test_actual_stan_density_matches_builtin_lupus(self):
        import bridgestan as bs
        from prioreffect_v2.models import BuiltinModel
        s = load_posterior_spec(EXAMPLE)
        builtin = BuiltinModel('lupus_logistic')
        data = builtin.validate_data(s['data'])
        points = np.array([[-2., -1.], [0., 0.], [-1., 3.], [2., 7.]])
        priors = [s['priors']['baseline']]+[
            {**s['priors']['interest'], **{k:v for k,v in c.items() if k!='prior_id'}} for c in s['prior_grid']]
        for hyper in priors:
            stan = bs.StanModel(os.environ['PRIORLENS_STAN_TEST_LIBRARY'], {**s['data'], **hyper})
            actual = np.array([stan.log_density(x, propto=True, jacobian=True) for x in points])
            prior = dict(family='student_t',loc=hyper['prior_loc'],scale=hyper['prior_scale'],df=hyper['prior_df'])
            expected = builtin.logkernel(points, data, prior)
            np.testing.assert_allclose(actual-actual[0], expected-expected[0], rtol=1e-12, atol=1e-12)


if __name__ == '__main__': unittest.main()
