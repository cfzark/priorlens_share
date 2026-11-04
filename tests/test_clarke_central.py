"""Independent integration oracles and end-to-end Clarke central-region tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import integrate, stats

from prioreffect_v2.core import Coordinate, TargetSpec, Measure, DefinitionError, NumericalFailure
from prioreffect_v2.central_tkl import central_truncated_kl
from prioreffect_v2.regions import central_region
from prioreffect_v2.models import BuiltinModel
from prioreffect_v2.methods import resolve_method, run_method
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task, merge

ROOT = Path(__file__).resolve().parents[1]


def scalar_measure(rv, size=512):
    x = rv.ppf((np.arange(size) + .5) / size)
    return Measure(x, np.ones(size), logpdf=lambda z: rv.logpdf(z[:, 0]), ppf=rv.ppf, cdf=rv.cdf)


def scalar_oracle(p, q, lo, hi):
    pa, qa = p.cdf(hi) - p.cdf(lo), q.cdf(hi) - q.cdf(lo)
    expectation = integrate.quad(lambda x: p.pdf(x) * (p.logpdf(x) - q.logpdf(x)), lo, hi,
                                 epsabs=1e-11, epsrel=1e-11)[0]
    return expectation / pa + np.log(qa / pa)


class ClarkeCentral(unittest.TestCase):
    def test_scalar_quad_oracles_and_common_region(self):
        pairs = [(stats.norm(.3, .7), stats.norm(1.5, 1.2)),
                 (stats.gamma(3., scale=.5), stats.gamma(5., scale=.3)),
                 (stats.beta(2., 5.), stats.beta(4., 3.)),
                 (stats.t(4, loc=.2, scale=.8), stats.t(8, loc=1., scale=1.1))]
        for p, q in pairs:
            with self.subTest(family=p.dist.name):
                result = central_truncated_kl(scalar_measure(p), scalar_measure(q))
                lo, hi = p.ppf([.025, .975])
                self.assertAlmostEqual(result["loss"], scalar_oracle(p, q, lo, hi), places=8)
                diag = result["region_diagnostics"]
                self.assertAlmostEqual(diag["p_mass"], .95, places=8)
                self.assertAlmostEqual(diag["q_mass"], q.cdf(hi) - q.cdf(lo), places=8)
                self.assertEqual(diag["region_source"], "candidate_baseline_posterior")

    def test_independent_2d_normal_oracle(self):
        u = (np.arange(64) + .5) / 64
        a, b = np.meshgrid(stats.norm.ppf(u), stats.norm.ppf(u))
        points = np.column_stack([a.ravel(), b.ravel()])
        p_rv = stats.multivariate_normal([0, 0], np.eye(2))
        q_rv = stats.multivariate_normal([.7, -.2], np.diag([1.5**2, .8**2]))
        p = Measure(points, np.ones(len(points)), logpdf=p_rv.logpdf)
        q = Measure(points * [1.5, .8] + [.7, -.2], np.ones(len(points)), logpdf=q_rv.logpdf)
        result = central_truncated_kl(p, q)
        r = central_region(p)
        expected = sum(scalar_oracle(stats.norm(), stats.norm(mu, sd), lo, hi)
                       for mu, sd, lo, hi in zip([.7, -.2], [1.5, .8], r.lower, r.upper))
        self.assertAlmostEqual(result["loss"], expected, places=9)
        self.assertAlmostEqual(result["region_diagnostics"]["p_mass"],
                               np.prod(stats.norm.cdf(r.upper) - stats.norm.cdf(r.lower)), places=9)

    def test_correlated_2d_independent_integration(self):
        rng = np.random.default_rng(38)
        p_rv = stats.multivariate_normal([0, .2], [[1, .6], [.6, 1.5]])
        q_rv = stats.multivariate_normal([.8, -.3], [[2, -.4], [-.4, 1.]])
        p = Measure(p_rv.rvs(512, random_state=rng), np.ones(512), logpdf=p_rv.logpdf)
        q = Measure(q_rv.rvs(512, random_state=rng), np.ones(512), logpdf=q_rv.logpdf)
        result = central_truncated_kl(p, q)
        r = central_region(p)
        def integral(f):
            return integrate.dblquad(lambda y, x: f([x, y]), r.lower[0], r.upper[0],
                lambda x: r.lower[1], lambda x: r.upper[1], epsabs=1e-8, epsrel=1e-8)[0]
        pa, qa = integral(p_rv.pdf), integral(q_rv.pdf)
        numerator = integral(lambda z: p_rv.pdf(z) * (p_rv.logpdf(z) - q_rv.logpdf(z)))
        self.assertAlmostEqual(result["loss"], numerator / pa + np.log(qa / pa), places=7)

    def test_self_comparison(self):
        p = scalar_measure(stats.gamma(3., scale=.5))
        self.assertAlmostEqual(central_truncated_kl(p, p)["loss"], 0., places=13)
        x = np.random.default_rng(9).normal(size=(256, 2))
        joint = Measure(x, np.ones(256), logpdf=stats.multivariate_normal([0, 0], np.eye(2)).logpdf)
        self.assertAlmostEqual(central_truncated_kl(joint, joint)["loss"], 0., places=13)

    def test_kde_density_normalization(self):
        rng = np.random.default_rng(32)
        p = Measure(rng.normal(size=80), np.ones(80))
        q = Measure(rng.normal(.7, 1.2, size=80), np.ones(80))
        result = central_truncated_kl(p, q, density_engine="kde")
        r = central_region(p)
        kp, kq = stats.gaussian_kde(p.values.T), stats.gaussian_kde(q.values.T)
        pa, qa = kp.integrate_box_1d(r.lower[0], r.upper[0]), kq.integrate_box_1d(r.lower[0], r.upper[0])
        numerator = integrate.quad(lambda x: float(kp([x])[0] * (kp.logpdf([x])[0] - kq.logpdf([x])[0])),
                                   r.lower[0], r.upper[0], epsabs=1e-10)[0]
        self.assertAlmostEqual(result["loss"], numerator / pa + np.log(qa / pa), places=8)
        self.assertAlmostEqual(result["region_diagnostics"]["q_mass"], qa, places=8)

    def test_tiny_prior_mass_uses_log_normalization(self):
        result = central_truncated_kl(scalar_measure(stats.norm()), scalar_measure(stats.norm(50, 1)))
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertLess(result["region_diagnostics"]["log_q_mass"], -1000)

    def test_failure_is_not_silently_published(self):
        p, q = scalar_measure(stats.norm()), scalar_measure(stats.norm(2, .7))
        with self.assertRaisesRegex(NumericalFailure, "did not converge") as caught:
            central_truncated_kl(p, q, orders=(2, 3), tolerance=1e-14)
        self.assertEqual(len(caught.exception.context["integration_history"]), 2)
        kernel = Measure(p.values, p.weights, log_kernel=lambda z: stats.norm.logpdf(z[:, 0]))
        with self.assertRaises(DefinitionError):
            central_truncated_kl(kernel, q)
        with self.assertRaisesRegex(NumericalFailure, "positive-volume"):
            central_truncated_kl(Measure(np.ones((10, 2)), np.ones(10)), Measure(np.ones((10, 2)), np.ones(10)))

    def test_configuration_guards(self):
        for settings in [{"preset": "clarke_kl_v2", "region": "central"},
                         {"preset": "clarke_central_tkl_v2", "region": "hdr"},
                         {"preset": "clarke_central_tkl_v2", "alpha": 1.},
                         {"preset": "clarke_central_tkl_v2", "central_tkl_orders": [32, 16]},
                         {"preset": "clarke_central_tkl_v2", "central_tkl_tolerance": 0},
                         {"preset": "clarke_tkl_v2", "central_tkl_orders": [16, 32]}]:
            with self.subTest(settings=settings), self.assertRaises(DefinitionError):
                resolve_method(settings)
        spec = load_spec(ROOT / "configs/clarke_central_smoke.yaml")
        spec["methods"] = [{"preset": "clarke_tkl_v2", "region": {"wrong_target": "central"}}]
        with self.assertRaises(DefinitionError):
            resolved_tasks(spec)

    def test_improper_exclusions_and_selected_diagnostics(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [-1, 0, 1, 2]})
        prior = {"family": "nig", "mu0": .5, "tau0_sq": .2, "kappa": 6., "eta": 5.}
        target = TargetSpec("mu", (Coordinate("mu"),))
        method = resolve_method({"preset": "clarke_central_tkl_v2", "posterior_draws": 128})
        result, rows, _, _ = run_method(model, data, prior, {"family": "jeffreys"}, target, method, [0, 1, 3], {}, 123)
        self.assertEqual(result["status"], "matched", result)
        self.assertEqual(result["candidate_counts"]["excluded_improper"], 9)
        self.assertEqual(rows[0]["status"], "no_admissible_candidate")
        self.assertEqual(result["selected_region_diagnostics"], rows[-1]["region_diagnostics"])
        self.assertAlmostEqual(result["selected_region_diagnostics"]["p_mass"], .95, places=7)

    def test_six_task_workflow(self):
        spec = load_spec(ROOT / "configs/clarke_central_smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = directory
            plan = prepare(spec)
            self.assertEqual(plan["n_pending"], 6)
            for task in range(6):
                result = run_task(plan["manifest"], task)
                self.assertEqual(result["status"], "matched")
            self.assertEqual(merge(directory)["status"], "pass")
            results = [json.loads(p.read_text()) for p in Path(directory).glob("leaves/*/result.json")]
            central = [r for r in results if r["method_id"] == "clarke_central_tkl_v2"]
            self.assertEqual(len(central), 2)
            self.assertTrue(all(r["selected_region_diagnostics"]["converged"] for r in central))


if __name__ == "__main__":
    unittest.main(verbosity=2)
