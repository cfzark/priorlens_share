"""Density ranks, posterior mass, reciprocity and Lupus integration contracts."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
from scipy.stats import qmc

from prioreffect_v2.core import Coordinate, TargetSpec, Measure, DefinitionError, NumericalFailure
from prioreffect_v2.density_hdr import density_hdr_region
from prioreffect_v2.experiments import resolved_tasks
from prioreffect_v2.losses import Coverage, coverage_terms
from prioreffect_v2.models import BuiltinModel, lupus_data
from prioreffect_v2.quadrature import WeightedPosterior

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lupus_pdf_hdr_smoke import smoke_spec


class PDFHDRTests(unittest.TestCase):
    def test_pdf_rank_not_importance_weight_and_all_boundary_ties(self):
        values = np.array([[0.], [1.], [2.], [3.]])
        # Highest posterior atom weight is NOT highest density.
        logpdf = lambda z: np.array([4., 3., 3., 0.])[z[:, 0].astype(int)]
        p = Measure(values, [.1, .2, .6, .1], logpdf=logpdf)
        region = density_hdr_region(p, .5)
        self.assertEqual(region.log_cutoff, 3.)
        self.assertAlmostEqual(region.own_mass, .9)
        self.assertAlmostEqual(region.strict_mass, .1)
        self.assertAlmostEqual(region.boundary_mass, .8)
        self.assertAlmostEqual(region.mass(p), .9)

    def test_kernel_constant_cancels_and_pdf_log_ranking_agree(self):
        x = np.linspace(-3, 3, 101)[:, None]
        lp = lambda z: -.5 * z[:, 0] ** 2
        p = Measure(x, np.exp(lp(x)), logpdf=lp)
        q = Measure(x, p.weights, log_kernel=lambda z: lp(z) + 10000.)
        a, b = density_hdr_region(p), density_hdr_region(q)
        self.assertAlmostEqual(b.log_cutoff - a.log_cutoff, 10000.)
        self.assertAlmostEqual(a.mass(q), b.mass(p))
        order = np.argsort(-np.exp(lp(x)))
        cut = np.exp(lp(x))[order[np.searchsorted(np.cumsum(p.weights[order]), .95)]]
        self.assertAlmostEqual(a.own_mass, float(p.weights @ (np.exp(lp(x)) >= cut)))

    def test_gaussian_hdr_ellipse_and_reciprocal_analytic_mass(self):
        z = stats.norm.ppf(qmc.Sobol(2, scramble=True, seed=123).random_base2(15))
        cov = np.array([[1., .8], [.8, 1.]])
        chol = np.linalg.cholesky(cov)
        shift = np.array([.6, -.3])
        p = Measure(z @ chol.T, np.ones(len(z)), logpdf=stats.multivariate_normal([0., 0.], cov).logpdf)
        q = Measure(z @ chol.T + shift, np.ones(len(z)), logpdf=stats.multivariate_normal(shift, cov).logpdf)
        rp = density_hdr_region(p)
        radius2 = 2 * (float(p.logpdf(np.zeros((1, 2)))) - rp.log_cutoff)
        self.assertAlmostEqual(radius2, stats.chi2.ppf(.95, 2), delta=.02)
        reference = stats.ncx2.cdf(radius2, 2, shift @ np.linalg.solve(cov, shift))
        result = Coverage(region="pdf_hdr")(p, q)
        self.assertAlmostEqual(result["cp1"][0], reference, delta=.002)
        self.assertAlmostEqual(result["cp2"][0], reference, delta=.002)
        self.assertLess(abs(rp.own_mass - .95), 1 / len(z))

    def test_reciprocal_direction_and_loss_options(self):
        x = np.linspace(-5, 5, 1001)[:, None]
        p = Measure(x, stats.norm.pdf(x[:, 0]), logpdf=lambda z: stats.norm.logpdf(z[:, 0]))
        q = Measure(x, stats.norm.pdf(x[:, 0], 1., .6), logpdf=lambda z: stats.norm.logpdf(z[:, 0], 1., .6))
        rp, rq = density_hdr_region(p), density_hdr_region(q)
        for family, gap, cap in [("historical", 1., False), ("direct", 0., True)]:
            result = Coverage(region="pdf_hdr", loss_family=family, gap_weight=gap, cap_nominal=cap)(p, q)
            self.assertEqual(result["cp1"], [rq.mass(p)])
            self.assertEqual(result["cp2"], [rp.mass(q)])
            expected = coverage_terms(rq.mass(p), rp.mass(q), .95, family, gap, cap)
            self.assertAlmostEqual(result["loss"], expected["loss"])

    def test_cache_and_no_kde(self):
        x = np.linspace(-4, 4, 401)[:, None]
        p = Measure(x, stats.norm.pdf(x[:, 0]), logpdf=lambda z: stats.norm.logpdf(z[:, 0]))
        compare = Coverage(region="pdf_hdr", cache_prepared=True)
        with patch("scipy.stats.gaussian_kde", side_effect=AssertionError("No KDE allowed")):
            first = compare(p, p)
            region = compare._prepared[id(p)][1][0]
            self.assertEqual(first, compare(p, p))
            self.assertIs(region, compare._prepared[id(p)][1][0])
        self.assertEqual(first, Coverage(region="pdf_hdr")(p, p))

    def test_invalid_density_and_zero_weight_support(self):
        x = np.array([[0.], [1.]])
        with self.assertRaises(DefinitionError):
            density_hdr_region(Measure(x, [1, 1]))
        for lp in (lambda z: np.full(len(z), np.nan), lambda z: np.zeros((len(z), 1)),
                   lambda z: np.full(len(z), np.inf), lambda z: np.full(len(z), -np.inf)):
            with self.assertRaises(NumericalFailure):
                density_hdr_region(Measure(x, [1, 1], logpdf=lp))
        p = Measure(x, [1, 0], logpdf=lambda z: np.where(z[:, 0] == 0, 0., -np.inf))
        self.assertEqual(density_hdr_region(p).own_mass, 1.)
        self.assertEqual(density_hdr_region(p).mass(Measure([[1]], [1])), 0.)
        with self.assertRaises(DefinitionError):
            Coverage(region="pdf_hdr", quantile_rule="numpy_linear")(p, p)

    def test_lupus_joint_kernel_and_affine_coordinates(self):
        model = BuiltinModel("lupus_logistic")
        data = model.validate_data(lupus_data())
        prior = dict(family="student_t", df=7, loc=[-1., 3.], scale=[10., 2.5])
        x = np.random.default_rng(1).normal(size=(100, 2)) + [-1, 3]
        post = WeightedPosterior(model, data, prior, x, np.ones(100), dict(engine="test"))
        raw = post.target(TargetSpec("raw", (Coordinate("beta0"), Coordinate("beta1"))))
        scaled = post.target(TargetSpec("scaled", (Coordinate("beta1", scale=2), Coordinate("beta0", scale=3))))
        np.testing.assert_allclose(raw.log_kernel(x), model.loglik(x, data) + model.logprior(x, prior))
        np.testing.assert_allclose(scaled.log_kernel(scaled.values), raw.log_kernel(x) + np.log(6))
        self.assertEqual(density_hdr_region(raw).own_mass, density_hdr_region(scaled).own_mass)
        marginal = post.target(TargetSpec("beta0", (Coordinate("beta0"),)))
        with self.assertRaisesRegex(DefinitionError, "marginal"):
            density_hdr_region(marginal)

    def test_standard_planner_and_paired_seeds(self):
        spec = smoke_spec("unused_pdf_hdr_test")
        tasks = resolved_tasks(spec)
        self.assertEqual(len(tasks), 6)
        for prior in ("near", "far"):
            self.assertEqual(len({t["seed"] for t in tasks if t["prior_id"] == prior}), 1)
        self.assertEqual(spec["inference"]["sobol_power"], 15)
        self.assertEqual(spec["inference"]["sobol_max_power"], 15)
        invalid = copy.deepcopy(spec)
        invalid["targets"][0]["coordinates"] = [dict(variable="beta0")]
        with self.assertRaisesRegex(DefinitionError, "joint"):
            resolved_tasks(invalid)
        # Joint HDR + central marginals is an explicit valid composition.
        spec["targets"].append(dict(id="beta0", coordinates=[dict(variable="beta0")]))
        for method in spec["methods"]:
            method["region"] = dict(beta_joint=method["region"], beta0="central")
        self.assertEqual(len(resolved_tasks(spec)), 12)


if __name__ == "__main__":
    unittest.main()
