"""Paper-definition gates, analytic oracles, and hierarchical covariance checks."""
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prioreffect_v2.core import Coordinate, TargetSpec, DefinitionError, NumericalFailure
from prioreffect_v2.models import BuiltinModel, SCHOOLS_Y, SCHOOLS_S
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.methods import resolve_method, run_method
from prioreffect_v2.experiments import load_spec, resolved_tasks
from prioreffect_v2.morita_versions import (epsilon_prior, run_version, validate_version,
    select_curve, precision_logdet, school_prior_moments, school_grid_moments)

ROOT = Path(__file__).resolve().parents[1]
SCHOOL_PRIOR = {"family": "schools", "mu_mean": 2., "mu_sd": 5.,
                "tau_kind": "lognormal", "tau_scale": 6., "tau_log_sd": .3}


def schools_target(kind="log", joint=True):
    coordinates = (Coordinate("mu", scale=10), Coordinate("tau", kind, scale=100 if kind == "square" else 1, reference=10))
    return TargetSpec("hyper", coordinates if joint else coordinates[1:])


class MoritaVersions(unittest.TestCase):
    def test_three_classic_analytic_and_fractional_ess(self):
        cases = [("normal_known", {"sigma": 2.}, {"family": "normal", "mean": 9., "sd": 1.}, "mu", 4.),
                 ("poisson_gamma", {}, {"family": "gamma", "shape": 7., "rate": 3.}, "lambda", 3.),
                 ("beta_binomial", {"trials": 4}, {"family": "beta", "a": 3., "b": 7.}, "p", 2.5)]
        for name, options, prior, variable, limit in cases:
            model = BuiltinModel(name, **options)
            data = model.validate_data({"y": [0, 1]})
            target = TargetSpec(variable, (Coordinate(variable, scale=3),))
            method = resolve_method({"preset": "morita_2008_conjugate_1d_v2", "epsilon_c": 10000})
            result, rows, _, accuracy = run_method(model, data, prior, {}, target, method, [0, 2, 5, 10], {}, 32)
            self.assertAlmostEqual(result["epss"], limit * .9999, places=11)
            self.assertTrue(result["match_bracketed"])
            self.assertFalse(result["observed_responses_used"])
            self.assertEqual(accuracy, [])

    def test_nig_legacy_formula_golden(self):
        spec = importlib.util.spec_from_file_location("nig_morita_legacy", ROOT / "scripts/run_nig2d_morita_heatmap.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [-1, 0, 1]})
        prior = {"family": "nig", "mu0": 8., "tau0_sq": .2, "kappa": 6., "eta": 5.}
        method = resolve_method({"preset": "morita_nig_joint_trace_project_v2", "epsilon_c": 10000})
        for transform, name in [("identity", "mu_sigma2"), ("log", "mu_log_sigma2")]:
            target = TargetSpec("joint", (Coordinate("mu"), Coordinate("sigma2", transform)))
            result, rows, _ = run_version(model, data, prior, target, method, [0, 5, 20], PosteriorProvider(model), 1)
            legacy = module.morita_components(1., .25, .2, name, 10000)
            self.assertAlmostEqual(result["epss_unrestricted"], legacy["ess_unrestricted"], places=12)
            for row in rows:
                self.assertAlmostEqual(row["candidate_curvature"], legacy["D_epsilon_0_trace"] + row["m"] * legacy["Dq_slope_trace"], places=12)

    def test_school_epsilon_preserves_target_means_and_inflates_variances(self):
        model = BuiltinModel("eight_schools")
        for kind in ["identity", "log", "square"]:
            for joint in [False, True]:
                target = schools_target(kind, joint)
                q = epsilon_prior(model, SCHOOL_PRIOR, target, 20)
                mean, cov = school_prior_moments(SCHOOL_PRIOR, target)
                qm, qc = school_prior_moments(q, target)
                np.testing.assert_allclose(qm, mean, atol=1e-12)
                np.testing.assert_allclose(qc, 20 * cov, rtol=1e-12)
                if not joint:
                    self.assertEqual(q["mu_sd"], SCHOOL_PRIOR["mu_sd"])

    def test_school_full_covariance_not_diagonal_only(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        post = model.optimized(data, SCHOOL_PRIOR)
        target = schools_target()
        mean, covariance = school_grid_moments(post, target)
        self.assertGreater(abs(covariance[0, 1]), 1e-4)
        samples = target.values(post.sample(150000, np.random.default_rng(17)))
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=.006)
        np.testing.assert_allclose(np.cov(samples.T), covariance, atol=.002)
        self.assertGreater(precision_logdet(covariance), precision_logdet(np.diag(np.diag(covariance))))

    def test_school_mc_ignores_observed_y_and_uses_plugin_not_hyperprior_draws(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        method = resolve_method({"preset": "morita_2012_case2_schools_mc_v2", "epsilon_c": 10,
            "morita_design": "new_schools_empirical_s", "prior_predictive_reps": 2})
        target = schools_target()
        settings = {"backend": "optimized", "school_grid_size": 201}
        a = run_method(model, data, SCHOOL_PRIOR, {}, target, method, [0, 1, 3], settings, 31)
        changed = {**data, "y": [999.] * 8}
        b = run_method(model, changed, SCHOOL_PRIOR, {}, target, method, [0, 1, 3], settings, 31)
        self.assertEqual(a[:3], b[:3])
        self.assertEqual(a[0]["status"], "matched")
        np.testing.assert_allclose(a[0]["hyperparameter_plugin_natural"], [2., 6.])
        self.assertTrue(any(r["engine"] == "morita_covariance_refinement" for r in a[3]))
        self.assertFalse(a[0]["epsilon_sensitivity_verified"])

    def test_average_determinants_not_determinant_of_average(self):
        from prioreffect_v2.morita_versions import schools_curve
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        target = schools_target()
        q = epsilon_prior(model, SCHOOL_PRIOR, target, 10)
        # Determinants are 1 and 1/8; desired average is 9/16, not 1/3.
        with patch("prioreffect_v2.morita_versions.school_checked_moments", side_effect=[
                (np.zeros(2), np.eye(2)), (np.zeros(2), np.diag([2., 4.]))]):
            _, rows, _ = schools_curve(model, data, SCHOOL_PRIOR, q, target, {"prior_predictive_reps": 2}, [1], PosteriorProvider(model), 42)
        self.assertAlmostEqual(np.exp(rows[0]["candidate_log_mean_precision_determinant"]), 9 / 16)

    def test_finite_grid_boundary_does_not_prove_infinite_ess(self):
        rows = [{"m": 0, "x": 1.}, {"m": 2, "x": 2.}]
        result = select_curve(rows, 10., "x")
        self.assertEqual(result["epss"], 2.)
        self.assertFalse(result["match_bracketed"])
        self.assertTrue(result["finite_grid_only"])

    def test_log_interpolation_is_in_information_not_log_information(self):
        rows = [{"m": 0, "x": np.log(1)}, {"m": 4, "x": np.log(9)}]
        result = select_curve(rows, np.log(3), "x", log_values=True)
        self.assertAlmostEqual(result["epss"], 1.)

    def test_log_information_underflow_does_not_create_false_roots(self):
        rows = [{"m": 0, "x": -1000.}, {"m": 1, "x": -999.}, {"m": 2, "x": 1000.}]
        result = select_curve(rows, -1001., "x", log_values=True)
        self.assertFalse(result["match_bracketed"])
        self.assertEqual(result["epss_grid"], 0)

    def test_gates_no_silent_source_relabeling(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        method = {"preset": "morita_2012_case2_schools_mc_v2", "epsilon_c": 10,
                  "morita_design": "new_schools_empirical_s"}
        for changes in [{"epsilon_c": 1}, {"epsilon_c": None}, {"epsilon_prior": SCHOOL_PRIOR}, {"morita_design": "fixed_K"}]:
            with self.assertRaises(DefinitionError):
                validate_version(model, data, SCHOOL_PRIOR, schools_target(), {**method, **changes})
        with self.assertRaises(DefinitionError):
            validate_version(model, data, {**SCHOOL_PRIOR, "tau_kind": "half_cauchy"}, schools_target(), method)
        with self.assertRaises(NumericalFailure):
            precision_logdet(np.diag([1., -1.]))
        with self.assertRaises(DefinitionError):
            resolve_method({"preset": "morita_curvature_v2", "epsilon_c": 10})

    def test_versioned_planner_analytic_needs_no_smoke_mc_defaults(self):
        spec = load_spec(ROOT / "configs/normal_smoke.yaml")
        spec["run_mode"] = "production"
        spec["methods"] = [{"preset": "morita_2008_conjugate_1d_v2", "epsilon_c": 10000}]
        spec["candidates"]["meaning"] = "pseudo_sample_size"
        tasks = resolved_tasks(spec)
        self.assertEqual(len(tasks), 3)
        self.assertNotIn("prior_predictive_reps", tasks[0]["method"])

    def test_2012_fixed_K_example_is_not_new_school_count(self):
        # Section 3.2: Var(mu)=.01, gamma^2=1, K=10 => no finite m.
        K, gamma2, var_mu, sigma2 = 10, 1., .01, 1.
        self.assertLess(K * var_mu - gamma2, 0)
        # Change design: m=20, increase K; K=105 and TOTAL observations=2100.
        within_school_n = 20
        required_K = (gamma2 + sigma2 / within_school_n) / var_mu
        self.assertAlmostEqual(required_K, 105)
        self.assertAlmostEqual(required_K * within_school_n, 2100)


if __name__ == "__main__":
    unittest.main()
