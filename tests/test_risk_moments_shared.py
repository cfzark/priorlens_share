"""Mathematical gates, deterministic moments and shared-data RNG contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import digamma, polygamma

from prioreffect_v2.core import Coordinate, TargetSpec, UndefinedMoment, UncertifiedMoment, DefinitionError, digest
from prioreffect_v2.models import BuiltinModel, SCHOOLS_Y, SCHOOLS_S
from prioreffect_v2.moments import check_target_moments, marginal_summary
from prioreffect_v2.methods import resolve_method, run_method
from prioreffect_v2.risk_components import RiskSampler, RiskDatasetBank
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task, NumericalErrorForCLI

ROOT = Path(__file__).resolve().parents[1]
BASE = {"family": "jeffreys"}
INFO = {"family": "nig", "mu0": .3, "tau0_sq": .5, "kappa": 4., "eta": 3.}


class RiskMomentsShared(unittest.TestCase):
    def setUp(self):
        self.nig = BuiltinModel("nig")
        self.joint = TargetSpec("joint", (Coordinate("mu"), Coordinate("sigma2", "log")))

    def data(self, n):
        return self.nig.validate_data({"y": np.linspace(-1, 1, n).tolist()})

    def test_jeffreys_mean_variance_threshold_matrix(self):
        # Includes sigma^4 as a transformed target, not just raw dimensions.
        requirements = [(Coordinate("mu"), 1, 3), (Coordinate("mu"), 2, 4),
            (Coordinate("sigma2"), 1, 4), (Coordinate("sigma2"), 2, 6),
            (Coordinate("sigma2", "log"), 1, 2), (Coordinate("sigma2", "log"), 2, 2),
            (Coordinate("sigma2", "square"), 1, 6), (Coordinate("sigma2", "square"), 2, 10)]
        for c, order, minimum in requirements:
            for n in range(2, 11):
                with self.subTest(coordinate=c, order=order, n=n):
                    self.nig.check_posterior_admissibility(self.data(n), BASE)
                    target = TargetSpec("t", (c,))
                    if n < minimum:
                        with self.assertRaises(UndefinedMoment):
                            check_target_moments(self.nig, self.data(n), BASE, target, order)
                    else:
                        self.assertTrue(check_target_moments(self.nig, self.data(n), BASE, target, order)["certified"])

    def test_proper_nig_prior_does_not_imply_finite_moments(self):
        data = self.nig.validate_data({"y": []})
        prior = {**INFO, "kappa": .4}
        self.nig.optimized(data, prior)  # Proper prior/posterior.
        with self.assertRaises(UndefinedMoment):
            check_target_moments(self.nig, data, prior, TargetSpec("mu", (Coordinate("mu"),)), 1)

    def test_schools_tau_tail_threshold_and_focal_moments(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": [1.], "s": [2.]})
        prior = {"family": "schools", "mu_mean": 0., "mu_sd": 10., "tau_kind": "half_cauchy", "tau_scale": 10.}
        self.assertTrue(check_target_moments(model, data, prior, TargetSpec("tau", (Coordinate("tau"),)), 1)["certified"])
        with self.assertRaises(UndefinedMoment):
            check_target_moments(model, data, prior, TargetSpec("tau", (Coordinate("tau"),)), 2)
        with self.assertRaises(UndefinedMoment):
            check_target_moments(model, data, prior, TargetSpec("tau2", (Coordinate("tau", "square"),)), 1)
        for c in (Coordinate("mu"), Coordinate("tau", "log"), Coordinate("theta[1]")):
            target = TargetSpec("t", (c,), "school_1" if c.variable.startswith("theta") else None)
            self.assertTrue(check_target_moments(model, data, prior, target, 2)["certified"])
        prior = {**prior, "tau_kind": "half_t", "tau_df": 4.}
        with self.assertRaises(UndefinedMoment):
            check_target_moments(model, data, prior, TargetSpec("tau2", (Coordinate("tau", "square"),)), 2, sample_size=0)

    def test_lupus_low_df_is_uncertified_not_claimed_infinite(self):
        model = BuiltinModel("lupus_logistic")
        prior = {"family": "student_t", "loc": [0., 0.], "scale": [1., 1.], "df": 1.}
        data = {"y": [0, 1], "x": [0, 1]}
        with self.assertRaises(UncertifiedMoment):
            check_target_moments(model, data, prior, TargetSpec("b", (Coordinate("beta0"),)), 2)
        self.assertTrue(check_target_moments(model, data, prior, TargetSpec("p", (Coordinate("p_x_0"),)), 2)["certified"])

    def test_external_adapter_must_certify_moments(self):
        model = SimpleNamespace(name="custom")
        with self.assertRaises(UncertifiedMoment):
            check_target_moments(model, {}, {}, self.joint, 2)
        model.check_target_moments = lambda *args: {"certified": True, "proof": "test hook"}
        self.assertTrue(check_target_moments(model, {}, {}, self.joint, 2)["certified"])

    def test_joint_risk_rejects_undefined_moment_even_in_legacy_mode(self):
        for mode in ("marginal", "legacy_joint"):
            method = resolve_method({"preset": "reimherr_project_bootstrap_v2", "risk_moments": mode, "target_reference": [0., 0.], "risk_reps": 2})
            for backend in ("optimized", "importance"):
                with patch("prioreffect_v2.inference.importance_posterior", side_effect=AssertionError("Reject before IS fit")):
                    result = run_method(self.nig, self.data(3), INFO, BASE, self.joint, method, [0], {"backend": backend}, 123)[0]
                self.assertEqual(result["status"], "undefined_moment")
                self.assertIsNone(result["epss"])

    def test_jones_checks_second_moment_but_cp_does_not_require_it(self):
        for family, expected in (("jones", "undefined_moment"), ("cp", "matched")):
            method = resolve_method({"preset": family+"_subsample_deletion_v2", "paths": 1, "region": "central", "posterior_draws": 64})
            result = run_method(self.nig, self.data(3), INFO, BASE, self.joint, method, [0], {}, 123)[0]
            self.assertEqual(result["status"], expected)

    def test_squared_mean_does_not_require_variance(self):
        method = resolve_method({"preset": "wiesenfarth_project_plugin_v2", "theta_star": [0., 1.], "target_reference": [0.], "risk_reps": 2})
        target = TargetSpec("mu", (Coordinate("mu"),))
        result = run_method(self.nig, self.data(3), INFO, BASE, target, method, [0], {}, 123)[0]
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["risk_components"]["moment_order_required"], 1)

    def test_nig_joint_moments_analytic_and_seed_draw_invariant(self):
        post = self.nig.optimized(self.data(8), INFO)
        target = TargetSpec("joint", (Coordinate("mu", scale=10), Coordinate("sigma2", "log", scale=2, reference=3)))
        p = post.params
        expected_mean = [p["mu"]/10, (np.log(p["eta"])-digamma(p["kappa"])-np.log(3))/2]
        expected_var = [p["eta"]/(p["kappa"]-1)/p["lambda"]/100, polygamma(1, p["kappa"])/4]
        with patch.object(post, "sample", side_effect=AssertionError("Risk must not sample posterior")):
            a = marginal_summary(post, target, 2, 8, 1)
            b = marginal_summary(post, target, 2, 8192, 999)
        np.testing.assert_allclose(a[0], expected_mean, atol=1e-15)
        np.testing.assert_allclose(a[1], expected_var, atol=1e-15)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_log_and_logit_conjugate_moments_are_analytic(self):
        model = BuiltinModel("beta_binomial")
        post = model.optimized(model.validate_data({"y": [2, 4]}), {"family": "beta", "a": 2., "b": 3.})
        target = TargetSpec("logit", (Coordinate("p", "logit"),))
        mean, var, _ = marginal_summary(post, target, 2, 2)
        a, b = post.rv.args
        self.assertAlmostEqual(mean[0], digamma(a)-digamma(b))
        self.assertAlmostEqual(var[0], polygamma(1, a)+polygamma(1, b))

    def test_schools_joint_moments_do_not_sample_and_match_grid_refinement(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        prior = {"family": "schools", "mu_mean": 0., "mu_sd": 25., "tau_kind": "half_t", "tau_scale": 25., "tau_df": 4.}
        post = model.optimized(data, prior)
        refined = model.optimized(data, prior, {"school_grid_size": 1601})
        for coordinate in (Coordinate("tau", "log", reference=10), Coordinate("tau", "square", scale=100)):
            target = TargetSpec("joint", (Coordinate("mu", scale=10), coordinate))
            with patch.object(post, "sample", side_effect=AssertionError("Risk must not sample posterior")):
                a = marginal_summary(post, target, 2, 8, 1)
                b = marginal_summary(post, target, 2, 8192, 999)
            reference = marginal_summary(refined, target, 2)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_allclose(a[0], reference[0], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(a[1], reference[1], rtol=1e-4, atol=1e-6)

    def test_identical_priors_have_exact_zero_m0_gap_all_combinations(self):
        for family in ("reimherr_project_bootstrap_v2", "wiesenfarth_project_plugin_v2"):
            for sampling in ("bootstrap", "likelihood"):
                method = resolve_method({"preset": family, "sampling": sampling, "risk_reps": 4, "risk_data_seed": 321})
                result, curves, _, _ = run_method(self.nig, self.data(8), INFO, INFO, self.joint, method, [0], {}, 123)
                self.assertEqual(result["status"], "matched")
                self.assertEqual(curves[0]["loss"], 0.)

    def test_shared_data_survives_grid_budget_and_order_changes(self):
        sampler = RiskSampler(self.nig, self.data(8), "likelihood", [0., 1.])
        first, second = RiskDatasetBank(sampler, 12), RiskDatasetBank(sampler, 12)
        a = first.draw(8, 0)
        second.draw(7, 0); second.draw(8, 9)
        self.assertEqual(a, second.draw(8, 0))
        self.assertIs(a, first.draw(8, 0))
        self.assertNotEqual(a["y"], first.draw(9, 0)["y"][:8])  # Explicitly NOT nested.
        changed = RiskDatasetBank(RiskSampler(self.nig, self.data(8), "likelihood", [1., 1.]), 12)
        self.assertNotEqual(first.law, changed.law)

    def test_planner_shares_seed_across_prior_cells_and_methods(self):
        spec = load_spec(ROOT/"configs/risk_components_smoke.yaml")
        spec["prior_grid"] = [{"prior_id": "a", "mean": 0.}, {"prior_id": "b", "mean": 1.}]
        tasks = resolved_tasks(spec)
        seeds = {t["method"]["risk_data_seed"] for t in tasks}
        self.assertEqual(len(seeds), 1)
        self.assertGreater(len({t["seed"] for t in tasks}), 1)
        spec["replication_id"] = 1
        self.assertNotEqual(next(iter(seeds)), resolved_tasks(spec)[0]["method"]["risk_data_seed"])

    def test_shared_datasets_across_method_and_prior_with_common_data_seed(self):
        checksums = []
        for family in ("reimherr_project_bootstrap_v2", "wiesenfarth_project_plugin_v2"):
            for mean in (0., 3.):
                method = resolve_method({"preset": family, "sampling": "bootstrap", "risk_reps": 2, "risk_data_seed": 321})
                result = run_method(self.nig, self.data(8), {**INFO, "mu0": mean}, INFO, self.joint, method, [-1, 0, 1], {}, int(123+mean))[0]
                self.assertEqual(result["status"], "matched")
                checksums.append(result["risk_components"]["data_checksums"])
        self.assertTrue(all(c == checksums[0] for c in checksums))

    def test_planner_rejects_impossible_candidate_moment(self):
        spec = load_spec(ROOT/"configs/risk_components_smoke.yaml")
        spec.update(model={"name": "nig"}, data={"y": [-1, 0, 1, 2]}, priors={"interest": INFO, "baseline": BASE}, prior_grid=[{"prior_id": "one"}], methods=[{"preset": "reimherr_project_bootstrap_v2"}], candidates={"values": [-1, 0], "meaning": "baseline_size_offset", "unit": "observation"})
        with self.assertRaises(UndefinedMoment):
            resolved_tasks(spec)

    def test_plugin_nuisance_mean_checked_even_for_mu_only_risk(self):
        method = resolve_method({"preset": "wiesenfarth_project_plugin_v2", "risk_reps": 2})
        # E[mu] exists at n=3, but automatic likelihood generation needs E[sigma²].
        result = run_method(self.nig, self.data(3), INFO, BASE, TargetSpec("mu", (Coordinate("mu"),)), method, [0], {}, 123)[0]
        self.assertEqual(result["status"], "undefined_moment")

    def test_worker_persists_undefined_moment_without_done(self):
        spec = load_spec(ROOT/"configs/risk_components_smoke.yaml")
        spec["methods"] = spec["methods"][:1]
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = directory
            plan = prepare(spec)
            with patch("prioreffect_v2.methods.risk_matching", side_effect=UndefinedMoment("test certificate failure")):
                with self.assertRaises(NumericalErrorForCLI):
                    run_task(plan["manifest"], 0)
            leaves = list((Path(directory)/"leaves").iterdir())
            self.assertEqual(json.loads((leaves[0]/"result.json").read_text())["status"], "undefined_moment")
            self.assertFalse((leaves[0]/"done.json").exists())

    def test_new_controls_reject_invalid_values_and_wrong_families(self):
        for extra in ({"risk_coupling": "nested"}, {"risk_moments": "sample_anyway"}, {"risk_data_seed": -1}):
            with self.assertRaises(DefinitionError):
                resolve_method({"preset": "reimherr_project_bootstrap_v2", **extra})
        with self.assertRaises(DefinitionError):
            resolve_method({"preset": "cp_source_faithful", "risk_coupling": "shared_by_size"})


if __name__ == "__main__":
    unittest.main()
