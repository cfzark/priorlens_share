"""Reference, loss, sampler and matching-direction contracts for risk methods."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from prioreffect_v2.core import Coordinate, TargetSpec, Measure, DefinitionError, NumericalFailure
from prioreffect_v2.models import BuiltinModel
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.methods import resolve_method, risk_matching
from prioreffect_v2.losses import posterior_risk
from prioreffect_v2.risk_components import resolve_reference, RiskSampler, validate_risk_settings
from prioreffect_v2.experiments import load_spec, resolved_tasks
from pathlib import Path


class RiskComponents(unittest.TestCase):
    def setUp(self):
        self.model = BuiltinModel("normal_known")
        self.data = self.model.validate_data({"y": [-1., 0., 1., 2.]})
        self.base = {"family": "normal", "mean": -2., "sd": 1.}
        self.info = {"family": "normal", "mean": 1., "sd": .5}
        self.target = TargetSpec("mu", (Coordinate("mu"),))

    def method(self, name="reimherr", **extra):
        preset = "reimherr_project_bootstrap_v2" if name == "reimherr" else "wiesenfarth_project_plugin_v2"
        return resolve_method({"preset": preset, "risk_reps": 2, "posterior_draws": 128, **extra})

    def test_defaults(self):
        r, w = self.method(), self.method("wiesenfarth")
        self.assertEqual((r["sampling"], r["risk"]), ("bootstrap", "posterior_mse"))
        self.assertEqual((w["sampling"], w["risk"]), ("likelihood", "squared_mean"))
        self.assertEqual(r["reference_rule"], "baseline_posterior_mean")

    def test_four_combinations_keep_matching_direction(self):
        for family in ("reimherr", "wiesenfarth"):
            for sampling in ("bootstrap", "likelihood"):
                with self.subTest(family=family, sampling=sampling):
                    method = self.method(family, sampling=sampling, theta_star=[.5], target_reference=[.5])
                    provider = PosteriorProvider(self.model)
                    with patch.object(self.model, "subset", wraps=self.model.subset) as subset, patch.object(self.model, "simulate", wraps=self.model.simulate) as simulate, patch.object(provider, "fit", wraps=provider.fit) as fit:
                        result, curves, _ = risk_matching(self.model, self.data, self.info, self.base, self.target, method, [-1, 0, 1], provider, 12)
                    counts = [len(c.args[0]["y"]) for c in fit.call_args_list]
                    self.assertEqual(counts, [4, 4, 3, 3, 4, 4, 5, 5] if family == "reimherr" else [4, 4, 5, 5, 4, 4, 3, 3])
                    self.assertEqual([c.args[1] for c in fit.call_args_list], ([self.info]*2+[self.base]*6) if family == "reimherr" else ([self.base]*2+[self.info]*6))
                    self.assertEqual(simulate.call_count, 6 if sampling == "likelihood" else 0)
                    self.assertEqual(subset.call_count, 6 if sampling == "bootstrap" else 0)
                    self.assertEqual(result["risk_components"]["sampling"], sampling)
                    self.assertEqual(len(curves), 3)

    def test_reference_baseline_not_observed_mean(self):
        theta, reference, audit = resolve_reference(self.model, self.data, self.base, self.target, self.method(), PosteriorProvider(self.model), 4)
        self.assertIsNone(theta)  # Bootstrap never needs a likelihood parameter.
        self.assertAlmostEqual(reference[0], 0.)
        self.assertNotEqual(reference[0], np.mean(self.data["y"]))
        self.assertEqual(audit["target_reference_source"], "baseline_target_posterior_mean")

    def test_likelihood_plugin_uses_baseline_parameter_mean(self):
        theta, reference, _ = resolve_reference(self.model, self.data, self.base, self.target, self.method("wiesenfarth"), PosteriorProvider(self.model), 4)
        np.testing.assert_array_equal(theta, reference)
        self.assertAlmostEqual(theta[0], 0.)

    def test_explicit_theta_does_not_override_baseline_reference(self):
        theta, reference, _ = resolve_reference(self.model, self.data, self.base, self.target, self.method("wiesenfarth", theta_star=[8.]), PosteriorProvider(self.model), 4)
        self.assertEqual(theta[0], 8.)
        self.assertAlmostEqual(reference[0], 0.)

    def test_explicit_reference_without_theta_for_multivariate_bootstrap(self):
        model = BuiltinModel("nig")
        target = TargetSpec("mu", (Coordinate("mu"),))
        method = self.method(target_reference=[0.])
        with patch.object(PosteriorProvider, "fit", side_effect=AssertionError("unneeded reference fit")):
            theta, reference, _ = resolve_reference(model, self.data, {}, target, method, PosteriorProvider(model), 4)
        self.assertIsNone(theta)
        np.testing.assert_array_equal(reference, [0.])

    def test_nonlinear_reference_is_mean_of_transform(self):
        model = BuiltinModel("nig")
        target = TargetSpec("joint", (Coordinate("mu", scale=10), Coordinate("sigma2", "log")))
        base = {"family": "nig", "mu0": .3, "tau0_sq": 1., "kappa": 4., "eta": 3.}
        theta, reference, _ = resolve_reference(model, self.data, base, target, self.method("wiesenfarth"), PosteriorProvider(model), 4)
        self.assertAlmostEqual(theta[0]/10, reference[0])
        self.assertGreater(np.log(theta[1])-reference[1], .01)
        # Joint reference does not depend on arbitrary joint Monte Carlo samples.
        _, other, _ = resolve_reference(model, self.data, base, target, self.method("wiesenfarth", posterior_draws=32), PosteriorProvider(model), 8)
        np.testing.assert_array_equal(reference, other)

    def test_binomial_parameter_reference_not_predictive_count(self):
        model = BuiltinModel("beta_binomial", trials=20)
        data = model.validate_data({"y": [2, 4]})
        theta, reference, _ = resolve_reference(model, data, {"family": "beta", "a": 2., "b": 3.}, TargetSpec("p", (Coordinate("p"),)), self.method("wiesenfarth"), PosteriorProvider(model), 4)
        self.assertAlmostEqual(reference[0], 8/45)
        self.assertAlmostEqual(theta[0], 8/45)
        self.assertNotEqual(reference[0], 20*8/45)

    def test_legacy_reference_and_loss_override(self):
        method = self.method("wiesenfarth", reference_rule="legacy_plugin", risk="posterior_mse")
        theta, reference, audit = resolve_reference(self.model, self.data, self.base, self.target, method, PosteriorProvider(self.model), 4)
        np.testing.assert_array_equal(theta, [.5])
        np.testing.assert_array_equal(reference, [.5])
        self.assertEqual(audit["theta_star_source"], "legacy_observed_mle")

    def test_1d_and_2d_loss_identities(self):
        for mean, covariance, reference in (([2.], [[3.]], [1.]), ([2., 3.], [[3., 1.], [1., 4.]], [1., 1.])):
            measure = Measure(np.asarray(mean)[None, :], np.ones(1), mean=np.asarray(mean), covariance=np.asarray(covariance))
            bias = np.sum((np.asarray(mean)-reference)**2)
            self.assertEqual(posterior_risk(measure, reference, "squared_mean"), bias)
            self.assertEqual(posterior_risk(measure, reference, "posterior_mse"), bias+np.trace(covariance))

    def test_all_loss_types_reject_nonfinite(self):
        measure = Measure([[0.]], [1.], mean=np.array([np.inf]), covariance=np.array([[np.inf]]))
        for metric in ("squared_mean", "posterior_mse", "posterior_variance"):
            with self.assertRaises(NumericalFailure):
                posterior_risk(measure, [0.], metric)

    def test_bootstrap_preserves_design_pairs(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": [1., 2., 3.], "s": [4., 5., 6.]})
        out = RiskSampler(model, data, "bootstrap").draw(20, np.random.default_rng(6))
        self.assertTrue(set(zip(out["y"], out["s"])).issubset(set(zip(data["y"], data["s"]))))
        self.assertTrue(all(i.startswith("aux_") for i in out["unit_ids"]))

    def test_invalid_settings_and_focal_rejected(self):
        for extra in ({"sampling": "ppd"}, {"reference_rule": "ppd_mean"}, {"risk": "mse"}, {"theta_star": [np.nan]}, {"target_reference": 1.}):
            with self.assertRaises(DefinitionError):
                self.method(**extra)
        with self.assertRaises(DefinitionError):
            resolve_method({"preset": "cp_source_faithful", "sampling": "bootstrap"})
        for method in (self.method(theta_star=[1., 2.]), self.method(target_reference=[1., 2.])):
            with self.assertRaises(DefinitionError):
                validate_risk_settings(method, self.model, self.target)
        with self.assertRaises(DefinitionError):
            validate_risk_settings(self.method(), BuiltinModel("eight_schools"), TargetSpec("focal", (Coordinate("theta[1]"),), "school_1"))
        with self.assertRaises(DefinitionError):
            validate_risk_settings(self.method(theta_star=[0., -1.]), BuiltinModel("nig"))

    def test_planner_accepts_auto_multivariate_reference(self):
        spec = load_spec(Path(__file__).resolve().parents[1]/"configs/normal_smoke.yaml")
        spec.update(model={"name": "nig"}, priors={"interest": {"family": "nig", "mu0": 0., "tau0_sq": 1., "kappa": 4., "eta": 3.}, "baseline": {"family": "jeffreys"}}, prior_grid=[{"prior_id": "one"}], methods=[{"preset": "wiesenfarth_project_plugin_v2"}], candidates={"values": [0, 1], "meaning": "informative_size_reduction", "unit": "observation"})
        self.assertEqual(len(resolved_tasks(spec)), 1)
        old = copy.deepcopy(spec)
        old["methods"][0]["reference_rule"] = "legacy_plugin"
        with self.assertRaises(DefinitionError):
            resolved_tasks(old)


if __name__ == "__main__":
    unittest.main()
