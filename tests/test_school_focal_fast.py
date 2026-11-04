"""Focal identity, conditional methods, and reference/accelerator golden tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prioreffect_v2.core import Coordinate, TargetSpec, DefinitionError, NumericalFailure
from prioreffect_v2.models import BuiltinModel, SCHOOLS_Y, SCHOOLS_S
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.focal import FocalAnchor, PROTOCOL
from prioreffect_v2.risk_components import RiskSampler, RiskDatasetBank, resolve_reference
from prioreffect_v2.school_acceleration import batch_posteriors
from prioreffect_v2.methods import resolve_method, run_method, clarke
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task_group, merge
from prioreffect_v2.losses import Coverage
from prioreffect_v2.moments import marginal_summary


INFO = {"family": "schools", "mu_mean": 5., "mu_sd": 10., "tau_kind": "lognormal", "tau_scale": 6., "tau_log_sd": .3}
BASE = {**INFO, "mu_mean": 0., "mu_sd": 25., "tau_log_sd": .7}
REFERENCE = {"backend": "optimized", "posterior_reuse": "none", "school_grid_size": 101}
FAST = {**REFERENCE, "school_engine": "batched", "school_batch_size": 4, "school_cache_by_data": True, "school_cache_size": 32}
ROOT = Path(__file__).resolve().parents[1]


def focal(j=1):
    return TargetSpec(f"school_{j}", (Coordinate(f"theta[{j}]"),), f"school_{j}")


class SchoolFocalFast(unittest.TestCase):
    def setUp(self):
        self.model = BuiltinModel("eight_schools")
        self.data = self.model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})

    def method(self, family="reimherr", sampling="bootstrap"):
        return resolve_method({"preset": "reimherr_project_bootstrap_v2" if family == "reimherr" else "wiesenfarth_project_plugin_v2",
            "focal_protocol": PROTOCOL, "sampling": sampling, "risk_reps": 3,
            "risk_data_seed": 99, "posterior_draws": 64})

    def test_all_eight_anchors_both_samplers_counts_and_no_duplication(self):
        for j in range(1, 9):
            anchor = FocalAnchor(self.model, self.data, f"school_{j}")
            for sampling in ("bootstrap", "likelihood"):
                sampler = RiskSampler(self.model, self.data, sampling, [1., 5.], anchor.unit_id)
                for count in (1, 8, 12):
                    candidate = sampler.draw(count, np.random.default_rng(123))
                    anchor.check(candidate)
                    self.assertEqual(len(candidate["y"]), count)
                    self.model.validate_target(focal(j), candidate)
                    if sampling == "bootstrap":
                        original = {(self.data["y"][i], self.data["s"][i]) for i in anchor.pool}
                        self.assertTrue(set(zip(candidate["y"][1:], candidate["s"][1:])).issubset(original))

    def test_focal_reference_is_baseline_marginal_mean(self):
        for j in (1, 3, 8):
            target = focal(j)
            theta, reference, _ = resolve_reference(self.model, self.data, BASE, target,
                self.method("wiesenfarth", "likelihood"), PosteriorProvider(self.model, REFERENCE), 123)
            post = self.model.optimized(self.data, BASE, REFERENCE)
            self.assertAlmostEqual(reference[0], post.target(target).mean[0], places=12)
            self.assertEqual(len(theta), 2)

    def test_shared_risk_keys_include_focal_identity(self):
        a = RiskDatasetBank(RiskSampler(self.model, self.data, "bootstrap", None, "school_1"), 123)
        b = RiskDatasetBank(RiskSampler(self.model, self.data, "bootstrap", None, "school_2"), 123)
        self.assertNotEqual(a.law, b.law)
        self.assertEqual(a.draw(5, 0)["unit_ids"][0], "school_1")
        with self.assertRaises(DefinitionError):
            a.draw(0, 0)

    def test_batched_grid_matches_scalar_for_all_prior_families(self):
        datasets = [RiskSampler(self.model, self.data, "bootstrap").draw(8, np.random.default_rng(i)) for i in range(5)]
        priors = [INFO, {**INFO, "tau_kind": "half_t", "tau_df": 4.},
                  {**INFO, "tau_kind": "half_cauchy"}, {**INFO, "tau_kind": "bounded_uniform", "tau_scale": 100.}]
        for prior in priors:
            fast = batch_posteriors(self.model, datasets, prior, REFERENCE)
            for data, post in zip(datasets, fast):
                reference = self.model.optimized(data, prior, REFERENCE)
                for field in ("eta", "masses", "mu_cond_mean", "mu_cond_var", "interval_cdf"):
                    np.testing.assert_allclose(getattr(post, field), getattr(reference, field), rtol=2e-13, atol=2e-13)
                self.assertEqual(post.accuracy["n_expansions"], reference.accuracy["n_expansions"])
                a, b = post.sample(64, np.random.default_rng(8)), reference.sample(64, np.random.default_rng(8))
                for name in a:
                    np.testing.assert_allclose(a[name], b[name], rtol=2e-12, atol=2e-12)

    def test_batch_empty_and_anchor_only_datasets(self):
        for count in (0, 1):
            datasets = [self.model.subset(self.data, list(range(count)))] * 2
            posts = batch_posteriors(self.model, datasets, INFO, REFERENCE)
            reference = self.model.optimized(datasets[0], INFO, REFERENCE)
            np.testing.assert_allclose(posts[0].masses, reference.masses, atol=1e-14)

    def test_focal_risk_reference_fast_all_four_combinations(self):
        for family in ("reimherr", "wiesenfarth"):
            for sampling in ("bootstrap", "likelihood"):
                method = self.method(family, sampling)
                a = run_method(self.model, self.data, INFO, BASE, focal(3), method, [-1, 0, 1], REFERENCE, 123)
                b = run_method(self.model, self.data, INFO, BASE, focal(3), method, [-1, 0, 1], FAST, 123)
                self.assertEqual(a[0]["status"], "matched")
                self.assertEqual(a[0]["epss"], b[0]["epss"])
                self.assertEqual(a[0]["risk_components"]["data_checksums"], b[0]["risk_components"]["data_checksums"])
                for x, y in zip(a[1], b[1]):
                    np.testing.assert_allclose([x[k] for k in ("reference_risk", "candidate_risk", "loss")], [y[k] for k in ("reference_risk", "candidate_risk", "loss")], rtol=1e-11, atol=1e-11)

    def test_risk_zero_auxiliary_is_valid_but_zero_total_is_not(self):
        a = run_method(self.model, self.data, INFO, BASE, focal(), self.method(), [-7], FAST, 123)
        self.assertEqual(a[0]["status"], "matched")
        with self.assertRaises(DefinitionError):
            run_method(self.model, self.data, INFO, BASE, focal(), self.method(), [-8], FAST, 123)

    def test_jones_cp_focal_reference_fast(self):
        for family in ("jones", "cp"):
            for scheme in ("subsample_deletion", "bootstrap_augmentation"):
                method = {"preset": f"{family}_{scheme}_v2", "paths": 2, "posterior_draws": 32, "quantile_grid_size": 12}
                if family == "cp":
                    method.update(region="equal_tail", cache_regions=True)
                method = resolve_method(method)
                a = run_method(self.model, self.data, INFO, BASE, focal(2), method, [0, 1], REFERENCE, 123)
                b = run_method(self.model, self.data, INFO, BASE, focal(2), method, [0, 1], FAST, 123)
                self.assertEqual(a[0]["path_checksum"], b[0]["path_checksum"])
                self.assertEqual(a[0]["epss"], b[0]["epss"])
                np.testing.assert_allclose([r["loss"] for r in a[1]], [r["loss"] for r in b[1]], rtol=1e-10, atol=1e-11)

    def test_clarke_conditions_both_sides_on_one_anchor(self):
        method = resolve_method({"preset": "clarke_kl_v2", "focal_protocol": PROTOCOL, "max_candidates": 10})
        provider = PosteriorProvider(self.model, REFERENCE)
        with patch.object(provider, "fit", wraps=provider.fit) as fit:
            result, curves, _ = clarke(self.model, self.data, INFO, INFO, focal(3), method, [0, 1], provider, 123)
        self.assertEqual(result["epss"], 0)
        self.assertEqual(result["candidate_count"], 8)  # anchor only, then seven auxiliaries
        self.assertAlmostEqual(curves[0]["loss"], 0., places=12)
        for call in fit.call_args_list:
            FocalAnchor(self.model, self.data, "school_3").check(call.args[0])
        self.assertEqual(len(fit.call_args_list[0].args[0]["y"]), 1)
        self.assertEqual(result["interpretation"], "conditional_anchor_Clarke_project_adaptation")

    def test_focal_clarke_hdr_and_central_zero_auxiliary(self):
        for preset in ("clarke_tkl_v2", "clarke_central_tkl_v2"):
            method = resolve_method({"preset": preset, "focal_protocol": PROTOCOL, "max_candidates": 1})
            result, curves, _ = clarke(self.model, self.data, INFO, INFO, focal(8), method, [0], PosteriorProvider(self.model, REFERENCE), 123)
            self.assertEqual(result["epss"], 0)
            self.assertAlmostEqual(curves[0]["loss"], 0., places=12)

    def test_ppd_focal_paths_and_losses_preserved(self):
        for preset in ("jones_source_adaptation_v2", "cp_source_faithful", "cp_directional_ppd"):
            method = {"preset": preset, "paths": 2, "posterior_draws": 32, "quantile_grid_size": 12}
            if preset.startswith("cp"):
                method["region"] = "equal_tail"
            method = resolve_method(method)
            a = run_method(self.model, self.data, INFO, BASE, focal(3), method, [0, 1], REFERENCE, 123)
            b = run_method(self.model, self.data, INFO, BASE, focal(3), method, [0, 1], FAST, 123)
            self.assertEqual(a[0]["path_checksum"], b[0]["path_checksum"])
            self.assertEqual(a[0]["source_checksum"], b[0]["source_checksum"])
            np.testing.assert_allclose([r["loss"] for r in a[1]], [r["loss"] for r in b[1]], atol=1e-11)

    def test_quantile_cache_uses_identical_solver_once(self):
        from scipy.optimize import brentq
        post = self.model.optimized(self.data, INFO, FAST)
        measure = post.target(focal())
        with patch("scipy.optimize.brentq", wraps=brentq) as solver:
            a = measure.quantile([.1, .9])
            b = measure.quantile([.1, .9])
            self.assertEqual(solver.call_count, 2)
            np.testing.assert_array_equal(a, b)
            a[:] = 0
            np.testing.assert_array_equal(measure.quantile([.1, .9]), b)

    def test_batched_failure_is_not_silently_excluded(self):
        provider = PosteriorProvider(self.model, {**FAST, "max_expansions": 0})
        with self.assertRaises(NumericalFailure) as context:
            provider.fit_many([self.data, self.data], INFO, [1, 2])
        self.assertEqual(context.exception.context["input_rows"], [0])

    def test_cp_region_cache_preserves_all_region_results(self):
        model = BuiltinModel("normal_known")
        t = TargetSpec("mu", (Coordinate("mu"),))
        p = model.optimized({"y": [1, 2]}, {"family": "normal", "mean": 0, "sd": 2}).target(t)
        q = model.optimized({"y": [1, 3]}, {"family": "normal", "mean": 0, "sd": 3}).target(t)
        for region in ("equal_tail", "central", "exact_density", "kde"):
            reference = Coverage(region=region, alphas=(.05, .1))(p, q)
            fast = Coverage(region=region, alphas=(.05, .1), cache_prepared=True)
            self.assertEqual(fast(p, q), reference)
            self.assertEqual(fast(p, q), reference)
            self.assertEqual(len(fast._prepared), 2)

    def test_cache_bounded_and_seed_changes_do_not_resample_target(self):
        provider = PosteriorProvider(self.model, {**FAST, "school_cache_size": 2})
        a = provider.fit(self.data, INFO, 1)
        b = provider.fit(self.data, INFO, 999)
        self.assertIs(a, b)
        for i in range(5):
            provider.fit(self.model.subset(self.data, [i]), INFO, i)
        self.assertEqual(len(provider.cache), 2)
        for backend in ("importance", "general"):
            with self.assertRaises(DefinitionError):
                PosteriorProvider(self.model, {**FAST, "backend": backend})

    def test_risk_cache_reuses_baseline_but_not_changed_prior_or_anchor(self):
        provider = PosteriorProvider(self.model, FAST)
        method = self.method()
        a = run_method(self.model, self.data, INFO, BASE, focal(), method, [0, 1], FAST, 123, provider=provider)
        b = run_method(self.model, self.data, {**INFO, "mu_mean": 9.}, BASE, focal(), method, [0, 1], FAST, 123, provider=provider)
        self.assertEqual(a[0]["risk_components"]["risk_cache_hits"], 0)
        self.assertEqual(b[0]["risk_components"]["risk_cache_hits"], 2)
        c = run_method(self.model, self.data, INFO, BASE, focal(2), method, [0, 1], FAST, 123, provider=provider)
        self.assertEqual(c[0]["risk_components"]["risk_cache_hits"], 0)

    def test_planner_filters_morita_and_grouped_workflow(self):
        spec = load_spec(ROOT / "configs/eight_schools_smoke.yaml")
        spec.update(priors={"interest": INFO, "baseline": BASE}, prior_grid=[{"prior_id": "one"}], inference=FAST)
        spec["methods"] = [self.method(), {"preset": "morita_2012_case2_schools_mc_v2", "targets": ["hyper_log"],
            "epsilon_c": 10, "morita_design": "new_schools_empirical_s", "prior_predictive_reps": 2,
            "candidates": {"meaning": "pseudo_sample_size", "unit": "school", "values": [0, 1]}}]
        spec["candidates"] = {"meaning": "baseline_size_offset", "unit": "school", "values": [0, 1]}
        # resolve_method accepts user fields, not resolved preset-internal fields.
        spec["methods"][0] = {k: v for k, v in spec["methods"][0].items() if k != "id"}
        self.assertEqual(len(resolved_tasks(spec)), 4)
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = directory
            plan = prepare(spec)
            results = run_task_group(plan["manifest"], [0, 1, 2, 3])
            self.assertEqual(len(results), 4)
            self.assertEqual(merge(directory)["status"], "pass")
        bad = copy.deepcopy(spec)
        bad["methods"][1]["targets"] = ["school_1_marginal"]
        with self.assertRaises(DefinitionError):
            resolved_tasks(bad)


if __name__ == "__main__":
    unittest.main()
