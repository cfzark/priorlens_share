"""Regression tests for the September correctness audit and central CP regions."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import stats

from prioreffect_v2.core import (Coordinate, TargetSpec, Measure, DefinitionError,
    InadmissiblePosterior, NumericalFailure, digest, seed_for, seed_scope)
from prioreffect_v2.models import BuiltinModel, SCHOOLS_Y, SCHOOLS_S, lupus_data
from prioreffect_v2.inference import PosteriorProvider
from prioreffect_v2.quadrature import WeightedPosterior
from prioreffect_v2.paths import build_paths
from prioreffect_v2.methods import clarke, resolve_method, run_method
from prioreffect_v2.losses import Coverage, kl
from prioreffect_v2.regions import central_region
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task, merge

ROOT = Path(__file__).resolve().parents[1]
NIG = {"family": "nig", "mu0": 0., "tau0_sq": .2, "kappa": 6., "eta": 5.}
JEFFREYS = {"family": "jeffreys"}
JOINT = TargetSpec("joint", (Coordinate("mu"), Coordinate("sigma2", "log")))


class AuditFixes(unittest.TestCase):
    def test_clarke_feasible_domain_and_full_candidate_ids(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [-1, 0, 1, 2]})
        result, rows, _ = clarke(model, data, NIG, JEFFREYS, JOINT,
            {"posterior_draws": 64}, [0, 1, 3], PosteriorProvider(model), 123)
        self.assertEqual(result["candidate_counts"],
            {"enumerated": 25, "admissible": 16, "excluded_improper": 9, "numerical_failed": 0})
        self.assertEqual([r["status"] for r in rows[:2]], ["no_admissible_candidate"] * 2)
        # Independent feasible-set enumeration; retain full candidate numbering.
        reference = model.optimized(model.subset(data, []), NIG).target(JOINT, 64, 123)
        losses = []
        for candidate_id, ix in enumerate(itertools.combinations_with_replacement(range(4), 3), start=5):
            if len(set(ix)) == 1:
                continue
            post = model.optimized(model.subset(data, ix), JEFFREYS)
            losses.append((kl(post.target(JOINT, 64, 123), reference), candidate_id))
        best_loss, best_id = min(losses)
        self.assertEqual(rows[2]["candidate_id"], best_id)
        self.assertAlmostEqual(rows[2]["loss"], best_loss)

    def test_clarke_all_improper_is_not_unmatched(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [1., 1., 1.]})
        result, rows, _, _ = run_method(model, data, NIG, JEFFREYS, JOINT,
            resolve_method("clarke_kl_v2"), [3], {}, 123)
        self.assertEqual(result["status"], "inadmissible")
        self.assertIsNone(result["epss"])
        self.assertEqual(result["candidate_counts"]["excluded_improper"], 10)
        self.assertEqual(rows[0]["status"], "no_admissible_candidate")

    def test_proper_numerical_failure_not_excluded(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [-1, 0, 1, 2]})
        provider = PosteriorProvider(model)
        original = provider.fit
        def fit(current, prior, *args, **kwargs):
            if len(set(current["y"])) > 1 and prior == JEFFREYS:
                raise NumericalFailure("deliberate solver failure")
            return original(current, prior, *args, **kwargs)
        with patch.object(provider, "fit", side_effect=fit):
            with self.assertRaises(NumericalFailure) as caught:
                clarke(model, data, NIG, JEFFREYS, JOINT, {}, [3], provider, 123)
        self.assertEqual(caught.exception.context["counts_at_m"]["numerical_failed"], 1)

    def test_admissibility_independent_of_backend(self):
        model = BuiltinModel("nig")
        for backend in ("optimized", "importance", "general"):
            with self.assertRaises(InadmissiblePosterior):
                PosteriorProvider(model, {"backend": backend}).fit({"y": [1, 1, 1]}, JEFFREYS)
        # Large common offsets must not turn nonconstant data into an exclusion.
        post = model.optimized({"y": [1e8, 1e8 + 1, 1e8 + 2]}, JEFFREYS)
        self.assertGreater(post.params["eta"], 0)

    def test_whole_path_redraw_proper_nested_reproducible(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [-1., 0., 1., 2.]})
        p, q = model.optimized(data, NIG), model.optimized(data, JEFFREYS)
        check = lambda current: model.check_posterior_admissibility(current, JEFFREYS)
        banks = [build_paths(model, data, p, q, [0, 1], "bootstrap_deletion", 100,
            np.random.default_rng(123), admissibility_check=check,
            invalid_path_policy="redraw_whole_path") for _ in range(2)]
        bank = banks[0]
        self.assertGreater(bank.diagnostics["rejected_paths"], 0)
        self.assertEqual(bank.checksum, banks[1].checksum)
        self.assertEqual(bank.diagnostics, banks[1].diagnostics)
        for full, small in zip(bank.candidates[::2], bank.candidates[1::2]):
            self.assertEqual(full["positive_data"]["y"][:3], small["positive_data"]["y"])
            self.assertEqual(small["positive_data"], small["negative_data"])
            check(small["positive_data"])
        with self.assertRaises(InadmissiblePosterior):
            build_paths(model, data, p, q, [0, 1], "bootstrap_deletion", 100,
                np.random.default_rng(123), admissibility_check=check, invalid_path_policy="error")

    def test_redraw_limit_and_policy_preflight(self):
        model = BuiltinModel("nig")
        data = model.validate_data({"y": [1., 1., 1., 1.]})
        post = model.optimized(data, NIG)
        with self.assertRaisesRegex(InadmissiblePosterior, "redraw limit"):
            build_paths(model, data, post, post, [1], "bootstrap_deletion", 1,
                np.random.default_rng(1), invalid_path_policy="redraw_whole_path", max_path_redraws=2,
                admissibility_check=lambda d: model.check_posterior_admissibility(d, JEFFREYS))
        spec = load_spec(ROOT / "configs/nig_smoke.yaml")
        spec["methods"] = [{"preset": "cp_bootstrap_deletion_v2", "region": "central", "paths": 2}]
        spec["candidates"] = {"values": [0, 1], "meaning": "number_deleted", "unit": "observation"}
        with self.assertRaisesRegex(DefinitionError, "invalid_path_policy"):
            resolved_tasks(spec)
        spec["methods"][0]["invalid_path_policy"] = "redraw_whole_path"
        self.assertTrue(resolved_tasks(spec))

    def test_focal_mismatch_and_reordered_original_identity(self):
        model = BuiltinModel("eight_schools")
        data = model.validate_data({"y": SCHOOLS_Y, "s": SCHOOLS_S})
        bad = TargetSpec("bad", (Coordinate("theta[3]"),), "school_1")
        good = TargetSpec("third", (Coordinate("theta[3]"),), "school_3")
        reduced = model.subset(data, [2, 0, 4])
        prior = {"family": "schools", "mu_mean": 0., "mu_sd": 25., "tau_kind": "half_t", "tau_df": 4., "tau_scale": 25.}
        grid = model.optimized(reduced, prior, {"school_grid_size": 101})
        weighted = WeightedPosterior(model, reduced, prior,
            np.array([[1., 2.], [3., 4.]]), np.ones(2), {"engine": "test"})
        for posterior in (grid, weighted):
            with self.assertRaises(DefinitionError):
                posterior.target(bad)
            self.assertTrue(np.isfinite(posterior.target(good).mean).all())
        spec = load_spec(ROOT / "configs/eight_schools_smoke.yaml")
        spec["targets"][2]["coordinates"][0]["variable"] = "theta[3]"
        with self.assertRaisesRegex(DefinitionError, "same original school"):
            resolved_tasks(spec)

    def test_lupus_scale_required_only_for_predictions(self):
        model = BuiltinModel("lupus_logistic")
        data = lupus_data()
        data.pop("x_center"); data.pop("x_divisor")
        data = model.validate_data(data)
        p = WeightedPosterior(model, data, {}, np.array([[-1., 3.], [-.5, 2.]]), np.ones(2), {"engine": "test"})
        self.assertEqual(p.target(TargetSpec("beta", (Coordinate("beta0"),))).values.shape, (2, 1))
        with self.assertRaisesRegex(DefinitionError, "x_center"):
            p.target(TargetSpec("p0", (Coordinate("p_x_0"),)))
        for divisor in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(DefinitionError):
                model.validate_data({**data, "x_center": 0., "x_divisor": divisor})
        valid = {**data, "x_center": .5, "x_divisor": 2.}
        self.assertEqual(model.subset(valid, [0, 1])["x_center"], .5)
        p.data = valid
        np.testing.assert_allclose(p.target(TargetSpec("p0", (Coordinate("p_x_0"),))).values[:, 0],
                                   stats.logistic.cdf([-1 - .75, -.5 - .5]))

    def test_seed_versions_replay_and_scope(self):
        expected = int(digest([123, "fit", 1])[:8], 16)
        self.assertEqual(seed_for(123, "fit", 1, scheme="sha256_32_v1"), expected)
        with seed_scope("sha256_64_v2"):
            wide = seed_for(123, "fit", 1)
            self.assertEqual(wide, int(digest([123, "fit", 1])[:16], 16))
            self.assertGreater(wide, 2**32)
        self.assertEqual(seed_for(123, "fit", 1), expected)
        seeds = {seed_for(123, "task", i, scheme="sha256_64_v2") for i in range(100000)}
        self.assertEqual(len(seeds), 100000)  # Regression, not a mathematical guarantee.
        spec = load_spec(ROOT / "configs/normal_smoke.yaml")
        spec["seed_scheme"] = "sha256_64_v2"
        first = resolved_tasks(spec)
        spec["prior_grid"].append({"prior_id": "new", "mean": 2.})
        self.assertTrue({t["task_key"] for t in first} <= {t["task_key"] for t in resolved_tasks(spec)})

    def test_production_budgets_explicit(self):
        spec = load_spec(ROOT / "configs/normal_smoke.yaml")
        spec["run_mode"] = "production"
        with self.assertRaisesRegex(DefinitionError, "explicit"):
            resolved_tasks(spec)
        for m in spec["methods"]:
            m.update(posterior_draws=128, quantile_grid_size=128)
        self.assertTrue(resolved_tasks(spec))

    def test_runtime_errors_recorded_but_not_completed(self):
        spec = load_spec(ROOT / "configs/normal_smoke.yaml")
        spec["methods"] = spec["methods"][:1]
        spec["prior_grid"] = spec["prior_grid"][:1]
        for error, status in [(DefinitionError("deliberate definition error"), "definition_error"),
                              (RuntimeError("deliberate bug"), "unexpected_error")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                spec["output_root"] = directory
                plan = prepare(spec)
                with patch("prioreffect_v2.experiments.run_method", side_effect=error):
                    with self.assertRaises(type(error)):
                        run_task(plan["manifest"], 0)
                leaf = next((Path(directory) / "leaves").iterdir())
                result = json.loads((leaf / "result.json").read_text())
                self.assertEqual(result["status"], status)
                self.assertIn("traceback", result)
                self.assertFalse((leaf / "done.json").exists())
                self.assertEqual(len(list((leaf / "failures").glob("*.json"))), 1)
                with self.assertRaises(FileNotFoundError):
                    merge(directory)
                # A later successful retry has precisely the required done files;
                # the failure audit remains but is not published as a result leaf.
                run_task(plan["manifest"], 0)
                self.assertEqual(merge(directory)["status"], "pass")


class CentralRegions(unittest.TestCase):
    def test_scalar_equivalent_to_existing_equal_tail(self):
        settings = [("normal_known", {"y": [-1, 0, 1]}, {"family": "normal", "mean": 1., "sd": 1.}, {"family": "flat"}, "mu"),
                    ("poisson_gamma", {"y": [1, 2, 3]}, {"family": "gamma", "shape": 3., "rate": 2.}, JEFFREYS, "lambda"),
                    ("beta_binomial", {"y": [1, 2, 3]}, {"family": "beta", "a": 2., "b": 3.}, JEFFREYS, "p"),
                    ("nig", {"y": [-1, 0, 1]}, NIG, JEFFREYS, "mu")]
        for name, data, prior, baseline, variable in settings:
            model = BuiltinModel(name)
            target = TargetSpec("scalar", (Coordinate(variable),))
            p, q = [model.optimized(data, pr).target(target) for pr in (prior, baseline)]
            for family, weight, cap in [("historical", 1., False), ("direct", 0., True), ("direct", 1., True)]:
                kw = {"alphas": (.05, .1), "loss_family": family, "gap_weight": weight, "cap_nominal": cap}
                old, new = Coverage(**kw)(p, q), Coverage(region="central", **kw)(p, q)
                for key in old:
                    self.assertEqual(new[key], old[key])

    def test_joint_nominal_mass_not_product_of_marginal_intervals(self):
        u = (np.arange(100) + .5) / 100
        x, y = np.meshgrid(u, u)
        p = Measure(np.column_stack([x.ravel(), y.ravel()]), np.ones(10000))
        region = central_region(p, .05)
        self.assertGreaterEqual(region.own_mass, .95)
        self.assertLess(region.own_mass, .97)
        self.assertLess(region.tail_probability, .025)
        product = float(p.weights @ np.all((p.values >= .025) & (p.values <= .975), axis=1))
        self.assertLess(product, .93)
        result = Coverage(region="central")(p, p)
        self.assertEqual(result["cp1"], result["cp2"])
        self.assertAlmostEqual(result["cp1"][0], region.own_mass)

    def test_correlated_weighted_joint_and_cross_coverage(self):
        rng = np.random.default_rng(415)
        z = rng.multivariate_normal([0, 0], [[1, .8], [.8, 1]], 3000)
        p = Measure(z, rng.uniform(.2, 1., len(z)))
        q = Measure(z + [1., -.3], rng.uniform(.2, 1., len(z)))
        result = Coverage(region="central", alphas=(.05, .2))(p, q)
        for k, alpha in enumerate((.05, .2)):
            rp, rq = central_region(p, alpha), central_region(q, alpha)
            self.assertGreaterEqual(rp.own_mass, 1 - alpha)
            self.assertLess(rp.own_mass - (1 - alpha), 4 * p.weights.max())
            manual = float(p.weights @ np.all((p.values >= rq.lower) & (p.values <= rq.upper), axis=1))
            self.assertAlmostEqual(result["cp1"][k], manual)
            self.assertEqual(result["region_diagnostics"][k]["p"]["own_mass"], rp.own_mass)

    def test_holdout_mass_and_positive_affine_invariance(self):
        rng = np.random.default_rng(991)
        covariance = [[1., .65], [.65, 2.]]
        p = Measure(rng.multivariate_normal([0, 0], covariance, 16000), np.ones(16000))
        region = central_region(p)
        holdout = Measure(rng.multivariate_normal([0, 0], covariance, 30000), np.ones(30000))
        self.assertLess(abs(region.mass(holdout) - .95), .012)
        transformed = central_region(Measure(p.values * [10., .5] + [2., -1.], p.weights))
        np.testing.assert_allclose(transformed.lower, region.lower * [10., .5] + [2., -1.], atol=1e-12)
        np.testing.assert_allclose(transformed.upper, region.upper * [10., .5] + [2., -1.], atol=1e-12)

    def test_tied_support_reports_overshoot_without_density(self):
        p = Measure(np.ones((10, 2)), np.ones(10))
        result = Coverage(region="central")(p, p)
        self.assertAlmostEqual(result["region_diagnostics"][0]["p"]["own_mass"], 1.)
        self.assertAlmostEqual(result["region_diagnostics"][0]["p"]["mass_error"], .05)
        self.assertAlmostEqual(result["adequacy"], .05)

    def test_central_region_spec_and_calibration_isolation(self):
        from prioreffect_v2.calibration import context
        spec = load_spec(ROOT / "configs/nig_smoke.yaml")
        spec["methods"] = [{"preset": "cp_nested_directional_v2", "paths": 1, "region": "central"}]
        central = resolved_tasks(spec)[0]
        spec["methods"][0]["region"] = "exact_density"
        hdr = resolved_tasks(spec)[0]
        self.assertNotEqual(digest(context(central, ["mu0"])), digest(context(hdr, ["mu0"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
