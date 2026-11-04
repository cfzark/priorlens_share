"""Definition and workflow tests independent of legacy results."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from prioreffect_v2.core import Coordinate,TargetSpec,Measure,DefinitionError,NumericalFailure,validate_grid,digest
from prioreffect_v2.models import BuiltinModel,SCHOOLS_Y,SCHOOLS_S
from prioreffect_v2.inference import PosteriorProvider,weight_diagnostics
from prioreffect_v2.selection import BidirectionalSelector
from prioreffect_v2.losses import Coverage,kl
from prioreffect_v2.paths import build_paths
from prioreffect_v2.experiments import load_spec,resolved_tasks,prepare,run_task,merge
from prioreffect_v2.methods import morita
from prioreffect_v2.quadrature import WeightedPosterior

ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def test_grid_rejects_silent_rounding(self):
        for grid in [[0,.5],[1,1],[2,1],[-1,0],[float('nan')]]:
            with self.assertRaises(DefinitionError): validate_grid(grid,"number_augmented",8)

    def test_deletion_bound(self):
        with self.assertRaises(DefinitionError): validate_grid([0,8],"number_deleted",8)

    def test_joint_duplicate_variable(self):
        with self.assertRaises(DefinitionError): TargetSpec("bad",(Coordinate("mu"),Coordinate("mu")))

    def test_transform_inverse_and_jacobian(self):
        x=np.array([.2,.3,.7])
        for c in [Coordinate("x",scale=10),Coordinate("x","log",reference=10),Coordinate("x","square",scale=100),Coordinate("x","logit")]:
            np.testing.assert_allclose(c.inverse(c.forward(x)),x,atol=1e-12)
            numerical=(c.forward(x+1e-6)-c.forward(x-1e-6))/2e-6
            np.testing.assert_allclose(np.exp(c.log_jacobian(x)),numerical,rtol=1e-8)

    def test_ties_and_unmatched(self):
        rows=[{"m":m,"direction":d,"loss":1.,"adequacy":.2,"gap":.01} for m in [1,2] for d in ["negative","positive"]]
        self.assertEqual(BidirectionalSelector("positive")(rows)["signed_m"],1)
        self.assertEqual(BidirectionalSelector("negative")(rows)["signed_m"],-1)
        self.assertIsNone(BidirectionalSelector(delta=.1)(rows)["signed_m"])
        rows[0]["loss"]=float('nan')
        with self.assertRaises(NumericalFailure): BidirectionalSelector(delta=.1)(rows)

    def test_self_coverage_and_kl(self):
        model=BuiltinModel("beta_binomial")
        p=model.optimized({"y":[2,3,5]}, {"family":"beta","a":2.,"b":3.})
        t=TargetSpec("p",(Coordinate("p"),));measure=p.target(t)
        result=Coverage()(measure,measure)
        np.testing.assert_allclose(result["cp1"],[.95],atol=1e-12)
        self.assertAlmostEqual(result["loss"],0)
        self.assertAlmostEqual(kl(measure,measure),0)

    def test_equal_tail_monotone_invariance(self):
        model=BuiltinModel("beta_binomial")
        p=model.optimized({"y":[2,3,5]},{"family":"beta","a":2.,"b":3.})
        q=model.optimized({"y":[2,3,5]},{"family":"beta","a":4.,"b":2.})
        a=TargetSpec("p",(Coordinate("p"),));b=TargetSpec("logit",(Coordinate("p","logit"),))
        self.assertAlmostEqual(Coverage()(p.target(a),q.target(a))["loss"],Coverage()(p.target(b),q.target(b))["loss"],places=12)

    def test_kernel_is_not_normalized_density(self):
        from scipy.stats import norm
        x = norm.ppf((np.arange(1000)+.5)/1000)[:,None]
        p = Measure(x,np.ones(1000),log_kernel=lambda z: norm.logpdf(z[:,0])+100)
        q = Measure(x,np.ones(1000),log_kernel=lambda z: norm.logpdf(z[:,0])-50)
        # Constants do not affect HDR membership, but cannot be used as KL densities.
        result = Coverage(region="exact_density")(p,q)
        # Weighted finite-support HDR boundaries can differ by tied nodes; the
        # normalizer constants must not produce a macroscopic coverage change.
        np.testing.assert_allclose(result["cp1"],[.95],atol=3/len(x))
        np.testing.assert_allclose(result["cp2"],[.95],atol=3/len(x))
        for truncated in [False,True]:
            with self.assertRaises(DefinitionError): kl(p,q,truncated=truncated)

    def test_joint_density_not_exposed_as_marginal(self):
        model = BuiltinModel("nig")
        prior = {"family":"nig","mu0":0.,"tau0_sq":.2,"kappa":6.,"eta":5.}
        data = model.validate_data({"y":[-1,0,1]})
        original = model.optimized(data,prior)
        draws = original.sample(1000,np.random.default_rng(1))
        post = WeightedPosterior(model,data,prior,np.column_stack([draws[v] for v in model.variables]),np.ones(1000),{"engine":"test"})
        marginal = post.target(TargetSpec("mu",(Coordinate("mu"),)))
        joint = post.target(TargetSpec("joint",(Coordinate("mu"),Coordinate("sigma2","log"))))
        self.assertIsNone(marginal.logpdf)
        self.assertIsNone(marginal.log_kernel)
        self.assertIsNone(joint.logpdf)
        self.assertIsNotNone(joint.log_kernel)

    def test_normal_morita_known_information(self):
        model = BuiltinModel("normal_known")
        data = model.validate_data({"y":[-1,0,1]})
        prior = {"family":"normal","mean":0.,"sd":np.sqrt(.2)}
        method = {"epsilon_prior":{"family":"normal","mean":0.,"sd":np.sqrt(200)},"prior_predictive_reps":2}
        result, curves, _ = morita(model,data,prior,{},model.default_targets()[0],method,list(range(8)),PosteriorProvider(model),123)
        self.assertEqual(result["epss"],5)
        np.testing.assert_allclose([r["candidate_curvature"] for r in curves],np.arange(8)+.005,atol=2e-6)

    def test_worker_rejects_runtime_change(self):
        spec = load_spec(ROOT/"configs/normal_smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = directory
            plan = prepare(spec)
            with patch("prioreffect_v2.experiments.runtime_fingerprint",return_value={"changed":True}):
                with self.assertRaises(DefinitionError): run_task(plan["manifest"],0)

    def test_calibration_roundtrip_and_context_gate(self):
        from prioreffect_v2.calibration import calibrate
        spec = load_spec(ROOT/"configs/normal_smoke.yaml")
        spec["prior_grid"] = [{"prior_id":"center","mean":0.}]
        spec["methods"] = [{"preset":"cp_source_faithful","paths":2,"region":"equal_tail"}]
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = str(Path(directory)/"calibration")
            plan = prepare(spec)
            run_task(plan["manifest"],0)
            merge(spec["output_root"])
            artifact = Path(directory)/"delta.json"
            result = calibrate([spec["output_root"]],artifact,location_fields=["mean"])
            self.assertEqual(len(result["groups"]),1)
            spec["methods"][0]["delta_artifact"] = str(artifact)
            spec["prior_grid"] = [{"prior_id":"conflict","mean":3.}]
            task = resolved_tasks(spec)[0]
            self.assertIsNotNone(task["method"]["delta"])
            spec["priors"]["interest"]["sd"] = 2.
            with self.assertRaises(DefinitionError): resolved_tasks(spec)

    def test_worker_rejects_task_tampering(self):
        spec = load_spec(ROOT/"configs/normal_smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"] = directory
            plan = prepare(spec)
            task = next((Path(directory)/"tasks").glob("*.json"))
            payload = json.loads(task.read_text())
            payload["seed"] += 1
            task.write_text(json.dumps(payload))
            import pandas as pd
            manifest = pd.read_csv(plan["manifest"],sep="\t")
            task_id = manifest[manifest.task_file == str(task)].iloc[0].task_id
            with self.assertRaises(DefinitionError): run_task(plan["manifest"],task_id)

    def test_focal_school_paths(self):
        model=BuiltinModel("eight_schools");data=model.validate_data({"y":SCHOOLS_Y,"s":SCHOOLS_S})
        prior={"family":"schools","mu_mean":0.,"mu_sd":25.,"tau_kind":"half_t","tau_df":4.,"tau_scale":25.}
        p=model.optimized(data,prior)
        for scheme in ["subsample_deletion","bootstrap_deletion","bootstrap_augmentation"]:
            bank=build_paths(model,data,p,p,[0,1,2],scheme,2,np.random.default_rng(1),"school_1")
            for row in bank.candidates:
                current=row["positive_data"]
                self.assertEqual(current["unit_ids"].count("school_1"),1)
                i=current["unit_ids"].index("school_1")
                self.assertEqual(current["y"][i],28.)
                self.assertEqual(current["s"][i],15.)

    def test_is_degeneracy_is_failure(self):
        lw=np.full(1000,-1e6);lw[0]=0
        _,diagnostic=weight_diagnostics(lw,np.arange(1000)[:,None])
        self.assertFalse(diagnostic["converged"])

    def test_reuse_fallback(self):
        model=BuiltinModel("normal_known");data=model.validate_data({"y":[-1,0,1]})
        prior={"family":"normal","mean":0.,"sd":1.}
        provider=PosteriorProvider(model,{"posterior_reuse":"importance","is_min_ess":100000000})
        anchor=provider.fit(data,prior)
        post=provider.fit(model.validate_data({"y":[20,30]}),prior,anchor=anchor)
        self.assertTrue(post.accuracy["fallback_used"])
        self.assertEqual(post.engine,"analytic")

    def test_no_legacy_imports(self):
        import ast
        for path in (ROOT/"src/prioreffect_v2").glob("*.py"):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom): self.assertNotEqual(node.module,"prioreffect")
                if isinstance(node,ast.Import):
                    for alias in node.names: self.assertFalse(alias.name.startswith("prioreffect."))

    def test_source_faithful_scope(self):
        spec=load_spec(ROOT/"configs/eight_schools_smoke.yaml")
        spec["methods"]=[{"preset":"jones_source_faithful"}]
        with self.assertRaises(DefinitionError): resolved_tasks(spec)

    def test_grid_ids_preserve_seed_on_refinement(self):
        spec=load_spec(ROOT/"configs/normal_smoke.yaml")
        before={t["task_key"] for t in resolved_tasks(spec)}
        spec["prior_grid"].insert(1,{"prior_id":"refined","mean":-.5})
        after={t["task_key"] for t in resolved_tasks(spec)}
        self.assertTrue(before.issubset(after))

    def test_workflow_resume_corruption_and_completion(self):
        spec=load_spec(ROOT/"configs/normal_smoke.yaml")
        spec["prior_grid"]=spec["prior_grid"][:1];spec["methods"]=spec["methods"][:2]
        with tempfile.TemporaryDirectory() as directory:
            spec["output_root"]=directory
            plan=prepare(spec)
            run_task(plan["manifest"],0)
            missing=prepare(spec,resume=True)
            self.assertEqual(missing["n_pending"],1)
            run_task(missing["manifest"],0)
            result=merge(directory)
            self.assertEqual(result["status"],"pass")
            with self.assertRaises(DefinitionError): prepare(spec,resume=True)
            # Corrupt source row must be rejected by the merger.
            leaf=next((Path(directory)/"leaves").glob("*/result.json"))
            leaf.write_text('{}')
            with self.assertRaises(DefinitionError): merge(directory)


if __name__=="__main__": unittest.main(verbosity=2)
