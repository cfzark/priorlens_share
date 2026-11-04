"""Resolved requests, immutable task manifests and strict publication."""
from __future__ import annotations
import copy
import csv
import json
import os
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .core import DefinitionError, InadmissiblePosterior, NumericalFailure, SEED_SCHEMES, TargetSpec, digest, native, seed_for, validate_grid, write_json
from .models import BuiltinModel
from .inference import load_adapter
from .methods import resolve_method, run_method, validate_clarke_settings
from .risk_components import validate_risk_settings, validate_risk_moments
from .morita_versions import VERSIONS as MORITA_VERSIONS, validate_version as validate_morita_version


def runtime_fingerprint(spec):
    """Freeze executable code, registered external sources, and numeric versions."""
    package = Path(__file__).resolve().parent
    files = {str(p.relative_to(package)): digest(p.read_text()) for p in sorted(package.glob("*.py"))}
    for p in sorted((package / "stan").glob("*.stan")):
        files["stan/" + p.name] = digest(p.read_text())
    registration = spec["model"]
    external = [registration[k] for k in ("adapter", "stan") if k in registration]
    external += registration.get("extra_files", [])
    external += [m["delta_artifact"] for m in spec["methods"] if isinstance(m, dict) and m.get("delta_artifact")]
    for name in external:
        p = Path(name)
        files[str(p.resolve())] = digest(p.read_text())
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "PyYAML", "cmdstanpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"files": files, "versions": versions}


def checked_leaf(leaf, scientific_hash, key):
    done = json.loads((leaf / "done.json").read_text())
    required = {"result.json", "curves.json", "paths.json", "posterior_accuracy.json"}
    if done.get("scientific_hash") != scientific_hash or done.get("task_key") != key or set(done.get("files", {})) != required:
        raise DefinitionError(f"Leaf provenance or artifact coverage mismatch: {key}")
    for name, expected in done["files"].items():
        if digest((leaf / name).read_text()) != expected:
            raise DefinitionError(f"Corrupt leaf {key}/{name}; preserve it for audit, do not overwrite")
    result = json.loads((leaf / "result.json").read_text())
    if result.get("task_key") != key or result.get("scientific_hash") != scientific_hash:
        raise DefinitionError("Result identity mismatch")
    if result.get("status") not in {"matched", "unmatched"}:
        raise DefinitionError("Numerical failure cannot be merged as unmatched")
    return result


def load_spec(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise DefinitionError("schema_version: 1 is required")
    allowed = {"schema_version", "model", "data", "priors", "prior_grid", "targets", "methods", "candidates", "inference", "execution", "seed", "seed_scheme", "run_mode", "output_root", "replication_id"}
    if set(spec) - allowed:
        raise DefinitionError(f"Unknown fields: {set(spec) - allowed}")
    for field in ["model", "data", "priors", "targets", "methods", "candidates", "output_root"]:
        if field not in spec:
            raise DefinitionError(f"Missing required field {field}")
    if isinstance(spec["data"], str):
        with (path.parent / spec["data"]).open() as f:
            spec["data"] = json.load(f)
    spec["inference"] = {"backend": "auto", "posterior_reuse": "none", **spec.get("inference", {})}
    spec["seed"] = int(spec.get("seed", 20260312))
    spec.setdefault("seed_scheme", "sha256_64_v2")
    spec.setdefault("run_mode", "production")
    for method in spec["methods"]:
        if isinstance(method, dict) and method.get("delta_artifact"):
            method["delta_artifact"] = str((path.parent / method["delta_artifact"]).resolve())
    if isinstance(spec.get("prior_grid"), str):
        with (path.parent / spec["prior_grid"]).open() as f:
            spec["prior_grid"] = json.load(f) if str(spec["prior_grid"]).endswith(".json") else list(csv.DictReader(f))
    if "adapter" in spec["model"]:
        for key in ["adapter", "stan"]:
            spec["model"][key] = str((path.parent / spec["model"][key]).resolve())
        spec["model"]["extra_files"] = [str((path.parent / p).resolve()) for p in spec["model"].get("extra_files", [])]
    # Output paths are deliberately repository-relative; data/adapter files are spec-relative.
    spec["output_root"] = str(Path(spec["output_root"]).resolve())
    return spec


def model_for(spec):
    registration = spec["model"]
    return load_adapter(registration) if "adapter" in registration else BuiltinModel(registration["name"], **registration.get("options", {}))


def set_field(obj, path, value):
    keys = path.split(".")
    current = obj
    for key in keys[:-1]:
        if isinstance(current, list):
            current = current[int(key)]
        elif key in current:
            current = current[key]
        else:
            raise DefinitionError(f"Prior grid addresses nonexistent field {path}")
    final = int(keys[-1]) if isinstance(current, list) else keys[-1]
    if isinstance(current, dict) and final not in current:
        raise DefinitionError(f"Prior grid addresses nonexistent field {path}")
    old = current[final]
    if isinstance(old, (float, int)):
        value = float(value)
    current[final] = value


def resolved_tasks(spec):
    seed_scheme = spec.get("seed_scheme", "sha256_64_v2")
    if seed_scheme not in SEED_SCHEMES:
        raise DefinitionError("Unknown seed_scheme")
    mode = spec.get("run_mode", "production")
    if mode not in {"smoke", "production"}:
        raise DefinitionError("run_mode must be smoke or production")
    model = model_for(spec)
    from .school_acceleration import validate_settings
    validate_settings(model, spec["inference"])
    data = model.validate_data(spec["data"])
    if not len(data["y"]):
        raise DefinitionError("Observed analysis data must be nonempty")
    if spec["inference"]["backend"] not in {"auto", "optimized", "general", "importance"}:
        raise DefinitionError("Unknown posterior backend")
    if spec["inference"]["posterior_reuse"] not in {"none", "importance"}:
        raise DefinitionError("posterior_reuse must be none or importance")
    if "interest" not in spec["priors"] or "baseline" not in spec["priors"]:
        raise DefinitionError("Both full interest and baseline prior specifications are required")
    if hasattr(model, "validate_prior"):
        model.validate_prior(spec["priors"]["baseline"])
    targets = [TargetSpec.parse(t) for t in spec["targets"]]
    if len({t.id for t in targets}) != len(targets):
        raise DefinitionError("Duplicate target IDs")
    methods = [resolve_method(m) for m in spec["methods"]]
    for method in methods:
        if "targets" in method and (not isinstance(method["targets"], list) or not method["targets"]
                or len(set(method["targets"])) != len(method["targets"])
                or set(method["targets"]) - {t.id for t in targets}):
            raise DefinitionError("Method targets must be a nonempty unique list of declared target IDs")
    if mode == "production":
        for method in methods:
            required = {"posterior_draws"}
            required |= {"jones": {"paths", "quantile_grid_size"}, "cp": {"paths"}, "lin": {"paths"},
                         "reimherr": {"risk_reps"}, "wiesenfarth": {"risk_reps"},
                         "morita": {"prior_predictive_reps"}, "clarke": {"max_candidates"}}[method["name"]]
            if method.get("preset") in MORITA_VERSIONS:
                # Analytic entries use neither posterior samples nor pseudo-data MC.
                required = {"prior_predictive_reps"} if method["preset"] == "morita_2012_case2_schools_mc_v2" else set()
            if method["name"] == "jones" and any(t.dimension == 2 and ("targets" not in method or t.id in method["targets"]) for t in targets):
                required.add("projections")
            if required - method.keys():
                raise DefinitionError(f"Production method {method['id']} requires explicit {sorted(required - method.keys())}; tiny defaults are smoke-only")
    if len({m["id"] for m in methods}) != len(methods):
        raise DefinitionError("Duplicate method IDs")
    grid_rows = spec.get("prior_grid", [{"prior_id": "interest"}])
    if not isinstance(grid_rows, list) or not grid_rows or len({r["prior_id"] for r in grid_rows}) != len(grid_rows):
        raise DefinitionError("prior_grid must have unique stable prior_id values")
    tasks = []
    for row in grid_rows:
        prior = copy.deepcopy(spec["priors"]["interest"])
        for key, value in row.items():
            if key != "prior_id":
                set_field(prior, key, value)
        if hasattr(model, "validate_prior"):
            model.validate_prior(prior)
        for method0 in methods:
            if method0["name"] == "lin":
                from .lin_runner import validate_design
                chosen_targets = [t for t in targets if "targets" not in method0 or t.id in method0["targets"]]
                validate_design(model, data, prior, spec["priors"]["baseline"], chosen_targets)
                if "adapter" in spec["model"]:
                    raise DefinitionError("Lin uses registered built-in data generators")
            if method0.get("classic_only") and model.name not in {"normal_known", "poisson_gamma", "beta_binomial"}:
                raise DefinitionError("jones_source_faithful is reserved for classic conjugate 1D")
            for target, target_dict in zip(targets, spec["targets"]):
                if "targets" in method0 and target.id not in method0["targets"]:
                    continue
                method = {**method0}
                if isinstance(method.get("region"), dict):
                    method["region"] = method["region"].get(target.id)
                if method["name"] == "clarke":
                    validate_clarke_settings(method)
                if method["name"] == "cp" and method.get("region") not in {"equal_tail", "central", "kde", "exact_density", "pdf_hdr"}:
                    raise DefinitionError("CP requires an explicit region per target")
                if method["name"] == "cp" and method["region"] == "equal_tail" and target.dimension != 1:
                    raise DefinitionError("Joint target requires central, kde, exact_density, or pdf_hdr region")
                if method["name"] == "cp" and method.get("region") == "pdf_hdr":
                    if method.get("quantile_rule", "weighted_cdf") != "weighted_cdf":
                        raise DefinitionError("pdf_hdr requires weighted density ranks, not numpy_linear")
                    if model.name == "lupus_logistic" and (
                            {c.variable for c in target.coordinates} != {"beta0", "beta1"}
                            or any(c.transform != "identity" for c in target.coordinates)):
                        raise DefinitionError("Lupus pdf_hdr currently requires joint(beta0,beta1) "
                                              "in affine coefficient coordinates; use central for marginals")
                if hasattr(model, "validate_target"):
                    model.validate_target(target, data)
                if target.focal_unit_id and data["unit_ids"].count(target.focal_unit_id) != 1:
                    raise DefinitionError("Original focal unit must occur exactly once")
                if model.name == "eight_schools":
                    for c in target.coordinates:
                        if c.variable not in model.variables and not c.variable.startswith("theta["):
                            raise DefinitionError(f"Unsupported school target {c.variable}")
                elif model.name != "lupus_logistic" and any(c.variable not in model.variables for c in target.coordinates):
                    raise DefinitionError("Target variable not declared by model")
                candidates = method.get("candidates", spec["candidates"])
                if candidates.get("unit") != model.unit:
                    raise DefinitionError(f"Candidate unit must be {model.unit}")
                expected = {"reimherr": "baseline_size_offset", "wiesenfarth": "informative_size_reduction", "clarke": "pseudo_sample_size", "morita": "pseudo_sample_size", "lin": "informative_fresh_size_reduction"}.get(method["name"], "number_deleted" if method.get("scheme", "").endswith("deletion") else "number_augmented")
                if candidates.get("meaning") != expected:
                    raise DefinitionError(f"{method['id']} requires candidate meaning {expected}")
                if method["name"] == "lin":
                    from .lin import signed_grid
                    grid = signed_grid(len(data["y"]), candidates["values"])
                    if len(data["y"]) in grid:
                        model.check_posterior_admissibility(model.subset(data, [], auxiliary=True), prior)
                else:
                    grid = validate_grid(candidates["values"], expected, len(data["y"]), candidates.get("min_retained", 3 if model.name == "nig" else 1))
                if method["name"] in {"reimherr", "wiesenfarth"}:
                    validate_risk_settings(method, model, target)
                    validate_risk_moments(model, data, prior, spec["priors"]["baseline"], target, method, grid)
                    if method.get("risk_coupling") == "shared_by_size":
                        method.setdefault("risk_data_seed", seed_for(spec["seed"], spec.get("replication_id", 0), "risk_data", method.get("path_group", "shared"), scheme=seed_scheme))
                if target.focal_unit_id:
                    from .focal import validate_focal
                    validate_focal(model, data, target, method)
                    if method["name"] == "morita":
                        raise DefinitionError("Morita focal epsilon prior is not defined; use Case 2 hyperparameter targets")
                    if method["name"] == "clarke" and method.get("focal_protocol") != "fixed_observed_anchor_v1":
                        raise DefinitionError("Conditional Clarke requires explicit focal_protocol: fixed_observed_anchor_v1")
                versioned_morita = method.get("preset") in MORITA_VERSIONS
                if versioned_morita:
                    validate_morita_version(model, data, prior, target, method)
                    if method["preset"] == "morita_2012_case2_schools_mc_v2" and (
                            spec["inference"]["backend"] not in {"auto", "optimized"}
                            or spec["inference"]["posterior_reuse"] != "none"):
                        raise DefinitionError("Morita 2012 Schools covariance reference requires optimized/auto and no importance reuse")
                if method["name"] == "morita" and not versioned_morita and "epsilon_prior" not in method:
                    raise DefinitionError("Specify Morita epsilon_prior")
                if method["name"] == "morita" and not versioned_morita and hasattr(model, "validate_prior"):
                    model.validate_prior(method["epsilon_prior"])
                if method.get("delta") is not None and method["name"] != "cp":
                    raise DefinitionError("Adequacy constraint belongs to CP")
                if method.get("invalid_path_policy") == "redraw_whole_path" and not hasattr(model, "check_posterior_admissibility"):
                    raise DefinitionError("Adapter must supply check_posterior_admissibility for whole-path redraw")
                if model.name == "nig" and method.get("scheme") == "bootstrap_deletion" and any(p["family"] == "jeffreys" for p in (prior, spec["priors"]["baseline"])) and "invalid_path_policy" not in method:
                    raise DefinitionError("NIG/Jeffreys bootstrap deletion requires explicit invalid_path_policy: error or redraw_whole_path")
                # Clarke's feasible domain explicitly excludes improper candidates,
                # including m=0; a size with no admissible candidates is recorded.
                tasks.append({"prior_id": row["prior_id"], "prior": prior, "baseline": spec["priors"]["baseline"],
                    "target": target_dict, "method": method, "candidates": {**candidates, "values": grid.tolist()},
                    "data": data, "model": spec["model"], "inference": spec["inference"],
                    "seed": seed_for(spec["seed"], spec.get("replication_id", 0), row["prior_id"], method.get("path_group", method.get("scheme", method["id"])), scheme=seed_scheme),
                    "seed_scheme": seed_scheme, "run_mode": mode,
                    "replication_id": spec.get("replication_id", 0)})
                if method["name"] == "lin":
                    tasks[-1].update(seed=20260312, seed_scheme="sha256_64_v2",
                        shared_targets=[t for t in spec["targets"] if "targets" not in method or t["id"] in method["targets"]])
    for task in tasks:
        from .calibration import apply_calibration
        apply_calibration(task)
        task["task_key"] = digest(task)
    return tasks


def prepare(spec, resume=False):
    tasks = resolved_tasks(spec)
    root = Path(spec["output_root"])
    science = {k: v for k, v in spec.items() if k not in {"execution", "output_root"}}
    runtime = runtime_fingerprint(spec)
    source_hash = digest(runtime)
    scientific_hash = digest([science, source_hash])
    if (root / "validation.json").exists():
        raise DefinitionError("Completed root is immutable; select a new output_root")
    if (root / "resolved_spec.json").exists():
        frozen = json.loads((root / "resolved_spec.json").read_text())
        if frozen["scientific_hash"] != scientific_hash:
            raise DefinitionError("Existing root has different scientific settings or code")
        if not resume:
            raise DefinitionError("Root already planned; use explicit missing-only resume")
    write_json(root / "resolved_spec.json", {"spec": spec, "scientific_hash": scientific_hash,
        "source_hash": source_hash, "runtime": runtime, "execution_hash": digest(spec.get("execution", {})), "expected_tasks": len(tasks)})
    manifest = root / ("missing_manifest.tsv" if resume else "manifest.tsv")
    pending = []
    for task in tasks:
        task["scientific_hash"] = scientific_hash
        task_path = root / "tasks" / (task["task_key"] + ".json")
        write_json(task_path, task)
        done = root / "leaves" / task["task_key"] / "done.json"
        if resume and done.exists():
            checked_leaf(done.parent, scientific_hash, task["task_key"])
            continue
        pending.append({"task_id": len(pending), "task_key": task["task_key"], "task_file": str(task_path.resolve())})
    pd.DataFrame(pending, columns=["task_id", "task_key", "task_file"]).to_csv(manifest, sep="\t", index=False)
    return {"manifest": str(manifest), "n_pending": len(pending), "expected_tasks": len(tasks), "scientific_hash": scientific_hash}


def run_task(manifest, task_id, *, provider=None):
    manifest = Path(manifest).resolve()
    rows = pd.read_csv(manifest, sep="\t")
    matches = rows[rows.task_id == int(task_id)]
    if len(matches) != 1:
        raise DefinitionError("Task ID is not unique in manifest")
    task = json.loads(Path(matches.iloc[0].task_file).read_text())
    root = manifest.parent
    frozen = json.loads((root / "resolved_spec.json").read_text())
    if digest(runtime_fingerprint(frozen["spec"])) != frozen["source_hash"]:
        raise DefinitionError("Runtime sources/dependencies changed since plan; create a new output root")
    actual_key = digest({k: v for k, v in task.items() if k not in {"task_key", "scientific_hash"}})
    if actual_key != task["task_key"] or actual_key != matches.iloc[0].task_key or task["scientific_hash"] != frozen["scientific_hash"]:
        raise DefinitionError("Task definition differs from frozen manifest")
    leaf = root / "leaves" / task["task_key"]
    if (leaf / "done.json").exists():
        raise DefinitionError("Task already completed")
    leaf.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (leaf / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (leaf / "done.json").exists():
            raise DefinitionError("Task completed by another worker")
        import time
        import traceback
        start = time.perf_counter()
        caught = None
        try:
            model = model_for(task)
            result, curves, paths, accuracy = run_method(model, task["data"], task["prior"], task["baseline"], TargetSpec.parse(task["target"]), task["method"], task["candidates"]["values"], task["inference"], task["seed"], seed_scheme=task.get("seed_scheme", "sha256_32_v1"), provider=provider,
                replication_id=task.get("replication_id", 0), shared_targets=task.get("shared_targets"))
        except Exception as error:
            # Persist context without relabelling configuration/programming errors
            # as CP unmatched, or letting a partial task acquire a done marker.
            caught = error
            status = ("inadmissible" if isinstance(error, InadmissiblePosterior) else
                      "definition_error" if isinstance(error, DefinitionError) else
                      "numerical_failure" if isinstance(error, NumericalFailure) else "unexpected_error")
            result = {"epss": None, "status": status, "error": str(error), "error_type": type(error).__name__,
                      "failure_context": getattr(error, "context", {}), "traceback": traceback.format_exc()}
            curves, paths, accuracy = [], [], getattr(error, "posterior_accuracy", [])
        result.update(task_key=task["task_key"], prior_id=task["prior_id"], method_id=task["method"]["id"], target_id=task["target"]["id"], replication_id=task["replication_id"], scientific_hash=task["scientific_hash"], wall_seconds=time.perf_counter()-start)
        result.update(seed=task["seed"], seed_scheme=task.get("seed_scheme", "sha256_32_v1"), run_mode=task.get("run_mode", "legacy"))
        write_json(leaf / "result.json", result)
        for name, table in [("curves", curves), ("paths", paths), ("posterior_accuracy", accuracy)]:
            write_json(leaf / (name + ".json"), table)
        if result["status"] not in {"matched", "unmatched"}:
            write_json(leaf / "failures" / f"{time.time_ns()}.json", result)
            if caught is not None:
                raise caught
            raise NumericalErrorForCLI(result["error"])
        hashes = {name: digest((leaf / name).read_text()) for name in ("result.json", "curves.json", "paths.json", "posterior_accuracy.json")}
        write_json(leaf / "done.json", {"scientific_hash": task["scientific_hash"], "task_key": task["task_key"], "files": hashes})
    return result


class NumericalErrorForCLI(RuntimeError):
    pass


def run_task_group(manifest, task_ids):
    """Same immutable task leaves, one process with bounded posterior reuse.

    Grouping changes scheduling only. It does not alter task keys or seeds.
    Use the saved missing manifest for recovery, never overwrite done leaves.
    """
    from .inference import PosteriorProvider
    ids = list(task_ids)
    rows = pd.read_csv(manifest, sep="\t")
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(rows.task_id):
        raise DefinitionError("Task group needs unique existing task IDs")
    tasks = {int(r.task_id): json.loads(Path(r.task_file).read_text()) for r in rows.itertuples() if r.task_id in ids}
    providers, results = {}, []
    for task_id in ids:
        task = tasks[task_id]
        key = digest([task["model"], task["inference"]])
        if key not in providers:
            providers[key] = PosteriorProvider(model_for(task), task["inference"])
        results.append(run_task(manifest, task_id, provider=providers[key]))
    return results


def merge(root):
    root = Path(root)
    frozen = json.loads((root / "resolved_spec.json").read_text())
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t")
    if manifest.task_key.duplicated().any() or len(manifest) != frozen["expected_tasks"]:
        raise DefinitionError("Original task coverage mismatch")
    results = []
    for key in manifest.task_key:
        leaf = root / "leaves" / key
        results.append(checked_leaf(leaf, frozen["scientific_hash"], key))
    frame = pd.DataFrame(results)
    frame.to_csv(root / "selected.csv", index=False)
    validation = {"status": "pass", "scientific_hash": frozen["scientific_hash"], "expected_tasks": len(manifest), "completed_tasks": len(results), "selected_hash": digest((root / "selected.csv").read_text())}
    write_json(root / "validation.json", validation)
    return validation
