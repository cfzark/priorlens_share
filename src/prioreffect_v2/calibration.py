"""Auditable CP adequacy thresholds from declared no-conflict calibration runs.

Statistic: per path min_{direction,m} A(m). Its quantile calibrates the chance
of having at least one adequate candidate, NOT pointwise coverage accuracy.
Independent no-conflict data replications must be supplied by the researcher.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .core import DefinitionError, digest, write_json


def context(task, location_fields):
    prior = copy.deepcopy(task["prior"])
    for field in location_fields:
        parts = field.split(".")
        current = prior
        for part in parts[:-1]:
            current = current[int(part)] if isinstance(current, list) else current[part]
        key = int(parts[-1]) if isinstance(current, list) else parts[-1]
        current[key] = "CALIBRATED_LOCATION_AXIS"
    method = {k: v for k, v in task["method"].items() if k not in {"id", "delta", "delta_artifact", "delta_provenance", "secondary"}}
    return {"model": task["model"], "n": len(task["data"]["y"]), "design": {k: v for k, v in task["data"].items() if k != "y"}, "prior_strength": prior,
            "baseline": task["baseline"], "target": task["target"], "method": method,
            "candidates": task["candidates"], "inference": task["inference"]}


def calibrate(roots, output, quantile=.95, location_fields=()):
    from .experiments import checked_leaf
    if not 0 < quantile < 1:
        raise DefinitionError("Calibration quantile must be in (0,1)")
    output = Path(output)
    if output.exists():
        raise DefinitionError("Calibration artifacts are immutable")
    groups, sources = {}, []
    for name in roots:
        root = Path(name).resolve()
        validation = json.loads((root / "validation.json").read_text())
        if validation["status"] != "pass":
            raise DefinitionError("Only validated calibration runs are accepted")
        for path in sorted((root / "tasks").glob("*.json")):
            task = json.loads(path.read_text())
            if task["method"]["name"] != "cp":
                continue
            if task["method"].get("delta") is not None:
                raise DefinitionError("Calibration input must be unconstrained CP, not an already thresholded run")
            leaf = root / "leaves" / task["task_key"]
            checked_leaf(leaf, validation["scientific_hash"], task["task_key"])
            curves = json.loads((leaf / "curves.json").read_text())
            signature = context(task, location_fields)
            group = groups.setdefault(digest(signature), {"context": signature, "statistics": [], "sources": []})
            for path_id in sorted({r["path"] for r in curves}):
                group["statistics"].append(min(r["adequacy"] for r in curves if r["path"] == path_id))
            group["sources"].append({"root": str(root), "task_key": task["task_key"], "replication_id": task["replication_id"], "curves_hash": digest(curves)})
        sources.append({"root": str(root), "validation": validation})
    if not groups:
        raise DefinitionError("No CP calibration paths found")
    for group in groups.values():
        group["delta"] = float(np.quantile(group["statistics"], quantile, method="higher"))
        group["n_paths"] = len(group["statistics"])
    artifact = {"schema_version": 1, "statistic": "pathwise_minimum_candidate_adequacy", "quantile": quantile,
                "quantile_method": "higher", "no_conflict_design": "explicitly_declared_by_user_not_inferred",
                "location_fields": list(location_fields), "groups": groups, "sources": sources,
                "warning": "Paths within a dataset are not independent data replications. Tiny smoke calibration is not a scientific delta estimate."}
    write_json(output, artifact)
    return artifact


def apply_calibration(task):
    path = task["method"].get("delta_artifact")
    if not path:
        return
    if task["method"].get("delta") is not None:
        raise DefinitionError("Choose fixed delta OR delta_artifact, not both")
    artifact = json.loads(Path(path).read_text())
    if artifact.get("statistic") != "pathwise_minimum_candidate_adequacy":
        raise DefinitionError("Unsupported CP calibration statistic")
    key = digest(context(task, artifact["location_fields"]))
    if key not in artifact["groups"]:
        raise DefinitionError("CP calibration does not match model/strength/baseline/target/grid/inference/path settings")
    task["method"] = {**task["method"], "delta": artifact["groups"][key]["delta"],
                      "delta_provenance": {"artifact_hash": digest(artifact), "group": key}}
