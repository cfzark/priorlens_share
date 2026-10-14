import argparse
import json
from pathlib import Path
from .core import native
from .experiments import load_spec, resolved_tasks, prepare, run_task, run_task_group, merge


def main():
    parser = argparse.ArgumentParser(description="PriorEffect v2: explicit model/target/method composition")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["validate-spec", "plan", "run"]:
        p = sub.add_parser(command)
        p.add_argument("--spec", required=True)
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("run-task")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p = sub.add_parser("run-task-group")
    p.add_argument("--manifest", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-ids", type=int, nargs="+")
    group.add_argument("--group-index", type=int)
    p.add_argument("--group-size", type=int, default=1)
    p = sub.add_parser("merge")
    p.add_argument("--root", required=True)
    p = sub.add_parser("calibrate-cp")
    p.add_argument("--root", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantile", type=float, default=.95)
    p.add_argument("--location-field", action="append", default=[])
    p.add_argument("--declare-no-conflict", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "calibrate-cp":
        from .calibration import calibrate
        result = calibrate(args.root, args.output, args.quantile, args.location_field)
    elif args.command == "run-task":
        result = run_task(args.manifest, args.task_id)
    elif args.command == "run-task-group":
        ids = args.task_ids
        if ids is None:
            import pandas as pd
            if args.group_index < 0 or args.group_size < 1:
                parser.error("group-index must be nonnegative and group-size positive")
            rows = pd.read_csv(args.manifest, sep="\t")
            start = args.group_index * args.group_size
            ids = rows.task_id.iloc[start:start + args.group_size].tolist()
        result = run_task_group(args.manifest, ids)
    elif args.command == "merge":
        result = merge(args.root)
    else:
        spec = load_spec(args.spec)
        if args.command == "validate-spec":
            tasks = resolved_tasks(spec)
            result = [{k: t[k] for k in ["prior_id", "target", "method", "candidates", "inference"]} for t in tasks]
        else:
            result = prepare(spec, args.resume)
            if args.command == "run":
                if result["n_pending"] and (spec["inference"].get("school_cache_by_data") or spec["inference"].get("school_engine") == "batched"):
                    run_task_group(result["manifest"], range(result["n_pending"]))
                else:
                    for task_id in range(result["n_pending"]):
                        run_task(result["manifest"], task_id)
                result = merge(spec["output_root"])
    print(json.dumps(native(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
