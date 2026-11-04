"""Small paired real-data CP region smoke, not a production submission.

Uses the standard V2 task/strict-merge contracts and fixed IS15. The three
methods share pseudo-data and fit seeds. No prior, historic grid or result
is modified. Only joint coefficients are tested here.
"""
import argparse
import json
from pathlib import Path

from prioreffect_v2.core import write_json
from prioreffect_v2.experiments import prepare, run_task_group, merge
from prioreffect_v2.models import lupus_data


def smoke_spec(output):
    common = dict(preset="cp_bootstrap_deletion_v2", paths=2, posterior_draws=32768,
                  path_group="paired_pdf_hdr_smoke", alphas=[.05], cache_regions=True)
    return dict(
        schema_version=1, run_mode="smoke", seed=20260312, seed_scheme="sha256_64_v2",
        model=dict(name="lupus_logistic"), data=lupus_data(),
        priors=dict(interest=dict(family="student_t", df=7, loc=[-1., 3.], scale=[10., 2.5]),
                    baseline=dict(family="student_t", df=7, loc=[0., 0.], scale=[10., 1000.])),
        prior_grid=[dict(prior_id="near", **{"loc.0": -1., "loc.1": 3.}),
                    dict(prior_id="far", **{"loc.0": 3., "loc.1": 11.})],
        targets=[dict(id="beta_joint", coordinates=[dict(variable="beta0"), dict(variable="beta1")])],
        candidates=dict(values=[0, 10, 20], meaning="number_deleted", unit="observation"),
        methods=[dict(common, id="central_H1", region="central", loss_family="historical", gap_weight=1.),
                 dict(common, id="pdf_hdr_H1", region="pdf_hdr", loss_family="historical", gap_weight=1.),
                 dict(common, id="pdf_hdr_D0_cap", region="pdf_hdr", loss_family="direct", gap_weight=0., cap_nominal=True)],
        inference=dict(backend="importance", posterior_reuse="none", sobol_power=15,
                       sobol_max_power=15, is_min_ess=500),
        output_root=str(Path(output).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New empty output root; never overwritten")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        parser.error("Fresh output root required")
    spec = smoke_spec(root)
    # Standard task definitions include source fingerprints and precise seeds.
    plan = prepare(spec)
    write_json(root / "input_spec.json", spec)
    run_task_group(plan["manifest"], range(plan["n_pending"]))
    certificate = merge(root)
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
