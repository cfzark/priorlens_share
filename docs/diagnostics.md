# Results and diagnostics

## Standard output

All method presets, including Lin, use the same layout:

| File / directory | Contents |
|---|---|
| resolved_spec.json | Resolved configuration and numerical runtime fingerprint |
| manifest.tsv | Task IDs, keys and definition files |
| tasks/ | Individual task definitions |
| leaves/<task_key>/result.json | EPSS, target/method identity and summary diagnostics |
| leaves/<task_key>/curves.json | Candidate objective values |
| leaves/<task_key>/paths.json | Pathwise selections |
| leaves/<task_key>/posterior_accuracy.json | Posterior numerical diagnostics |
| leaves/<task_key>/done.json | Completed-leaf artifact hashes |
| selected.csv | Merged results |
| validation.json | Merge completeness and integrity |

R/W risk-summary additionally saves the full raw-risk bank and mean/median
reductions in its own output directory.

## Status and interpretation

| Field / status | Meaning |
|---|---|
| matched | A candidate was selected |
| unmatched | The CP absolute-adequacy rule had no eligible candidate |
| inadmissible | A requested posterior is improper |
| undefined_moment | A loss requires a moment that does not exist |
| numerical_failure | The posterior or loss calculation failed its numerical checks |
| boundary_rate | Fraction of selections on the candidate boundary |
| match_rate | Fraction of paths satisfying the selection rule |
| path_checksum | Identity of the generated path bank |
| tie_count / runner_up_gap | Exact ties and separation from the next candidate |

With a CP adequacy gate, EPSS averages the matched paths and match_rate gives
their fraction. Lin records both selected cross-coverages, tie rate and the
mean runner-up gap.

Posterior accuracy, matching adequacy and file integrity have separate outputs.
IS weight ESS measures sampler efficiency; EPSS measures the prior comparison.
Between-dataset SD describes outer-replication variation; within-path summaries
describe internal Monte Carlo variation.

## Resume

```bash
priorlens run --spec examples/nig/all_methods.yaml --output results/nig --resume
```

The runner checks completed leaves and schedules missing tasks from the same
configuration. Failed attempts remain recorded under each leaf's failures/.
Completed merged experiments use a new output directory for subsequent runs.

## Numerical review

For a few fixed observed and candidate datasets, compare moments, quantiles,
cross-region probabilities and distances across numerical budgets. Replay the
same paths to measure selection changes. [Posterior computation](posterior.md)
lists the backend-specific diagnostics; [configuration](configuration.md)
defines data units and target coordinates.
