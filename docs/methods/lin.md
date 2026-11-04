# Lin: one-sided fresh-data matching

The observed baseline posterior P=q_b(theta|X_N) is fixed. Compare it with
Q_m=q_pi(theta|Y_(1:N-m)) over a signed integer grid in [-N,N].

## Presets

| Preset | Generating parameter |
|---|---|
| lin_ppd_v1 | One baseline observed-posterior draw per path |
| lin_plugin_v1 | Baseline natural-coordinate posterior means |

Generate a fresh vector of length 2N per path and retain nested prefixes.
The two variants share design draws and response-noise streams. For hierarchical
data, global mu,tau are shared within a path and each new school has its own
local effect.

## Objective and selection

Use central regions of nominal mass c=1-alpha. With reciprocal probabilities
c1=P(R_Q), c2=Q(R_P), capped D0 is

L(m) = [min(c1,c)-c]^2 + [min(c2,c)-c]^2.

Select m separately in each path, then average. Exact ties choose smallest
|m|, followed by negative m. m=N evaluates the informative prior; m=0 uses N
fresh rows; negative m uses more than N rows.

## Configuration

```yaml
methods:
  - preset: lin_ppd_v1
    id: lin_ppd
    paths: 2
    posterior_draws: 512
    alpha: 0.05
  - preset: lin_plugin_v1
    id: lin_plugin
    paths: 2
    posterior_draws: 512
    alpha: 0.05
candidates:
  meaning: informative_fresh_size_reduction
  unit: observation
  values: [-1, 0, 1]
replication_id: -1
```

Use unit: school for Eight Schools. In a mixed-method experiment, put this
candidate mapping inside each Lin method entry.

```bash
priorlens run --spec examples/nig/all_methods.yaml --output results/nig
priorlens run --spec examples/nig/lin_presets.yaml --output results/lin
```

Lin uses the same plan, run-task, run-task-group, resume and merge commands as
the other methods. Each target has an independent result leaf. Within a worker
group, the targets share candidate fits, baseline regions and a path bank.

## Computation

The numerical kernels preserve lin_nested_v1: seed base 20260312, with model,
replication and path identities. Set replication_id to distinguish datasets;
-1 is used by the real-data examples. The experiment seed controls the other
method families; Lin retains its fixed base.

The path bank prepares both generating laws and therefore uses finite baseline
means of the natural model parameters. Conjugate scalar targets use analytic
CDFs/PPFs. Joint targets use common natural-coordinate posterior draws.
The preset fixes central regions, capped D0 and the stated tie rule.

Supported models: normal_known, nig, eight_schools and lupus_logistic; targets
are population-parameter marginals/joints. See [models](../models.md) for the
full capability table.

## Compatibility

The older `priorlens lin --spec .../lin.yaml` command translates its wrapper
configuration into Lin presets and executes the same manifest workflow.
Results are now in selected.csv with per-task curves.json and paths.json.
