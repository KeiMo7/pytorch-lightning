# Multi-validation-set checkpoint selection

`MultiMetricModelCheckpoint` is a side-mounted Lightning `Checkpoint` callback
that maintains an independent "best checkpoint" decision for **several
validation sets** (e.g. old customers / new customers / long-tail classes)
instead of forcing one number to represent all of them.

It lives outside `lightning.pytorch`: the training loop, callback registration
and the existing `ModelCheckpoint` are untouched. No new dependency is added
(stdlib + Lightning/torch only).

## Usage

```python
from lightning.pytorch import Trainer
from multi_metric_ckpt import MultiMetricModelCheckpoint, RouteSpec

cb = MultiMetricModelCheckpoint(
    routes=[
        RouteSpec("old_customer", "val/old_auroc", mode="max", patience=2),
        RouteSpec("new_customer", "val/new_auroc", mode="max", patience=2),
        RouteSpec("long_tail",    "val/tail_auroc", mode="max", patience=3),
    ],
    dirpath="checkpoints/",
    joint_mode=False,   # True => save only when every route is no-worse
    save_top_k=3,
)
Trainer(callbacks=[cb]).fit(model)
```

Each route just needs its metric logged with `self.log(monitor, value)` during
validation, exactly as for `ModelCheckpoint`.

## How the requirements map to the implementation

| Requirement | Where it lives |
| --- | --- |
| Independent best record + patience per route | `RouteState` in `routes.py`; one state per `RouteSpec` |
| Record which route/epoch/metric won on each file | `CheckpointRecord.winner_routes` / `reason` |
| Simultaneous wins merged into one file, with a reason | `_finalize_epoch` (per-route mode), tie-break by configured route order |
| Best history + patience survive resume | full state returned from `state_dict()`, restored in `load_state_dict()` |
| Per-route history + current artifact path in a report | `report()` and `multi_metric_report.json` (same in-memory state) |
| Missing/empty route flagged unavailable, others decide | freshness check in `on_validation_end`; route marked `available=False` |
| Joint decision (all routes no-worse + >=1 improvement) | `joint_mode=True`, `_joint_decision` |
| Single-metric behaviour unchanged when the feature is off | native `ModelCheckpoint` is not modified; parity tested at `save_top_k=1` |
| Logging order independence | metrics are buffered in `_pending` and consumed in configured route order |
| Per-route decision/save opt-out (`participate_decision`/`participate_save`) | `RouteSpec`; record-only routes never decide or save |
| Bounded retained files, explicit eviction, no orphans | FIFO `_evict_if_needed`; temp-file cleanup; `reconcile_disk` |

### Hard guarantees

* **One source of truth.** Decision, saving and the JSON report all read/write
  the same `_routes` / `_checkpoints` state. A report path never points at a
  missing file: when a best file is FIFO-evicted (or found missing on resume),
  the route pointer is repointed to a retained winner or cleared.
* **Metadata rides inside the checkpoint.** Everything needed to resume
  (`schema_version`, per-route best/history/patience, checkpoint records,
  sequence counter) is in the Lightning checkpoint under
  `checkpoint["callbacks"][<state_key>]`, never only in memory or logs.
* **Atomic writes.** Bytes go to a temp file then `os.replace`; a failure
  removes the temp file, rolls back the sequence counter, the record and any
  eviction, and raises `MultiMetricWriteError`. Later epochs recognise the
  state instead of believing a phantom checkpoint exists.
* **Strict restore.** A schema mismatch or missing field raises
  `MultiMetricStateMismatchError` naming the route and field; it never silently
  restarts from zero. `strict_restore=False` downgrades that to a warning.
* **Non-finite metrics** mark the route unavailable for that epoch (with a
  warning) and exclude it from comparison; the other routes continue.

### Deliberate design choice: FIFO vs. score-based eviction

The single-metric `ModelCheckpoint` ranks retained files by the monitored
score. With several routes there is no single comparable score, so eviction is
**explicit FIFO** (oldest sequence first), documented in the `save_top_k`
docstring and pinned by `test_eviction_rule_is_explicit_fifo`. At
`save_top_k=1` the two rules coincide. The checkpoint resumed from is
protected from eviction.

## Layout

```
multi_metric_ckpt/
  __init__.py     public API
  exceptions.py   MultiMetricCheckpointError hierarchy
  routes.py       RouteSpec, RouteState, CheckpointRecord, schema validation
  callback.py     MultiMetricModelCheckpoint (hooks, decisions, IO, report)
tests/multi_metric/
  helpers.py      MultiRouteModel + trainer/report helpers
  test_multi_metric_checkpoint.py   16 acceptance tests
output/
  demo_e2e.py     end-to-end run incl. interrupt/resume
  e2e_run/        produced checkpoints + JSON report
```

## Run

```bash
# acceptance tests
PYTHONPATH=src:. python -m pytest tests/multi_metric -o addopts=""

# end-to-end demo (writes output/e2e_run/)
PYTHONPATH=src:. python output/demo_e2e.py
```
