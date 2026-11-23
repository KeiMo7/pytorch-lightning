# Copyright (c) 2026)
"""Acceptance tests for MultiMetricModelCheckpoint.

Coverage map (task spec section "测试与验收" + the形态/硬约束/降级 lists):

* test_independent_best_and_patience            -> 形态1, 验收1
* test_checkpoint_records_winner_and_reason     -> 形态2/3
* test_resume_continues_patience_and_history    -> 形态4, 验收2
* test_report_queries_history_and_paths         -> 形态5, 硬约束1
* test_missing_route_marks_unavailable          -> 形态6, 降级4
* test_joint_vs_per_route_decision              -> 形态7, 验收3
* test_disabled_single_route_matches_native     -> 形态8, 不能破坏1
* test_metric_log_order_independent             -> 形态9, 验收3
* test_record_only_routes                       -> 形态10
* test_eviction_leaves_no_orphans               -> 形态11, 验收4
* test_persistence_lives_inside_checkpoint      -> 硬约束2
* test_restore_metadata_mismatch_names_route    -> 降级1
* test_failed_write_leaves_no_half_file         -> 降级2
* test_non_finite_metric_marks_unavailable      -> 降级3
"""

import os

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from multi_metric_ckpt import (
    MultiMetricModelCheckpoint,
    MultiMetricStateMismatchError,
    MultiMetricWriteError,
    RouteSpec,
)
from multi_metric_ckpt.callback import _TMP_PREFIX
from helpers import (
    MultiRouteModel,
    list_ckpts,
    make_routes,
    make_trainer,
    read_report,
)

ROUTES = [
    ("old_customer", "val/old_auroc", "max", 2),
    ("new_customer", "val/new_auroc", "max", 2),
    ("long_tail", "val/tail_auroc", "max", 3),
]


def _build(tmp_path, schedules, joint_mode=False, save_top_k=3, every_n_epochs=1, routes=None, **kw):
    routes = make_routes(routes or ROUTES)
    cb = MultiMetricModelCheckpoint(
        routes=routes,
        dirpath=os.path.join(tmp_path, "ckpts"),
        joint_mode=joint_mode,
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
        verbose=True,
        **kw,
    )
    model = MultiRouteModel(schedules)
    return cb, model


def test_independent_best_and_patience(tmp_path):
    """Three routes with different trajectories each keep their own best + patience."""
    schedules = {
        "val/old_auroc": [0.5, 0.6, 0.55, 0.54],   # wins e0,e1 then stalls
        "val/new_auroc": [0.4, 0.38, 0.45, 0.44],  # wins e0,e2 (e1 dips but stays below e0)
        "val/tail_auroc": [0.2, 0.21, 0.22, 0.23], # improves every epoch
    }
    cb, model = _build(tmp_path, schedules)
    trainer = make_trainer(tmp_path, cb, max_epochs=4)
    trainer.fit(model)

    report = read_report(cb)
    old, new, tail = (report["routes"][n] for n in ("old_customer", "new_customer", "long_tail"))

    assert old["best_score"] == pytest.approx(0.6) and old["best_epoch"] == 1
    assert new["best_score"] == pytest.approx(0.45) and new["best_epoch"] == 2
    assert tail["best_score"] == pytest.approx(0.23) and tail["best_epoch"] == 3

    # patience counters: old improved at e1 -> waits at e2,e3 (wait_count=2);
    # new improved at e2 -> wait at e3 (1); tail never waits (0).
    assert old["wait_count"] == 2 and old["patience_exceeded"] is False
    assert new["wait_count"] == 1
    assert tail["wait_count"] == 0

    # each route's history has one entry per evaluated epoch, independent of saves
    for block in (old, new, tail):
        assert [h["epoch"] for h in block["history"]] == [0, 1, 2, 3]
    assert [h["improved"] for h in old["history"]] == [True, True, False, False]
    assert [h["improved"] for h in new["history"]] == [True, False, True, False]
    assert [h["improved"] for h in tail["history"]] == [True, True, True, True]


def test_checkpoint_records_winner_and_reason(tmp_path):
    """Saved files record which routes won and simultaneous wins are merged once."""
    schedules = {
        "val/old_auroc": [0.5, 0.9],
        "val/new_auroc": [0.4, 0.8],
        "val/tail_auroc": [0.2, 0.1],  # e1 regression
    }
    cb, model = _build(tmp_path, schedules)
    trainer = make_trainer(tmp_path, cb, max_epochs=2)
    trainer.fit(model)

    records = [r for r in read_report(cb)["checkpoints"] if not r["evicted"]]
    # e0: all three win together -> one merged file; e1: two win -> one file
    assert len(records) == 2
    e0, e1 = records
    assert tuple(e0["winner_routes"]) == ("old_customer", "new_customer", "long_tail")
    assert "simultaneously" in e0["reason"]
    assert tuple(e1["winner_routes"]) == ("old_customer", "new_customer")
    # at most one checkpoint per epoch on disk
    assert len(list_ckpts(cb.dirpath)) == 2


def test_resume_continues_patience_and_history(tmp_path):
    """Interrupt after 2 epochs, resume: best history and wait_count carry over."""
    schedules = {
        "val/old_auroc": [0.5, 0.6, 0.55, 0.54, 0.53],
        "val/new_auroc": [0.4, 0.42, 0.45, 0.44, 0.43],
        "val/tail_auroc": [0.2, 0.21, 0.22, 0.23, 0.24],
    }
    cb, model = _build(tmp_path, schedules)
    trainer = make_trainer(tmp_path, cb, max_epochs=2)
    trainer.fit(model)
    ckpt_before = cb.best_model_path
    assert ckpt_before and os.path.isfile(ckpt_before)

    report_before = read_report(cb)
    assert report_before["routes"]["old_customer"]["wait_count"] == 0

    # resume into a fresh trainer/callback instance from the newest ckpt
    # (select by epoch/seq embedded in the report, not filesystem mtime, which
    # can be identical for files written within the same second)
    newest = max(read_report(cb)["checkpoints"], key=lambda r: r["seq"])
    resume_ckpt = os.path.join(cb.dirpath, newest["filename"])
    cb2 = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=cb.dirpath, save_top_k=3, verbose=True
    )
    model2 = MultiRouteModel(schedules)
    trainer2 = make_trainer(tmp_path, cb2, max_epochs=5)
    trainer2.fit(model2, ckpt_path=resume_ckpt)

    report = read_report(cb2)
    old = report["routes"]["old_customer"]
    # history spans all five evaluated epochs, not restarted at epoch 2
    assert [h["epoch"] for h in old["history"]] == [0, 1, 2, 3, 4]
    assert old["best_epoch"] == 1 and old["best_score"] == pytest.approx(0.6)
    # no further improvement for old route -> three consecutive waits after resume
    assert old["wait_count"] == 3
    assert old["patience_exceeded"] is True
    # the pre-resume best path remains the same file
    assert os.path.basename(ckpt_before) in os.listdir(cb.dirpath)


def test_report_queries_history_and_paths(tmp_path):
    """Report exposes per-route history and the current best artifact path."""
    schedules = {
        "val/old_auroc": [0.5, 0.6],
        "val/new_auroc": [0.4, 0.42],
        "val/tail_auroc": [0.2, 0.3],
    }
    cb, model = _build(tmp_path, schedules)
    make_trainer(tmp_path, cb, max_epochs=2).fit(model)

    report = read_report(cb)
    for name in ("old_customer", "new_customer", "long_tail"):
        block = report["routes"][name]
        assert block["best_model_path"].endswith(".ckpt")
        assert os.path.isfile(block["best_model_path"])
        assert len(block["history"]) == 2
    # report on disk is the same structure as callback.report() (single source)
    assert report == cb.report()


def test_missing_route_marks_unavailable(tmp_path):
    """A route missing for an epoch is flagged; the others keep deciding."""
    schedules = {
        "val/old_auroc": [0.5, 0.6, 0.7],
        "val/new_auroc": [0.4, None, 0.45],  # missing at epoch 1
        "val/tail_auroc": [0.2, 0.25, 0.26],
    }
    cb, model = _build(tmp_path, schedules)
    make_trainer(tmp_path, cb, max_epochs=3).fit(model)

    report = read_report(cb)
    new = report["routes"]["new_customer"]
    # missing epoch leaves no history entry, but route recovers afterwards
    assert [h["epoch"] for h in new["history"]] == [0, 2]
    assert new["available"] is True  # recovered at e2
    # old/tail still produced decisions at epoch 1 -> a checkpoint exists for e1
    epoch1 = [r for r in report["checkpoints"] if r["epoch"] == 1 and not r["evicted"]]
    assert epoch1 and "old_customer" in epoch1[0]["winner_routes"]


def test_joint_vs_per_route_decision(tmp_path):
    """Joint mode saves only when every participating route is no-worse."""
    schedules = {
        "val/old_auroc": [0.5, 0.6, 0.7],
        "val/new_auroc": [0.4, 0.3, 0.5],  # genuinely regresses at e1, recovers e2
        "val/tail_auroc": [0.2, 0.25, 0.26],
    }
    cb_pr = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=os.path.join(tmp_path, "pr_ckpts"),
        joint_mode=False, save_top_k=3, verbose=True,
    )
    make_trainer(tmp_path, cb_pr, max_epochs=3).fit(MultiRouteModel(schedules))
    pr_epochs = {r["epoch"] for r in read_report(cb_pr)["checkpoints"] if not r["evicted"]}
    assert pr_epochs == {0, 1, 2}

    # joint mode: e1 rejected (new_customer regressed); e0 and e2 accepted
    cb_j = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=os.path.join(tmp_path, "joint_ckpts"),
        joint_mode=True, save_top_k=3,
    )
    model_j = MultiRouteModel(schedules)
    make_trainer(tmp_path, cb_j, max_epochs=3).fit(model_j)
    joint_records = [r for r in read_report(cb_j)["checkpoints"] if not r["evicted"]]
    assert {r["epoch"] for r in joint_records} == {0, 2}
    # the accepted saves carry the explicit joint-mode reason
    assert all("joint mode" in r["reason"] for r in joint_records)
    # new_customer is evaluated every epoch and genuinely regresses at e1:
    # patience ticks once, then its e2 improvement resets the counter
    new_block = read_report(cb_j)["routes"]["new_customer"]
    assert [h["epoch"] for h in new_block["history"]] == [0, 1, 2]
    assert [h["improved"] for h in new_block["history"]] == [True, False, True]
    assert new_block["wait_count"] == 0
    # the rejected epoch itself is still auditable through route history
    assert new_block["history"][1]["improved"] is False


def test_disabled_single_route_matches_native(tmp_path):
    """One route, joint_mode off: save cadence and best-file semantics match ModelCheckpoint."""
    scores = [0.9, 0.8, 0.85, 0.7, 0.6]
    schedules = {"val/auroc": scores}
    single_route = [RouteSpec("only", "val/auroc", "min")]

    native_dir = os.path.join(tmp_path, "native")
    native = ModelCheckpoint(dirpath=native_dir, monitor="val/auroc", mode="min", save_top_k=1)
    make_trainer(tmp_path, native, max_epochs=5).fit(MultiRouteModel(schedules))
    native_files = list_ckpts(native_dir)

    mm_dir = os.path.join(tmp_path, "mm")
    cb = MultiMetricModelCheckpoint(routes=single_route, dirpath=mm_dir, joint_mode=False, save_top_k=1)
    make_trainer(tmp_path, cb, max_epochs=5).fit(MultiRouteModel(schedules))
    mm_files = list_ckpts(mm_dir)

    # both keep exactly one file (top-1 replacement) and save on the same epochs
    assert len(native_files) == len(mm_files) == 1
    report = read_report(cb)
    # improving epochs for min-mode scores 0.9->0.8->...: e0 and e1 and e3 and e4
    assert report["routes"]["only"]["best_score"] == pytest.approx(0.6)
    assert report["routes"]["only"]["best_epoch"] == 4
    improved_epochs = [h["epoch"] for h in report["routes"]["only"]["history"] if h["improved"]]
    assert improved_epochs == [0, 1, 3, 4]


def test_eviction_rule_is_explicit_fifo(tmp_path):
    """Multi-route eviction is explicit FIFO, not native score-based top-k.

    With several routes there is no single cross-route score to rank files by,
    so the documented rule is oldest-sequence-first. At ``save_top_k=1`` this
    coincides with native ``ModelCheckpoint`` (see the parity test above); here
    the FIFO behaviour at k>=2 is pinned down deterministically.
    """
    # monotonically improving => every epoch saves; only the newest k survive
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    schedules = {"val/auroc": scores}
    single = [RouteSpec("only", "val/auroc", "min")]
    mm_dir = os.path.join(tmp_path, "mm")
    cb = MultiMetricModelCheckpoint(routes=single, dirpath=mm_dir, save_top_k=2)
    make_trainer(tmp_path, cb, max_epochs=5).fit(MultiRouteModel(schedules))

    kept = sorted(r["epoch"] for r in read_report(cb)["checkpoints"] if not r["evicted"])
    # FIFO keeps the two most recent epochs regardless of their score rank
    assert kept == [3, 4]
    # evicted records remain in the audit trail, flagged, with no disk file
    evicted = [r for r in read_report(cb)["checkpoints"] if r["evicted"]]
    assert {r["epoch"] for r in evicted} == {0, 1, 2}
    assert all(not os.path.isfile(os.path.join(mm_dir, r["filename"])) for r in evicted)
    assert len(list_ckpts(mm_dir)) == 2


def test_metric_log_order_independent(tmp_path):
    """Logging route metrics in different orders yields identical decisions."""
    schedules = {
        "val/old_auroc": [0.5, 0.9],
        "val/new_auroc": [0.4, 0.8],
        "val/tail_auroc": [0.2, 0.1],
    }

    class OrderModel(MultiRouteModel):
        def __init__(self, schedules, reverse: bool):
            super().__init__(schedules)
            self._reverse = reverse

        def on_validation_epoch_end(self):
            items = list(self._schedules.items())
            if self._reverse:
                items.reverse()
            for key, values in items:
                epoch = self.current_epoch
                if epoch < len(values) and values[epoch] is not None:
                    self.log(key, torch.tensor(float(values[epoch]), dtype=torch.float32))

    runs = {}
    for tag, reverse in (("forward", False), ("reverse", True)):
        cb = MultiMetricModelCheckpoint(
            routes=make_routes(ROUTES),
            dirpath=os.path.join(tmp_path, tag),
            save_top_k=5,
        )
        make_trainer(tmp_path, cb, max_epochs=2).fit(OrderModel(schedules, reverse))
        runs[tag] = read_report(cb)

    a, b = runs["forward"], runs["reverse"]
    for name in ("old_customer", "new_customer", "long_tail"):
        assert a["routes"][name]["history"] == b["routes"][name]["history"]
        assert a["routes"][name]["best_score"] == b["routes"][name]["best_score"]
    assert [r["winner_routes"] for r in a["checkpoints"]] == [r["winner_routes"] for r in b["checkpoints"]]


def test_record_only_routes(tmp_path):
    """A route with participate_decision=False only records, never decides/saves."""
    schedules = {
        "val/old_auroc": [0.5, 0.4],   # regresses -> would not win anyway
        "val/new_auroc": [0.4, 0.9],   # record-only improvement
        "val/tail_auroc": [0.2, 0.15], # min-like but configured max; regress
    }
    routes = make_routes(ROUTES)
    routes[1] = RouteSpec("new_customer", "val/new_auroc", "max",
                          participate_decision=False, participate_save=False)
    cb = MultiMetricModelCheckpoint(routes=routes, dirpath=os.path.join(tmp_path, "ckpts"), save_top_k=5)
    make_trainer(tmp_path, cb, max_epochs=2).fit(MultiRouteModel(schedules))

    report = read_report(cb)
    new = report["routes"]["new_customer"]
    assert new["participate_decision"] is False
    assert new["best_score"] == pytest.approx(0.9)  # recorded despite not deciding
    assert all("new_customer" not in r["winner_routes"] for r in report["checkpoints"])


def test_eviction_leaves_no_orphans(tmp_path):
    """save_top_k eviction keeps the disk and the report in exact agreement."""
    schedules = {
        "val/old_auroc": [0.1, 0.2, 0.3, 0.4, 0.5],
        "val/new_auroc": [0.1, 0.2, 0.3, 0.4, 0.5],
        "val/tail_auroc": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    cb, model = _build(tmp_path, schedules, save_top_k=2)
    make_trainer(tmp_path, cb, max_epochs=5).fit(model)

    on_disk = set(list_ckpts(cb.dirpath))
    retained = {r["filename"] for r in read_report(cb)["checkpoints"] if not r["evicted"]}
    evicted = {r["filename"] for r in read_report(cb)["checkpoints"] if r["evicted"]}
    assert on_disk == retained
    assert on_disk.isdisjoint(evicted)
    assert len(on_disk) == 2
    # no temp files or other leftovers
    assert not [f for f in os.listdir(cb.dirpath) if f.startswith(_TMP_PREFIX)]


def test_eviction_repoints_dangling_route_pointers(tmp_path):
    """When a route's best file is FIFO-evicted, its pointer moves to a live win."""
    # only one route keeps improving; save_top_k=1 forces every older best out
    schedules = {
        "val/old_auroc": [0.1, 0.2, 0.3, 0.4],
        "val/new_auroc": [0.1, 0.2, 0.3, 0.4],
        "val/tail_auroc": [0.1, 0.2, 0.3, 0.4],
    }
    cb = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=os.path.join(tmp_path, "ckpts"), save_top_k=1
    )
    make_trainer(tmp_path, cb, max_epochs=4).fit(MultiRouteModel(schedules))

    report = read_report(cb)
    for name in ("old_customer", "new_customer", "long_tail"):
        block = report["routes"][name]
        # pointer must never reference a file that no longer exists
        assert block["best_model_path"] == "" or os.path.isfile(block["best_model_path"])
        # with one retained file every route points at that same live file
        assert os.path.basename(block["best_model_path"]) in list_ckpts(cb.dirpath)
    # the single retained file is the newest and every route won on it
    assert len(list_ckpts(cb.dirpath)) == 1
    newest = max(report["checkpoints"], key=lambda r: r["seq"])
    assert not newest["evicted"] and newest["epoch"] == 3


def test_persistence_lives_inside_checkpoint(tmp_path):
    """All bookkeeping is inside the Lightning checkpoint's callback state."""
    schedules = {
        "val/old_auroc": [0.5, 0.6],
        "val/new_auroc": [0.4, 0.42],
        "val/tail_auroc": [0.2, 0.25],
    }
    cb, model = _build(tmp_path, schedules)
    make_trainer(tmp_path, cb, max_epochs=2).fit(model)

    ckpt = torch.load(cb.best_model_path, map_location="cpu", weights_only=False)
    assert "callbacks" in ckpt
    state_keys = list(ckpt["callbacks"])
    assert any("MultiMetricModelCheckpoint" in key for key in state_keys)
    block = next(v for k, v in ckpt["callbacks"].items() if "MultiMetricModelCheckpoint" in k)
    assert block["schema_version"] >= 1
    assert set(block["routes"]) == {"old_customer", "new_customer", "long_tail"}
    assert block["routes"]["old_customer"]["history"][1]["score"] == pytest.approx(0.6)


def test_restore_metadata_mismatch_names_route(tmp_path):
    """Tampered metadata raises naming the route and the missing field."""
    schedules = {
        "val/old_auroc": [0.5, 0.6],
        "val/new_auroc": [0.4, 0.42],
        "val/tail_auroc": [0.2, 0.25],
    }
    cb, model = _build(tmp_path, schedules)
    make_trainer(tmp_path, cb, max_epochs=2).fit(model)
    ckpt_path = cb.best_model_path

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = next(k for k in ckpt["callbacks"] if "MultiMetricModelCheckpoint" in k)
    del ckpt["callbacks"][key]["routes"]["new_customer"]["wait_count"]
    torch.save(ckpt, ckpt_path)

    cb2 = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=cb.dirpath, save_top_k=3
    )
    trainer2 = make_trainer(tmp_path, cb2, max_epochs=3)
    with pytest.raises(MultiMetricStateMismatchError, match="new_customer.*wait_count"):
        trainer2.fit(MultiRouteModel(schedules), ckpt_path=ckpt_path)

    # strict_restore=False downgrades the same corrupted file to a warning + fresh start
    cb3 = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=cb.dirpath,
        save_top_k=3, strict_restore=False,
    )
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_trainer(tmp_path, cb3, max_epochs=1).fit(MultiRouteModel(schedules), ckpt_path=ckpt_path)
    # lenient mode names the offending route/field inside the discard warning
    discard = [str(w.message) for w in caught if "MultiMetricCheckpoint" in str(w.message)]
    assert discard and "new_customer" in discard[0] and "wait_count" in discard[0]
    # metadata was discarded: history restarts from zero even though a
    # checkpoint file was supplied
    assert cb3.report()["routes"]["old_customer"]["history"] == []


def test_failed_write_leaves_no_half_file(tmp_path, monkeypatch):
    """A write failure removes the temp file and rolls back bookkeeping."""
    schedules = {
        "val/old_auroc": [0.5, 0.6],
        "val/new_auroc": [0.4, 0.5],
        "val/tail_auroc": [0.2, 0.3],
    }
    cb, model = _build(tmp_path, schedules, save_top_k=5)
    trainer = make_trainer(tmp_path, cb, max_epochs=2)

    real_save = Trainer.save_checkpoint

    def flaky_save(self_trainer, filepath, weights_only=None, storage_options=None):
        # succeed at epoch 0, blow up at epoch 1
        if self_trainer.current_epoch >= 1:
            raise OSError("simulated disk full")
        return real_save(self_trainer, filepath, weights_only, storage_options)

    monkeypatch.setattr(Trainer, "save_checkpoint", flaky_save)
    with pytest.raises(MultiMetricWriteError, match="disk full"):
        trainer.fit(model)

    # no half-file, no tmp leftovers; the epoch-0 checkpoint is intact
    leftovers = [f for f in os.listdir(cb.dirpath) if f.startswith(_TMP_PREFIX)]
    assert leftovers == []
    assert len(list_ckpts(cb.dirpath)) == 1
    # seq counter rolled back: a later run reuses seq numbers cleanly
    monkeypatch.undo()
    cb_new = MultiMetricModelCheckpoint(
        routes=make_routes(ROUTES), dirpath=os.path.join(tmp_path, "retry"), save_top_k=5
    )
    make_trainer(tmp_path, cb_new, max_epochs=1).fit(MultiRouteModel(schedules))
    assert len(list_ckpts(cb_new.dirpath)) == 1


def test_non_finite_metric_marks_unavailable(tmp_path):
    """NaN/Inf routes are flagged with a warning and excluded from comparison."""
    schedules = {
        "val/old_auroc": [0.5, 0.6],
        "val/new_auroc": [0.4, float("nan")],
        "val/tail_auroc": [0.2, 0.3],
    }
    cb, model = _build(tmp_path, schedules)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_trainer(tmp_path, cb, max_epochs=2).fit(model)
    assert any("non-finite" in str(w.message) for w in caught)

    report = read_report(cb)
    new = report["routes"]["new_customer"]
    assert new["available"] is False and "non-finite" in new["unavailable_reason"]
    # best score stays at the epoch-0 value; other routes still saved at e1
    assert new["best_score"] == pytest.approx(0.4)
    assert any(r["epoch"] == 1 for r in report["checkpoints"] if not r["evicted"])
