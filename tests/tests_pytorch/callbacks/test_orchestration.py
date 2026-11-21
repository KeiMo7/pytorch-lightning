# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
import os
from unittest import mock

import pytest
import torch

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import (
    After,
    Before,
    EarlyStopping,
    ExclusiveResource,
    LearningRateFinder,
    ModelCheckpoint,
    MutuallyExclusive,
    with_constraints,
)
from lightning.pytorch.callbacks.orchestration import (
    CallbackSource,
    DanglingConstraintError,
    OrchestrationConflictError,
    OrchestrationCycleError,
    OrchestrationMismatchError,
    _capability_of,
    orchestration_fingerprint,
    orchestration_requested,
    resolve_callback_orchestration,
)
from lightning.pytorch.demos.boring_classes import BoringModel


def _named(name):
    """Create a callback class with the given qualname (its orchestration capability id)."""
    return type(name, (Callback,), {"__qualname__": name})


def _resolve(entries, extra=()):
    return resolve_callback_orchestration(entries, extra_constraints=extra)


# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------


def test_no_constraints_means_orchestration_not_requested():
    assert not orchestration_requested([Callback(), EarlyStopping("monitor")])
    assert not orchestration_requested([])


def test_with_constraints_does_not_change_callback_identity():
    callback = EarlyStopping("monitor")
    state_key_before = callback.state_key
    with_constraints(callback, Before("EarlyStopping", "ModelCheckpoint"))
    assert callback.state_key == state_key_before
    assert type(callback).__qualname__ == "EarlyStopping"
    assert _capability_of(callback) == "EarlyStopping"


def test_declare_via_class_attribute():
    class MyCallback(Callback):
        __qualname__ = "MyCallback"
        orchestration_constraints = (After("MyCallback", "Other"),)

    other = _named("Other")()
    mine = MyCallback()
    result = _resolve([(mine, CallbackSource.USER), (other, CallbackSource.USER)])
    assert result.report.order == ["Other", "MyCallback"]


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_hard_before_after_and_insertion_order_independence():
    a, b, c = _named("A")(), _named("B")(), _named("C")()
    with_constraints(a, Before("A", "B"))
    with_constraints(b, Before("B", "C"))
    orders = set()
    for perm in itertools.permutations([a, b, c]):
        result = _resolve([(cb, CallbackSource.USER) for cb in perm])
        orders.add(tuple(result.report.order))
    assert orders == {("A", "B", "C")}


def test_determinism_across_sources_and_shuffles():
    """The same set of participants and constraints resolves identically regardless of insertion order and of how
    sources are interleaved."""
    classes = [_named(f"Cb{i}") for i in range(30)]
    instances = [cls() for cls in classes]
    with_constraints(instances[5], After("Cb5", "Cb10"))
    with_constraints(instances[20], Before("Cb20", "Cb3"))
    sources = [CallbackSource.USER, CallbackSource.MODEL, CallbackSource.PLUGIN, CallbackSource.TRAINER]
    reference = None
    for seed in range(5):
        shuffled = list(instances)
        import random

        random.Random(seed).shuffle(shuffled)
        entries = [(cb, sources[(i + seed) % len(sources)]) for i, cb in enumerate(shuffled)]
        result = _resolve(entries)
        if reference is None:
            reference = result.report.order
        assert result.report.order == reference


def test_constraints_filled_by_explainable_tiebreak():
    a, b = _named("Zeta")(), _named("Alpha")()
    with_constraints(a, Before("Zeta", "Zeta"))  # self constraint is a no-op, forces the orchestrated path
    result = _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)])
    assert result.report.order == ["Alpha", "Zeta"]  # canonical tie-break: capability id ascending
    assert any("canonical tie-break" in rule for rule in result.report.inferred)


def test_soft_cycle_is_broken_and_reported():
    a, b = _named("A")(), _named("B")()
    extra = [Before("A", "B", hard=False), Before("B", "A", hard=False)]
    result = _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)], extra)
    assert len(result.callbacks) == 2
    assert len(result.report.relaxed) == 1
    assert "cycle" in result.report.relaxed[0]["reason"]


def test_legacy_implicit_rules_active_in_orchestrated_path():
    """The historical guarantees (tuner first, checkpoint last) are emulated as soft constraints."""
    checkpoint = ModelCheckpoint()
    tuner = LearningRateFinder()
    other = _named("Other")()
    with_constraints(other, Before("Other", "Other"))  # no-op, only to trigger orchestration
    result = _resolve(
        [(checkpoint, CallbackSource.TRAINER), (other, CallbackSource.USER), (tuner, CallbackSource.TRAINER)]
    )
    assert result.report.order[0] == "LearningRateFinder"
    assert result.report.order[-1] == "ModelCheckpoint"
    assert any(c["origin"] == "legacy:checkpoint-last" for c in result.report.satisfied)


def test_legacy_rules_can_be_overridden_by_hard_constraints():
    checkpoint = ModelCheckpoint()
    other = _named("Other")()
    with_constraints(other, After("Other", "ModelCheckpoint"))  # explicitly run after the checkpoint
    result = _resolve([(checkpoint, CallbackSource.TRAINER), (other, CallbackSource.USER)])
    assert result.report.order == ["ModelCheckpoint", "Other"]
    assert any(c["origin"] == "legacy:checkpoint-last" for c in result.report.relaxed)


# ---------------------------------------------------------------------------
# input boundaries
# ---------------------------------------------------------------------------


def test_empty_and_single_callback():
    assert _resolve([]).callbacks == []
    only = _named("Only")()
    result = _resolve([(only, CallbackSource.USER)])
    assert result.callbacks == [only]


def test_same_instance_registered_twice_is_deduplicated():
    instance = _named("Dup")()
    with_constraints(instance, Before("Dup", "Other"))
    other = _named("Other")()
    result = _resolve(
        [(instance, CallbackSource.USER), (instance, CallbackSource.PLUGIN), (other, CallbackSource.USER)]
    )
    assert result.report.order == ["Dup", "Other"]
    assert len(result.report.dropped) == 1
    assert "twice" in result.report.dropped[0]["reason"]


def test_same_capability_from_different_sources_resolves_by_precedence():
    user_instance, model_instance, plugin_instance = _named("Cap")(), _named("Cap")(), _named("Cap")()
    with_constraints(user_instance, Before("Cap", "Cap"))  # trigger orchestration
    entries = [
        (plugin_instance, CallbackSource.PLUGIN),
        (user_instance, CallbackSource.USER),
        (model_instance, CallbackSource.MODEL),
    ]
    result = _resolve(entries)
    assert result.callbacks == [model_instance]  # model source wins, independent of registration order
    assert len(result.report.dropped) == 2


def test_same_capability_same_source_gets_disambiguated():
    first, second = _named("Cap")(), _named("Cap")()
    trigger = _named("Trigger")()
    with_constraints(trigger, Before("Trigger", "Trigger"))  # no-op, only to trigger orchestration
    entries = [(first, CallbackSource.USER), (second, CallbackSource.USER), (trigger, CallbackSource.USER)]
    result = _resolve(entries)
    assert result.report.order[:2] == ["Cap", "Cap#2"]
    # referencing the ambiguous bare capability in a hard constraint fails loudly
    with pytest.raises(DanglingConstraintError, match="ambiguous"):
        _resolve(entries, [Before("Cap", "Cap#2")])


def test_stateful_and_stateless_mix_resolves():
    class Stateful(Callback):
        __qualname__ = "Stateful"

        def state_dict(self):
            return {"x": 1}

        def load_state_dict(self, state_dict):
            pass

    stateful, stateless = Stateful(), _named("Stateless")()
    with_constraints(stateless, After("Stateless", "Stateful"))
    result = _resolve([(stateless, CallbackSource.USER), (stateful, CallbackSource.USER)])
    assert result.report.order == ["Stateful", "Stateless"]


# ---------------------------------------------------------------------------
# pathological inputs
# ---------------------------------------------------------------------------


def test_hard_cycle_fails_with_participants_and_sources():
    a, b, c = _named("A")(), _named("B")(), _named("C")()
    extra = [Before("A", "B"), Before("B", "C"), Before("C", "A", origin="plugin")]
    entries = [(a, CallbackSource.USER), (b, CallbackSource.MODEL), (c, CallbackSource.PLUGIN)]
    with pytest.raises(OrchestrationCycleError) as excinfo:
        _resolve(entries, extra)
    message = str(excinfo.value)
    assert all(f"`{cap}`" in message for cap in "ABC")
    assert all(source in message for source in ("user", "model", "plugin"))


def test_dangling_hard_constraint_fails_at_startup():
    a = _named("A")()
    with_constraints(a, Before("A", "DoesNotExist"))
    with pytest.raises(DanglingConstraintError, match="DoesNotExist"):
        _resolve([(a, CallbackSource.USER)])


def test_dangling_soft_constraint_is_reported_not_fatal():
    a = _named("A")()
    with_constraints(a, Before("A", "DoesNotExist", hard=False))
    result = _resolve([(a, CallbackSource.USER)])
    assert result.callbacks == [a]
    assert len(result.report.relaxed) == 1


def test_contradictory_duplicate_constraints_fail():
    a, b = _named("A")(), _named("B")()
    extra = [Before("A", "B"), Before("B", "A")]  # the same constraint declared twice with opposite directions
    with pytest.raises(OrchestrationCycleError):
        _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)], extra)


def test_two_hard_last_constraints_conflict():
    a, b = _named("A")(), _named("B")()
    extra = [After("A", "*"), After("B", "*")]
    with pytest.raises(OrchestrationConflictError, match="last"):
        _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)], extra)


def test_hard_mutual_exclusion_conflict_reports_options():
    ema, swa = _named("EMA")(), _named("SWA")()
    extra = [MutuallyExclusive("EMA", "SWA")]
    with pytest.raises(OrchestrationConflictError) as excinfo:
        _resolve([(ema, CallbackSource.USER), (swa, CallbackSource.USER)], extra)
    assert "Resolution options" in str(excinfo.value)


def test_soft_mutual_exclusion_keeps_higher_precedence_source():
    ema, swa = _named("EMA")(), _named("SWA")()
    extra = [MutuallyExclusive("EMA", "SWA", hard=False)]
    entries = [(ema, CallbackSource.USER), (swa, CallbackSource.MODEL)]
    result = _resolve(entries, extra)
    assert result.callbacks == [swa]
    assert result.report.dropped[0]["capability"] == "EMA"


def test_exclusive_resource_hard_conflict():
    a, b = _named("A")(), _named("B")()
    extra = [ExclusiveResource("A", "gpu-memory"), ExclusiveResource("B", "gpu-memory")]
    with pytest.raises(OrchestrationConflictError, match="gpu-memory"):
        _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)], extra)


def test_exclusive_resource_soft_claim_loses_to_hard():
    a, b = _named("A")(), _named("B")()
    extra = [ExclusiveResource("A", "gpu-memory"), ExclusiveResource("B", "gpu-memory", hard=False)]
    result = _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)], extra)
    assert len(result.callbacks) == 2  # both still run; the soft claim is reported as relaxed
    assert any("gpu-memory" in c.get("reason", "") for c in result.report.relaxed)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_report_is_machine_readable_and_fingerprint_stable():
    a, b = _named("A")(), _named("B")()
    with_constraints(a, Before("A", "B"))
    result = _resolve([(b, CallbackSource.PLUGIN), (a, CallbackSource.USER)])
    report = result.report.as_dict()
    json.dumps(report)  # must be JSON-serializable for archiving
    assert report["order"] == ["A", "B"]
    assert report["sources"] == {"A": "user", "B": "plugin"}
    assert report["fingerprint"] == orchestration_fingerprint(["A", "B"])
    # same inputs, different insertion order -> identical fingerprint
    result2 = _resolve([(a, CallbackSource.USER), (b, CallbackSource.PLUGIN)])
    assert result2.report.fingerprint == result.report.fingerprint


def test_report_to_human_mentions_order_and_constraints():
    a, b = _named("A")(), _named("B")()
    with_constraints(a, Before("A", "B"))
    result = _resolve([(a, CallbackSource.USER), (b, CallbackSource.USER)])
    text = result.report.to_human()
    assert "final order" in text
    assert "A" in text
    assert "fingerprint" in text


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def test_trainer_without_constraints_keeps_legacy_order_and_stays_silent(tmp_path):
    callback_a, callback_b = _named("ZZZ")(), _named("AAA")()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[callback_a, callback_b],
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # legacy behavior: ModelCheckpoint pushed to the end, everything else in insertion order
    assert trainer.callbacks == [callback_a, callback_b, *trainer.callbacks[2:]]
    assert isinstance(trainer.callbacks[-1], ModelCheckpoint)
    assert trainer.callbacks[:2] == [callback_a, callback_b]
    assert trainer._callback_orchestration_report is None


def test_trainer_with_constraints_resolves_order(tmp_path):
    first, second = _named("First")(), _named("Second")()
    with_constraints(second, Before("Second", "First"))
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[first, second],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    assert trainer.callbacks == [second, first]
    assert trainer._callback_orchestration_report is not None
    assert trainer._callback_orchestration_report.order == ["Second", "First"]


def test_trainer_constraint_violation_fails_at_init(tmp_path):
    a, b = _named("A")(), _named("B")()
    with_constraints(a, Before("A", "B"))
    with_constraints(b, Before("B", "A"))
    with pytest.raises(OrchestrationCycleError):
        Trainer(
            default_root_dir=tmp_path,
            callbacks=[a, b],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )


def test_model_configure_callbacks_participate_in_orchestration(tmp_path):
    class ModelWithCallbacks(BoringModel):
        def configure_callbacks(self):
            callback = _named("FromModel")()
            with_constraints(callback, Before("FromModel", "FromUser"))
            return [callback]

    user_callback = _named("FromUser")()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[user_callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
    )
    model = ModelWithCallbacks()
    trainer.fit(model)
    names = [_capability_of(cb) for cb in trainer.callbacks]
    assert names.index("FromModel") < names.index("FromUser")


# ---------------------------------------------------------------------------
# checkpoint fingerprint and resume guard
# ---------------------------------------------------------------------------


def _fit_and_save(tmp_path, callbacks=()):
    checkpoint = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[*callbacks, checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    trainer.fit(BoringModel())
    return str(tmp_path / "last.ckpt")


def test_checkpoint_contains_orchestration_fingerprint(tmp_path):
    ckpt_path = _fit_and_save(tmp_path)
    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert "callback_orchestration" in checkpoint
    saved = checkpoint["callback_orchestration"]
    assert saved["fingerprint"] == orchestration_fingerprint(saved["order"])
    assert "ModelCheckpoint" in saved["order"]


def test_resume_with_matching_order_is_quiet(tmp_path):
    ckpt_path = _fit_and_save(tmp_path)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    with mock.patch(
        "lightning.pytorch.trainer.connectors.checkpoint_connector.rank_zero_warn"
    ) as warn_mock:
        trainer.fit(BoringModel(), ckpt_path=ckpt_path)
    for call in warn_mock.call_args_list:
        assert "orchestration" not in str(call)


def test_resume_with_reordered_callbacks_warns_and_reports_diff(tmp_path):
    first, second = _named("First")(), _named("Second")()
    ckpt_path = _fit_and_save(tmp_path, callbacks=[first, second])

    trainer = Trainer(
        default_root_dir=tmp_path,
        # same capabilities, different insertion order -> different legacy order -> fingerprint mismatch
        callbacks=[second, first, ModelCheckpoint(dirpath=tmp_path)],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    with mock.patch(
        "lightning.pytorch.trainer.connectors.checkpoint_connector.rank_zero_warn"
    ) as warn_mock:
        trainer.fit(BoringModel(), ckpt_path=ckpt_path)
    messages = [str(call) for call in warn_mock.call_args_list]
    assert any("orchestration fingerprint" in message for message in messages)
    # machine-readable diff is archived on the trainer
    mismatch = trainer._callback_orchestration_mismatch
    assert mismatch["expected_order"] != mismatch["actual_order"]
    assert set(mismatch["expected_order"]) == set(mismatch["actual_order"])


def test_resume_mismatch_strict_mode_raises(tmp_path):
    first, second = _named("First")(), _named("Second")()
    ckpt_path = _fit_and_save(tmp_path, callbacks=[first, second])
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[second, first, ModelCheckpoint(dirpath=tmp_path)],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    with mock.patch.dict(os.environ, {"PL_CALLBACK_ORCHESTRATION_STRICT": "1"}), pytest.raises(
        OrchestrationMismatchError, match="orchestration fingerprint"
    ):
        trainer.fit(BoringModel(), ckpt_path=ckpt_path)


def test_resume_mismatch_force_env_continues_without_warning(tmp_path):
    first, second = _named("First")(), _named("Second")()
    ckpt_path = _fit_and_save(tmp_path, callbacks=[first, second])
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[second, first, ModelCheckpoint(dirpath=tmp_path)],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    with mock.patch.dict(os.environ, {"PL_CALLBACK_ORCHESTRATION_FORCE": "1"}), mock.patch(
        "lightning.pytorch.trainer.connectors.checkpoint_connector.rank_zero_warn"
    ) as warn_mock:
        trainer.fit(BoringModel(), ckpt_path=ckpt_path)
    for call in warn_mock.call_args_list:
        assert "orchestration" not in str(call)


def test_resume_legacy_checkpoint_without_fingerprint_takes_compat_path(tmp_path):
    ckpt_path = _fit_and_save(tmp_path)
    # simulate a checkpoint saved before orchestration tracking existed
    checkpoint = torch.load(ckpt_path, weights_only=False)
    del checkpoint["callback_orchestration"]
    torch.save(checkpoint, ckpt_path)

    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
    )
    with mock.patch(
        "lightning.pytorch.trainer.connectors.checkpoint_connector.rank_zero_info"
    ) as info_mock:
        trainer.fit(BoringModel(), ckpt_path=ckpt_path)  # must not fail
    messages = [str(call) for call in info_mock.call_args_list]
    assert any("orchestration fingerprint" in message for message in messages)
