# Copyright (c) 2026
"""Multi-validation-set checkpoint callback.

Design contract (see the task specification):

* Decision, persistence and reporting share one in-memory state; the JSON
  report is a dump of that state, never a separate computation.
* All bookkeeping is returned from :meth:`state_dict`, so Lightning persists
  it inside every ``.ckpt`` and restores it on resume.
* The training loop and callback registration are untouched; this is a plain
  :class:`~lightning.pytorch.callbacks.Checkpoint` subclass registered like
  any other callback.
* No third-party dependency: only the stdlib plus Lightning/torch.
"""

import json
import logging
import os
import tempfile
from typing import Any, Iterable, Optional, Sequence, Union

import torch
from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import _is_local_file_protocol, get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from multi_metric_ckpt.exceptions import (
    MultiMetricStateMismatchError,
    MultiMetricWriteError,
)
from multi_metric_ckpt.routes import (
    STATE_SCHEMA_VERSION,
    CheckpointRecord,
    RouteSpec,
    RouteState,
    validate_state_dict_layout,
)

log = logging.getLogger(__name__)

REPORT_FILENAME = "multi_metric_report.json"
# Prefix for in-progress atomic writes; also the pattern cleaned up on setup so
# a crashed process never leaves an unrecognised half-file behind.
_TMP_PREFIX = ".mmckpt-tmp-"


class MultiMetricModelCheckpoint(Checkpoint):
    r"""Track one best checkpoint per validation route, with optional joint
    decisions.

    Args:
        routes: One or more :class:`RouteSpec` objects. For convenience a
            single route may also be given as ``(monitor, mode)`` /
            ``(monitor, mode, name)``. With exactly one route and
            ``joint_mode=False`` the observable save semantics match
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
            (``save_top_k=1``, epoch cadence).
        dirpath: Directory for checkpoints and the JSON report. Resolved like
            ``ModelCheckpoint`` when ``None`` (logger dir / default root).
        filename: Optional ``str.format``-style template. ``{epoch}``,
            ``{step}``, ``{seq}`` and every logged metric key are available.
            Defaults to ``"mmckpt-epoch{epoch}-step{step}-seq{seq}"``.
        joint_mode: When ``True`` a checkpoint is saved only when *every
            decision-participating, available* route is no worse than its
            historical best and at least one route strictly improves. When
            ``False`` (default) any winning route triggers a save; simultaneous
            wins are merged into one file by an explicit rule.
        save_top_k: Maximum number of retained checkpoints (``-1`` =
            unlimited). Eviction is explicit FIFO (oldest sequence first):
            with several routes there is no single cross-route score to rank
            files by, unlike the single-metric ``ModelCheckpoint`` which ranks
            by monitored score. At ``save_top_k=1`` the two coincide (the most
            recent best replaces the previous one). The file the run was
            resumed from is never evicted (matches Lightning semantics); routes
            whose best pointer referenced an evicted file are repointed to a
            retained winner or cleared, so a report path never dangles.
        every_n_epochs: A decision/save pass only runs for epochs
        ``(epoch + 1) % every_n_epochs == 0``. Default ``1``.
        save_weights_only: Forwarded to ``Trainer.save_checkpoint``.
        strict_restore: When ``True`` (default) a missing/mismatching metadata
            field raises :class:`MultiMetricStateMismatchError`. Set to
            ``False`` to accept a fresh start with a warning instead.
        verbose: Emit ``INFO`` log lines for every decision.

    Example::

        cb = MultiMetricModelCheckpoint(
            routes=[
                RouteSpec("old_customer", "val/old_auroc", "max", patience=3),
                RouteSpec("new_customer", "val/new_auroc", "max", patience=3),
                RouteSpec("long_tail", "val/tail_auroc", "max", patience=5),
            ],
            dirpath="ckpts/",
            joint_mode=False,
            save_top_k=3,
        )
        trainer = Trainer(callbacks=[cb])
    """

    FILE_EXTENSION = ".ckpt"

    def __init__(
        self,
        routes: Union[RouteSpec, Sequence[RouteSpec], Sequence[str], Sequence[tuple]],
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        joint_mode: bool = False,
        save_top_k: int = 1,
        every_n_epochs: int = 1,
        save_weights_only: bool = False,
        strict_restore: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        parsed = self._parse_routes(routes)
        if not parsed:
            raise ValueError("At least one validation route is required.")
        names = [r.name for r in parsed]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate route names are not allowed: {names}")
        monitors = [r.monitor for r in parsed]
        if len(set(monitors)) != len(monitors):
            raise ValueError(f"Each route must monitor a distinct metric key, got: {monitors}")
        if save_top_k == 0 or save_top_k < -1:
            raise ValueError("`save_top_k` must be -1 (unlimited) or a positive integer.")
        if every_n_epochs < 1:
            raise ValueError("`every_n_epochs` must be >= 1.")

        self.route_specs: tuple[RouteSpec, ...] = tuple(parsed)
        self.dirpath = os.path.realpath(os.path.expanduser(dirpath)) if dirpath and _is_local_file_protocol(str(dirpath)) else dirpath
        self.filename = filename
        self.joint_mode = bool(joint_mode)
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.save_weights_only = save_weights_only
        self.strict_restore = strict_restore
        self.verbose = verbose

        self._routes: dict[str, RouteState] = {r.name: RouteState(r) for r in parsed}
        # seq -> CheckpointRecord, insertion ordered; evicted records stay
        # flagged so the historical decision trail remains auditable.
        self._checkpoints: dict[int, CheckpointRecord] = {}
        self._next_seq = 0
        self._last_finalized_epoch: Optional[int] = None
        # Metrics observed during the current validation epoch, buffered so
        # that the order in which routes log their metrics cannot change the
        # decision. Keyed by route name.
        self._pending: dict[str, tuple[float, int]] = {}
        self._fs = get_filesystem(dirpath if dirpath else "")
        self._resume_ckpt_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Setup / properties
    # ------------------------------------------------------------------ #
    @property
    @override
    def state_key(self) -> str:
        # The key encodes the full route layout: resuming with a different
        # layout makes Lightning warn about the unmatched callback state, and
        # our own load_state_dict then reports precisely which route differs.
        return self._generate_state_key(
            joint_mode=self.joint_mode,
            routes=tuple((r.name, r.monitor, r.mode) for r in self.route_specs),
            every_n_epochs=self.every_n_epochs,
        )

    @property
    def report_path(self) -> str:
        return os.path.join(self.dirpath or "", REPORT_FILENAME)

    @property
    def best_model_path(self) -> str:
        """Path of the most recently produced retained checkpoint, or ``""``.

        In per-route mode prefer :meth:`best_path_for_route`.
        """
        retained = [r for r in self._checkpoints.values() if not r.evicted]
        if not retained:
            return ""
        return os.path.join(self.dirpath or "", retained[-1].filename)

    def best_path_for_route(self, route: str) -> str:
        """Path of the current best checkpoint recorded for ``route``."""
        state = self._routes[route]
        if state.best_checkpoint_seq is None:
            return ""
        record = self._checkpoints[state.best_checkpoint_seq]
        return os.path.join(self.dirpath or "", record.filename)

    def patience_exceeded(self, route: str) -> bool:
        state = self._routes[route]
        return state.spec.patience > 0 and state.wait_count > state.spec.patience

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self.dirpath is None:
            self.dirpath = self._resolve_ckpt_dir(trainer)
        self.dirpath = trainer.strategy.broadcast(self.dirpath)
        self._fs = get_filesystem(self.dirpath)
        if trainer.is_global_zero and stage == "fit":
            self._recover_orphan_temp_files()
            os.makedirs(self.dirpath, exist_ok=True)
            self._write_report()

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # ``setup`` runs before the checkpoint connector assigns
        # ``trainer.ckpt_path``; by fit start it is populated. Record it here so
        # eviction can protect the file this run was resumed from.
        if trainer.ckpt_path is not None:
            path = str(trainer.ckpt_path)
            self._resume_ckpt_path = os.path.realpath(path) if _is_local_file_protocol(path) else path
        if trainer.is_global_zero:
            # Report whatever state is current (fresh or restored in
            # load_state_dict). Disk reconciliation deliberately happens only
            # after metadata restore, never before, so pre-resume checkpoints
            # cannot be mistaken for orphans.
            self._write_report()

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip(trainer):
            return
        step = int(trainer.global_step)
        # Buffer every fresh, finite metric. ``callback_metrics`` retains the
        # previous validation's values, so freshness cannot be judged from the
        # value alone: the internal result collection flags a metric as
        # ``has_reset=True`` until it is re-logged in the current run.
        # Non-finite values mark the route unavailable instead of poisoning
        # comparisons.
        for state in self._routes.values():
            if not self._metric_is_fresh(trainer, state.spec.monitor):
                continue
            value = trainer.callback_metrics.get(state.spec.monitor)
            if value is None:
                continue
            tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
            if not torch.isfinite(tensor):
                state.available = False
                state.unavailable_reason = f"non-finite metric {state.spec.monitor}={float(tensor)} at step {step}"
                rank_zero_warn(
                    f"[MultiMetricCheckpoint] route {state.name!r} marked unavailable: {state.unavailable_reason}."
                    " It will not participate in this epoch's decision."
                )
                # A fresh NaN also invalidates any buffered value.
                self._pending.pop(state.name, None)
                state.last_seen_step = step
                state.last_seen_value = None
                continue
            numeric = float(tensor)
            state.last_seen_step = step
            state.last_seen_value = numeric
            self._pending[state.name] = (numeric, step)

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip(trainer):
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._finalize_epoch(trainer)

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Validation-only runs (no training epochs) may leave a buffered epoch.
        if self._pending and not self._should_skip(trainer):
            self._finalize_epoch(trainer)

    # ------------------------------------------------------------------ #
    # Decision core
    # ------------------------------------------------------------------ #
    def _finalize_epoch(self, trainer: "pl.Trainer") -> None:
        """Consume the buffered metrics of one epoch and produce <=1 save."""
        epoch = int(trainer.current_epoch)
        step = int(trainer.global_step)
        if self._last_finalized_epoch == epoch:
            # Defensive: multiple finalisation triggers in one epoch must be a
            # no-op so patience/history are never double-counted.
            self._pending.clear()
            return

        winners: list[str] = []
        record_only_updates: list[str] = []
        evaluated: list[str] = []

        for state in self._routes.values():
            fresh = self._pending.get(state.name)
            if fresh is None:
                # No fresh metric this epoch: missing route / empty loader /
                # skipped validation. Mark unavailable; other routes continue.
                if state.available:
                    state.available = False
                    state.unavailable_reason = f"metric {state.spec.monitor!r} missing at epoch {epoch}"
                continue
            value, value_step = fresh
            state.available = True
            state.unavailable_reason = None
            evaluated.append(state.name)

            is_win = self._is_improvement(state, value)
            # History records every evaluation, win or not.
            state.history.append(
                {
                    "epoch": epoch,
                    "step": value_step,
                    "score": value,
                    "best_before": state.best_score,
                    "improved": is_win,
                    "participates_decision": state.spec.participate_decision,
                }
            )
            if is_win:
                state.best_score = value
                state.best_epoch = epoch
                state.wait_count = 0
                if state.spec.participate_decision:
                    winners.append(state.name)
                else:
                    record_only_updates.append(state.name)
            else:
                state.wait_count += 1

        self._last_finalized_epoch = epoch
        self._pending.clear()

        if not evaluated:
            # Nothing comparable this epoch (all routes missing/non-finite).
            self._write_report()
            return

        decision_routes = [s for s in self._routes.values() if s.spec.participate_decision and s.available]

        if self.joint_mode:
            save_now, reason = self._joint_decision(decision_routes, winners, epoch)
            effective_winners = winners if save_now else []
        else:
            saving_winners = [
                name for name in winners if self._routes[name].spec.participate_save and self._routes[name].available
            ]
            save_now = bool(saving_winners)
            effective_winners = saving_winners
            if save_now:
                if len(saving_winners) > 1:
                    reason = (
                        "per-route mode: "
                        + ", ".join(saving_winners)
                        + " improved simultaneously; merged into one checkpoint, winner selected by route order"
                    )
                else:
                    reason = f"per-route mode: route {saving_winners[0]!r} improved over its historical best"
            else:
                reason = "no participating route improved"

        if record_only_updates:
            reason = (reason + "; " if reason else "") + (
                "record-only routes updated without save: " + ", ".join(record_only_updates)
            )

        if not save_now:
            if self.verbose:
                log.info("[MultiMetricCheckpoint] epoch %s: no checkpoint saved (%s).", epoch, reason)
            self._write_report()
            return

        # One file per epoch at most. The winner recorded on the file is the
        # first in *configured* route order — an explicit, stable tie-break,
        # independent of metric logging order within the epoch.
        winner_routes = tuple(sorted(effective_winners, key=lambda n: self._route_index(n)))
        scores = {name: self._routes[name].history[-1]["score"] for name in evaluated}
        seq = self._next_seq
        filename = self._format_filename(trainer, epoch, step, seq, scores)
        record = CheckpointRecord(
            seq=seq,
            filename=filename,
            epoch=epoch,
            step=step,
            scores=scores,
            winner_routes=winner_routes,
            reason=reason,
        )

        # Evict *before* writing, and register the incoming record first, so
        # the metadata captured inside the new file already reflects the
        # post-eviction disk including the incoming file itself: a checkpoint
        # never advertises a sibling as retained while it is about to be
        # deleted, and never omits itself. On write failure both the record and
        # any eviction are rolled back.
        evicted_this_round: list[CheckpointRecord] = []
        self._register_record(record)
        try:
            evicted_this_round = self._evict_if_needed(trainer, incoming=record)
            record = self._atomic_save(trainer, record=record)
        except BaseException:
            self._checkpoints.pop(record.seq, None)
            for victim in evicted_this_round:
                victim.evicted = False
            raise

        # The produced checkpoint becomes the "current best" artifact of every
        # winning route; non-winners keep pointing at their previous best.
        for name in winner_routes:
            self._routes[name].best_checkpoint_seq = record.seq
        # In joint mode a save also certifies the non-improving-but-equal
        # participating routes; their best pointer is refreshed only when they
        # have none yet.
        if self.joint_mode:
            for state in decision_routes:
                if state.best_checkpoint_seq is None:
                    state.best_checkpoint_seq = record.seq

        self._write_report()
        if self.verbose:
            log.info("[MultiMetricCheckpoint] epoch %s: saved %s (%s).", epoch, record.filename, reason)

    def _joint_decision(
        self, decision_routes: list[RouteState], winners: list[str], epoch: int
    ) -> tuple[bool, str]:
        """Joint rule: all participating+available routes no-worse, >=1 win."""
        if not decision_routes:
            return False, "joint mode: no participating route available"
        no_worse = all(
            state.best_score is None or not self._is_worse(state, state.history[-1]["score"])
            for state in decision_routes
        )
        any_win = bool(winners)
        if no_worse and any_win:
            return True, (
                "joint mode: every participating route is no-worse than its historical best and "
                + ", ".join(winners)
                + " improved"
            )
        worse = [s.name for s in decision_routes if s.best_score is not None and self._is_worse(s, s.history[-1]["score"])]
        if worse:
            return False, "joint mode: rejected because route(s) regressed: " + ", ".join(worse)
        return False, "joint mode: no route improved"

    def _is_improvement(self, state: RouteState, value: float) -> bool:
        if state.best_score is None:
            return True
        if state.spec.mode == "min":
            return value < state.best_score - state.spec.min_delta
        return value > state.best_score + state.spec.min_delta

    def _is_worse(self, state: RouteState, value: float) -> bool:
        """Strict regression vs. the historical best (equality is acceptable)."""
        if state.best_score is None:
            return False
        if state.spec.mode == "min":
            return value > state.best_score
        return value < state.best_score

    # ------------------------------------------------------------------ #
    # Atomic saving / eviction / orphan recovery
    # ------------------------------------------------------------------ #
    def _atomic_save(self, trainer: "pl.Trainer", record: CheckpointRecord) -> CheckpointRecord:
        """Write ``record`` atomically (temp file + ``os.replace``).

        The record is already registered by the caller so its metadata is part
        of the checkpoint being written. On failure the temp file is removed
        and the sequence counter is rolled back; the caller also rolls back the
        record registration and any eviction.
        """
        seq = record.seq
        self._next_seq = seq + 1
        filename = record.filename

        if not trainer.is_global_zero:
            # Non-zero ranks still participate in trainer.save_checkpoint's
            # collective below; the bookkeeping is identical on every rank.
            trainer.save_checkpoint(self._full_path(filename), self.save_weights_only)
            return record

        final_path = self._full_path(filename)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=self.FILE_EXTENSION, dir=self.dirpath)
        os.close(tmp_fd)
        try:
            # All ranks enter save_checkpoint; rank 0 writes the bytes into the
            # temp file first, then publishes via atomic os.replace.
            trainer.save_checkpoint(tmp_path, self.save_weights_only)
            os.replace(tmp_path, final_path)
        except BaseException as exc:  # includes KeyboardInterrupt
            self._safe_unlink(tmp_path)
            self._next_seq -= 1
            raise MultiMetricWriteError(
                f"Failed to atomically write checkpoint {final_path!r}; any partial file was removed. "
                f"Bookkeeping rolled back. Underlying error: {exc!r}"
            ) from exc
        return record

    def _register_record(self, record: CheckpointRecord) -> None:
        self._checkpoints[record.seq] = record

    def _evict_if_needed(
        self, trainer: "pl.Trainer", incoming: Optional[CheckpointRecord] = None
    ) -> list[CheckpointRecord]:
        """FIFO eviction down to ``save_top_k`` retained files.

        Explicit rule: oldest sequence number first. The checkpoint the run
        was resumed from is protected (same rule as Lightning's
        ``ModelCheckpoint``): it is never removed, and when it would otherwise
        be the FIFO victim, eviction skips it and removes the next-oldest
        *unprotected* file. The resume file itself counts toward the quota, so
        ``save_top_k`` is always satisfied and no orphan remains.

        The caller registers ``incoming`` before calling this, so the incoming
        record is already part of the retained pool and its metadata is
        consistent. Returns the records that were flagged and deleted, so a
        later write failure can roll them back.
        """
        evicted: list[CheckpointRecord] = []
        if self.save_top_k == -1:
            return evicted
        retained = [r for r in self._checkpoints.values() if not r.evicted]
        # Protected files are taken out of the candidate pool but keep their
        # quota slot; evict the oldest unprotected files until the pool fits.
        protected = [r for r in retained if self._is_resume_checkpoint(self._full_path(r.filename))]
        candidates = [r for r in retained if r not in protected]
        keep_unprotected = max(0, self.save_top_k - len(protected))
        # Keep the newest ``keep_unprotected``; evict the older remainder.
        # Guard against keep >= len(candidates) (no eviction in that case).
        if len(candidates) > keep_unprotected:
            to_evict = candidates[: len(candidates) - keep_unprotected]
        else:
            to_evict = []
        evicted_seqs = {r.seq for r in to_evict}
        for victim in to_evict:
            victim.evicted = True
            self._delete_file(trainer, self._full_path(victim.filename))
            evicted.append(victim)
            if self.verbose:
                log.info(
                    "[MultiMetricCheckpoint] evicted %s (FIFO, save_top_k=%s).",
                    victim.filename, self.save_top_k,
                )
        # Keep every route's best pointer live: repoint routes whose best
        # artifact was just evicted to their newest still-retained winning
        # checkpoint, or clear the pointer (never point at a missing file).
        if evicted_seqs:
            self._repoint_routes_after_eviction(evicted_seqs)
        return evicted

    def _repoint_routes_after_eviction(self, evicted_seqs: set[int]) -> None:
        """Move dangling route best pointers onto a retained checkpoint."""
        live_seqs = [r.seq for r in self._checkpoints.values() if not r.evicted]
        live_set = set(live_seqs)
        for state in self._routes.values():
            if state.best_checkpoint_seq not in evicted_seqs:
                continue
            # newest retained record this route won on (highest seq first)
            fallback = next(
                (seq for seq in reversed(live_seqs) if state.name in self._checkpoints[seq].winner_routes),
                None,
            )
            state.best_checkpoint_seq = fallback if fallback in live_set else None

    def _is_resume_checkpoint(self, path: str) -> bool:
        if self._resume_ckpt_path is None or not _is_local_file_protocol(str(path)):
            return False
        try:
            return os.path.realpath(path) == os.path.realpath(self._resume_ckpt_path)
        except OSError:
            return False

    def _recover_orphan_temp_files(self) -> None:
        """Remove half-written temp files left by an interrupted write."""
        if not self.dirpath or not os.path.isdir(self.dirpath):
            return
        for name in os.listdir(self.dirpath):
            if name.startswith(_TMP_PREFIX):
                path = os.path.join(self.dirpath, name)
                rank_zero_warn(f"[MultiMetricCheckpoint] removing unfinished temporary checkpoint {path!r}.")
                self._safe_unlink(path)

    def reconcile_disk(self) -> list[str]:
        """Cross-check bookkeeping against the filesystem; return orphan files.

        Removes ``.ckpt`` files in ``dirpath`` that no record references and
        flags records whose file vanished. Called automatically on
        :meth:`setup`; exposed for tests/operators.
        """
        orphans: list[str] = []
        if not self.dirpath or not os.path.isdir(self.dirpath):
            return orphans
        known = {r.filename for r in self._checkpoints.values() if not r.evicted}
        for name in os.listdir(self.dirpath):
            if not name.endswith(self.FILE_EXTENSION) or name.startswith(_TMP_PREFIX):
                continue
            if name not in known:
                orphans.append(os.path.join(self.dirpath, name))
        for path in orphans:
            self._safe_unlink(path)
        missing_seqs: set[int] = set()
        for record in self._checkpoints.values():
            if not record.evicted and not os.path.isfile(self._full_path(record.filename)):
                record.evicted = True
                missing_seqs.add(record.seq)
                rank_zero_warn(
                    f"[MultiMetricCheckpoint] recorded checkpoint {record.filename!r} is missing on disk;"
                    " marking it evicted in the report."
                )
        if missing_seqs:
            self._repoint_routes_after_eviction(missing_seqs)
        return orphans

    # ------------------------------------------------------------------ #
    # Persistence: callback state + JSON report (same source of truth)
    # ------------------------------------------------------------------ #
    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "joint_mode": self.joint_mode,
            "routes": {name: state.to_dict() for name, state in self._routes.items()},
            "checkpoints": [record.to_dict() for record in self._checkpoints.values()],
            "next_ckpt_seq": self._next_seq,
            "last_finalized_epoch": self._last_finalized_epoch,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        route_names = {r.name for r in self.route_specs}
        try:
            validate_state_dict_layout(state_dict, route_names)
        except MultiMetricStateMismatchError as exc:
            if self.strict_restore:
                raise
            rank_zero_warn(
                f"[MultiMetricCheckpoint] strict_restore=False: discarding incompatible metadata ({exc})."
            )
            return

        # Reject configuration drift explicitly instead of silently mixing.
        if state_dict["joint_mode"] != self.joint_mode:
            msg = (
                f"joint_mode changed: checkpoint has {state_dict['joint_mode']!r},"
                f" callback is configured with {self.joint_mode!r}."
            )
            if self.strict_restore:
                raise MultiMetricStateMismatchError(msg)
            rank_zero_warn(f"[MultiMetricCheckpoint] {msg} Discarding metadata.")
            return

        restored_routes: dict[str, RouteState] = {}
        for spec in self.route_specs:
            block = state_dict["routes"][spec.name]
            if block["monitor"] != spec.monitor or block["mode"] != spec.mode:
                msg = (
                    f"Route {spec.name!r}: monitor/mode drift (checkpoint: "
                    f"{block['monitor']!r}/{block['mode']!r}, now: {spec.monitor!r}/{spec.mode!r})."
                )
                if self.strict_restore:
                    raise MultiMetricStateMismatchError(msg)
                rank_zero_warn(f"[MultiMetricCheckpoint] {msg} Discarding metadata.")
                return
            restored_routes[spec.name] = RouteState.from_dict(block)

        self._routes = restored_routes
        self._checkpoints = {
            record["seq"]: CheckpointRecord.from_dict(record) for record in state_dict["checkpoints"]
        }
        self._next_seq = int(state_dict["next_ckpt_seq"])
        self._last_finalized_epoch = state_dict["last_finalized_epoch"]

        # Metadata vs. disk cross-check: the report must never claim a best
        # file that is not there. Records already flagged ``evicted`` are
        # retained only for the audit trail and are not expected on disk.
        if self.dirpath and _is_local_file_protocol(str(self.dirpath)):
            for record in self._checkpoints.values():
                if record.evicted:
                    continue
                if not os.path.isfile(self._full_path(record.filename)):
                    msg = (
                        f"Checkpoint record seq={record.seq} ({record.filename!r}) is referenced by metadata"
                        " but missing on disk."
                    )
                    if self.strict_restore:
                        raise MultiMetricStateMismatchError(msg)
                    rank_zero_warn(f"[MultiMetricCheckpoint] {msg}")
                    record.evicted = True
            self.reconcile_disk()

    def report(self) -> dict[str, Any]:
        """Structured report: per-route best history + current artifact path."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "dirpath": self.dirpath,
            "joint_mode": self.joint_mode,
            "save_top_k": self.save_top_k,
            "routes": {
                name: {
                    "monitor": state.spec.monitor,
                    "mode": state.spec.mode,
                    "available": state.available,
                    "unavailable_reason": state.unavailable_reason,
                    "participate_decision": state.spec.participate_decision,
                    "participate_save": state.spec.participate_save,
                    "patience": state.spec.patience,
                    "wait_count": state.wait_count,
                    "patience_exceeded": self.patience_exceeded(name),
                    "best_score": state.best_score,
                    "best_epoch": state.best_epoch,
                    "best_checkpoint_seq": state.best_checkpoint_seq,
                    "best_model_path": self.best_path_for_route(name),
                    "history": list(state.history),
                }
                for name, state in self._routes.items()
            },
            "checkpoints": [record.to_dict() for record in self._checkpoints.values()],
        }

    def _write_report(self) -> None:
        if not self.dirpath or not _is_local_file_protocol(str(self.dirpath)):
            return
        os.makedirs(self.dirpath, exist_ok=True)
        final_path = self.report_path
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".mmckpt-report-", suffix=".tmp", dir=self.dirpath)
        try:
            with os.fdopen(tmp_fd, "w") as fp:
                json.dump(self.report(), fp, indent=2, sort_keys=True)
            os.replace(tmp_path, final_path)
        except BaseException:
            self._safe_unlink(tmp_path)
            raise

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _should_skip(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)
            or trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
        )

    def _metric_is_fresh(self, trainer: "pl.Trainer", monitor: str) -> bool:
        """Return whether ``monitor`` was re-logged during the current run.

        ``trainer.callback_metrics`` keeps stale entries from earlier
        validation runs. The result collection marks every entry with
        ``has_reset=True`` once a validation run ends; entries that receive a
        new ``log()`` during the next run flip back to ``False``. Matching by
        the metric's metadata name keeps this independent of internal key
        prefixes (fx name, dataloader suffix).
        """
        results = getattr(trainer, "_results", None)
        if results is None:
            # Degraded path: unknown Trainer internals. Only trust a metric we
            # have never seen before at a strictly greater step.
            return monitor in trainer.callback_metrics
        for result_metric in results.values():
            meta = getattr(result_metric, "meta", None)
            if meta is None or getattr(meta, "name", None) != monitor:
                continue
            return not bool(getattr(result_metric, "has_reset", True))
        return False

    def _route_index(self, name: str) -> int:
        for i, spec in enumerate(self.route_specs):
            if spec.name == name:
                return i
        raise KeyError(name)

    def _full_path(self, filename: str) -> str:
        return os.path.join(self.dirpath or "", filename)

    def _format_filename(
        self, trainer: "pl.Trainer", epoch: int, step: int, seq: int, scores: dict[str, float]
    ) -> str:
        template = self.filename or "mmckpt-epoch{epoch}-step{step}-seq{seq}"
        fields: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "seq": seq,
        }
        # Metric names may contain "/"; expose both the raw key and a flattened
        # alias so filename templates stay filesystem-friendly.
        for key, value in trainer.callback_metrics.items():
            fields[key] = float(value) if isinstance(value, Tensor) else value
            fields.setdefault(key.replace("/", "_"), fields[key])
        for key, value in scores.items():
            fields.setdefault(key, value)
            fields.setdefault(key.replace("/", "_"), value)
        try:
            stem = template.format(**fields)
        except KeyError:
            # Unknown template placeholder: fall back rather than crash the run.
            stem = f"mmckpt-epoch{epoch}-step{step}-seq{seq}"
        return f"{stem}{self.FILE_EXTENSION}"

    def _resolve_ckpt_dir(self, trainer: "pl.Trainer") -> str:
        if len(trainer.loggers) > 0 and trainer.loggers[0].save_dir is not None:
            logger = trainer.loggers[0]
            version = logger.version if isinstance(logger.version, str) else f"version_{logger.version}"
            return os.path.join(logger.save_dir, str(logger.name), version, "checkpoints")
        return os.path.join(trainer.default_root_dir, "checkpoints")

    def _delete_file(self, trainer: "pl.Trainer", path: str) -> None:
        trainer.strategy.remove_checkpoint(path)

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            rank_zero_warn(f"[MultiMetricCheckpoint] could not remove {path!r}: {exc!r}")

    @staticmethod
    def _parse_routes(routes: Any) -> list[RouteSpec]:
        if isinstance(routes, RouteSpec):
            return [routes]
        if not isinstance(routes, Iterable):
            raise TypeError("`routes` must be a RouteSpec or a sequence of RouteSpec.")
        parsed: list[RouteSpec] = []
        for i, item in enumerate(routes):
            if isinstance(item, RouteSpec):
                parsed.append(item)
            elif isinstance(item, str):
                parsed.append(RouteSpec(name=f"route_{i}", monitor=item))
            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    monitor, mode = item
                    parsed.append(RouteSpec(name=f"route_{i}", monitor=monitor, mode=mode))
                elif len(item) == 3:
                    monitor, mode, name = item
                    parsed.append(RouteSpec(name=name, monitor=monitor, mode=mode))
                else:
                    raise ValueError(f"Route tuple must be (monitor, mode) or (monitor, mode, name), got {item!r}")
            else:
                raise TypeError(f"Unsupported route specification: {item!r}")
        return parsed
