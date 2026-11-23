# Copyright (c) 2026
"""Route specifications and per-route bookkeeping state.

Everything in this module is plain data: no ``torch`` dependency beyond what
``state_dict``/``load_state_dict`` round-tripping needs, so the structures can
be serialised both into the Lightning checkpoint (callback state) and into the
human-readable JSON report without duplication.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Mode = Literal["min", "max"]

# Schema version stored in every checkpoint and every report. Bump whenever the
# persisted layout below changes; ``load_state_dict`` refuses unknown versions
# instead of guessing.
STATE_SCHEMA_VERSION = 1

# Required top-level keys in the persisted callback state.
_STATE_KEYS = (
    "schema_version",
    "joint_mode",
    "routes",
    "checkpoints",
    "next_ckpt_seq",
    "last_finalized_epoch",
)

# Required keys inside a persisted per-route block.
_ROUTE_STATE_KEYS = (
    "name",
    "monitor",
    "mode",
    "participate_decision",
    "participate_save",
    "best_score",
    "best_epoch",
    "best_checkpoint_seq",
    "patience",
    "wait_count",
    "min_delta",
    "available",
    "unavailable_reason",
    "history",
    "last_seen_step",
    "last_seen_value",
)

# Required keys inside a persisted checkpoint record.
_CHECKPOINT_RECORD_KEYS = (
    "seq",
    "filename",
    "epoch",
    "step",
    "scores",
    "winner_routes",
    "reason",
    "evicted",
)


@dataclass(frozen=True)
class RouteSpec:
    """Description of one validation route (one validation set).

    Args:
        name: Stable identifier of the validation set, e.g. ``"old_customer"``.
            Used in file names, reports and error messages; never changes.
        monitor: Metric key logged through ``self.log(...)`` for this route.
        mode: ``"min"`` for losses, ``"max"`` for accuracies/AUROC/...
        min_delta: Minimum improvement over the historical best to count as a
            real win. Applied in the metric's natural direction.
        patience: Number of *evaluated* epochs without improvement before the
            route's patience counter reaches ``patience`` (mirrors
            ``EarlyStopping`` semantics; purely informational, never stops
            training on its own).
        participate_decision: When ``False`` the route is *record-only*: its
            best/history are maintained but it can never trigger a save and is
            ignored by joint decisions.
        participate_save: When ``False`` the route may still win a decision
            (and reset its own patience), but a checkpoint is not written on
            its behalf; a route that does not participate in the decision also
            cannot participate in saving.
    """

    name: str
    monitor: str
    mode: Mode = "min"
    min_delta: float = 0.0
    patience: int = 0
    participate_decision: bool = True
    participate_save: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("`RouteSpec.name` must be a non-empty string.")
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError(f"Route {self.name!r}: `monitor` must be a non-empty string.")
        if self.mode not in ("min", "max"):
            raise ValueError(f"Route {self.name!r}: `mode` must be 'min' or 'max', got {self.mode!r}.")
        if self.min_delta < 0:
            raise ValueError(f"Route {self.name!r}: `min_delta` must be >= 0, got {self.min_delta}.")
        if self.patience < 0:
            raise ValueError(f"Route {self.name!r}: `patience` must be >= 0, got {self.patience}.")
        if not self.participate_decision and self.participate_save:
            # A record-only route must not be able to cause a save indirectly.
            object.__setattr__(self, "participate_save", False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteState:
    """Mutable per-route bookkeeping. One instance per :class:`RouteSpec`."""

    spec: RouteSpec
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_checkpoint_seq: Optional[int] = None
    wait_count: int = 0
    # ``available`` flips to ``False`` when a metric is non-finite or the
    # validation set turns out to be missing/empty. It flips back to ``True``
    # automatically once a finite metric is observed again.
    available: bool = True
    unavailable_reason: Optional[str] = None
    # Best-score history: one entry per epoch where the route was evaluated.
    history: list[dict[str, Any]] = field(default_factory=list)
    # Global step at which the monitored metric was last observed; used to
    # tell a fresh metric apart from a stale value left in ``callback_metrics``.
    last_seen_step: Optional[int] = None
    # Value observed at ``last_seen_step``; identical (value, step) pairs seen
    # on a later validation run are stale leftovers, not fresh measurements.
    last_seen_value: Optional[float] = None

    @property
    def name(self) -> str:
        return self.spec.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "monitor": self.spec.monitor,
            "mode": self.spec.mode,
            "participate_decision": self.spec.participate_decision,
            "participate_save": self.spec.participate_save,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_checkpoint_seq": self.best_checkpoint_seq,
            "patience": self.spec.patience,
            "wait_count": self.wait_count,
            "min_delta": self.spec.min_delta,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "history": list(self.history),
            "last_seen_step": self.last_seen_step,
            "last_seen_value": self.last_seen_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteState":
        for key in _ROUTE_STATE_KEYS:
            if key not in data:
                # The caller (callback.load_state_dict) adds route context.
                raise KeyError(key)
        spec = RouteSpec(
            name=data["name"],
            monitor=data["monitor"],
            mode=data["mode"],
            min_delta=data["min_delta"],
            patience=data["patience"],
            participate_decision=data["participate_decision"],
            participate_save=data["participate_save"],
        )
        state = cls(spec=spec)
        state.best_score = data["best_score"]
        state.best_epoch = data["best_epoch"]
        state.best_checkpoint_seq = data["best_checkpoint_seq"]
        state.wait_count = data["wait_count"]
        state.available = data["available"]
        state.unavailable_reason = data["unavailable_reason"]
        state.history = list(data["history"])
        state.last_seen_step = data["last_seen_step"]
        state.last_seen_value = data["last_seen_value"]
        return state


@dataclass
class CheckpointRecord:
    """One retained checkpoint on disk and the decision that produced it."""

    seq: int
    filename: str
    epoch: int
    step: int
    scores: dict[str, float]
    winner_routes: tuple[str, ...]
    reason: str
    evicted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "filename": self.filename,
            "epoch": self.epoch,
            "step": self.step,
            "scores": dict(self.scores),
            "winner_routes": list(self.winner_routes),
            "reason": self.reason,
            "evicted": self.evicted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointRecord":
        for key in _CHECKPOINT_RECORD_KEYS:
            if key not in data:
                raise KeyError(key)
        return cls(
            seq=data["seq"],
            filename=data["filename"],
            epoch=data["epoch"],
            step=data["step"],
            scores=dict(data["scores"]),
            winner_routes=tuple(data["winner_routes"]),
            reason=data["reason"],
            evicted=data["evicted"],
        )


def validate_state_dict_layout(state: dict[str, Any], route_names: set[str]) -> None:
    """Validate restored callback metadata before any field is consumed.

    Raises :class:`MultiMetricStateMismatchError` naming the exact route and
    field that is missing, or the mismatched top-level structure. This is the
    "never silently restart from zero" guard.
    """
    from multi_metric_ckpt.exceptions import MultiMetricStateMismatchError

    missing = [key for key in _STATE_KEYS if key not in state]
    if missing:
        raise MultiMetricStateMismatchError(
            f"Multi-metric checkpoint metadata is invalid: missing top-level field(s) {missing}."
        )
    if state["schema_version"] != STATE_SCHEMA_VERSION:
        raise MultiMetricStateMismatchError(
            f"Multi-metric checkpoint metadata schema_version={state['schema_version']!r} is not supported"
            f" (this build supports {STATE_SCHEMA_VERSION})."
        )
    if not isinstance(state["routes"], dict):
        raise MultiMetricStateMismatchError("Restored field `routes` must be a mapping.")

    for name in route_names:
        block = state["routes"].get(name)
        if block is None:
            raise MultiMetricStateMismatchError(
                f"Route {name!r}: no metadata block found in the checkpoint (field `routes.{name}` missing)."
                " Refusing to restart this route from zero."
            )
        for key in _ROUTE_STATE_KEYS:
            if key not in block:
                raise MultiMetricStateMismatchError(
                    f"Route {name!r}: restored metadata is missing field `{key}`."
                )
        if block.get("name") != name:
            raise MultiMetricStateMismatchError(
                f"Route {name!r}: restored block carries mismatched name={block.get('name')!r}."
            )

    for raw in state["checkpoints"]:
        for key in _CHECKPOINT_RECORD_KEYS:
            if key not in raw:
                raise MultiMetricStateMismatchError(
                    f"Checkpoint record seq={raw.get('seq')!r}: missing field `{key}`."
                )
