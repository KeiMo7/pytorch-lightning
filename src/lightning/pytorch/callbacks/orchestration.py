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
"""Declarative orchestration for the Trainer's callback list.

This module lets users and callback authors declare *ordering constraints* between callbacks (``before``/``after``,
mutual exclusion, exclusive resources) with a hard/soft severity, and resolves them into a single deterministic
execution order.

Design guarantees:

- **Determinism**: the resolved order is a pure function of the set of participants (capability id + source) and the
  declared constraints. It does not depend on insertion order, on the source a callback came from, or on the number
  of processes. Unconstrained degrees of freedom are filled by a canonical tie-break (capability id, ascending).
- **Backward compatibility**: when no callback declares constraints and no external constraints are passed, the
  resolver is not used at all and the Trainer keeps the legacy behavior (tuner callbacks first, checkpoint callbacks
  last, everything else in insertion order) with no extra output.
- **Auditability**: the resolution produces an :class:`OrchestrationReport` (human-readable and machine-readable)
  including the final order, which constraints were satisfied or relaxed, and which rules were inferred. A
  fingerprint of the final order is stored in checkpoints and verified on resume.

"""

import hashlib
import heapq
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)

#: Target that matches every other participant, used to express "first"/"last" positions.
ALL = "*"


class CallbackSource(Enum):
    """Origin of a callback in the merged Trainer callback list."""

    #: Injected by the Trainer itself (default ModelCheckpoint, progress bar, Timer, ModelSummary, ...)
    TRAINER = "trainer"
    #: Explicitly passed by the user via ``Trainer(callbacks=[...])``
    USER = "user"
    #: Returned by ``LightningModule.configure_callbacks``
    MODEL = "model"
    #: Registered by external packages through the ``lightning.pytorch.callbacks_factory`` entry point
    PLUGIN = "plugin"


#: Lower value wins when two sources provide the same capability. This mirrors the historical merge semantics
#: (model callbacks override trainer callbacks, user callbacks prevent defaults, plugins are appended last) but makes
#: the precedence explicit and independent of registration order.
_SOURCE_PRECEDENCE = {
    CallbackSource.MODEL: 0,
    CallbackSource.USER: 1,
    CallbackSource.PLUGIN: 2,
    CallbackSource.TRAINER: 3,
}


class OrchestrationError(MisconfigurationException):
    """Base class for callback orchestration failures. Raised before training starts."""


class OrchestrationCycleError(OrchestrationError):
    """A set of hard ordering constraints forms a cycle."""


class DanglingConstraintError(OrchestrationError):
    """A hard constraint references a capability that is not part of the run."""


class OrchestrationConflictError(OrchestrationError):
    """Hard constraints are mutually unsatisfiable (e.g. two capabilities must both run last)."""


class OrchestrationMismatchError(OrchestrationError):
    """The resolved callback order does not match the fingerprint stored in a resumed checkpoint."""


@dataclass(frozen=True)
class OrchestrationConstraint:
    """A single declarative constraint between callback capabilities.

    Args:
        kind: One of ``"before"``, ``"after"``, ``"mutually_exclusive"``, ``"exclusive_resource"``.
        subject: Capability id the constraint applies to.
        target: For ``before``/``after``: another capability id or ``"*"`` (everything). For
            ``mutually_exclusive``: the other capability id. For ``exclusive_resource``: the resource name.
        hard: Hard constraints must be satisfied or resolution fails. Soft constraints are best-effort: they are
            dropped (and reported) when they cannot be satisfied.
        origin: Free-form label of who declared the constraint, used in error messages and the report.

    """

    kind: str
    subject: str
    target: str
    hard: bool = True
    origin: str = "user"

    def describe(self) -> str:
        level = "hard" if self.hard else "soft"
        if self.kind == "exclusive_resource":
            return f"[{level}] `{self.subject}` exclusively owns resource `{self.target}` (declared by {self.origin})"
        if self.kind == "mutually_exclusive":
            return f"[{level}] `{self.subject}` and `{self.target}` are mutually exclusive (declared by {self.origin})"
        return f"[{level}] `{self.subject}` must run {self.kind} `{self.target}` (declared by {self.origin})"


def Before(subject: str, target: str, hard: bool = True, origin: str = "user") -> OrchestrationConstraint:
    """Declare that ``subject`` must run before ``target`` (``"*"`` = everything else)."""
    return OrchestrationConstraint("before", subject, target, hard=hard, origin=origin)


def After(subject: str, target: str, hard: bool = True, origin: str = "user") -> OrchestrationConstraint:
    """Declare that ``subject`` must run after ``target`` (``"*"`` = everything else)."""
    return OrchestrationConstraint("after", subject, target, hard=hard, origin=origin)


def MutuallyExclusive(first: str, second: str, hard: bool = True, origin: str = "user") -> OrchestrationConstraint:
    """Declare that only one of the two capabilities may stay in the run."""
    return OrchestrationConstraint("mutually_exclusive", first, second, hard=hard, origin=origin)


def ExclusiveResource(subject: str, resource: str, hard: bool = True, origin: str = "user") -> OrchestrationConstraint:
    """Declare that ``subject`` exclusively owns a resource (e.g. ``"gpu-memory"``)."""
    return OrchestrationConstraint("exclusive_resource", subject, resource, hard=hard, origin=origin)


def with_constraints(callback: Callback, *constraints: OrchestrationConstraint) -> Callback:
    """Attach orchestration constraints to a callback instance without modifying its class.

    This is the escape hatch for expressing constraints on callbacks whose source code cannot or should not be
    changed (third-party callbacks, legacy code). The callback's identity (``state_key``, class name, log prefixes)
    is untouched.

    """
    existing = tuple(getattr(callback, "orchestration_constraints", ()) or ())
    callback.orchestration_constraints = existing + tuple(constraints)  # type: ignore[attr-defined]
    return callback


def _capability_of(callback: Callback) -> str:
    """The orchestration capability id of a callback.

    Defaults to the class qualname (mirroring the default ``state_key``). A callback class may define
    ``orchestration_capability`` to share a capability id across related classes. This never modifies the callback's
    persisted identity.

    """
    capability = getattr(callback, "orchestration_capability", None)
    if isinstance(capability, str) and capability:
        return capability
    return type(callback).__qualname__


def _declared_constraints_of(callback: Callback) -> tuple[OrchestrationConstraint, ...]:
    constraints = getattr(callback, "orchestration_constraints", None)
    if not constraints:
        return ()
    return tuple(constraints)


def orchestration_requested(
    callbacks: Iterable[Callback], extra_constraints: Iterable[OrchestrationConstraint] = ()
) -> bool:
    """Whether any constraint is declared for this set of callbacks.

    When ``False``, the Trainer uses the legacy reordering path and the orchestration machinery stays completely
    silent.

    """
    if any(True for _ in extra_constraints):
        return True
    return any(_declared_constraints_of(cb) for cb in callbacks)


def orchestration_fingerprint(order: Iterable[str]) -> str:
    """A stable fingerprint of an ordered list of capability ids."""
    payload = json.dumps(list(order), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def orchestration_order_diff(expected: Iterable[str], actual: Iterable[str]) -> str:
    """A human-readable diff between the orchestration stored in a checkpoint and the one resolved for this run."""
    expected, actual = list(expected), list(actual)
    lines = [
        f"  expected order (from checkpoint): {expected}",
        f"  actual order (this run):          {actual}",
    ]
    expected_pos = {capability: i for i, capability in enumerate(expected)}
    actual_pos = {capability: i for i, capability in enumerate(actual)}
    moved = [c for c in actual if c in expected_pos and expected_pos[c] != actual_pos[c]]
    if moved:
        lines.append(f"  capabilities at a different position: {moved}")
    missing = [c for c in expected if c not in actual_pos]
    if missing:
        lines.append(f"  capabilities present at save time but missing now: {missing}")
    added = [c for c in actual if c not in expected_pos]
    if added:
        lines.append(f"  capabilities added since the checkpoint was saved: {added}")
    return "\n".join(lines)


def _orchestration_env_flag(name: str) -> bool:
    """Whether a boolean orchestration toggle is enabled in the environment."""
    import os

    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


@dataclass
class _Participant:
    callback: Callback
    source: CallbackSource
    capability: str
    constraints: list[OrchestrationConstraint] = field(default_factory=list)


@dataclass
class OrchestrationReport:
    """The auditable outcome of a resolution.

    Attributes:
        order: Capability ids in final execution order.
        sources: Mapping of capability id -> source (``user``/``model``/``plugin``/``trainer``).
        satisfied: Machine-readable list of constraints that hold in the final order.
        relaxed: Machine-readable list of soft constraints that could not be honored, with the reason.
        inferred: Human-readable list of fill-in rules the resolver applied.
        dropped: Callbacks removed during resolution (duplicate instances, capability collisions, mutual
            exclusion), with the reason.
        fingerprint: Fingerprint of ``order``; stored in checkpoints and verified on resume.

    """

    order: list[str]
    sources: dict[str, str]
    satisfied: list[dict[str, Any]]
    relaxed: list[dict[str, Any]]
    inferred: list[str]
    dropped: list[dict[str, Any]]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable version, suitable for archiving alongside the run."""
        return {
            "order": list(self.order),
            "sources": dict(self.sources),
            "satisfied": [dict(c) for c in self.satisfied],
            "relaxed": [dict(c) for c in self.relaxed],
            "inferred": list(self.inferred),
            "dropped": [dict(d) for d in self.dropped],
            "fingerprint": self.fingerprint,
        }

    def to_human(self) -> str:
        """Human-readable summary of the resolved orchestration."""
        lines = ["Callback orchestration resolved:", "  final order:"]
        for i, capability in enumerate(self.order):
            lines.append(f"    {i + 1}. {capability} (source: {self.sources.get(capability, '?')})")
        if self.satisfied:
            lines.append("  satisfied constraints:")
            lines.extend(f"    - {c['description']}" for c in self.satisfied)
        if self.relaxed:
            lines.append("  relaxed (soft, not honored) constraints:")
            lines.extend(f"    - {c['description']} -- {c['reason']}" for c in self.relaxed)
        if self.dropped:
            lines.append("  dropped callbacks:")
            lines.extend(f"    - {d['capability']} (source: {d['source']}) -- {d['reason']}" for d in self.dropped)
        if self.inferred:
            lines.append("  inferred rules:")
            lines.extend(f"    - {rule}" for rule in self.inferred)
        lines.append(f"  fingerprint: {self.fingerprint}")
        return "\n".join(lines)


@dataclass
class OrchestrationResult:
    callbacks: list[Callback]
    report: OrchestrationReport


def _constraint_dict(constraint: OrchestrationConstraint, reason: Optional[str] = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": constraint.kind,
        "subject": constraint.subject,
        "target": constraint.target,
        "hard": constraint.hard,
        "origin": constraint.origin,
        "description": constraint.describe(),
    }
    if reason is not None:
        entry["reason"] = reason
    return entry


def _legacy_implicit_constraints(participant: _Participant) -> list[OrchestrationConstraint]:
    """The historical implicit ordering guarantees, expressed as soft constraints.

    Tuner callbacks run first and checkpoint callbacks run last. They are soft so that explicitly declared hard
    constraints can override them; when they get relaxed, the report says so.

    """
    # imported lazily to avoid a circular import at package init time
    from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
    from lightning.pytorch.callbacks.checkpoint import Checkpoint
    from lightning.pytorch.callbacks.lr_finder import LearningRateFinder

    callback = participant.callback
    if isinstance(callback, (BatchSizeFinder, LearningRateFinder)):
        return [Before(participant.capability, ALL, hard=False, origin="legacy:tuner-first")]
    if isinstance(callback, Checkpoint):
        return [After(participant.capability, ALL, hard=False, origin="legacy:checkpoint-last")]
    return []


def _find_cycle(capabilities: list[str], edges: dict[str, set[str]]) -> Optional[list[str]]:
    """Return one cycle as a list of capability ids, or ``None`` if the graph is acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(capabilities, WHITE)
    stack: list[str] = []

    def visit(node: str) -> Optional[list[str]]:
        color[node] = GRAY
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if color.get(nxt, BLACK) == GRAY:
                return stack[stack.index(nxt):] + [nxt]
            if color.get(nxt, BLACK) == WHITE:
                cycle = visit(nxt)
                if cycle is not None:
                    return cycle
        stack.pop()
        color[node] = BLACK
        return None

    for capability in sorted(capabilities):
        if color[capability] == WHITE:
            cycle = visit(capability)
            if cycle is not None:
                return cycle
    return None


def _reachable(edges: dict[str, set[str]], start: str, goal: str) -> bool:
    """Whether ``goal`` is reachable from ``start`` following the edges."""
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        if node == goal:
            return True
        for nxt in edges.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def resolve_callback_orchestration(
    entries: Iterable[tuple[Callback, CallbackSource]],
    extra_constraints: Iterable[OrchestrationConstraint] = (),
) -> OrchestrationResult:
    """Resolve a set of callbacks and constraints into a single deterministic execution order.

    Args:
        entries: ``(callback, source)`` pairs. The same instance registered twice is deduplicated (first occurrence
            wins, recorded in the report).
        extra_constraints: Constraints declared outside of the callbacks (e.g. via
            :func:`with_constraints` on a wrapper, or by the Trainer).

    Returns:
        The ordered callbacks plus an :class:`OrchestrationReport`.

    Raises:
        OrchestrationCycleError: hard before/after constraints form a cycle.
        DanglingConstraintError: a hard constraint references an unknown capability.
        OrchestrationConflictError: hard constraints are mutually unsatisfiable.

    """
    dropped: list[dict[str, Any]] = []
    inferred: list[str] = []
    satisfied: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ 1. dedup identical instances
    participants: list[_Participant] = []
    seen_instances: dict[int, _Participant] = {}
    for callback, source in entries:
        if id(callback) in seen_instances:
            kept = seen_instances[id(callback)]
            dropped.append(
                {
                    "capability": kept.capability,
                    "source": source.value,
                    "reason": f"same instance registered twice; kept the registration from source "
                    f"'{kept.source.value}'",
                }
            )
            continue
        participant = _Participant(
            callback=callback,
            source=source,
            capability=_capability_of(callback),
            constraints=list(_declared_constraints_of(callback)),
        )
        participants.append(participant)
        seen_instances[id(callback)] = participant

    # ------------------------------------------------------- 2. capability collisions across sources
    by_capability: dict[str, list[_Participant]] = {}
    for participant in participants:
        by_capability.setdefault(participant.capability, []).append(participant)

    final: list[_Participant] = []
    ambiguous: dict[str, list[str]] = {}
    for capability in sorted(by_capability):
        group = by_capability[capability]
        if len(group) == 1:
            final.append(group[0])
            continue
        sources = {p.source for p in group}
        if len(sources) > 1:
            # deterministic: source precedence decides, never registration order
            winner = min(group, key=lambda p: (_SOURCE_PRECEDENCE[p.source], p.source.value))
            final.append(winner)
            for loser in group:
                if loser is not winner:
                    dropped.append(
                        {
                            "capability": capability,
                            "source": loser.source.value,
                            "reason": f"capability '{capability}' also provided by source "
                            f"'{winner.source.value}', which takes precedence",
                        }
                    )
        else:
            # same source, same capability, different instances: keep all under suffixed capabilities
            suffixed = []
            for index, participant in enumerate(group):
                participant.capability = capability if index == 0 else f"{capability}#{index + 1}"
                suffixed.append(participant.capability)
                final.append(participant)
            ambiguous[capability] = suffixed
            inferred.append(
                f"capability '{capability}' is provided by {len(group)} instances from the same source; they were"
                f" disambiguated as {suffixed}. Constraints must target the suffixed ids."
            )

    # ------------------------------------------------------------- 3. gather and normalize constraints
    constraints: list[OrchestrationConstraint] = list(extra_constraints)
    for participant in final:
        constraints.extend(participant.constraints)
        constraints.extend(_legacy_implicit_constraints(participant))
    # identical constraints declared twice collapse into one
    constraints = sorted(set(constraints), key=lambda c: (c.kind, c.subject, c.target, c.hard, c.origin))

    present = {p.capability for p in final}

    def _expand(target: str) -> set[str]:
        """Expand a constraint target into concrete capability ids."""
        if target == ALL:
            return set(present)
        return {target}

    # ------------------------------------------------------------- 4. mutual exclusion
    remaining_constraints: list[OrchestrationConstraint] = []
    for constraint in constraints:
        if constraint.kind != "mutually_exclusive":
            remaining_constraints.append(constraint)
            continue
        a, b = constraint.subject, constraint.target
        a_present, b_present = a in present, b in present
        if not a_present and not b_present:
            if constraint.hard:
                raise DanglingConstraintError(
                    f"Constraint {constraint.describe()} references capabilities that are not part of this run."
                    f" Available capabilities: {sorted(present)}."
                )
            relaxed.append(_constraint_dict(constraint, reason="neither capability is part of this run"))
            continue
        if a_present != b_present:
            # only one of the two is registered: no conflict can arise, the constraint holds trivially
            satisfied.append(_constraint_dict(constraint))
            continue
        if constraint.hard:
            pa = next(p for p in final if p.capability == a)
            pb = next(p for p in final if p.capability == b)
            raise OrchestrationConflictError(
                f"Hard mutual exclusion conflict: `{a}` (source: {pa.source.value}) and `{b}` (source:"
                f" {pb.source.value}) are both registered but declared mutually exclusive by {constraint.origin}."
                f" Resolution options: (1) remove one of the two callbacks, (2) downgrade the constraint to soft"
                f" (`hard=False`) so the resolver keeps the higher-precedence one deterministically."
            )
        # soft: keep the higher-precedence source; tie-break by capability id for determinism
        pa = next(p for p in final if p.capability == a)
        pb = next(p for p in final if p.capability == b)
        # the loser is the one with the *lower* precedence (larger precedence value); capability id breaks ties
        loser = max((pa, pb), key=lambda p: (_SOURCE_PRECEDENCE[p.source], p.capability))
        winner = pb if loser is pa else pa
        final.remove(loser)
        present.discard(loser.capability)
        dropped.append(
            {
                "capability": loser.capability,
                "source": loser.source.value,
                "reason": f"soft mutual exclusion with '{winner.capability}' (declared by {constraint.origin});"
                f" '{winner.capability}' (source: {winner.source.value}) takes precedence",
            }
        )
        satisfied.append(_constraint_dict(constraint))
    constraints = remaining_constraints

    # ------------------------------------------------------------- 5. exclusive resources
    remaining_constraints = []
    resource_owners: dict[str, list[OrchestrationConstraint]] = {}
    for constraint in constraints:
        if constraint.kind != "exclusive_resource":
            remaining_constraints.append(constraint)
            continue
        if constraint.subject not in present:
            remaining_constraints.append(constraint)  # dangling; handled below
            continue
        resource_owners.setdefault(constraint.target, []).append(constraint)
        satisfied.append(_constraint_dict(constraint))
    for resource, owners in sorted(resource_owners.items()):
        hard_owners = [c for c in owners if c.hard]
        if len(hard_owners) > 1:
            claimants = ", ".join(f"`{c.subject}` (declared by {c.origin})" for c in hard_owners)
            raise OrchestrationConflictError(
                f"Hard exclusive-resource conflict: resource `{resource}` is exclusively claimed by {claimants}."
                f" Resolution options: (1) remove all but one claimant, (2) downgrade all but one claim to soft"
                f" (`hard=False`)."
            )
        if hard_owners and len(owners) > 1:
            for c in owners:
                if not c.hard:
                    reason = f"resource `{resource}` is hard-owned by another capability"
                    relaxed.append(_constraint_dict(c, reason=reason))
    constraints = remaining_constraints

    # ------------------------------------------------------------- 6. reference validation
    validated: list[OrchestrationConstraint] = []
    for constraint in constraints:
        refs = [ref for ref in (constraint.subject, constraint.target) if ref != ALL]
        ambiguous_refs = [ref for ref in refs if ref in ambiguous]
        unknown = [ref for ref in refs if ref not in present and ref not in ambiguous]
        problem: Optional[str] = None
        if ambiguous_refs:
            options = {ref: ambiguous[ref] for ref in ambiguous_refs}
            problem = f"references ambiguous capability {ambiguous_refs}; use one of the disambiguated ids: {options}"
        elif unknown:
            problem = f"references unknown capability {unknown}; available capabilities: {sorted(present)}"
        if problem is None:
            validated.append(constraint)
        elif constraint.hard:
            raise DanglingConstraintError(f"Constraint {constraint.describe()} {problem}.")
        else:
            relaxed.append(_constraint_dict(constraint, reason=problem))
    constraints = validated

    # ------------------------------------------------------------- 7. hard ordering graph + cycle check
    def _ordering_edges(cs: Iterable[OrchestrationConstraint]) -> dict[str, set[str]]:
        edges: dict[str, set[str]] = {p.capability: set() for p in final}
        for c in cs:
            if c.kind == "before":
                for target in _expand(c.target):
                    if target != c.subject and target in edges:
                        edges[c.subject].add(target)
            elif c.kind == "after":
                for target in _expand(c.target):
                    if target != c.subject and target in edges:
                        edges[target].add(c.subject)
        return edges

    hard_constraints = [c for c in constraints if c.hard]
    soft_constraints = [c for c in constraints if not c.hard]

    # friendlier message for the common "two capabilities must both be last/first" case
    for kind, position in (("after", "last"), ("before", "first")):
        anchored = [c for c in hard_constraints if c.kind == kind and c.target == ALL]
        if len(anchored) > 1:
            claimants = ", ".join(f"`{c.subject}` (declared by {c.origin})" for c in anchored)
            raise OrchestrationConflictError(
                f"Hard ordering conflict: {claimants} all require to run {position}. Only one capability can be"
                f" {position}. Resolution options: (1) relax all but one to soft (`hard=False`), (2) order them"
                f" explicitly relative to each other instead of against `*`."
            )

    hard_edges = _ordering_edges(hard_constraints)
    cycle = _find_cycle(sorted(present), hard_edges)
    if cycle is not None:
        sources = {p.capability: p.source.value for p in final}
        path = " -> ".join(f"`{c}` (source: {sources[c]})" for c in cycle)
        involved = [c.describe() for c in hard_constraints if c.subject in cycle]
        raise OrchestrationCycleError(
            f"Hard ordering constraints form a cycle: {path}. Constraints involved: {involved}. Remove or soften"
            f" one of them to break the cycle."
        )
    satisfied.extend(_constraint_dict(c) for c in hard_constraints)

    # ------------------------------------------------------------- 8. integrate soft edges
    edges = {node: set(targets) for node, targets in hard_edges.items()}
    for constraint in sorted(soft_constraints, key=lambda c: (c.kind, c.subject, c.target, c.origin)):
        if constraint.kind == "before":
            pairs = [(constraint.subject, t) for t in sorted(_expand(constraint.target))]
        else:  # after
            pairs = [(t, constraint.subject) for t in sorted(_expand(constraint.target))]
        pairs = [(s, t) for s, t in pairs if s != t and s in edges and t in edges]
        if not pairs:
            # nothing to order against (e.g. the only participant): the constraint holds vacuously
            satisfied.append(_constraint_dict(constraint))
        elif all(not _reachable(edges, t, s) for s, t in pairs):
            for s, t in pairs:
                edges[s].add(t)
            satisfied.append(_constraint_dict(constraint))
        else:
            relaxed.append(_constraint_dict(constraint, reason="would create a cycle with higher-priority constraints"))

    # ------------------------------------------------------------- 9. deterministic topological sort
    indegree = dict.fromkeys(edges, 0)
    for node, targets in edges.items():
        for target in targets:
            indegree[target] += 1

    heap = [(capability, capability) for capability, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, capability = heapq.heappop(heap)
        order.append(capability)
        for target in sorted(edges[capability]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, (target, target))
    if len(order) != len(final):  # pragma: no cover - guarded by the cycle check above
        raise OrchestrationCycleError("Ordering constraints form a cycle that could not be resolved.")

    inferred.append(
        "pairs not ordered by any constraint were placed by the canonical tie-break (capability id, ascending);"
        " the result is independent of insertion order, callback source, and process count"
    )

    # ------------------------------------------------------------- 10. report
    by_capability_final = {p.capability: p for p in final}
    ordered_callbacks = [by_capability_final[capability].callback for capability in order]
    report = OrchestrationReport(
        order=order,
        sources={p.capability: p.source.value for p in final},
        satisfied=satisfied,
        relaxed=relaxed,
        inferred=inferred,
        dropped=dropped,
        fingerprint=orchestration_fingerprint(order),
    )
    return OrchestrationResult(callbacks=ordered_callbacks, report=report)
