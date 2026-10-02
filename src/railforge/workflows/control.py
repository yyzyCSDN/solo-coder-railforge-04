from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal
from threading import RLock
from typing import Any, Callable, Mapping, TypeVar

from railforge.audit.trail import Trail
from railforge.braking.brake_percentage import brake_percentage
from railforge.billing.demurrage import charge
from railforge.cargo.segregation import violations
from railforge.consist.integrity import validate
from railforge.customs.holds import active
from railforge.energy.regen import net_energy
from railforge.eta.event_fusion import fused_delay
from railforge.events.cursor import CursorConsumer, FeedEvent
from railforge.events.stream import EventStream
from railforge.intermodal.connections import feasible
from railforge.inventory.wagon_pool import allocate
from railforge.locomotive.tractive_effort import available_effort
from railforge.maintenance.release import releasable
from railforge.possession.windows import conflicts as possession_conflicts
from railforge.routing.clearance import route_ok
from railforge.routing.oversize_clearance import check as oversize_clearance_check
from railforge.signaling.block_occupancy import reserve
from railforge.storage.ops_store import VersionedStore
from railforge.timetable.meets import conflicts as meet_conflicts
from railforge.wagon.axle_load import route_ok as axle_route_ok
from railforge.yard.hump import classify
from railforge.yard.switching import lock_route
from railforge.crew.duty import legal as duty_legal

T = TypeVar("T")
BROKEN_CONTROL_TOPICS: set[str] = set()


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value


@dataclass(frozen=True)
class OperationReceipt:
    operation_id: str
    topic: str
    subject: str
    result: Any
    event: Any
    audit_index: int
    version: int


class RailForgeWorkflowControl:
    """Atomic domain -> event -> audit workflow used by every feature surface."""

    def __init__(self, stream: EventStream | None = None, trail: Trail | None = None,
                 store: VersionedStore | None = None):
        self.stream = stream or EventStream()
        self.trail = trail or Trail()
        self.store = store or VersionedStore()
        self._receipts: dict[str, OperationReceipt] = {}
        self._consumers: dict[str, CursorConsumer] = {}
        self._oversize_lock = RLock()

    @property
    def receipts(self) -> Mapping[str, OperationReceipt]:
        return dict(self._receipts)

    def _commit(self, operation_id: str, topic: str, subject: str, at: datetime,
                action: Callable[[], T], evidence: Mapping[str, Any] | None = None) -> OperationReceipt:
        previous = self._receipts.get(operation_id)
        if previous is not None:
            if previous.topic != topic or previous.subject != subject:
                raise ValueError("operation id reused for a different workflow")
            return previous
        result = action()
        version = self.store.write(subject, _safe(result))
        payload = {
            "operation_id": operation_id,
            "status": "committed",
            "control_path": "legacy" if topic in BROKEN_CONTROL_TOPICS else "atomic",
            "version": version,
            "result": _safe(result),
            **dict(evidence or {}),
        }
        event = self.stream.publish(operation_id, at, topic, subject, payload)
        audit = self.trail.append("railforge-control", topic, subject, at)
        receipt = OperationReceipt(operation_id, topic, subject, result, event, audit.index, version)
        self._receipts[operation_id] = receipt
        return receipt

    def _publish(self, operation_id: str, topic: str, subject: str, at: datetime,
                 result: Any, status: str, version: int,
                 evidence: Mapping[str, Any] | None = None) -> OperationReceipt:
        previous = self._receipts.get(operation_id)
        if previous is not None:
            if previous.topic != topic or previous.subject != subject:
                raise ValueError("operation id reused for a different workflow")
            return previous
        payload = {
            "operation_id": operation_id,
            "status": status,
            "control_path": "atomic",
            "version": version,
            "result": _safe(result),
            **dict(evidence or {}),
        }
        event = self.stream.publish(operation_id, at, topic, subject, payload)
        audit = self.trail.append("railforge-control", topic, subject, at)
        receipt = OperationReceipt(operation_id, topic, subject, result, event, audit.index, version)
        self._receipts[operation_id] = receipt
        return receipt

    def run(self, operation_id: str, topic: str, subject: str, at: datetime,
            action: Callable[[], T], evidence: Mapping[str, Any] | None = None) -> OperationReceipt:
        return self._commit(operation_id, topic, subject, at, action, evidence)

    def reserve_block(self, operation_id, candidate, existing, clearance, at):
        return self._commit(operation_id, "dispatch.block.reserved", candidate.train, at,
                            lambda: reserve(candidate, existing, clearance))

    def validate_consist(self, operation_id, rows, subject, at):
        return self._commit(operation_id, "consist.validated", subject, at, lambda: validate(rows))

    def assess_brakes(self, operation_id, rows, gradient, subject, at):
        return self._commit(operation_id, "braking.assessed", subject, at,
                            lambda: brake_percentage(rows, gradient))

    def plan_traction(self, operation_id, rows, adhesion_factor, subject, at):
        return self._commit(operation_id, "traction.planned", subject, at,
                            lambda: available_effort(rows, adhesion_factor))

    def assign_duty(self, operation_id, rows, max_duty, max_drive, max_span, subject, at):
        def action():
            result = duty_legal(rows, max_duty, max_drive, max_span)
            if not result:
                raise ValueError("duty is not legal")
            return result
        return self._commit(operation_id, "crew.duty.assigned", subject, at, action)

    def book_meet(self, operation_id, rows, subject, at):
        return self._commit(operation_id, "timetable.meet.booked", subject, at,
                            lambda: meet_conflicts(rows))

    def classify_yard(self, operation_id, cuts, tracks, occupied, subject, at):
        return self._commit(operation_id, "yard.classified", subject, at,
                            lambda: classify(cuts, tracks, occupied))

    def check_axles(self, operation_id, wagon, max_axle, max_gross, subject, at):
        return self._commit(operation_id, "wagon.axle.checked", subject, at,
                            lambda: axle_route_ok(wagon, max_axle, max_gross))

    def book_intermodal(self, operation_id, inbound, outbound, unload, load, cutoff, subject, at):
        return self._commit(operation_id, "intermodal.booked", subject, at,
                            lambda: feasible(inbound, outbound, unload, load, cutoff))

    def protect_possession(self, operation_id, candidate, rows, subject, at):
        return self._commit(operation_id, "possession.protected", subject, at,
                            lambda: possession_conflicts(candidate, rows))

    def release_maintenance(self, operation_id, evidence, policy, check_at, subject, at):
        def action():
            result = releasable(evidence, policy, check_at)
            if not result:
                raise ValueError("maintenance evidence incomplete")
            return result
        return self._commit(operation_id, "maintenance.released", subject, at, action)

    def consume_event(self, operation_id, event: FeedEvent, subject, at):
        consumer = self._consumers.setdefault(event.partition, CursorConsumer())
        return self._commit(operation_id, "events.cursor.consumed", subject, at,
                            lambda: consumer.accept(event))

    def charge_demurrage(self, operation_id, arrival, release, tariff, subject, at):
        return self._commit(operation_id, "billing.demurrage.charged", subject, at,
                            lambda: charge(arrival, release, tariff))

    def check_customs(self, operation_id, rows, scope, check_at, subject, at):
        return self._commit(operation_id, "customs.hold.checked", subject, at,
                            lambda: active(rows, scope, check_at))

    def check_segregation(self, operation_id, rows, rules, subject, at):
        return self._commit(operation_id, "cargo.segregation.checked", subject, at,
                            lambda: violations(rows, rules))

    def calculate_energy(self, operation_id, rows, subject, at):
        return self._commit(operation_id, "energy.calculated", subject, at,
                            lambda: net_energy(rows))

    def lock_switch_route(self, operation_id, route, switches, owner, subject, at):
        return self._commit(operation_id, "yard.route.locked", subject, at,
                            lambda: lock_route(route, switches, owner))

    def check_clearance(self, operation_id, profile, segments, subject, at):
        return self._commit(operation_id, "routing.clearance.checked", subject, at,
                            lambda: route_ok(profile, segments))

    def check_oversize_clearance(self, operation_id, consist, sections, blockades, subject,
                                at, reference_version=0):
        decision = oversize_clearance_check(consist, sections, blockades, at)
        if not decision.allowed:
            failure = decision.first_failure
            return self._publish(
                operation_id, "routing.oversize_clearance.rejected", subject, at,
                decision, "rejected", reference_version,
                {"reference_version": reference_version,
                 "first_failed_section": failure.section_id,
                 "reason": failure.reason,
                 "failure": _safe(failure)})
        return self._publish(
            operation_id, "routing.oversize_clearance.checked", subject, at,
            decision, "checked", reference_version,
            {"reference_version": reference_version})

    def reserve_oversize_path(self, operation_id, consist, sections, blockades, subject,
                              at, expected_version, reference_version=None):
        decision = oversize_clearance_check(consist, sections, blockades, at)
        if reference_version is None:
            reference_version = expected_version
        if not decision.allowed:
            failure = decision.first_failure
            return self._publish(
                operation_id, "routing.oversize_clearance.rejected", subject, at,
                decision, "rejected", reference_version,
                {"reference_version": reference_version,
                 "first_failed_section": failure.section_id,
                 "reason": failure.reason,
                 "failure": _safe(failure)})

        return self._commit_oversize_reservation(
            operation_id, subject, at, expected_version, reference_version, decision)

    def _commit_oversize_reservation(self, operation_id, subject, at, expected_version,
                                     reference_version, decision):
        with self._oversize_lock:
            key = "oversize-path:" + subject + ":" + ":".join(decision.checked_sections)
            current = self.store.read(key)
            actual_version = current.version if current else 0
            if expected_version != actual_version:
                return self._publish_oversize_version_conflict(
                    operation_id, subject, at, expected_version, actual_version,
                    reference_version, decision.checked_sections)

            reservation = {
                "reserved": True,
                "revoked": False,
                "subject": subject,
                "route": decision.checked_sections,
                "reference_version": reference_version,
            }
            try:
                version = self.store.write(key, _safe(reservation), expected_version)
            except RuntimeError:
                current = self.store.read(key)
                actual_version = current.version if current else 0
                return self._publish_oversize_version_conflict(
                    operation_id, subject, at, expected_version, actual_version,
                    reference_version, decision.checked_sections)
            return self._publish(
                operation_id, "routing.oversize_path.reserved", subject, at,
                reservation, "reserved", version,
                {"reference_version": reference_version})

    def _publish_oversize_version_conflict(self, operation_id, subject, at,
                                           expected_version, actual_version,
                                           reference_version, route):
        with self._oversize_lock:
            if actual_version:
                key = "oversize-path:" + subject + ":" + ":".join(route)
                current = self.store.read(key)
                if current and isinstance(current.value, dict) and current.value.get("reserved"):
                    revoked = dict(current.value)
                    revoked.update({"reserved": False, "revoked": True, "reason": "version conflict"})
                    self.store.write(key, _safe(revoked), actual_version)
                    actual_version += 1
            result = {
                "reserved": False,
                "revoked": True,
                "reason": "version conflict",
                "expected_version": expected_version,
                "actual_version": actual_version,
                "reference_version": reference_version,
                "route": route,
            }
            return self._publish(
                operation_id, "routing.oversize_path.revoked", subject, at,
                result, "revoked", actual_version,
                {"reference_version": reference_version})

    def allocate_wagons(self, operation_id, rows, wagon_type, location, on, count, required_days, forbidden, subject, at):
        return self._commit(operation_id, "inventory.wagons.allocated", subject, at,
                            lambda: allocate(rows, wagon_type, location, on, count, required_days, forbidden))

    def fuse_eta(self, operation_id, rows, max_lateness, subject, at):
        return self._commit(operation_id, "eta.fused", subject, at,
                            lambda: fused_delay(rows, max_lateness))

