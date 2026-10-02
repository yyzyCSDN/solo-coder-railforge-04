from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock

from railforge.core.time import utc


@dataclass(frozen=True)
class ConsistVehicle:
    """One vehicle of the consist; axle_loads_t carries the eccentric (offset) axle weights."""
    position: int
    wagon: str
    height_m: float
    width_m: float
    axle_loads_t: tuple[float, ...]


@dataclass(frozen=True)
class PlannedSection:
    section: str
    enter_at: datetime
    exit_at: datetime


@dataclass(frozen=True)
class GaugeSection:
    section: str
    max_height_m: float
    max_width_m: float
    max_axle_t: float
    max_gross_t: float


@dataclass(frozen=True)
class TemporaryBlockade:
    blockade_id: str
    section: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class GaugeCheckRequest:
    train: str
    consist: tuple[ConsistVehicle, ...]
    route: tuple[PlannedSection, ...]
    clearances: tuple[GaugeSection, ...]
    blockades: tuple[TemporaryBlockade, ...]
    data_version: int


@dataclass(frozen=True)
class GaugeRejection:
    section: str
    reason: str
    detail: str


@dataclass(frozen=True)
class GaugeVerdict:
    train: str
    data_version: int
    passed: bool
    sections: tuple[str, ...]
    checked_sections: tuple[str, ...]
    first_failure: GaugeRejection | None


@dataclass(frozen=True)
class GaugeReservation:
    reservation_id: str
    train: str
    sections: tuple[str, ...]
    data_version: int
    reserved_at: datetime
    status: str = "active"
    revoked_at: datetime | None = None
    revoke_reason: str | None = None


def consist_gross_t(consist: tuple[ConsistVehicle, ...]) -> float:
    return sum(sum(v.axle_loads_t) for v in consist)


def consist_peak_axle_t(consist: tuple[ConsistVehicle, ...]) -> float:
    return max(max(v.axle_loads_t) for v in consist)


def _validate_consist(consist: tuple[ConsistVehicle, ...]) -> tuple[ConsistVehicle, ...]:
    if not consist:
        raise ValueError("empty consist")
    ordered = tuple(sorted(consist, key=lambda v: v.position))
    if [v.position for v in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("position gap")
    wagons = [v.wagon for v in ordered]
    if len(wagons) != len(set(wagons)):
        raise ValueError("duplicate wagon")
    for v in ordered:
        if v.height_m <= 0 or v.width_m <= 0:
            raise ValueError("vehicle envelope")
        if not v.axle_loads_t or min(v.axle_loads_t) < 0:
            raise ValueError("axle loads")
    return ordered


def _check_section(section, enter, leave, clearances, blockades,
                   height, width, peak_axle, gross) -> GaugeRejection | None:
    # Deterministic priority: blockade, missing data, envelope, then loads.
    for bsec, start, end, bid in blockades:
        if bsec == section and max(enter, start) < min(leave, end):
            return GaugeRejection(section, "blockade", f"temporary blockade {bid} active")
    c = clearances.get(section)
    if c is None:
        return GaugeRejection(section, "clearance-missing", "no clearance data for section")
    if height > c.max_height_m:
        return GaugeRejection(section, "height", f"height {height} exceeds {c.max_height_m}")
    if width > c.max_width_m:
        return GaugeRejection(section, "width", f"width {width} exceeds {c.max_width_m}")
    if peak_axle > c.max_axle_t:
        return GaugeRejection(section, "axle", f"peak axle {peak_axle} exceeds {c.max_axle_t}")
    if gross > c.max_gross_t:
        return GaugeRejection(section, "gross", f"gross {gross} exceeds {c.max_gross_t}")
    return None


def verify(request: GaugeCheckRequest) -> GaugeVerdict:
    """Full-route clearance check; stops at and reports the first failed section."""
    consist = _validate_consist(tuple(request.consist))
    if not request.route:
        raise ValueError("empty route")
    if request.data_version < 1:
        raise ValueError("data version")
    clearances = {}
    for c in request.clearances:
        if c.section in clearances:
            raise ValueError("duplicate clearance section")
        clearances[c.section] = c
    blockades = []
    for b in request.blockades:
        start, end = utc(b.start), utc(b.end)
        if end <= start:
            raise ValueError("blockade window")
        blockades.append((b.section, start, end, b.blockade_id))
    height = max(v.height_m for v in consist)
    width = max(v.width_m for v in consist)
    peak_axle = consist_peak_axle_t(consist)
    gross = consist_gross_t(consist)
    sections = tuple(p.section for p in request.route)
    checked: list[str] = []
    for planned in request.route:
        enter, leave = utc(planned.enter_at), utc(planned.exit_at)
        if leave <= enter:
            raise ValueError("section window")
        checked.append(planned.section)
        failure = _check_section(planned.section, enter, leave, clearances, blockades,
                                 height, width, peak_axle, gross)
        if failure is not None:
            return GaugeVerdict(request.train, request.data_version, False,
                                sections, tuple(checked), failure)
    return GaugeVerdict(request.train, request.data_version, True,
                        sections, tuple(checked), None)


class GaugePathRegistry:
    """Reservations gated on a passing verdict; any version conflict revokes them."""

    def __init__(self):
        self._lock = RLock()
        self._passes: dict[tuple[str, tuple[str, ...], int], GaugeVerdict] = {}
        self._reservations: dict[str, GaugeReservation] = {}
        self._latest_version = 0

    @property
    def latest_version(self) -> int:
        with self._lock:
            return self._latest_version

    def record_pass(self, verdict: GaugeVerdict) -> None:
        if not verdict.passed:
            raise ValueError("verdict not passed")
        with self._lock:
            key = (verdict.train, verdict.sections, verdict.data_version)
            self._passes[key] = verdict
            self._latest_version = max(self._latest_version, verdict.data_version)

    def reserve(self, reservation_id: str, train: str, sections: tuple[str, ...],
                data_version: int, at: datetime) -> GaugeReservation:
        with self._lock:
            existing = self._reservations.get(reservation_id)
            if existing is not None:
                same = (existing.train, existing.sections, existing.data_version)
                if existing.status == "active" and same == (train, sections, data_version):
                    return existing
                raise ValueError("reservation id reused")
            if (train, sections, data_version) not in self._passes:
                raise ValueError("clearance not verified")
            if data_version != self._latest_version:
                raise ValueError("stale clearance version")
            r = GaugeReservation(reservation_id, train, sections, data_version, utc(at))
            self._reservations[reservation_id] = r
            return r

    def revalidate(self, reservation_id: str, current_version: int,
                   at: datetime) -> tuple[str, GaugeReservation]:
        with self._lock:
            r = self._reservations.get(reservation_id)
            if r is None:
                raise ValueError("unknown reservation")
            if r.status == "revoked":
                return ("revoked", r)
            if current_version != r.data_version:
                r = replace(r, status="revoked", revoked_at=utc(at),
                            revoke_reason="version-conflict")
                self._reservations[reservation_id] = r
                return ("revoked", r)
            return ("confirmed", r)

    def revoke_stale(self, current_version: int, at: datetime) -> list[GaugeReservation]:
        with self._lock:
            self._latest_version = max(self._latest_version, current_version)
            out = []
            for rid in sorted(self._reservations):
                r = self._reservations[rid]
                if r.status == "active" and r.data_version != current_version:
                    r = replace(r, status="revoked", revoked_at=utc(at),
                                revoke_reason="version-conflict")
                    self._reservations[rid] = r
                    out.append(r)
            return out

    def get(self, reservation_id: str) -> GaugeReservation | None:
        with self._lock:
            return self._reservations.get(reservation_id)

    def active(self) -> list[GaugeReservation]:
        with self._lock:
            return sorted((r for r in self._reservations.values() if r.status == "active"),
                          key=lambda r: r.reservation_id)
