from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OversizeWagon:
    wagon_id: str
    gross_t: float
    axles: int
    height_m: float = 0.0
    left_offset_m: float = 0.0
    right_offset_m: float = 0.0
    front_fraction: float = 0.5
    axle_loads_t: tuple[float, ...] | None = None

    def max_axle_load_t(self) -> float:
        if self.axle_loads_t is not None:
            if len(self.axle_loads_t) != self.axles or any(x < 0 for x in self.axle_loads_t):
                raise ValueError("oversize wagon")
            return max(self.axle_loads_t, default=0.0)
        if self.axles <= 0 or self.gross_t < 0 or not 0 <= self.front_fraction <= 1:
            raise ValueError("oversize wagon")
        half = self.axles // 2
        if half == 0:
            return self.gross_t
        front = self.gross_t * self.front_fraction / half
        rear = self.gross_t * (1 - self.front_fraction) / half
        return max(front, rear)


@dataclass(frozen=True)
class ClearanceSection:
    section_id: str
    max_height_m: float
    max_width_m: float = 0.0
    max_gross_t: float = 0.0
    max_axle_t: float = 0.0
    max_left_offset_m: float | None = None
    max_right_offset_m: float | None = None

    @classmethod
    def symmetric(cls, section_id: str, max_height_m: float, max_width_m: float,
                  max_gross_t: float, max_axle_t: float):
        return cls(section_id, max_height_m, max_width_m, max_gross_t, max_axle_t)

    @property
    def left_limit_m(self) -> float:
        return self.max_left_offset_m if self.max_left_offset_m is not None else self.max_width_m / 2

    @property
    def right_limit_m(self) -> float:
        return self.max_right_offset_m if self.max_right_offset_m is not None else self.max_width_m / 2


@dataclass(frozen=True)
class TemporaryBlockade:
    blockade_id: str
    section_id: str
    start: datetime | None = None
    end: datetime | None = None
    reason: str = "temporary blockade"


@dataclass(frozen=True)
class ClearanceFailure:
    section_id: str
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ClearanceDecision:
    allowed: bool
    checked_sections: tuple[str, ...]
    first_failure: ClearanceFailure | None = None

    @property
    def first_failed_section(self) -> str | None:
        return None if self.first_failure is None else self.first_failure.section_id

    @property
    def reason(self) -> str | None:
        return None if self.first_failure is None else self.first_failure.reason


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo is not None else value


def _blockade_active(blockade: TemporaryBlockade, at: datetime | None) -> bool:
    if blockade.start is None and blockade.end is None:
        return True
    if at is None:
        return True
    moment = _as_utc(at)
    start = _as_utc(blockade.start) if blockade.start is not None else None
    end = _as_utc(blockade.end) if blockade.end is not None else None
    if start is not None and moment < start:
        return False
    if end is not None and moment >= end:
        return False
    return True


def _as_oversize_wagon(row: object) -> OversizeWagon:
    if isinstance(row, OversizeWagon):
        return row
    if hasattr(row, "axles") and (hasattr(row, "wagon_id") or hasattr(row, "id")):
        wagon_id = getattr(row, "wagon_id", getattr(row, "id", ""))
        gross = getattr(row, "gross_t", None)
        if gross is None and hasattr(row, "tare_t") and hasattr(row, "payload_t"):
            gross = row.tare_t + row.payload_t
        gross = 0.0 if gross is None else gross
        axle_loads = getattr(row, "axle_loads_t", None)
        return OversizeWagon(
            str(wagon_id),
            float(gross),
            int(row.axles),
            height_m=float(getattr(row, "height_m", 0.0)),
            left_offset_m=float(getattr(row, "left_offset_m", 0.0)),
            right_offset_m=float(getattr(row, "right_offset_m", 0.0)),
            front_fraction=float(getattr(row, "front_fraction", 0.5)),
            axle_loads_t=tuple(axle_loads) if axle_loads is not None else None,
        )
    raise TypeError("oversize wagon required")


def _wagon_rows(consist: Sequence[object]) -> list[OversizeWagon]:
    return [_as_oversize_wagon(row) for row in consist]


def check(consist: list[OversizeWagon], sections: list[ClearanceSection],
          blockades: list[TemporaryBlockade] | None = None,
          at: datetime | None = None) -> ClearanceDecision:
    """Check every section in route order and stop at the first failure."""
    if not consist or not sections:
        raise ValueError("oversize route requires consist and sections")
    consist = _wagon_rows(consist)
    blockades = list(blockades or [])
    checked = []
    for section in sections:
        checked.append(section.section_id)
        active = [b for b in blockades
                  if b.section_id == section.section_id and _blockade_active(b, at)]
        if active:
            blockade = min(active, key=lambda b: b.blockade_id)
            return ClearanceDecision(
                False, tuple(checked),
                ClearanceFailure(section.section_id, "temporary_blockade", {
                    "blockade_id": blockade.blockade_id,
                    "reason": blockade.reason,
                }))

        gross = sum(w.gross_t for w in consist)
        if gross > section.max_gross_t:
            return ClearanceDecision(
                False, tuple(checked),
                ClearanceFailure(section.section_id, "gross_weight", {
                    "gross_t": gross,
                    "limit_t": section.max_gross_t,
                }))

        for wagon in consist:
            if wagon.height_m > section.max_height_m:
                return ClearanceDecision(
                    False, tuple(checked),
                    ClearanceFailure(section.section_id, "height", {
                        "wagon_id": wagon.wagon_id,
                        "height_m": wagon.height_m,
                        "limit_m": section.max_height_m,
                    }))
            if wagon.left_offset_m > section.left_limit_m:
                return ClearanceDecision(
                    False, tuple(checked),
                    ClearanceFailure(section.section_id, "lateral_offset", {
                        "wagon_id": wagon.wagon_id,
                        "side": "left",
                        "offset_m": wagon.left_offset_m,
                        "limit_m": section.left_limit_m,
                    }))
            if wagon.right_offset_m > section.right_limit_m:
                return ClearanceDecision(
                    False, tuple(checked),
                    ClearanceFailure(section.section_id, "lateral_offset", {
                        "wagon_id": wagon.wagon_id,
                        "side": "right",
                        "offset_m": wagon.right_offset_m,
                        "limit_m": section.right_limit_m,
                    }))
            axle_load = wagon.max_axle_load_t()
            if axle_load > section.max_axle_t:
                return ClearanceDecision(
                    False, tuple(checked),
                    ClearanceFailure(section.section_id, "axle_load", {
                        "wagon_id": wagon.wagon_id,
                        "axle_load_t": axle_load,
                        "limit_t": section.max_axle_t,
                    }))

    return ClearanceDecision(True, tuple(checked))


def route_clearance(consist: list[OversizeWagon], sections: list[ClearanceSection],
                    blockades: list[TemporaryBlockade] | None = None,
                    at: datetime | None = None) -> bool:
    return check(consist, sections, blockades, at).allowed


# Public aliases used by callers that name infrastructure records as segments.
OversizeConsistItem = OversizeWagon
ClearanceSegment = ClearanceSection
RouteSection = ClearanceSection
Blockade = TemporaryBlockade
OversizeClearanceFailure = ClearanceFailure
OversizeClearanceDecision = ClearanceDecision
check_route = check
check_oversize_route = check
route_allowed = route_clearance
