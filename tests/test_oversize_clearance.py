import unittest
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.routing.oversize_clearance import (
    ClearanceSection,
    OversizeWagon,
    TemporaryBlockade,
    check,
)
from railforge.wagon.axle_load import Wagon as WeightedWagon

U = timezone.utc


def wagon(wagon_id="W1", gross=90.0, axles=4, height=4.1, left=1.4, right=1.4, front=0.5):
    return OversizeWagon(wagon_id, gross, axles, height, left, right, front)


def section(section_id="S1", height=4.2, width=3.0, gross=100.0, axle=25.0,
            left=None, right=None):
    return ClearanceSection(section_id, height, width, gross, axle, left, right)


class OversizeClearanceTests(unittest.TestCase):
    def test_reports_first_failed_section_and_not_later_failure(self):
        sections = [
            section("S1", height=4.0),
            section("S2", height=3.0),
        ]

        decision = check([wagon(height=4.1)], sections)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.first_failed_section, "S1")
        self.assertEqual(decision.reason, "height")
        self.assertEqual(decision.checked_sections, ("S1",))
        self.assertEqual(decision.first_failure.details["limit_m"], 4.0)

    def test_checks_offset_axle_gross_weight_and_temporary_blockades(self):
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        active_block = TemporaryBlockade("B1", "S1", at, at + timedelta(hours=1))

        self.assertEqual(check([wagon(left=1.6)], [section()]).reason, "lateral_offset")
        self.assertEqual(check([wagon(front=0.8)], [section(axle=16.0)]).reason, "axle_load")
        self.assertEqual(
            check([WeightedWagon("legacy", 30.0, 60.0, 4, 0.8)], [section(axle=30.0)]).reason,
            "axle_load")
        self.assertEqual(check([wagon(gross=101.0)], [section()]).reason, "gross_weight")
        self.assertEqual(
            check([wagon()], [section()], [active_block], at).reason,
            "temporary_blockade")

        future_block = TemporaryBlockade("B2", "S1", at + timedelta(hours=1),
                                         at + timedelta(hours=2))
        self.assertTrue(check([wagon()], [section()], [future_block], at).allowed)

    def test_check_only_publishes_auditable_rejection(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)

        receipt = service.features.check_oversize_clearance(
            "op-check-reject", [wagon(height=4.5)], [section("S1", height=4.0)], [], 3, at)

        self.assertFalse(receipt.result.allowed)
        self.assertEqual(receipt.event.payload["status"], "rejected")
        self.assertEqual(receipt.event.payload["first_failed_section"], "S1")
        self.assertEqual(receipt.event.payload["reference_version"], 3)
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_rejection_is_auditable_and_blocks_path_reservation(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        sections = [section("S1", height=4.0), section("S2")]

        receipt = service.features.reserve_oversize_path(
            "op-reject", [wagon(height=4.1)], sections, [], 0, at)

        self.assertEqual(receipt.event.payload["status"], "rejected")
        self.assertEqual(receipt.event.payload["first_failed_section"], "S1")
        self.assertEqual(receipt.topic, "routing.oversize_clearance.rejected")
        self.assertIsNone(service.workflow().store.read("oversize-path:op-reject"))
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_successful_check_reserves_path_and_stale_version_is_revoked(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        sections = [section("S1"), section("S2")]

        reserved = service.features.reserve_oversize_path(
            "op-reserve", [wagon()], sections, [], 0, at, reference_version=7, subject="T1")
        self.assertEqual(reserved.event.payload["status"], "reserved")
        self.assertEqual(reserved.version, 1)
        self.assertEqual(reserved.event.payload["reference_version"], 7)

        stale = service.features.reserve_oversize_path(
            "op-stale", [wagon()], sections, [], 0, at, reference_version=8, subject="T1")
        self.assertEqual(stale.event.payload["status"], "revoked")
        self.assertEqual(stale.result["actual_version"], 2)
        self.assertEqual(stale.event.payload["reference_version"], 8)
        self.assertEqual(stale.topic, "routing.oversize_path.revoked")
        stored = service.workflow().store.read("oversize-path:T1:S1:S2").value
        self.assertTrue(stored["revoked"])
        self.assertFalse(stored["reserved"])
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())


if __name__ == "__main__":
    unittest.main()
