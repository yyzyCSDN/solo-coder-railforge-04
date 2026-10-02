import unittest
from datetime import datetime, timezone, timedelta

from railforge.api import RailForgeService
from railforge.routing.gauge_clearance import (
    ConsistVehicle,
    GaugeCheckRequest,
    GaugePathRegistry,
    GaugeSection,
    PlannedSection,
    TemporaryBlockade,
    verify,
)

U = timezone.utc
AT = datetime(2026, 1, 1, 8, tzinfo=U)
H = timedelta(hours=1)


def make_request(**overrides):
    base = dict(
        train="T9001",
        consist=(
            ConsistVehicle(1, "W1", 3.0, 2.5, (20.0, 20.0, 20.0, 20.0)),
            ConsistVehicle(2, "W2", 3.2, 2.5, (24.0, 18.0, 22.0, 20.0)),  # eccentric load
        ),
        route=(
            PlannedSection("S1", AT, AT + H),
            PlannedSection("S2", AT + H, AT + 2 * H),
            PlannedSection("S3", AT + 2 * H, AT + 3 * H),
        ),
        clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S2", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        ),
        blockades=(),
        data_version=1,
    )
    base.update(overrides)
    return GaugeCheckRequest(**base)


class PureVerifyTests(unittest.TestCase):
    def test_pass_checks_every_section(self):
        verdict = verify(make_request())
        self.assertTrue(verdict.passed)
        self.assertIsNone(verdict.first_failure)
        self.assertEqual(verdict.checked_sections, ("S1", "S2", "S3"))
        self.assertEqual(verdict.sections, ("S1", "S2", "S3"))
        self.assertEqual(verdict.data_version, 1)

    def test_first_failed_section_stops_scan(self):
        # S2 fails on height and S3 would fail on axle; only S2 may be reported.
        verdict = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S2", 3.1, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 23.0, 400.0),
        )))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.first_failure.section, "S2")
        self.assertEqual(verdict.first_failure.reason, "height")
        self.assertEqual(verdict.checked_sections, ("S1", "S2"))

    def test_eccentric_peak_axle_is_checked(self):
        # W2 carries 24t on one axle due to offset loading.
        verdict = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 23.0, 400.0),
            GaugeSection("S2", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        )))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.first_failure.section, "S1")
        self.assertEqual(verdict.first_failure.reason, "axle")

    def test_gross_and_width_and_missing_clearance(self):
        gross = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 100.0),
            GaugeSection("S2", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        )))
        self.assertEqual(gross.first_failure.reason, "gross")
        width = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 2.4, 25.0, 400.0),
            GaugeSection("S2", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        )))
        self.assertEqual(width.first_failure.reason, "width")
        missing = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        )))
        self.assertEqual(missing.first_failure.section, "S2")
        self.assertEqual(missing.first_failure.reason, "clearance-missing")

    def test_temporary_blockade_overlap(self):
        hit = verify(make_request(blockades=(
            TemporaryBlockade("B1", "S2", AT + timedelta(minutes=90), AT + 2 * H),
        )))
        self.assertFalse(hit.passed)
        self.assertEqual(hit.first_failure.section, "S2")
        self.assertEqual(hit.first_failure.reason, "blockade")
        self.assertIn("B1", hit.first_failure.detail)
        # Touching windows do not overlap; blockade on another section is irrelevant.
        ok = verify(make_request(blockades=(
            TemporaryBlockade("B2", "S2", AT + 2 * H, AT + 3 * H),
            TemporaryBlockade("B3", "S9", AT, AT + 3 * H),
        )))
        self.assertTrue(ok.passed)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            verify(make_request(consist=()))
        with self.assertRaises(ValueError):
            verify(make_request(consist=(
                ConsistVehicle(1, "W1", 3.0, 2.5, (20.0, 20.0)),
                ConsistVehicle(3, "W3", 3.0, 2.5, (20.0, 20.0)),
            )))
        with self.assertRaises(ValueError):
            verify(make_request(consist=(
                ConsistVehicle(1, "W1", 3.0, 2.5, (20.0, 20.0)),
                ConsistVehicle(2, "W1", 3.0, 2.5, (20.0, 20.0)),
            )))
        with self.assertRaises(ValueError):
            verify(make_request(consist=(ConsistVehicle(1, "W1", 3.0, 2.5, (-1.0, 20.0)),)))
        with self.assertRaises(ValueError):
            verify(make_request(route=()))
        with self.assertRaises(ValueError):
            verify(make_request(data_version=0))
        with self.assertRaises(ValueError):
            verify(make_request(blockades=(TemporaryBlockade("B", "S1", AT + H, AT),)))
        with self.assertRaises(ValueError):
            verify(make_request(route=(PlannedSection("S1", AT + H, AT),)))


class WorkflowTests(unittest.TestCase):
    def test_rejection_event_is_auditable(self):
        service = RailForgeService()
        request = make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S2", 3.1, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        ))
        receipt = service.workflow().check_gauge_clearance("op-rej", request, AT)
        self.assertEqual(receipt.event.topic, "routing.gauge.rejected")
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["first_failed_section"], "S2")
        self.assertEqual(receipt.result.first_failure.section, "S2")
        stored = service.workflow().store.read("T9001")
        self.assertIsNotNone(stored)
        self.assertFalse(stored.value["passed"])
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_reserve_only_after_pass(self):
        service = RailForgeService()
        with self.assertRaises(ValueError):
            service.workflow().reserve_gauge_path("op-r", "RSV1", "T9001",
                                                  ["S1", "S2", "S3"], 1, AT)
        receipt = service.workflow().check_gauge_clearance("op-c", make_request(), AT)
        self.assertEqual(receipt.event.topic, "routing.gauge.cleared")
        reserved = service.workflow().reserve_gauge_path("op-r", "RSV1", "T9001",
                                                         ["S1", "S2", "S3"], 1, AT)
        self.assertEqual(reserved.event.topic, "routing.gauge.path_reserved")
        self.assertEqual(reserved.result.status, "active")
        self.assertEqual([r.reservation_id for r in service.workflow().gauge_paths.active()],
                         ["RSV1"])

    def test_reserve_at_stale_version_rejected(self):
        service = RailForgeService()
        service.workflow().check_gauge_clearance("op-v1", make_request(), AT)
        service.workflow().check_gauge_clearance("op-v2", make_request(data_version=2), AT)
        with self.assertRaises(ValueError):
            service.workflow().reserve_gauge_path("op-r", "RSV1", "T9001",
                                                  ["S1", "S2", "S3"], 1, AT)
        ok = service.workflow().reserve_gauge_path("op-r2", "RSV2", "T9001",
                                                   ["S1", "S2", "S3"], 2, AT)
        self.assertEqual(ok.result.data_version, 2)

    def test_version_conflict_revokes_reservation(self):
        service = RailForgeService()
        reserved = service.features.clear_out_of_gauge_route("op-f", make_request(), "RSV1", AT)
        self.assertEqual(reserved.event.topic, "routing.gauge.path_reserved")
        published = service.features.publish_gauge_data_version("op-pub", 2, AT + 4 * H)
        self.assertEqual(published.result["revoked"], ["RSV1"])
        reservation = service.workflow().gauge_paths.get("RSV1")
        self.assertEqual(reservation.status, "revoked")
        self.assertEqual(reservation.revoke_reason, "version-conflict")
        self.assertEqual(service.workflow().gauge_paths.active(), [])
        topics = [e.topic for e in service.workflow().stream.snapshot()]
        self.assertIn("routing.gauge.reservation_revoked", topics)
        self.assertIn("routing.gauge.data_published", topics)
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_revalidate_confirms_matching_version(self):
        service = RailForgeService()
        service.features.clear_out_of_gauge_route("op-f", make_request(), "RSV1", AT)
        confirmed = service.workflow().revalidate_gauge_path("op-re", "RSV1", 1, AT)
        self.assertEqual(confirmed.event.topic, "routing.gauge.reservation_confirmed")
        revoked = service.workflow().revalidate_gauge_path("op-re2", "RSV1", 7, AT)
        self.assertEqual(revoked.event.topic, "routing.gauge.reservation_revoked")
        self.assertEqual(revoked.result.status, "revoked")
        with self.assertRaises(ValueError):
            service.workflow().revalidate_gauge_path("op-re3", "NOPE", 1, AT)

    def test_feature_rejects_without_reserving(self):
        service = RailForgeService()
        bad = make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S2", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 150.0),
        ))
        receipt = service.features.clear_out_of_gauge_route("op-x", bad, "RSV9", AT)
        self.assertEqual(receipt.event.topic, "routing.gauge.rejected")
        self.assertEqual(receipt.result.first_failure.section, "S3")
        self.assertIsNone(service.workflow().gauge_paths.get("RSV9"))

    def test_operations_are_idempotent(self):
        service = RailForgeService()
        first = service.workflow().check_gauge_clearance("op-i", make_request(), AT)
        self.assertIs(first, service.workflow().check_gauge_clearance("op-i", make_request(), AT))
        r1 = service.workflow().reserve_gauge_path("op-ir", "RSV1", "T9001",
                                                   ["S1", "S2", "S3"], 1, AT)
        self.assertIs(r1, service.workflow().reserve_gauge_path("op-ir", "RSV1", "T9001",
                                                                ["S1", "S2", "S3"], 1, AT))

    def test_registry_rejects_unpassed_verdict(self):
        registry = GaugePathRegistry()
        verdict = verify(make_request(clearances=(
            GaugeSection("S1", 4.0, 3.0, 25.0, 400.0),
            GaugeSection("S2", 3.1, 3.0, 25.0, 400.0),
            GaugeSection("S3", 4.0, 3.0, 25.0, 400.0),
        )))
        with self.assertRaises(ValueError):
            registry.record_pass(verdict)


if __name__ == "__main__":
    unittest.main()
