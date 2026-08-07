"""Test session-sticky global rank routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import Request
from routing import StickyRoundRobinRouter


def request(session_name: str, turn_number: int = 0) -> Request:
    """Build one small request used only for routing tests."""
    return Request(
        request_name=f"{session_name}/turn-{turn_number}",
        session_name=session_name,
        turn_number=turn_number,
        arrives_at_ns=0,
        input_tokens=10,
        output_tokens=10,
        prefix_key=None,
    )


class StickyRoundRobinRouterTest(unittest.TestCase):
    """Verify global rank routing retains session locality across ticks."""

    def test_new_sessions_rotate_and_existing_sessions_stick(self) -> None:
        """Route new sessions round-robin without moving an existing session."""
        router = StickyRoundRobinRouter()
        first = router.route(
            "job",
            (request("s0"), request("s1"), request("s2"), request("s3")),
            {"rank_0": 1.0, "rank_1": 1.0, "rank_2": 1.0},
        )

        self.assertEqual([item.session_name for item in first["rank_0"]], ["s0", "s3"])
        self.assertEqual([item.session_name for item in first["rank_1"]], ["s1"])
        self.assertEqual([item.session_name for item in first["rank_2"]], ["s2"])

        second = router.route(
            "job",
            (request("s1", 1), request("s4")),
            {"rank_0": 1.0, "rank_1": 1.0, "rank_2": 1.0},
        )

        self.assertEqual([item.session_name for item in second["rank_1"]], ["s1", "s4"])

    def test_removed_rank_reassigns_its_sessions(self) -> None:
        """Drop a stale assignment so the next session turn gets a live rank."""
        router = StickyRoundRobinRouter()
        router.route("job", (request("s0"), request("s1")), {"rank_0": 1.0, "rank_1": 1.0})
        router.remove_rank("job", "rank_1")

        routed = router.route("job", (request("s1", 1),), {"rank_0": 1.0, "rank_2": 1.0})

        self.assertEqual([item.session_name for item in routed["rank_0"]], ["s1"])

    def test_negative_share_is_rejected(self) -> None:
        """Reject an invalid Koi rank traffic share."""
        with self.assertRaises(ValueError):
            StickyRoundRobinRouter().route("job", (), {"rank_0": -1.0})

    def test_new_sessions_follow_koi_traffic_shares(self) -> None:
        """Route ten new sessions in the exact 60/40 Koi rank proportion."""
        routed = StickyRoundRobinRouter().route(
            "job",
            tuple(request(f"s{number}") for number in range(10)),
            {"rank_0": 0.6, "rank_1": 0.4},
        )

        self.assertEqual(len(routed["rank_0"]), 6)
        self.assertEqual(len(routed["rank_1"]), 4)


if __name__ == "__main__":
    unittest.main()
