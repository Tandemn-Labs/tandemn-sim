"""Test contextual-bandit terminal-job handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from contextual_bandit import ContextualBanditPlannerDriver, StoreSnapshot


class ContextualBanditPlannerDriverTest(unittest.TestCase):
    """Verify Contextual Bandit omits terminal jobs from later plans."""

    def test_terminal_jobs_produce_no_actions(self) -> None:
        """Keep terminal jobs out of future placement plans."""
        driver = object.__new__(ContextualBanditPlannerDriver)
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver.config_policy = object()
        driver._select_placements = lambda problems, capacities: {}
        snapshot = StoreSnapshot({}, (), (), (), ({"job_id": "finished"},))

        self.assertEqual(driver._build_actions(snapshot), [])


if __name__ == "__main__":
    unittest.main()
