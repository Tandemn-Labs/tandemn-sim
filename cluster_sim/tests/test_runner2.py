from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from runner2 import PlanRunner


class _Jobs:
    def __init__(self) -> None:
        self.transitions = []

    def running_jobs(self, user_id):
        return [SimpleNamespace(job=SimpleNamespace(job_id="job_1"), ranks=[])]

    def waiting_jobs(self, user_id):
        return []

    def paused_jobs(self, user_id):
        return []

    def transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))


class PlanRunnerTest(unittest.TestCase):
    def test_hard_stop_leaves_rank_teardown_to_orca(self) -> None:
        runner = PlanRunner.__new__(PlanRunner)
        runner.user_id = "user_1"
        runner.jobs = _Jobs()

        runner._finish_submitted_jobs("SIMULATION_HARD_STOP")

        self.assertEqual(runner.jobs.transitions[0][0][0], "job_1")


if __name__ == "__main__":
    unittest.main()
