from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import BatchJob, OnlineJob, Scenario  # noqa: E402
from runner2 import LogicalClock, PlanRunner  # noqa: E402


class _Jobs:
    def __init__(self) -> None:
        self.transitions = []

    def transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))


class PlanRunnerTest(unittest.TestCase):
    def test_finishes_each_job_only_at_its_scheduled_stop(self) -> None:
        online = OnlineJob("online", "online-id", "model", 0, 5, {})
        batch = BatchJob("batch", "batch-id", "model", 0, 4, 10, {})
        runner = PlanRunner.__new__(PlanRunner)
        runner.scenario = Scenario("test", 5, 10, (online,), (batch,))
        runner.jobs = _Jobs()
        runner._submitted_jobs = {"online", "batch"}
        runner._finished_jobs = set()

        runner._finish_jobs_due(5)
        runner._finish_jobs_due(10)
        runner._finish_jobs_due(10)

        self.assertEqual(
            [transition[0][0] for transition in runner.jobs.transitions],
            ["online-id", "batch-id"],
        )

    def test_tick_limit_does_not_finish_jobs_early(self) -> None:
        online = OnlineJob("online", "online-id", "model", 0, 10_000_000, {})
        runner = PlanRunner.__new__(PlanRunner)
        runner.scenario = Scenario("test", 5_000_000, 10_000_000, (online,), ())
        runner.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        runner.jobs = _Jobs()
        runner._submitted_jobs = {"online"}
        runner._finished_jobs = set()
        runner.run_tick = lambda _tick: None

        runner.run(max_ticks=1)

        self.assertEqual(runner.jobs.transitions, [])

    @mock.patch("runner2.sleep")
    @mock.patch("runner2.monotonic", side_effect=[0.0, 0.0, 1.0])
    def test_waits_until_the_wall_clock_stop(self, _clock, sleep) -> None:
        online = OnlineJob("online", "online-id", "model", 0, 2_000_000_000, {})
        runner = PlanRunner.__new__(PlanRunner)
        runner.scenario = Scenario("test", 1_000_000_000, 2_000_000_000, (online,), ())
        runner.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        runner.jobs = _Jobs()
        runner._submitted_jobs = {"online"}
        runner._finished_jobs = set()
        runner.run_tick = lambda _tick: None

        runner.run()

        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(1.0)])
        self.assertEqual(len(runner.jobs.transitions), 1)


if __name__ == "__main__":
    unittest.main()
