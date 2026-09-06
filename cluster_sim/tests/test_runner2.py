from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import BatchJob, OnlineJob, Scenario  # noqa: E402
from runner2 import LogicalClock, PlanRunner, _clock_start  # noqa: E402


class _Jobs:
    def __init__(self) -> None:
        self.transitions = []

    def transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))


class PlanRunnerTest(unittest.TestCase):
    @mock.patch("runner2.datetime")
    def test_default_clock_start_uses_real_utc(
        self, mocked_datetime: mock.Mock
    ) -> None:
        now = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
        mocked_datetime.now.return_value = now

        self.assertEqual(_clock_start(None), now)

        mocked_datetime.now.assert_called_once_with(UTC)

    def test_koi_mode_skips_planner_poll(self) -> None:
        runner = PlanRunner.__new__(PlanRunner)
        runner.scenario = Scenario("test", 5, 10, (), ())
        runner.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        runner._submit_jobs_before = mock.Mock()
        runner._finish_jobs_due = mock.Mock()
        runner.planner = mock.Mock()

        runner.planner = None
        result = runner.run_tick(0)

        runner._submit_jobs_before.assert_any_call(1)
        runner._finish_jobs_due.assert_called_once_with(0)
        self.assertEqual(result.plans, ())

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

    def test_hard_stop_finishes_jobs_with_later_scheduled_stops(self) -> None:
        online = OnlineJob("online", "online-id", "model", 0, 20, {})
        runner = PlanRunner.__new__(PlanRunner)
        runner.scenario = Scenario("test", 5, 10, (online,), ())
        runner.jobs = _Jobs()
        runner._submitted_jobs = {"online"}
        runner._finished_jobs = set()

        runner._finish_jobs_due(10, force=True)

        args, kwargs = runner.jobs.transitions[0]
        self.assertEqual(args[0], "online-id")
        self.assertEqual(kwargs["finish_reason"], "SIMULATION_HARD_STOP")

    @mock.patch("runner2._submit_waiting_job")
    @mock.patch("tandemn_system_data.clients.ModelCatalogStore")
    @mock.patch("runner2.datetime")
    def test_submission_uses_real_utc_not_the_logical_clock(
        self,
        mocked_datetime: mock.Mock,
        catalog_store: mock.Mock,
        submit_waiting_job: mock.Mock,
    ) -> None:
        submitted_at = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
        mocked_datetime.now.return_value = submitted_at
        catalog_store.return_value.get.return_value = None
        job = OnlineJob("online", "online-id", "model", 0, 5, {})
        runner = PlanRunner.__new__(PlanRunner)
        runner.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        runner.postgres = mock.sentinel.postgres
        runner.user_id = "tenant"
        runner.jobs = mock.sentinel.jobs
        runner._submitted_jobs = set()

        runner._submit(job)

        mocked_datetime.now.assert_called_once_with(UTC)
        submit_waiting_job.assert_called_once_with(
            job,
            "tenant",
            mock.sentinel.jobs,
            submitted_at,
            prefix_cache_enabled=False,
        )
        self.assertEqual(runner._submitted_jobs, {"online"})

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
