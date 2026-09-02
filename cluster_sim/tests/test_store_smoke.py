"""Opt-in Store and Orca lifecycle smoke test for the cluster simulator."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TANDEMN_ROOT = ROOT.parent
sys.path[:0] = [
    str(ROOT / "cluster_sim"),
    str(TANDEMN_ROOT / "tandemn-store" / "src"),
    str(TANDEMN_ROOT / "Tandemn-orca" / "src"),
]

from models import OnlineJob, Scenario
from planning import ScriptedPlannerDriver
from runner import (
    ClusterSimulator,
    LogicalClock,
    active_ranks,
    apply_pending_plans,
    submit_waiting_job,
    verify_store_user,
)
from scenario import load_scenario


@unittest.skipUnless(
    os.getenv("CLUSTER_SIM_STORE_SMOKE") == "1",
    "set CLUSTER_SIM_STORE_SMOKE=1 with a migrated local Tandemn Store",
)
class StoreSmokeTest(unittest.TestCase):
    """Verify waiting job -> Store plan -> Orca rank lifecycle."""

    def test_place_creates_running_job_and_rank(self) -> None:
        """Place one YAML-derived online job through Store and Noop Orca."""
        from tandemn_system_data.clients import JobStore, PlanStore, PostgresClient
        from tandemn_system_data.db.orm import UserRow
        from tandemn_system_data.ids import new_job_id, new_user_id
        from tandemn_system_data.models import ActionType, PlanAction
        from tandemn_system_data.models._base import utc_now
        from tandemn_system_data.models.enums import JobStatus

        postgres = PostgresClient()
        user_id = new_user_id()
        with postgres.begin() as session:
            session.add(UserRow(user_id=user_id, name="cluster-sim-smoke", created_at=utc_now()))

        try:
            scenario = load_scenario(ROOT / "benchmarks/simulation_3h/overview.yaml")
            job = replace(scenario.online_jobs[0], store_job_id=new_job_id())
            jobs = JobStore(postgres)
            plans = PlanStore(postgres)
            submitted_at = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
            verify_store_user(user_id, postgres)
            submit_waiting_job(job, user_id, jobs, submitted_at)
            self.assertEqual(jobs.get(job.store_job_id).created_at, submitted_at)

            action = PlanAction(
                job_id=job.store_job_id,
                type=ActionType.PLACE,
                target_p99_ttft_ms=600,
                target_p99_tpot_ms=60,
                ladder=[
                    {
                        "role": "aggregate",
                        "rank_id": "rank_0",
                        "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
                        "config": {
                            "instance_type": "sim.h100.8gpu",
                            "gpu_type": "H100-SXM-80GB",
                            "gpu_count": 1,
                            "tp": 1,
                            "pp": 1,
                        },
                        "n_replicas": 1,
                    }
                ],
            )
            result = ScriptedPlannerDriver(
                user_id,
                plans,
                {0: (action,)},
            ).create_plans(0, utc_now())

            self.assertEqual(len(result.plans), 1)
            self.assertEqual(apply_pending_plans(user_id, postgres, result.plans), 1)
            self.assertEqual(plans.get(result.plans[0].plan_id).status, "applied")
            self.assertEqual(jobs.get(job.store_job_id).status, JobStatus.RUNNING)
            self.assertEqual(len(active_ranks(job.store_job_id, jobs)), 1)
        finally:
            with postgres.begin() as session:
                user = session.get(UserRow, user_id)
                if user is not None:
                    session.delete(user)

    def test_coordinator_submits_between_ticks_with_logical_time(self) -> None:
        """Submit a T=6 job after the T=5 plan and before the T=10 plan."""
        from tandemn_system_data.clients import JobStore, PlanStore, PostgresClient
        from tandemn_system_data.db.orm import UserRow
        from tandemn_system_data.ids import new_job_id, new_user_id
        from tandemn_system_data.models import ActionType, PlanAction
        from tandemn_system_data.models._base import utc_now
        from tandemn_system_data.models.enums import JobStatus

        postgres = PostgresClient()
        user_id = new_user_id()
        with postgres.begin() as session:
            session.add(UserRow(user_id=user_id, name="cluster-sim-coordinator", created_at=utc_now()))

        try:
            source = load_scenario(ROOT / "benchmarks/simulation_3h/overview.yaml")
            first = replace(
                source.online_jobs[0],
                store_job_id=new_job_id(),
                starts_at_ns=0,
                stops_at_ns=900_000_000_000,
            )
            second = replace(
                source.online_jobs[1],
                store_job_id=new_job_id(),
                starts_at_ns=360_000_000_000,
                stops_at_ns=900_000_000_000,
            )
            scenario = Scenario("coordinator-smoke", 300_000_000_000, 600_000_000_000, (first, second), ())
            action = PlanAction(
                job_id=first.store_job_id,
                type=ActionType.PLACE,
                target_p99_ttft_ms=600,
                target_p99_tpot_ms=60,
                ladder=[
                    {
                        "role": "aggregate",
                        "rank_id": "rank_0",
                        "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
                        "config": {
                            "instance_type": "sim.h100.8gpu",
                            "gpu_type": "H100-SXM-80GB",
                            "gpu_count": 1,
                            "tp": 1,
                            "pp": 1,
                        },
                        "n_replicas": 1,
                    }
                ],
            )
            epoch = datetime(2026, 1, 1, tzinfo=UTC)
            jobs = JobStore(postgres)
            simulator = ClusterSimulator(
                scenario,
                user_id,
                postgres,
                ScriptedPlannerDriver(user_id, PlanStore(postgres), {0: (action,)}),
                LogicalClock(epoch),
            )

            simulator.run_tick(0)
            self.assertEqual(jobs.get(first.store_job_id).created_at, epoch)
            self.assertEqual(jobs.get(first.store_job_id).status, JobStatus.RUNNING)
            self.assertIsNone(jobs.get(second.store_job_id))

            simulator.run_tick(1)
            self.assertEqual(
                jobs.get(second.store_job_id).created_at,
                datetime(2026, 1, 1, 0, 6, tzinfo=UTC),
            )
            self.assertEqual(jobs.get(second.store_job_id).status, JobStatus.WAITING)
            self.assertEqual(simulator.clock.now(), datetime(2026, 1, 1, 0, 10, tzinfo=UTC))
        finally:
            with postgres.begin() as session:
                user = session.get(UserRow, user_id)
                if user is not None:
                    session.delete(user)


if __name__ == "__main__":
    unittest.main()
