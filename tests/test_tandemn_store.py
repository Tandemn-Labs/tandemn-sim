"""Method-neutral Tandemn Store adapter checks."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from data_generation.tandemn_store import (
    TandemnPlanPublisher,
    TandemnStoreAdapter,
    TandemnStoreError,
)


class FakeModel:
    """Expose the Pydantic serialization boundary used by Store models."""

    def __init__(self, **values: object) -> None:
        self.values = values

    def model_dump(self, *, mode: str) -> dict[str, object]:
        """Return a JSON-shaped model document."""
        if mode != "json":
            raise AssertionError("state reader must request JSON serialization")
        return self.values


class FakeResourceMap:
    """Provide one instance-atomic H100 environment."""

    version = 7
    updated_at = "2026-08-07T18:00:00+00:00"
    market = ["reserved"]

    def scheduling_summary(self) -> dict[str, dict[str, object]]:
        """Return total capacity in Tandemn Store's flattened shape."""
        return {
            "reserved|aws|us-east-1|use1-az1|H100": {
                "total": 16,
                "total_instances": 2,
                "gpu_type": "H100",
                "market": "reserved",
                "cloud": "aws",
                "region": "us-east-1",
                "zone": "use1-az1",
                "pools": [
                    {
                        "instance_type": "p5.48xlarge",
                        "total": 16,
                        "total_instances": 2,
                        "gpus_per_instance": 8,
                    }
                ],
            }
        }


class FakeJobStore:
    """Return one job in each lifecycle group."""

    def running_jobs(self, user_id: str) -> list[FakeModel]:
        """Return one active job with one active chain."""
        return [
            FakeModel(
                job={
                    "job_id": "job-running",
                    "user_id": user_id,
                    "status": "running",
                },
                chains=[
                    {
                        "chain_id": "chain-running",
                        "status": "running",
                        "target_node": "reserved|aws|us-east-1|use1-az1|H100",
                        "shape_json": {
                            "count": 4,
                            "instance_type": "p5.48xlarge",
                        },
                    }
                ],
            )
        ]

    def waiting_jobs(self, user_id: str) -> list[FakeModel]:
        """Return one pending job."""
        return [FakeModel(job_id="job-pending", user_id=user_id, status="waiting")]

    def paused_jobs(self, user_id: str) -> list[FakeModel]:
        """Return one paused job."""
        return [FakeModel(job_id="job-paused", user_id=user_id, status="paused")]

    def finished_jobs(self, user_id: str) -> list[FakeModel]:
        """Return one successfully completed job."""
        return [
            FakeModel(
                job_id="job-completed",
                user_id=user_id,
                status="finished",
                finish_reason=None,
            )
        ]


class LegacyFakeJobStore(FakeJobStore):
    """Match the current Store API without a public finished_jobs read."""

    finished_jobs = None

    def __init__(self) -> None:
        self._jobs_with_status = mock.Mock(return_value=[])


class MultiUserJobStore:
    """Return distinct jobs for every requested Store user."""

    def running_jobs(self, user_id: str) -> list[FakeModel]:
        return []

    def waiting_jobs(self, user_id: str) -> list[FakeModel]:
        return [FakeModel(job_id=f"job-{user_id}", user_id=user_id, status="waiting")]

    def paused_jobs(self, user_id: str) -> list[FakeModel]:
        return []

    def finished_jobs(self, user_id: str) -> list[FakeModel]:
        return []


class FakePlanStore:
    """Record plans written through the production publisher boundary."""

    def __init__(self) -> None:
        self.plans: list[object] = []

    def create(self, plan: object) -> object:
        self.plans.append(plan)
        return plan


class TandemnStoreAdapterTests(unittest.TestCase):
    """Verify the shared cluster snapshot."""

    def test_reads_resources_and_all_required_job_groups(self) -> None:
        job_store = FakeJobStore()
        clients = SimpleNamespace(
            PostgresClient=lambda: object(),
            JobStore=lambda _client: job_store,
            ResourceMapStore=lambda _client, user_id: SimpleNamespace(
                get=lambda: FakeResourceMap()
            ),
        )
        models = SimpleNamespace(JobStatus=SimpleNamespace(FINISHED="finished"))

        with mock.patch(
            "data_generation.tandemn_store.import_module",
            side_effect=(clients, models),
        ):
            snapshot = TandemnStoreAdapter("usr-test").read_cluster_state()

        environment = snapshot.resources["reserved|aws|us-east-1|use1-az1|H100"]
        self.assertEqual(environment["total"], 16)
        self.assertEqual(environment["used"], 8)
        self.assertEqual(environment["free"], 8)
        self.assertEqual(snapshot.running_jobs[0]["job_id"], "job-running")
        self.assertEqual(
            snapshot.running_jobs[0]["active_chains"][0]["chain_id"],
            "chain-running",
        )
        self.assertEqual(snapshot.waiting_jobs[0]["job_id"], "job-pending")
        self.assertEqual(snapshot.paused_jobs[0]["job_id"], "job-paused")
        self.assertEqual(snapshot.finished_jobs[0]["job_id"], "job-completed")

        state = snapshot.to_dict()
        self.assertEqual(state["resource_map"]["version"], 7)
        self.assertEqual(
            set(state["jobs"]),
            {"by_id", "running", "waiting", "paused", "finished"},
        )
        self.assertEqual(
            state["jobs"]["by_id"]["job-running"]["status"],
            "running",
        )
        json.dumps(state)

    def test_uses_current_store_finished_status_reader(self) -> None:
        job_store = LegacyFakeJobStore()
        clients = SimpleNamespace(
            PostgresClient=lambda: object(),
            JobStore=lambda _client: job_store,
            ResourceMapStore=lambda _client, user_id: SimpleNamespace(
                get=lambda: FakeResourceMap()
            ),
        )
        models = SimpleNamespace(JobStatus=SimpleNamespace(FINISHED="finished"))

        with mock.patch(
            "data_generation.tandemn_store.import_module",
            side_effect=(clients, models),
        ):
            TandemnStoreAdapter("usr-test").read_cluster_state()

        job_store._jobs_with_status.assert_called_once_with("usr-test", "finished")

    def test_reads_jobs_for_every_benchmark_user(self) -> None:
        job_store = MultiUserJobStore()
        clients = SimpleNamespace(
            PostgresClient=lambda: object(),
            JobStore=lambda _client: job_store,
            ResourceMapStore=lambda _client, user_id: SimpleNamespace(
                get=lambda: FakeResourceMap()
            ),
        )
        models = SimpleNamespace(JobStatus=SimpleNamespace(FINISHED="finished"))

        with mock.patch(
            "data_generation.tandemn_store.import_module",
            side_effect=(clients, models),
        ):
            snapshot = TandemnStoreAdapter(
                "operator",
                job_user_ids=("tenant-a", "tenant-b"),
            ).read_cluster_state()

        self.assertEqual(
            set(snapshot.jobs_by_id()),
            {"job-operator", "job-tenant-a", "job-tenant-b"},
        )

    def test_publishes_canonical_plan_for_orca(self) -> None:
        plan_store = FakePlanStore()
        clients = SimpleNamespace(
            PostgresClient=lambda: object(),
            PlanStore=lambda _client: plan_store,
        )

        def plan_action(**values: object) -> object:
            return SimpleNamespace(**values)

        def plan(**values: object) -> object:
            return SimpleNamespace(plan_id="plan-test", **values)

        models = SimpleNamespace(PlanAction=plan_action, Plan=plan)
        with mock.patch(
            "data_generation.tandemn_store.import_module",
            side_effect=(clients, models),
        ):
            publisher = TandemnPlanPublisher("operator")
            plan_id = publisher.publish_plan(
                {
                    "tick_rationale": "test tick",
                    "actions": [
                        {
                            "job_id": "job-waiting",
                            "type": "place",
                            "ladder": [{"env": ["reserved", "aws"]}],
                            "target_tps": 12.0,
                            "tenant_id": "ignored-by-store-schema",
                        }
                    ],
                },
                "gpt_only",
            )

        self.assertEqual(plan_id, "plan-test")
        stored_plan = plan_store.plans[0]
        self.assertEqual(stored_plan.user_id, "operator")
        self.assertEqual(stored_plan.koi_version, "gpt_only")
        self.assertEqual(stored_plan.actions[0].job_id, "job-waiting")

    def test_reports_missing_store_dependency(self) -> None:
        with mock.patch(
            "data_generation.tandemn_store.import_module",
            side_effect=ImportError,
        ):
            with self.assertRaisesRegex(
                TandemnStoreError,
                "tandemn-store is required",
            ):
                TandemnStoreAdapter("usr-test")


if __name__ == "__main__":
    unittest.main()
