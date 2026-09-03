"""Test the Store-only simulation plan executor."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from simulation_orca import apply_pending_plans, normalize_job_spec
from tandemn_system_data.models.enums import (
    ActionType,
    JobKind,
    JobStatus,
    RankRole,
    RankStatus,
    ReasonCode,
)
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank

from data_generation.real_scenario import (
    Manifest,
    ManifestJob,
    State,
    StoreJob,
    run_scenario,
)

OLD_RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
NEW_RANK_ID = "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8"


class _JobStore:
    def __init__(self, statuses: dict[str, JobStatus], ranks: list[Rank] = ()) -> None:
        self.statuses = dict(statuses)
        self.ranks = {rank.rank_id: rank for rank in ranks}

    def get(self, job_id: str):
        status = self.statuses.get(job_id)
        if status is None:
            return None
        return SimpleNamespace(
            status=status,
            kind=JobKind.ONLINE,
            spec_json={"model_id": "model-a"},
        )

    def transition(self, job_id, to, expected):
        if self.statuses.get(job_id) not in expected:
            return False
        self.statuses[job_id] = to
        return True

    def launch_ranks(self, ranks):
        for rank in ranks:
            self.ranks[rank.rank_id] = rank
        return ranks

    def active_ranks(self, job_id):
        return [
            rank
            for rank in self.ranks.values()
            if rank.job_id == job_id
            and rank.status in (RankStatus.LAUNCHING, RankStatus.RUNNING)
        ]

    def set_rank_status(self, rank_id, to, expected, *, reason_code=None):
        rank = self.ranks.get(rank_id)
        if rank is None or rank.status not in expected:
            return False
        self.ranks[rank_id] = rank.model_copy(
            update={"status": to, "reason_code": reason_code}
        )
        return True


class _PlanStore:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.applied = []

    def unapplied(self, _user_id):
        return self.plans

    def mark_applied(self, plan_id):
        self.applied.append(plan_id)
        return True


class SimulationOrcaTest(unittest.TestCase):
    def test_normalize_job_spec_derives_traffic_from_concurrency(self) -> None:
        spec = normalize_job_spec(
            {
                "model_id": "model-a",
                "concurrency": 20,
                "osl_token_avg": 10,
                "target_p99_ttft_ms": 1_000,
                "target_p99_tpot_ms": 100,
            },
            JobKind.ONLINE,
        )

        self.assertEqual(spec["job_features"]["_traffic_mode"], "concurrency")
        self.assertEqual(spec["job_features"]["max_concurrent_streaming"], 20)
        self.assertEqual(spec["job_features"]["request_arrival_rate"], 10.0)

    def test_place_materializes_a_launching_rank(self) -> None:
        plan = Plan(
            user_id="user-a",
            actions=[
                PlanAction(
                    job_id="job-a",
                    type=ActionType.PLACE,
                    ladder=[_ladder(OLD_RANK_ID)],
                )
            ],
        )
        jobs = _JobStore({"job-a": JobStatus.WAITING})
        plans = _PlanStore([plan])

        with (
            patch("simulation_orca.JobStore", return_value=jobs),
            patch("simulation_orca.PlanStore", return_value=plans),
        ):
            self.assertEqual(apply_pending_plans("user-a", object()), 1)

        rank = jobs.ranks[OLD_RANK_ID]
        self.assertEqual(jobs.statuses["job-a"], JobStatus.RUNNING)
        self.assertEqual(rank.status, RankStatus.LAUNCHING)
        self.assertEqual(rank.shape_json["model_id"], "model-a")
        self.assertEqual(plans.applied, [plan.plan_id])

    def test_place_hands_running_store_job_to_real_scenario(self) -> None:
        plan = Plan(
            user_id="user-a",
            actions=[
                PlanAction(
                    job_id="job-a",
                    type=ActionType.PLACE,
                    ladder=[_ladder(OLD_RANK_ID)],
                )
            ],
        )
        jobs = _JobStore({"job-a": JobStatus.WAITING})

        with (
            patch("simulation_orca.JobStore", return_value=jobs),
            patch("simulation_orca.PlanStore", return_value=_PlanStore([plan])),
        ):
            apply_pending_plans("user-a", object())

        store = SimpleNamespace(
            list_jobs=lambda _user_ids: (
                StoreJob("job-a", jobs.statuses["job-a"], "authored-job"),
            ),
            finish_job=lambda *_args: True,
        )
        manifest = Manifest(
            "handoff",
            0.01,
            (
                ManifestJob(
                    "authored-job",
                    "user-a",
                    ROOT / "benchmarks" / "15min_real" / "jobs" / "code-chat.yaml",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "data_generation.real_scenario.resolve_endpoint",
                    return_value="http://127.0.0.1:28001",
                ),
                patch("data_generation.real_scenario.check_ready", return_value=True),
                patch(
                    "data_generation.real_scenario._launch",
                    side_effect=lambda tracked, *_args: setattr(
                        tracked, "state", State.DONE
                    ),
                ) as launch,
            ):
                self.assertEqual(
                    run_scenario(
                        manifest,
                        store,
                        poll_interval=0.01,
                        ready_timeout=1,
                        output_dir=Path(directory),
                        router_config_dir=Path(directory),
                        source_cloud="sim",
                        source_region="region-1",
                        source_zone="zone-1",
                    ),
                    0,
                )

        self.assertEqual(launch.call_args.args[0].store_job_id, "job-a")

    def test_swap_then_preempt_replaces_and_stops_ranks(self) -> None:
        old_rank = Rank(
            rank_id=OLD_RANK_ID,
            job_id="job-a",
            plan_id="plan-old",
            role=RankRole.AGGREGATE,
            shape_json={"count": 1},
            n_replicas=1,
            status=RankStatus.RUNNING,
        )
        plan = Plan(
            user_id="user-a",
            actions=[
                PlanAction(
                    job_id="job-a",
                    type=ActionType.SWAP,
                    ladder=[_ladder(NEW_RANK_ID)],
                ),
                PlanAction(job_id="job-a", type=ActionType.PREEMPT),
            ],
        )
        jobs = _JobStore({"job-a": JobStatus.RUNNING}, [old_rank])

        with (
            patch("simulation_orca.JobStore", return_value=jobs),
            patch("simulation_orca.PlanStore", return_value=_PlanStore([plan])),
        ):
            apply_pending_plans("user-a", object())

        self.assertEqual(jobs.statuses["job-a"], JobStatus.PAUSED)
        self.assertEqual(jobs.ranks[OLD_RANK_ID].status, RankStatus.STOPPED)
        self.assertEqual(jobs.ranks[NEW_RANK_ID].status, RankStatus.STOPPED)
        self.assertEqual(jobs.ranks[NEW_RANK_ID].reason_code, ReasonCode.PREEMPTED)


def _ladder(rank_id: str) -> dict:
    return {
        "role": "aggregate",
        "rank_id": rank_id,
        "env": ["reserved", "sim", "region-1", "zone-1", "H100"],
        "config": {"instance_type": "sim.h100.8gpu", "gpu_count": 1},
        "n_replicas": 1,
    }


if __name__ == "__main__":
    unittest.main()
