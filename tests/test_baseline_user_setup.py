"""Test baseline-user setup helpers and opt-in Store isolation."""

from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from tandemn_system_data.clients import PostgresClient
from tandemn_system_data.db.orm import (
    EventRow,
    EvidenceRowRow,
    GpuMetricRow,
    JobRow,
    KoiCausalEdgeRow,
    KoiCausalMechanismRow,
    KoiCausalNodeRow,
    PlanRow,
    RankRow,
    ResourceMapRow,
    UserRow,
)
from tandemn_system_data.ids import (
    new_event_id,
    new_gpu_metric_id,
    new_job_id,
    new_plan_id,
    new_rank_id,
    new_user_id,
)

from benchmarks.baseline_user_setup import (
    baseline_name,
    parse_args,
    reset_user,
    resource_map_body,
)
from benchmarks.seed_benchmark_store import load_resource_map

ROOT = Path(__file__).resolve().parents[1]


class BaselineUserSetupTest(unittest.TestCase):
    """Keep baseline identity and resource-map cloning deterministic."""

    def test_baseline_name_is_stable(self) -> None:
        self.assertEqual(
            baseline_name("bestfit", "20260819T010000Z"),
            "benchmark-baseline:bestfit:20260819T010000Z",
        )

    def test_baseline_name_rejects_unsafe_run_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "run-id"):
            baseline_name("bestfit", "run/id")

    def test_resource_map_copy_excludes_the_source_user(self) -> None:
        _, resource_map = load_resource_map(
            ROOT / "benchmarks/resource_map_benchmarking.jsonl"
        )
        body = resource_map_body(resource_map)

        self.assertEqual(set(body), {"market", "clouds"})
        self.assertEqual(
            sum(pool.total_gpus for *_, pool in resource_map.iter_machine_pools()),
            200,
        )

    def test_koi_root_is_optional_only_at_parse_time(self) -> None:
        args = parse_args(["--baseline", "koi", "--run-id", "run-1"])

        self.assertIsNone(args.koi_root)
        self.assertEqual(args.benchmark_id, "simulation_3h")

    def test_selects_the_run_resource_map(self) -> None:
        args = parse_args(
            [
                "--baseline",
                "bestfit",
                "--run-id",
                "run-1",
                "--benchmark-id",
                "15min_real",
            ]
        )

        self.assertEqual(args.benchmark_id, "15min_real")

    @unittest.skipUnless(
        os.getenv("BASELINE_USER_SETUP_SMOKE") == "1",
        "set BASELINE_USER_SETUP_SMOKE=1 after static Store seeding",
    )
    def test_reset_removes_only_the_target_users_runtime_state(self) -> None:
        _, resource_map = load_resource_map(
            ROOT / "benchmarks/resource_map_benchmarking.jsonl"
        )
        client = PostgresClient()
        target_name = f"baseline-reset-smoke:{new_user_id()}"
        target_id = reset_user(client, name=target_name, resource_map=resource_map)
        other_id = new_user_id()
        job_id = new_job_id()
        rank_id = new_rank_id()
        now = datetime.now(UTC)

        try:
            with client.begin() as session:
                session.add(
                    UserRow(
                        user_id=other_id, name="baseline-smoke-other", created_at=now
                    )
                )
                session.add_all(
                    [
                        JobRow(
                            job_id=job_id,
                            user_id=target_id,
                            kind="inference",
                            spec_json={},
                            input_source={},
                            output_target={},
                            status="running",
                            finish_reason=None,
                            error_message=None,
                            created_at=now,
                            finished_at=None,
                        ),
                        RankRow(
                            rank_id=rank_id,
                            job_id=job_id,
                            plan_id=None,
                            role="aggregate",
                            shape_json={},
                            n_replicas=1,
                            status="running",
                            reason_code=None,
                            created_at=now,
                            updated_at=now,
                        ),
                        PlanRow(
                            plan_id=new_plan_id(),
                            user_id=target_id,
                            koi_version=None,
                            tick_rationale="smoke",
                            actions_json=[],
                            status="pending",
                            created_at=now,
                        ),
                        GpuMetricRow(
                            metric_id=new_gpu_metric_id(),
                            ts=now,
                            job_id=job_id,
                            gpu_uuid="smoke-gpu",
                            rank_id=rank_id,
                            chain_index=0,
                            local_rank="0",
                            role="aggregate",
                            node_name=None,
                            instance_type=None,
                            model_name=None,
                            metrics_json={},
                            created_at=now,
                        ),
                        EventRow(
                            event_id=new_event_id(),
                            user_id=target_id,
                            job_id=job_id,
                            rank_id=rank_id,
                            type="smoke",
                            payload_json={},
                            created_at=now,
                        ),
                        EvidenceRowRow(
                            row_id="evidence-smoke",
                            user_id=target_id,
                            tick=1,
                            job_id=job_id,
                            rank_id=rank_id,
                            deploy_timestamp_utc=0.0,
                            payload_json={},
                            created_at=now,
                        ),
                        KoiCausalNodeRow(
                            user_id=target_id,
                            node_id="smoke-node",
                            node_type="X",
                            description=None,
                            unit=None,
                        ),
                        KoiCausalEdgeRow(
                            user_id=target_id,
                            edge_id="smoke-edge",
                            src="smoke-node",
                            dst="smoke-node",
                            src_type="X",
                            dst_type="X",
                            status="active",
                            alpha=1.0,
                            beta=1.0,
                            visit_count=0,
                            last_touched_tick=None,
                            q_histogram_json={},
                            envs_seen_json=[],
                            q3_frequency=0.0,
                        ),
                        KoiCausalMechanismRow(
                            user_id=target_id,
                            mechanism_id="smoke-mechanism",
                            name="smoke",
                            edge_ids_json=[],
                            scope_json={},
                            narrative="smoke",
                            status="active",
                            archived_reason=None,
                            alpha=1.0,
                            beta=1.0,
                            visit_count=0,
                            last_touched_tick=None,
                            q_histogram_json={},
                            envs_seen_json=[],
                            inspection_count=0,
                        ),
                    ]
                )

            self.assertEqual(
                reset_user(client, name=target_name, resource_map=resource_map),
                target_id,
            )
            with client.session() as session:
                counts = [
                    session.scalar(
                        select(func.count())
                        .select_from(JobRow)
                        .where(JobRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(RankRow)
                        .join(JobRow, RankRow.job_id == JobRow.job_id)
                        .where(JobRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(PlanRow)
                        .where(PlanRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(GpuMetricRow)
                        .where(GpuMetricRow.job_id == job_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(EventRow)
                        .where(EventRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(EvidenceRowRow)
                        .where(EvidenceRowRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(KoiCausalNodeRow)
                        .where(KoiCausalNodeRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(KoiCausalEdgeRow)
                        .where(KoiCausalEdgeRow.user_id == target_id)
                    ),
                    session.scalar(
                        select(func.count())
                        .select_from(KoiCausalMechanismRow)
                        .where(KoiCausalMechanismRow.user_id == target_id)
                    ),
                ]
                self.assertEqual(counts, [0] * len(counts))
                self.assertIsNotNone(session.get(ResourceMapRow, target_id))
                self.assertIsNotNone(session.get(UserRow, other_id))
        finally:
            with client.begin() as session:
                session.execute(
                    delete(GpuMetricRow).where(GpuMetricRow.job_id == job_id)
                )
                session.execute(delete(EventRow).where(EventRow.user_id == target_id))
                session.execute(
                    delete(UserRow).where(UserRow.user_id.in_([target_id, other_id]))
                )


if __name__ == "__main__":
    unittest.main()
