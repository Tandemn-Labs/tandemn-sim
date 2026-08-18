"""Test experiment trace paths."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from trace import (
    append_event,
    cluster_window,
    gpu_sample,
    job_admitted,
    job_finished,
    job_state_changed,
    job_submitted,
    job_window,
    placement_observed,
    plan_validation,
    planner_finished,
    pool_window,
    rank_failed,
    rank_window,
    read_events,
    resource_snapshot,
    run_finished,
    run_started,
    tick_started,
    trace_path,
)


class TracePathTest(unittest.TestCase):
    """Verify the stable trace layout."""

    def test_uses_one_file_per_run(self) -> None:
        """Separate scenario, benchmark, method, and run in the path."""
        path = trace_path(Path("traces"), "simulation", "mixed", "koi", "run-1")

        self.assertEqual(path, Path("traces/simulation/mixed/koi/run-1.jsonl"))

    def test_appends_one_json_event(self) -> None:
        """Create a nested trace file and preserve the supplied event."""
        with tempfile.TemporaryDirectory() as directory:
            path = trace_path(Path(directory), "simulation", "mixed", "koi", "run-1")

            append_event(path, {"run_id": "run-1", "kind": "run_started"})

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"kind": "run_started", "run_id": "run-1"}\n',
            )

    def test_reads_events_in_append_order(self) -> None:
        """Preserve trace event order when compiling a completed run."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            append_event(path, {"kind": "run_started"})
            append_event(path, {"kind": "run_finished"})

            events = read_events(path)

        self.assertEqual([event["kind"] for event in events], ["run_started", "run_finished"])

    def test_run_finished_records_terminal_time(self) -> None:
        """Preserve the experiment completion timestamp without aggregation."""
        event = run_finished("2026-01-01T03:00:00+00:00", status="completed")

        self.assertEqual(event["kind"], "run_finished")
        self.assertEqual(event["finished_at"], "2026-01-01T03:00:00+00:00")
        self.assertFalse(event["truncated"])
        self.assertIsNone(event["termination_reason"])
        self.assertEqual(event["status"], "completed")

    def test_run_started_keeps_trace_identity(self) -> None:
        """Record the immutable context needed to interpret every later event."""
        event = run_started(
            "run-1",
            "simulation",
            "mixed",
            "koi",
            "2026-01-01T00:00:00+00:00",
            300,
        )

        self.assertEqual(event["trace_version"], 4)
        self.assertEqual(event["admission_definition"], "first_place_plan_created")
        self.assertEqual(event["method"], "koi")

    def test_tick_started_keeps_logical_start_time(self) -> None:
        """Record the boundary before a planner receives one control tick."""
        event = tick_started(4, "2026-01-01T00:20:00+00:00")

        self.assertEqual(event["tick"], 4)
        self.assertEqual(event["started_at"], "2026-01-01T00:20:00+00:00")

    def test_resource_snapshot_preserves_pool_facts(self) -> None:
        """Keep raw ResourceMap capacity and price data for later analysis."""
        pools = [{"pool_id": "pool-a", "gpu_count": 8, "price_per_instance_hour": 10.0}]
        event = resource_snapshot(4, pools)

        self.assertEqual(event["tick"], 4)
        self.assertEqual(event["pools"], pools)

    def test_placement_observed_keeps_canonical_rank_facts(self) -> None:
        """Distinguish the applied rank from the requested planner payload."""
        shape = {"instance_type": "sim.h100.8gpu", "count": 1}
        event = placement_observed(4, "plan-4", "job-a", "rank-a", 2, shape)

        self.assertEqual(event["rank_id"], "rank-a")
        self.assertEqual(event["n_replicas"], 2)
        self.assertEqual(event["shape"], shape)

    def test_planner_finished_preserves_outcome_facts(self) -> None:
        """Record planner timing and the plans created for one tick."""
        event = planner_finished(
            4,
            "2026-01-01T00:20:00+00:00",
            "2026-01-01T00:20:14+00:00",
            14_000,
            [
                {
                    "plan_id": "plan-4",
                    "actions": [
                        {"job_id": "job-a", "type": "place"},
                        {"job_id": "job-b", "type": "swap"},
                        {"job_id": "job-c", "type": "defer"},
                        {"job_id": "job-d", "type": "keep"},
                    ],
                }
            ],
            ("job-a", "job-b"),
            False,
            None,
        )

        self.assertEqual(event["decision_time_ms"], 14_000)
        self.assertEqual(event["plans"][0]["plan_id"], "plan-4")
        self.assertFalse(event["timed_out"])
        self.assertEqual(event["suggested_action_count"], 2)

    def test_plan_validation_preserves_post_apply_proof(self) -> None:
        event = plan_validation(4, "plan-4", 2, True)

        self.assertEqual(event["kind"], "plan_validation")
        self.assertEqual(event["applied_action_count"], 2)
        self.assertTrue(event["selected_plan_valid"])

    def test_job_admitted_uses_first_place_time(self) -> None:
        """Preserve the scheduler admission timestamps for JQT derivation."""
        event = job_admitted(
            "job-a",
            "plan-4",
            "2026-01-01T00:06:00+00:00",
            "2026-01-01T00:10:00+00:00",
            "logical_place_plan_created",
        )

        self.assertEqual(event["action_type"], "place")
        self.assertEqual(event["plan_id"], "plan-4")
        self.assertEqual(event["admission_source"], "logical_place_plan_created")

    def test_rank_window_keeps_actual_and_predicted_trajectories(self) -> None:
        """Preserve rank facts without inventing predictions for baselines."""
        event = rank_window(
            4,
            "job-a",
            "rank-0",
            (0,),
            "2026-01-01T00:20:00+00:00",
            "2026-01-01T00:25:00+00:00",
            {"kv_cache_util": [0.72, 0.74]},
            {"p99_ttft_ms": [840, 860]},
            sample_bucket_starts=(
                "2026-01-01T00:20:00+00:00",
                "2026-01-01T00:20:10+00:00",
            ),
        )

        self.assertIsNone(event["v_hat"])
        self.assertEqual(event["y_actual"], {"p99_ttft_ms": [840, 860]})
        self.assertEqual(event["chain_indices"], [0])
        self.assertEqual(event["telemetry_scope"], "observed_tick")
        self.assertIsNone(event["projection_method"])
        self.assertEqual(
            event["sample_bucket_starts"],
            ["2026-01-01T00:20:00+00:00", "2026-01-01T00:20:10+00:00"],
        )
        self.assertEqual(event["aggregation_policy_version"], 2)

    def test_job_window_keeps_raw_values_and_slo_flags(self) -> None:
        """Store per-window facts without deriving an experiment-wide rate."""
        event = job_window(
            4,
            "job-a",
            "online",
            "2026-01-01T00:20:00+00:00",
            "2026-01-01T00:25:00+00:00",
            840,
            42,
            1550,
            0.000012,
            1000,
            50,
            True,
            True,
            100,
            80,
            20,
            0.8,
            "completed_requests",
            offered_request_count=100,
        )

        self.assertEqual(event["p99_ttft_ms"], 840)
        self.assertTrue(event["ttft_slo_met"])
        self.assertEqual(event["cost_per_token"], 0.000012)
        self.assertEqual(event["attempted_request_count"], 100)
        self.assertEqual(event["completed_request_count"], 80)
        self.assertEqual(event["unfinished_request_count"], 20)
        self.assertEqual(event["completion_ratio"], 0.8)
        self.assertEqual(event["latency_scope"], "completed_requests")
        self.assertIsNone(event["slo_pass_count"])
        self.assertEqual(event["slo_count_scope"], "completed_requests")
        self.assertIsNone(event["ttft_pass_count"])
        self.assertIsNone(event["tpot_pass_count"])
        self.assertEqual(event["offered_request_count"], 100)
        self.assertIsNone(event["admitted_request_count"])
        self.assertIsNone(event["qualified_output_tokens"])
        self.assertEqual(event["request_count_scope"], "tick_requests")

    def test_batch_window_separates_probe_and_campaign_counts(self) -> None:
        event = job_window(
            4,
            "batch-a",
            "batch",
            "2026-01-01T00:20:00+00:00",
            "2026-01-01T00:25:00+00:00",
            840,
            42,
            1550,
            0.000012,
            1000,
            50,
            None,
            None,
            request_count_scope="batch_probe",
            batch_latency_diagnostic=True,
            probe_attempted_request_count=100,
            probe_completed_request_count=80,
            probe_unfinished_request_count=20,
            campaign_total_request_count=10_000,
            campaign_completed_in_window=3_000,
            campaign_completed_request_count=5_000,
            campaign_remaining_request_count=5_000,
            campaign_request_per_sec=10,
        )

        self.assertEqual(event["request_count_scope"], "batch_probe")
        self.assertTrue(event["batch_latency_diagnostic"])
        self.assertEqual(event["probe_completed_request_count"], 80)
        self.assertEqual(event["campaign_completed_request_count"], 5_000)

    def test_pool_window_exposes_zero_filled_capacity(self) -> None:
        """Keep missing telemetry distinct from measured zero utilization."""
        event = pool_window(4, "pool-a", 8, 6, 3, 42.5, 75)

        self.assertEqual(event["missing_gpu_count"], 2)
        self.assertEqual(event["zero_filled_gpu_count"], 2)
        self.assertEqual(event["available_gpu_count"], 5)

    def test_cluster_window_keeps_whole_and_allocated_costs(self) -> None:
        """Separate whole-cluster cost from active GPU capacity cost."""
        event = cluster_window(
            4,
            "2026-01-01T00:20:00+00:00",
            "2026-01-01T00:25:00+00:00",
            5.3333,
            2.3333,
            32.773,
            28.677,
            2,
            8,
            11900,
        )

        self.assertEqual(event["capacity_gpu_hours"], 5.3333)
        self.assertEqual(event["capacity_gpu_seconds"], 5.3333 * 3600)
        self.assertEqual(event["allocated_gpu_seconds"], 2.3333 * 3600)
        self.assertEqual(event["busy_gpu_seconds"], 2.3333 * 3600)
        self.assertEqual(event["allocated_cost"], 28.677)
        self.assertEqual(event["jobs_admitted_in_tick"], 2)
        self.assertEqual(event["admission_count_so_far"], 8)
        self.assertEqual(event["rolling_admission_time_ms"], 11900)

    def test_job_submitted_preserves_slo_and_deadline_inputs(self) -> None:
        """Freeze the workload facts required for later JCT and SLO analysis."""
        event = job_submitted(
            "batch-a",
            "batch",
            "2026-01-01T00:06:00+00:00",
            "tenant-a",
            None,
            None,
            "2026-01-01T02:06:00+00:00",
        )

        self.assertEqual(event["job_kind"], "batch")
        self.assertEqual(event["tenant_id"], "tenant-a")
        self.assertEqual(event["deadline_at"], "2026-01-01T02:06:00+00:00")

    def test_job_finished_marks_terminal_batch_outcome(self) -> None:
        """Preserve completion and deadline facts without calculating JCT."""
        event = job_finished(
            "batch-a",
            "2026-01-01T01:55:00+00:00",
            True,
            None,
            True,
            0,
            deadline_outcome="met",
            offered_request_count=100,
            completed_request_count=92,
            unfinished_request_count=8,
            ttft_pass_count=88,
            tpot_pass_count=85,
            slo_pass_count=80,
            qualified_output_tokens=120_000,
            slo_count_scope="all_offered_requests",
        )

        self.assertTrue(event["completed_successfully"])
        self.assertTrue(event["deadline_slo_met"])
        self.assertIsNone(event["finish_reason"])
        self.assertEqual(event["unserved_request_count"], 0)
        self.assertEqual(event["deadline_outcome"], "met")
        self.assertEqual(event["offered_request_count"], 100)
        self.assertEqual(event["completed_request_count"], 92)
        self.assertEqual(event["unfinished_request_count"], 8)
        self.assertEqual(event["ttft_pass_count"], 88)
        self.assertEqual(event["tpot_pass_count"], 85)
        self.assertEqual(event["slo_pass_count"], 80)
        self.assertEqual(event["qualified_output_tokens"], 120_000)
        self.assertEqual(event["slo_count_scope"], "all_offered_requests")

    def test_job_state_changed_preserves_queue_transition(self) -> None:
        event = job_state_changed(
            "job-a", "WAITING", "RUNNING", "2026-01-01T00:40:00+00:00", "ranks_materialized")

        self.assertEqual(event["from_status"], "WAITING")
        self.assertEqual(event["to_status"], "RUNNING")

    def test_rank_failed_records_oom_degradation_and_released_capacity(self) -> None:
        """Expose one rank-local failure without declaring a job SLO outcome."""
        event = rank_failed(
            "job-a",
            "rank-a",
            "2026-01-01T00:40:00+00:00",
            "OOM",
            8,
            "Model size exceeds total NPU memory",
        )

        self.assertEqual(event["reason"], "OOM")
        self.assertEqual(event["service_impact"], "degraded")
        self.assertNotIn("slo_outcome", event)
        self.assertEqual(event["released_gpu_count"], 8)
        self.assertIn("memory", event["error_detail"])

    def test_gpu_sample_keeps_idle_store_telemetry(self) -> None:
        """Keep actual idle GPU rows distinct from pool-level zero filling."""
        event = gpu_sample(
            4,
            "2026-01-01T00:20:10+00:00",
            "GPU-1",
            None,
            None,
            None,
            "node-a",
            "p5.48xlarge",
            {"sm_utilization": 0.0, "gpu_mem_used_fraction": 0.0},
        )

        self.assertIsNone(event["job_id"])
        self.assertIn("chain_index", event)
        self.assertNotIn("chain_id", event)
        self.assertEqual(event["metrics"]["sm_utilization"], 0.0)


if __name__ == "__main__":
    unittest.main()
