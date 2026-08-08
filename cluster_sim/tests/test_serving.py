"""Test per-rank LLMServingSim input generation."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import RankMetricSample, Request
from serving import (
    CSV_HEADER,
    RankRun,
    RankResult,
    build_rank_cluster,
    build_rank_deployments,
    commit_tick_metrics,
    gpu_metrics_from_results,
    parse_rank_result,
    prepare_rank_run,
    resolve_rank_hardware,
    _tick_kvcache_hit_rate,
    write_rank_workload,
)


class ServingInputTest(unittest.TestCase):
    """Verify Store chains become homogeneous rank simulation inputs."""

    def setUp(self) -> None:
        """Create one job and two replicas in one TP=2 rank."""
        self.job = SimpleNamespace(
            job_id="job_1",
            spec_json={"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        )
        shape = {
            "rank_id": "rank_0",
            "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
            "count": 2,
            "tp": 2,
            "pp": 1,
            "ep": 1,
            "instance_type": "sim.h100.8gpu",
            "target_p99_ttft_ms": 600,
            "max_num_seq": 64,
            "max_num_batched_tokens": 4096,
            "prefix_cache_enabled": True,
        }
        self.chains = (
            SimpleNamespace(chain_id="chain_b", shape_json=dict(shape)),
            SimpleNamespace(chain_id="chain_a", shape_json=dict(shape)),
        )
        self.hardware = {
            "profile": "H100",
            "memory_gb": 80,
            "memory_bandwidth_gb_s": 3350,
            "link_bandwidth_gb_s": 900,
        }

    def test_rank_chains_build_one_homogeneous_cluster(self) -> None:
        """Create two deterministic rank instances with TP=2 each."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        cluster = build_rank_cluster(deployment, self.hardware)

        self.assertEqual([binding.chain_id for binding in deployment.chains], ["chain_a", "chain_b"])
        self.assertEqual(cluster["num_nodes"], 1)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 2)
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 2)
        self.assertEqual(cluster["nodes"][0]["instances"][0]["tp_size"], 2)

    def test_store_hardware_catalog_resolves_rank_profile(self) -> None:
        """Resolve one rank's Store catalog entry into H100 simulator facts."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        catalog = {
            "regions": [
                {
                    "cloud": "sim",
                    "region": "region-1",
                    "instance_types": [
                        {
                            "instance_type": "sim.h100.8gpu",
                            "accelerators": [
                                {
                                    "name": "H100-SXM-80GB",
                                    "canonical_gpu_name": "H100-SXM-80GB",
                                    "memory_mib_each": 81_920,
                                    "gpu_bandwidth_gbps": 3_350,
                                    "nvlink_bandwidth_gbps": 900,
                                }
                            ],
                            "network": {"network_cards": [{"peak_bandwidth_gbps": 400}]},
                        }
                    ],
                }
            ]
        }

        hardware = resolve_rank_hardware(deployment, catalog)

        self.assertEqual(
            hardware,
            {
                "profile": "H100",
                "memory_gb": 80,
                "memory_bandwidth_gb_s": 3_350,
                "link_bandwidth_gb_s": 900,
                "link_latency_ns": 0,
            },
        )

    def test_rank_workload_rebases_arrivals_and_preserves_prefixes(self) -> None:
        """Write rank-local LLMServingSim rows from global logical arrivals."""
        requests = (
            Request("r0", "s0", 0, 305 * 1_000_000_000, 100, 20, "job/prefix-75"),
            Request("r1", "s1", 0, 330 * 1_000_000_000, 50, 10, "job/prefix-75"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            write_rank_workload(requests, path, 300 * 1_000_000_000)
            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([row["arrival_time_ns"] for row in rows], [5_000_000_000, 30_000_000_000])
        self.assertEqual(rows[0]["input_tok_ids"][:50], rows[1]["input_tok_ids"])
        self.assertEqual(len(rows[0]["input_tok_ids"]), 75)

    def test_prepare_rank_run_writes_isolated_input_files(self) -> None:
        """Prepare one unique rank directory before its Docker process starts."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "outputs" / "run-rank-0"
            rank_run = prepare_rank_run(
                deployment,
                (Request("r0", "s0", 0, 0, 16, 8, None),),
                self.hardware,
                root,
                run_dir,
                "servingsim_docker",
                "run-rank-0",
                0,
                30,
            )

            self.assertEqual(rank_run.cleanup_dir, run_dir)
            self.assertTrue((run_dir / "cluster.json").is_file())
            self.assertTrue((run_dir / "workload.jsonl").is_file())
            self.assertEqual(rank_run.output_path, run_dir / "requests.csv")
            self.assertEqual(rank_run.metrics_path, run_dir / "metrics.jsonl")

    def test_parse_rank_result_reads_csv_and_final_metrics(self) -> None:
        """Read one completed row per request and one final metric line per replica."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "requests.csv"
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
                writer.writeheader()
                for request_id, instance_id in enumerate((0, 1)):
                    writer.writerow(
                        {
                            "instance id": instance_id,
                            "request id": request_id,
                            "model": deployment.model_name,
                            "input": 100,
                            "output": 20,
                            "arrival": 0,
                            "end_time": 1000,
                            "latency": 1000,
                            "queuing_delay": 0,
                            "TTFT": 100,
                            "TPOT": 10,
                            "ITL": 10,
                            "npu_prefix_hit_tokens": 50,
                            "prefix_cache_requested_tokens": 100,
                            "prefix_cache_hit_ratio": 0.5,
                        }
                    )
            metrics_path = Path(directory) / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "sim_time_ns": 10_000_000_000,
                                "sample_interval_ns": 10_000_000_000,
                                "sample_kind": "periodic",
                                "instance_id": 0,
                                "kv_cache_util": 0.5,
                            }
                        ),
                        json.dumps(
                            {
                                "sim_time_ns": 10_000_000_000,
                                "sample_interval_ns": 10_000_000_000,
                                "sample_kind": "periodic",
                                "instance_id": 1,
                                "kv_cache_util": 0.6,
                            }
                        ),
                        json.dumps(
                            {
                                "sim_time_ns": 11_000_000_000,
                                "sample_kind": "final",
                                "instance_id": 0,
                                "kv_cache_util": 0.0,
                            }
                        ),
                    )
                )
                + "\n"
            )
            rank_run = RankRun(deployment, (), output_path, metrics_path, 2, 30)
            completed = subprocess.CompletedProcess(
                (),
                0,
                "\n".join(
                    (
                        "SIM_METRICS instance=0 kv_cache_util=0.5",
                        "SIM_METRICS instance=1 kv_cache_util=0.6",
                        "All Request Has Been Exited",
                    )
                ),
                "",
            )
            result = parse_rank_result(rank_run, completed)

        self.assertEqual(len(result.request_rows), 2)
        self.assertEqual(len(result.metric_samples), 2)
        self.assertEqual(result.metric_samples[0].metrics["kv_cache_util"], 0.5)
        self.assertEqual(result.metric_samples[1].metrics["kv_cache_util"], 0.6)

    def test_parse_rank_result_rejects_missing_replica_metrics(self) -> None:
        """Require final runtime metrics from every chain replica."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "requests.csv"
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
                writer.writeheader()
                for request_id in range(2):
                    writer.writerow(dict.fromkeys(CSV_HEADER, request_id))
            metrics_path = Path(directory) / "metrics.jsonl"
            metrics_path.write_text(
                json.dumps(
                    {
                        "sim_time_ns": 10_000_000_000,
                        "sample_interval_ns": 10_000_000_000,
                        "sample_kind": "periodic",
                        "instance_id": 0,
                    }
                )
                + "\n"
            )
            rank_run = RankRun(deployment, (), output_path, metrics_path, 2, 30)
            completed = subprocess.CompletedProcess(
                (),
                0,
                "SIM_METRICS instance=0 kv_cache_util=0.5\nAll Request Has Been Exited",
                "",
            )
            with self.assertRaises(ValueError):
                parse_rank_result(rank_run, completed)

    def test_rank_results_expand_to_one_metric_per_simulated_gpu(self) -> None:
        """Convert two TP=2 chains into four canonical Store GPU metrics."""
        deployment = build_rank_deployments(self.job, self.chains)[0]
        result = RankResult(
            deployment=deployment,
            request_rows=(
                {
                    "instance id": 0,
                    "input": 100,
                    "output": 100,
                    "end_time": 2_000_000_000,
                    "TTFT": 100_000_000,
                    "TPOT": 10_000_000,
                    "npu_prefix_hit_tokens": 25,
                    "prefix_cache_requested_tokens": 100,
                    "prefix_cache_hit_ratio": 0.25,
                },
                {
                    "instance id": 1,
                    "input": 200,
                    "output": 200,
                    "end_time": 2_000_000_000,
                    "TTFT": 200_000_000,
                    "TPOT": 20_000_000,
                    "npu_prefix_hit_tokens": 100,
                    "prefix_cache_requested_tokens": 200,
                    "prefix_cache_hit_ratio": 0.5,
                },
            ),
            metric_samples=(
                RankMetricSample(10_000_000_000, 10_000_000_000, 0, {"kv_cache_util": 0.5}),
                RankMetricSample(10_000_000_000, 10_000_000_000, 1, {"kv_cache_util": 0.7}),
            ),
            stdout="",
            stderr="",
        )
        metrics = gpu_metrics_from_results(
            self.job,
            (deployment,),
            (result,),
            datetime(2026, 1, 1, tzinfo=UTC),
            300_000_000_000,
            {"sim.h100.8gpu": 10.0},
        )

        periodic = [metric for metric in metrics if metric.ts == datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)]
        self.assertEqual(len(periodic), 4)
        chain_a = [metric for metric in periodic if metric.chain_id == "chain_a"]
        self.assertEqual([metric.local_rank for metric in chain_a], ["0", "1"])
        self.assertEqual(chain_a[0].rank_id, "rank_0")
        self.assertEqual(chain_a[0].throughput_token_per_sec, 10.0)
        self.assertEqual(chain_a[0].p99_ttft_ms, 100.0)
        self.assertEqual(chain_a[0].kv_cache_util, 0.5)

        tick_summary = [metric for metric in metrics if metric.kvcache_hit_rate is not None]
        self.assertEqual(len(tick_summary), 4)
        self.assertTrue(all(metric.kvcache_hit_rate == 125 / 300 for metric in tick_summary))
        self.assertTrue(all(metric.kv_cache_util is None for metric in tick_summary))

    def test_tick_cache_hit_rate_is_token_weighted(self) -> None:
        rows = (
            {"npu_prefix_hit_tokens": 1, "prefix_cache_requested_tokens": 10},
            {"npu_prefix_hit_tokens": 90, "prefix_cache_requested_tokens": 90},
        )

        self.assertEqual(_tick_kvcache_hit_rate(rows), 91 / 100)
        self.assertIsNone(_tick_kvcache_hit_rate(()))
        with self.assertRaises(ValueError):
            _tick_kvcache_hit_rate(
                ({"npu_prefix_hit_tokens": 11, "prefix_cache_requested_tokens": 10},)
            )

    def test_commit_tick_metrics_uses_one_store_batch(self) -> None:
        """Delegate one completed tick's fully staged metrics to put_many once."""
        calls = []
        store = SimpleNamespace(put_many=lambda metrics: calls.append(tuple(metrics)))

        commit_tick_metrics(store, ("metric_a", "metric_b"))

        self.assertEqual(calls, [("metric_a", "metric_b")])


if __name__ == "__main__":
    unittest.main()
