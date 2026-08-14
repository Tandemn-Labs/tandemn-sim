"""Test per-rank LLMServingSim input generation."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
    completed_request_summary,
    gpu_metrics_from_results,
    parse_rank_result,
    prepare_rank_run,
    resolve_rank_hardware,
    _read_metric_samples,
    _run_rank,
    _run_summary,
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
        self.ranks = (
            SimpleNamespace(rank_id="rank_a", n_replicas=2, shape_json=dict(shape)),
        )
        self.hardware = {
            "profile": "H100",
            "memory_gb": 80,
            "memory_bandwidth_gb_s": 3350,
            "link_bandwidth_gb_s": 900,
        }

    def test_rank_chains_build_one_homogeneous_cluster(self) -> None:
        """Create two deterministic rank instances with TP=2 each."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        cluster = build_rank_cluster(deployment, self.hardware)

        self.assertEqual([binding.chain_index for binding in deployment.chains], [0, 1])
        self.assertEqual(cluster["num_nodes"], 1)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 2)
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 2)
        self.assertEqual(cluster["nodes"][0]["instances"][0]["tp_size"], 2)
        self.assertEqual(cluster["nodes"][0]["instances"][0]["fp"], "bf16")

    def test_rank_cluster_uses_store_fp16_name(self) -> None:
        """Use LLMServingSim's FP16 profile convention for Store float16 models."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        deployment.shape["weight_dtype"] = "float16"

        cluster = build_rank_cluster(deployment, self.hardware)

        self.assertEqual(cluster["nodes"][0]["instances"][0]["fp"], "fp16")
        self.assertEqual(cluster["nodes"][0]["instances"][0]["dtype"], "float16")

    def test_rank_deployment_uses_per_replica_gpu_count(self) -> None:
        """Keep total DP footprint separate from one simulated replica's GPU count."""
        shape = {**self.ranks[0].shape_json, "count": 8, "tp": 2, "pp": 1, "dp": 4}
        rank = SimpleNamespace(rank_id="rank_a", n_replicas=4, shape_json=shape)

        deployment = build_rank_deployments(self.job, (rank,))[0]

        self.assertEqual(deployment.shape["count"], 2)
        self.assertEqual(len(deployment.chains), 4)
        self.assertTrue(all(chain.gpu_count == 2 for chain in deployment.chains))

    def test_store_hardware_catalog_resolves_rank_profile(self) -> None:
        """Resolve one rank's Store catalog entry into H100 simulator facts."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
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

    def test_store_hardware_catalog_reuses_same_cloud_instance_across_regions(self) -> None:
        """Use identical cloud instance facts when a catalog omits one region."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        deployment = replace(
            deployment,
            shape={
                **deployment.shape,
                "env": ["reserved", "sim", "region-2", "zone-1", "H100-SXM-80GB"],
            },
        )
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

        self.assertEqual(resolve_rank_hardware(deployment, catalog)["profile"], "H100")

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
        deployment = build_rank_deployments(self.job, self.ranks)[0]
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
            self.assertEqual(
                rank_run.command[
                    rank_run.command.index("--max-wall-time-seconds") + 1
                ],
                "300",
            )

    def test_rank_run_receives_shared_deadline(self) -> None:
        """Append one tick deadline to an active simulator command."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        rank_run = RankRun(deployment, (), ("python3", "-m", "serving"), Path(), Path(), 0, 330)
        parsed = SimpleNamespace()
        completed = subprocess.CompletedProcess((), 0, "", "")

        with (
            patch("serving.time", return_value=1_000),
            patch("serving.subprocess.run", return_value=completed) as run,
            patch("serving.parse_rank_result", return_value=parsed),
        ):
            result = _run_rank(rank_run, 1_300)

        self.assertIs(result, parsed)
        self.assertEqual(
            run.call_args.args[0][-2:],
            ("--wall-deadline-epoch-seconds", "1300"),
        )

    def test_rank_queued_past_shared_deadline_is_not_started(self) -> None:
        """Return an unfinished result without launching a late simulator."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        with tempfile.TemporaryDirectory() as directory:
            cleanup_dir = Path(directory) / "rank"
            cleanup_dir.mkdir()
            rank_run = RankRun(deployment, (), (), Path(), Path(), 0, 330, cleanup_dir)
            with (
                patch("serving.time", return_value=1_301),
                patch("serving.subprocess.run") as run,
            ):
                result = _run_rank(rank_run, 1_300)

            self.assertFalse(cleanup_dir.exists())

        run.assert_not_called()
        self.assertEqual(result.status, "wall_time_limit")
        self.assertEqual(result.completed_requests, ())

    def test_parse_rank_result_reads_csv_and_final_metrics(self) -> None:
        """Read one completed row per request and one final metric line per replica."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "requests.csv"
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
                writer.writeheader()
                for request_id, instance_id in ((1, 0), (0, 1)):
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
                        json.dumps(
                            {
                                "sim_time_ns": 11_000_000_000,
                                "sample_kind": "final",
                                "instance_id": 1,
                                "kv_cache_util": 0.0,
                            }
                        ),
                    )
                )
                + "\n"
            )
            requests = (
                Request("request_0", "session", 0, 0, 100, 20, None),
                Request("request_1", "session", 0, 0, 100, 20, None),
            )
            rank_run = RankRun(deployment, requests, (), output_path, metrics_path, 2, 30)
            completed = subprocess.CompletedProcess(
                (),
                0,
                "\n".join(
                    (
                        "SIM_METRICS instance=0 kv_cache_util=0.5",
                        "SIM_METRICS instance=1 kv_cache_util=0.6",
                        "All Request Has Been Exited",
                        'SIM_RUN_SUMMARY {"status":"completed","loaded_requests":2,"completed_requests":2,"unfinished_requests":0,"sim_time_ns":11000}',
                    )
                ),
                "",
            )
            result = parse_rank_result(rank_run, completed)

        self.assertEqual(len(result.request_rows), 2)
        self.assertEqual(len(result.metric_samples), 4)
        self.assertEqual(result.metric_samples[0].metrics["kv_cache_util"], 0.5)
        self.assertEqual(result.metric_samples[1].metrics["kv_cache_util"], 0.6)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.completed_requests, (requests[1], requests[0]))
        self.assertEqual(result.sim_time_ns, 11_000)

    def test_run_summary_rejects_missing_malformed_and_unknown_status(self) -> None:
        """Only LLMServingSim's two explicit terminal states are accepted."""
        for output in (
            "",
            "SIM_RUN_SUMMARY {",
            'SIM_RUN_SUMMARY {"status":"crashed","loaded_requests":1,"completed_requests":0,"unfinished_requests":1,"sim_time_ns":0}',
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                _run_summary(output)

    def test_parse_rank_result_rejects_invalid_request_ids(self) -> None:
        """Completed rows must name a unique, rank-local request index."""
        rank = SimpleNamespace(rank_id="rank_ids", n_replicas=1, shape_json=self.ranks[0].shape_json)
        deployment = build_rank_deployments(self.job, (rank,))[0]
        requests = (
            Request("request_0", "session", 0, 0, 100, 20, None),
            Request("request_1", "session", 0, 0, 100, 20, None),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.jsonl"
            metrics_path.write_text(json.dumps({
                "sim_time_ns": 1, "sample_interval_ns": 1,
                "sample_kind": "periodic", "instance_id": 0,
            }) + "\n")
            for ids in ((0.5, 1), (-1, 1), (2, 1), (0, 0)):
                output_path = root / "requests.csv"
                with output_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
                    writer.writeheader()
                    for request_id in ids:
                        writer.writerow(dict.fromkeys(CSV_HEADER, request_id))
                rank_run = RankRun(deployment, requests, (), output_path, metrics_path, 2, 30)
                completed = subprocess.CompletedProcess(
                    (), 0,
                    'SIM_RUN_SUMMARY {"status":"completed","loaded_requests":2,"completed_requests":2,"unfinished_requests":0,"sim_time_ns":1000}',
                    "",
                )
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    parse_rank_result(rank_run, completed)

    def test_metric_samples_reject_fractional_instance_ids(self) -> None:
        """Metric identities must not silently truncate fractional instance IDs."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps({
                "sim_time_ns": 1, "sample_interval_ns": 1,
                "sample_kind": "periodic", "instance_id": 0.5,
            }) + "\n")
            with self.assertRaises(ValueError):
                _read_metric_samples(path, {0})

    def test_metric_samples_keep_final_only_telemetry(self) -> None:
        """Short completed runs still produce telemetry when only final rows exist."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("\n".join(json.dumps({
                "sim_time_ns": 5, "sample_kind": "final", "instance_id": instance_id,
                "kv_cache_util": 0.5 + instance_id / 10,
            }) for instance_id in (0, 1)) + "\n")
            samples = _read_metric_samples(path, {0, 1})

        self.assertEqual(len(samples), 2)
        self.assertTrue(all(sample.sample_interval_ns == 5 for sample in samples))

    def test_rank_local_request_ids_do_not_cross_rank_runs(self) -> None:
        """Every rank's request ID zero resolves against that rank's input order."""
        rank = SimpleNamespace(rank_id="rank_local", n_replicas=1, shape_json=self.ranks[0].shape_json)
        deployment = build_rank_deployments(self.job, (rank,))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = subprocess.CompletedProcess(
                (), 0,
                'SIM_RUN_SUMMARY {"status":"completed","loaded_requests":1,"completed_requests":1,"unfinished_requests":0,"sim_time_ns":1000}',
                "",
            )
            results = []
            for name in ("rank_a_request", "rank_b_request"):
                output_path = root / f"{name}.csv"
                output_path.write_text(",".join(CSV_HEADER) + "\n0,0,model,1,1,0,1,1,0,1,1,[],0,0,0\n")
                metrics_path = root / f"{name}.jsonl"
                metrics_path.write_text(json.dumps({
                    "sim_time_ns": 1, "sample_interval_ns": 1,
                    "sample_kind": "periodic", "instance_id": 0,
                }) + "\n")
                request = Request(name, "session", 0, 0, 1, 1, None)
                results.append(parse_rank_result(
                    RankRun(deployment, (request,), (), output_path, metrics_path, 1, 30), completed))

        self.assertEqual([result.completed_requests[0].request_name for result in results], [
            "rank_a_request", "rank_b_request",
        ])

    def test_parse_rank_result_rejects_missing_replica_metrics(self) -> None:
        """Require final runtime metrics from every chain replica."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
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
            requests = (
                Request("request_0", "session", 0, 0, 100, 20, None),
                Request("request_1", "session", 0, 0, 100, 20, None),
            )
            rank_run = RankRun(deployment, requests, (), output_path, metrics_path, 2, 30)
            completed = subprocess.CompletedProcess(
                (),
                0,
                'SIM_RUN_SUMMARY {"status":"completed","loaded_requests":2,"completed_requests":2,"unfinished_requests":0,"sim_time_ns":1000}',
                "",
            )
            with self.assertRaises(ValueError):
                parse_rank_result(rank_run, completed)

    def test_parse_rank_result_accepts_completed_rows_after_wall_time_limit(self) -> None:
        """Accept the completed-only artifacts emitted by a timed-out ServingSim run."""
        rank = SimpleNamespace(rank_id="rank_partial", n_replicas=1, shape_json=self.ranks[0].shape_json)
        deployment = build_rank_deployments(self.job, (rank,))[0]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "requests.csv"
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
                writer.writeheader()
                writer.writerow(dict.fromkeys(CSV_HEADER, 0))
            metrics_path = Path(directory) / "metrics.jsonl"
            metrics_path.write_text(json.dumps({
                "sim_time_ns": 1, "sample_interval_ns": 1,
                "sample_kind": "periodic", "instance_id": 0,
            }) + "\n")
            requests = (
                Request("request_0", "session", 0, 0, 100, 20, None),
                Request("request_1", "session", 0, 0, 100, 20, None),
            )
            rank_run = RankRun(deployment, requests, (), output_path, metrics_path, 2, 30)
            completed = subprocess.CompletedProcess(
                (), 0,
                'SIM_RUN_SUMMARY {"status":"wall_time_limit","loaded_requests":2,"completed_requests":1,"unfinished_requests":1,"sim_time_ns":1000,"admitted_request_ids":[1]}',
                "",
            )
            result = parse_rank_result(rank_run, completed)

        self.assertEqual(len(result.request_rows), 1)
        self.assertEqual(len(result.metric_samples), 1)
        self.assertEqual(result.status, "wall_time_limit")
        self.assertEqual(result.completed_requests, (requests[0],))
        self.assertEqual(result.admitted_requests, (requests[1],))
        self.assertEqual(result.sim_time_ns, 1_000)

    def test_parse_rank_result_accepts_zero_completed_requests_after_wall_time_limit(self) -> None:
        """A cutoff before the first completion is still a valid simulator result."""
        rank = SimpleNamespace(rank_id="rank_empty", n_replicas=1, shape_json=self.ranks[0].shape_json)
        deployment = build_rank_deployments(self.job, (rank,))[0]
        requests = (Request("request_0", "session", 0, 0, 100, 20, None),)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "requests.csv"
            output_path.write_text(",".join(CSV_HEADER) + "\n")
            metrics_path = Path(directory) / "metrics.jsonl"
            metrics_path.write_text(json.dumps({
                "sim_time_ns": 1, "sample_interval_ns": 1,
                "sample_kind": "periodic", "instance_id": 0,
            }) + "\n")
            rank_run = RankRun(deployment, requests, (), output_path, metrics_path, 1, 30)
            completed = subprocess.CompletedProcess(
                (), 0,
                'SIM_RUN_SUMMARY {"status":"wall_time_limit","loaded_requests":1,"completed_requests":0,"unfinished_requests":1,"sim_time_ns":0}',
                "",
            )
            result = parse_rank_result(rank_run, completed)

        self.assertEqual(result.completed_requests, ())

    def test_rank_results_expand_to_one_metric_per_simulated_gpu(self) -> None:
        """Convert two TP=2 replicas into four canonical Store GPU metrics."""
        deployment = build_rank_deployments(self.job, self.ranks)[0]
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
        replica_zero = [metric for metric in periodic if metric.chain_index == 0]
        self.assertEqual([metric.local_rank for metric in replica_zero], ["0", "1"])
        self.assertEqual(replica_zero[0].rank_id, "rank_a")
        self.assertEqual(replica_zero[0].throughput_token_per_sec, 10.0)
        self.assertEqual(replica_zero[0].p99_ttft_ms, 100.0)
        self.assertEqual(replica_zero[0].kv_cache_util, 0.5)

        tick_summary = [metric for metric in metrics if metric.kvcache_hit_rate is not None]
        self.assertEqual(len(tick_summary), 4)
        self.assertTrue(all(metric.kvcache_hit_rate == 125 / 300 for metric in tick_summary))
        self.assertTrue(all(metric.kv_cache_util is None for metric in tick_summary))

    def test_completed_request_summary_merges_rows_across_ranks(self) -> None:
        """Calculate one percentile from all completed job requests."""
        def result(ttft_values):
            return RankResult(
                SimpleNamespace(),
                tuple(
                    {"TTFT": value * 1_000_000, "TPOT": value * 100_000, "output": 200}
                    for value in ttft_values
                ),
                (),
                "",
                "",
            )

        summary = completed_request_summary((result((1, 100)), result((2, 3))), 10, 1)

        self.assertEqual(summary["completed_request_count"], 4)
        self.assertAlmostEqual(summary["p99_ttft_ms"], 97.09)
        self.assertAlmostEqual(summary["p99_tpot_ms"], 9.709)
        self.assertEqual(summary["ttft_pass_count"], 3)
        self.assertEqual(summary["ttft_fail_count"], 1)
        self.assertEqual(summary["tpot_pass_count"], 3)
        self.assertEqual(summary["tpot_fail_count"], 1)
        self.assertEqual(summary["slo_pass_count"], 3)
        self.assertEqual(summary["slo_fail_count"], 1)
        self.assertEqual(summary["qualified_output_tokens"], 600)

    def test_encoder_summary_uses_latency_as_ttft(self) -> None:
        result = RankResult(
            SimpleNamespace(),
            ({"latency": 50_000_000, "TTFT": -1, "TPOT": 0, "output": 0},),
            (), "", "",
        )

        summary = completed_request_summary((result,), 60, 1, "embeddings")

        self.assertEqual(summary["p99_ttft_ms"], 50)
        self.assertIsNone(summary["p99_tpot_ms"])
        self.assertEqual(summary["ttft_pass_count"], 1)
        self.assertIsNone(summary["tpot_pass_count"])
        self.assertEqual(summary["slo_pass_count"], 1)

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
