"""Opt-in real LLMServingSim smoke test for one rank and two requests."""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
TANDEMN_ROOT = ROOT.parent
LLMSERVINGSIM_ROOT = TANDEMN_ROOT / "tandemn-LLMServingSim"
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import Request
from serving import (
    RankRun,
    build_rank_cluster,
    build_rank_command,
    build_rank_deployments,
    run_ranks,
    write_rank_workload,
)


@unittest.skipUnless(
    os.getenv("CLUSTER_SIM_SERVINGSIM_SMOKE") == "1",
    "set CLUSTER_SIM_SERVINGSIM_SMOKE=1 with servingsim_docker running",
)
class LLMServingSimSmokeTest(unittest.TestCase):
    """Exercise one real rank command through CSV and metrics parsing."""

    def test_one_rank_two_requests(self) -> None:
        """Run the shipped RTXPRO6000 Llama profile through the adapter."""
        job = SimpleNamespace(
            job_id="job_servingsim_smoke",
            spec_json={"model_id": "meta-llama/Llama-3.1-8B"},
        )
        rank = SimpleNamespace(
            rank_id="rank_servingsim_smoke",
            n_replicas=1,
            shape_json={
                "count": 1,
                "tp": 1,
                "pp": 1,
                "ep": 1,
                "precision": "bf16",
                "instance_type": "sim.rtxpro6000.1gpu",
                "max_num_seq": 8,
                "max_num_batched_tokens": 256,
                "prefix_cache_enabled": True,
            },
        )
        deployment = build_rank_deployments(job, (rank,))[0]
        run_name = f"cluster-smoke-{uuid4().hex}"
        run_dir = LLMSERVINGSIM_ROOT / "outputs" / "cluster-sim-smoke" / run_name
        cluster_path = run_dir / "cluster.json"
        workload_path = run_dir / "workload.jsonl"
        output_path = run_dir / "requests.csv"
        metrics_path = run_dir / "metrics.jsonl"
        run_dir.mkdir(parents=True)
        completed = False

        try:
            cluster = build_rank_cluster(
                deployment,
                {
                    "profile": "RTXPRO6000",
                    "memory_gb": 96,
                    "memory_bandwidth_gb_s": 1597,
                    "link_bandwidth_gb_s": 16,
                    "link_latency_ns": 20_000,
                },
            )
            cluster_path.write_text(json.dumps(cluster))
            requests = (
                Request("request_0", "session_0", 0, 0, 16, 8, "smoke/prefix-8"),
                Request("request_1", "session_1", 0, 15_000_000_000, 24, 12, "smoke/prefix-8"),
            )
            write_rank_workload(requests, workload_path, 0)
            command = build_rank_command(
                "servingsim_docker",
                LLMSERVINGSIM_ROOT,
                cluster_path,
                workload_path,
                output_path,
                metrics_path,
                run_name,
                2,
            )
            result = run_ranks(
                (RankRun(deployment, requests, command, output_path, metrics_path, 2, 300),), 1
            )[0]

            self.assertEqual(len(result.request_rows), 2)
            self.assertTrue(result.metric_samples)
            completed = True
        finally:
            if completed:
                shutil.rmtree(run_dir)


if __name__ == "__main__":
    unittest.main()
