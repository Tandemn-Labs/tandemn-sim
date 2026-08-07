"""Tests for the single-job AIPerf controller."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_generation.controller import (
    ROOT,
    build_aiperf_config,
    find_job,
    run_job,
)


class ControllerTests(unittest.TestCase):
    """Verify benchmark job resolution and AIPerf startup."""

    def test_find_job_resolves_an_exact_job_id(self) -> None:
        path = find_job("job-benchmark_timing_3hr-0-1")

        self.assertEqual(path.name, "chat-free-core.yaml")

    def test_find_job_rejects_an_unknown_job_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown benchmark job"):
            find_job("missing-job")

    def test_build_config_rejects_non_aiperf_jobs(self) -> None:
        path = (
            ROOT
            / "benchmarks"
            / "simulation_3h"
            / "jobs"
            / "batch-mini-nightly-eval.yaml"
        )

        with self.assertRaisesRegex(ValueError, "does not use AIPerf"):
            build_aiperf_config(path)

    def test_build_config_materializes_prompt_token_shapes(self) -> None:
        path = find_job("job-benchmark_timing_3hr-0-1")

        config = build_aiperf_config(path)

        prompts = config["benchmark"]["dataset"]["prompts"]
        self.assertEqual(
            prompts["isl"],
            {"mean": 900.0, "min": 225.0, "max": 2200.0},
        )
        self.assertEqual(
            prompts["osl"],
            {"mean": 300.0, "min": 75.0, "max": 700.0},
        )
        self.assertNotIn("schedule", config)
        self.assertNotIn("spec", config)

    def test_build_config_materializes_ranking_passage_tokens(self) -> None:
        path = find_job("job-benchmark_timing_3hr-7-2")

        config = build_aiperf_config(path)

        passage_tokens = config["benchmark"]["dataset"]["rankings"]["passageTokens"]
        self.assertAlmostEqual(passage_tokens["mean"], (512 - 8.031) / 3.688)
        self.assertEqual(passage_tokens["min"], 127.0)
        self.assertAlmostEqual(passage_tokens["max"], (2048 - 35) / 6)

    @mock.patch("data_generation.controller.subprocess.run")
    def test_run_job_starts_one_aiperf_process(
        self,
        run_process: mock.Mock,
    ) -> None:
        run_process.return_value = subprocess.CompletedProcess((), 0)

        exit_code = run_job("job-benchmark_timing_3hr-0-1")

        self.assertEqual(exit_code, 0)
        command = run_process.call_args.args[0]
        self.assertEqual(command[:3], ("aiperf", "profile", "--config"))
        self.assertFalse(run_process.call_args.kwargs["check"])

    @mock.patch("data_generation.controller.subprocess.run")
    def test_run_job_returns_the_aiperf_exit_code(
        self,
        run_process: mock.Mock,
    ) -> None:
        run_process.return_value = subprocess.CompletedProcess((), 17)

        self.assertEqual(run_job("job-benchmark_timing_3hr-0-1"), 17)

    def test_all_runtime_configs_are_valid_for_aiperf(self) -> None:
        job_paths = sorted((ROOT / "benchmarks").glob("*/jobs/*.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aiperf.yaml"
            for job_path in job_paths:
                document = yaml.safe_load(job_path.read_text(encoding="utf-8"))
                if document["schedule"]["driver"] != "aiperf":
                    continue
                with self.subTest(job_id=document["job_id"]):
                    config = build_aiperf_config(job_path)
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    completed = subprocess.run(
                        ("aiperf", "config", "validate", str(path)),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        completed.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
