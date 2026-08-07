"""Benchmark controller lifecycle checks."""

import json
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from unittest import mock

import yaml

from data_generation.controller import (
    Job,
    JobKind,
    JobState,
    RunningJob,
    discard_aiperf_inputs,
    ensure_job_artifacts,
    job_command,
    load_benchmark,
    materialize_runtime_config,
    run_benchmark,
    sync_job_records,
)
from data_generation.controllers.gpt import run_gpt_controller
from data_generation.scheduling import (
    DecisionFailureCategory,
    DecisionStatus,
    OptimizationGoal,
    SchedulerJob,
    SchedulerWorkload,
    SchedulingDecision,
)
from data_generation.submission import SubmissionRequest, SubmissionResult
from data_generation.tandemn_store import ClusterStateSnapshot

ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    """Check benchmark loading, job lifecycles, pause, and resume."""

    def test_three_hour_benchmark_loads_all_jobs(self) -> None:
        benchmark = load_benchmark(ROOT / "benchmarks/simulation_3h/benchmark.yaml")

        self.assertEqual(benchmark.benchmark_id, "simulation_3h")
        self.assertEqual(benchmark.reoptimization_tick_minutes, 5)
        self.assertEqual(len(benchmark.jobs), 26)
        self.assertEqual(sum(job.enabled for job in benchmark.jobs), 26)
        self.assertEqual(len({job.config_path for job in benchmark.jobs}), 26)
        self.assertTrue(all(job.submission is not None for job in benchmark.jobs))
        self.assertTrue(
            all(
                job.submission is not None
                and job.submission.spec_json.get("precision") == "fp16"
                for job in benchmark.jobs
            )
        )
        self.assertEqual(benchmark.poll_interval_seconds, 60)

        token_fields = {
            "isl_token_avg",
            "isl_token_min",
            "isl_token_max",
            "isl_distribution_type",
            "osl_token_avg",
            "osl_token_min",
            "osl_token_max",
            "osl_distribution_type",
        }
        for job in benchmark.jobs:
            document = yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
            features = document["spec"]["features"]
            workload = document["workload"]
            self.assertTrue(token_fields <= features.keys(), job.job_id)
            if job.kind == JobKind.BATCH:
                self.assertNotIn("io", workload["dataset"], job.job_id)
                continue
            dataset = workload["benchmark"]["dataset"]
            if "rankings" in dataset:
                self.assertNotIn(
                    "passageTokens",
                    dataset["rankings"],
                    job.job_id,
                )
            else:
                prompts = dataset.get("prompts") or {}
                self.assertNotIn("isl", prompts, job.job_id)
                self.assertNotIn("osl", prompts, job.job_id)

    def test_aiperf_summary_export_excludes_raw_payload_level(self) -> None:
        """Build an AI-Perf command that requests aggregate output only."""
        benchmark = load_benchmark(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        job = next(job for job in benchmark.jobs if job.driver == "aiperf")

        command = job_command(
            job,
            Path("/tmp/ephemeral-aiperf"),
            aiperf_export_level="summary",
        )

        export_level_index = command.index("--export-level")
        self.assertEqual(command[export_level_index + 1], "summary")

    def test_discard_mode_creates_no_request_files_and_nulls_inputs(self) -> None:
        """Prevent controller and AI-Perf prompt exports from retaining text."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            job_directory = ensure_job_artifacts(
                run_directory,
                "ephemeral-job",
                create_record_files=False,
            )
            discard_aiperf_inputs(job_directory)
            inputs_path = job_directory / "inputs.json"
            temporary_inputs_path = job_directory / "inputs.tmp"

            temporary_inputs_path.write_text("must not persist", encoding="utf-8")
            temporary_inputs_path.replace(inputs_path)

            self.assertTrue(inputs_path.is_symlink())
            self.assertEqual(inputs_path.read_text(encoding="utf-8"), "")
            self.assertFalse(temporary_inputs_path.exists())
            self.assertFalse((job_directory / "prompts.jsonl").exists())
            self.assertFalse((job_directory / "responses.jsonl").exists())

    def test_online_job_stops_at_its_end_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.001,
                    process_seconds=5,
                )
            )

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="online-test",
            )

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)

    def test_batch_job_can_finish_after_its_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="batch",
                    end_field="deadline_at_offset_minutes",
                    end_minute=0.001,
                    process_seconds=0.1,
                )
            )

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="batch-test",
            )

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)
            self.assertEqual(summary.deadline_misses, frozenset({"test-job"}))

    def test_paused_job_restarts_when_benchmark_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.002,
                    process_seconds=5,
                )
            )
            real_sleep = time.sleep
            sleep_calls = 0

            def interrupt_first_poll(seconds: float) -> None:
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    raise KeyboardInterrupt
                real_sleep(seconds)

            with mock.patch(
                "data_generation.controller.time.sleep",
                side_effect=interrupt_first_poll,
            ):
                paused = run_benchmark(
                    benchmark,
                    root / "artifacts",
                    run_id="resume-test",
                )

            self.assertEqual(
                paused.states["test-job"],
                JobState.PAUSED,
            )

            resumed = run_benchmark(
                benchmark,
                root / "artifacts",
                resume_run_id="resume-test",
            )
            checkpoint_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "unmanaged"
                / "resume-test"
                / "benchmark_state.json"
            )
            job_path = checkpoint_path.parent / "jobs" / "test-job"
            state = json.loads(checkpoint_path.read_text())
            saved_job = state["jobs"][0]

            self.assertEqual(resumed.states["test-job"], JobState.COMPLETED)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(saved_job["job_id"], "test-job")
            self.assertEqual(saved_job["launch_count"], 2)
            self.assertIsNotNone(saved_job["paused_at"])
            self.assertIsNotNone(saved_job["resumed_at"])
            self.assertTrue((job_path / "prompts.jsonl").is_file())
            self.assertTrue((job_path / "responses.jsonl").is_file())

    def test_resume_reuses_the_saved_tandemn_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.002,
                    process_seconds=5,
                    with_submission=True,
                )
            )
            submitter = FakeSubmitter()
            real_sleep = time.sleep
            sleep_calls = 0

            def interrupt_first_poll(seconds: float) -> None:
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    raise KeyboardInterrupt
                real_sleep(seconds)

            with mock.patch(
                "data_generation.controller.time.sleep",
                side_effect=interrupt_first_poll,
            ):
                paused = run_benchmark(
                    benchmark,
                    root / "artifacts",
                    run_id="submission-resume-test",
                    job_submitter=submitter,
                )

            self.assertEqual(paused.states["test-job"], JobState.PAUSED)

            resumed = run_benchmark(
                benchmark,
                root / "artifacts",
                resume_run_id="submission-resume-test",
                job_submitter=submitter,
            )
            state_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "unmanaged"
                / "submission-resume-test"
                / "benchmark_state.json"
            )
            saved_job = json.loads(state_path.read_text())["jobs"][0]

            self.assertEqual(resumed.states["test-job"], JobState.COMPLETED)
            self.assertEqual(len(submitter.requests), 1)
            self.assertEqual(saved_job["tandemn_job_id"], "tandemn-test-job")
            self.assertEqual(saved_job["endpoint_url"], submitter.endpoint_url)

    def test_submission_waits_for_endpoint_before_starting_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.01,
                    process_seconds=0.02,
                    with_submission=True,
                )
            )
            submitter = FakeSubmitter(defer_endpoint=True)

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="deferred-endpoint-test",
                job_submitter=submitter,
            )

            state_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "unmanaged"
                / "deferred-endpoint-test"
                / "benchmark_state.json"
            )
            saved_job = json.loads(state_path.read_text())["jobs"][0]

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)
            self.assertEqual(len(submitter.requests), 1)
            self.assertGreaterEqual(len(submitter.polls), 2)
            self.assertEqual(saved_job["tandemn_job_id"], "tandemn-test-job")
            self.assertEqual(saved_job["endpoint_url"], submitter.endpoint_url)
            self.assertEqual(saved_job["launch_count"], 1)

    def test_runtime_aiperf_config_owns_the_returned_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_config = root / "source.yaml"
            source_config.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "benchmark-job/v1",
                        "job_id": "online-job",
                        "schedule": {},
                        "spec": {
                            "user_id": "test-user",
                            "optimization_goal": "latency",
                            "features": {},
                        },
                        "workload": {
                            "schemaVersion": "2.0",
                            "benchmark": {
                                "model": "test-model",
                                "endpoint": {
                                    "type": "chat",
                                    "streaming": True,
                                },
                            },
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            job = Job(
                job_id="online-job",
                enabled=True,
                output_to_terminal=False,
                kind=JobKind.ONLINE,
                driver="aiperf",
                config_path=source_config,
                command=None,
                start_minute=0,
                stop_minute=1,
                deadline_minute=None,
                hard_stop_minute=1,
                parent_job_id=None,
                submission=SubmissionRequest(
                    user_id="test-user",
                    kind="online",
                    spec_json={"model_id": "test-model"},
                ),
            )
            artifact_directory = root / "artifacts"
            artifact_directory.mkdir()

            runtime_config = materialize_runtime_config(
                job,
                artifact_directory,
                "https://runtime.example/v1/chat/completions",
            )
            source = yaml.safe_load(source_config.read_text())
            runtime = yaml.safe_load(runtime_config.read_text())

            self.assertNotIn(
                "url",
                source["workload"]["benchmark"]["endpoint"],
            )
            self.assertNotIn("job", runtime)
            self.assertEqual(
                runtime["benchmark"]["endpoint"]["url"],
                "https://runtime.example/v1/chat/completions",
            )

    def test_runtime_aiperf_uses_the_canonical_job_token_shape(self) -> None:
        benchmark = load_benchmark(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        code_job = next(
            job
            for job in benchmark.jobs
            if job.job_id == "job-benchmark_timing_3hr-0-3"
        )
        rerank_job = next(
            job
            for job in benchmark.jobs
            if job.job_id == "job-benchmark_timing_3hr-7-2"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            code_runtime = materialize_runtime_config(
                code_job,
                artifact_directory,
                "https://runtime.example/v1/chat/completions",
            )
            code_document = yaml.safe_load(code_runtime.read_text(encoding="utf-8"))
            code_prompts = code_document["benchmark"]["dataset"]["prompts"]
            self.assertEqual(code_prompts["isl"]["mean"], 1800)
            self.assertEqual(code_prompts["osl"]["mean"], 120)

            rerank_runtime = materialize_runtime_config(
                rerank_job,
                artifact_directory,
                "https://runtime.example/rerank",
            )
            rerank_document = yaml.safe_load(rerank_runtime.read_text(encoding="utf-8"))
            rankings = rerank_document["benchmark"]["dataset"]["rankings"]
            aggregate_mean = (
                rankings["queryTokens"]["mean"]
                + rankings["passages"]["mean"] * rankings["passageTokens"]["mean"]
            )
            self.assertAlmostEqual(aggregate_mean, 512)

    def test_scheduler_placement_precedes_sending_and_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.002,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            benchmark = replace(
                benchmark,
                jobs=tuple(
                    replace(
                        job,
                        workload=SchedulerWorkload(
                            input_tokens=10,
                            output_tokens=10,
                            request_rate=1,
                        ),
                    )
                    for job in benchmark.jobs
                ),
            )
            method = FakeSchedulingMethod()
            submitter = FakeSubmitter()

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="scheduled-send-test",
                job_submitter=submitter,
                scheduling_method=method,
            )

            placement_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "scheduled-send-test"
                / "jobs"
                / "test-job"
                / "placement.json"
            )
            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)
            self.assertEqual(method.scheduled_jobs, ["test-job"])
            self.assertEqual(method.released_jobs, ["test-job"])
            self.assertEqual(len(submitter.requests), 1)
            self.assertTrue(placement_path.is_file())
            plan_path = placement_path.parents[2] / "placement.json"
            plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_action = plan_document["ticks"][0]["plan"]["actions"][0]
            self.assertEqual(plan_action["job_id"], "test-job")
            self.assertEqual(plan_action["type"], "place")
            final_action = plan_document["ticks"][-1]["plan"]["actions"][0]
            self.assertEqual(final_action["type"], "preempt")

    def test_deferred_job_is_retried_with_the_previous_cluster_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.004,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            benchmark = replace(
                benchmark,
                jobs=tuple(
                    replace(
                        job,
                        workload=SchedulerWorkload(
                            input_tokens=10,
                            output_tokens=10,
                            request_rate=1,
                        ),
                    )
                    for job in benchmark.jobs
                ),
            )
            method = FakeSchedulingMethod(defer_first=True)

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="deferred-retry-test",
                job_submitter=FakeSubmitter(),
                scheduling_method=method,
            )

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)
            self.assertGreaterEqual(len(method.scheduled_jobs), 2)
            self.assertIsNone(method.past_plans[0])
            second_plan = method.past_plans[1]
            self.assertIsNotNone(second_plan)
            self.assertEqual(second_plan["actions"][0]["type"], "defer")

            plan_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "deferred-retry-test"
                / "placement.json"
            )
            plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
            action_types = [
                tick["plan"]["actions"][0]["type"] for tick in plan_document["ticks"]
            ]
            self.assertEqual(action_types[0], "defer")
            self.assertIn("place", action_types)
            self.assertEqual(action_types[-1], "preempt")

    def test_planner_retries_deferred_jobs_on_the_next_minute_tick(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=3,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            benchmark = replace(
                benchmark,
                jobs=tuple(
                    replace(
                        job,
                        workload=SchedulerWorkload(
                            input_tokens=10,
                            output_tokens=10,
                            request_rate=1,
                        ),
                    )
                    for job in benchmark.jobs
                ),
            )
            method = FakeSchedulingMethod(defer_first=True)

            summary = run_gpt_controller(
                benchmark,
                root / "artifacts",
                method,
                state_provider=FakeStateProvider(),
                sleep=lambda _seconds: None,
                run_id="planner-deferred-retry-test",
            )

            self.assertEqual(len(method.scheduled_jobs), 2)
            self.assertEqual(
                method.scheduled_ticks,
                [("test-job",), ("test-job",)],
            )
            second_plan = method.past_plans[1]
            self.assertIsNotNone(second_plan)
            self.assertEqual(second_plan["actions"][0]["type"], "defer")
            self.assertEqual(
                summary.decisions["test-job"].status,
                DecisionStatus.ACCEPTED,
            )

            plan_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "planner-deferred-retry-test"
                / "placement.json"
            )
            plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [tick["plan"]["actions"][0]["type"] for tick in plan_document["ticks"]],
                ["defer", "place", "place", "place"],
            )

    def test_gpt_controller_appends_jobs_and_updates_types_from_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=3,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            first_job = replace(
                benchmark.jobs[0],
                workload=SchedulerWorkload(
                    input_tokens=10,
                    output_tokens=10,
                    request_rate=1,
                ),
            )
            second_job = replace(
                first_job,
                job_id="second-job",
                start_minute=1,
            )
            benchmark = replace(benchmark, jobs=(first_job, second_job))
            method = FakeSchedulingMethod()
            state_provider = FakeStateProvider(
                snapshots=(
                    cluster_state(),
                    cluster_state(running=("test-job",)),
                    cluster_state(
                        running=("second-job",),
                        finished=("test-job",),
                    ),
                    cluster_state(
                        running=("second-job",),
                        finished=("test-job",),
                    ),
                )
            )

            run_gpt_controller(
                benchmark,
                root / "artifacts",
                method,
                state_provider=state_provider,
                sleep=lambda _seconds: None,
                run_id="planner-batch-test",
            )

            self.assertEqual(
                method.scheduled_ticks,
                [("test-job",), ("second-job",)],
            )
            plan_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "planner-batch-test"
                / "placement.json"
            )
            plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(len(plan_document["ticks"]), 4)
            self.assertEqual(
                [
                    action["type"]
                    for action in plan_document["ticks"][0]["plan"]["actions"]
                ],
                ["place"],
            )
            self.assertEqual(
                [
                    action["type"]
                    for action in plan_document["ticks"][1]["plan"]["actions"]
                ],
                ["keep", "place"],
            )
            self.assertEqual(
                [
                    action["type"]
                    for action in plan_document["ticks"][2]["plan"]["actions"]
                ],
                ["preempt", "keep"],
            )
            self.assertEqual(
                [
                    action["job_id"]
                    for action in plan_document["ticks"][2]["plan"]["actions"]
                ],
                ["test-job", "second-job"],
            )

    def test_gpt_controller_submits_jobs_and_publishes_state_legal_plans(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=1,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            benchmark = replace(
                benchmark,
                jobs=tuple(
                    replace(
                        job,
                        workload=SchedulerWorkload(
                            input_tokens=10,
                            output_tokens=10,
                            request_rate=1,
                        ),
                    )
                    for job in benchmark.jobs
                ),
            )
            method = FakeSchedulingMethod()
            submitter = ExactJobSubmitter()
            publisher = FakePlanPublisher()
            state_provider = FakeStateProvider(
                snapshots=(
                    cluster_state(),
                    cluster_state(waiting=("test-job",)),
                    cluster_state(running=("test-job",)),
                )
            )

            run_gpt_controller(
                benchmark,
                root / "artifacts",
                method,
                state_provider=state_provider,
                plan_publisher=publisher,
                job_submitter=submitter,
                sleep=lambda _seconds: None,
                run_id="store-loop-test",
            )

            self.assertEqual(submitter.requests[0].job_id, "test-job")
            self.assertEqual(method.scheduled_ticks, [("test-job",)])
            self.assertEqual(
                [plan["actions"][0]["type"] for plan in publisher.plans],
                ["place", "keep"],
            )

    def test_gpt_controller_does_not_place_paused_or_publish_finished_jobs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=1,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            method = FakeSchedulingMethod()
            publisher = FakePlanPublisher()
            state_provider = FakeStateProvider(
                snapshots=(
                    cluster_state(paused=("test-job",)),
                    cluster_state(finished=("test-job",)),
                )
            )

            run_gpt_controller(
                benchmark,
                root / "artifacts",
                method,
                state_provider=state_provider,
                plan_publisher=publisher,
                sleep=lambda _seconds: None,
                run_id="terminal-history-test",
            )

            self.assertEqual(method.scheduled_ticks, [])
            self.assertEqual(publisher.plans[0]["actions"], [])
            self.assertEqual(publisher.plans[1]["actions"], [])
            plan_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "terminal-history-test"
                / "placement.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [tick["plan"]["actions"][0]["type"] for tick in plan["ticks"]],
                ["defer", "preempt"],
            )

    def test_gpt_controller_uses_reoptimization_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=5,
                    process_seconds=0.01,
                    with_submission=True,
                )
            )
            benchmark = replace(benchmark, reoptimization_tick_minutes=2)
            benchmark = replace(
                benchmark,
                jobs=tuple(
                    replace(
                        job,
                        workload=SchedulerWorkload(
                            input_tokens=10,
                            output_tokens=10,
                            request_rate=1,
                        ),
                    )
                    for job in benchmark.jobs
                ),
            )
            sleep_calls: list[float] = []

            run_gpt_controller(
                benchmark,
                root / "artifacts",
                FakeSchedulingMethod(),
                state_provider=FakeStateProvider(),
                sleep=sleep_calls.append,
                run_id="tick-interval-test",
            )

            plan_path = (
                root
                / "artifacts"
                / "adhoc"
                / "test-benchmark"
                / "fake"
                / "tick-interval-test"
                / "placement.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [tick["offset_minutes"] for tick in plan["ticks"]],
                [0, 2, 4, 5],
            )
            self.assertEqual(sleep_calls, [120, 120, 60])

    def test_aiperf_raw_records_are_split_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            job_path = run_directory / "jobs" / "online-job"
            raw_path = job_path / "raw_records" / "raw_records_worker.jsonl"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "x_request_id": "request-1",
                            "request_start_ns": 1_750_000_000_000_000_000,
                            "request_end_ns": 1_750_000_001_000_000_000,
                        },
                        "payload": {"messages": [{"content": "hello"}]},
                        "status": 200,
                        "responses": [{"content": "world"}],
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = RunningJob(
                job=Job(
                    job_id="online-job",
                    enabled=True,
                    output_to_terminal=False,
                    kind=JobKind.ONLINE,
                    driver="aiperf",
                    config_path=run_directory / "job.yaml",
                    command=None,
                    start_minute=0,
                    stop_minute=1,
                    deadline_minute=None,
                    hard_stop_minute=1,
                    parent_job_id=None,
                    submission=None,
                ),
                state=JobState.RUNNING,
            )

            sync_job_records(item, run_directory)
            sync_job_records(item, run_directory)
            resumed_item = RunningJob(job=item.job, state=JobState.RUNNING)
            sync_job_records(resumed_item, run_directory)

            prompts = read_jsonl(job_path / "prompts.jsonl")
            responses = read_jsonl(job_path / "responses.jsonl")
            self.assertEqual(len(prompts), 1)
            self.assertEqual(len(responses), 1)
            self.assertEqual(prompts[0]["request_id"], "request-1")
            self.assertEqual(responses[0]["request_id"], "request-1")
            self.assertEqual(prompts[0]["request"]["messages"][0]["content"], "hello")
            self.assertEqual(responses[0]["response"][0]["content"], "world")


def write_benchmark(
    root: Path,
    *,
    kind: str,
    end_field: str,
    end_minute: float,
    process_seconds: float,
    with_submission: bool = False,
) -> Path:
    """Write one small command-backed benchmark."""
    jobs_directory = root / "jobs"
    jobs_directory.mkdir()
    config = jobs_directory / "test-job.yaml"
    spec: dict[str, object] | None = None
    if with_submission:
        spec = {
            "user_id": "test-user",
            "optimization_goal": "latency",
            "model_id": "test-model",
            "features": {},
        }
    schedule: dict[str, object] = {
        "enabled": True,
        "kind": kind,
        "driver": "external_mock_data",
        "command": [
            sys.executable,
            "-c",
            f"import time; time.sleep({process_seconds})",
        ],
        "start_at_offset_minutes": 0,
        end_field: end_minute,
    }
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "benchmark-job/v1",
                "job_id": "test-job",
                "schedule": schedule,
                "spec": spec,
                "workload": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = root / "benchmark.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "benchmark-manifest/v1",
                "benchmark": {
                    "id": "test-benchmark",
                    "status": "executable",
                    "jobs": "jobs",
                    "poll_interval_seconds": 0.01,
                    "shutdown_grace_seconds": 0.05,
                    "hard_stop_at_offset_minutes": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


class FakeSubmitter:
    """Return one stable endpoint while recording submission attempts."""

    endpoint_url = "https://runtime.example/v1/chat/completions"

    def __init__(self, *, defer_endpoint: bool = False) -> None:
        self.requests: list[SubmissionRequest] = []
        self.defer_endpoint = defer_endpoint
        self.polls: list[str] = []

    def submit(self, request: SubmissionRequest) -> SubmissionResult:
        self.requests.append(request)
        return SubmissionResult(
            tandemn_job_id="tandemn-test-job",
            endpoint_url=None if self.defer_endpoint else self.endpoint_url,
        )

    def poll_endpoint(self, tandemn_job_id: str) -> str | None:
        self.polls.append(tandemn_job_id)
        if len(self.polls) < 2:
            return None
        return self.endpoint_url


class ExactJobSubmitter:
    """Return the benchmark-owned job ID used for Store reconciliation."""

    def __init__(self) -> None:
        self.requests: list[SubmissionRequest] = []

    def submit(self, request: SubmissionRequest) -> SubmissionResult:
        self.requests.append(request)
        if request.job_id is None:
            raise AssertionError("GPT jobs must carry their canonical job_id")
        return SubmissionResult(tandemn_job_id=request.job_id)

    def poll_endpoint(self, tandemn_job_id: str) -> str | None:
        return None


class FakePlanPublisher:
    """Record the state-legal subset published for Orca."""

    def __init__(self) -> None:
        self.plans: list[Mapping[str, object]] = []

    def publish_plan(
        self,
        plan: Mapping[str, object],
        method_name: str,
    ) -> str:
        self.plans.append(plan)
        return f"plan-{len(self.plans)}"


class FakeSchedulingMethod:
    """Accept one placement and record its allocation lifecycle."""

    method_name = "fake"

    def __init__(self, *, defer_first: bool = False) -> None:
        self.scheduled_jobs: list[str] = []
        self.released_jobs: list[str] = []
        self.past_plans: list[Mapping[str, object] | None] = []
        self.scheduled_ticks: list[tuple[str, ...]] = []
        self.defer_first = defer_first

    def schedule_tick(
        self,
        jobs: tuple[SchedulerJob, ...],
        past_plan: Mapping[str, object] | None = None,
        *,
        cluster_state: ClusterStateSnapshot | None = None,
        seen_jobs: tuple[SchedulerJob, ...] | None = None,
    ) -> Mapping[str, SchedulingDecision]:
        if cluster_state is None:
            raise AssertionError("controller must supply Tandemn Store state")
        self.scheduled_ticks.append(tuple(job.job_id for job in jobs))
        return {job.job_id: self.schedule(job, past_plan) for job in jobs}

    def schedule(
        self,
        job: SchedulerJob,
        past_plan: Mapping[str, object] | None = None,
    ) -> SchedulingDecision:
        self.scheduled_jobs.append(job.job_id)
        self.past_plans.append(past_plan)
        before = {"available": 1}
        if self.defer_first and len(self.scheduled_jobs) == 1:
            return SchedulingDecision(
                job_id=job.job_id,
                method=self.method_name,
                status=DecisionStatus.REJECTED,
                optimization_goal=OptimizationGoal.THROUGHPUT,
                candidates=(),
                resource_snapshot_before=before,
                resource_snapshot_after=before,
                failure_category=DecisionFailureCategory.INSUFFICIENT_CAPACITY,
                reason="No capacity is currently available",
            )
        after = {"available": 0}
        return SchedulingDecision(
            job_id=job.job_id,
            method=self.method_name,
            status=DecisionStatus.ACCEPTED,
            optimization_goal=OptimizationGoal.THROUGHPUT,
            candidates=(),
            resource_snapshot_before=before,
            resource_snapshot_after=after,
            pool_id="test-pool",
            accelerator="test-accelerator",
            devices=1,
            latency_ms=1,
            throughput_tokens_per_second=1,
            configuration={"parallel": "tp1"},
        )

    def release(self, job_id: str) -> None:
        self.released_jobs.append(job_id)

    def resource_snapshot(self) -> dict[str, object]:
        return {
            "available": 1 if self.released_jobs else 0,
        }


class FakeStateProvider:
    """Return one Store snapshot per controller tick."""

    def __init__(
        self,
        snapshots: tuple[ClusterStateSnapshot, ...] | None = None,
    ) -> None:
        self.snapshots = snapshots or (cluster_state(),)
        self.calls = 0

    def read_cluster_state(self) -> ClusterStateSnapshot:
        """Return the next snapshot, retaining the last state when exhausted."""
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return self.snapshots[index]


def cluster_state(
    *,
    running: tuple[str, ...] = (),
    waiting: tuple[str, ...] = (),
    paused: tuple[str, ...] = (),
    finished: tuple[str, ...] = (),
) -> ClusterStateSnapshot:
    """Build a Store snapshot with jobs indexed by benchmark job_id."""
    return ClusterStateSnapshot(
        user_id="usr-test",
        resource_map_version=1,
        resource_map_updated_at="2026-08-07T18:00:00+00:00",
        resources={},
        running_jobs=tuple(
            {"job_id": job_id, "status": "running", "active_chains": []}
            for job_id in running
        ),
        waiting_jobs=tuple(
            {"job_id": job_id, "status": "waiting"} for job_id in waiting
        ),
        paused_jobs=tuple({"job_id": job_id, "status": "paused"} for job_id in paused),
        finished_jobs=tuple(
            {"job_id": job_id, "status": "finished"} for job_id in finished
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL artifact into records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
