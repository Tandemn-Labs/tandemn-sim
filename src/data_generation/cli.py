"""Run one benchmark-method pair or an authoritative experiment matrix."""

from __future__ import annotations

import argparse
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from data_generation.batch_dataset_generator import generate_batch_dataset
from data_generation.benchmarks.registry import BenchmarkDefinition, resolve_benchmark
from data_generation.controller import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BENCHMARK,
    ROOT,
    interrupt_controller,
    load_benchmark,
    log_benchmark,
    run_benchmark,
    run_placement_only,
)
from data_generation.experiments.registry import load_experiment
from data_generation.methods.registry import list_methods, load_method

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSelection:
    """One concrete experiment, benchmark, and method execution."""

    experiment_id: str
    benchmark: BenchmarkDefinition
    method_id: str


def main() -> None:
    """Parse the repository-level benchmark interface and execute it."""
    load_dotenv(ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_batch is not None:
        _run_batch_generation(parser, args)
        return
    if args.output is not None or args.records is not None:
        parser.error("--output and --records require --generate-batch")
    if args.legacy_benchmark is not None and args.benchmark is not None:
        parser.error("use either the positional benchmark or --benchmark, not both")
    if args.placement_only and args.resume is not None:
        parser.error("--placement-only cannot be resumed")
    if args.discard_request_data and args.aiperf_export_level != "summary":
        parser.error("--discard-request-data requires --aiperf-export-level summary")

    benchmark_filter = args.benchmark or args.legacy_benchmark
    try:
        selections = resolve_run_selections(
            experiment_id=args.experiment,
            benchmark_filter=benchmark_filter,
            method_filter=args.method,
            validate_only=args.validate_only,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    if args.validate_only:
        LOGGER.info("[validation] selections=%d", len(selections))
        return

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_controller)
    try:
        for selection in selections:
            execute_selection(selection, args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    """Create the single top-level CLI for benchmarks and experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "legacy_benchmark",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmark",
        help="benchmark catalog ID or explicit benchmark.yaml path",
    )
    parser.add_argument(
        "--method",
        help="method catalog ID",
    )
    parser.add_argument(
        "--experiment",
        help="experiment catalog ID; optional benchmark/method flags filter its matrix",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--aiperf-export-level",
        choices=("summary", "records", "raw"),
        default="raw",
        help="AI-Perf output detail; summary excludes per-request payloads",
    )
    parser.add_argument(
        "--discard-request-data",
        action="store_true",
        help="do not create controller request files and discard AI-Perf inputs",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="resume a paused benchmark run without advancing paused time",
    )
    parser.add_argument(
        "--placement-only",
        action="store_true",
        help="simulate placements without waiting or starting data senders",
    )
    parser.add_argument(
        "--generate-batch",
        type=Path,
        metavar="WORKLOAD",
        help="generate batch JSONL from a batch workload YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL output path; otherwise records print to the terminal",
    )
    parser.add_argument(
        "--records",
        type=int,
        help="optional generation limit used with --generate-batch",
    )
    return parser


def resolve_run_selections(
    *,
    experiment_id: str | None,
    benchmark_filter: str | None,
    method_filter: str | None,
    validate_only: bool,
) -> tuple[RunSelection, ...]:
    """Resolve direct flags or an experiment matrix into concrete runs."""
    if experiment_id is None:
        benchmark = resolve_benchmark(benchmark_filter or DEFAULT_BENCHMARK)
        if method_filter is None:
            if validate_only:
                _validate_benchmark(benchmark)
                return ()
            raise ValueError("--method is required when --experiment is not used")
        selection = RunSelection("adhoc", benchmark, method_filter)
        _validate_selection(selection, validate_only=validate_only)
        return (selection,)

    experiment = load_experiment(experiment_id)
    benchmark_ids = _filtered_values(
        experiment.benchmarks,
        benchmark_filter,
        "benchmark",
    )
    method_ids = _filtered_values(experiment.methods, method_filter, "method")
    selections = tuple(
        RunSelection(
            experiment.experiment_id,
            resolve_benchmark(benchmark_id),
            method_id,
        )
        for benchmark_id in benchmark_ids
        for method_id in method_ids
    )
    for selection in selections:
        _validate_selection(selection, validate_only=validate_only)
    return selections


def execute_selection(selection: RunSelection, args: argparse.Namespace) -> None:
    """Run one fully resolved benchmark-method selection."""
    benchmark = load_benchmark(selection.benchmark.manifest)
    method = load_method(selection.method_id, selection.benchmark.manifest.parent)
    LOGGER.info(
        "[selection] experiment=%s benchmark=%s method=%s",
        selection.experiment_id,
        benchmark.benchmark_id,
        selection.method_id,
    )
    if not args.placement_only:
        log_benchmark(benchmark)
    if args.placement_only:
        run_placement_only(
            benchmark,
            args.artifact_root,
            method,
            experiment_id=selection.experiment_id,
            method_id=selection.method_id,
        )
        return
    run_benchmark(
        benchmark,
        args.artifact_root,
        resume_run_id=args.resume,
        aiperf_export_level=args.aiperf_export_level,
        discard_request_data=args.discard_request_data,
        scheduling_method=method,
        experiment_id=selection.experiment_id,
        method_id=selection.method_id,
    )


def _validate_selection(
    selection: RunSelection,
    *,
    validate_only: bool,
) -> None:
    methods = list_methods()
    try:
        method = methods[selection.method_id]
    except KeyError as error:
        raise ValueError(f"Unknown method: {selection.method_id}") from error
    if not validate_only and method.status != "implemented":
        raise ValueError(
            f"Method {selection.method_id} is cataloged but not implemented"
        )
    if not validate_only:
        _validate_benchmark(selection.benchmark)


def _validate_benchmark(benchmark: BenchmarkDefinition) -> None:
    if benchmark.status != "executable":
        raise ValueError(
            f"Benchmark {benchmark.benchmark_id} is {benchmark.status}, not executable"
        )
    load_benchmark(benchmark.manifest)


def _filtered_values(
    declared: tuple[str, ...],
    selected: str | None,
    name: str,
) -> tuple[str, ...]:
    if selected is None:
        return declared
    if selected not in declared:
        raise ValueError(f"{name} {selected} is not in the selected experiment")
    return (selected,)


def _run_batch_generation(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    try:
        generate_batch_dataset(
            args.generate_batch,
            args.output,
            record_limit=args.records,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
