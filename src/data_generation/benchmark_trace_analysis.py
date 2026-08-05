"""Generate job-specific traffic analyses from a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from data_generation.traffic_trace_analysis import (
    _parse_timestamp_seconds,
    analyze_trace,
)

LOGGER = logging.getLogger(__name__)
SELECTION_BUCKET_SECONDS = 60
SELECTION_STEP_BUCKETS = 5


@dataclass(frozen=True)
class TrafficRecord:
    """A source request reduced to traffic-analysis fields."""

    timestamp: float
    elapsed: float | None
    input_tokens: int
    output_tokens: int

    def as_row(self) -> dict[str, float | int | None]:
        """Return the normalized row accepted by the shared analyzer."""
        return {
            "timestamp_seconds": self.timestamp,
            "elapsed_seconds": self.elapsed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class JobSpec:
    """Describe one benchmark job's trace-analysis source and selection."""

    slug: str
    job_id: str
    pattern: str
    duration_minutes: int
    source_id: str
    source_kind: str
    source_path: Path
    variant: str = ""
    filter_description: str = "none"
    reuse_window_from: str | None = None
    exclude_windows_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowSelection:
    """Record a deterministic source window selected for one job."""

    start: float
    end: float
    score: float
    requests: int


@dataclass(frozen=True)
class BenchmarkManifest:
    """Hold resolved job definitions and their output directory."""

    output_directory: Path
    jobs: tuple[JobSpec, ...]


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load a benchmark's trace-analysis configuration."""
    raw_data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("Manifest must be a YAML mapping")

    data = cast(dict[str, Any], raw_data)
    if data.get("schema_version") != "traffic-analysis/v1":
        raise ValueError("Unsupported manifest schema version")

    raw_sources = data.get("sources")
    raw_jobs = data.get("jobs")
    if not isinstance(raw_sources, dict) or not isinstance(raw_jobs, list):
        raise ValueError("Manifest requires sources and jobs")

    sources = cast(dict[str, dict[str, Any]], raw_sources)
    jobs_data = cast(list[dict[str, Any]], raw_jobs)
    jobs: list[JobSpec] = []
    for job in jobs_data:
        source_id = str(job["source"])
        source = sources[source_id]
        reuse_window = job.get("reuse_window_from")
        jobs.append(
            JobSpec(
                slug=str(job["slug"]),
                job_id=str(job["job_id"]),
                pattern=str(job["pattern"]),
                duration_minutes=int(job["duration_minutes"]),
                source_id=source_id,
                source_kind=str(source["kind"]),
                source_path=path.parent / str(source["path"]),
                variant=str(job.get("variant", "")),
                filter_description=str(job.get("filter_description", "none")),
                reuse_window_from=(
                    str(reuse_window) if reuse_window is not None else None
                ),
                exclude_windows_from=tuple(
                    str(slug) for slug in job.get("exclude_windows_from", ())
                ),
            )
        )

    slugs = [job.slug for job in jobs]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Job slugs must be unique")

    output_directory = path.parent / str(data.get("output_directory", "traces"))
    return BenchmarkManifest(output_directory, tuple(jobs))


def _iter_records(spec: JobSpec) -> Iterator[TrafficRecord]:
    if spec.source_kind == "azure":
        yield from _iter_azure(spec.source_path)
    elif spec.source_kind == "burstgpt":
        yield from _iter_burstgpt(spec.source_path, spec.variant)
    elif spec.source_kind == "bailian":
        yield from _iter_bailian(spec.source_path)
    elif spec.source_kind == "ragpulse":
        yield from _iter_ragpulse(spec.source_path, spec.variant)
    elif spec.source_kind == "normalized_csv":
        yield from _iter_normalized_csv(spec.source_path)
    elif spec.source_kind == "mooncake":
        yield from _iter_mooncake(spec.source_path)
    elif spec.source_kind == "swiss":
        yield from _iter_swiss(spec.source_path)
    else:
        raise ValueError(f"Unsupported source kind: {spec.source_kind}")


def _iter_azure(path: Path) -> Iterator[TrafficRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            yield TrafficRecord(
                timestamp=_parse_timestamp_seconds(row[0]),
                elapsed=None,
                input_tokens=int(row[1]),
                output_tokens=int(row[2]),
            )


def _iter_burstgpt(path: Path, variant: str) -> Iterator[TrafficRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            input_tokens = int(row[4])
            output_tokens = int(row[5])
            model = row[3]
            log_type = row[7]
            if variant == "gpt4_conversation" and not (
                model == "GPT-4" and log_type == "Conversation log"
            ):
                continue
            if variant == "short_api" and not (
                log_type == "API log" and input_tokens <= 512 and output_tokens <= 32
            ):
                continue
            if variant == "api" and log_type != "API log":
                continue
            if variant == "safety_api" and not (
                log_type == "API log"
                and 256 <= input_tokens <= 2048
                and 64 <= output_tokens <= 512
            ):
                continue
            yield TrafficRecord(
                timestamp=float(row[0]),
                elapsed=float(row[2]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


def _iter_bailian(path: Path) -> Iterator[TrafficRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield TrafficRecord(
                timestamp=float(row["timestamp"]),
                elapsed=None,
                input_tokens=int(row["input_length"]),
                output_tokens=int(row["output_length"]),
            )


def _load_token_lengths(path: Path, id_column: str) -> dict[int, int]:
    lengths = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[int(row[id_column])] = int(row["token_length"])
    return lengths


def _iter_ragpulse(path: Path, variant: str) -> Iterator[TrafficRecord]:
    data_root = path.parent
    user_lengths = _load_token_lengths(
        data_root / "4_user_input.jsonl", "user_input_id"
    )
    passage_lengths = _load_token_lengths(data_root / "2_passages.jsonl", "passage_id")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hashes = row["hash_ids"]
            user_tokens = sum(
                user_lengths.get(int(identifier), 0)
                for identifier in hashes["user_input"]
            )
            passage_tokens = sum(
                passage_lengths.get(int(identifier), 0)
                for identifier in hashes["passages_ids"]
            )
            if variant == "embedding":
                input_tokens = user_tokens
                output_tokens = 0
            elif variant == "rerank":
                input_tokens = user_tokens + passage_tokens
                output_tokens = 0
            else:
                input_tokens = int(row["input_length"])
                output_tokens = int(row["output_length"])
            if input_tokens <= 0:
                continue
            yield TrafficRecord(
                timestamp=float(row["timestamp"]),
                elapsed=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


def _iter_normalized_csv(path: Path) -> Iterator[TrafficRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield TrafficRecord(
                timestamp=float(row["timestamp_seconds"]),
                elapsed=float(row["elapsed_seconds"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
            )


def _iter_mooncake(path: Path) -> Iterator[TrafficRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            input_tokens = int(row["input_length"])
            if input_tokens < 24_000:
                continue
            yield TrafficRecord(
                timestamp=float(row["timestamp"]),
                elapsed=None,
                input_tokens=input_tokens,
                output_tokens=int(row["output_length"]),
            )


def _iter_swiss(path: Path) -> Iterator[TrafficRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            input_tokens = row.get("input_tokens")
            output_tokens = row.get("output_tokens")
            start = row.get("start_time")
            end = row.get("end_time")
            if not all(
                isinstance(value, (int, float))
                for value in (input_tokens, output_tokens, start, end)
            ):
                continue
            if int(input_tokens) < 3_000 or not 300 <= int(output_tokens) <= 1_500:
                continue
            elapsed = float(end) - float(start)
            if elapsed < 0:
                continue
            yield TrafficRecord(
                timestamp=float(start),
                elapsed=elapsed,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )


def _minute_counts(
    spec: JobSpec,
) -> tuple[dict[int, int], int, int]:
    counts: dict[int, int] = defaultdict(int)
    minimum = math.inf
    maximum = -math.inf
    for record in _iter_records(spec):
        bucket = math.floor(record.timestamp / SELECTION_BUCKET_SECONDS)
        counts[bucket] += 1
        minimum = min(minimum, bucket)
        maximum = max(maximum, bucket)
    if not counts:
        raise ValueError(f"No records matched {spec.slug}")
    return counts, int(minimum), int(maximum)


def _candidate_starts(last_start: int) -> list[int]:
    starts = list(range(0, last_start + 1, SELECTION_STEP_BUCKETS))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _windows_overlap(
    start: float,
    end: float,
    excluded: Sequence[WindowSelection],
) -> bool:
    return any(start < window.end and end > window.start for window in excluded)


def _select_window(
    spec: JobSpec,
    excluded: Sequence[WindowSelection] = (),
) -> WindowSelection:
    sparse_counts, minimum, maximum = _minute_counts(spec)
    counts = [sparse_counts.get(bucket, 0) for bucket in range(minimum, maximum + 1)]
    requested_buckets = math.ceil(spec.duration_minutes * 60 / SELECTION_BUCKET_SECONDS)
    window_buckets = min(requested_buckets, len(counts))
    last_start = len(counts) - window_buckets
    starts = _candidate_starts(last_start)
    totals = [sum(counts[start : start + window_buckets]) for start in starts]
    maximum_total = max(totals)
    minimum_total = min(maximum_total, max(10, math.floor(maximum_total * 0.25)))

    best: WindowSelection | None = None
    for start_index, total in zip(starts, totals, strict=True):
        if total < minimum_total:
            continue
        start = (minimum + start_index) * SELECTION_BUCKET_SECONDS
        end = start + window_buckets * SELECTION_BUCKET_SECONDS
        if _windows_overlap(start, end, excluded):
            continue
        values = counts[start_index : start_index + window_buckets]
        score = _pattern_score(values, spec.pattern)
        candidate = WindowSelection(start, end, score, total)
        if best is None or (candidate.score, -candidate.start) > (
            best.score,
            -best.start,
        ):
            best = candidate
    if best is None and excluded:
        return _select_window(spec)
    if best is None:
        raise ValueError(f"No candidate window found for {spec.slug}")
    return best


def _pattern_score(counts: Sequence[int], pattern: str) -> float:
    mean = statistics.fmean(counts)
    if mean == 0:
        return -math.inf
    variance = statistics.pvariance(counts)
    coefficient_of_variation = math.sqrt(variance) / mean
    peak_to_mean = max(counts) / mean
    p95_to_mean = _percentile(counts, 0.95) / mean
    midpoint = max(1, len(counts) // 2)
    first_mean = statistics.fmean(counts[:midpoint])
    second_mean = statistics.fmean(counts[midpoint:])
    trend = abs(second_mean - first_mean) / mean

    if pattern == "Bursty":
        return 1.5 * p95_to_mean + 0.25 * peak_to_mean + coefficient_of_variation
    if pattern == "Constant":
        return -(coefficient_of_variation + 0.25 * peak_to_mean + trend)
    if pattern == "Poisson":
        fano_factor = variance / mean
        return -(abs(fano_factor - 1) + 0.25 * coefficient_of_variation + trend)
    if pattern == "Diurnal":
        smoothed = _rolling_mean(counts, min(5, len(counts)))
        smooth_range = (max(smoothed) - min(smoothed)) / mean
        roughness = (
            statistics.fmean(
                abs(right - left)
                for left, right in zip(smoothed, smoothed[1:], strict=False)
            )
            / mean
            if len(smoothed) > 1
            else 0.0
        )
        return smooth_range + trend - 0.5 * roughness
    raise ValueError(f"Unsupported traffic pattern: {pattern}")


def _rolling_mean(values: Sequence[int], width: int) -> list[float]:
    if width <= 1:
        return [float(value) for value in values]
    return [
        statistics.fmean(values[index : index + width])
        for index in range(len(values) - width + 1)
    ]


def _percentile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _selected_records(
    spec: JobSpec,
    selection: WindowSelection,
) -> list[dict[str, float | int | None]]:
    rows = [
        record.as_row()
        for record in _iter_records(spec)
        if selection.start <= record.timestamp < selection.end
    ]
    if not rows:
        raise ValueError(f"Selected window for {spec.slug} is empty")
    return rows


def generate_analyses(
    manifest: BenchmarkManifest,
) -> dict[str, WindowSelection]:
    """Generate all job-specific analysis JSON files."""
    manifest.output_directory.mkdir(parents=True, exist_ok=True)
    selections: dict[str, WindowSelection] = {}
    for spec in manifest.jobs:
        if spec.reuse_window_from is not None:
            selection = selections[spec.reuse_window_from]
        else:
            excluded = [
                selections[slug]
                for slug in spec.exclude_windows_from
                if slug in selections
            ]
            selection = _select_window(spec, excluded)
        rows = _selected_records(spec, selection)
        analysis = analyze_trace(
            rows,
            bucket_seconds=1,
            window_start=selection.start,
            window_end=selection.end,
        )
        output = manifest.output_directory / f"{spec.slug}_analysis.json"
        output.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selections[spec.slug] = selection
        LOGGER.info(
            "%s\t%s\t%.3f\t%.3f\t%d\t%.6f",
            spec.slug,
            spec.source_id,
            selection.start,
            selection.end,
            len(rows),
            selection.score,
        )
    return selections


def main() -> None:
    """Run a benchmark trace-analysis manifest."""
    parser = argparse.ArgumentParser(
        description="Generate job traffic analyses from a benchmark manifest."
    )
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_analyses(load_manifest(cast(Path, arguments.manifest)))


if __name__ == "__main__":
    main()
