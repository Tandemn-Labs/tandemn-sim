"""Normalize and analyze a request-traffic trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TIMESTAMP_COLUMN = "timestamp_seconds"
ELAPSED_COLUMN = "elapsed_seconds"
INPUT_TOKENS_COLUMN = "input_tokens"
OUTPUT_TOKENS_COLUMN = "output_tokens"

COLUMN_ALIASES = {
    TIMESTAMP_COLUMN: (
        "timestamp_seconds",
        "timestamp",
        "start_time",
        "arrival_timestamp",
        "arrival_time",
        "request_timestamp",
        "request_time",
        "request_start_time",
        "invocation_timestamp",
    ),
    ELAPSED_COLUMN: (
        "elapsed_seconds",
        "elapsed_time",
        "elapsed",
        "duration_seconds",
        "duration",
        "latency_seconds",
    ),
    INPUT_TOKENS_COLUMN: (
        "input_tokens",
        "input",
        "input_length",
        "request_tokens",
        "prompt_tokens",
        "context_tokens",
    ),
    OUTPUT_TOKENS_COLUMN: (
        "output_tokens",
        "output",
        "output_length",
        "response_tokens",
        "completion_tokens",
        "generated_tokens",
    ),
}


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Load records, normalize column names, and discard unused columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {path}")

    suffix = path.suffix.lower()
    rows = []
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=1):
                rows.append(_normalize_row(dict(row), row_number))
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Line {line_number} contains invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} must contain a JSON object")
                rows.append(_normalize_row(row, line_number))
    else:
        raise ValueError("Input must be a .csv, .jsonl, or .ndjson file")

    if not rows:
        raise ValueError("Trace is empty")
    return rows


def analyze_trace(
    rows: Sequence[dict[str, Any]],
    bucket_seconds: int = 1,
    window_start: float | None = None,
    window_end: float | None = None,
) -> dict[str, Any]:
    """Calculate request-rate, interarrival, and token metrics."""
    if not rows:
        raise ValueError("Trace is empty")
    if bucket_seconds <= 0:
        raise ValueError("Bucket width must be positive")

    parsed = [
        (
            _parse_timestamp_seconds(row[TIMESTAMP_COLUMN]),
            _nonnegative_int(row[INPUT_TOKENS_COLUMN], INPUT_TOKENS_COLUMN),
            _nonnegative_int(row[OUTPUT_TOKENS_COLUMN], OUTPUT_TOKENS_COLUMN),
        )
        for row in rows
    ]
    parsed.sort(key=lambda item: item[0])
    timestamps = [timestamp for timestamp, _, _ in parsed]
    start = timestamps[0] if window_start is None else window_start
    end = window_end
    if end is not None and end <= start:
        raise ValueError("Window end must be later than window start")
    if timestamps[0] < start or (end is not None and timestamps[-1] >= end):
        raise ValueError("Trace timestamps must fall inside the analysis window")
    counts = _bucket_counts(timestamps, start, bucket_seconds, end)
    rates = [count / bucket_seconds for count in counts]
    interarrival_ms = [
        (right - left) * 1_000
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ]

    result = {
        "request_rate": {
            "bucket_seconds": bucket_seconds,
            **_summary(rates, "rps"),
            "coefficient_of_variation": _coefficient_of_variation(rates),
            "peak_to_mean_ratio": _ratio(max(rates), statistics.fmean(rates)),
        },
        "interarrival": {
            **_summary(interarrival_ms, "ms"),
            "coefficient_of_variation": _coefficient_of_variation(interarrival_ms),
        },
        "tokens": {
            "input": _summary([float(row[1]) for row in parsed], "tokens"),
            "output": _summary([float(row[2]) for row in parsed], "tokens"),
        },
    }
    _validate_finite(result)
    _round_floats(result)
    return result


def _bucket_counts(
    timestamps: Sequence[float],
    start: float,
    bucket_seconds: int,
    end: float | None = None,
) -> list[int]:
    if end is None:
        bucket_count = math.floor((timestamps[-1] - start) / bucket_seconds) + 1
    else:
        bucket_count = math.ceil((end - start) / bucket_seconds)
    counts = [0] * bucket_count
    for timestamp in timestamps:
        offset = timestamp - start
        counts[math.floor(offset / bucket_seconds)] += 1
    return counts


def _summary(values: Sequence[float], unit: str) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            f"mean_{unit}": None,
            f"median_{unit}": None,
            f"standard_deviation_{unit}": None,
            f"minimum_{unit}": None,
            f"p50_{unit}": None,
            f"p90_{unit}": None,
            f"p95_{unit}": None,
            f"p99_{unit}": None,
            f"maximum_{unit}": None,
        }
    return {
        "count": len(values),
        f"mean_{unit}": statistics.fmean(values),
        f"median_{unit}": statistics.median(values),
        f"standard_deviation_{unit}": (
            statistics.pstdev(values) if len(values) > 1 else 0.0
        ),
        f"minimum_{unit}": min(values),
        f"p50_{unit}": _percentile(values, 0.50),
        f"p90_{unit}": _percentile(values, 0.90),
        f"p95_{unit}": _percentile(values, 0.95),
        f"p99_{unit}": _percentile(values, 0.99),
        f"maximum_{unit}": max(values),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _normalize_row(
    row: dict[str, Any],
    row_number: int,
) -> dict[str, Any]:
    normalized_source = {
        _normalize_column_name(name): value for name, value in row.items()
    }
    normalized_row = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_source:
                normalized_row[canonical_name] = normalized_source[alias]
                break
        else:
            if canonical_name == ELAPSED_COLUMN:
                normalized_row[canonical_name] = _derive_elapsed(normalized_source)
                continue
            raise ValueError(
                f"Row {row_number} is missing required column {canonical_name!r}"
            )
    return normalized_row


def _normalize_column_name(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def _derive_elapsed(row: dict[str, Any]) -> float | None:
    start = row.get("start_time")
    end = row.get("end_time")
    if start is None or end is None:
        return None
    elapsed = _parse_timestamp_seconds(end) - _parse_timestamp_seconds(start)
    if elapsed < 0:
        raise ValueError("end_time must not be earlier than start_time")
    return elapsed


def _nonnegative_float(value: Any, column: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{column} must contain finite nonnegative numbers")
    return parsed


def _nonnegative_int(value: Any, column: str) -> int:
    parsed = _nonnegative_float(value, column)
    if not parsed.is_integer():
        raise ValueError(f"{column} must contain whole numbers")
    return int(parsed)


def _parse_timestamp_seconds(value: Any) -> float:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).timestamp()
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(
            f"{TIMESTAMP_COLUMN} must contain valid nonnegative timestamps"
        )
    return numeric


def _validate_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Analysis produced NaN or Infinity")
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite(child)
    elif isinstance(value, list):
        for child in value:
            _validate_finite(child)


def _round_floats(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, float):
                value[key] = round(child, 3)
            else:
                _round_floats(child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, float):
                value[index] = round(child, 3)
            else:
                _round_floats(child)


def main() -> None:
    """Run the trace analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze a normalized CSV or JSON Lines traffic trace."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bucket-seconds", type=int, default=1)
    args = parser.parse_args()

    try:
        result = analyze_trace(
            load_trace(args.input),
            bucket_seconds=args.bucket_seconds,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
