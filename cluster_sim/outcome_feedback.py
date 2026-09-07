"""Store-visible placement outcome feedback for learning planners."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class GpuMetricStore(Protocol):
    """Read persisted simulator metrics for one canonical rank window."""

    def rows_for_rank_window(
        self,
        user_id: str,
        job_id: str,
        rank_id: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[object]: ...


class EventLog(Protocol):
    """Read terminal Store events in event-ID order."""

    def read_after(
        self,
        event_id: str | None,
        *,
        types: set[str],
    ) -> Sequence[object]: ...


@dataclass(frozen=True)
class PlacementFingerprint:
    """Identify one job-specific physical serving configuration."""

    job_id: str
    model_id: str
    environment: str
    instance_type: str | None
    gpu_type: str
    tp: int
    pp: int
    dp: int
    n_replicas: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-native form exposed to evolved policies."""
        count = self.tp * self.pp
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "environment": self.environment,
            "instance_type": self.instance_type,
            "gpu_type": self.gpu_type,
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "gpu_count": count,
            "count": count,
            "n_replicas": self.n_replicas if self.n_replicas is not None else self.dp,
        }


@dataclass(frozen=True)
class Outcome:
    """One observed Store outcome for a planned placement fingerprint."""

    fingerprint: PlacementFingerprint
    p99_ttft_ms: float | None
    p99_tpot_ms: float | None
    throughput_token_per_sec: float | None
    batch_deadline_met: bool | None
    reward: float
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-native measured results without Store capabilities."""
        return {
            **self.fingerprint.as_dict(),
            "p99_ttft_ms": self.p99_ttft_ms,
            "p99_tpot_ms": self.p99_tpot_ms,
            "throughput_token_per_sec": self.throughput_token_per_sec,
            "batch_deadline_met": self.batch_deadline_met,
            "reward": self.reward,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class _PendingOutcome:
    """Keep one planned action open until Store exposes its outcome."""

    fingerprint: PlacementFingerprint
    rank_id: str
    window_start: datetime
    is_batch: bool
    required_tps: float | None
    ttft_target_ms: float | None
    tpot_target_ms: float | None


class OutcomeFeedback:
    """Read delayed Store outcomes and retain the latest evidence per placement."""

    def __init__(self, user_id: str) -> None:
        """Initialize an unbound feedback ledger for one Store user."""
        self.user_id = user_id
        self.metric_store: GpuMetricStore | None = None
        self.event_log: EventLog | None = None
        self._event_cursor: str | None = None
        self._pending: dict[str, _PendingOutcome] = {}
        self._outcomes: dict[PlacementFingerprint, Outcome] = {}

    def bind(self, metric_store: GpuMetricStore, event_log: EventLog) -> None:
        """Attach the Store readers used at the start of each planning tick."""
        self.metric_store = metric_store
        self.event_log = event_log

    def record(
        self,
        *,
        fingerprint: PlacementFingerprint,
        rank_id: str,
        window_start: datetime,
        is_batch: bool,
        required_tps: float | None,
        ttft_target_ms: float | None,
        tpot_target_ms: float | None,
    ) -> None:
        """Track one applied-intent placement until Store exposes evidence."""
        self._pending[fingerprint.job_id] = _PendingOutcome(
            fingerprint=fingerprint,
            rank_id=rank_id,
            window_start=window_start,
            is_batch=is_batch,
            required_tps=required_tps,
            ttft_target_ms=ttft_target_ms,
            tpot_target_ms=tpot_target_ms,
        )

    def has_pending_batch(self, job_id: str) -> bool:
        """Return whether a batch action awaits its terminal deadline result."""
        pending = self._pending.get(job_id)
        return pending is not None and pending.is_batch

    def observe(self, logical_time: datetime) -> tuple[Outcome, ...]:
        """Consume newly available Store metrics and terminal batch events."""
        if self.metric_store is None or self.event_log is None:
            return ()
        observed = []
        for event in self.event_log.read_after(
            self._event_cursor,
            types={"batch.outcome"},
        ):
            self._event_cursor = _event_id(event)
            if _event_user_id(event) != self.user_id:
                continue
            job_id = _event_job_id(event)
            if job_id is None:
                continue
            pending = self._pending.get(job_id)
            if pending is None or not pending.is_batch:
                continue
            outcome = self._read_outcome(
                pending,
                logical_time,
                batch_deadline_met=_event_deadline_met(event),
            )
            if outcome is None:
                continue
            self._pending.pop(job_id, None)
            self._outcomes[outcome.fingerprint] = outcome
            observed.append(outcome)

        for job_id, pending in tuple(self._pending.items()):
            if pending.is_batch:
                continue
            outcome = self._read_outcome(pending, logical_time)
            if outcome is None:
                continue
            self._pending.pop(job_id, None)
            self._outcomes[outcome.fingerprint] = outcome
            observed.append(outcome)
        return tuple(observed)

    def for_fingerprint(
        self,
        fingerprint: PlacementFingerprint,
    ) -> dict[str, object] | None:
        """Return the latest public outcome for one candidate configuration."""
        outcome = self._outcomes.get(fingerprint)
        return None if outcome is None else outcome.as_dict()

    def outcomes_for_job_ids(
        self,
        job_ids: Sequence[str],
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic JSON evidence for the current decision jobs."""
        requested = set(job_ids)
        outcomes = [
            outcome.as_dict()
            for fingerprint, outcome in self._outcomes.items()
            if fingerprint.job_id in requested
        ]
        return tuple(
            sorted(
                outcomes,
                key=lambda outcome: (
                    str(outcome["job_id"]),
                    str(outcome["environment"]),
                    str(outcome["instance_type"]),
                    int(outcome["tp"]),
                    int(outcome["pp"]),
                    int(outcome["dp"]),
                ),
            )
        )

    def _read_outcome(
        self,
        pending: _PendingOutcome,
        logical_time: datetime,
        *,
        batch_deadline_met: bool | None = None,
    ) -> Outcome | None:
        """Aggregate one rank window and score it with the shared reward contract."""
        assert self.metric_store is not None
        metrics = self.metric_store.rows_for_rank_window(
            self.user_id,
            pending.fingerprint.job_id,
            pending.rank_id,
            pending.window_start,
            logical_time,
        )
        if not metrics and batch_deadline_met is None:
            return None
        ttft = worst_metric(metrics, "p99_ttft_ms")
        tpot = worst_metric(metrics, "p99_tpot_ms")
        throughput = mean_rank_throughput(metrics)
        if pending.is_batch:
            assert batch_deadline_met is not None
            reward = 0.4 * throughput_score(throughput, pending.required_tps) + 0.6 * (
                1.0 if batch_deadline_met else -1.0
            )
        else:
            reward = (
                0.4 * latency_score(ttft, pending.ttft_target_ms)
                + 0.4 * latency_score(tpot, pending.tpot_target_ms)
                + 0.2 * throughput_score(throughput, pending.required_tps)
            )
        return Outcome(
            fingerprint=pending.fingerprint,
            p99_ttft_ms=ttft,
            p99_tpot_ms=tpot,
            throughput_token_per_sec=throughput,
            batch_deadline_met=batch_deadline_met,
            reward=reward,
            sample_count=len(metrics),
        )


def fingerprint(
    *,
    job_id: str,
    model_id: str,
    environment: str,
    instance_type: str | None,
    gpu_type: str,
    tp: int,
    pp: int,
    dp: int,
    n_replicas: int | None = None,
) -> PlacementFingerprint:
    """Build one canonical placement identity from scheduler-visible fields."""
    return PlacementFingerprint(
        job_id=job_id,
        model_id=model_id,
        environment=environment,
        instance_type=instance_type,
        gpu_type=gpu_type,
        tp=tp,
        pp=pp,
        dp=dp,
        n_replicas=n_replicas,
    )


def latency_score(observed: float | None, target: float | None) -> float:
    """Score lower latency against one p99 target."""
    if observed is None or target is None or target <= 0:
        return -1.0
    return max(-1.0, min(1.0, 1 - observed / target))


def throughput_score(observed: float | None, required: float | None) -> float:
    """Score delivered output throughput against one required rate."""
    if observed is None or required is None or required <= 0:
        return -1.0
    return max(-1.0, min(1.0, observed / required - 1))


def worst_metric(metrics: Sequence[object], field: str) -> float | None:
    """Return the worst finite rank value across one Store telemetry window."""
    values = [_finite_nonnegative(getattr(metric, field, None)) for metric in metrics]
    present = [value for value in values if value is not None]
    return None if not present else max(present)


def mean_rank_throughput(metrics: Sequence[object]) -> float | None:
    """Average rank throughput without double counting GPU-level metric rows."""
    by_timestamp: dict[object, float] = {}
    for metric in metrics:
        value = _finite_nonnegative(getattr(metric, "throughput_token_per_sec", None))
        if value is None:
            continue
        timestamp = getattr(metric, "ts", None)
        by_timestamp[timestamp] = max(by_timestamp.get(timestamp, 0.0), value)
    return None if not by_timestamp else sum(by_timestamp.values()) / len(by_timestamp)


def _finite_nonnegative(value: object) -> float | None:
    """Read one finite nonnegative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _event_id(event: object) -> str | None:
    """Read one optional event ID from Store's event model or a test double."""
    value = getattr(event, "event_id", None)
    return value if isinstance(value, str) else None


def _event_user_id(event: object) -> str | None:
    """Read one optional Store user ID from an event."""
    value = getattr(event, "user_id", None)
    return value if isinstance(value, str) else None


def _event_job_id(event: object) -> str | None:
    """Read one optional Store job ID from an event."""
    value = getattr(event, "job_id", None)
    return value if isinstance(value, str) else None


def _event_deadline_met(event: object) -> bool:
    """Read the terminal batch deadline outcome from one Store event."""
    payload = getattr(event, "payload_json", None)
    return isinstance(payload, Mapping) and payload.get("deadline_met") is True
