"""Normalize Koi's native debug JSONL into experiment trace events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trace import read_events


def normalize_koi_state(event: dict[str, Any]) -> dict[str, object] | None:
    """Translate one Koi state event while preserving its raw diagnostic payload."""
    if event.get("kind") != "state":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    return {
        "kind": "koi_state",
        "tick": event.get("tick"),
        "observed_at": event.get("ts"),
        "state": payload.get("state"),
        "duration_ms": payload.get("duration_ms"),
        "payload": payload,
    }


def normalize_koi_tick_summary(event: dict[str, Any]) -> dict[str, object] | None:
    """Translate one Koi tick summary while preserving its raw planner facts."""
    if event.get("kind") != "tick_summary":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    return {
        "kind": "koi_tick_summary",
        "tick": event.get("tick"),
        "observed_at": event.get("ts"),
        "state_durations_ms": payload.get("state_durations_ms"),
        "error": payload.get("error"),
        "payload": payload,
    }


def normalize_koi_llm_calls(event: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Translate Koi LLM-call records, retaining provider usage when available."""
    if event.get("kind") != "llm_calls":
        return ()
    payload = event.get("payload")
    calls = payload.get("calls") if isinstance(payload, dict) else None
    if not isinstance(calls, list):
        return ()
    normalized = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        elapsed_sec = call.get("elapsed_sec")
        normalized.append(
            {
                "kind": "koi_llm_call",
                "tick": event.get("tick"),
                "observed_at": event.get("ts"),
                "call_index": call.get("call_index", index),
                "elapsed_ms": elapsed_sec * 1_000 if isinstance(elapsed_sec, (int, float)) else None,
                "prompt_tokens": call.get("prompt_tokens"),
                "completion_tokens": call.get("completion_tokens"),
                "total_tokens": call.get("total_tokens"),
                "error": call.get("error"),
            }
        )
    return tuple(normalized)


def normalize_koi_evidence(event: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Translate S2 evidence rows while preserving Koi's raw V/Y facts."""
    if event.get("kind") != "state":
        return ()
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("state") != "S2_VALIDATE":
        return ()
    rows = payload.get("evidence_rows")
    if not isinstance(rows, list):
        return ()
    return tuple(
        {
            "kind": "koi_evidence",
            "tick": event.get("tick"),
            "observed_at": event.get("ts"),
            "job_id": row.get("job_id"),
            "rank_id": row.get("rank_id"),
            "payload": row,
        }
        for row in rows
        if isinstance(row, dict)
    )


def normalize_koi_rank_predictions(event: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Extract Koi's S2 observed and predicted V/Y into rank-joinable facts."""
    if event.get("kind") != "state":
        return ()
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("state") != "S2_VALIDATE":
        return ()
    rows = payload.get("evidence_rows")
    if not isinstance(rows, list):
        return ()
    return tuple(
        {
            "kind": "koi_rank_prediction",
            "tick": event.get("tick"),
            "observed_at": event.get("ts"),
            "job_id": row.get("job_id"),
            "rank_id": row.get("rank_id"),
            "v_hat": row.get("v_predicted") or row.get("v_hat"),
            "y_hat": row.get("y_predicted") or row.get("y_hat"),
            "v_actual": row.get("v_observed") or row.get("v_actual"),
            "y_actual": row.get("y_observed") or row.get("y_actual"),
        }
        for row in rows
        if isinstance(row, dict)
    )


def normalize_koi_cusum(event: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Translate S2 mechanism diagnostics without changing CUSUM semantics."""
    if event.get("kind") != "state":
        return ()
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("state") != "S2_VALIDATE":
        return ()
    diagnostics = payload.get("mechanism_diagnostics")
    if not isinstance(diagnostics, list):
        return ()
    return tuple(
        {
            "kind": "koi_cusum",
            "tick": event.get("tick"),
            "observed_at": event.get("ts"),
            "job_id": diagnostic.get("job_id"),
            "rank_id": diagnostic.get("rank_id"),
            "mechanism_id": diagnostic.get("mechanism_id"),
            "payload": diagnostic,
        }
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
    )


def normalize_koi_regret(event: dict[str, Any]) -> dict[str, object] | None:
    """Translate S3 slow-state diagnostics containing Koi regret facts."""
    if event.get("kind") != "state":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("state") != "S3_SLOW_UPDATE":
        return None
    slow_state = payload.get("new_slow_state")
    if not isinstance(slow_state, dict):
        return None
    return {
        "kind": "koi_regret",
        "tick": event.get("tick"),
        "observed_at": event.get("ts"),
        "regret_slope": slow_state.get("regret_slope"),
        "payload": slow_state,
    }


def normalize_koi_candidate(event: dict[str, Any]) -> dict[str, object] | None:
    """Translate Koi's S4 candidate plan without replacing its plan payload."""
    if event.get("kind") != "state":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("state") != "S4_AGENTIC_PLAN":
        return None
    candidate_plan = payload.get("candidate_plan")
    if not isinstance(candidate_plan, dict):
        return None
    return {
        "kind": "koi_candidate",
        "tick": event.get("tick"),
        "observed_at": event.get("ts"),
        "payload": candidate_plan,
    }


def normalize_koi_search_candidates(event: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Translate Koi's scored search candidates from its agent trace."""
    if event.get("kind") != "agent_trace":
        return ()
    payload = event.get("payload")
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return ()
    return tuple(
        {
            "kind": "koi_search_candidate",
            "tick": event.get("tick"),
            "observed_at": event.get("ts"),
            "payload": candidate,
        }
        for candidate in events
        if isinstance(candidate, dict) and candidate.get("kind") == "kp_candidate_scored"
    )


def compile_koi_events(path: Path) -> tuple[dict[str, object], ...]:
    """Compile one native Koi event log into common experiment trace events."""
    compiled = []
    for event in read_events(path):
        state = normalize_koi_state(event)
        if state is not None:
            compiled.append(state)
        summary = normalize_koi_tick_summary(event)
        if summary is not None:
            compiled.append(summary)
        compiled.extend(normalize_koi_llm_calls(event))
        compiled.extend(normalize_koi_evidence(event))
        compiled.extend(normalize_koi_rank_predictions(event))
        compiled.extend(normalize_koi_cusum(event))
        regret = normalize_koi_regret(event)
        if regret is not None:
            compiled.append(regret)
        candidate = normalize_koi_candidate(event)
        if candidate is not None:
            compiled.append(candidate)
        compiled.extend(normalize_koi_search_candidates(event))
    return tuple(compiled)
