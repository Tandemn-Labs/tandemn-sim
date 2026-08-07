"""Submit benchmark jobs to Tandemn before their traffic generators start."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol
from urllib.parse import urlsplit

WORKLOAD_DEFAULTS = {
    "isl_token_avg": 4000,
    "isl_token_min": 1,
    "isl_token_max": 8000,
    "isl_distribution_type": "LogNormal",
    "osl_token_avg": 1000,
    "osl_token_min": 1,
    "osl_token_max": 4000,
    "osl_distribution_type": "LogNormal",
    "request_arrival_rate": 1.0,
    "request_arrival_pattern": "Poisson",
    "peak_to_mean_ratio": 2,
    "workload_prefix_concentration": 0,
    "multi_turn_ratio": 0.5,
    "shared_prefix_length_avg": 500,
    "sliding_window_attention": 0,
    "is_session_affinity": True,
    "total_token_budget": 1_000_000,
    "priority_class": "STANDARD",
    "max_concurrent_streaming": 10,
    "preemption_policy": "lifo",
    "router_policy": "kv_router",
}
TRAFFIC_ALIASES = {
    "requestRate": "request_arrival_rate",
    "concurrency": "max_concurrent_streaming",
}
TRAFFIC_MODE_KEY = "_traffic_mode"
TRAFFIC_MODE_REQUEST_RATE = "request_rate"
TRAFFIC_MODE_CONCURRENCY = "concurrency"
REQUEST_RATE_KEYS = frozenset({"requestRate", "request_arrival_rate"})
CONCURRENCY_KEYS = frozenset({"concurrency", "max_concurrent_streaming"})
USER_FEATURE_KEYS = frozenset(
    {
        *WORKLOAD_DEFAULTS,
        *TRAFFIC_ALIASES,
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
    }
)
DIST_TYPES = {"Constant", "Uniform", "Normal", "LogNormal", "Exponential"}
ARRIVAL_PATTERNS = {"Poisson", "Constant", "Bursty", "Diurnal"}
PRIORITY_CLASSES = {"HIGH", "STANDARD", "LOW"}
PREEMPTION_POLICIES = {"lifo", "fifo"}
ROUTER_POLICIES = {"kv_router"}


class SubmissionError(RuntimeError):
    """Report one job submission or response-contract failure."""


@dataclass(frozen=True)
class SubmissionRequest:
    """User-owned Tandemn job inputs loaded from a canonical job spec."""

    user_id: str
    kind: str
    spec_json: Mapping[str, object]
    job_id: str | None = None


@dataclass(frozen=True)
class SubmissionResult:
    """Runtime identity and routing returned by Tandemn."""

    tandemn_job_id: str
    endpoint_url: str | None = None

    def __post_init__(self) -> None:
        """Enforce the scheduler response contract at the boundary."""
        if not self.tandemn_job_id:
            raise SubmissionError("Tandemn submission returned no job_id")
        if self.endpoint_url is not None:
            validate_endpoint_url(self.endpoint_url)


class JobSubmitter(Protocol):
    """Transport boundary between the benchmark controller and Tandemn."""

    def submit(self, request: SubmissionRequest) -> SubmissionResult:
        """Submit one job and return its Tandemn runtime identity."""

    def poll_endpoint(self, tandemn_job_id: str) -> str | None:
        """Return a deployed endpoint, or None while deployment is pending."""


class TandemnStoreSubmitter:
    """Submit through tandemn-system-data's production JobStore path."""

    def submit(self, request: SubmissionRequest) -> SubmissionResult:
        """Normalize, store, and return one scheduler submission result."""
        try:
            clients = import_module("tandemn_system_data.clients")
            models = import_module("tandemn_system_data.models")
        except ImportError as error:
            raise SubmissionError(
                "tandemn-system-data is required to submit benchmark jobs"
            ) from error

        if request.kind == "online":
            tandemn_kind = models.JobKind.ONLINE
        elif request.kind == "batch":
            tandemn_kind = models.JobKind.BATCH
        else:
            raise SubmissionError(f"unsupported Tandemn job kind: {request.kind}")

        try:
            spec_json = normalize_job_spec(dict(request.spec_json), request.kind)
            job_values = {
                "user_id": request.user_id,
                "kind": tandemn_kind,
                "spec_json": spec_json,
            }
            if request.job_id is not None:
                job_values["job_id"] = request.job_id
            job = models.Job(**job_values)
            response = clients.JobStore(clients.PostgresClient()).submit(job)
        except (OSError, ValueError) as error:
            raise SubmissionError(str(error)) from error

        endpoint_url = response_value(response, "endpoint_url", required=False)
        tandemn_job_id = response_value(response, "job_id", required=False)
        if tandemn_job_id is None:
            tandemn_job_id = str(job.job_id)
        return SubmissionResult(
            tandemn_job_id=tandemn_job_id,
            endpoint_url=endpoint_url,
        )

    def poll_endpoint(self, tandemn_job_id: str) -> str | None:
        """Read a job's endpoint from Tandemn Store while it deploys.

        The store lookup is intentionally kept at this boundary because the
        final Tandemn response shape is still being defined. The adapter
        accepts the current/future common JobStore method names and response
        shapes without exposing them to the benchmark controller.
        """
        try:
            clients = import_module("tandemn_system_data.clients")
        except ImportError as error:
            raise SubmissionError(
                "tandemn-system-data is required to poll benchmark jobs"
            ) from error

        store = clients.JobStore(clients.PostgresClient())
        lookup = getattr(store, "get_job", None) or getattr(store, "get", None)
        if lookup is None:
            raise SubmissionError(
                "Tandemn JobStore does not yet expose a job lookup method"
            )
        try:
            response = lookup(tandemn_job_id)
        except (OSError, ValueError) as error:
            raise SubmissionError(str(error)) from error

        endpoint_url = response_value(response, "endpoint_url", required=False)
        if endpoint_url is None:
            endpoint_url = response_value(response, "url", required=False)
        if endpoint_url is not None:
            validate_endpoint_url(endpoint_url)
        return endpoint_url


def response_value(
    response: object,
    key: str,
    *,
    required: bool = True,
) -> str | None:
    """Read a response value from either a mapping or response object."""
    value: object
    if isinstance(response, Mapping):
        value = response.get(key)
    else:
        value = getattr(response, key, None)
    if isinstance(value, str) and value:
        return value
    if required:
        raise SubmissionError(f"Tandemn submission returned no {key}")
    return None


def validate_endpoint_url(endpoint_url: str) -> None:
    """Require a complete HTTP endpoint returned by Tandemn."""
    parsed = urlsplit(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SubmissionError(
            f"Tandemn returned an invalid endpoint_url: {endpoint_url!r}"
        )


def normalize_job_spec(spec: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return canonical spec_json using the production submitter contract."""
    model_id = spec.get("model_id")
    if not model_id:
        raise SubmissionError("submission spec_json must include model_id")

    provided_request_rate = has_any_user_key(spec, REQUEST_RATE_KEYS)
    provided_concurrency = has_any_user_key(spec, CONCURRENCY_KEYS)
    if provided_request_rate and provided_concurrency:
        raise SubmissionError(
            "submission spec_json must provide request rate or concurrency, not both"
        )

    features = dict(WORKLOAD_DEFAULTS)
    if isinstance(spec.get("job_features"), dict):
        features.update(canonical_user_features(spec["job_features"]))
    features.update(canonical_user_features(spec))
    features["model_id"] = model_id
    features["type"] = kind
    canonicalize_runtime_policy(features)

    if kind == "online":
        for key in ("target_p99_ttft_ms", "target_p99_tpot_ms"):
            if features.get(key) is None:
                raise SubmissionError(f"submission spec_json missing {key}")
        features.setdefault("deadline_hrs", 0)
        derive_online_traffic(features, provided_request_rate, provided_concurrency)
    elif kind == "batch":
        deadline = features.get("deadline_hrs")
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, int | float)
            or deadline <= 0
        ):
            raise SubmissionError("submission deadline_hrs must be a positive number")
    else:
        raise SubmissionError(f"unsupported submission kind: {kind}")

    features["pd_ratio"] = positive_float(
        features,
        "isl_token_avg",
    ) / positive_float(features, "osl_token_avg")
    validate_features(features)
    out = dict(spec)
    out["model_id"] = model_id
    out["job_features"] = features
    return out


def canonical_user_features(source: Mapping[str, Any]) -> dict[str, Any]:
    """Select and canonicalize user-owned job features."""
    return {
        TRAFFIC_ALIASES.get(key, key): value
        for key, value in source.items()
        if key in USER_FEATURE_KEYS
    }


def has_any_user_key(spec: Mapping[str, Any], keys: frozenset[str]) -> bool:
    """Return whether a traffic intent was explicitly supplied."""
    if any(key in spec for key in keys):
        return True
    nested = spec.get("job_features")
    return isinstance(nested, dict) and any(key in nested for key in keys)


def derive_online_traffic(
    features: dict[str, Any],
    provided_request_rate: bool,
    provided_concurrency: bool,
) -> None:
    """Preserve the user's traffic mode and derive Koi's companion field."""
    if not provided_request_rate and not provided_concurrency:
        features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_REQUEST_RATE
        return
    service_time_s = target_service_time_seconds(features)
    if provided_request_rate:
        features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_REQUEST_RATE
        request_rate = positive_float(features, "request_arrival_rate")
        features["request_arrival_rate"] = request_rate
        features["max_concurrent_streaming"] = max(
            1,
            math.ceil(request_rate * service_time_s),
        )
        return
    features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_CONCURRENCY
    concurrency = max(
        1,
        math.ceil(positive_float(features, "max_concurrent_streaming")),
    )
    features["max_concurrent_streaming"] = concurrency
    features["request_arrival_rate"] = concurrency / service_time_s


def target_service_time_seconds(features: Mapping[str, Any]) -> float:
    """Estimate target request service time from submitted SLOs."""
    ttft_ms = positive_float(features, "target_p99_ttft_ms")
    tpot_ms = positive_float(features, "target_p99_tpot_ms")
    osl = positive_float(features, "osl_token_avg")
    return (ttft_ms + tpot_ms * osl) / 1000.0


def positive_float(features: Mapping[str, Any], key: str) -> float:
    """Read one positive numeric feature."""
    value = features.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SubmissionError(f"submission field {key} must be a positive number")
    return float(value)


def canonicalize_runtime_policy(features: dict[str, Any]) -> None:
    """Replace unsupported policies with production defaults."""
    force_allowed(features, "preemption_policy", PREEMPTION_POLICIES, "lifo")
    force_allowed(features, "router_policy", ROUTER_POLICIES, "kv_router")


def force_allowed(
    features: dict[str, Any],
    key: str,
    allowed: set[str],
    default: str,
) -> None:
    """Use a supported runtime policy value."""
    if features.get(key) not in allowed:
        features[key] = default


def validate_features(features: Mapping[str, Any]) -> None:
    """Validate the production submitter's bounded feature enums and numbers."""
    for key in (
        "isl_token_avg",
        "isl_token_min",
        "isl_token_max",
        "osl_token_avg",
        "osl_token_min",
        "osl_token_max",
        "request_arrival_rate",
        "max_concurrent_streaming",
    ):
        positive_float(features, key)
    for key, allowed in (
        ("isl_distribution_type", DIST_TYPES),
        ("osl_distribution_type", DIST_TYPES),
        ("request_arrival_pattern", ARRIVAL_PATTERNS),
        ("priority_class", PRIORITY_CLASSES),
    ):
        if features[key] not in allowed:
            raise SubmissionError(
                f"submission field {key} must be one of {sorted(allowed)}"
            )
