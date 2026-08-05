"""AIC client boundary and NVIDIA AIC SDK implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Protocol

from data_generation.scheduling import OptimizationGoal

_AIC_CANDIDATE_LIMIT = 1000


class AicEvaluationError(RuntimeError):
    """Report an AIC dependency or evaluation failure."""


class AicUnavailableError(ValueError):
    """Report that the NVIDIA AIC package cannot run in this environment."""


@dataclass(frozen=True)
class AicRequest:
    """Inputs for one homogeneous AIC recommendation."""

    model: str
    system: str
    available_devices: int
    backend: str
    backend_version: str | None
    database_mode: str
    serving_mode: str
    input_tokens: int
    output_tokens: int
    target_ttft_ms: float
    target_tpot_ms: float
    optimization_goal: OptimizationGoal
    target_concurrency: float | None = None
    target_request_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the request as a JSON object."""
        return asdict(self)


@dataclass(frozen=True)
class AicResult:
    """Normalized rank-one aggregate configuration returned by AIC."""

    devices: int
    replicas: int
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    throughput_tokens_per_second: float
    configuration: Mapping[str, object]


class AicClient(Protocol):
    """Evaluate one job against one homogeneous hardware system."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Return AIC's rank-one aggregate configuration."""


class NvidiaAicClient:
    """Call the public AIC recommendation API and retain aggregate serving."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Run AIC load matching for one hardware system."""
        if request.serving_mode != "agg":
            raise AicEvaluationError("Only aggregate serving is supported here")
        try:
            from aiconfigurator.cli.api import cli_recommend
            from aiconfigurator.sdk.picking import pick_optimization_type
        except ImportError as error:
            raise AicUnavailableError(
                "AIC is not installed; run `uv sync --group dynamo`"
            ) from error

        previous_logging_threshold = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = cli_recommend(
                    model_path=request.model,
                    system=request.system,
                    target_request_rate=request.target_request_rate,
                    target_concurrency=request.target_concurrency,
                    backend=request.backend,
                    backend_version=request.backend_version,
                    database_mode=request.database_mode,
                    isl=request.input_tokens,
                    osl=request.output_tokens,
                    ttft=request.target_ttft_ms,
                    tpot=request.target_tpot_ms,
                    strict_sla=True,
                    top_n=_AIC_CANDIDATE_LIMIT,
                )
        except SystemExit as error:
            raise AicEvaluationError(str(error)) from error
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        finally:
            logging.disable(previous_logging_threshold)

        frame = result.best_configs.get("agg")
        if frame is None or frame.empty:
            raise AicEvaluationError("AIC returned no aggregate configuration")
        try:
            fitting_frame = frame[
                frame["total_gpus_needed"] <= request.available_devices
            ]
            objective_input = fitting_frame if not fitting_frame.empty else frame
            if request.optimization_goal == OptimizationGoal.LATENCY:
                frame = objective_input.sort_values(
                    by=["ttft", "tpot", "total_gpus_needed"],
                    ascending=[True, True, True],
                ).head(1)
            else:
                picked = pick_optimization_type(
                    pareto_df=objective_input,
                    optimization_type=request.optimization_goal.value,
                    total_gpus=request.available_devices,
                    serving_mode=request.serving_mode,
                    top_n=1,
                )
                frame = picked["best_config_df"]
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        if frame is None or frame.empty:
            raise AicEvaluationError(
                "AIC returned no configuration for the optimization goal"
            )
        row = frame.iloc[0]
        configuration = _json_value(row.to_dict())
        if not isinstance(configuration, Mapping):
            raise AicEvaluationError("AIC returned an invalid configuration")

        replicas = _positive_int(row.get("replicas_needed", 1), "replicas")
        devices = _positive_int(
            row.get("total_gpus_needed", row.get("num_total_gpus")),
            "total_gpus_needed",
        )
        per_replica_throughput = _positive_float(
            row.get("tokens/s"),
            "tokens/s",
        )
        ttft_ms = _positive_float(row.get("ttft"), "ttft")
        tpot_ms = _positive_float(row.get("tpot"), "tpot")
        request_latency_ms = float(
            row.get(
                "request_latency",
                ttft_ms + tpot_ms * max(request.output_tokens - 1, 0),
            )
        )
        return AicResult(
            devices=devices,
            replicas=replicas,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            request_latency_ms=request_latency_ms,
            throughput_tokens_per_second=(per_replica_throughput * replicas),
            configuration=configuration,
        )


def _json_value(value: object) -> object:
    """Convert pandas and NumPy scalar containers to JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return item_method()
    return value


def _positive_int(value: object, name: str) -> int:
    converted = _json_value(value)
    if isinstance(converted, bool) or not isinstance(converted, int | float):
        raise AicEvaluationError(f"AIC returned invalid {name}")
    number = int(converted)
    if number <= 0:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number


def _positive_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AicEvaluationError(f"AIC returned invalid {name}") from error
    if number <= 0:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number
