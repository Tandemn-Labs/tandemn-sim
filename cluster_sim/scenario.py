"""Build deterministic workload slices from benchmark manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from models import BatchJob, BatchProgress, OnlineJob, Request, Scenario

SECOND = 1_000_000_000


@dataclass(frozen=True)
class TokenDistribution:
    """Bounded token-length distribution used by the simulator."""

    kind: str
    minimum: int
    average: int
    maximum: int
    sigma: float = 0.8

    def sample(self, random_source: random.Random) -> int:
        """Sample one token length within the configured bounds."""
        if self.kind == "constant":
            return self.average
        location = _bounded_lognormal_location(
            self.minimum,
            self.average,
            self.maximum,
            self.sigma,
        )
        sampled = round(random_source.lognormvariate(location, self.sigma))
        return max(self.minimum, min(self.maximum, sampled))


def load_scenario(overview_path: Path) -> Scenario:
    """Load every enabled online and batch job from one benchmark overview."""
    settings = yaml.safe_load(overview_path.read_text(encoding="utf-8"))["benchmark"]
    jobs_directory = overview_path.parent / settings["jobs"]
    online_jobs = []
    batch_jobs = []
    loaded_job_ids = set()

    for job_path in sorted(jobs_directory.glob("*.yaml")):
        document = yaml.safe_load(job_path.read_text(encoding="utf-8"))
        schedule = document["schedule"]
        if not schedule["enabled"]:
            continue
        job_id = document["job_id"]
        loaded_job_ids.add(job_id)
        features = document["spec"]["features"]
        workload = document["workload"]
        kind = schedule["kind"]

        if kind == "online":
            benchmark = workload["benchmark"]
            online_jobs.append(
                OnlineJob(
                    name=job_id,
                    store_job_id=job_id,
                    model_name=str(benchmark["model"]),
                    starts_at_ns=_ns(schedule["start_at_offset_minutes"]),
                    stops_at_ns=_ns(schedule["stop_at_offset_minutes"]),
                    workload={
                        "features": features,
                        "phases": benchmark["phases"],
                        "seed": workload["randomSeed"],
                        "task": benchmark["endpoint"]["type"],
                    },
                )
            )
            continue

        dataset = workload["dataset"]
        partitioning = dataset["partitioning"]
        batch_jobs.append(
            BatchJob(
                name=job_id,
                store_job_id=job_id,
                model_name=str(dataset["model"]),
                releases_at_ns=_ns(schedule["start_at_offset_minutes"]),
                deadline_at_ns=_ns(schedule["deadline_at_offset_minutes"]),
                hard_stop_at_ns=_ns(schedule["hard_stop_at_offset_minutes"]),
                workload={
                    "features": features,
                    "input": _distribution(features, "isl"),
                    "output": _distribution(features, "osl"),
                    "record_count": dataset["record_count"],
                    "requests_per_shard": partitioning["requests_per_shard"],
                    "seed": dataset["seed"],
                    "task": dataset["request_format"],
                },
            )
        )

    definition_path = overview_path.parent / settings["definition"]
    for job in _document_jobs(definition_path):
        job_id = str(job["job_id"])
        if job_id in loaded_job_ids or job["kind"] != "batch":
            continue
        spec = job["spec_json"]
        features = spec["job_features"]
        source = job["input_source"]
        schedule = job["schedule"]
        batch_jobs.append(
            BatchJob(
                name=job_id,
                store_job_id=job_id,
                model_name=str(spec["model_id"]),
                releases_at_ns=_ns(schedule["release_at_offset_minutes"]),
                deadline_at_ns=_ns(schedule["deadline_at_offset_minutes"]),
                hard_stop_at_ns=_ns(settings["hard_stop_at_offset_minutes"]),
                workload={
                    "features": features,
                    "input": _distribution(features, "isl"),
                    "output": _distribution(features, "osl"),
                    "record_count": (
                        int(source["shard_count"])
                        * int(source["requests_per_shard"])
                    ),
                    "requests_per_shard": source["requests_per_shard"],
                    "seed": int.from_bytes(
                        hashlib.sha256(job_id.encode()).digest()[:8], "big"
                    ),
                    "task": "chat",
                },
            )
        )

    return Scenario(
        benchmark_name=settings["id"],
        tick_duration_ns=_ns(settings["reoptimization_tick_minutes"]),
        drain_end_ns=_ns(settings["hard_stop_at_offset_minutes"]),
        online_jobs=tuple(online_jobs),
        batch_jobs=tuple(batch_jobs),
    )


def _document_jobs(path: Path) -> tuple[dict[str, object], ...]:
    """Read canonical jobs from fenced JSON benchmark events."""
    jobs = []
    for block in re.findall(
        r"```json\s*(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL
    ):
        jobs.extend(json.loads(block).get("jobs", []))
    return tuple(jobs)


def materialize_online_requests(job: OnlineJob) -> tuple[Request, ...]:
    """Generate one continuous request stream for an online job's lifetime."""
    seed = int(job.workload["seed"])
    arrivals = []
    phase_start_ns = job.starts_at_ns
    for number, phase in enumerate(job.workload["phases"]):
        for offset_ns in _arrivals(phase, _rng(seed, f"arrival:{number}")):
            if phase_start_ns + offset_ns < job.stops_at_ns:
                arrivals.append(phase_start_ns + offset_ns)
        phase_start_ns += _ns(phase["duration"] / 60)

    features = job.workload["features"]
    input_tokens = _distribution(features, "isl")
    output_tokens = _distribution(features, "osl")
    token_rng = _rng(seed, "tokens")
    session_rng = _rng(seed, "sessions")
    prefix_keys = _prefix_keys(job.name, int(features["shared_prefix_length_avg"]))
    requests = []
    request_number = 0
    session_number = 0

    while request_number < len(arrivals):
        turns = 4 if session_rng.random() < features["multi_turn_ratio"] else 1
        session_name = f"{job.name}/session-{session_number:08d}"
        prefix_key = (
            prefix_keys[session_rng.randrange(len(prefix_keys))]
            if session_rng.random() < features["workload_prefix_concentration"]
            else None
        )
        for turn_number in range(turns):
            if request_number == len(arrivals):
                break
            requests.append(
                Request(
                    request_name=f"{job.name}/request-{request_number:08d}",
                    session_name=session_name,
                    turn_number=turn_number,
                    arrives_at_ns=arrivals[request_number],
                    input_tokens=input_tokens.sample(token_rng),
                    output_tokens=output_tokens.sample(token_rng),
                    prefix_key=prefix_key,
                )
            )
            request_number += 1
        session_number += 1
    return tuple(requests)


def requests_in_tick(
    requests: tuple[Request, ...], start_ns: int, end_ns: int
) -> tuple[Request, ...]:
    """Return one half-open logical tick from a continuous request stream."""
    return tuple(
        request for request in requests if start_ns <= request.arrives_at_ns < end_ns
    )


def build_batch_probe(job: BatchJob, progress: BatchProgress) -> tuple[Request, ...]:
    """Build the next variable-length batch shard for one service measurement."""
    count = min(
        int(job.workload["requests_per_shard"]),
        progress.total_requests - progress.completed_requests,
    )
    requests = []
    for number in range(
        progress.next_request_number, progress.next_request_number + count
    ):
        tokens = _rng(int(job.workload["seed"]), f"batch:{number}")
        requests.append(
            Request(
                request_name=f"{job.name}/request-{number:08d}",
                session_name=f"{job.name}/session-{number:08d}",
                turn_number=0,
                arrives_at_ns=0,
                input_tokens=job.workload["input"].sample(tokens),
                output_tokens=job.workload["output"].sample(tokens),
                prefix_key=None,
            )
        )
    return tuple(requests)


def _arrivals(phase: dict[str, object], rng: random.Random):
    """Yield phase-relative arrivals from the existing AIPerf phase fields."""
    rate = float(phase["rate"])
    ramp = float(phase.get("rateRamp", 0))
    duration = float(phase["duration"])
    maximum = _intensity(duration, rate, ramp)
    intensity = 0.0

    while True:
        if phase["type"] == "constant":
            intensity += 1
        elif phase["type"] == "poisson":
            intensity += rng.expovariate(1)
        else:
            smoothness = float(phase["smoothness"])
            intensity += rng.gammavariate(smoothness, 1 / smoothness)
        if intensity >= maximum:
            return
        yield round(_time_for_intensity(intensity, rate, ramp) * SECOND)


def _intensity(seconds: float, rate: float, ramp: float) -> float:
    """Return cumulative offered requests at one phase-relative time."""
    if not ramp:
        return rate * seconds
    if seconds <= ramp:
        return rate * seconds**2 / (2 * ramp)
    return rate * ramp / 2 + rate * (seconds - ramp)


def _time_for_intensity(intensity: float, rate: float, ramp: float) -> float:
    """Invert the phase's cumulative offered-request function."""
    if not ramp:
        return intensity / rate
    ramp_intensity = rate * ramp / 2
    if intensity <= ramp_intensity:
        return math.sqrt(2 * ramp * intensity / rate)
    return ramp + (intensity - ramp_intensity) / rate


def _distribution(features: dict[str, object], prefix: str) -> TokenDistribution:
    """Build the existing bounded token sampler from benchmark feature fields."""
    return TokenDistribution(
        str(features[f"{prefix}_distribution_type"]).lower(),
        int(features[f"{prefix}_token_min"]),
        int(features[f"{prefix}_token_avg"]),
        int(features[f"{prefix}_token_max"]),
    )


@cache
def _bounded_lognormal_location(
    minimum: int,
    average: int,
    maximum: int,
    sigma: float,
) -> float:
    """Find the lognormal location whose clipped mean matches the target."""
    lower = math.log(minimum) - (4 * sigma)
    upper = math.log(maximum) + (4 * sigma)
    for _ in range(60):
        midpoint = (lower + upper) / 2
        mean = _bounded_lognormal_mean(
            midpoint,
            sigma,
            minimum,
            maximum,
        )
        if mean < average:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _bounded_lognormal_mean(
    location: float,
    sigma: float,
    minimum: int,
    maximum: int,
) -> float:
    """Return the analytic mean of a clipped lognormal distribution."""
    minimum_score = (math.log(minimum) - location) / sigma
    maximum_score = (math.log(maximum) - location) / sigma
    unbounded_mean = math.exp(location + (sigma**2 / 2))
    below_minimum = minimum * _normal_cdf(minimum_score)
    within_bounds = unbounded_mean * (
        _normal_cdf(maximum_score - sigma) - _normal_cdf(minimum_score - sigma)
    )
    above_maximum = maximum * (1 - _normal_cdf(maximum_score))
    return below_minimum + within_bounds + above_maximum


def _normal_cdf(value: float) -> float:
    """Return the standard normal cumulative distribution."""
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _prefix_keys(job_name: str, average_length: int) -> tuple[str, ...]:
    """Create six hot-prefix families for cache-hit variation.

    ``workload_prefix_concentration`` selects whether a session is hot; a hot
    session then chooses one stable family so repeated turns can reuse a
    realistic mix of short and long cached context.
    """
    lengths = (
        average_length // 4,
        average_length // 2,
        average_length * 3 // 4,
        average_length,
        average_length * 5 // 4,
        average_length * 3 // 2,
    )
    return tuple(f"{job_name}/prefix-{max(1, length)}" for length in lengths)


def _rng(seed: int, name: str) -> random.Random:
    """Derive one stable independent random stream."""
    value = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def _ns(minutes: float | None) -> int:
    """Convert one required benchmark-minute value to nanoseconds."""
    if minutes is None:
        raise ValueError("missing benchmark timing")
    return round(minutes * 60 * SECOND)
