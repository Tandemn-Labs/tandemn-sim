"""Canonical Store rank identity helpers."""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from tandemn_system_data.ids import new_rank_id


def rank_id(known: Mapping[str, object]) -> str:
    """Preserve an existing rank ID or allocate a canonical Store ID."""
    return str(known["rank_id"]) if known.get("rank_id") else new_rank_id()


def active_rank_id(running_job: Mapping[str, object]) -> str | None:
    """Return the sole active rank identity eligible for a single-rank swap."""
    ranks = running_job.get("active_ranks")
    if not isinstance(ranks, Sequence) or isinstance(ranks, (str, bytes)):
        return None
    if len(ranks) != 1:
        return None
    rank = ranks[0]
    if not isinstance(rank, Mapping):
        return None
    value = rank.get("rank_id")
    if not isinstance(value, str) or not value:
        return None
    return value


def reconcile_swap_rank_ids(
    running_job: Mapping[str, object],
    action: MutableMapping[str, object],
) -> bool:
    """Assign swap ladder IDs and report whether the deployment is unchanged.

    An active ID is reusable when the normalized deployment is identical after
    excluding DP. A DP-only change therefore keeps its ID, while any other
    deployment change receives a fresh identity.
    """
    active_values = running_job.get("active_ranks")
    ladder = action.get("ladder")
    if not isinstance(active_values, Sequence) or isinstance(
        active_values, (str, bytes)
    ):
        raise ValueError("running job active_ranks must be a sequence")
    if not isinstance(ladder, list):
        raise ValueError("swap action ladder must be a list")

    active = []
    for value in active_values:
        if not isinstance(value, Mapping):
            raise ValueError("active rank must be an object")
        rank_name = value.get("rank_id")
        replicas = value.get("n_replicas")
        if not isinstance(rank_name, str) or not rank_name:
            raise ValueError("active rank is missing rank_id")
        if type(replicas) is not int or replicas <= 0:
            raise ValueError("active rank is missing positive n_replicas")
        active.append(
            {
                "rank_id": rank_name,
                "replicas": replicas,
                "signature": _active_signature(value),
            }
        )

    desired = []
    for value in ladder:
        if not isinstance(value, MutableMapping):
            raise ValueError("swap ladder rank must be a mutable object")
        replicas = value.get("n_replicas", value.get("chains", 1))
        if type(replicas) is not int or replicas <= 0:
            raise ValueError("swap ladder rank is missing positive n_replicas")
        desired.append(
            {
                "rank": value,
                "replicas": replicas,
                "signature": _desired_signature(running_job, action, value),
                "match": None,
                "exact": False,
            }
        )

    unmatched = set(range(len(active)))
    for exact in (True, False):
        for proposed in desired:
            if proposed["match"] is not None:
                continue
            for index in sorted(unmatched):
                current = active[index]
                if current["signature"] != proposed["signature"]:
                    continue
                replicas_match = current["replicas"] == proposed["replicas"]
                if replicas_match is not exact:
                    continue
                proposed["match"] = index
                proposed["exact"] = exact
                proposed["rank"]["rank_id"] = current["rank_id"]
                unmatched.remove(index)
                break

    for proposed in desired:
        if proposed["match"] is None:
            proposed["rank"]["rank_id"] = new_rank_id()

    return (
        len(active) == len(desired)
        and not unmatched
        and all(
            proposed["match"] is not None and proposed["exact"]
            for proposed in desired
        )
    )


def _active_signature(rank: Mapping[str, object]) -> object:
    shape = rank.get("shape_json")
    if not isinstance(shape, Mapping):
        raise ValueError("active rank is missing shape_json")
    return _signature(rank.get("role"), shape)


def _desired_signature(
    running_job: Mapping[str, object],
    action: Mapping[str, object],
    rank: Mapping[str, object],
) -> object:
    config = rank.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("swap ladder rank is missing config")
    shape: dict[str, Any] = dict(config)
    env = rank.get("env")
    if env is not None:
        shape["env"] = env
    if rank.get("mechanism_id") is not None:
        shape["mechanism_id"] = rank["mechanism_id"]
    for key in ("predicted_y", "predicted_v"):
        if isinstance(rank.get(key), Mapping):
            shape[key] = dict(rank[key])
    for action_key, shape_key in (
        ("target_p99_ttft_ms", "target_p99_ttft_ms"),
        ("target_p99_tpot_ms", "target_p99_tpot_ms"),
    ):
        if action.get(action_key) is not None:
            shape[shape_key] = action[action_key]

    spec = running_job.get("spec_json")
    if not shape.get("model_id") and isinstance(spec, Mapping) and spec.get("model_id"):
        shape["model_id"] = str(spec["model_id"])
    if not shape.get("gpu_type") and isinstance(env, (list, tuple)) and len(env) >= 5:
        shape["gpu_type"] = str(env[4])
    shape.setdefault("engine_name", "vllm")
    shape.setdefault("sp", 1)
    shape.setdefault("ep", 1)
    shape.setdefault("cp", 1)
    return _signature(rank.get("role"), shape)


def _signature(role: object, shape: Mapping[str, object]) -> object:
    normalized = dict(shape)
    count = normalized.get("gpu_count", normalized.get("count"))
    normalized.pop("gpu_count", None)
    if count is not None:
        normalized["count"] = count
    env = normalized.pop("env", normalized.pop("pool_id", None))
    if env is not None:
        normalized["env"] = _environment(env)
    normalized.pop("dp", None)
    normalized.setdefault("sp", 1)
    normalized.setdefault("ep", 1)
    normalized.setdefault("cp", 1)
    normalized.setdefault("engine_name", "vllm")
    role_value = getattr(role, "value", role)
    return str(role_value), _freeze(normalized)


def _environment(value: object) -> object:
    if isinstance(value, str):
        return tuple(value.split("|"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(part) for part in value)
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value
