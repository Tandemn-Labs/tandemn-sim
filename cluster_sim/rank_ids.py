"""Canonical Store rank identity helpers."""

from collections.abc import Mapping, Sequence

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
