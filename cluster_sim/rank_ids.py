"""Canonical Store rank identity helpers."""

from collections.abc import Mapping

from tandemn_system_data.ids import new_rank_id


def rank_id(known: Mapping[str, object]) -> str:
    """Preserve an existing rank ID or allocate a canonical Store ID."""
    return str(known["rank_id"]) if known.get("rank_id") else new_rank_id()
