"""Route sessions across rank deployments in the cluster simulator."""

from __future__ import annotations

from typing import Sequence

from models import Request


class StickyRoundRobinRouter:
    """Assign new sessions round-robin and retain healthy rank assignments."""

    def __init__(self) -> None:
        """Create empty per-job session assignments and round-robin cursors."""
        self._assignments: dict[tuple[str, str], str] = {}
        self._cursors: dict[str, int] = {}

    def route(
        self,
        job_name: str,
        requests: Sequence[Request],
        rank_names: Sequence[str],
    ) -> dict[str, tuple[Request, ...]]:
        """Partition complete sessions among active ranks for one logical tick."""
        if not rank_names or len(set(rank_names)) != len(rank_names):
            raise ValueError("routing requires unique active rank names")

        routed = {rank_name: [] for rank_name in rank_names}
        cursor = self._cursors.get(job_name, 0)
        active_ranks = set(rank_names)

        for request in requests:
            key = job_name, request.session_name
            rank_name = self._assignments.get(key)
            if rank_name not in active_ranks:
                rank_name = rank_names[cursor % len(rank_names)]
                self._assignments[key] = rank_name
                cursor += 1
            routed[rank_name].append(request)

        self._cursors[job_name] = cursor
        return {rank_name: tuple(values) for rank_name, values in routed.items()}

    def remove_rank(self, job_name: str, rank_name: str) -> None:
        """Drop assignments for a rank removed by a swap or terminal event."""
        for key, assigned_rank in tuple(self._assignments.items()):
            if key[0] == job_name and assigned_rank == rank_name:
                del self._assignments[key]
