"""Route sessions across rank deployments in the cluster simulator."""

from __future__ import annotations

from typing import Mapping, Sequence

from models import Request


class StickyRoundRobinRouter:
    """Assign new sessions by Koi traffic share and retain healthy assignments."""

    def __init__(self) -> None:
        """Create empty per-job session assignments and weighted credits."""
        self._assignments: dict[tuple[str, str], str] = {}
        self._credits: dict[str, dict[str, float]] = {}

    def route(
        self,
        job_name: str,
        requests: Sequence[Request],
        rank_shares: Mapping[str, float],
    ) -> dict[str, tuple[Request, ...]]:
        """Partition sessions by Koi's traffic shares for one logical tick."""
        if not rank_shares or any(share < 0 for share in rank_shares.values()):
            raise ValueError("routing requires nonnegative rank traffic shares")
        total = sum(rank_shares.values())
        if total <= 0:
            raise ValueError("routing requires a positive total traffic share")

        shares = {rank_name: share / total for rank_name, share in rank_shares.items()}
        routed = {rank_name: [] for rank_name in shares}
        credits = self._credits.setdefault(job_name, {})
        for rank_name in tuple(credits):
            if rank_name not in shares:
                del credits[rank_name]
        for rank_name in shares:
            credits.setdefault(rank_name, 0.0)

        for request in requests:
            key = job_name, request.session_name
            rank_name = self._assignments.get(key)
            if rank_name not in shares:
                for candidate, share in shares.items():
                    credits[candidate] += share
                best_credit = max(credits[candidate] for candidate in shares)
                rank_name = next(
                    candidate
                    for candidate in shares
                    if credits[candidate] >= best_credit - 1e-12
                )
                credits[rank_name] -= 1.0
                self._assignments[key] = rank_name
            routed[rank_name].append(request)

        return {rank_name: tuple(values) for rank_name, values in routed.items()}

    def remove_rank(self, job_name: str, rank_name: str) -> None:
        """Drop assignments for a rank removed by a swap or terminal event."""
        for key, assigned_rank in tuple(self._assignments.items()):
            if key[0] == job_name and assigned_rank == rank_name:
                del self._assignments[key]
        self._credits.get(job_name, {}).pop(rank_name, None)
