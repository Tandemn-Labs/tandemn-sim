"""Test delayed Store outcome feedback for learning planners."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from outcome_feedback import OutcomeFeedback, fingerprint


class OutcomeFeedbackTest(unittest.TestCase):
    """Verify Store metrics become bounded empirical placement evidence."""

    def test_online_metrics_use_contextual_bandit_reward(self) -> None:
        """Score rank telemetry with the shared TTFT, TPOT, and TPS weights."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        placement = _fingerprint("online")
        feedback = OutcomeFeedback("user")
        feedback.bind(
            SimpleNamespace(
                rows_for_rank_window=lambda *_args: (
                    SimpleNamespace(
                        ts=start,
                        p99_ttft_ms=250.0,
                        p99_tpot_ms=25.0,
                        throughput_token_per_sec=150.0,
                    ),
                )
            ),
            SimpleNamespace(read_after=lambda *_args, **_kwargs: ()),
        )
        feedback.record(
            fingerprint=placement,
            rank_id="rank-online",
            window_start=start,
            is_batch=False,
            required_tps=100.0,
            ttft_target_ms=500.0,
            tpot_target_ms=50.0,
        )

        outcomes = feedback.observe(start + timedelta(minutes=5))

        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0].reward, 0.5)
        self.assertEqual(outcomes[0].sample_count, 1)
        self.assertEqual(feedback.for_fingerprint(placement), outcomes[0].as_dict())
        self.assertEqual(
            feedback.outcomes_for_job_ids(("online",)),
            (outcomes[0].as_dict(),),
        )

    def test_batch_waits_for_deadline_event(self) -> None:
        """Keep a batch action pending until its terminal Store outcome exists."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        placement = _fingerprint("batch")
        feedback = OutcomeFeedback("user")
        feedback.bind(
            SimpleNamespace(
                rows_for_rank_window=lambda *_args: (
                    SimpleNamespace(
                        ts=start,
                        p99_ttft_ms=100.0,
                        p99_tpot_ms=10.0,
                        throughput_token_per_sec=100.0,
                    ),
                )
            ),
            SimpleNamespace(
                read_after=lambda *_args, **_kwargs: (
                    SimpleNamespace(
                        event_id="event-1",
                        user_id="user",
                        job_id="batch",
                        payload_json={"deadline_met": True},
                    ),
                )
            ),
        )
        feedback.record(
            fingerprint=placement,
            rank_id="rank-batch",
            window_start=start,
            is_batch=True,
            required_tps=100.0,
            ttft_target_ms=None,
            tpot_target_ms=None,
        )

        self.assertTrue(feedback.has_pending_batch("batch"))
        outcomes = feedback.observe(start + timedelta(minutes=5))

        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0].reward, 0.6)
        self.assertTrue(outcomes[0].batch_deadline_met)
        self.assertFalse(feedback.has_pending_batch("batch"))


def _fingerprint(job_id: str):
    """Create one deterministic per-job placement identity for feedback tests."""
    return fingerprint(
        job_id=job_id,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        environment="reserved|aws|us-east|use1-az1|H100",
        instance_type=None,
        gpu_type="H100",
        tp=1,
        pp=1,
        dp=1,
    )
