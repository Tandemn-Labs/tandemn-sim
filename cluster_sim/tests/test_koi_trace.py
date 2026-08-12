"""Test Koi debug-log normalization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from koi_trace import (
    compile_koi_events,
    normalize_koi_candidate,
    normalize_koi_evidence,
    normalize_koi_cusum,
    normalize_koi_llm_calls,
    normalize_koi_rank_predictions,
    normalize_koi_regret,
    normalize_koi_search_candidates,
    normalize_koi_state,
    normalize_koi_tick_summary,
)
from trace import append_event


class KoiTraceTest(unittest.TestCase):
    """Verify Koi debug facts remain available in the common trace."""

    def test_normalizes_state_with_raw_payload(self) -> None:
        """Preserve state timing and diagnostics without reinterpreting them."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S2_VALIDATE",
                "duration_ms": 12.5,
                "mechanism_diagnostics": [{"id": "m-1"}],
            },
        }

        normalized = normalize_koi_state(event)

        self.assertEqual(normalized["kind"], "koi_state")
        self.assertEqual(normalized["state"], "S2_VALIDATE")
        self.assertEqual(normalized["payload"], event["payload"])

    def test_ignores_non_state_events(self) -> None:
        """Leave other Koi event kinds for their dedicated normalizers."""
        self.assertIsNone(normalize_koi_state({"kind": "tick_summary"}))

    def test_normalizes_tick_summary_with_planner_facts(self) -> None:
        """Preserve Koi timing and plan diagnostics for common-trace analysis."""
        event = {
            "kind": "tick_summary",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state_durations_ms": {"S4_AGENTIC_PLAN": 1200},
                "error": None,
                "deploy_acks": [{"plan_id": "plan-4"}],
            },
        }

        normalized = normalize_koi_tick_summary(event)

        self.assertEqual(normalized["kind"], "koi_tick_summary")
        self.assertEqual(normalized["state_durations_ms"], {"S4_AGENTIC_PLAN": 1200})
        self.assertEqual(normalized["payload"], event["payload"])

    def test_normalizes_llm_calls_with_nullable_token_usage(self) -> None:
        """Keep today's missing provider usage null without estimating tokens."""
        event = {
            "kind": "llm_calls",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {"calls": [{"elapsed_sec": 2.5, "error": None}]},
        }

        normalized = normalize_koi_llm_calls(event)

        self.assertEqual(normalized[0]["elapsed_ms"], 2500.0)
        self.assertIsNone(normalized[0]["total_tokens"])

    def test_normalizes_s2_evidence_rows(self) -> None:
        """Keep Koi evidence V/Y observations and predictions unmodified."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S2_VALIDATE",
                "evidence_rows": [
                    {
                        "job_id": "job-a",
                        "rank_id": "rank-0",
                        "v_observed": {"kv_cache_util": 0.7},
                    }
                ],
            },
        }

        normalized = normalize_koi_evidence(event)

        self.assertEqual(normalized[0]["job_id"], "job-a")
        self.assertEqual(normalized[0]["payload"], event["payload"]["evidence_rows"][0])

    def test_extracts_rank_prediction_facts_from_evidence(self) -> None:
        """Expose Koi V/Y predictions without changing the evidence payload."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S2_VALIDATE",
                "evidence_rows": [
                    {
                        "job_id": "job-a",
                        "rank_id": "rank-0",
                        "v_predicted": {"kv_cache_util": 0.7},
                        "y_observed": {"p99_ttft_ms": 840},
                    }
                ],
            },
        }

        normalized = normalize_koi_rank_predictions(event)

        self.assertEqual(normalized[0]["v_hat"], {"kv_cache_util": 0.7})
        self.assertEqual(normalized[0]["y_actual"], {"p99_ttft_ms": 840})

    def test_normalizes_s2_cusum_diagnostics(self) -> None:
        """Keep Koi's CUSUM trajectory and thresholds in the common trace."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S2_VALIDATE",
                "mechanism_diagnostics": [
                    {
                        "job_id": "job-a",
                        "rank_id": "rank-0",
                        "mechanism_id": "mechanism-a",
                        "s_plus": [0.0, 1.0],
                    }
                ],
            },
        }

        normalized = normalize_koi_cusum(event)

        self.assertEqual(normalized[0]["mechanism_id"], "mechanism-a")
        self.assertEqual(normalized[0]["payload"]["s_plus"], [0.0, 1.0])

    def test_normalizes_s3_regret_diagnostics(self) -> None:
        """Preserve the slow-loop regret slope without deriving a score."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S3_SLOW_UPDATE",
                "new_slow_state": {"regret_slope": 0.12, "z_star_t": {}},
            },
        }

        normalized = normalize_koi_regret(event)

        self.assertEqual(normalized["regret_slope"], 0.12)
        self.assertEqual(normalized["payload"]["z_star_t"], {})

    def test_normalizes_s4_candidate_plan(self) -> None:
        """Keep the unvalidated Koi plan available for search analysis."""
        event = {
            "kind": "state",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "state": "S4_AGENTIC_PLAN",
                "candidate_plan": {"actions": [{"job_id": "job-a", "type": "place"}]},
            },
        }

        normalized = normalize_koi_candidate(event)

        self.assertEqual(normalized["payload"], event["payload"]["candidate_plan"])

    def test_normalizes_scored_search_candidates(self) -> None:
        """Keep each agent-scored candidate as a raw search fact."""
        event = {
            "kind": "agent_trace",
            "tick": 4,
            "ts": "2026-01-01T00:20:00+00:00",
            "payload": {
                "events": [
                    {"kind": "kp_candidate_scored", "score": 0.8},
                    {"kind": "repl_exec"},
                ]
            },
        }

        normalized = normalize_koi_search_candidates(event)

        self.assertEqual(normalized[0]["payload"]["score"], 0.8)

    def test_compiles_native_koi_events(self) -> None:
        """Collect all supported Koi facts from one JSONL log."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_event(
                path,
                {
                    "kind": "state",
                    "tick": 4,
                    "ts": "2026-01-01T00:20:00+00:00",
                    "payload": {
                        "state": "S3_SLOW_UPDATE",
                        "new_slow_state": {"regret_slope": 0.12},
                    },
                },
            )
            append_event(
                path,
                {
                    "kind": "llm_calls",
                    "tick": 4,
                    "ts": "2026-01-01T00:20:00+00:00",
                    "payload": {"calls": [{"elapsed_sec": 2.5}]},
                },
            )

            compiled = compile_koi_events(path)

        self.assertEqual([event["kind"] for event in compiled], ["koi_state", "koi_regret", "koi_llm_call"])


if __name__ == "__main__":
    unittest.main()
