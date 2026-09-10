"""Test rank identity reconciliation for swap plans."""

# ruff: noqa: E402

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from rank_ids import reconcile_swap_rank_ids
from tandemn_system_data.ids import is_valid_id

ACTIVE_ID = "rank_01ARZ3NDEKTSV4RRFFQ69G5FAV"
ENV = ["reserved", "gcp", "us-central1", "default", "H100"]


def _config(*, dp: int = 1, tp: int = 1, engine_version: str = "0.16.0"):
    return {
        "engine_name": "vllm",
        "engine_version": engine_version,
        "model_id": "model",
        "precision": "fp16",
        "instance_type": "a3-highgpu-8g",
        "gpu_type": "H100",
        "gpu_count": tp,
        "count": tp,
        "tp": tp,
        "pp": 1,
        "dp": dp,
        "ep": 1,
        "sp": 1,
        "cp": 1,
        "rank_traffic_share": 1.0,
    }


def _running(*, dp: int = 1, tp: int = 1, rank_id: str = ACTIVE_ID):
    return {
        "job_id": "job",
        "spec_json": {"model_id": "model"},
        "active_ranks": [
            {
                "rank_id": rank_id,
                "role": "aggregate",
                "n_replicas": dp,
                "shape_json": {
                    **_config(dp=dp, tp=tp),
                    "env": ENV,
                    "target_p99_ttft_ms": 700.0,
                    "target_p99_tpot_ms": 55.0,
                },
            }
        ],
    }


def _action(*, dp: int = 1, tp: int = 1, engine_version: str = "0.16.0"):
    return {
        "job_id": "job",
        "type": "swap",
        "target_p99_ttft_ms": 700.0,
        "target_p99_tpot_ms": 55.0,
        "ladder": [
            {
                "rank_id": "placeholder",
                "role": "aggregate",
                "env": list(ENV),
                "config": _config(dp=dp, tp=tp, engine_version=engine_version),
                "n_replicas": dp,
                "mechanism_id": None,
            }
        ],
    }


class RankIdentityTest(unittest.TestCase):
    def test_exact_deployment_is_a_noop(self) -> None:
        action = _action()

        self.assertTrue(reconcile_swap_rank_ids(_running(), action))
        self.assertEqual(action["ladder"][0]["rank_id"], ACTIVE_ID)

    def test_dp_only_change_reuses_rank_id(self) -> None:
        action = _action(dp=2)

        self.assertFalse(reconcile_swap_rank_ids(_running(), action))
        self.assertEqual(action["ladder"][0]["rank_id"], ACTIVE_ID)

    def test_non_dp_change_allocates_fresh_rank_id(self) -> None:
        for field, action in (
            ("tp", _action(tp=2)),
            ("runtime", _action(engine_version="0.17.0")),
        ):
            with self.subTest(field=field):
                proposed = copy.deepcopy(action)
                self.assertFalse(reconcile_swap_rank_ids(_running(), proposed))
                replacement = proposed["ladder"][0]["rank_id"]
                self.assertTrue(is_valid_id(replacement, "rank"))
                self.assertNotEqual(replacement, ACTIVE_ID)

    def test_mixed_ladder_retains_only_unchanged_rank(self) -> None:
        second_id = "rank_01ARZ3NDEKTSV4RRFFQ69G5FAW"
        running = _running()
        second = copy.deepcopy(running["active_ranks"][0])
        second["rank_id"] = second_id
        second["shape_json"]["tp"] = 2
        second["shape_json"]["gpu_count"] = 2
        second["shape_json"]["count"] = 2
        running["active_ranks"].append(second)
        action = _action()
        changed = copy.deepcopy(action["ladder"][0])
        changed["config"]["tp"] = 4
        changed["config"]["gpu_count"] = 4
        changed["config"]["count"] = 4
        action["ladder"].append(changed)

        self.assertFalse(reconcile_swap_rank_ids(running, action))
        self.assertEqual(action["ladder"][0]["rank_id"], ACTIVE_ID)
        replacement = action["ladder"][1]["rank_id"]
        self.assertTrue(is_valid_id(replacement, "rank"))
        self.assertNotIn(replacement, {ACTIVE_ID, second_id})


if __name__ == "__main__":
    unittest.main()
