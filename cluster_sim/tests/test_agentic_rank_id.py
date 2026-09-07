"""Test Agentic rank identity at the Store/Orca boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from agentic import AgenticPlacement, _placement_ladder
from models import OnlineJob
from simulation_orca import ladder_to_ranks
from tandemn_system_data.ids import is_valid_id, new_job_id, new_plan_id


class AgenticRankIdTest(unittest.TestCase):
    """Verify Agentic placement ladders create canonical Store ranks."""

    def test_place_ladder_creates_a_canonical_rank_id(self) -> None:
        job = OnlineJob(
            name="waiting",
            store_job_id="waiting",
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            starts_at_ns=0,
            stops_at_ns=1,
            workload={"features": {}, "phases": []},
        )
        with patch("agentic._materialize_launch_config", return_value={}):
            ladder = _placement_ladder(
                job,
                {"model_id": job.model_name, "ep": 1},
                AgenticPlacement(
                    decision="place",
                    environment="reserved|aws|us-east-1|use1-az1|H100",
                    instance_type="p5.48xlarge",
                configuration={"replicas_needed": 2, "tp": 2, "pp": 1, "dp": 3},
                ),
                {
                    "reserved|aws|us-east-1|use1-az1|H100": {
                        "gpu_type": "H100",
                    "total": 48,
                    "free": 48,
                        "pools": [
                            {
                                "instance_type": "p5.48xlarge",
                            "total": 48,
                            "free": 48,
                                "gpus_per_instance": 8,
                            }
                        ],
                    }
                },
                {job.model_name: {}},
            )

        ranks = ladder_to_ranks(
            [ladder],
            job_id=new_job_id(),
            plan_id=new_plan_id(),
        )

        self.assertEqual(len(ranks), 1)
        self.assertTrue(is_valid_id(ranks[0].rank_id, "rank"))
        self.assertEqual(ranks[0].shape_json["count"], 2)
        self.assertEqual(ranks[0].n_replicas, 6)


if __name__ == "__main__":
    unittest.main()
