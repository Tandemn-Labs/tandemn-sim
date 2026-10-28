"""Exercise proxy correction across the AIC planner/deployment boundary."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic import (
    AicEvaluationError,
    HardwareCandidate,
    ResourceTarget,
    _evaluate_aic_request,
    _normalize_result,
    _place_action,
    _policy_hardware_candidates,
    _sweep_aic_request,
)
from aic_proxy import SupportedProfile
from planning import PlannerContext, _build_aic
from serving import build_rank_cluster

from cluster_sim.tests.test_aic import _moe_request, _moe_row, _store_job
from cluster_sim.tests.test_aic_proxy import NEXT_MODEL, _model


class AicProxyIntegrationTest(unittest.TestCase):
    """Validate corrected sizing with real SDK pickers and isolated estimates."""

    def setUp(self) -> None:
        """Provide a target unavailable to AIC and a compatible measured proxy."""
        self.target = _model()
        self.proxy = SupportedProfile(
            replace(
                self.target, model_id="proxy-model", num_experts=128, active_experts=8
            ),
            "h100_sxm",
            "vllm",
            "0.22.0",
        )
        self.request = replace(
            _moe_request(),
            model=NEXT_MODEL,
            available_devices=12,
            parallelism_candidates=((4, 1),),
            target_request_rate=35,
            target_ttft_ms=250,
            target_tpot_ms=15,
            optimization_goal="latency",
        )
        self.raw = {
            **_moe_row(),
            "ttft": 100,
            "tpot": 10,
            "tokens/s": 2000,
            "seq/s": 20,
        }
        stack = self.enterContext(ExitStack())
        stack.enter_context(
            patch("aic_proxy.load_model_profile", return_value=self.target)
        )
        self.inventory = stack.enter_context(
            patch("aic_proxy.load_supported_profiles", return_value=(self.proxy,))
        )
        self.factors = stack.enter_context(
            patch("aic_proxy.phase_correction_factors", return_value=(0.5, 0.8))
        )
        self.sweep = stack.enter_context(
            patch("aic._sweep_aic_request", side_effect=self._sweep)
        )

    def _sweep(self, request):
        if request.model == NEXT_MODEL:
            raise AicEvaluationError("Qwen3NextForCausalLM is not supported")
        return pd.DataFrame([self.raw]), 128

    def test_corrected_capacity_sizes_replicas_and_deploys_original_target(
        self,
    ) -> None:
        """A proxy's two-replica estimate becomes three actual target replicas."""
        result = _evaluate_aic_request(self.request)
        self.assertEqual(result.replicas, 3)
        self.assertEqual(result.devices, 12)
        self.assertEqual(result.num_experts, 512)
        self.assertAlmostEqual(
            result.throughput_tokens_per_second, 3 * 1517.2413793103449
        )
        target = ResourceTarget(
            "reserved|aws|region|zone|H100", "H100", None, {"free": 12}, None
        )
        placement = _normalize_result(
            result, self.request, HardwareCandidate("h100_sxm", (target,), 12)
        )
        job = _store_job({"job_id": "job-next"})
        job["spec_json"]["model_id"] = NEXT_MODEL
        action = _place_action("job-next", job, placement, "vllm")
        rank = action["ladder"][0]
        shape = rank["config"]
        self.assertEqual(shape["model_id"], NEXT_MODEL)
        self.assertEqual(shape["ep"], 4)
        self.assertEqual(rank["n_replicas"], 3)
        metadata = shape["aic_prediction"]
        self.assertEqual(metadata["prediction_model_id"], "proxy-model")
        self.assertEqual(metadata["requested_model_id"], NEXT_MODEL)
        self.assertEqual(metadata["corrected_metrics"]["ttft"], 200)
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=rank["rank_id"],
                model_name=NEXT_MODEL,
                shape=shape,
                chains=(object(),) * 3,
            ),
            {
                "profile": "H100",
                "memory_gb": 80,
                "memory_bandwidth_gb_s": 3350,
                "link_bandwidth_gb_s": 900,
            },
        )
        self.assertTrue(
            all(
                instance["model_name"] == NEXT_MODEL
                for instance in cluster["nodes"][0]["instances"]
            )
        )

    def test_raw_proxy_sla_pass_cannot_hide_corrected_target_breach(self) -> None:
        """Screen latency after correction instead of accepting a fast proxy."""
        with self.assertRaisesRegex(AicEvaluationError, "under SLA"):
            _evaluate_aic_request(replace(self.request, target_ttft_ms=150))

    def test_native_estimate_is_preferred_without_proxy_scaling(self) -> None:
        """Use native per-replica capacity when the requested model is supported."""
        self.sweep.side_effect = None
        self.sweep.return_value = (pd.DataFrame([self.raw]), 512)
        result = _evaluate_aic_request(self.request)
        self.assertEqual(result.replicas, 2)
        self.assertEqual(result.configuration["aic_prediction"]["mode"], "native")
        self.inventory.assert_not_called()
        self.factors.assert_not_called()

    def test_corrected_replica_footprint_must_fit_available_capacity(self) -> None:
        """Do not retain proxy replica counts or silently cap the target load."""
        with self.assertRaisesRegex(AicEvaluationError, "within capacity"):
            _evaluate_aic_request(replace(self.request, available_devices=8))

    def test_proxy_search_expands_sla_before_correcting_faster_target(self) -> None:
        """A slower proxy's raw SLA violation must not prematurely prune a target."""
        self.raw["ttft"] = 300
        self.factors.return_value = (2, 2)
        result = _evaluate_aic_request(self.request)
        self.assertEqual(result.ttft_ms, 150)
        self.assertEqual(self.sweep.call_args.args[0].target_ttft_ms, 250 * 20)

    def test_ranking_and_factors_use_each_candidate_topology_and_batch(self) -> None:
        """Correction can reverse which TP configuration has the best latency."""

        def sweep(request):
            if request.model == NEXT_MODEL:
                raise AicEvaluationError("unsupported")
            tp = request.parallelism_candidates[0][0]
            return pd.DataFrame(
                [
                    {
                        **self.raw,
                        "tp": tp,
                        "moe_ep": tp,
                        "num_total_gpus": tp,
                        "parallel": f"tp{tp}pp1dp1etp1ep{tp}",
                        "bs": tp * 2,
                        "ttft": 80 if tp == 4 else 120,
                    }
                ]
            ), 128

        self.sweep.side_effect = sweep
        self.factors.side_effect = lambda *args, **kwargs: (
            (0.4, 1) if kwargs["tp"] == 4 else (1, 1)
        )
        result = _evaluate_aic_request(
            replace(
                self.request,
                parallelism_candidates=((4, 1), (8, 1)),
                available_devices=32,
                target_request_rate=1,
            )
        )
        self.assertEqual(result.configuration["tp"], 8)
        self.assertEqual(result.ttft_ms, 120)
        self.assertEqual(
            [call.kwargs["batch_size"] for call in self.factors.call_args_list], [8, 16]
        )

    def test_unusable_prediction_profile_tries_next_ranked_proxy(self) -> None:
        """Profile fallback occurs within the current call without runtime feedback."""
        second = replace(
            self.proxy, model=replace(self.proxy.model, model_id="second-proxy")
        )
        self.inventory.return_value = (self.proxy, second)

        def sweep(request):
            if request.model != "second-proxy":
                raise AicEvaluationError("profile unavailable")
            return pd.DataFrame([self.raw]), 128

        self.sweep.side_effect = sweep
        result = _evaluate_aic_request(self.request)
        self.assertEqual(
            result.configuration["aic_prediction"]["prediction_model_id"],
            "second-proxy",
        )
        self.assertEqual(result.configuration["aic_prediction"]["proxy_rank"], 2)
        self.assertEqual(self.sweep.call_count, 6)

    def test_concurrency_sizing_keeps_slot_count(self) -> None:
        """Slower per-request service does not create or remove concurrency slots."""
        result = _evaluate_aic_request(
            replace(self.request, target_request_rate=None, target_concurrency=17)
        )
        self.assertEqual(result.replicas, 3)
        self.assertEqual(result.configuration["concurrency"], 8)

    def test_expired_deadline_does_not_start_prediction(self) -> None:
        """Keep native and proxy search within the existing call deadline."""
        with self.assertRaisesRegex(AicEvaluationError, "deadline expired"):
            _evaluate_aic_request(self.request, deadline=0)
        self.sweep.assert_not_called()

    def test_local_unsupported_architecture_skips_native_download(self) -> None:
        """Use known local Next architecture instead of refetching it from HF."""
        with patch("aiconfigurator.sdk.utils.get_model_config_from_model_path") as sdk:
            with self.assertRaisesRegex(
                AicEvaluationError, "architecture.*not supported"
            ):
                _sweep_aic_request(
                    replace(self.request, model_configs_root=Path("models"))
                )
        sdk.assert_not_called()

    def test_aic_builder_receives_local_model_metadata_without_koi(self) -> None:
        """The existing evaluator-root argument supplies unsupported target metadata."""
        context = PlannerContext(
            "user", object(), object(), object(), None, [], Path("evaluator")
        )
        with patch("aic.AicPlannerDriver.build") as build:
            _build_aic(context)
        self.assertEqual(
            build.call_args.args[2]["model_configs_root"], "evaluator/configs/model"
        )

    def test_hardware_candidates_keep_distinct_target_precisions(self) -> None:
        """One AIC system cannot mix FP16/BF16 target correction assumptions."""
        a10g = ResourceTarget("a10g", "A10G", None, {"free": 8}, None)
        l4 = ResourceTarget("l4", "L4", None, {"free": 8}, None)
        candidates = _policy_hardware_candidates(
            (HardwareCandidate("l4", (a10g, l4), 8),),
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(
            [candidate.targets for candidate in candidates], [(a10g,), (l4,)]
        )


if __name__ == "__main__":
    unittest.main()
