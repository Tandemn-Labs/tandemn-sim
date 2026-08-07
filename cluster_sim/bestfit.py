"""Run static benchmark-aware best-fit placement in the cluster simulator."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

from models import CreatedPlan, PlannerResult, Scenario

BEST_FIT_CONFIGURATIONS: dict[
    str,
    dict[str, dict[str, object]],
] = {
    "real_life_12h": {
        "job-benchmark_timing_real_12hr_192gpu-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-5": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-6": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-10-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-10-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-10-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-45-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-45-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-45-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-90-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-90-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-150-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-150-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-210-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-210-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-270-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-270-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-360-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_real_12hr_192gpu-360-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-420-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-420-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-480-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-480-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-480-3": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-570-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_12hr_192gpu-570-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-630-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 12},
        },
        "job-benchmark_timing_real_12hr_192gpu-630-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_12hr_192gpu-690-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-690-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
    },
    "real_life_24h": {
        "job-benchmark_timing_real_24hr_192gpu-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-5": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-6": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-0-7": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_24hr_192gpu-10-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_24hr_192gpu-10-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-10-3": {
            "hardware": "A100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-30-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-30-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-60-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-60-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-120-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-120-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-180-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-180-2": {
            "hardware": "A100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_24hr_192gpu-300-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_24hr_192gpu-300-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-420-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-420-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-540-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-540-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-600-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-690-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_24hr_192gpu-720-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-720-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-840-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_real_24hr_192gpu-840-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-930-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_24hr_192gpu-930-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-960-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_24hr_192gpu-960-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-1080-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-1080-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-1200-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_24hr_192gpu-1200-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_24hr_192gpu-1320-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 12},
        },
        "job-benchmark_timing_real_24hr_192gpu-1320-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_24hr_192gpu-1410-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_24hr_192gpu-1410-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
    },
    "real_life_3h": {
        "job-benchmark_timing_real_3hr_192gpu-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-0-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-0-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-0-5": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-12-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-12-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-12-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-25-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_3hr_192gpu-25-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-25-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_3hr_192gpu-45-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_3hr_192gpu-45-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_3hr_192gpu-70-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-70-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-70-3": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-95-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_3hr_192gpu-95-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_real_3hr_192gpu-120-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_3hr_192gpu-145-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 3},
        },
    },
    "simulation_12h": {
        "job-benchmark_timing_12hr-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-5": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-6": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-10-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-10-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-10-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-25-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-45-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-45-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-90-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-90-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-135-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-135-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-180-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-180-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-240-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-240-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr-300-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-330-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-375-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-420-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 40},
        },
        "job-benchmark_timing_12hr-420-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-465-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-480-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-510-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 3},
        },
        "job-benchmark_timing_12hr-510-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-570-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-570-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-630-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 24},
        },
        "job-benchmark_timing_12hr-630-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-690-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-690-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 6},
        },
    },
    "simulation_24h": {
        "job-benchmark_timing_24hr-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-0-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-5": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-6": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-7": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-0-8": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-10-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-10-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-10-3": {
            "hardware": "A100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-10-4": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-30-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-30-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-60-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-60-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_24hr-120-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-120-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-180-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-180-2": {
            "hardware": "A100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-240-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-300-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-300-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-420-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 32},
        },
        "job-benchmark_timing_24hr-420-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-540-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-540-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-540-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-600-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-690-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-720-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-720-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-840-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 64},
        },
        "job-benchmark_timing_24hr-840-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-930-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-930-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-960-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-960-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_24hr-1020-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-1020-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-1080-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 6},
        },
        "job-benchmark_timing_24hr-1080-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_24hr-1140-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-1200-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 32},
        },
        "job-benchmark_timing_24hr-1200-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_24hr-1320-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 48},
        },
        "job-benchmark_timing_24hr-1320-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 32},
        },
        "job-benchmark_timing_24hr-1410-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_24hr-1410-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 32},
        },
    },
    "simulation_3h": {
        "job-benchmark_timing_3hr-0-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-4": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-5": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-7-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-7-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-7-3": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_3hr-15-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-15-2": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-15-3": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-32-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_3hr-32-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-49-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 20},
        },
        "job-benchmark_timing_3hr-49-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_3hr-67-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_3hr-67-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-67-3": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-85-1": {
            "hardware": "L40S-48GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-100-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-100-2": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-125-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-125-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 12},
        },
        "job-benchmark_timing_3hr-145-1": {
            "hardware": "H100-SXM-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-145-2": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 8, "pp": 1, "dp": 3},
        },
        "job-benchmark_timing_3hr-160-1": {
            "hardware": "MI300X-192GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
    },
}


_HARDWARE_ALIASES = {
    "h100": "h100-sxm-80gb",
    "h100-sxm": "h100-sxm-80gb",
    "h100-sxm-80gb": "h100-sxm-80gb",
    "a100": "a100-sxm-80gb",
    "a100-sxm": "a100-sxm-80gb",
    "a100-sxm-80gb": "a100-sxm-80gb",
    "a100-pcie": "a100-pcie-40gb",
    "a100-pcie-40gb": "a100-pcie-40gb",
    "a10g": "a10g-24gb",
    "a10g-24gb": "a10g-24gb",
    "l40s": "l40s-48gb",
    "l40s-48gb": "l40s-48gb",
    "mi300x": "mi300x-192gb",
    "mi300x-192gb": "mi300x-192gb",
}


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout a best-fit control tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResourceTarget:
    """One concrete Store target matching a mapped accelerator."""

    environment_key: str
    environment: dict[str, Any]
    pool: dict[str, Any] | None
    gpu_type: str
    instance_type: str | None


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose the Store resource-map fields used by this adapter."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class BestFitPlannerDriver:
    """Create canonical Store plans from fixed per-benchmark placements."""

    method_name = "bestfit"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        plan_store: PlanStoreClient,
        backend: str,
    ) -> None:
        """Keep Store clients and select the scenario's static mapping."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.plan_store = plan_store
        self.backend = backend

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
    ) -> BestFitPlannerDriver:
        """Build best-fit with Tandemn Store public clients."""
        from tandemn_system_data.clients import (
            JobStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )

        postgres = postgres_client or PostgresClient()
        return cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            plan_store=PlanStore(postgres),
            backend=_text(settings.get("backend"), "backend"),
        )

    def create_plans(
        self,
        tick_number: int,
        logical_time: datetime,
    ) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        del logical_time
        try:
            snapshot = self._read_store_snapshot()
            actions = self._build_actions(snapshot)
            if not actions:
                return PlannerResult(tick_number=tick_number, plans=())

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(tick_number, actions),
                actions=actions,
            )
            self.plan_store.create(plan)
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
            )
        except Exception as error:
            return PlannerResult(
                tick_number=tick_number,
                plans=(),
                degraded=True,
                error=repr(error),
            )

    def _read_store_snapshot(self) -> StoreSnapshot:
        """Freeze resources and canonical lifecycle groups for one tick."""
        resource_map = self.resource_map_store.get()
        running = tuple(self.job_store.running_jobs(self.user_id))
        waiting = tuple(self.job_store.waiting_jobs(self.user_id))
        paused = tuple(self.job_store.paused_jobs(self.user_id))
        finished = tuple(self._finished_jobs())
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=tuple(_running_job_payload(job) for job in running),
            waiting_jobs=tuple(_model_payload(job) for job in waiting),
            paused_jobs=tuple(_model_payload(job) for job in paused),
            finished_jobs=tuple(_model_payload(job) for job in finished),
        )

    def _finished_jobs(self) -> Sequence[object]:
        """Read terminal jobs through the Store's available lifecycle API."""
        reader = getattr(self.job_store, "finished_jobs", None)
        if callable(reader):
            return cast(Sequence[object], reader(self.user_id))
        status_reader = getattr(self.job_store, "_jobs_with_status", None)
        if not callable(status_reader):
            raise RuntimeError("JobStore does not expose finished jobs")

        from tandemn_system_data.models import JobStatus

        return cast(
            Sequence[object],
            status_reader(self.user_id, JobStatus.FINISHED),
        )

    def _build_actions(self, snapshot: StoreSnapshot) -> list[dict[str, object]]:
        """Place waiting jobs greedily and include every lifecycle action."""
        benchmark = BEST_FIT_CONFIGURATIONS.get(self.benchmark_name, {})
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = _text(waiting.get("job_id"), "waiting Store job_id")
            mapped = benchmark.get(job_id)
            placement = None if mapped is None else _mapped_placement(mapped, resources)
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            target, tp, pp, dp = placement
            _reserve(target, tp * pp * dp)
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    target,
                    tp,
                    pp,
                    dp,
                    self.backend,
                )
            )
        for paused in snapshot.paused_jobs:
            actions.append({"job_id": paused["job_id"], "type": "defer"})
        for finished in snapshot.finished_jobs:
            actions.append({"job_id": finished["job_id"], "type": "preempt"})
        _validate_complete_job_actions(snapshot, actions)
        return actions


def _mapped_placement(
    mapped: Mapping[str, object],
    resources: dict[str, dict[str, Any]],
) -> tuple[ResourceTarget, int, int, int] | None:
    """Resolve one mapped hardware layout against current Store capacity."""
    hardware = _text(mapped.get("hardware"), "mapped hardware")
    configuration = _mapping(mapped.get("configuration"), "mapped configuration")
    tp = _positive_int(configuration.get("tp"), "tp")
    pp = _positive_int(configuration.get("pp"), "pp")
    dp = _positive_int(configuration.get("dp"), "dp")
    targets = _matching_targets(resources, hardware)
    feasible = []
    for target in targets:
        footprint = _target_footprint(target, tp * pp * dp)
        if footprint is None or footprint > _target_available(target):
            continue
        feasible.append(
            (
                _target_available(target) - footprint,
                target.environment_key,
                target.instance_type or "",
                target,
            )
        )
    if not feasible:
        return None
    return min(feasible, key=lambda value: value[:3])[3], tp, pp, dp


def _matching_targets(
    resources: dict[str, dict[str, Any]],
    hardware: str,
) -> tuple[ResourceTarget, ...]:
    """Return deterministic concrete targets for one mapped accelerator."""
    expected = _canonical_hardware(hardware)
    targets = []
    for environment_key, environment in sorted(resources.items()):
        gpu_type = str(environment.get("gpu_type") or environment_key.split("|")[-1])
        if _canonical_hardware(gpu_type) != expected:
            continue
        pools = environment.get("pools", [])
        if not isinstance(pools, list):
            raise ValueError("resource pools must be a list")
        for pool in pools or [None]:
            if pool is not None and not isinstance(pool, dict):
                raise ValueError("resource pool must be an object")
            targets.append(
                ResourceTarget(
                    environment_key=environment_key,
                    environment=environment,
                    pool=pool,
                    gpu_type=gpu_type,
                    instance_type=(
                        None
                        if pool is None
                        else _optional_text(pool.get("instance_type"))
                    ),
                )
            )
    return tuple(targets)


def _place_action(
    job_id: str,
    store_job: Mapping[str, object],
    target: ResourceTarget,
    tp: int,
    pp: int,
    dp: int,
    backend: str,
) -> dict[str, object]:
    """Combine deterministic Store fields with one mapped placement."""
    known = _known_job_fields(store_job)
    gpu_count = tp * pp * dp
    return {
        "job_id": job_id,
        "type": "place",
        "tenant_id": known.get("tenant_id") or store_job.get("user_id"),
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "mechanism_id": known.get("mechanism_id"),
        "budget_ref": known.get("budget_ref"),
        "rationale": known.get("rationale"),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": known.get("rank_id") or "rank_0",
                "env": target.environment_key.split("|"),
                "config": {
                    "model_id": known.get("model_id"),
                    "engine_name": known.get("engine_name") or backend,
                    "instance_type": target.instance_type,
                    "gpu_type": target.gpu_type,
                    "gpu_count": gpu_count,
                    "count": known.get("count"),
                    "tp": tp,
                    "pp": pp,
                    "dp": dp,
                    "ep": known.get("ep"),
                    "max_num_seq": known.get("max_num_seq"),
                    "max_num_batched_tokens": known.get("max_num_batched_tokens"),
                    "gpu_mem_util": known.get("gpu_mem_util"),
                    "prefix_cache_enabled": known.get("prefix_cache_enabled"),
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": 1,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def _target_available(target: ResourceTarget) -> int:
    """Return current capacity for one concrete target."""
    available = int(target.environment.get("free", 0))
    if target.pool is not None:
        available = min(available, int(target.pool.get("free", 0)))
    return max(0, available)


def _target_footprint(target: ResourceTarget, gpu_count: int) -> int | None:
    """Translate mapped GPUs into the target's allocation unit."""
    if target.pool is None:
        return gpu_count
    allocation_kind = str(
        target.pool.get("allocation_kind")
        or target.pool.get("allocation_unit")
        or "instance"
    ).lower()
    if allocation_kind == "gpu":
        return gpu_count
    gpus_per_instance = _positive_int(
        target.pool.get("gpus_per_instance"),
        "gpus_per_instance",
    )
    if gpu_count > gpus_per_instance:
        return None
    return gpus_per_instance


def _reserve(target: ResourceTarget, gpu_count: int) -> None:
    """Reserve a mapped placement before evaluating the next waiting job."""
    footprint = _target_footprint(target, gpu_count)
    if footprint is None or footprint > _target_available(target):
        raise RuntimeError("selected best-fit placement no longer fits")
    target.environment["free"] = int(target.environment["free"]) - footprint
    if target.pool is not None:
        target.pool["free"] = int(target.pool["free"]) - footprint


def _resources_with_free_capacity(
    resource_map: ResourceMap,
    running_jobs: Sequence[object],
) -> dict[str, dict[str, Any]]:
    """Normalize Store totals and subtract active chain allocations."""
    raw = resource_map.scheduling_summary()
    default_market = _default_market(resource_map)
    resources = {
        _normalize_env_key(str(key), default_market): dict(value)
        for key, value in raw.items()
    }
    used_by_env: dict[str, int] = {}
    used_by_pool: dict[tuple[str, str], int] = {}
    for running in running_jobs:
        payload = _model_payload(running)
        for chain_value in payload.get("chains", []):
            chain = _mapping(chain_value, "active chain")
            shape = _mapping(chain.get("shape_json", {}), "chain shape")
            raw_env = (
                chain.get("target_node") or shape.get("env") or shape.get("pool_id")
            )
            if raw_env is None:
                continue
            env_key = _normalize_env_key(_env_key(raw_env), default_market)
            count = _positive_int(shape.get("count"), "active chain count")
            environment = resources.get(env_key)
            if environment is None:
                continue
            pool = _select_pool(
                environment,
                _optional_text(shape.get("instance_type")),
            )
            footprint = count
            if pool is not None:
                allocation_kind = str(
                    pool.get("allocation_kind")
                    or pool.get("allocation_unit")
                    or "instance"
                ).lower()
                if allocation_kind != "gpu":
                    footprint = _positive_int(
                        pool.get("gpus_per_instance"),
                        "gpus_per_instance",
                    )
                pool_key = (env_key, str(pool.get("instance_type")))
                used_by_pool[pool_key] = used_by_pool.get(pool_key, 0) + footprint
            used_by_env[env_key] = used_by_env.get(env_key, 0) + footprint

    for env_key, environment in resources.items():
        environment["free"] = max(
            0,
            int(environment.get("total", 0)) - used_by_env.get(env_key, 0),
        )
        pools = [dict(pool) for pool in environment.get("pools", [])]
        for pool in pools:
            pool_key = (env_key, str(pool.get("instance_type")))
            pool["free"] = max(
                0,
                int(pool.get("total", 0)) - used_by_pool.get(pool_key, 0),
            )
        environment["pools"] = pools
    return resources


def _running_job_payload(running_job: object) -> dict[str, Any]:
    """Flatten a Store running-job aggregate for the frozen snapshot."""
    payload = _model_payload(running_job)
    job = dict(_mapping(payload.get("job"), "running job"))
    job["active_chains"] = payload.get("chains", [])
    return job


def _model_payload(value: object) -> dict[str, Any]:
    """Serialize one Store model or aggregate."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    job = getattr(value, "job", None)
    chains = getattr(value, "chains", None)
    if job is not None and chains is not None:
        return {
            "job": _model_payload(job),
            "chains": [_model_payload(chain) for chain in chains],
        }
    raise TypeError(f"Store value {type(value).__name__} is not serializable")


def _known_job_fields(store_job: Mapping[str, object]) -> dict[str, object]:
    """Flatten scheduler-visible fields from one Store job."""
    spec_value = store_job.get("spec_json")
    spec = spec_value if isinstance(spec_value, Mapping) else {}
    features_value = spec.get("job_features")
    features = features_value if isinstance(features_value, Mapping) else {}
    known: dict[str, object] = {}
    for source in (store_job, spec, features):
        for key, value in source.items():
            if value is not None:
                known.setdefault(str(key), value)
    return known


def _target_tps(known: Mapping[str, object]) -> float | None:
    """Derive output-token throughput from canonical Store job fields."""
    output_tokens = _number(known.get("osl_token_avg"))
    if output_tokens is None:
        return None
    kind = getattr(known.get("kind"), "value", known.get("kind"))
    try:
        if kind == "batch":
            token_budget = _positive_number(
                known.get("total_token_budget"),
                "total_token_budget",
            )
            deadline_hours = _positive_number(
                known.get("deadline_hrs"),
                "deadline_hrs",
            )
            input_tokens = _positive_number(
                known.get("isl_token_avg"),
                "isl_token_avg",
            )
            request_rate = token_budget / (
                (input_tokens + output_tokens) * deadline_hours * 3600
            )
        else:
            request_rate = _positive_number(
                known.get("request_arrival_rate"),
                "request_arrival_rate",
            )
    except ValueError:
        return None
    return round(request_rate * output_tokens, 3)


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job."""
    store_ids: list[str] = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
        snapshot.paused_jobs,
        snapshot.finished_jobs,
    ):
        store_ids.extend(_text(job.get("job_id"), "Store job_id") for job in jobs)
    action_ids = [_text(action.get("job_id"), "plan job_id") for action in actions]
    if len(store_ids) != len(set(store_ids)):
        raise ValueError("Store snapshot contains duplicate job_id values")
    if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(store_ids):
        raise ValueError("plan must contain exactly one action for every Store job")


def _tick_rationale(
    tick_number: int,
    actions: Sequence[Mapping[str, object]],
) -> str:
    """Summarize the deterministic action set persisted for one tick."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"Best-fit cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


def _select_pool(
    environment: Mapping[str, Any],
    instance_type: str | None,
) -> dict[str, Any] | None:
    """Resolve one environment pool from an active chain shape."""
    pools = environment.get("pools", [])
    if not isinstance(pools, list):
        raise ValueError("resource pools must be a list")
    typed_pools = []
    for value in pools:
        if not isinstance(value, dict):
            raise ValueError("resource pool must be an object")
        typed_pools.append(value)
    if not typed_pools:
        return None
    if instance_type is not None:
        return next(
            (
                pool
                for pool in typed_pools
                if pool.get("instance_type") == instance_type
            ),
            None,
        )
    return typed_pools[0] if len(typed_pools) == 1 else None


def _default_market(resource_map: ResourceMap) -> str:
    """Read the resource map's default market."""
    markets = getattr(resource_map, "market", None)
    if isinstance(markets, (list, tuple)) and markets:
        return str(markets[0])
    return str(markets or "reserved")


def _normalize_env_key(env_key: str, default_market: str) -> str:
    """Normalize legacy four-part environment keys."""
    parts = env_key.split("|")
    return "|".join((default_market, *parts)) if len(parts) == 4 else env_key


def _env_key(value: object) -> str:
    """Return one canonical pipe-delimited environment key."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(part) for part in value)
    return str(value)


def _canonical_hardware(value: str) -> str:
    """Normalize benchmark and Store accelerator labels."""
    key = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return _HARDWARE_ALIASES.get(key, key)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require one mapping value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    """Return optional nonempty text."""
    return None if value is None else _text(value, "optional text")


def _positive_int(value: object, name: str) -> int:
    """Require one positive integer value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str) -> float:
    """Require one positive numeric value."""
    number = _number(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _number(value: object) -> float | None:
    """Return one optional numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
