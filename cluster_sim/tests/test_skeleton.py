"""Verify the Phase 1 standalone cluster-simulator skeleton."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


CLUSTER_SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_SIM_ROOT))


class SkeletonTest(unittest.TestCase):
    """Ensure every reviewed standalone skeleton module can load."""

    def test_modules_import(self) -> None:
        """Import every Phase 1 module without external dependencies."""
        for name in (
            "cli",
            "models",
            "planning",
            "routing",
            "runner",
            "scenario",
            "serving",
        ):
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
