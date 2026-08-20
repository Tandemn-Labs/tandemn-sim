"""Test normalized planner LLM usage artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from llm_usage import LlmUsage, OpenAIUsageClient, append_usage, usage_path


class LlmUsageTest(unittest.TestCase):
    """Keep one minimal usage shape beside each run trace."""

    def test_openai_wrapper_records_chat_and_responses_calls(self) -> None:
        response = SimpleNamespace(
            model="model-used",
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=10),
        )
        resource = SimpleNamespace(create=lambda **_kwargs: response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=resource),
            responses=resource,
        )
        usage = []
        tracked = OpenAIUsageClient(client, 2, "model-requested", usage)

        tracked.chat.completions.create()
        tracked.responses.create()

        self.assertEqual(
            usage,
            [
                LlmUsage(2, 40, 10, "model-used"),
                LlmUsage(2, 40, 10, "model-used"),
            ],
        )

    def test_writes_adjacent_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_file = Path(directory) / "run-1.jsonl"
            path = usage_path(trace_file)

            append_usage(path, LlmUsage(3, 120, 30, "model-used"))

            self.assertEqual(path, Path(directory) / "run-1.llm-usage.jsonl")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "tick_number": 3,
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "model": "model-used",
                },
            )


if __name__ == "__main__":
    unittest.main()
