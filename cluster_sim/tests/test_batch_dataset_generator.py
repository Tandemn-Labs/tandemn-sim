"""Validate deterministic batch dataset materialization."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from benchmarks.generate_batch_dataset import _records


class FakePrompts:
    """Return predictable prompt text without loading a tokenizer."""

    tokenizer = None

    def __init__(self) -> None:
        """Use the prompt factory as its minimal tokenizer."""
        self.tokenizer = self

    def generate(self, mean: int, stddev: int) -> str:
        """Encode the requested size in the generated text."""
        return f"prompt-{mean}-{stddev}"

    def encode(self, text: str) -> list[int]:
        """Return one token for test prompt text."""
        return [len(text)]

    def decode(self, tokens: list[int]) -> str:
        """Return test prompt text for the supplied tokens."""
        return f"prompt-{len(tokens)}"


class BatchDatasetGeneratorTest(unittest.TestCase):
    """Keep single- and multi-turn output batch-compatible."""

    def test_materializes_exact_request_count(self) -> None:
        """Preserve total requests while grouping multi-turn sessions."""
        source = yaml.safe_load(
            (
                Path(__file__).resolve().parents[2]
                / "benchmarks/15min_real/jobs/small-batch-eval.yaml"
            ).read_text(encoding="utf-8")
        )
        features = source["spec"]["features"]

        rows = list(
            _records(
                features,
                9,
                "test",
                "test-model",
                "/v1/chat/completions",
                "multi",
                2,
                3,
                4,
                random.Random(1),
                FakePrompts(),
            )
        )
        self.assertEqual(len(rows), 9)
        self.assertTrue(
            all(
                set(row) == {"custom_id", "method", "url", "body"}
                and row["method"] == "POST"
                and row["url"] == "/v1/chat/completions"
                for row in rows
            )
        )
        self.assertEqual(rows[0]["body"]["messages"][0]["role"], "user")
        self.assertEqual(set(rows[0]["body"]), {"model", "messages"})
        self.assertEqual(rows[0]["body"]["model"], "test-model")
        self.assertEqual(
            [message["role"] for message in rows[1]["body"]["messages"]],
            ["user", "assistant", "user"],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(len(path.read_text().splitlines()), len(rows))


if __name__ == "__main__":
    unittest.main()
