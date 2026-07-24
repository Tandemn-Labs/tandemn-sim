"""Batch dataset generation checks."""

import json
import tempfile
import unittest
from pathlib import Path

from aiperf.common.tokenizer import Tokenizer

from data_generation.batch_dataset_generator import generate_batch_dataset

ROOT = Path(__file__).resolve().parents[1]


class BatchDatasetGeneratorTests(unittest.TestCase):
    """Check finite JSONL generation from a batch job YAML."""

    def test_generates_api_ready_records_with_requested_prompt_lengths(self) -> None:
        config = (
            ROOT
            / "benchmarks"
            / "simulation_3h"
            / "jobs"
            / "batch-mini-nightly-eval.yaml"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dataset.jsonl"
            summary = generate_batch_dataset(config, output, record_limit=3)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        tokenizer = Tokenizer.from_pretrained("builtin")
        self.assertEqual(summary.record_count, 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["method"], "POST")
        self.assertEqual(records[0]["url"], "/v1/chat/completions")
        prompts = [record["body"]["messages"][0]["content"] for record in records]
        self.assertEqual(len(set(prompts)), 3)
        for record in records:
            body = record["body"]
            prompt = body["messages"][0]["content"]
            self.assertGreaterEqual(len(tokenizer.encode(prompt)), 450)
            self.assertLessEqual(len(tokenizer.encode(prompt)), 6000)
            self.assertGreaterEqual(body["max_tokens"], 300)
            self.assertLessEqual(body["max_tokens"], 4000)


if __name__ == "__main__":
    unittest.main()
