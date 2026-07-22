"""Generate request previews from AIPerf workload configurations."""

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from aiperf.common import random_generator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.loader.plan import load_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "workloads"
LOGGER = logging.getLogger(__name__)


def generate(job: str, count: int = 5, full_prompt: bool = False) -> None:
    """Generate and log request previews for one workload."""
    if count < 1:
        raise ValueError("count must be at least 1")

    config = _workload_config(job)
    for request in _generated_requests(config, count, full_prompt):
        LOGGER.info("%s", json.dumps(request, ensure_ascii=False))


def _workload_config(job: str) -> Path:
    config = WORKLOADS / f"{job}.yaml"
    if config.exists():
        return config

    available = ", ".join(path.stem for path in sorted(WORKLOADS.glob("*.yaml")))
    raise ValueError(f"Unknown job '{job}'. Available jobs: {available}")


def _generated_requests(
    config: Path,
    count: int,
    full_prompt: bool,
) -> Iterator[dict[str, object]]:
    plan = load_benchmark_plan(config)
    seed = plan.variation_seeds[0]
    random_generator.reset()
    random_generator.init(seed)

    run = BenchmarkRun(
        benchmark_id="preview-only",
        sweep_id=plan.sweep_id,
        cfg=plan.configs[0],
        variation=plan.variations[0],
        artifact_dir=ROOT,
        random_seed=seed,
    )
    tokenizer = Tokenizer.from_pretrained(
        run.cfg.tokenizer.name or "builtin",
        trust_remote_code=run.cfg.tokenizer.trust_remote_code,
        revision=run.cfg.tokenizer.revision,
    )
    conversations = SyntheticDatasetComposer(
        run=run, tokenizer=tokenizer
    ).create_dataset()

    generated = 0
    for conversation in conversations:
        for turn_number, turn in enumerate(conversation.turns, start=1):
            prompt = "\n".join(
                content for text in turn.texts for content in text.contents
            )
            messages: list[dict[str, str]] = []
            if conversation.system_message:
                messages.append(
                    {"role": "system", "content": conversation.system_message}
                )
            if conversation.user_context_message:
                messages.append(
                    {"role": "user", "content": conversation.user_context_message}
                )
            messages.append({"role": "user", "content": prompt})

            visible_messages = messages if full_prompt else _shorten_messages(messages)
            yield {
                "request_number": generated + 1,
                "session_id": conversation.session_id,
                "turn": turn_number,
                "conversation_turns": len(conversation.turns),
                "turn_delay_ms": turn.delay,
                "model": turn.model or run.cfg.get_model_names()[0],
                "stream": run.cfg.endpoint.streaming,
                "base_input_tokens": sum(
                    len(tokenizer.encode(message["content"])) for message in messages
                ),
                "max_tokens": turn.max_tokens,
                "messages": visible_messages,
            }
            generated += 1
            if generated == count:
                return


def _shorten_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    shortened = []
    for message in messages:
        content = message["content"]
        preview = content[:200] + ("..." if len(content) > 200 else "")
        shortened.append({"role": message["role"], "content": preview})
    return shortened


def main() -> None:
    """Run the workload generation command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="AIPerf workload data generation.")
    commands = parser.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser(
        "generate", help="Generate and print requests for one workload."
    )
    generate_parser.add_argument("job")
    generate_parser.add_argument("--count", type=int, default=5)
    generate_parser.add_argument("--full-prompt", action="store_true")
    args = parser.parse_args()

    try:
        generate(args.job, count=args.count, full_prompt=args.full_prompt)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
