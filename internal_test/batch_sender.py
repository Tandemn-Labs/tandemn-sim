"""Stream one complete batch dataset to the internal test receiver."""

import argparse
import http.client
import json
import logging
import queue
import random
import threading
from pathlib import Path
from urllib.parse import quote, urlsplit

from data_generation.batch_dataset_generator import (
    SyntheticPromptGenerator,
    build_request_record,
    load_batch_dataset_spec,
)

LOGGER = logging.getLogger(__name__)
WORKERS = 16
PROGRESS_INTERVAL = 1_000
RequestItem = tuple[str, bytes] | None


def main() -> None:
    """Generate and send every configured record with bounded memory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    args = parse_args()
    send_batch(
        args.config,
        args.job_id,
        args.receiver,
        record_limit=args.records,
    )


def parse_args() -> argparse.Namespace:
    """Parse the controller-provided batch sender arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--records", type=int)
    return parser.parse_args()


def send_batch(
    config_path: Path,
    job_id: str,
    receiver: str,
    *,
    record_limit: int | None = None,
) -> None:
    """Generate and concurrently send one complete finite dataset."""
    spec = load_batch_dataset_spec(config_path)
    record_count = spec.record_count if record_limit is None else record_limit
    if record_count <= 0 or record_count > spec.record_count:
        raise ValueError("record_limit must be between 1 and the configured count")

    parsed_receiver = urlsplit(receiver)
    if parsed_receiver.scheme != "http" or not parsed_receiver.hostname:
        raise ValueError("receiver must be an http URL with a hostname")
    receiver_prefix = (
        f"{parsed_receiver.path.rstrip('/')}/jobs/{quote(job_id, safe='')}"
    )
    port = parsed_receiver.port or 80
    requests: queue.Queue[RequestItem] = queue.Queue(maxsize=WORKERS * 2)
    failures: list[str] = []
    failure_lock = threading.Lock()
    workers = [
        threading.Thread(
            target=request_worker,
            args=(
                requests,
                parsed_receiver.hostname,
                port,
                failures,
                failure_lock,
            ),
        )
        for _ in range(WORKERS)
    ]
    for worker in workers:
        worker.start()

    LOGGER.info(
        "[batch-sender] job=%s dataset=%s records=%d workers=%d",
        job_id,
        spec.dataset_id,
        record_count,
        WORKERS,
    )
    random_source = random.Random(spec.seed)
    prompt_generator = SyntheticPromptGenerator(spec.seed)
    for index in range(record_count):
        input_tokens = spec.input_tokens.sample(random_source)
        output_tokens = spec.output_tokens.sample(random_source)
        prompt = prompt_generator.generate(input_tokens)
        record = build_request_record(spec, index, prompt, output_tokens)
        body = json.dumps(record["body"], separators=(",", ":")).encode()
        requests.put((f"{receiver_prefix}{record['url']}", body))
        if (index + 1) % PROGRESS_INTERVAL == 0:
            LOGGER.info(
                "[batch-sender] job=%s generated=%d/%d",
                job_id,
                index + 1,
                record_count,
            )

    for _ in workers:
        requests.put(None)
    requests.join()
    for worker in workers:
        worker.join()
    if failures:
        raise RuntimeError(
            f"{len(failures)} batch requests failed; first error: {failures[0]}"
        )
    LOGGER.info("[batch-sender] job=%s completed records=%d", job_id, record_count)


def request_worker(
    requests: queue.Queue[RequestItem],
    host: str,
    port: int,
    failures: list[str],
    failure_lock: threading.Lock,
) -> None:
    """Send queued requests over one reusable local HTTP connection."""
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        while True:
            item = requests.get()
            try:
                if item is None:
                    return
                path, body = item
                connection.request(
                    "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                if not 200 <= response.status < 300:
                    with failure_lock:
                        failures.append(f"HTTP {response.status} for {path}")
            except (OSError, http.client.HTTPException) as error:
                connection.close()
                connection = http.client.HTTPConnection(host, port, timeout=30)
                with failure_lock:
                    failures.append(str(error))
            finally:
                requests.task_done()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
