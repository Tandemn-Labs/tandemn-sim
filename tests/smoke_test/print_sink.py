"""Print locally generated AIPerf requests and return minimal mock responses."""

import itertools
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"
PORT = 8765
LOGGER = logging.getLogger(__name__)
REQUEST_NUMBERS = itertools.count(1)


class RequestHandler(BaseHTTPRequestHandler):
    """Handle the endpoint types used by the smoke jobs."""

    def do_GET(self) -> None:
        """Return a basic health response."""
        self.send_json({"status": "ready"})

    def do_POST(self) -> None:
        """Log one generated request and return a compatible response."""
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "request body must be JSON"}, status=400)
            return

        LOGGER.info(
            "[request %04d] %s",
            next(REQUEST_NUMBERS),
            json.dumps(request_summary(self.path, payload), sort_keys=True),
        )

        if self.path.endswith("/embeddings"):
            input_count = len(payload.get("input", []))
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [0.0],
                        }
                        for index in range(input_count)
                    ],
                }
            )
        elif self.path.endswith("/rerank"):
            self.send_json(
                [
                    {"index": index, "score": 1.0 / (index + 1)}
                    for index, _ in enumerate(payload.get("texts", []))
                ]
            )
        elif payload.get("stream"):
            self.send_streaming_chat()
        else:
            self.send_json(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "smoke"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

    def send_json(self, body: object, status: int = 200) -> None:
        """Send one JSON response."""
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_streaming_chat(self) -> None:
        """Send a minimal OpenAI-compatible streaming response."""
        event = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "smoke"},
                    "finish_reason": "stop",
                }
            ],
        }
        encoded = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the default HTTP access log."""


def request_summary(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the useful request shape without printing full synthetic prompts."""
    summary: dict[str, Any] = {"path": path}

    if path.endswith("/embeddings"):
        inputs = payload.get("input", [])
        summary.update(
            {
                "kind": "embeddings",
                "model": payload.get("model"),
                "inputs": len(inputs),
                "input_characters": sum(len(item) for item in inputs),
                "preview": preview(inputs[0]) if inputs else "",
            }
        )
    elif path.endswith("/rerank"):
        passages = payload.get("texts", [])
        summary.update(
            {
                "kind": "rerank",
                "query_characters": len(payload.get("query", "")),
                "passages": len(passages),
                "passage_characters": sum(len(item) for item in passages),
                "query_preview": preview(payload.get("query", "")),
            }
        )
    else:
        messages = payload.get("messages", [])
        summary.update(
            {
                "kind": "chat",
                "model": payload.get("model"),
                "max_completion_tokens": payload.get(
                    "max_completion_tokens",
                    payload.get("max_tokens"),
                ),
                "stream": payload.get("stream"),
                "messages": [
                    {
                        "role": message.get("role"),
                        "characters": content_length(message.get("content")),
                        "preview": preview(message.get("content")),
                    }
                    for message in messages
                ],
            }
        )

    return summary


def content_length(content: object) -> int:
    """Measure text inside a simple or multipart message."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return 0


def preview(content: object) -> str:
    """Return a short single-line preview of generated content."""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        return ""
    return " ".join(content.split())[:80]


def main() -> None:
    """Run the terminal request sink."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    LOGGER.info("[request-sink] listening on http://%s:%d", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("[request-sink] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
