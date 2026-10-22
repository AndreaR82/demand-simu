"""A scripted stand-in for vLLM's Anthropic Messages endpoint, for dry runs.

Claude Code talks to it exactly as it would to vLLM (ANTHROPIC_BASE_URL), but
the "model" is a fixed script: in every engineer turn it runs `git status`,
appends a line to SIMULATED.md, then ends the turn. Requests without a Bash
tool (background work) get a one-line text reply. It can also play the
engineer, on /v1/chat/completions or on /v1/messages with a JSON output
format: one follow-up, then accept. That is enough to exercise
simulate_engineers.py end to end without a GPU or an API bill, and the request
log previews what Run 0 measures: calls per turn, background calls, startup
calls and prompt size (chars / 4, not a real tokenizer).

    python mock_messages_server.py --port 8099 --log mock_requests.jsonl
"""

import argparse
import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def approx_tokens(body: dict) -> int:
    return len(json.dumps([body.get("system"), body.get("messages"), body.get("tools")])) // 4


def has_block(msg: dict, kind: str) -> bool:
    c = msg.get("content")
    return isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == kind for b in c)


def engineer_decision(messages: list[dict], replies: int) -> str:
    """The engineer's JSON: `replies` follow-ups after the opener, then accept."""
    said = sum(m.get("role") == "assistant" for m in messages)  # the opener is the first
    if said <= replies:
        return json.dumps({"action": "reply", "message": "Thanks. Can you double-check the edge cases?"})
    return json.dumps({"action": "accept", "message": ""})


def plan(body: dict, replies: int) -> tuple[str, list[dict], str]:
    """(kind, content blocks, stop_reason) for this request."""
    if (body.get("output_config") or {}).get("format"):  # simulate_engineers.py --engineer anthropic
        return "engineer", [{"type": "text", "text": engineer_decision(body.get("messages") or [], replies)}], "end_turn"
    tools = {t.get("name") for t in body.get("tools") or []}
    if "Bash" not in tools:
        return "background", [{"type": "text", "text": "Simulated reply."}], "end_turn"

    msgs = body.get("messages") or []
    # The engineer's prompts are user messages with no tool_result in them; tool
    # results can carry text blocks too (system reminders), so text is no signal.
    human = [m for m in msgs if m.get("role") == "user" and not has_block(m, "tool_result")]
    turn = len(human)
    step = 0
    for m in reversed(msgs):
        if m.get("role") == "user" and not has_block(m, "tool_result"):
            break
        if m.get("role") == "assistant" and has_block(m, "tool_use"):
            step += 1

    script = ["git status --short", f"echo 'simulated edit, turn {turn}' >> SIMULATED.md"]
    if step < len(script):
        return "agent", [
            {"type": "text", "text": f"Step {step + 1}: running a command."},
            {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": "Bash",
             "input": {"command": script[step], "description": "simulated step"}},
        ], "tool_use"
    return "agent", [{"type": "text", "text":
                      f"Done with turn {turn}: checked the tree and appended a line to "
                      "SIMULATED.md. Tell me if you want anything else changed."}], "end_turn"


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-messages/0.1"

    def log_message(self, fmt, *args):  # quiet; the JSONL log is the record
        pass

    def record(self, **rec) -> None:
        if self.server.log_path:
            with LOCK, open(self.server.log_path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "method": self.command,
                                    "path": self.path, **rec}) + "\n")

    def send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self.record(status=200)
            return self.send_json(200, {"data": [{"type": "model", "id": self.server.model,
                                                  "display_name": self.server.model}],
                                        "has_more": False})
        self.record(status=404)
        self.send_json(404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}})

    def do_HEAD(self):
        self.record(status=200)
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        path = self.path.split("?")[0]
        n_in = approx_tokens(body)
        if path == "/v1/messages/count_tokens":
            self.record(status=200, approx_in=n_in)
            return self.send_json(200, {"input_tokens": n_in})
        if path == "/v1/chat/completions":  # simulate_engineers.py --engineer openai
            self.record(status=200, kind="engineer", model=body.get("model"))
            text = engineer_decision(body.get("messages") or [], self.server.engineer_replies)
            return self.send_json(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
                                        "created": int(time.time()), "model": body.get("model"),
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": text}}]})
        if path != "/v1/messages":
            self.record(status=404)
            return self.send_json(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})

        kind, blocks, stop = plan(body, self.server.engineer_replies)
        if kind != "engineer" and random.random() < self.server.error_rate:
            self.record(status=529, kind=kind)
            return self.send_json(529, {"type": "error", "error": {"type": "overloaded_error", "message": "mock"}})
        n_out = max(1, len(json.dumps(blocks)) // 4)
        self.record(status=200, kind=kind, stream=bool(body.get("stream")), model=body.get("model"),
                    max_tokens=body.get("max_tokens"), n_messages=len(body.get("messages") or []),
                    n_tools=len(body.get("tools") or []), approx_in=n_in, approx_out=n_out, stop=stop,
                    session=self.headers.get("x-claude-code-session-id"),
                    beta=self.headers.get("anthropic-beta"), ua=self.headers.get("user-agent"))
        msg = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
               "model": body.get("model") or self.server.model, "content": [],
               "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": n_in, "output_tokens": 1}}
        time.sleep(self.server.ttft)
        if not body.get("stream"):
            time.sleep(n_out * self.server.tpot)
            msg.update(content=blocks, stop_reason=stop)
            msg["usage"]["output_tokens"] = n_out
            return self.send_json(200, msg)

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        def event(name: str, data: dict) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        event("message_start", {"type": "message_start", "message": msg})
        for i, b in enumerate(blocks):
            if b["type"] == "text":
                event("content_block_start", {"type": "content_block_start", "index": i,
                                              "content_block": {"type": "text", "text": ""}})
                words = b["text"].split(" ")
                for j, w in enumerate(words):
                    time.sleep(self.server.tpot)
                    event("content_block_delta", {"type": "content_block_delta", "index": i,
                                                  "delta": {"type": "text_delta",
                                                            "text": w + (" " if j < len(words) - 1 else "")}})
            else:
                event("content_block_start", {"type": "content_block_start", "index": i,
                                              "content_block": {**b, "input": {}}})
                time.sleep(self.server.tpot)
                event("content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "input_json_delta",
                                                        "partial_json": json.dumps(b["input"])}})
            event("content_block_stop", {"type": "content_block_stop", "index": i})
        event("message_delta", {"type": "message_delta",
                                "delta": {"stop_reason": stop, "stop_sequence": None},
                                "usage": {"output_tokens": n_out}})
        event("message_stop", {"type": "message_stop"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--model", default="mock-model", help="name reported by /v1/models")
    ap.add_argument("--ttft", type=float, default=0.2, help="seconds before the first event")
    ap.add_argument("--tpot", type=float, default=0.01, help="seconds per streamed chunk")
    ap.add_argument("--engineer-replies", type=int, default=1,
                    help="as the engineer: follow-ups after the opener before accepting")
    ap.add_argument("--error-rate", type=float, default=0.0,
                    help="fraction of model calls answered 529 overloaded, to exercise retry counting")
    ap.add_argument("--log", help="append one JSON line per request here")
    a = ap.parse_args()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    srv.model, srv.ttft, srv.tpot, srv.log_path = a.model, a.ttft, a.tpot, a.log
    srv.engineer_replies, srv.error_rate = a.engineer_replies, a.error_rate
    print(f"mock Messages API on http://{a.host}:{a.port}  log={a.log or '-'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
