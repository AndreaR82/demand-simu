"""Drive Claude Code headless over a task set, with an LLM playing the engineer.

Implements §4 of claude-code-agent-loadtest.md. Per task: one long-lived
`claude -p --input-format stream-json` process in a sandbox gets the opener;
when the agent's `result` arrives, the engineer LLM sees what a person would
(the reply, the diff stat, the check result) and answers, until it accepts,
abandons or runs out of turns. Engineer think time is modelled and logged, and
only `think × --time-scale` is slept (0 while recording; §6.3 puts it back
into the trace).

Writes, per session, --out/<task>__<session8>/: stream.jsonl (Claude Code's
events), engineer.jsonl (one line per engineer turn, wall-clock timestamps for
§6.3), claude.stderr and cc/ (CLAUDE_CONFIG_DIR, transcripts in cc/projects/);
plus one summary line per session in --out/sessions.jsonl.

    # dry run, no GPU and no API bill: the mock plays the model and the engineer
    python mock_messages_server.py --port 8099 &
    python simulate_engineers.py --tasks example_tasks.jsonl --sandbox local \
        --base-url http://127.0.0.1:8099 --model mock-model --max-context-tokens 131072 \
        --engineer openai --engineer-url http://127.0.0.1:8099/v1 --engineer-model mock \
        --out runs/dry

    # Run 1: docker sandboxes, capture proxy in front of vLLM, engineer on Claude
    python simulate_engineers.py --tasks tasks.jsonl --sandbox docker --network cc-test \
        --base-url http://proxy:3001 --model $MODEL --max-context-tokens 131072 \
        --engineer anthropic --concurrency 4 --out runs/run1

--sandbox local runs the agent on this host with every permission granted: use
it with the mock or inside a throwaway VM, never with a real model on your
laptop. The docker sandbox has not been run yet (no Docker where this was
written); Run 0 is where it gets proven.
"""

import argparse
import json
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

Z90 = 1.2816  # standard normal quantile at 0.90

ENGINEER_PROMPT = """\
You are a software engineer using an AI coding agent in your own repository.
You know exactly what you want; it is in <intent>. The agent does not see <intent>.

Rules:
- Reveal intent the way a busy engineer would: start vague, add detail only when the
  agent asks or gets it wrong. Never paste <intent> verbatim.
- Reply in at most 2-3 sentences. No praise, no summaries of what the agent did.
- If the change is wrong, say what is wrong as you would notice it (a failing test,
  a wrong file, a missed case), not how to fix it.
- Patience: after {patience} replies that don't move things forward, abandon.
- Accept as soon as the change does what <intent> asks and the check passes; don't
  ask for polish you wouldn't ask for in real life.

Output JSON only: {{"action": "reply" | "accept" | "abandon", "message": "<text or empty>"}}

<intent>{intent}</intent>"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["reply", "accept", "abandon"]},
                   "message": {"type": "string"}},
    "required": ["action", "message"],
    "additionalProperties": False,
}

# Snapshot the working tree (untracked files included) as a tree object, through
# a throwaway index so the agent never sees anything staged.
SNAPSHOT = ('i=$(mktemp -u) && cp "$(git rev-parse --git-path index)" "$i" && '
            'GIT_INDEX_FILE="$i" git add -A && GIT_INDEX_FILE="$i" git write-tree; rm -f "$i"')

GIT_IDENTITY = {"GIT_AUTHOR_NAME": "Simulated Engineer", "GIT_AUTHOR_EMAIL": "engineer@example.invalid",
                "GIT_COMMITTER_NAME": "Simulated Engineer", "GIT_COMMITTER_EMAIL": "engineer@example.invalid"}

LIVE: set = set()  # sandboxes to tear down on Ctrl-C
WRITE_LOCK = threading.Lock()


def run_sh(argv: list[str], cwd=None, env=None, timeout=None) -> tuple[int, str]:
    """Run, return (exit code, stdout+stderr); on timeout kill the whole process group."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return 124, out + f"\n[timed out after {timeout}s]"


# --------------------------------------------------------------------------- sandboxes

class LocalSandbox:
    """A clone outside any repo (so no parent CLAUDE.md leaks in), on this host. Not isolated."""

    def __init__(self, task: dict, sdir: Path, a):
        self.task, self.sdir = task, sdir
        self.workdir = Path(tempfile.mkdtemp(prefix="ccsim-", dir=a.workroot)) / "repo"
        self.config_dir = str(sdir / "cc")
        self.proc = None

    def start(self, env: dict) -> None:
        home = self.sdir / "home"
        home.mkdir(parents=True)
        keep = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR") if k in os.environ}
        self.env = {**keep, "HOME": str(home), **GIT_IDENTITY, **env}
        rc, out = run_sh(["git", "clone", "-q", self.task["repo"], str(self.workdir)])
        if rc:
            raise RuntimeError(f"git clone failed: {out.strip()}")

    def sh(self, cmd: str, timeout=None) -> tuple[int, str]:
        return run_sh(["sh", "-c", cmd], cwd=self.workdir, env={**os.environ, **GIT_IDENTITY}, timeout=timeout)

    def claude(self, argv: list[str], stderr) -> subprocess.Popen:
        self.proc = subprocess.Popen(["claude", *argv], cwd=self.workdir, env=self.env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        return self.proc

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        run_sh(["rm", "-rf", str(self.workdir.parent)])


class DockerSandbox:
    """One container per session from the task's image; Claude Code must be installed in it."""

    def __init__(self, task: dict, sdir: Path, a):
        self.task, self.sdir, self.network = task, sdir, a.network
        self.name = "ccsim-" + sdir.name.replace("__", "-")
        self.workdir = task.get("workdir", "/repo")
        self.config_dir = "/cc"
        self.proc = None

    def start(self, env: dict) -> None:
        argv = ["docker", "run", "-d", "--name", self.name, "-w", self.workdir, "--entrypoint", "tail"]
        if self.network:
            argv += ["--network", self.network]
        for k, v in {**GIT_IDENTITY, **env}.items():
            argv += ["-e", f"{k}={v}"]
        rc, out = run_sh(argv + [self.task["image"], "-f", "/dev/null"])
        if rc:
            raise RuntimeError(f"docker run failed: {out.strip()}")

    def sh(self, cmd: str, timeout=None) -> tuple[int, str]:
        return run_sh(["docker", "exec", self.name, "sh", "-c", cmd], timeout=timeout)

    def claude(self, argv: list[str], stderr) -> subprocess.Popen:
        self.proc = subprocess.Popen(["docker", "exec", "-i", self.name, "claude", *argv], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        return self.proc

    def stop(self) -> None:
        run_sh(["docker", "cp", f"{self.name}:{self.config_dir}", str(self.sdir / "cc")])
        run_sh(["docker", "rm", "-f", self.name])


# --------------------------------------------------------------------------- the agent

class Agent:
    """One `claude -p --input-format stream-json` process; every event is copied to stream.jsonl."""

    def __init__(self, proc: subprocess.Popen, log):
        self.proc, self.log = proc, log
        self.lines = queue.Queue()
        self.eof = False
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, text: str) -> None:
        self.proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n")
        self.proc.stdin.flush()

    def wait_result(self, timeout: float) -> tuple[dict | None, str, dict]:
        """Read events until this turn's `result`. Returns (result, why it stopped, counters)."""
        deadline = time.monotonic() + timeout
        calls, n = set(), {"api_retries": 0, "subagent_events": 0}
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None, "turn_timeout", n
            if line is None:
                self.eof = True
                return None, "claude_exited", n
            self.log.write(line)
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = (ev.get("type"), ev.get("subtype"))
            if kind[0] == "assistant":
                calls.add(ev.get("message", {}).get("id"))
                n["subagent_events"] += ev.get("parent_tool_use_id") is not None
            elif kind == ("system", "api_retry"):
                n["api_retries"] += 1
            elif kind == ("system", "init"):
                n["version"] = ev.get("claude_code_version")
                n["context_model"] = ev.get("model")
            elif kind[0] == "result":
                n["calls_seen"] = len(calls)
                return ev, "result", n

    def close(self, timeout: float = 60) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=timeout)  # let it flush the transcript
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        while not self.eof:
            line = self.lines.get()
            self.eof = line is None
            self.log.write(line or "")


# --------------------------------------------------------------------------- the engineer

class Engineer:
    """The simulated engineer: a prompt, a private conversation, one JSON decision per turn."""

    def __init__(self, a, intent: str):
        self.a = a
        self.system = ENGINEER_PROMPT.format(patience=a.patience, intent=intent)
        self.history: list[dict] = []

    def opener(self, given: str | None) -> dict:
        self.history.append({"role": "user", "content": "The session starts now. Write your first message to the agent."})
        d = {"action": "reply", "message": given} if given else self._ask()
        self.history.append({"role": "assistant", "content": json.dumps({k: d[k] for k in ("action", "message")})})
        return d

    def decide(self, shown: str) -> dict:
        self.history.append({"role": "user", "content": shown})
        d = self._ask()
        self.history.append({"role": "assistant", "content": json.dumps({k: d[k] for k in ("action", "message")})})
        return d

    def _ask(self) -> dict:
        err = None
        for _ in range(2):  # one retry on malformed output
            try:
                text = ENGINEER_BACKENDS[self.a.engineer](self.a, self.system, self.history)
                d = json.loads(text[text.find("{"): text.rfind("}") + 1])
                if d.get("action") in ("reply", "accept", "abandon") and (d["action"] != "reply" or d.get("message")):
                    return {"action": d["action"], "message": d.get("message") or ""}
                err = f"bad decision: {text[:200]}"
            except Exception as e:  # network, refusal, JSON: all end the session the same way
                err = f"{type(e).__name__}: {e}"
        return {"action": "abandon", "message": "", "error": err}


def engineer_openai(a, system: str, history: list[dict]) -> str:
    """Any OpenAI-compatible chat endpoint, e.g. a second vLLM."""
    body = {"model": a.engineer_model, "messages": [{"role": "system", "content": system}, *history],
            "max_tokens": 400, "temperature": a.engineer_temperature, "seed": a.engineer_seed,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(a.engineer_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": f"Bearer {os.environ.get(a.engineer_key_env, 'dummy')}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


_anthropic_client = None


def engineer_anthropic(a, system: str, history: list[dict]) -> str:
    """Claude via the official SDK. Sampling parameters are rejected by the API, so no temperature/seed."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # only this backend needs it: pip install anthropic
        # Explicit base_url: an exported ANTHROPIC_BASE_URL meant for Claude Code would
        # otherwise send the engineer to the server under test.
        _anthropic_client = anthropic.Anthropic(base_url=a.engineer_url or "https://api.anthropic.com",
                                                api_key=os.environ.get(a.engineer_key_env))
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if a.engineer_fallbacks else {}
    r = _anthropic_client.beta.messages.create(
        model=a.engineer_model, max_tokens=16000, system=system, messages=history,
        output_config={"effort": a.engineer_effort, "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
        **extra)
    if r.stop_reason == "refusal":
        raise RuntimeError("engineer model refused")
    return next(b.text for b in r.content if b.type == "text")


ENGINEER_BACKENDS = {"openai": engineer_openai, "anthropic": engineer_anthropic}


def think_time(a, rng: random.Random, reply: str, changed_lines: int, message: str) -> dict:
    """§4.3: modelled seconds between the agent's result and the engineer's next message."""
    t = {"read": len(reply.split()) / a.read_wps,
         "review": min(a.review_cap_s, changed_lines * a.review_s_per_line),
         "pause": rng.lognormvariate(math.log(a.pause_p50), math.log(a.pause_p90 / a.pause_p50) / Z90),
         "type": len(message.split()) / a.type_wps}
    t["total"] = sum(t.values())
    return {k: round(v, 2) for k, v in t.items()}


def render_shown(turn: int, max_turns: int, reply: str, diffstat: str, check: tuple | None) -> str:
    parts = [f"<agent_reply>\n{reply.strip() or '(no text)'}\n</agent_reply>",
             f"<diff_stat>\n{diffstat.strip() or 'no changes'}\n</diff_stat>"]
    if check:
        cmd, rc, tail = check
        parts.append(f"<check command={json.dumps(cmd)} result=\"{'pass' if rc == 0 else f'fail (exit {rc})'}\">\n"
                     f"{tail}\n</check>")
    parts.append(f"Agent turn {turn} of at most {max_turns}. Answer with the JSON object only.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- one session

def claude_env(a, config_dir: str) -> dict:
    env = {"ANTHROPIC_BASE_URL": a.base_url, "ANTHROPIC_API_KEY": "dummy", "ANTHROPIC_AUTH_TOKEN": "dummy",
           # ANTHROPIC_MODEL as well as the tier defaults: with only the defaults set,
           # Claude Code 2.1.308 runs "<model>[1m]", assumes a 1M window and ignores
           # CLAUDE_CODE_MAX_CONTEXT_TOKENS.
           "ANTHROPIC_MODEL": a.model, "ANTHROPIC_DEFAULT_OPUS_MODEL": a.model,
           "ANTHROPIC_DEFAULT_SONNET_MODEL": a.model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": a.model,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
           "CLAUDE_CONFIG_DIR": config_dir}
    for flag, var in (("max_context_tokens", "CLAUDE_CODE_MAX_CONTEXT_TOKENS"),
                      ("autocompact_window", "CLAUDE_CODE_AUTO_COMPACT_WINDOW"),
                      ("max_output_tokens", "CLAUDE_CODE_MAX_OUTPUT_TOKENS")):
        if getattr(a, flag):
            env[var] = str(getattr(a, flag))
    env.update(kv.split("=", 1) for kv in a.env)
    return env


def run_session(a, task: dict, out: Path) -> dict:
    sid = str(uuid.uuid4())
    sdir = out / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', task['id'])}__{sid[:8]}"
    sdir.mkdir(parents=True)
    rng = random.Random(f"{a.seed}:{task['id']}")
    max_turns = int(task.get("max_engineer_turns") or a.max_engineer_turns)
    s = {"task": task["id"], "category": task.get("category"), "session_id": sid, "dir": str(sdir),
         "outcome": None, "engineer_turns": 0, "agent_calls": 0, "api_retries": 0, "agent_s": 0.0,
         "api_s": 0.0, "think_modelled_s": 0.0, "think_slept_s": 0.0, "final_check_rc": None}
    sb = None
    t_start = time.time()
    try:
        sb = (DockerSandbox if a.sandbox == "docker" else LocalSandbox)(task, sdir, a)
        LIVE.add(sb)
        sb.start(claude_env(a, sb.config_dir))
        for cmd in filter(None, [f"git checkout -q {task['base_commit']}" if task.get("base_commit") else None,
                                 task.get("setup")]):
            rc, o = sb.sh(cmd, timeout=a.check_timeout)
            if rc:
                raise RuntimeError(f"`{cmd}` exited {rc}: {o.strip()[-300:]}")
        prev_tree = base_tree = sb.sh(SNAPSHOT)[1].strip()

        with open(sdir / "stream.jsonl", "w") as log, open(sdir / "engineer.jsonl", "w") as elog, \
                open(sdir / "claude.stderr", "w") as err:
            agent = Agent(sb.claude(["-p", "--input-format", "stream-json", "--output-format", "stream-json",
                                     "--verbose", "--dangerously-skip-permissions",
                                     "--disallowed-tools", "AskUserQuestion", "--session-id", sid,
                                     *a.claude_arg], err), log)
            eng = Engineer(a, task["intent"])
            d = eng.opener(task.get("opener"))
            if d.get("error"):
                raise RuntimeError(f"engineer: {d['error']}")
            agent.send(d["message"])
            api_ms_total = 0
            elog.write(json.dumps({"turn": 0, "t_sent": round(time.time(), 3), "action": "reply",
                                   "message": d["message"], "source": "task" if task.get("opener") else "engineer"}) + "\n")
            for turn in range(1, max_turns + 1):
                s["engineer_turns"] = turn
                res, why, n = agent.wait_result(a.turn_timeout)
                if res is None:
                    s["outcome"] = why
                    break
                t_result = time.time()
                for k in ("version", "context_model"):  # from the turn's init event
                    s[k] = n.get(k) or s.get(k)
                s["context_window"] = [m.get("contextWindow") for m in (res.get("modelUsage") or {}).values()]
                s["agent_calls"] += res.get("num_turns") or 0
                s["api_retries"] += n["api_retries"]
                s["agent_s"] += (res.get("duration_ms") or 0) / 1000
                api_ms = (res.get("duration_api_ms") or 0) - api_ms_total  # cumulative over the process
                api_ms_total += api_ms
                s["api_s"] = api_ms_total / 1000
                rec = {"turn": turn, "t_result": round(t_result, 3),
                       "agent": {"is_error": res.get("is_error"), "terminal_reason": res.get("terminal_reason"),
                                 "num_turns": res.get("num_turns"), "calls_seen": n.get("calls_seen"),
                                 "api_retries": n["api_retries"], "subagent_events": n["subagent_events"],
                                 "duration_ms": res.get("duration_ms"), "duration_api_ms": api_ms,
                                 "usage": {k: (res.get("usage") or {}).get(k) for k in
                                           ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                            "cache_creation_input_tokens")}}}
                if res.get("is_error"):  # retries exhausted, overflow, ...: an SLO failure, not a turn
                    elog.write(json.dumps(rec) + "\n")
                    s["outcome"] = f"agent_error: {res.get('terminal_reason')}: {(res.get('result') or '')[:120]}"
                    break

                tree = sb.sh(SNAPSHOT)[1].strip()
                diffstat = sb.sh(f"git diff --stat {base_tree} {tree}")[1]
                changed = sum(int(x) for line in sb.sh(f"git diff --numstat {prev_tree} {tree}")[1].splitlines()
                              for x in line.split("\t")[:2] if x.isdigit())
                prev_tree = tree
                check = None
                if task.get("check"):
                    rc, o = sb.sh(task["check"], timeout=a.check_timeout)
                    check = (task["check"], rc, "\n".join(o.rstrip().splitlines()[-a.check_tail:]))
                    s["final_check_rc"] = rc

                reply = res.get("result") or ""
                d = eng.decide(render_shown(turn, max_turns, reply, diffstat, check))
                think = think_time(a, rng, reply, changed, d["message"] if d["action"] == "reply" else "")
                rec.update(changed_lines=changed, check_rc=check[1] if check else None,
                           action=d["action"], message=d["message"], think=think)
                if d.get("error"):
                    rec["error"] = d["error"]
                if d["action"] != "reply" or turn == max_turns:
                    elog.write(json.dumps(rec) + "\n")
                    s["outcome"] = {"accept": "accepted", "abandon": "abandoned"}.get(d["action"], "max_turns")
                    if d.get("error"):
                        s["outcome"] = "engineer_error"
                    break
                sleep = think["total"] * a.time_scale
                time.sleep(sleep)
                s["think_modelled_s"] += think["total"]
                s["think_slept_s"] += sleep
                agent.send(d["message"])
                rec["t_sent"] = round(time.time(), 3)
                rec["recorded_gap_s"] = round(rec["t_sent"] - t_result, 3)  # §6.3 replaces this with think.total
                elog.write(json.dumps(rec) + "\n")
                elog.flush()
            agent.close()
    except Exception as e:
        s["outcome"] = f"error: {type(e).__name__}: {e}"
    finally:
        if sb:
            sb.stop()
            LIVE.discard(sb)
    s["wall_s"] = round(time.time() - t_start, 1)
    for k in ("agent_s", "api_s", "think_modelled_s", "think_slept_s"):
        s[k] = round(s[k], 1)
    return s


# --------------------------------------------------------------------------- main

def load_tasks(path: str) -> list[dict]:
    base = Path(path).resolve().parent
    tasks = []
    for line in open(path):
        if line.strip():
            t = json.loads(line)
            if "repo" in t and not re.match(r"^[\w+.-]+://|^[\w.-]+@", t["repo"]):
                t["repo"] = str((base / t["repo"]).resolve())  # local path, relative to the tasks file
            tasks.append(t)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", required=True, help="JSONL, one task per line (§3)")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--limit", type=int, help="run only the first N tasks")
    ap.add_argument("--concurrency", type=int, default=1, help="sessions at a time")
    ap.add_argument("--sandbox", choices=["local", "docker"], required=True)
    ap.add_argument("--workroot", help="local: where the repo clones go (default: system temp)")
    ap.add_argument("--network", help="docker: network that reaches only the proxy")
    g = ap.add_argument_group("Claude Code (§5)")
    g.add_argument("--base-url", required=True, help="the capture proxy, or the mock")
    g.add_argument("--model", required=True, help="vLLM --served-model-name")
    g.add_argument("--max-context-tokens", type=int, help="CLAUDE_CODE_MAX_CONTEXT_TOKENS, i.e. max_model_len")
    g.add_argument("--autocompact-window", type=int, help="CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    g.add_argument("--max-output-tokens", type=int, help="CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    g.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for Claude Code")
    g.add_argument("--claude-arg", action="append", default=[], help="extra CLI argument for claude")
    g.add_argument("--turn-timeout", type=float, default=3600, help="seconds to wait for one agent turn")
    g = ap.add_argument_group("engineer (§4)")
    g.add_argument("--engineer", choices=sorted(ENGINEER_BACKENDS), required=True)
    g.add_argument("--engineer-url", help="openai: base URL ending in /v1; anthropic: default api.anthropic.com")
    g.add_argument("--engineer-model", help="required for openai; anthropic default: claude-opus-5-5")
    g.add_argument("--engineer-key-env", default="ENGINEER_API_KEY", help="env var holding the engineer's API key")
    g.add_argument("--engineer-temperature", type=float, default=0.7, help="openai only")
    g.add_argument("--engineer-seed", type=int, default=1234, help="openai only")
    g.add_argument("--engineer-effort", default="medium", help="anthropic only")
    g.add_argument("--no-engineer-fallbacks", dest="engineer_fallbacks", action="store_false",
                   help="anthropic: don't send fallbacks=default (server-side retry of refusals)")
    g.add_argument("--patience", type=int, default=3)
    g.add_argument("--max-engineer-turns", type=int, default=6, help="when the task doesn't set it")
    g.add_argument("--check-timeout", type=float, default=900)
    g.add_argument("--check-tail", type=int, default=20, help="lines of check output the engineer sees")
    g = ap.add_argument_group("engineer timing (§4.3, illustrative defaults)")
    g.add_argument("--time-scale", type=float, default=0.0, help="fraction of modelled think time to sleep")
    g.add_argument("--read-wps", type=float, default=4.0)
    g.add_argument("--review-s-per-line", type=float, default=1.0)
    g.add_argument("--review-cap-s", type=float, default=120.0)
    g.add_argument("--pause-p50", type=float, default=60.0)
    g.add_argument("--pause-p90", type=float, default=600.0)
    g.add_argument("--type-wps", type=float, default=0.7)
    g.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.engineer == "openai" and not (a.engineer_url and a.engineer_model):
        ap.error("--engineer openai needs --engineer-url and --engineer-model")
    a.engineer_model = a.engineer_model or "claude-opus-5-5"

    tasks = load_tasks(a.tasks)[: a.limit]
    if a.sandbox == "local":
        print("WARNING: --sandbox local runs the agent on this host with all permissions.", file=sys.stderr)
        if a.workroot:
            Path(a.workroot).mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    print(f"{len(tasks)} tasks, {a.concurrency} at a time -> {out}", flush=True)

    results = []
    pool = ThreadPoolExecutor(max_workers=a.concurrency)
    try:
        futs = [pool.submit(run_session, a, t, out) for t in tasks]
        for f in as_completed(futs):
            s = f.result()
            results.append(s)
            with WRITE_LOCK, open(out / "sessions.jsonl", "a") as fh:
                fh.write(json.dumps(s) + "\n")
            print(f"[{len(results)}/{len(tasks)}] {s['task']:30s} {s['outcome']:14s} "
                  f"turns={s['engineer_turns']} calls={s['agent_calls']} retries={s['api_retries']} "
                  f"agent={s['agent_s']}s think={s['think_modelled_s']}s wall={s['wall_s']}s", flush=True)
    except KeyboardInterrupt:
        print("interrupted: stopping sandboxes", file=sys.stderr)
        for sb in list(LIVE):
            sb.stop()
        os._exit(130)  # workers may be mid-sleep or mid-request; don't wait for them
    pool.shutdown()

    done = [s for s in results if s["agent_calls"]]
    print(f"outcomes {dict(Counter(s['outcome'].split(':')[0] for s in results))}")
    if done:
        turns = sum(s["engineer_turns"] for s in done)
        print(f"calls/engineer turn {sum(s['agent_calls'] for s in done) / turns:.1f}  "
              f"agent s/turn {sum(s['agent_s'] for s in done) / turns:.1f}  "
              f"modelled think s/turn {sum(s['think_modelled_s'] for s in done) / max(1, turns - len(done)):.0f}  "
              f"api retries {sum(s['api_retries'] for s in done)}")


if __name__ == "__main__":
    main()
