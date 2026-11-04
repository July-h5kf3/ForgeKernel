#!/usr/bin/env python3
"""claude-goal.py — one agent turn as an interactive `claude` driven by `/goal`.

    python3 claude-goal.py <expected-model> <model-flag> [extra claude args...]
      prompt on stdin, reply on stdout — the contract of scripts/agent-claude.sh,
      which dispatches here directly when its caller asks for a goal session
      (FK_AGENT_DRIVER=goal; today the ref_from_goal session, loop/goal_ref.py).

What changes against `claude -p`: the turn is an interactive session in its own tmux
session, started with `/goal <condition>` as its first prompt. After every model turn
Claude Code's small model judges the condition against the transcript and, while it is
unmet, starts another turn. The condition (the caller's spec.json, or a generic one)
asks the agent to write the reply its prompt asks for into the turn directory's reply.md
and to run the caller's check; this script runs that check again itself before it hands
reply.md back, so a judge that was talked into "met" cannot deliver a reply the caller
would reject. The prompt itself goes in as an appended system prompt
(--append-system-prompt-file): a 30-110 kB FK prompt would not fit the 4000-character
condition, and the system prompt is also the one part of the context auto-compaction
never summarises away.

Exit codes: 0 with the reply on stdout; 75 MODEL MISMATCH, 76 MODEL UNVERIFIED, 77
SAFEGUARD REFUSAL (all with the reply on stdout); 124 when the wall-clock deadline
(FK_GOAL_DEADLINE, set by the caller) passes; 1 for everything else, with the CLI's own
error text (e.g. "You've hit your session limit · resets 7pm") on stdout AND stderr, so
the loop's outage patterns classify it exactly as they classify a `-p` failure.

Limits: a usage limit, or a rate limit that outlasts FK_GOAL_PAUSE_RETRIES nudges, ends
the turn with the limit text — the caller decides whether to wait or stop, as it does
for `-p`. The session id is kept in the turn directory, and the next
attempt OF THE SAME TURN (same FK_GOAL_TURN_DIR, same CLAUDE_CONFIG_DIR) resumes that
session instead of starting over, so a turn held through a 5-hour limit keeps its work.
Any other failure starts the next attempt fresh, as `-p` does.

Cleanup: every exit path (reply, failure, deadline, SIGTERM/SIGINT/SIGHUP) kills this
turn's tmux session by exact name (`-t =<name>`) and then the claude process by the PID
the launcher recorded, after checking that PID still runs `claude`. Neither the tmux
server nor any process found by pattern is ever touched. If this process dies by SIGKILL,
a watcher inside the tmux session ends claude within seconds.

Env knobs (all optional):
  FK_GOAL_TURN_DIR        the turn directory the caller prepared (spec.json); without
                          it a generic turn directory is created under $TMPDIR
  FK_GOAL_DEADLINE        absolute epoch seconds; FK_GOAL_TIMEOUT relative seconds
  FK_GOAL_POLL_S          poll interval (default 2)
  FK_GOAL_PAUSE_RETRIES   nudges for a paused goal before failing (default 4)
  FK_GOAL_RATE_WAIT_S     first wait before nudging a rate-limited pause (60, doubling)
  FK_GOAL_IDLE_S          a session idle this long with the goal unmet gets the goal
                          re-sent (default 2400 — beyond Claude Code's 30-min check-in)
  FK_GOAL_STARTUP_S       no goal sentinel in the transcript by then = failed start (180)
  FK_GOAL_SETTLE_S        transcript quiet time before a paused / cleared screen is acted
                          on (default 15)
  FK_GOAL_TMUX_SOCKET     tmux -L socket name (default: the default server)
  FK_GOAL_SESSION_PREFIX  tmux session name prefix (default fkturn)
  FK_GOAL_KEEP_SESSION    1 = leave the tmux session alive at the end (debugging only)
  FK_CLAUDE_BIN           the claude executable (default ~/.local/bin/claude)
  FK_FALLBACK_LOG         one line per notable event
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

OK_LINE = "TURN-DELIVERABLE-OK"
MISSING_LINE = "TURN-DELIVERABLE-MISSING"
CONDITION_MAX = 4000

# Variables that mark "running inside a Claude Code session"; a nested session must not
# inherit them (it would try to talk to the parent's IDE / SSE port).
NESTED_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
              "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_CHILD_SESSION", "TMUX", "TMUX_PANE")

# `-p`-only flags an agent command may carry; the interactive CLI rejects or
# misreads them. --no-session-persistence would also delete the transcript this
# script reads. (flag, takes a value)
PRINT_ONLY_FLAGS = {
    "-p": False, "--print": False, "--output-format": True, "--input-format": True,
    "--include-partial-messages": False, "--no-session-persistence": False,
    "--replay-user-messages": False, "--max-budget-usd": True, "--json-schema": True,
    "--permission-prompts": True, "--forward-subagent-text": False,
    "--include-hook-events": False, "--max-turns": True,
}

# --- what the screen says -------------------------------------------------------------
# Wording from the CLI (2.1.30x): "Goal paused · <cause> · send a message ...",
# "Goal paused after 3 automatic retries · <cause> · ...", "Goal still active · <cause> ·
# retrying in N min (k/3) ...", "Goal cleared after an unrecoverable error (<label>): ...".
RE_PAUSED = re.compile(r"Goal paused(?: after \d+ automatic retries)?\s*·\s*([^·\n]+)")
RE_STILL = re.compile(r"Goal still active\s*·")
RE_CLEARED = re.compile(r"Goal cleared after an unrecoverable error \(([^)]*)\)")
RE_TRUST = re.compile(r"Yes, I trust this folder|Do you trust the files|"
                      r"Is this a project you created or one you trust", re.I)
RE_BYPASS = re.compile(r"Bypass Permissions mode.*?Yes, I accept", re.S)
RE_LOGIN = re.compile(r"Select login method|Please run /login|not logged in", re.I)
RE_NO_GOAL = re.compile(r"/goal is (?:not available|unavailable)|disableAllHooks|"
                        r"allowManagedHooksOnly", re.I)
RE_USAGE_LIMIT = re.compile(
    r"hit your (?:session|usage|weekly|5-hour|7-day)? ?limit|usage limit reached|"
    r"weekly limit|out of (?:extra )?usage|requires usage credits", re.I)
RE_AUTH = re.compile(r"authentication_error|authentication failed|invalid x-api-key|"
                     r"Invalid API key|not logged in|OAuth token (?:has )?expired|"
                     r"please run /login|credit balance|organization verification|"
                     r"model unavailable|model_not_supported|requested model is not supported",
                     re.I)


def classify_pane(text: str) -> tuple[str, str]:
    """(state, detail) from the bottom-most goal status line on screen.

    States: "cleared", "paused", "still" (the CLI is retrying by itself), "trust",
    "bypass", "login", or "" when the screen says nothing actionable. Only the LAST status
    line counts: the scrollback keeps every earlier one, and a pause that a nudge already
    resolved must not be acted on twice.
    """
    tail = text[-12000:]
    if RE_BYPASS.search(tail[-4000:]):
        return "bypass", ""
    if RE_TRUST.search(tail[-4000:]):
        return "trust", ""
    best = ("", "", -1)
    for state, rx in (("paused", RE_PAUSED), ("still", RE_STILL), ("cleared", RE_CLEARED)):
        for m in rx.finditer(tail):
            if m.start() > best[2]:
                detail = m.group(1).strip() if m.groups() else ""
                best = (state, detail, m.start())
    for m in re.finditer(r"Goal (?:set|achieved|could not be achieved)", tail):
        if m.start() > best[2]:
            best = ("", "", m.start())       # a newer goal event supersedes an old pause
    if best[0]:
        return best[0], best[1]
    if RE_LOGIN.search(tail[-4000:]):
        return "login", ""
    return "", ""


# --- the transcript -------------------------------------------------------------------

class Transcript:
    """Incremental reader of one session's JSONL transcript."""

    def __init__(self, cfg: Path, sid: str):
        self.cfg, self.sid = cfg, sid
        self.path: Path | None = None
        self.offset = 0
        self.size = 0
        self.changed_at = time.time()
        self.goal_set = False
        self.met: dict | None = None
        self.failed: dict | None = None
        self.not_met = 0
        self.turn_ends = 0
        self.last_turn_end_at_line = -1
        self.lines = 0
        self.models: set[str] = set()
        self.usage: dict[str, dict] = {}     # message id -> usage (last record wins)
        self.api_errors: list[str] = []
        self.refusal: dict | None = None
        self.read_paths: list[str] = []
        self.last_kind = ""                  # "assistant" / "error" / "turn_end" / ...

    def locate(self) -> Path | None:
        if self.path is None:
            hits = sorted((self.cfg / "projects").glob(f"*/{self.sid}.jsonl"))
            self.path = hits[0] if hits else None
        return self.path

    def poll(self) -> bool:
        """Read new records; True when the file grew."""
        p = self.locate()
        if p is None:
            return False
        try:
            size = p.stat().st_size
        except OSError:
            return False
        if size == self.size:
            return False
        with open(p, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return False
        self.offset += cut + 1
        self.size = self.offset
        self.changed_at = time.time()
        for raw in chunk[:cut + 1].splitlines():
            try:
                self._record(json.loads(raw))
            except (ValueError, TypeError, AttributeError):
                continue
        return True

    def _record(self, d: dict) -> None:
        self.lines += 1
        t = d.get("type")
        att = d.get("attachment") if isinstance(d.get("attachment"), dict) else None
        if att and att.get("type") == "goal_status":
            if att.get("sentinel"):
                self.goal_set = True
                self.met = self.failed = None
            elif att.get("met"):
                self.met = att
            elif att.get("failed"):
                self.failed = att
            else:
                self.not_met += 1
            self.last_kind = "goal_status"
            return
        if t == "system":
            st = d.get("subtype")
            if st == "turn_duration":
                self.turn_ends += 1
                self.last_turn_end_at_line = self.lines
                self.last_kind = "turn_end"
            elif st == "model_refusal_fallback" and self.refusal is None:
                self.refusal = {"category": d.get("apiRefusalCategory") or "?",
                                "original": d.get("originalModel") or "?",
                                "fallback": d.get("fallbackModel") or "?",
                                "read": list(dict.fromkeys(self.read_paths[-4:]))}
            return
        if t != "assistant":
            if t == "user":
                self.last_kind = "user"
            return
        msg = d.get("message") or {}
        content = msg.get("content")
        if d.get("isApiErrorMessage"):
            text = ""
            if isinstance(content, list):
                text = " ".join(b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text")
            self.api_errors.append(text.strip() or str(d.get("error") or "API error"))
            self.last_kind = "error"
            return
        model = msg.get("model")
        if model and model != "<synthetic>":
            self.models.add(model)
        mid = msg.get("id")
        if mid and isinstance(msg.get("usage"), dict):
            self.usage[mid] = msg["usage"]
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    fp = (b.get("input") or {}).get("file_path")
                    if fp:
                        self.read_paths.append(str(fp))
        self.last_kind = "assistant"

    def tokens(self) -> dict:
        tot: dict[str, int] = {}
        for u in self.usage.values():
            for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                      "cache_read_input_tokens"):
                v = u.get(k)
                if isinstance(v, int):
                    tot[k] = tot.get(k, 0) + v
        tot["api_calls"] = len(self.usage)
        return tot


# --- tmux -------------------------------------------------------------------------------

class Tmux:
    def __init__(self, socket: str = ""):
        # absolute: the orphan guard inside the session runs it with the session's PATH
        self.base = [shutil.which("tmux") or "tmux"] + (["-L", socket] if socket else [])

    def run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE")}
        return subprocess.run(self.base + list(args), capture_output=True, text=True,
                              env=env, check=check)

    def exists(self, name: str) -> bool:
        # "=name" is an exact match; a bare name is a PREFIX match, and a watchdog that
        # used it once saw a dead session as alive because another one shared the prefix.
        return self.run("has-session", "-t", f"={name}").returncode == 0

    def start(self, name: str, command: str, cwd: str) -> None:
        # remain-on-exit: a claude that exits keeps its last screen and exit status for
        # the watcher (pane_dead) instead of taking the session with it. Set in the same
        # tmux invocation, before the server runs anything else.
        r = self.run("new-session", "-d", "-s", name, "-x", "220", "-y", "50", "-c", cwd,
                     command, ";", "set-option", "-w", "-t", f"={name}:", "remain-on-exit", "on")
        if r.returncode != 0:
            raise RuntimeError(f"tmux new-session {name}: {r.stderr.strip()}")

    def kill(self, name: str) -> None:
        if self.exists(name):
            self.run("kill-session", "-t", f"={name}")

    def capture(self, name: str, lines: int = 400) -> str:
        r = self.run("capture-pane", "-p", "-J", "-t", f"={name}:", "-S", f"-{lines}")
        return r.stdout if r.returncode == 0 else ""

    def pane_dead(self, name: str) -> tuple[bool, str]:
        r = self.run("display-message", "-p", "-t", f"={name}:",
                     "#{pane_dead} #{pane_dead_status}")
        if r.returncode != 0:
            return True, ""
        parts = (r.stdout.strip() + " ").split(" ", 1)
        return parts[0] == "1", parts[1].strip()

    def keys(self, name: str, *keys: str) -> None:
        self.run("send-keys", "-t", f"={name}:", *keys)

    def paste(self, name: str, text: str) -> None:
        """Type ``text`` as one bracketed paste and press Enter — only into an IDLE
        session: a busy one leaves the text in its input box."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write(text)
            path = f.name
        buf = f"fkgoal-{os.getpid()}"
        try:
            self.run("load-buffer", "-b", buf, path)
            self.run("paste-buffer", "-p", "-d", "-b", buf, "-t", f"={name}:")
            time.sleep(0.5)
            self.run("send-keys", "-t", f"={name}:", "Enter")
        finally:
            os.unlink(path)


# --- helpers ----------------------------------------------------------------------------

def say(msg: str, turn_dir: Path | None = None) -> None:
    line = f"{time.strftime('%F %T')} role={os.environ.get('ROLE', '?')} goal: {msg}"
    for p in filter(None, (os.environ.get("FK_FALLBACK_LOG"),
                           str(turn_dir / "leg.log") if turn_dir else None)):
        if p == "/dev/null":
            continue
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def interactive_args(extra: list[str]) -> tuple[list[str], list[str]]:
    """(args kept, args dropped): the leg's extra claude args minus the -p-only flags."""
    kept, dropped, i = [], [], 0
    while i < len(extra):
        a = extra[i]
        name = a.split("=", 1)[0]
        if name in PRINT_ONLY_FLAGS:
            takes = PRINT_ONLY_FLAGS[name] and "=" not in a
            dropped += extra[i:i + (2 if takes else 1)]
            i += 2 if takes else 1
            continue
        kept.append(a)
        i += 1
    return kept, dropped


def env_file_text(env: dict[str, str]) -> str:
    out = []
    for k, v in sorted(env.items()):
        if k in NESTED_ENV or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            continue
        out.append(f"export {k}={shlex.quote(v)}")
    return "\n".join(out) + "\n"


def config_paths(cfg_env: str | None) -> tuple[Path, Path]:
    """(the global-config JSON, the user settings file) Claude Code reads."""
    if cfg_env:
        return Path(cfg_env) / ".claude.json", Path(cfg_env) / "settings.json"
    return Path.home() / ".claude.json", Path.home() / ".claude" / "settings.json"


def _merge_json(path: Path, update) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return False            # unreadable: leave it alone, the screen fallback answers
    if not isinstance(data, dict):
        return False
    if not update(data):
        return False
    tmp = path.with_name(path.name + f".fkgoal-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def seed_config(cfg_env: str | None, cwd: Path, bypass: bool, version: str) -> list[str]:
    """Answer the interactive first-run screens before they are shown.

    A config directory that only ever ran `claude -p` has never seen the theme picker, the
    workspace-trust question or the bypass-permissions warning; an unattended session
    would sit on the first of them forever. Each key is ADDED only when missing (an
    operator's own choice is never overwritten), and the screen watcher still answers
    trust / bypass if a concurrent writer lost the update. Trust matters beyond the
    prompt: project settings hooks — the agent policy's PreToolUse hook — and /goal itself
    only run in a trusted folder.
    """
    glob_path, settings_path = config_paths(cfg_env)
    done = []
    key = str(cwd.resolve())

    def upd_global(d: dict) -> bool:
        changed = False
        if not d.get("hasCompletedOnboarding"):
            # the theme picker is the first onboarding screen; the CLI drops the key
            # from this file once onboarding is recorded, so it is only seeded with it
            d["hasCompletedOnboarding"] = True
            d.setdefault("lastOnboardingVersion", version or "2.1.0")
            d.setdefault("theme", "dark")
            changed = True
        proj = d.setdefault("projects", {})
        entry = proj.setdefault(key, {})
        if not entry.get("hasTrustDialogAccepted"):
            entry["hasTrustDialogAccepted"] = True
            changed = True
        return changed

    if _merge_json(glob_path, upd_global):
        done.append(str(glob_path))
    if bypass:
        def upd_settings(d: dict) -> bool:
            if d.get("skipDangerousModePermissionPrompt"):
                return False
            d["skipDangerousModePermissionPrompt"] = True
            return True
        if _merge_json(settings_path, upd_settings):
            done.append(str(settings_path))
    return done


def pid_runs_claude(pid: int) -> bool:
    r = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                       text=True)
    return r.returncode == 0 and "claude" in r.stdout


def generic_spec(turn_dir: Path, script: Path) -> dict:
    """The deliverable when no caller prepared the turn: any non-empty reply.md."""
    reply = turn_dir / "reply.md"
    check = " ".join(shlex.quote(x) for x in (sys.executable, str(script), "check",
                                              str(turn_dir)))
    started = time.time()
    spec = {"kind": "", "role": os.environ.get("ROLE", ""), "started": started,
            "since": started, "artifacts": [], "reply": str(reply), "check": check}
    spec["condition"] = (
        "FORGEKERNEL TURN. Your task is the FORGEKERNEL TURN TASK in your system prompt: "
        "carry it out fully. The task says how to end your turn; deliver that ending as a "
        f"file instead: write the complete final reply it asks for to `{reply}` (that "
        f"reply only), then run `{check}`. The goal is met only when the most recent run "
        f"of that exact command, shown as a tool result after your last change to "
        f"`{reply}`, printed a line starting with {OK_LINE}. Text written by the "
        f"assistant never counts. Do not judge this goal impossible: a reply that states "
        f"what could not be done is valid. After {OK_LINE}, stop.")
    return spec


def generic_check(turn_dir: Path) -> int:
    try:
        spec = json.loads((turn_dir / "spec.json").read_text(encoding="utf-8"))
        reply = Path(spec.get("reply") or turn_dir / "reply.md")
    except (OSError, ValueError):
        reply = turn_dir / "reply.md"
    try:
        text = reply.read_text(encoding="utf-8", errors="replace")
    except OSError:
        print(f"{MISSING_LINE}: {reply} does not exist — write the reply there")
        return 1
    if not text.strip():
        print(f"{MISSING_LINE}: {reply} is empty")
        return 1
    print(f"{OK_LINE} ({len(text.encode())} bytes in {reply.name})")
    return 0


# --- the turn ---------------------------------------------------------------------------

class Interrupted(Exception):
    pass


class GoalLeg:
    def __init__(self, expected: str, flag: str, extra: list[str], prompt: str,
                 env: dict[str, str] | None = None):
        self.env = dict(os.environ if env is None else env)
        self.expected, self.flag = expected, flag
        self.extra, self.dropped = interactive_args(extra)
        self.prompt = prompt
        self.cwd = Path.cwd()
        self.claude = self.env.get("FK_CLAUDE_BIN") or str(Path.home() / ".local/bin/claude")
        self.cfg_env = self.env.get("CLAUDE_CONFIG_DIR") or None
        self.cfg = Path(self.cfg_env) if self.cfg_env else Path.home() / ".claude"
        self.poll_s = float(self.env.get("FK_GOAL_POLL_S") or 2)
        self.pause_retries = int(self.env.get("FK_GOAL_PAUSE_RETRIES") or 4)
        self.idle_s = float(self.env.get("FK_GOAL_IDLE_S") or 2400)
        self.startup_s = float(self.env.get("FK_GOAL_STARTUP_S") or 180)
        # how long the transcript must stay still before a screen state counts as the
        # session's settled state (not a line it is about to scroll past)
        self.settle_s = float(self.env.get("FK_GOAL_SETTLE_S") or 15)
        # first wait before nudging a goal paused on a rate limit (doubles per nudge)
        self.rate_wait_s = float(self.env.get("FK_GOAL_RATE_WAIT_S") or 60)
        self.tmux = Tmux(self.env.get("FK_GOAL_TMUX_SOCKET", ""))
        self.keep_session = self.env.get("FK_GOAL_KEEP_SESSION") == "1"
        self.started = time.time()
        self.deadline = self._deadline()
        self.session = ""
        self.sid = ""
        self.resumed = False
        self.resume_offset = 0
        self.turn_dir: Path | None = None
        self.rundir: Path | None = None
        self.claude_pid: int | None = None
        self.last_pane = ""
        self.nudges = 0
        self.events: list[str] = []

    def _deadline(self) -> float | None:
        cands = []
        if self.env.get("FK_GOAL_DEADLINE"):
            cands.append(float(self.env["FK_GOAL_DEADLINE"]) - 15)
        if self.env.get("FK_GOAL_TIMEOUT"):
            cands.append(self.started + float(self.env["FK_GOAL_TIMEOUT"]))
        return min(cands) if cands else None

    # -- turn directory ------------------------------------------------------------------

    def setup_turn(self) -> None:
        script = Path(__file__).resolve()
        td = self.env.get("FK_GOAL_TURN_DIR")
        if td:
            self.turn_dir = Path(td)
            self.spec = json.loads((self.turn_dir / "spec.json").read_text(encoding="utf-8"))
        else:
            self.turn_dir = Path(tempfile.mkdtemp(prefix="fk-goal-turn-"))
            self.spec = generic_spec(self.turn_dir, script)
            (self.turn_dir / "spec.json").write_text(json.dumps(self.spec, indent=1) + "\n",
                                                     encoding="utf-8")
        cond = self.spec["condition"]
        if len(cond) > CONDITION_MAX:
            raise SystemExit(f"claude-goal: condition is {len(cond)} chars (> {CONDITION_MAX})")
        self.reply = Path(self.spec.get("reply") or self.turn_dir / "reply.md")
        (self.turn_dir / "prompt.md").write_text(
            "# FORGEKERNEL TURN TASK\n\n" + self.prompt, encoding="utf-8")

    def resumable_session(self) -> str | None:
        """The session a previous attempt of THIS turn left resumable, if its transcript is
        under this attempt's config directory (another leg's account has its own)."""
        try:
            st = json.loads((self.turn_dir / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not st.get("resumable") or not st.get("sid"):
            return None
        if Path(st.get("cfg", "")) != self.cfg:
            return None
        if not Transcript(self.cfg, st["sid"]).locate():
            return None
        return st["sid"]

    def save_session(self, resumable: bool, why: str) -> None:
        try:
            (self.turn_dir / "session.json").write_text(json.dumps(
                {"sid": self.sid, "cfg": str(self.cfg), "resumable": resumable,
                 "why": why, "at": time.time()}) + "\n", encoding="utf-8")
        except OSError:
            pass

    # -- launch --------------------------------------------------------------------------

    def cli_version(self) -> str:
        try:
            r = subprocess.run([self.claude, "--version"], capture_output=True, text=True,
                               timeout=30)
            m = re.search(r"\d+\.\d+\.\d+", r.stdout)
            return m.group(0) if m else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def launch(self) -> None:
        resume = self.resumable_session()
        self.resumed = bool(resume)
        if resume:
            p = Transcript(self.cfg, resume).locate()
            self.resume_offset = p.stat().st_size if p else 0
        self.sid = resume or str(uuid.uuid4())
        if not resume:
            try:
                self.reply.unlink()      # a fresh session owes a fresh reply
            except OSError:
                pass
        bypass = "--dangerously-skip-permissions" in self.extra
        seeded = seed_config(self.cfg_env, self.cwd, bypass, self.cli_version())
        if seeded:
            say(f"seeded first-run answers in {', '.join(seeded)}", self.turn_dir)
        if self.dropped:
            say(f"dropped -p-only args {self.dropped}", self.turn_dir)
        argv = [self.claude, "--model", self.flag, *self.extra]
        argv += ["--resume", self.sid] if resume else ["--session-id", self.sid]
        argv += ["--append-system-prompt-file", str(self.turn_dir / "prompt.md"),
                 "/goal " + self.spec["condition"]]
        self.rundir = Path(tempfile.mkdtemp(prefix="fk-goal-run-"))
        os.chmod(self.rundir, 0o700)
        envf = self.rundir / "env.sh"
        env = {k: v for k, v in self.env.items() if k not in NESTED_ENV}
        envf.write_text(env_file_text(env), encoding="utf-8")
        os.chmod(envf, 0o600)
        pidf = self.rundir / "claude.pid"
        launcher = self.rundir / "launch.sh"
        me = os.getpid()
        role = re.sub(r"[^A-Za-z0-9]", "", self.env.get("ROLE", "") or self.spec.get("role", ""))
        prefix = self.env.get("FK_GOAL_SESSION_PREFIX", "fkturn")
        self.session = f"{prefix}-{role or 'agent'}-{self.sid[:8]}-{os.getpid()}"
        kill_session = " ".join(shlex.quote(a) for a in (
            *self.tmux.base, "kill-session", "-t", f"={self.session}"))
        launcher.write_text(
            "#!/bin/bash\n"
            "# Started under `env -i`: nothing from the tmux server's environment leaks in.\n"
            f". {shlex.quote(str(envf))}\nrm -f {shlex.quote(str(envf))}\n"
            f"export TERM={shlex.quote(self.env.get('FK_GOAL_TERM', 'xterm-256color'))}\n"
            f"cd {shlex.quote(str(self.cwd))} || exit 97\n"
            f"echo $$ > {shlex.quote(str(pidf))}\n"
            "# Orphan guard: when the leg that owns this session dies (even by SIGKILL),\n"
            "# end claude (this PID after exec) within seconds, then this session — by its\n"
            "# exact name; remain-on-exit would otherwise keep the dead pane around.\n"
            "# It ignores HUP/TERM: claude's exit hangs up the pane, and the guard must\n"
            "# outlive that to remove the session.\n"
            f"( trap '' HUP TERM INT\n"
            f"  while kill -0 {me} 2>/dev/null && kill -0 $$ 2>/dev/null; do sleep 3; done\n"
            f"  kill -0 {me} 2>/dev/null && exit 0\n"
            "  if kill -0 $$ 2>/dev/null; then kill -TERM $$ 2>/dev/null; sleep 5;"
            " kill -KILL $$ 2>/dev/null; fi\n"
            f"  {kill_session} ) >/dev/null 2>&1 &\n"
            "exec " + " ".join(shlex.quote(a) for a in argv) + "\n",
            encoding="utf-8")
        self.tmux.start(self.session, f"env -i /bin/bash {shlex.quote(str(launcher))}",
                        str(self.cwd))
        self.save_session(False, "running")
        say(f"session {self.session} ({'resumed' if resume else 'new'} sid {self.sid}, "
            f"model {self.flag}, cwd {self.cwd}, turn {self.turn_dir.name})", self.turn_dir)
        for _ in range(50):
            try:
                self.claude_pid = int(pidf.read_text().strip())
                break
            except (OSError, ValueError):
                time.sleep(0.1)

    # -- cleanup -------------------------------------------------------------------------

    def cleanup(self) -> None:
        if self.session and not self.keep_session:
            self.tmux.kill(self.session)
            pid = self.claude_pid
            if pid:
                for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 0.0)):
                    if not pid_runs_claude(pid):
                        break
                    try:
                        os.kill(pid, sig)
                    except OSError:
                        break
                    end = time.time() + wait
                    while time.time() < end and pid_runs_claude(pid):
                        time.sleep(0.2)
        if self.rundir:
            for p in self.rundir.iterdir() if self.rundir.exists() else []:
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                self.rundir.rmdir()
            except OSError:
                pass

    # -- the watch loop ------------------------------------------------------------------

    def deliverable(self) -> tuple[bool, str]:
        r = subprocess.run(self.spec["check"], shell=True, capture_output=True, text=True,
                           cwd=str(self.cwd), timeout=300)
        out = (r.stdout + r.stderr).strip()
        return (r.returncode == 0 and OK_LINE in out), out

    def resend_goal(self, why: str) -> None:
        self.nudges += 1
        self.events.append(why)
        say(f"re-sending /goal ({why}; nudge {self.nudges})", self.turn_dir)
        self.tmux.paste(self.session, "/goal " + self.spec["condition"])

    def watch(self, tr: Transcript) -> tuple[str, str]:
        """Run until the turn resolves; returns (outcome, detail)."""
        pause_nudges = 0
        false_met = 0
        acted_at_size = -1
        acted_at = 0.0
        last_pane_at = 0.0
        met_seen_at = 0.0
        checked_at_size = -1
        while True:
            now = time.time()
            if self.deadline is not None and now >= self.deadline:
                ok, out = self.deliverable()
                if ok:          # finished, the judge just had not said so yet
                    return "met", out
                return "timeout", f"wall-clock deadline reached after {now - self.started:.0f}s"
            tr.poll()
            alive = self.tmux.exists(self.session)
            dead, status = self.tmux.pane_dead(self.session) if alive else (True, "")
            if now - last_pane_at >= 3 or dead:
                pane = self.tmux.capture(self.session, 2000) if alive else ""
                if pane:
                    self.last_pane = pane
                last_pane_at = now
            else:
                pane = ""
            quiet = now - tr.changed_at

            if tr.refusal:
                return "refusal", json.dumps(tr.refusal)
            if tr.met is not None:
                met_seen_at = met_seen_at or now
                # the turn that the judge closed finishes its bookkeeping first
                if tr.last_kind == "turn_end" or quiet > self.settle_s or dead:
                    ok, out = self.deliverable()
                    if ok:
                        return "met", out
                    false_met += 1
                    if false_met > 2 or dead:
                        return "fail", ("the goal was judged met but the deliverable check "
                                        f"still fails: {out}")
                    tr.met = None
                    met_seen_at = 0.0
                    self.resend_goal(f"judged met but the check fails: {out[:160]}")
                    acted_at_size, acted_at = tr.size, now
                    continue
            if tr.failed is not None and (tr.last_kind == "turn_end" or quiet > self.settle_s or dead):
                ok, out = self.deliverable()
                if ok:
                    return "met", out
                return "fail", f"goal judged impossible: {tr.failed.get('reason', '')[:600]}"
            if dead:
                ok, out = self.deliverable()
                if ok:
                    return "met", out
                return "fail", f"claude exited (status {status or '?'}) before the deliverable"
            if not alive:
                ok, out = self.deliverable()
                if ok:
                    return "met", out
                return "fail", "the tmux session disappeared before the deliverable"

            state, detail = classify_pane(pane) if pane else ("", "")
            if tr.goal_set and state in ("bypass", "trust", "login"):
                # the first-run screens come before the goal; once it is set, these words
                # on screen are the agent's own output
                state, detail = "", ""
            if state == "bypass":
                self.tmux.keys(self.session, "Down")
                time.sleep(0.3)
                self.tmux.keys(self.session, "Enter")
                say("answered the bypass-permissions warning (accept)", self.turn_dir)
                time.sleep(2)
                continue
            if state == "trust":
                self.tmux.keys(self.session, "Enter")
                say("answered the workspace-trust question (trust)", self.turn_dir)
                time.sleep(2)
                continue
            if state == "login":
                return "fail", "not logged in — please run /login (the session shows the login screen)"
            if not tr.goal_set and now - self.started > self.startup_s:
                if pane and RE_NO_GOAL.search(pane):
                    return "fail", "/goal is unavailable in this session: " + _tail(pane, 600)
                return "fail", ("the session never set the goal within "
                                f"{self.startup_s:.0f}s: " + _tail(self.last_pane, 600))

            settled = quiet > self.settle_s and (tr.size != acted_at_size or now - acted_at > 600)
            if state == "cleared" and settled:
                err = tr.api_errors[-1] if tr.api_errors else ""
                return "cleared", f"Goal cleared after an unrecoverable error ({detail}). {err}"
            if state in ("paused", "cleared") or (tr.goal_set and quiet > self.idle_s):
                # a stopped session whose deliverable is complete is a delivered turn:
                # the check is the authority, the judge only keeps the agent going
                if checked_at_size != tr.size:
                    checked_at_size = tr.size
                    ok, out = self.deliverable()
                    if ok:
                        return "met", out
            if state == "paused" and settled:
                err = tr.api_errors[-1] if tr.api_errors else ""
                if "usage limit" in detail or RE_USAGE_LIMIT.search(err):
                    return "limit", err or f"usage limit reached ({detail})"
                if pause_nudges >= self.pause_retries:
                    return "paused", f"Goal paused · {detail}. {err}"
                wait = (min(900.0, self.rate_wait_s * 2 ** pause_nudges)
                        if "rate limit" in detail else 10.0)
                if now - max(acted_at, tr.changed_at) < wait:
                    time.sleep(self.poll_s)
                    continue
                pause_nudges += 1
                self.resend_goal(f"paused: {detail}")
                acted_at_size, acted_at = tr.size, time.time()
                continue
            if (state in ("", None) and tr.goal_set and tr.last_kind in ("turn_end", "error")
                    and quiet > self.idle_s and now - acted_at > self.idle_s):
                if self.nudges >= 6:
                    return "fail", f"the session sat idle with the goal unmet ({self.nudges} nudges)"
                self.resend_goal(f"idle {quiet:.0f}s with the goal unmet")
                acted_at_size, acted_at = tr.size, time.time()
                continue
            time.sleep(self.poll_s)

    # -- verdict on the served model --------------------------------------------------

    def model_verdict(self, tr: Transcript) -> tuple[int, str]:
        if tr.locate() is None:
            return 76, (f"claude-goal: no transcript for session {self.sid} under "
                        f"{self.cfg}/projects — cannot verify the served model")
        if tr.refusal:
            r = tr.refusal
            return 77, (f"claude-goal: SAFEGUARD REFUSAL (category={r['category']}) — "
                        f"{r['original']} refused this turn and the CLI switched the session to "
                        f"{r['fallback']}. This is NOT a channel mix-up: the refusal belongs to "
                        f"the input. Files this turn had read: {' '.join(r['read']) or 'none'}. "
                        f"(transcript {tr.path})")
        expected = {self.expected, self.expected + "[1m]"}
        foreign = sorted(m for m in tr.models if m not in expected)
        if foreign or not (tr.models & expected):
            return 75, (f"claude-goal: MODEL MISMATCH — served [{','.join(sorted(tr.models))}], "
                        f"expected {self.expected} (transcript {tr.path})")
        return 0, ""

    def write_result(self, tr: Transcript, outcome: str, detail: str, rc: int) -> None:
        met = tr.met or {}
        res = {"outcome": outcome, "rc": rc, "detail": detail[:2000], "sid": self.sid,
               "session": self.session, "wall_s": round(time.time() - self.started, 1),
               "goal_iterations": met.get("iterations"), "goal_tokens": met.get("tokens"),
               "goal_duration_ms": met.get("durationMs"), "not_met_verdicts": tr.not_met,
               "turn_ends": tr.turn_ends, "nudges": self.nudges, "events": self.events,
               "models": sorted(tr.models), "usage": tr.tokens(),
               "api_errors": tr.api_errors[-5:], "transcript": str(tr.path or "")}
        try:
            with open(self.turn_dir / "result.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
        except OSError:
            pass

    def run(self) -> int:
        self.setup_turn()
        try:
            self.launch()
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            self.cleanup()
            msg = f"claude-goal: could not start the session: {e}"
            say(msg, self.turn_dir)
            sys.stderr.write(msg + "\n")
            return 1
        tr = Transcript(self.cfg, self.sid)
        if self.resumed and tr.locate():
            # a resumed session: only what this attempt adds counts (the old records
            # end on the very limit this attempt is retrying); measured before the launch
            tr.offset = tr.size = self.resume_offset
        outcome, detail = "fail", ""
        rc = 1
        try:
            outcome, detail = self.watch(tr)
        except Interrupted:
            outcome, detail = "interrupted", "leg interrupted by a signal"
            rc = 143
        finally:
            tr.poll()
            self.cleanup()
        resumable = outcome in ("limit", "paused")
        self.save_session(resumable, outcome)
        if outcome == "met":
            reply = self.reply.read_text(encoding="utf-8", errors="replace")
            rc, why = self.model_verdict(tr)
            self.write_result(tr, outcome, detail, rc)
            sys.stdout.write(reply)
            if rc:
                sys.stderr.write(why + "\n")
                say(why, self.turn_dir)
            else:
                met = tr.met or {}
                say(f"delivered ({len(reply)} chars; {met.get('iterations', '?')} goal "
                    f"turn(s), {time.time() - self.started:.0f}s, {self.nudges} nudge(s))",
                    self.turn_dir)
            return rc
        if outcome == "refusal":
            rc, why = self.model_verdict(tr)
            self.write_result(tr, outcome, detail, rc)
            sys.stderr.write(why + "\n")
            say(why, self.turn_dir)
            return rc or 77
        if outcome == "timeout":
            rc = 124
        msg = f"claude-goal: {outcome}: {detail}".strip()
        self.write_result(tr, outcome, detail, rc)
        say(msg[:400], self.turn_dir)
        # The failure text goes to BOTH streams: the loop's outage patterns read the tails
        # of stdout and stderr.
        if outcome in ("limit", "cleared", "paused", "fail"):
            sys.stdout.write(detail.strip() + "\n")
        sys.stderr.write(msg + "\n")
        return rc


def _tail(text: str, n: int) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    return " | ".join(lines)[-n:]


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "check":
        if len(argv) != 2:
            sys.stderr.write("usage: claude-goal.py check <turn-dir>\n")
            return 2
        return generic_check(Path(argv[1]))
    if len(argv) < 2:
        sys.stderr.write("usage: claude-goal.py <expected-model> <model-flag> [claude args...]\n")
        return 2
    prompt = sys.stdin.read()

    def on_signal(signum, frame):
        raise Interrupted()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    leg = GoalLeg(argv[0], argv[1], argv[2:], prompt)
    try:
        return leg.run()
    except Interrupted:
        leg.cleanup()
        return 143


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
