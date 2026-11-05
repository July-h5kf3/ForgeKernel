"""scripts/claude-goal.py — one agent turn as an interactive `claude` driven by /goal.

The leg runs for real here: a real tmux (on a private socket, so no test can touch the
operator's sessions or server) and a stub `claude` (tests/fixtures/stub-claude.py) that
writes the transcript records and screen lines the real CLI writes. What is pinned:

- the contract scripts/agent-claude.sh has: prompt in, reply on stdout, rc 0 / 75 / 1 / 124,
  and the CLI's own limit / auth text on stdout+stderr for the layers above to classify;
- the deliverable is checked by the leg itself, so a judge that called the goal met
  without the reply gets the goal re-sent, not a turn ended on nothing;
- a paused goal is nudged, a usage limit ends the leg resumably and the next attempt of
  the same turn resumes the session;
- every exit path (reply, failure, deadline, SIGTERM) leaves no tmux session and no
  claude process behind.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LEG = _ROOT / "scripts" / "claude-goal.py"
_STUB = _ROOT / "tests" / "fixtures" / "stub-claude.py"

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="needs tmux")


def _leg_module():
    spec = importlib.util.spec_from_file_location("claude_goal", _LEG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The leg ships stdlib-only for the system python3 (3.9 on the macOS hosts); run it there
# when there is one, so a 3.10-only construct fails here and not in a run.
_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable


def _socket() -> str:
    return f"fkgoal-test-{os.getpid()}"


def _sessions() -> list[str]:
    r = subprocess.run(["tmux", "-L", _socket(), "ls", "-F", "#{session_name}"],
                       capture_output=True, text=True)
    return [s for s in r.stdout.split() if s] if r.returncode == 0 else []


def _stub_procs(tag: str) -> list[str]:
    r = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if tag in ln and "stub-claude" in ln]


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "bin" / "stub-claude"
    stub.parent.mkdir()
    shutil.copy(_STUB, stub)
    stub.chmod(0o755)
    ws = tmp_path / "ws"
    ws.mkdir()
    tag = f"tag{os.getpid()}x{time.time_ns() % 10**8}"
    e = {k: v for k, v in os.environ.items()
         if k not in ("TMUX", "TMUX_PANE", "FK_GOAL_TURN_DIR", "FK_GOAL_DEADLINE", "FK_GOAL_TIMEOUT")}
    e.update(FK_CLAUDE_BIN=str(stub), CLAUDE_CONFIG_DIR=str(tmp_path / "cfg"),
             FK_GOAL_TMUX_SOCKET=_socket(), FK_GOAL_POLL_S="0.2", FK_GOAL_SETTLE_S="0.6",
             FK_GOAL_STARTUP_S="20", FK_GOAL_TIMEOUT="60", ROLE="student",
             STUB_SERVED_MODEL="claude-test-1", STUB_ARGV_LOG=str(tmp_path / "argv.log"),
             STUB_LINES_LOG=str(tmp_path / "lines.log"), FK_FALLBACK_LOG=str(tmp_path / "fb.log"),
             STUB_TAG=tag)
    (tmp_path / "cfg").mkdir()
    yield {"env": e, "ws": ws, "tmp": tmp_path, "tag": tag}
    assert _sessions() == [], "a test left a tmux session behind"


def _run(ctx, mode="ok", *, expected="claude-test-1", extra=(), prompt="do the thing", **kw):
    e = dict(ctx["env"], STUB_MODE=mode, **kw)
    return subprocess.run([_PY, str(_LEG), expected, "test-model", "--dangerously-skip-permissions",
                           *extra], input=prompt, capture_output=True, text=True, cwd=ctx["ws"],
                          env=e, timeout=120)


# A caller's own deliverable check (loop/goal_ref.py brings one): this one accepts a reply
# with a record line or an EXHAUSTED ending, the way a reply parser would.
_CALLER_CHECK = """\
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text() if Path(sys.argv[1]).is_file() else ""
if any(ln.startswith(("decision_id:", "EXHAUSTED:")) for ln in text.splitlines()):
    print("TURN-DELIVERABLE-OK")
else:
    print("TURN-DELIVERABLE-MISSING: no decision_id: / EXHAUSTED: line")
    sys.exit(1)
"""


def _turn_dir(ctx):
    """A caller-prepared turn: spec.json with the caller's condition and check."""
    td = ctx["tmp"] / "turn"
    td.mkdir()
    (td / "check.py").write_text(_CALLER_CHECK)
    reply = td / "reply.md"
    check = f"{_PY} {td / 'check.py'} {reply}"
    spec = {"kind": "caller", "role": "student", "reply": str(reply), "check": check,
            "condition": (f"FORGEKERNEL TURN. Write the reply to `{reply}`, then run "
                          f"`{check}`; met when it printed TURN-DELIVERABLE-OK.")}
    (td / "spec.json").write_text(json.dumps(spec))
    return td


def test_a_met_goal_hands_back_the_reply_and_leaves_nothing_running(env):
    r = _run(env, STUB_REPLY="EXHAUSTED: nothing left\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "EXHAUSTED: nothing left\n"
    assert r.stderr == ""                     # the loop scans stderr for failure text
    assert _sessions() == [] and _stub_procs(env["tag"]) == []
    argv = json.loads((env["tmp"] / "argv.log").read_text().splitlines()[0])
    assert argv[-1].startswith("/goal FORGEKERNEL TURN")       # /goal is the first prompt
    assert "--append-system-prompt-file" in argv and "--session-id" in argv
    prompt_file = Path(argv[argv.index("--append-system-prompt-file") + 1])
    assert prompt_file.read_text().endswith("do the thing")


def test_a_caller_prepared_turn_is_checked_with_the_callers_check(env):
    td = _turn_dir(env)
    r = _run(env, FK_GOAL_TURN_DIR=str(td), STUB_REPLY="decision_id: D1.1\nchange: x\nfiles_touched: k.py\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("decision_id: D1.1")
    res = json.loads((td / "result.json").read_text().splitlines()[-1])
    assert res["outcome"] == "met" and res["goal_iterations"] == 1
    assert res["usage"]["output_tokens"] == 20 and res["models"] == ["claude-test-1"]


def test_a_reply_the_callers_check_rejects_is_not_delivered(env):
    """The judge said met, but the reply has no record line: the goal is re-sent (the
    stub then delivers the same unparseable reply), and after the bounded retries the leg
    fails instead of handing the caller a reply it would only reject."""
    td = _turn_dir(env)
    r = _run(env, "false_met", FK_GOAL_TURN_DIR=str(td), STUB_REPLY="prose only, no marker\n")
    assert r.returncode == 1
    assert "check still fails" in r.stdout + r.stderr
    assert "TURN-DELIVERABLE-MISSING" in r.stdout


def test_a_judge_fooled_without_a_reply_gets_the_goal_resent(env):
    r = _run(env, "false_met")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "EXHAUSTED: stub reply\n"
    assert "/goal FORGEKERNEL" in (env["tmp"] / "lines.log").read_text()


def test_a_served_model_mismatch_exits_75_with_the_reply(env):
    r = _run(env, "mismatch", expected="claude-opus-5")
    assert r.returncode == 75
    assert r.stdout == "EXHAUSTED: stub reply\n"
    assert "MODEL MISMATCH" in r.stderr and "claude-test-1" in r.stderr
    assert _sessions() == []


def test_a_usage_limit_ends_the_leg_with_the_cli_text_and_the_next_attempt_resumes(env):
    td = _turn_dir(env)
    r = _run(env, "resume_ok", FK_GOAL_TURN_DIR=str(td))
    assert r.returncode == 1
    # the CLI's own words, on both streams: FK_FAIL_RE / the hold's patterns read them
    assert "You've hit your session limit" in r.stdout
    assert "hit your session limit" in r.stderr
    st = json.loads((td / "session.json").read_text())
    assert st["resumable"] is True
    r2 = _run(env, "resume_ok", FK_GOAL_TURN_DIR=str(td))
    assert r2.returncode == 0, r2.stderr
    argvs = [json.loads(x) for x in (env["tmp"] / "argv.log").read_text().splitlines()]
    assert "--resume" in argvs[1] and argvs[1][argvs[1].index("--resume") + 1] == st["sid"]


def test_a_fresh_attempt_does_not_resume_after_a_non_limit_failure(env):
    td = _turn_dir(env)
    assert _run(env, "cleared", FK_GOAL_TURN_DIR=str(td)).returncode == 1
    assert _run(env, "ok", FK_GOAL_TURN_DIR=str(td)).returncode == 0
    argvs = [json.loads(x) for x in (env["tmp"] / "argv.log").read_text().splitlines()]
    assert "--resume" not in argvs[1] and "--session-id" in argvs[1]


def test_an_unrecoverable_clear_reports_the_auth_text(env):
    r = _run(env, "cleared")
    assert r.returncode == 1
    assert "authentication failed" in r.stdout and "Please run /login" in r.stdout


def test_a_rate_limit_pause_is_nudged_and_the_turn_completes(env):
    # a rate-limited pause waits 60 s (doubling) before its nudge; the knob shortens it
    r = _run(env, "pause_then_ok", FK_GOAL_RATE_WAIT_S="0.5")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "EXHAUSTED: stub reply\n"
    assert "/goal FORGEKERNEL" in (env["tmp"] / "lines.log").read_text()


def test_the_deadline_kills_the_session_and_exits_124(env):
    t0 = time.time()
    r = _run(env, "hang", FK_GOAL_TIMEOUT="3")
    assert r.returncode == 124
    assert time.time() - t0 < 30
    assert "deadline" in r.stderr
    assert _sessions() == [] and _stub_procs(env["tag"]) == []


def test_a_claude_that_exits_without_a_reply_fails_the_leg(env):
    r = _run(env, "exit")
    assert r.returncode == 1
    assert "claude exited" in r.stdout + r.stderr
    assert _sessions() == []


def test_sigterm_takes_the_session_down(env):
    e = dict(env["env"], STUB_MODE="hang")
    p = subprocess.Popen([_PY, str(_LEG), "claude-test-1", "m", "--dangerously-skip-permissions"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=env["ws"], env=e)
    p.stdin.write("x")
    p.stdin.close()
    for _ in range(100):
        if _sessions():
            break
        time.sleep(0.1)
    assert _sessions(), "the leg never started its session"
    time.sleep(1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    assert _sessions() == [] and _stub_procs(env["tag"]) == []


def test_sigkill_of_the_leg_is_cleaned_up_by_the_guard_in_the_session(env):
    e = dict(env["env"], STUB_MODE="hang")
    p = subprocess.Popen([_PY, str(_LEG), "claude-test-1", "m", "--dangerously-skip-permissions"],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, text=True, cwd=env["ws"], env=e)
    p.stdin.write("x")
    p.stdin.close()
    for _ in range(100):
        if _sessions():
            break
        time.sleep(0.1)
    time.sleep(1)
    p.kill()
    p.wait()
    for _ in range(100):           # guard polls every 3 s, then SIGTERM + up to 5 s
        if not _sessions():
            break
        time.sleep(0.2)
    assert _sessions() == [] and _stub_procs(env["tag"]) == []


# --- pure helpers --------------------------------------------------------------------

def test_print_only_flags_are_dropped_for_the_interactive_cli():
    m = _leg_module()
    kept, dropped = m.interactive_args(["--dangerously-skip-permissions", "--output-format", "json",
                                        "--setting-sources", "project", "-p",
                                        "--no-session-persistence", "--max-turns=5"])
    assert kept == ["--dangerously-skip-permissions", "--setting-sources", "project"]
    assert dropped == ["--output-format", "json", "-p", "--no-session-persistence", "--max-turns=5"]


def test_only_the_bottom_most_goal_status_counts():
    m = _leg_module()
    paused = "x\nGoal paused · the request was rate limited · send a message to retry\n"
    assert m.classify_pane(paused) == ("paused", "the request was rate limited")
    # a resolved pause scrolled up: the newer "Goal set" supersedes it
    assert m.classify_pane(paused + "❯ /goal ...\n  ⎿  Goal set: FORGEKERNEL\n") == ("", "")
    assert m.classify_pane("Goal still active · the API was unavailable · retrying in 1 min (1/3)")[0] == "still"
    assert m.classify_pane('Goal cleared after an unrecoverable error (context limit reached): "x"')[0:2] == \
        ("cleared", "context limit reached")
    assert m.classify_pane("WARNING: Claude Code running in Bypass Permissions mode\n ❯ No, exit\n"
                           "   Yes, I accept\n")[0] == "bypass"


def test_seeding_adds_only_what_is_missing(tmp_path):
    m = _leg_module()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "theme": "light",
                                                  "projects": {}, "userID": "u"}))
    (cfg / "settings.json").write_text(json.dumps({"model": "opus"}))
    done = m.seed_config(str(cfg), tmp_path, bypass=True, version="2.1.304")
    g = json.loads((cfg / ".claude.json").read_text())
    assert g["theme"] == "light" and g["userID"] == "u"           # untouched
    assert g["projects"][str(tmp_path.resolve())]["hasTrustDialogAccepted"] is True
    s = json.loads((cfg / "settings.json").read_text())
    assert s == {"model": "opus", "skipDangerousModePermissionPrompt": True}
    assert len(done) == 2
    assert m.seed_config(str(cfg), tmp_path, bypass=True, version="2.1.304") == []   # idempotent


def test_the_env_file_quotes_values_and_drops_nested_session_markers():
    m = _leg_module()
    text = m.env_file_text({"A": "it's $HOME", "CLAUDECODE": "1", "BASH_FUNC_x%%": "() {}", "B": ""})
    assert "CLAUDECODE" not in text and "BASH_FUNC" not in text
    out = subprocess.run(["bash", "-c", text + 'printf "%s|%s" "$A" "$B"'], capture_output=True,
                         text=True).stdout
    assert out == "it's $HOME|"


# --- how a turn reaches the goal session -----------------------------------------------

def _scripts_copy(tmp_path):
    d = tmp_path / "scripts"
    d.mkdir()
    for f in ("agent-claude.sh", "timeout-run.py"):
        shutil.copy(_ROOT / "scripts" / f, d / f)
    # the goal session replaced by a recorder: which args reached it
    (d / "claude-goal.py").write_text(
        '#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint("GOAL " + " ".join(sys.argv[1:]))\n'
    )
    (d / "claude-goal.py").chmod(0o755)
    return d


def test_agent_claude_dispatches_a_goal_turn(tmp_path):
    d = _scripts_copy(tmp_path)
    r = subprocess.run(["bash", str(d / "agent-claude.sh"), "claude-opus-5-5", "high"],
                       input="prompt", capture_output=True, text=True,
                       env=dict(os.environ, FK_AGENT_DRIVER="goal"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ("GOAL claude-opus-5-5 claude-opus-5-5 "
                                "--dangerously-skip-permissions --effort high")


def test_without_a_goal_request_agent_claude_runs_claude_p(tmp_path):
    d = _scripts_copy(tmp_path)
    fake = tmp_path / "claude"
    fake.write_text('#!/usr/bin/env bash\ncat >/dev/null\necho "P $*"\n')
    fake.chmod(0o755)
    r = subprocess.run(["bash", str(d / "agent-claude.sh"), "m"], input="prompt",
                       capture_output=True, text=True,
                       env=dict(os.environ, FK_CLAUDE_BIN=str(fake)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "P -p --model m --dangerously-skip-permissions"


def test_the_agent_scripts_pass_the_agent_policys_command_scan():
    """agent_policy.command_findings refuses a run whose agent command seems to change
    directory before claude starts (the project settings would not load); the agent
    scripts must not trip it."""
    sys.path.insert(0, str(_ROOT / "src"))
    from forgekernel.loop.agent_policy import command_findings
    assert command_findings([f"bash {_ROOT / 'scripts' / 'agent-claude.sh'} m m"], {}) == []
    assert command_findings([f"python3 {_ROOT / 'scripts' / 'claude-goal.py'} m m"], {}) == []


def test_a_limit_after_the_deliverable_is_complete_delivers(env):
    """The usage limit hit after the agent wrote and checked its reply (the judge never got
    to say so): the check is the authority, the turn is delivered, not retried."""
    td = _turn_dir(env)
    r = _run(env, "limit", FK_GOAL_TURN_DIR=str(td), STUB_REPLY_BEFORE_LIMIT="1",
             STUB_REPLY="EXHAUSTED: already done\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout == "EXHAUSTED: already done\n"
