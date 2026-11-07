"""Native harnesses and Orca transport tested with real child processes, no model calls."""
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from forgekernel import agents, orca_agent
from forgekernel.loop.agent_policy import AgentPolicy
from forgekernel.loop.cli import RealBackend
from forgekernel.loop.driver import AgentError
from forgekernel.task import AgentSpec, load_task
from test_task import MINIMAL, _kb, _task


def test_task_routes_codex_teacher_and_grok_student(tmp_path):
    task = load_task(_task(tmp_path, MINIMAL +
        '[agent.teacher]\nbackend="codex"\nmodel="default"\neffort="high"\n'
        '[agent.student]\nbackend="grok"\nmodel="default"\n'), kb_root=_kb(tmp_path))
    for role in ("teacher", "student"):
        spec = getattr(task, role)
        command = shlex.split(agents.agent_command(spec, role))
        assert command[:1] == ["env"] and command[1].startswith("PYTHONPATH=/")
        assert command[2:5] == [sys.executable, "-m", "forgekernel.orca_agent"]
        assert command[command.index("--agent") + 1] == spec.backend
        assert command[command.index("--role") + 1] == role


def test_changing_harness_does_not_inherit_a_claude_model(tmp_path):
    task = load_task(_task(tmp_path, MINIMAL +
        '[agent.teacher]\nbackend="codex"\n'
        '[agent.student]\nbackend="grok"\n'), kb_root=_kb(tmp_path))
    assert task.teacher.model == task.student.model == "default"
    shared = load_task(_task(tmp_path, MINIMAL + '[agent]\nbackend="codex"\n', name="shared"),
                       kb_root=tmp_path / "repo")
    assert shared.teacher.model == shared.student.model == "default"


@pytest.mark.parametrize("agent,key", [("codex", "FK_CODEX_BIN"), ("grok", "FK_GROK_BIN")])
def test_orca_worker_uses_the_callers_binary_override(tmp_path, agent, key):
    ws = tmp_path / "ws"
    ws.mkdir()
    policy = {"role": "teacher", "native": {}}
    argv = orca_agent.native_command({"agent": agent, "workspace": str(ws), "model": "default",
        "policy": policy, "env": {key: "/caller/bin/fixture"}}, tmp_path / "turn")
    assert argv[0] == "/caller/bin/fixture"


@pytest.fixture
def transport(tmp_path, monkeypatch):
    ws = tmp_path / "operator workspace"
    ws.mkdir()
    monkeypatch.chdir(ws)
    policy = AgentPolicy("source", tmp_path, student_online=True)
    # These binaries only echo and edit a file. Test policy translation separately;
    # the host running this transport test need not support Grok's Linux sandbox.
    monkeypatch.setattr(AgentPolicy, "no_cluster", lambda self, role: False)
    state = tmp_path / "orca-state"
    state.mkdir()
    fake_orca = tmp_path / "fake-orca"
    fake_orca.write_text(f"#!{sys.executable}\n" + '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
state = Path(os.environ["TEST_ORCA_STATE"])
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
result = {}
if args[:2] == ["terminal", "create"]:
    command = args[args.index("--command") + 1]
    proc = subprocess.Popen(["sh", "-c", command], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (state / "pid").write_text(str(proc.pid))
    result = {"terminal": {"handle": "owned-terminal"}}
elif args[:2] == ["terminal", "wait"]:
    time.sleep(0.03)
    print(json.dumps({"ok": False, "error": {"code": "timeout", "message": "timeout"}}))
    sys.exit(1)
elif args[:2] == ["terminal", "read"]:
    result = {"terminal": {"status": "running"}}
elif args[:2] == ["terminal", "close"]:
    try:
        os.killpg(int((state / "pid").read_text()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    result = {"close": {"ptyKilled": True}}
print(json.dumps({"ok": True, "result": result}))
''')
    fake_orca.chmod(0o755)
    fake_native = tmp_path / "fake-native"
    fake_native.write_text(f"#!{sys.executable}\n" + '''
import os, sys
from pathlib import Path
args = sys.argv[1:]
if "exec" in args:
    prompt = sys.stdin.read()
    Path(args[args.index("--output-last-message") + 1]).write_text("codex final")
else:
    prompt = Path(args[args.index("--prompt-file") + 1]).read_text()
    print("grok final")
Path("kernel.py").write_text(prompt)
assert os.environ["PRIVATE_TEST_TOKEN"] == "private-test-value"
''')
    fake_native.chmod(0o755)
    monkeypatch.setenv("ORCA_CLI_COMMAND", shlex.quote(str(fake_orca)))
    monkeypatch.setenv("TEST_ORCA_STATE", str(state))
    monkeypatch.setenv("PRIVATE_TEST_TOKEN", "private-test-value")
    monkeypatch.setenv("FK_CODEX_BIN", str(fake_native))
    monkeypatch.setenv("FK_GROK_BIN", str(fake_native))
    monkeypatch.delenv("ORCA_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ORCA_PAIRING_CODE", raising=False)
    return ws, policy, state, fake_native


@pytest.mark.parametrize("agent,role", [("codex", "teacher"), ("grok", "student")])
def test_orca_turn_edits_the_bound_workspace_and_returns_final_response(transport, agent, role):
    ws, policy, state, _ = transport
    policy.install(role, ws)
    rc, answer, error = orca_agent.run_turn(agent, role, "default", None, 10,
                                           "write this operator; $(do not execute)")
    assert rc == 0, error
    assert answer.strip() == f"{agent} final"
    assert (ws / "kernel.py").read_text() == "write this operator; $(do not execute)"
    calls = (state / "calls.jsonl").read_text()
    assert "private-test-value" not in calls and "write this operator" not in calls
    parsed = [json.loads(line) for line in calls.splitlines()]
    assert sum(c[:2] == ["terminal", "create"] for c in parsed) == 1
    assert parsed[-1][:2] == ["terminal", "close"]
    assert not (ws / ".grok/sandbox.toml").exists()


def test_real_backend_alternates_codex_and_grok_through_orca(transport):
    ws, policy, _, _ = transport
    teacher = agents.agent_command(AgentSpec("codex", "default", "high"), "teacher")
    student = agents.agent_command(AgentSpec("grok", "default", None), "student")
    backend = RealBackend("op", ws, student, "n", "e", teacher_agent_cmd=teacher,
                          agent_policy=policy, agent_retries=1)
    assert backend.run_agent("teacher instruction", role="teacher").strip() == "codex final"
    assert "teacher instruction" in (ws / "kernel.py").read_text()
    assert backend.run_agent("student instruction", role="student").strip() == "grok final"
    assert "student instruction" in (ws / "kernel.py").read_text()


def test_codex_preserves_filesystem_and_network_policy(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    ref = tmp_path / "private-reference"
    pol = AgentPolicy("blackbox", tmp_path, ref_source_paths=(ref,))
    pol.install("teacher", ws)
    policy = json.loads((ws / orca_agent.POLICY).read_text())
    args = orca_agent.native_command({"agent": "codex", "workspace": str(ws),
        "model": "chosen-model", "effort": "high", "policy": policy}, tmp_path / "turn")
    settings = tomllib.loads("\n".join(args[i+1] for i, a in enumerate(args) if a == "-c"))
    assert settings["permissions"]["forgekernel"]["filesystem"][str(ref)] == "deny"
    assert settings["approval_policy"] == "never"
    assert settings["model_reasoning_effort"] == "high"
    assert args[args.index("--model") + 1] == "chosen-model"


def test_grok_offline_policy_uses_kernel_deny_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    ws = tmp_path / "ws"
    ws.mkdir()
    ref = tmp_path / "ref"
    pol = AgentPolicy("source", tmp_path, ref_source_paths=(ref,))
    pol.install("student", ws)
    args = orca_agent.native_command({"agent": "grok", "workspace": str(ws),
        "model": "default", "policy": json.loads((ws / orca_agent.POLICY).read_text())},
        tmp_path / "turn")
    profiles = tomllib.loads((ws / ".grok/sandbox.toml").read_text())["profiles"]
    profile = profiles[args[args.index("--sandbox") + 1]]
    assert profile["restrict_network"] is True and str(ref) in profile["deny"]
    assert "--disable-web-search" in args and "--model" not in args


def test_macos_grok_student_uses_file_tools_and_leaves_ssh_to_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    ws = tmp_path / "ws"
    ws.mkdir()
    pol = AgentPolicy("source", tmp_path)
    pol.install("student", ws)
    args = orca_agent.native_command({"agent": "grok", "workspace": str(ws),
        "model": "default", "policy": json.loads((ws / orca_agent.POLICY).read_text())}, tmp_path / "turn")
    assert set(args[args.index("--tools") + 1].split(",")) == {
        "read_file", "search_replace", "grep", "list_dir"}
    assert "--disable-web-search" in args and "--no-subagents" in args
    profiles = tomllib.loads((ws / ".grok/sandbox.toml").read_text())["profiles"]
    assert profiles[args[args.index("--sandbox") + 1]]["restrict_network"] is False
    assert "use_tool" in args[args.index("--disallowed-tools") + 1].split(",")


def test_an_already_exited_orca_terminal_is_not_reported_as_uncertain(transport, monkeypatch):
    ws, policy, _, _ = transport
    policy.install("teacher", ws)
    actual = orca_agent._orca
    def observed(*args, **kw):
        if args[:2] == ("terminal", "wait"):
            return {"wait": {"satisfied": True}}
        if args[:2] == ("terminal", "close"):
            return {"close": {"ptyKilled": False}}
        return actual(*args, **kw)
    monkeypatch.setattr(orca_agent, "_orca", observed)
    rc, answer, error = orca_agent.run_turn("codex", "teacher", "default", None, 10, "p")
    assert rc == 0 and answer == "codex final", error


def test_missing_policy_is_refused_before_contacting_orca(transport):
    with pytest.raises(orca_agent.OrcaError, match="policy is missing"):
        orca_agent.run_turn("grok", "student", "default", None, 10, "p")
    assert not (transport[2] / "calls.jsonl").exists()


def test_worker_propagates_native_failure_and_timeout(transport, tmp_path):
    ws, policy, _, native = transport
    policy.install("teacher", ws)
    native.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('native failure')\nsys.exit(9)\n")
    rc, _, error = orca_agent.run_turn("codex", "teacher", "default", None, 10, "p")
    assert rc == 9 and "native failure" in error
    native.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    rc, _, error = orca_agent.run_turn("codex", "teacher", "default", None, 1, "p")
    assert rc == 124 and "timeout" in error


def test_uncertain_transport_does_not_relaunch_the_agent(tmp_path, monkeypatch):
    backend = RealBackend("op", tmp_path, "adapter", "n", "e")
    calls = []
    def turn(*args):
        calls.append(args)
        return 1, "", "FK_AGENT_UNCERTAIN: lost launch receipt"
    monkeypatch.setattr(backend, "_agent_turn", turn)
    with pytest.raises(AgentError, match="UNCERTAIN"):
        backend.run_agent("p")
    assert len(calls) == 1


def test_uncertain_launch_preserves_private_state(transport, tmp_path, monkeypatch):
    ws, policy, _, _ = transport
    policy.install("teacher", ws)
    spool = tmp_path / "private-spool"
    spool.mkdir(mode=0o700)
    monkeypatch.setattr(orca_agent.tempfile, "mkdtemp", lambda **kw: str(spool))
    calls = []
    def lost(*args, **kw):
        calls.append(args[:2])
        if args[:2] == ("terminal", "create"):
            raise subprocess.TimeoutExpired("orca", 60)
        return {}
    monkeypatch.setattr(orca_agent, "_orca", lost)
    with pytest.raises(orca_agent.OrcaError, match="FK_AGENT_UNCERTAIN"):
        orca_agent.run_turn("codex", "teacher", "default", None, 10, "private prompt")
    assert (spool / "request.json").is_file() and (spool.stat().st_mode & 0o077) == 0
    assert calls.count(("terminal", "create")) == 1


def test_grok_preserves_existing_user_sandbox_profiles(transport):
    ws, policy, _, _ = transport
    policy.install("teacher", ws)
    sandbox = ws / ".grok/sandbox.toml"
    sandbox.parent.mkdir()
    original = '[profiles.personal]\nextends="read-only"\n'
    sandbox.write_text(original)
    rc, _, error = orca_agent.run_turn("grok", "teacher", "default", None, 10, "p")
    assert rc == 0, error
    assert sandbox.read_text() == original


def test_linux_uses_orca_ide_instead_of_the_screen_reader(monkeypatch):
    monkeypatch.delenv("ORCA_CLI_COMMAND", raising=False)
    monkeypatch.delenv("ORCA_DEV_REPO_ROOT", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert orca_agent.orca_command() == ["orca-ide"]
