"""Native harnesses and Orca transport tested with real child processes, no model calls."""
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tomllib
from types import SimpleNamespace
from pathlib import Path

import pytest

from forgekernel import agents, orca_agent, agy
from forgekernel import task as task_module
from forgekernel.loop.agent_policy import AgentPolicy
from forgekernel.loop.cli import RealBackend
from forgekernel.loop.driver import AgentError
from forgekernel.task import AgentSpec, load_task
from test_task import MINIMAL, _kb, _task


@pytest.mark.parametrize("teacher", task_module.BACKENDS)
@pytest.mark.parametrize("student", task_module.BACKENDS)
def test_both_roles_choose_harness_model_effort_and_timeout_independently(tmp_path, teacher, student):
    task = load_task(_task(tmp_path, MINIMAL +
        f'[agent.teacher]\nbackend="{teacher}"\nmodel="teacher/custom-model"\n'
        'effort="high"\ntimeout_s=43\n'
        f'[agent.student]\nbackend="{student}"\nmodel="student/custom-model"\n'
        'effort="low"\ntimeout_s=17\n'), kb_root=_kb(tmp_path))
    for role, backend, effort, timeout in (("teacher", teacher, "high", 43),
                                          ("student", student, "low", 17)):
        spec = getattr(task, role)
        assert (spec.backend, spec.model, spec.effort, spec.timeout_s) == (
            backend, f"{role}/custom-model", effort, timeout)
        command = shlex.split(agents.agent_command(spec, role))
        if backend == "claude":
            assert command[2:] == [spec.model, effort, str(timeout)]
        else:
            assert command[0] == "env" and command[1].startswith("PYTHONPATH=/")
            assert command[2:5] == [sys.executable, "-m", "forgekernel.orca_agent"]
            assert command[command.index("--agent") + 1] == spec.backend
            assert command[command.index("--role") + 1] == role
            assert command[command.index("--model") + 1] == spec.model
            assert command[command.index("--effort") + 1] == effort
            assert command[command.index("--timeout") + 1] == str(timeout)


@pytest.mark.parametrize("before,after", [(a, b) for a in task_module.BACKENDS
                                         for b in task_module.BACKENDS if a != b])
def test_changing_shared_harness_resets_both_roles_models_and_providers(tmp_path, monkeypatch, before, after):
    base = task_module.read_global_config([])
    base["agent"] = {"backend": before, "model": "old-model", "provider": "old-gateway",
                     "timeout_s": 19, "teacher": {"model": "old-teacher-model"},
                     "student": {"model": "old-student-model"}}
    monkeypatch.setattr(task_module, "read_global_config", lambda problems: base)
    task = load_task(_task(tmp_path, MINIMAL + f'[agent]\nbackend="{after}"\n'),
                     kb_root=_kb(tmp_path))
    for spec in (task.teacher, task.student):
        assert (spec.backend, spec.model, spec.provider, spec.timeout_s) == (
            after, "default", "native", 19)


def test_shared_harness_override_uses_the_previous_effective_role(tmp_path, monkeypatch):
    base = task_module.read_global_config([])
    base["agent"] = {"backend": "grok", "model": "grok-chosen", "provider": "native",
                     "timeout_s": 19, "teacher": {"backend": "codex", "model": "codex-chosen"}}
    monkeypatch.setattr(task_module, "read_global_config", lambda problems: base)
    task = load_task(_task(tmp_path, MINIMAL + '[agent]\nbackend="grok"\n'), kb_root=_kb(tmp_path))
    assert (task.teacher.backend, task.teacher.model) == ("grok", "default")
    assert (task.student.backend, task.student.model) == ("grok", "grok-chosen")


def test_changing_harness_does_not_inherit_a_claude_model(tmp_path):
    task = load_task(_task(tmp_path, MINIMAL +
        '[agent.teacher]\nbackend="codex"\n'
        '[agent.student]\nbackend="grok"\n'), kb_root=_kb(tmp_path))
    assert task.teacher.model == task.student.model == "default"


@pytest.mark.parametrize("agent,key", [("codex", "FK_CODEX_BIN"), ("grok", "FK_GROK_BIN"),
                                      ("agy", "FK_AGY_BIN")])
def test_orca_worker_uses_the_callers_binary_override(tmp_path, agent, key):
    if agent == "agy" and sys.platform != "darwin":
        pytest.skip("AGY process isolation currently requires macOS")
    ws = tmp_path / "ws"
    ws.mkdir()
    policy = {"role": "teacher", "native": {}}
    argv = orca_agent.native_command({"agent": agent, "workspace": str(ws), "model": "default",
        "policy": policy, "env": {key: "/caller/bin/fixture"}}, tmp_path / "turn")
    assert "/caller/bin/fixture" in argv


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
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "exec" in args:
    prompt = sys.stdin.read()
    Path(args[args.index("--output-last-message") + 1]).write_text("codex final")
else:
    if "--input-format" in args:
        prompt = json.loads(sys.stdin.readline())["message"]["content"]
        print(json.dumps({"event":"result","result":{"status":"SUCCESS","response":"agy final"}}))
    else:
        prompt = Path(args[args.index("--prompt-file") + 1]).read_text()
        print("grok final")
Path("kernel.py").write_text(prompt)
Path("chosen-model.txt").write_text(args[args.index("--model") + 1] if "--model" in args else "default")
assert os.environ["PRIVATE_TEST_TOKEN"] == "private-test-value"
''')
    fake_native.chmod(0o755)
    monkeypatch.setenv("ORCA_CLI_COMMAND", shlex.quote(str(fake_orca)))
    monkeypatch.setenv("TEST_ORCA_STATE", str(state))
    monkeypatch.setenv("PRIVATE_TEST_TOKEN", "private-test-value")
    monkeypatch.setenv("FK_CODEX_BIN", str(fake_native))
    monkeypatch.setenv("FK_GROK_BIN", str(fake_native))
    monkeypatch.setenv("FK_AGY_BIN", str(fake_native))
    monkeypatch.delenv("ORCA_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ORCA_PAIRING_CODE", raising=False)
    return ws, policy, state, fake_native


@pytest.mark.parametrize("agent,role", [("codex", "teacher"), ("grok", "student"),
                                       ("agy", "teacher"), ("agy", "student")])
def test_orca_turn_edits_the_bound_workspace_and_returns_final_response(transport, agent, role):
    if agent == "agy" and sys.platform != "darwin":
        pytest.skip("AGY process isolation currently requires macOS")
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
    assert not list((ws / ".agents/agents").glob("forgekernel_*.md"))


@pytest.mark.parametrize("teacher_backend,student_backend", [
    (a, b) for a in ("codex", "grok", "agy") for b in ("codex", "grok", "agy")])
def test_real_backend_routes_independent_harnesses_and_models(transport, teacher_backend, student_backend):
    if "agy" in (teacher_backend, student_backend) and sys.platform != "darwin":
        pytest.skip("AGY process isolation currently requires macOS")
    ws, policy, _, _ = transport
    teacher = agents.agent_command(AgentSpec(teacher_backend, "teacher-model", "high"), "teacher")
    student = agents.agent_command(AgentSpec(student_backend, "student-model", None), "student")
    backend = RealBackend("op", ws, student, "n", "e", teacher_agent_cmd=teacher,
                          agent_policy=policy, agent_retries=1)
    assert backend.run_agent("teacher instruction", role="teacher").strip() == f"{teacher_backend} final"
    assert (ws / "chosen-model.txt").read_text() == "teacher-model"
    assert "teacher instruction" in (ws / "kernel.py").read_text()
    assert backend.run_agent("student instruction", role="student").strip() == f"{student_backend} final"
    assert (ws / "chosen-model.txt").read_text() == "student-model"
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
        if args[:2] == ("terminal", "wait") and "5000" in args:
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


def test_slow_orca_wait_keeps_the_active_turn(transport, monkeypatch):
    ws, policy, state, _ = transport
    policy.install("teacher", ws)
    actual = orca_agent._orca
    missed = False

    def observed(*args, **kw):
        nonlocal missed
        if args[:2] == ("terminal", "wait") and "1000" in args and not missed:
            missed = True
            raise subprocess.TimeoutExpired("orca", 10)
        return actual(*args, **kw)

    monkeypatch.setattr(orca_agent, "_orca", observed)
    rc, answer, error = orca_agent.run_turn("codex", "teacher", "default", None, 10, "p")
    calls = [json.loads(line)[:2] for line in (state / "calls.jsonl").read_text().splitlines()]
    assert missed and calls.count(["terminal", "create"]) == 1
    assert rc == 0 and answer == "codex final", error


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


@pytest.mark.parametrize("missing", ("launch receipt", "worker result"))
def test_uncertain_turn_preserves_private_state(transport, tmp_path, monkeypatch, missing):
    ws, policy, _, _ = transport
    policy.install("teacher", ws)
    spool = tmp_path / "private-spool"
    spool.mkdir(mode=0o700)
    monkeypatch.setattr(orca_agent.tempfile, "mkdtemp", lambda **kw: str(spool))
    calls = []
    def lost(*args, **kw):
        calls.append(args[:2])
        if args[:2] == ("terminal", "create"):
            if missing == "launch receipt":
                raise subprocess.TimeoutExpired("orca", 60)
            return {"terminal": {"handle": "owned-terminal"}}
        if args[:2] == ("terminal", "wait"):
            return {"wait": {"satisfied": True}}
        if args[:2] == ("terminal", "close"):
            return {"close": {"ptyKilled": False}}
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


@pytest.mark.parametrize("model", ("default", "explicit-claude-model"))
def test_claude_headless_turn_respects_model_choice(tmp_path, model):
    native = tmp_path / "claude"
    native.write_text(f"#!{sys.executable}\nimport json, sys\nsys.stdin.read()\nprint(json.dumps(sys.argv[1:]))\n")
    native.chmod(0o755)
    environment = {**os.environ, "FK_CLAUDE_BIN": str(native), "FK_AGENT_PY": sys.executable}
    environment.pop("FK_AGENT_DRIVER", None)
    environment.pop("FK_PROVIDER_KEY_ENV", None)
    script = Path(__file__).resolve().parents[1] / "scripts/agent-claude.sh"
    result = subprocess.run(["bash", str(script), model, "", "5"], input="prompt",
                            capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    if model == "default":
        assert "--model" not in args
    else:
        assert args[args.index("--model") + 1] == model


@pytest.mark.parametrize("expected,refusal,code", [
    ("default", None, 0), ("explicit-model", None, 75),
    ("default", {"category": "test", "original": "requested", "fallback": "other", "read": []}, 77),
])
def test_claude_goal_accepts_native_default_but_checks_explicit_models_and_refusals(tmp_path, expected, refusal, code):
    path = Path(__file__).resolve().parents[1] / "scripts/claude-goal.py"
    spec = importlib.util.spec_from_file_location("forgekernel_test_claude_goal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    leg = module.GoalLeg.__new__(module.GoalLeg)
    leg.expected, leg.sid, leg.cfg = expected, "test-session", tmp_path
    transcript = SimpleNamespace(locate=lambda: tmp_path / "transcript.jsonl", refusal=refusal,
                                 path=tmp_path / "transcript.jsonl", models={"served-default-model"})
    assert leg.model_verdict(transcript)[0] == code


@pytest.mark.parametrize("output,returncode,expected", [
    ('{"event":"result","result":{"status":"SUCCESS","response":"done"}}', 0, (0, "done", "")),
    ('{"event":"result","result":{"status":"ERROR","response":"","error":"location unavailable"}}',
     0, (1, "", "location unavailable")),
    ('{"event":"init","init":{}}', 0, (1, "", "agy returned no terminal result event")),
    ('{"event":"result","result":{"status":"SUCCESS","response":""}}',
     0, (1, "", "agy exited unsuccessfully or returned an empty response")),
])
def test_agy_requires_a_successful_nonempty_terminal_result(output, returncode, expected):
    assert agy.result(output, returncode) == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process sandbox")
def test_agy_process_cannot_read_reference_or_write_outside_its_working_copy(tmp_path):
    ws, turn = tmp_path / "ws", tmp_path / "turn"
    ws.mkdir()
    turn.mkdir()
    secret = ws / "reference.txt"
    secret.write_text("private reference")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    (turn / "request.json").write_text("private supervisor state")
    args = agy.command({"workspace": str(ws), "model": "chosen", "effort": "high",
        "policy": {"native": {"deny_read": [str(secret)]}}}, turn, {})
    agent = agy.agent_file(ws, turn)
    tools = json.loads(next(line[7:] for line in agent.read_text().splitlines() if line.startswith("tools: ")))
    assert set(tools) == set(agy.FILE_TOOLS) and "run_command" not in tools
    script = '''from pathlib import Path
import sys
ws, secret, outside, request, definition = map(Path, sys.argv[1:])
(ws / "kernel.py").write_text("allowed")
for path, mode in ((secret,"r"),(outside,"w"),(request,"r"),(definition,"w")):
    try:
        with path.open(mode): pass
    except PermissionError: continue
    raise AssertionError("sandbox allowed " + mode + " " + str(path))
print("FILE_BOUNDARY_OK")
'''
    result = subprocess.run([*args[:3], sys.executable, "-c", script,
                             str(ws), str(secret), str(outside), str(turn / "request.json"), str(agent)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and "FILE_BOUNDARY_OK" in result.stdout, result.stderr
    assert outside.read_text() == "keep"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process sandbox")
def test_agy_preserves_existing_agent_definitions(transport):
    ws, policy, _, _ = transport
    policy.install("teacher", ws)
    folder = ws / ".agents/agents"
    folder.mkdir(parents=True)
    personal = folder / "personal.md"
    personal.write_text("existing user agent")
    rc, answer, error = orca_agent.run_turn("agy", "teacher", "default", None, 10, "p")
    assert rc == 0 and answer.strip() == "agy final", error
    assert personal.read_text() == "existing user agent"
    assert list(folder.iterdir()) == [personal]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process sandbox")
def test_agy_refuses_to_replace_an_existing_definition(tmp_path):
    ws, turn = tmp_path / "ws", tmp_path / "turn"
    definition = agy.agent_file(ws, turn)
    definition.parent.mkdir(parents=True)
    definition.write_text("existing user data")
    with pytest.raises(FileExistsError):
        agy.command({"workspace": str(ws), "model": "default", "policy": {"native": {}}}, turn, {})
    agy.cleanup(ws, turn)
    assert definition.read_text() == "existing user data"


def test_agy_does_not_silently_drop_isolation_on_unsupported_platforms(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="requires macOS sandbox-exec"):
        agy.command({}, tmp_path / "turn", {})


def test_agy_reference_generation_is_refused_before_launch(transport, monkeypatch):
    ws, policy, state, _ = transport
    policy.install("teacher", ws)
    monkeypatch.setenv("FK_AGENT_DRIVER", "goal")
    with pytest.raises(orca_agent.OrcaError, match="cannot drive reference-generation"):
        orca_agent.run_turn("agy", "teacher", "default", None, 10, "p")
    assert not (state / "calls.jsonl").exists()
