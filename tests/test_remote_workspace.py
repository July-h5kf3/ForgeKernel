"""Local agents with SSH mirrors and remote job execution; no server-side agents."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel import launch, runs, taskcard, workspace
from forgekernel.__main__ import main
from forgekernel.localenv import MachineError, load_machine
from forgekernel.loop.cli import RealBackend
from forgekernel.submit.config import load_config
from forgekernel.submit.direct import DirectRunner, remote_tree
from forgekernel.task import load_task
from test_task import MINIMAL, _kb, _task


def _machine(root):
    return load_machine(root, environ={
        "FK_WORKSPACE_HOST": "gpu", "FK_WORKSPACE_DIR": str(root / "remote"),
        "FK_WORKSPACE_PYTHON": sys.executable,
    })


@pytest.fixture
def ssh(monkeypatch):
    """Execute the SSH payload in a local subprocess, retaining real tar/stdio boundaries."""
    actual = subprocess.run
    calls = []
    def run(argv, **kw):
        if argv[0] != "ssh":
            return actual(argv, **kw)
        calls.append(argv)
        return actual(["sh", "-c", argv[-1]], **kw)
    monkeypatch.setattr(workspace.subprocess, "run", run)
    return calls


def test_workspace_config_selects_remote_jobs_and_local_agents(tmp_path):
    (tmp_path / "local.env").write_text(
        'FK_WORKSPACE_HOST=gpu\nFK_WORKSPACE_DIR="/data/ldc/forge kernel"\n'
        'FK_WORKSPACE_ENV_FILE=/opt/env.sh\n')
    m = load_machine(tmp_path, environ={})
    assert m.workspace and m.delivery == "ssh"
    assert (m.remote_host, m.remote_workdir, m.remote_env_file) == (
        "gpu", "/data/ldc/forge kernel", "/opt/env.sh")
    assert workspace.load_workspace(tmp_path, {"FK_WORKSPACE_HOST": "other"}).host == "other"


@pytest.mark.parametrize("values", [
    {"FK_WORKSPACE_HOST": "-oProxyCommand=bad", "FK_WORKSPACE_DIR": "/data/fk"},
    {"FK_WORKSPACE_HOST": "gpu", "FK_WORKSPACE_DIR": "relative"},
    {"FK_WORKSPACE_HOST": "gpu", "FK_WORKSPACE_DIR": "/"},
    {"FK_WORKSPACE_HOST": "gpu", "FK_WORKSPACE_DIR": "/data/../home"},
    {"FK_WORKSPACE_HOST": "gpu", "FK_WORKSPACE_DIR": "/data/fk", "FK_WORKSPACE_ENV_FILE": "env.sh"},
])
def test_invalid_workspace_is_refused(tmp_path, values):
    with pytest.raises(MachineError):
        load_machine(tmp_path, environ=values)


def test_remote_shell_preserves_arguments_env_and_exit_code(tmp_path):
    root = tmp_path / "remote checkout's directory"
    directory = root / "op/run/jobs/op"
    env_file = tmp_path / "remote env.sh"
    env_file.write_text("cd /tmp\nexport FROM_ENV_SCRIPT=yes\n")
    config = workspace.SshWorkspace("gpu", str(root), str(env_file), sys.executable)
    script = ('import os, sys; assert os.environ["FROM_ENV_SCRIPT"] == "yes"; '
              'assert sys.argv[1] == "kernel; $(touch injected)"; '
              'assert os.getcwd() == sys.argv[2]; raise SystemExit(7)')
    command = config.command([sys.executable, "-c", script, "kernel; $(touch injected)", str(directory)], str(directory))
    assert subprocess.run(["sh", "-c", command[-1]]).returncode == 7
    assert not (directory / "injected").exists()
    assert "-m forgekernel" not in command[-1]


def test_cli_reads_local_tasks_without_launching_remote_forge(tmp_path, monkeypatch, capsys):
    task = _task(tmp_path, MINIMAL)
    monkeypatch.setenv("FK_WORKSPACE_HOST", "gpu")
    monkeypatch.setenv("FK_WORKSPACE_DIR", str(tmp_path / "remote"))
    def no_ssh(*a, **kw):
        raise AssertionError("dry-run must not contact SSH or remote agents")
    monkeypatch.setattr(workspace.subprocess, "run", no_ssh)
    assert main(["new", str(task), "--dry-run"]) == 0
    assert "agents: this machine; SSH workspace:" in capsys.readouterr().out


def test_sync_removes_stale_files_and_never_transfers_agent_credentials(tmp_path, ssh):
    local = tmp_path / "local"
    local.mkdir()
    (local / "kernel.py").write_text("version 1")
    (local / "stale.py").write_text("old")
    for name in (".git", ".codex", ".grok", ".claude", ".agents"):
        (local / name).mkdir()
        (local / name / "auth.json").write_text("secret")
    (local / "submit.local.toml").write_text('FK_REF_KEY="secret"')
    (local / "local.env").write_text("XAI_API_KEY=secret")
    (local / "outside-link").symlink_to(local / "local.env")
    remote = workspace.SshWorkspace("gpu", str(tmp_path / "remote"), python=sys.executable)
    target = Path(remote.run_dir("op", "run")) / "ws"
    remote.sync_tree(local, str(target))
    (local / "kernel.py").write_text("version 2")
    (local / "stale.py").unlink()
    remote.sync_tree(local, str(target))
    assert (target / "kernel.py").read_text() == "version 2"
    assert sorted(p.name for p in target.iterdir()) == [".forgekernel-mirror", "kernel.py"]
    assert all("-m forgekernel" not in call[-1] for call in ssh)


def test_sync_refuses_to_replace_existing_user_data(tmp_path, ssh):
    local = tmp_path / "local"
    local.mkdir()
    (local / "kernel.py").write_text("new")
    remote = workspace.SshWorkspace("gpu", str(tmp_path / "remote"), python=sys.executable)
    target = Path(remote.run_dir("op", "run")) / "ws"
    target.mkdir(parents=True)
    (target / "user-data").write_text("keep")
    with pytest.raises(workspace.WorkspaceError, match="unowned"):
        remote.sync_tree(local, str(target))
    assert (target / "user-data").read_text() == "keep"
    with pytest.raises(workspace.WorkspaceError, match="beneath"):
        remote.sync_tree(local, remote.directory)


def test_local_run_and_remote_mirrors_are_grouped_per_operator(tmp_path, monkeypatch, ssh):
    monkeypatch.setenv(runs.RUNS_ENV, str(tmp_path / "local-runs"))
    m = _machine(tmp_path)
    task = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    first, second = runs.new_run(task, machine=m), runs.new_run(task, machine=m)
    assert first.dir.parent == second.dir.parent == tmp_path / "local-runs/rmsnorm"
    assert first.dir != second.dir
    workspace.sync_run(first, m)
    context = workspace.run_context(first, m)
    target = Path(context["run_dir"])
    assert target.parent == tmp_path / "remote/rmsnorm"
    assert json.loads((target / "state/run.json").read_text())["id"] == first.id
    first.ws.mkdir()
    (first.ws / "kernel.py").write_text("student kernel")
    monkeypatch.setenv(workspace.CONTEXT_ENV, json.dumps(context))
    workspace.sync_agent_workspace(first.ws)
    assert (target / "ws/kernel.py").read_text() == "student kernel"
    assert {r.dir for r in runs.list_runs()} == {first.dir, second.dir}


def test_gpu_jobs_use_the_same_operator_directory_and_root_gpu_lock(tmp_path):
    m = _machine(tmp_path)
    task = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    run = runs.new_run(task, root=tmp_path / "runs", machine=m)
    for op in (task.name, "_rmsnormref"):
        tree = run.operators_dir / op
        tree.mkdir(parents=True)
        taskcard.write_submit(tree, task, m)
        cfg = load_config(op, base_dir=run.dir)
        assert cfg.exec_workspace == workspace.run_context(run, m)["run_dir"]
        assert remote_tree(cfg) == cfg.exec_workspace + "/jobs/" + op
        assert str(tmp_path / "remote/.gpu0.lock") in DirectRunner(cfg)._exec_cmd("all")


def test_sync_failure_does_not_repeat_a_successful_agent_turn(tmp_path, monkeypatch):
    backend = RealBackend("op", tmp_path, "local-agent", "n", "e", agent_retries=3)
    calls = []
    monkeypatch.setattr(backend, "_agent_turn", lambda *a: (calls.append(a) or (0, "done", "")))
    def fail(*a):
        raise workspace.WorkspaceError("SSH unavailable")
    monkeypatch.setattr(workspace, "sync_agent_workspace", fail)
    with pytest.raises(workspace.WorkspaceError):
        backend.run_agent("p")
    assert len(calls) == 1
