"""SSH mirrors for a locally controlled run. No agent code executes on the server."""
from __future__ import annotations

import io
import os
import re
import shlex
import subprocess
import tarfile
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from forgekernel.localenv import MachineError

KEYS = ("FK_WORKSPACE_HOST", "FK_WORKSPACE_DIR", "FK_WORKSPACE_ENV_FILE",
        "FK_WORKSPACE_PYTHON")
# Agent and reference-generation callbacks sync from the controller after the turn.
ACTIVE_RUN = ContextVar("forgekernel_ssh_run", default=None)
_PRIVATE = {".git", ".claude", ".codex", ".grok", ".agents", ".jobs", ".exec", "__pycache__"}

# Only marked, controller-owned mirrors can be replaced. A failed extraction
# leaves the previous mirror intact. The receiver needs only Python's stdlib.
_RECEIVE = '''import os, pathlib, shutil, sys, tarfile, tempfile
target = pathlib.Path(sys.argv[1])
marker = ".forgekernel-mirror"
if target.is_symlink() or (target.exists() and not (target / marker).is_file()):
    raise RuntimeError("refusing to replace an unowned workspace: " + str(target))
target.parent.mkdir(parents=True, exist_ok=True)
stage = pathlib.Path(tempfile.mkdtemp(prefix=".fk-transfer-", dir=target.parent))
backup = stage.with_name(stage.name + ".previous")
try:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as archive:
        for member in archive:
            name = pathlib.PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise RuntimeError("unsafe workspace member: " + member.name)
            dest = stage / str(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            dest.chmod(member.mode & 0o777)
    (stage / marker).write_text("forgekernel local controller\\n")
    if target.exists():
        target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup.exists():
            backup.rename(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    print("FK_WORKSPACE_SYNC_OK", flush=True)
finally:
    shutil.rmtree(stage, ignore_errors=True)
'''


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SshWorkspace:
    host: str
    directory: str
    env_file: str = ""
    python: str = "python3"

    def run_dir(self, operator: str, run: str) -> str:
        for part in (operator, run):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part):
                raise WorkspaceError(f"invalid workspace path component: {part!r}")
        return str(PurePosixPath(self.directory) / operator / run)

    def _check_target(self, target: str) -> None:
        path, root = PurePosixPath(target), PurePosixPath(self.directory)
        if ".." in path.parts or path == root or root not in path.parents:
            raise WorkspaceError(f"workspace target must be beneath {root}: {target}")

    def sync_tree(self, source: Path, target: str, exclude_top=()) -> None:
        self._check_target(target)
        source = Path(source).resolve()
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            for current, dirs, files in os.walk(source, followlinks=False):
                dirs[:] = [d for d in dirs if d not in _PRIVATE
                           and not (Path(current) / d).is_symlink()
                           and not (Path(current) == source and d in exclude_top)]
                for name in sorted(files):
                    path = Path(current) / name
                    if (name in _PRIVATE or name in {"local.env", ".env", ".forgekernel-mirror"}
                            or name.endswith(".local.toml") or path.is_symlink()
                            or not path.is_file()):
                        continue
                    archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
        command = shlex.join([self.python, "-c", _RECEIVE, target])
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                "--", self.host, command]
        try:
            result = subprocess.run(argv, input=payload.getvalue(),
                                    capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            raise WorkspaceError(f"workspace sync to {self.host}:{target} failed: {e}") from e
        if result.returncode:
            message = result.stderr.decode(errors="replace")[-1000:]
            raise WorkspaceError(f"workspace sync to {self.host}:{target} failed: {message}")
        if b"FK_WORKSPACE_SYNC_OK" not in result.stdout.splitlines():
            raise WorkspaceError(f"workspace sync to {self.host}:{target} was not acknowledged")


def parse_workspace(env: dict) -> SshWorkspace | None:
    values = {k: str(env.get(k, "")).strip() for k in KEYS}
    host, directory, env_file, python = (values[k] for k in KEYS)
    if not any(values.values()):
        return None
    problems = []
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        problems.append("FK_WORKSPACE_HOST must be a non-interactive SSH host alias")
    if (not directory.startswith("/") or ".." in PurePosixPath(directory).parts
            or directory.rstrip("/") in ("", "/root", "/home", "/tmp")):
        problems.append("FK_WORKSPACE_DIR must name an absolute dedicated workspace directory")
    if env_file and not env_file.startswith("/"):
        problems.append("FK_WORKSPACE_ENV_FILE must be an absolute path on the SSH host")
    if python.startswith("-"):
        problems.append("FK_WORKSPACE_PYTHON must name a Python executable")
    if problems:
        raise MachineError(problems)
    return SshWorkspace(host, directory, env_file, python or "python3")


def sync_run(run, machine) -> None:
    remote = machine.workspace
    if remote is None:
        return
    target = remote.run_dir(run.task_name, run.name)
    remote.sync_tree(run.dir, target + "/state", exclude_top=("ws", "operators"))
    if run.ws.is_dir():
        remote.sync_tree(run.ws, target + "/ws")


def sync_agent_workspace(source: Path) -> None:
    context = ACTIVE_RUN.get()
    if context is None:
        return
    run, remote = context
    relative = source.resolve().relative_to(run.dir.resolve())
    target = PurePosixPath(remote.run_dir(run.task_name, run.name)) / relative.as_posix()
    remote.sync_tree(source, str(target))
