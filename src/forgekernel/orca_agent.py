"""One headless Codex/Grok/AGY turn in an Orca-managed terminal.

Orca owns the terminal; a small trusted worker owns stdin, the native process,
timeout and an atomic result file. We never parse a TUI or treat an idle shell as
a completed turn. Files and the terminal belong to this invocation only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from forgekernel.paths import repo_root

UNCERTAIN = "FK_AGENT_UNCERTAIN"
POLICY = Path(".claude/forgekernel-agent-policy.json")


class OrcaError(RuntimeError):
    pass


def orca_command() -> list[str]:
    configured = os.environ.get("ORCA_CLI_COMMAND")
    if configured:
        return shlex.split(configured)
    if os.environ.get("ORCA_DEV_REPO_ROOT"):
        return ["orca-dev"]
    return ["orca-ide" if sys.platform.startswith("linux") else "orca"]


def _orca(*args: str, timeout: float = 60) -> dict:
    proc = subprocess.run([*orca_command(), *args, "--json"], capture_output=True,
                          text=True, timeout=timeout)
    try:
        obj = json.loads(proc.stdout)
    except ValueError as e:
        raise OrcaError(f"Orca {' '.join(args[:2])}: invalid reply (rc={proc.returncode})") from e
    if proc.returncode or not obj.get("ok"):
        err = obj.get("error", {})
        if args[:2] == ("terminal", "wait") and err.get("code") == "timeout":
            return {"wait": {"satisfied": False}}
        raise OrcaError(f"Orca {' '.join(args[:2])}: {err.get('code', proc.returncode)}: "
                        f"{err.get('message', 'command failed')}")
    return obj.get("result", {})


def _toml_table(values: dict) -> str:
    return "{" + ", ".join(json.dumps(k) + " = " + json.dumps(v) for k, v in values.items()) + "}"


def native_command(spec: dict, turn: Path) -> list[str]:
    """Translate the same role policy to each harness; never silently drop it."""
    environment = {**os.environ, **spec.get("env", {})}
    agent = spec["agent"]
    if agent == "agy":
        from forgekernel.agy import command
        return command(spec, turn, environment)
    policy = spec["policy"]
    native = policy.get("native", {})
    ws = Path(spec["workspace"])
    # Native agents cannot modify policy/config, or read the supervisor's credentials.
    deny = [*native.get("deny_read", []), str(turn / "request.json")]
    protected = [str(ws / p) for p in (".claude", ".codex", ".grok", ".agents")]
    read_only = [*native.get("deny_write", []), *protected]
    offline = bool(policy.get("offline") or policy.get("deny_urls"))
    shell_offline = offline or bool(policy.get("no_cluster"))
    model = [] if spec["model"] == "default" else ["--model", spec["model"]]
    if agent == "codex":
        fs = {"/": "read", str(ws): "write",
              **{p: "write" for p in native.get("allow_write", [])},
              **{p: "read" for p in read_only}, **{p: "deny" for p in deny}}
        settings = {
            "approval_policy": '"never"', "default_permissions": '"forgekernel"',
            "permissions.forgekernel.filesystem": _toml_table(fs),
            "permissions.forgekernel.network.enabled": str(not shell_offline).lower(),
            "web_search": '"disabled"' if offline else '"live"',
            "shell_environment_policy.exclude": '["*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*"]',
        }
        if spec.get("effort"):
            settings["model_reasoning_effort"] = json.dumps(spec["effort"])
        args = [environment.get("FK_CODEX_BIN", "codex"), "exec", "--strict-config", "--ignore-user-config",
                "--skip-git-repo-check", "--color", "never", *model,
                "--output-last-message", str(turn / "reply.txt")]
        for key, value in settings.items():
            args += ["-c", f"{key}={value}"]
        return [*args, "-"]
    if agent != "grok":
        raise OrcaError(f"unsupported Orca harness {agent!r}")
    profile = "forgekernel_" + turn.name.replace("-", "_")
    sandbox = ws / ".grok/sandbox.toml"
    sandbox.parent.mkdir(parents=True, exist_ok=True)
    # Preserve user profiles. Unique names avoid collisions with global profiles.
    with sandbox.open("a", encoding="utf-8") as fh:
        fh.write(f'\n[profiles.{profile}]\nextends = "workspace"\n'
                 f'restrict_network = {str(shell_offline and sys.platform == "linux").lower()}\n'
                 f'read_write = {json.dumps(native.get("allow_write", []))}\n'
                 f'read_only = {json.dumps(read_only)}\n'
                 f'deny = {json.dumps(deny)}\n')
    args = [environment.get("FK_GROK_BIN", "grok"), "--prompt-file", str(turn / "prompt.txt"),
            "--output-format", "plain", "--permission-mode", "dontAsk", "--allow", "*",
            "--no-subagents", "--sandbox", profile, *model]
    if offline:
        args += ["--disable-web-search"]
    if shell_offline and sys.platform == "darwin":
        # Grok's macOS sandbox cannot block child networking. A file-only toolset
        # leaves compilation/evaluation to the local controller's SSH job channel.
        args += ["--tools", "read_file,search_replace,grep,list_dir"]
        args += ["--disallowed-tools", "run_terminal_cmd,run_terminal_command,bash,web_search,"
                 "web_fetch,search_tool,use_tool,task,spawn_subagent"]
    if policy.get("no_cluster"):
        for command in ("ssh *", "scp *", "rsync *", "python* -m forgekernel.submit *"):
            args += ["--deny", f"Bash({command})"]
    if spec.get("effort"):
        args += ["--reasoning-effort", spec["effort"]]
    return args


def worker(request: Path) -> int:
    """Runs inside Orca's terminal, on the same host and filesystem as the caller."""
    from forgekernel.submit.direct import _kill_session
    turn = request.parent
    spec = json.loads(request.read_text(encoding="utf-8"))
    rc, error = 1, ""
    proc = None
    sandbox = Path(spec["workspace"]) / ".grok/sandbox.toml"
    previous_sandbox = sandbox.read_bytes() if sandbox.is_file() else None
    try:
        argv = native_command(spec, turn)
        env = {**os.environ, **spec["env"]}
        prompt = (turn / "prompt.txt").read_text()
        if spec["agent"] == "agy":
            env.update(AGY_CLI_DISABLE_AUTO_UPDATE="true", TMPDIR=str(turn / "tmp"))
            prompt = json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
        with (turn / "stdout.txt").open("w") as out, (turn / "stderr.txt").open("w") as err:
            proc = subprocess.Popen(argv, cwd=spec["workspace"], env=env,
                                    stdin=subprocess.PIPE, stdout=out, stderr=err,
                                    text=True, start_new_session=True)
            try:
                proc.communicate(prompt, timeout=spec["timeout"])
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                _kill_session(proc)
                proc.wait()
                rc, error = 124, "agent turn exceeded its timeout"
        reply = turn / ("reply.txt" if spec["agent"] == "codex" else "stdout.txt")
        answer = reply.read_text(encoding="utf-8") if reply.is_file() else ""
        if spec["agent"] == "agy":
            from forgekernel.agy import result
            rc, answer, protocol_error = result(answer, rc)
            error = error or protocol_error
        if rc == 0 and not answer.strip():
            rc, error = 1, "agent exited without a final response"
    except BaseException as e:
        if proc is not None and proc.poll() is None:
            _kill_session(proc)
            proc.wait()
        answer, error = "", f"{type(e).__name__}: {e}"
    finally:
        if spec["agent"] == "agy":
            from forgekernel.agy import cleanup
            cleanup(Path(spec["workspace"]), turn)
        if spec["agent"] == "grok" and sandbox.is_file():
            if previous_sandbox is None:
                sandbox.unlink()
            else:
                sandbox.write_bytes(previous_sandbox)
    result = turn / "result.json"
    temporary = result.with_suffix(".tmp")
    temporary.write_text(json.dumps({"returncode": rc, "answer": answer, "error": error}))
    temporary.replace(result)
    return rc


def run_turn(agent: str, role: str, model: str, effort: str | None,
             timeout: int, prompt: str) -> tuple[int, str, str]:
    ws = Path.cwd().resolve()
    if not (ws / POLICY).is_file():
        raise OrcaError("agent policy is missing; launch this adapter through ForgeKernel")
    policy = json.loads((ws / POLICY).read_text())
    if not isinstance(policy.get("native"), dict):
        raise OrcaError("native agent policy is missing; reinstall the ForgeKernel role policy")
    if policy.get("role") != role and os.environ.get("FK_AGENT_DRIVER") != "goal":
        raise OrcaError("agent policy role does not match this turn")
    if agent == "agy" and os.environ.get("FK_AGENT_DRIVER") == "goal":
        raise OrcaError("FK_AGENT_UNSUPPORTED: agy file-only turns cannot drive reference-generation commands; "
                        "provide a reference or choose a teacher with shell tools")
    if (agent == "grok" and sys.platform == "darwin"
            and (policy.get("offline") or policy.get("no_cluster") or policy.get("deny_urls"))):
        prompt = ("This turn exposes file tools only. Edit the local kernel and probe files; "
                  "the ForgeKernel controller synchronizes them and runs compilation, "
                  "correctness and performance jobs over SSH. Do not wait for a shell tool.\n\n" + prompt)
    if os.environ.get("FK_AGENT_DRIVER") == "goal":
        timeout = max(1, int(float(os.environ["FK_GOAL_DEADLINE"]) - time.time()))
    # Orca is the local execution host's runtime. Do not route local spool paths to a
    # paired server: agent processes and their private files must stay on this host.
    if os.environ.get("ORCA_ENVIRONMENT") or os.environ.get("ORCA_PAIRING_CODE"):
        raise OrcaError("paired Orca routing is incompatible with local agents; "
                        "use the Orca runtime on this machine")
    target = os.environ.get("FK_ORCA_WORKSPACE")
    if not target:
        _orca("repo", "add", "--path", str(repo_root()), "--kind", "folder")
        target = "path:" + str(repo_root())
    # Preserve the private spool on an uncertain launch/stop: removing it cannot stop
    # an already-started worker, and would discard its eventual result and provenance.
    turn = Path(tempfile.mkdtemp(prefix="fk-orca-"))
    settled = True
    try:
        (turn / "prompt.txt").write_text(prompt, encoding="utf-8")
        (turn / "request.json").write_text(json.dumps({
            "agent": agent, "role": role, "model": model, "effort": effort,
            "timeout": timeout, "workspace": str(ws), "policy": policy,
            "env": dict(os.environ),
        }))
        command = shlex.join([sys.executable, "-m", "forgekernel.orca_agent",
                              "--worker", str(turn / "request.json")])
        command = (f"cd {shlex.quote(str(repo_root()))} && "
                   f"export PYTHONPATH={shlex.quote(str(repo_root() / 'src'))} && exec " + command)
        handle = None
        completed = False
        exited = False
        result_path = turn / "result.json"
        try:
            try:
                settled = False
                result = _orca("terminal", "create", "--worktree", target, "--title",
                               f"forge {role}: {agent}", "--command", command)
                terminal = result.get("terminal", result)
                handle = terminal.get("handle")
                if not handle:
                    raise OrcaError("terminal create returned no handle")
                (turn / "terminal.json").write_text(json.dumps({"handle": handle}))
            except (OrcaError, subprocess.TimeoutExpired) as e:
                # A create with a lost reply may already have run the command.
                raise OrcaError(f"{UNCERTAIN}: terminal launch outcome unknown: {e}") from e
            deadline = time.monotonic() + timeout + 60
            while not result_path.is_file():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OrcaError("Orca worker did not return a result before its deadline")
                try:
                    observed = _orca("terminal", "wait", "--terminal", handle, "--for", "exit",
                                     "--timeout-ms", str(max(1, min(1000, int(remaining * 1000)))), timeout=10)
                except subprocess.TimeoutExpired:
                    # A missed status poll does not invalidate this terminal or its result.
                    continue
                exited = observed.get("wait", {}).get("satisfied") is True
                if exited and not result_path.is_file():
                    raise OrcaError("Orca worker exited without a result; check its terminal log")
            result = json.loads(result_path.read_text())
            completed = True
            stderr = (turn / "stderr.txt").read_text() if (turn / "stderr.txt").is_file() else ""
            return result["returncode"], result["answer"], result["error"] + "\n" + stderr
        finally:
            if handle:
                try:
                    if completed and not exited:
                        try:
                            observed = _orca("terminal", "wait", "--terminal", handle,
                                             "--for", "exit", "--timeout-ms", "5000", timeout=10)
                            exited = observed.get("wait", {}).get("satisfied") is True
                        except (OrcaError, subprocess.SubprocessError):
                            pass  # the close below must then prove it stopped the PTY
                    closed = _orca("terminal", "close", "--terminal", handle, timeout=30)
                    if closed.get("close", {}).get("ptyKilled") is not True and not (completed and exited):
                        raise OrcaError("Orca did not confirm that the terminal stopped")
                    settled = True
                except Exception as e:
                    raise OrcaError(f"{UNCERTAIN}: cannot confirm terminal cleanup: {e}; "
                                    f"state kept at {turn}") from e
    finally:
        if settled:
            shutil.rmtree(turn)
        else:
            print(f"{UNCERTAIN}: state kept at {turn}; inspect Orca before retrying", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path)
    p.add_argument("--agent", choices=("codex", "grok", "agy"))
    p.add_argument("--role", choices=("teacher", "student"))
    p.add_argument("--model", default="default")
    p.add_argument("--effort")
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args(argv)
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    if a.worker:
        return worker(a.worker)
    if not a.agent or not a.role or a.timeout <= 0:
        p.error("--agent, --role and a positive --timeout are required")
    try:
        rc, answer, error = run_turn(a.agent, a.role, a.model, a.effort, a.timeout, sys.stdin.read())
        sys.stdout.write(answer)
        sys.stderr.write(error)
        return rc
    except (OrcaError, OSError, subprocess.SubprocessError) as e:
        print(f"[orca-agent] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
