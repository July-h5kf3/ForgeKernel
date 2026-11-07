"""Agent commands from a task's ``[agent]`` (merged per role: forgekernel.task).

The task names backend, provider, model, effort and timeout. Claude uses
scripts/agent-claude.sh; Codex and Grok use headless turns in Orca-owned terminals.
"""

from __future__ import annotations

import os
import shlex
import sys

from forgekernel import provider as provider_mod
from forgekernel.paths import repo_root

AGENT_SCRIPT = "scripts/agent-claude.sh"


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    if spec.backend in ("codex", "grok"):
        provider_mod.grant(spec.backend, spec.provider, env)
        argv = [sys.executable, "-m", "forgekernel.orca_agent", "--agent", spec.backend,
                "--role", role, "--model", spec.model, "--timeout", str(spec.timeout_s)]
        if spec.effort:
            argv += ["--effort", spec.effort]
        pythonpath = str(repo_root() / "src")
        if os.environ.get("PYTHONPATH"):
            pythonpath += os.pathsep + os.environ["PYTHONPATH"]
        return shlex.join(["env", f"PYTHONPATH={pythonpath}", *argv])
    if spec.backend != "claude":
        raise ValueError(f"backend {spec.backend!r} is not supported")
    grant = provider_mod.grant(spec.backend, spec.provider, env)
    argv = ["bash", str(repo_root() / AGENT_SCRIPT), spec.model, spec.effort or "",
            str(spec.timeout_s)]
    prefix = ["env", *(f"{k}={v}" for k, v in grant.items())] if grant else []
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
