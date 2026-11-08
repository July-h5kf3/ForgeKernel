"""Antigravity CLI's streamed turn protocol and local macOS file isolation.

Native agent definitions: https://antigravity.google/docs/subagents?tab=cli
Headless protocol: https://antigravity.google/docs/cli/headless
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FILE_TOOLS = ("view_file", "list_dir", "grep_search", "find_by_name", "write_to_file",
              "replace_file_content", "multi_replace_file_content")


def agent_file(workspace: Path, turn: Path) -> Path:
    return workspace / ".agents" / "agents" / ("forgekernel_" + turn.name.replace("-", "_") + ".md")


def command(spec: dict, turn: Path, environment: dict) -> list[str]:
    # AGY has no per-invocation read-deny flag. Apply a process-level filesystem
    # boundary instead of silently ignoring the role's reference/key exclusions.
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise ValueError("agy currently requires macOS sandbox-exec for role file isolation")
    ws = Path(spec["workspace"]).resolve()
    native = spec["policy"]["native"]
    turn.mkdir(parents=True, exist_ok=True)
    (turn / "tmp").mkdir(exist_ok=True)
    definition = agent_file(ws, turn)
    definition.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "---\nname: " + definition.stem + "\ndescription: ForgeKernel file-only worker.\n"
        + "tools: " + json.dumps(FILE_TOOLS) + "\nmainAgent: true\nsubagent: false\n"
        + "inheritMcp: false\ncommandExecutionPolicy: off\n---\n"
        + "Edit local files according to the task. The ForgeKernel controller runs compilation, "
          "tests and SSH jobs. Use only the listed file tools.\n")
    with definition.open("x", encoding="utf-8") as output:
        try:
            (turn / "agy-agent-owned").touch()
            output.write(text)
        except BaseException:
            definition.unlink()
            raise

    home = Path(environment.get("HOME") or Path.home())
    cache = home / ".gemini" / "antigravity-cli"
    quoted = lambda path: json.dumps(str(Path(path).resolve()), ensure_ascii=False)
    writable = [ws, *native.get("allow_write", []), turn / "tmp", cache,
                home / ".gemini" / "antigravity"]
    rules = ["(version 1)", "(allow default)", "(deny file-write*)"]
    rules += [f"(allow file-write* (subpath {quoted(p)}))" for p in writable]
    rules += [f"(allow file-write* (literal {quoted(turn / name)}))"
              for name in ("stdout.txt", "stderr.txt")]
    protected = [*native.get("deny_write", []), *(ws / p for p in
                 (".claude", ".codex", ".grok", ".agents")), cache / "settings.json"]
    rules += [f"(deny file-write* (subpath {quoted(p)}))" for p in protected]
    rules += [f"(deny file-read* file-write* (subpath {quoted(p)}))"
              for p in [*native.get("deny_read", []), turn / "request.json",
                        turn / "result.json", turn / "result.tmp", turn / "terminal.json"]]
    sandbox = turn / "agy.sb"
    sandbox.write_text("\n".join(rules) + "\n", encoding="utf-8")
    argv = ["/usr/bin/sandbox-exec", "-f", str(sandbox), environment.get("FK_AGY_BIN") or "agy",
            "--input-format", "stream-json", "--output-format", "stream-json", "--sandbox",
            "--mode", "accept-edits", "--agent", definition.stem, "--disable-slash-commands"]
    if spec["model"] != "default":
        argv += ["--model", spec["model"]]
    if spec.get("effort"):
        argv += ["--effort", spec["effort"]]
    for directory in native.get("allow_write", []):
        argv += ["--add-dir", str(directory)]
    return argv


def cleanup(workspace: Path, turn: Path) -> None:
    if (turn / "agy-agent-owned").is_file():
        agent_file(workspace, turn).unlink(missing_ok=True)


def result(text: str, returncode: int) -> tuple[int, str, str]:
    terminal = None
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("event") == "result":
            terminal = record.get("result")
    if not isinstance(terminal, dict):
        return returncode or 1, "", "agy returned no terminal result event"
    answer = str(terminal.get("response") or "")
    if terminal.get("status") != "SUCCESS":
        return returncode or 1, answer, str(terminal.get("error") or terminal.get("status"))
    if returncode or not answer.strip():
        return returncode or 1, answer, "agy exited unsuccessfully or returned an empty response"
    return 0, answer, ""
