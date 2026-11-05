"""What a headless agent may reach from the machine it runs on: the network, and the
reference implementation's source.

The rules this module enforces:

- the **student** never goes online (no WebFetch / WebSearch, no outbound traffic from
  its shell) and never reads the reference source or a teacher-only tree;
- the **teacher** may go online, but in a closed-source (blackbox) run it may read the
  reference source neither on this machine nor on the web.

Until this existed both rules were prompt text only. Agents run as ``claude -p
--dangerously-skip-permissions`` on the host, so nothing stopped them: an audit of earlier
runs' agent transcripts found students fetching public implementations of the very
operator they were building and adapting them into kernel.py, which is exactly the
dependence a blackbox run is meant to measure the absence of.

Enforcement is delivered as a Claude Code *project* settings file written into the
agent's working directory right before each turn (``<cwd>/.claude/settings.json``):

- a project file reaches every way the agent is launched -- an absolute ``claude`` path,
  a wrapper script, ``--setting-sources project`` -- where a
  ``--settings`` flag or a ``claude`` shim on PATH would each miss some of them, and
  ``CLAUDE_CODE_MANAGED_SETTINGS_PATH`` is not honoured by the builds in use;
- ``permissions.deny`` removes WebFetch / WebSearch and refuses the file tools (and the
  shell readers Claude Code recognises) on the forbidden trees; deny rules hold under
  ``--dangerously-skip-permissions`` and inside subagents. A ``Read`` deny rule is what
  the file tools are checked against -- it covers Edit and Write on the same path and is
  applied best-effort to Grep and Glob, while a ``Glob(path)`` or ``Write(path)`` rule is
  accepted, never consulted, and warned about at startup;
- the Bash **sandbox** (Seatbelt on macOS) is what makes it hard: ``filesystem.denyRead``
  stops ``python3 -c "open(...)"`` as well as ``cat``, and the network layer refuses
  every outbound connection from the student's shell. Three things are needed there,
  because each alone has a hole: ``deniedDomains = ["*"]`` (an ``allowedDomains = []``
  alone only turns an unlisted host into a permission prompt, which bypass mode
  approves), the ``WebFetch(domain:*)`` deny rule (documented to leave sandboxed
  commands unable to reach any host, and honoured from project settings, which
  ``strictAllowlist`` is NOT -- it is read from user, managed and ``--settings``
  settings only), and ``failIfUnavailable`` so that a sandbox that cannot start stops
  the agent instead of silently running it unsandboxed;
- a PreToolUse **hook** is the last layer: it refuses a student's network-shaped shell
  commands, and refuses teacher tool calls that name a declared reference repository.
  It is a tripwire, not a boundary -- a fetch that never spells the repository out
  (a URL shortener, a vendored mirror, a script that builds the URL at runtime) walks
  past it, and only the closed-source teacher, which keeps the network on purpose,
  depends on it at all;
- writes are scoped per role: the teacher may write the run's repository (milestones,
  KB recipes), the student only its own working directory, so a turn cannot edit the
  prompts, the knowledge base or this policy's own hook script;
- the sandbox also refuses writes to the settings files themselves, and the file is
  rewritten from this module before every turn, so an edit could not outlive the turn.

Each turn's prompt is prefixed with a short notice of the same rules, so the agent does
not spend its turn discovering them by failure.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.loop import lang_mode
from forgekernel.paths import operators_dir

ROLE_STUDENT = "student"
ROLE_TEACHER = "teacher"
ORACLE_BLACKBOX = "blackbox"

SETTINGS_RELPATH = Path(".claude") / "settings.json"
# The hook's input: which role this turn runs as and what it may not name. Kept next to
# the settings file so one directory is both the thing the sandbox write-protects and
# the thing git excludes.
POLICY_RELPATH = Path(".claude") / "forgekernel-agent-policy.json"
FOREIGN_BACKUP_SUFFIX = ".pre-agent-policy"
HOOK_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "agent_policy_hook.py"

# The loop.toml [loop] keys a run declares its reference with.
KEY_REF_SOURCE_PATHS = "ref_source_paths"
KEY_REF_SOURCE_URLS = "ref_source_urls"
KEY_DSL_SOURCE_DIRS = "dsl_source_dirs"
# The CUDA mode's counterpart (loop/lang_mode.py): local CUDA / CUTLASS C++ header trees
# an offline CUDA student may read. Falls back to dsl_source_dirs, where older
# configs declared them.
KEY_CUDA_SOURCE_DIRS = "cuda_source_dirs"

# Shell commands that reach the network, denied by name for an offline role. Tripwires
# layered UNDER the sandbox, which is what actually refuses the connection: a deny rule
# does not match the same program by path or inside `sh -c`. They exist so the common
# spellings fail with a message that names the rule instead of a sandbox error.
NETWORK_BASH_PREFIXES = (
    "curl", "wget", "aria2c", "scp", "sftp", "rsync", "ssh", "nc", "ncat", "telnet",
    "git clone", "git fetch", "git pull", "git ls-remote", "git remote",
    "pip install", "pip download", "pip3 install", "pip3 download",
    "uv pip install", "uv add", "uv tool install",
    "npm install", "npx", "pnpm", "yarn", "bun install",
    "gh repo", "gh api", "gh release",
    "huggingface-cli download", "hf download",
    "conda install", "poetry add", "cargo add", "brew install",
)
# The ways to the GPU machine: what a student with the network on (student_online) is
# still denied — its GPU work goes through the loop, never through a job of its own (a
# job it starts is outside the driver's ledger: unrecorded and invisible to the run's
# own accounting).
CLUSTER_BASH_PREFIXES = ("ssh", "scp", "rsync", "python -m forgekernel.submit",
                         "python3 -m forgekernel.submit")


def _abs(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


@dataclass(frozen=True)
class AgentPolicy:
    """The per-run facts the settings are rendered from.

    ``ref_source_paths`` / ``ref_source_urls``: the reference implementation's checkouts
    or installs on this machine, and its repositories on the web (URL fragments such as
    ``github.com/deepseek-ai/DeepGEMM``). ``teacher_trees``: teacher-only trees the
    student must not read (reconnaissance / reference-pricing trees, a source run's
    oracle and the directory it sits in). ``repo_root``: the run's repository, which the
    teacher may write (milestones, KB recipes) and the student may not.
    ``secret_files``: the gitignored ``submit.local.toml`` of the operator card and of
    every teacher tree — the file a closed single-file reference's key rides in
    (``src/forgekernel/card_template/eval/closed_ref.py``). It sits INSIDE the run's tree,
    where the student and a blackbox teacher can otherwise read it; denied to both, the way the
    reference checkouts are, because the key opens the payload committed in the card.
    """

    oracle_mode: str
    repo_root: Path
    ref_source_paths: tuple[Path, ...] = ()
    ref_source_urls: tuple[str, ...] = ()
    teacher_trees: tuple[Path, ...] = ()
    dsl_source_dirs: tuple[Path, ...] = ()
    secret_files: tuple[Path, ...] = ()
    # The run's loop language: a CUDA-mode run's notice names CUDA / CUTLASS headers
    # where a CuTe DSL run's names the DSL package source (lang_mode).
    language: str = "cutedsl"
    # The student's deny list with the network on: the ref_from_goal session
    # (loop/goal_ref.py), which writes a reference from scratch and may look things up.
    # The GPU-machine CLIs stay off — its GPU channel is the loop's ledgered eval command.
    student_online: bool = False
    # Paths the agent may read but not write, by the file tools or the shell: the
    # ref_from_goal session's ledger, state and candidates (goal_ref.session_read_only_paths)
    # — the records its deliverable check and its acceptance trust.
    deny_write: tuple[Path, ...] = ()
    hook_python: str = field(default_factory=lambda: sys.executable)

    def _teacher_denied(self) -> bool:
        return self.oracle_mode == ORACLE_BLACKBOX

    # -- what each role is denied ------------------------------------------------------

    def deny_read(self, role: str) -> list[Path]:
        if role == ROLE_STUDENT:
            paths = [*self.ref_source_paths, *self.teacher_trees, *self.secret_files]
        elif self._teacher_denied():
            paths = [*self.ref_source_paths, *self.secret_files]
        else:
            # A source run's teacher distils FROM the reference source: denying it
            # would remove the run's input, not enforce anything.
            paths = []
        seen, out = set(), []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def deny_urls(self, role: str) -> list[str]:
        # The student has no network at all; the URL list matters for the teacher of a
        # blackbox run, and is harmless belt-and-braces for the student.
        if role == ROLE_STUDENT or self._teacher_denied():
            return list(self.ref_source_urls)
        return []

    def offline(self, role: str) -> bool:
        return role == ROLE_STUDENT and not self.student_online

    def no_cluster(self, role: str) -> bool:
        """The student never drives the GPU machine itself, online or not (offline, the
        network denial already covers it)."""
        return role == ROLE_STUDENT

    def allow_write(self, role: str) -> list[Path]:
        """Where Bash may write besides the working directory and the temp dirs.

        The teacher files milestones and KB recipes across the run's repository, so it
        gets the repository. The student writes one kernel file in its own working
        directory (the driver, not the agent, commits and moves anything else), so it
        gets nothing extra -- otherwise a student turn could rewrite the prompts, the
        knowledge base, a teacher-only tree it cannot even read, or this policy's hook.
        """
        return [] if role == ROLE_STUDENT else [self.repo_root]

    # -- rendering ---------------------------------------------------------------------

    def _deny_paths(self, role: str, workspace: Path) -> list[Path]:
        """``deny_read`` minus anything that contains the agent's working directory.

        Denying an ancestor of the working directory would take the agent's own tree
        away from it (a mis-declared ``ref_source_paths`` entry, an oracle whose parent
        directory is the run's own tree). The declaration is wrong in that case, but the
        failure mode must be a warning, not an agent that cannot read its own kernel.
        """
        ws = Path(workspace).resolve()
        out = []
        for p in self.deny_read(role):
            if p == ws or p in ws.parents:
                sys.stderr.write(
                    f"[agent-policy] warning: not denying {p} to the {role}: it contains "
                    f"the agent's working directory {ws}\n")
                continue
            out.append(p)
        return out

    def _deny_write_paths(self, workspace: Path) -> list[Path]:
        """``deny_write`` minus anything that contains the working directory (it would
        leave the agent unable to write its own kernel)."""
        ws = Path(workspace).resolve()
        return [p for p in self.deny_write if not (p == ws or p in ws.parents)]

    def settings(self, role: str, workspace: Path) -> dict:
        denied = self._deny_paths(role, workspace)
        read_only = self._deny_write_paths(workspace)
        deny: list[str] = []
        if self.offline(role):
            # The bare names remove the tools; the domain form is what also leaves
            # sandboxed commands unable to reach any host (and, unlike strictAllowlist,
            # is honoured from project settings).
            deny += ["WebFetch", "WebSearch", "WebFetch(domain:*)"]
            deny += [f"Bash({c}:*)" for c in NETWORK_BASH_PREFIXES]
            deny += [f"Bash({c}:*)" for c in CLUSTER_BASH_PREFIXES
                     if c not in NETWORK_BASH_PREFIXES]
        elif self.no_cluster(role):
            deny += [f"Bash({c}:*)" for c in CLUSTER_BASH_PREFIXES]
        for p in denied:
            # Both spellings: gitignore semantics make `p/**` everything UNDER p, which
            # for a file (a source run's oracle) is nothing at all. Read rules are the
            # ones the file tools are checked against; they cover Edit and Write too.
            deny.append(f"Read(/{p})")
            deny.append(f"Read(/{p}/**)")
        # The policy directory itself: not editable through the file tools (the sandbox
        # covers the shell), and neither is the hook script the policy runs.
        deny.append(f"Edit(/{workspace / '.claude'}/**)")
        deny.append(f"Edit(/{HOOK_SCRIPT})")
        for p in read_only:
            deny.append(f"Edit(/{p})")
            deny.append(f"Edit(/{p}/**)")

        sandbox: dict = {
            "enabled": True,
            # A sandbox that cannot start otherwise warns and runs the command
            # unsandboxed, which for this policy means no enforcement at all.
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": [str(p) for p in denied],
                "allowWrite": [str(p) for p in self.allow_write(role)],
                "denyWrite": [str(workspace / ".claude"), str(HOOK_SCRIPT),
                              *(str(p) for p in denied), *(str(p) for p in read_only)],
            },
        }
        if self.offline(role):
            sandbox["network"] = {"allowedDomains": [], "deniedDomains": ["*"],
                                  "strictAllowlist": True}

        out: dict = {"permissions": {"deny": deny}, "sandbox": sandbox}
        if self.offline(role) or self.no_cluster(role) or self.deny_urls(role):
            hook =" ".join(shlex.quote(x) for x in (
                self.hook_python, str(HOOK_SCRIPT), str(workspace / POLICY_RELPATH)))
            out["hooks"] = {"PreToolUse": [{
                "matcher": "WebFetch|WebSearch|Bash",
                "hooks": [{"type": "command", "command": hook}],
            }]}
        return out

    def policy_document(self, role: str) -> dict:
        """What the hook reads: this turn's role and what it may not reach."""
        return {
            "role": role,
            "oracle_mode": self.oracle_mode,
            "offline": self.offline(role),
            "no_cluster": self.no_cluster(role),
            "deny_urls": self.deny_urls(role),
        }

    def notice(self, role: str) -> str:
        lines = ["## Runtime environment constraints (enforced by the harness, not advice)"]
        cuda = lang_mode.is_cuda(self.language)
        if self.offline(role):
            lines.append(
                "- **No network this turn**: WebFetch / WebSearch are removed, and every outbound "
                "connection from Bash (curl, pip, git clone, python networking) is refused by the "
                "sandbox. Do not try to go online; look up interface details in the KB, the local "
                + ("CUDA / CUTLASS headers" if cuda else "DSL package sources")
                + " and your own probes.")
            lines.append(
                "- **You may write only your own working directory**: the rest of the repo "
                "(prompts, KB, other operator trees) is read-only to you, and the harness does "
                "the commits.")
            lines.append(
                "- **The evaluator belongs to the card**: `gpu_entry.sh`, `eval/`, `ref_bin/` and "
                "`.gitattributes` come from the card, and every job runs the card's copy; a delivery "
                "that modifies them is not evaluated at all and is REVERTed outright.")
            if self.dsl_source_dirs:
                lines.append(("- Readable local CUDA / CUTLASS header directories: " if cuda
                              else "- Readable local DSL package sources: ") + ", ".join(
                    f"`{p}`" for p in self.dsl_source_dirs))
        elif role == ROLE_STUDENT:
            lines.append(
                "- You may go online for public material (docs, forums, issues, papers), but "
                "**code you see online must not be copied** into your kernel: write the implementation "
                "yourself. Do not ssh to a GPU machine or submit jobs yourself; the GPU is reached only "
                "through the eval commands in your brief.")
            lines.append(
                "- **You may write only your own working directory**: the rest of the repo "
                "(prompts, KB, other operator trees) is read-only to you.")
        else:
            lines.append(
                "- You may go online for public material (docs, forums, issues, papers), but "
                "**code you see online must not be copied** into the milestone, the KB or a probe: "
                "describe mechanisms in your own words and certify them on the GPU per the KB rules.")
        denied = self.deny_read(role)
        if denied:
            what = ("the reference implementation's source, teacher-only trees and the sealed "
                    "reference's key files" if role == ROLE_STUDENT
                    else "the reference implementation's source and the sealed reference's key files")
            lines.append(f"- These paths are {what}; reading them is refused: " + ", ".join(
                f"`{p}`" for p in denied))
        if self.deny_write:
            lines.append("- These are the harness's records: readable, not writable "
                         "(a rewrite is refused): " + ", ".join(
                f"`{p}`" for p in self.deny_write))
        if role != ROLE_STUDENT and self.oracle_mode == ORACLE_BLACKBOX:
            lines.append(
                "- This is a blackbox run: **looking up the reference implementation's source "
                "online is forbidden** (including mirror sites, raw / api endpoints and package-manager "
                "downloads).")
        return "\n".join(lines) + "\n\n"

    # -- installation ------------------------------------------------------------------

    def install(self, role: str, workspace: Path) -> Path:
        """Write this role's settings into ``workspace`` and keep them out of git.

        Called before every turn: roles alternate in the same working directory, and
        rewriting from the harness each time means nothing the previous turn did to the
        directory survives into the next."""
        workspace = Path(workspace)
        settings_path = workspace / SETTINGS_RELPATH
        policy_path = workspace / POLICY_RELPATH
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        if settings_path.exists() and not policy_path.exists():
            # A settings file this module did not write: keep it aside rather than
            # silently replacing what someone configured by hand.
            backup = settings_path.with_name(settings_path.name + FOREIGN_BACKUP_SUFFIX)
            if not backup.exists():
                settings_path.replace(backup)
                sys.stderr.write(
                    f"[agent-policy] moved a pre-existing {settings_path} aside to "
                    f"{backup.name}; the harness owns this file from now on\n")
        policy_path.write_text(
            json.dumps(self.policy_document(role), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        settings_path.write_text(
            json.dumps(self.settings(role, workspace), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        # the repository's skills/, where Claude Code finds them (ForgeTrain's mirroring)
        from forgekernel import skills
        skills.mirror(workspace)
        _git_exclude(workspace, ".claude/")
        return settings_path


def _git_exclude(workspace: Path, pattern: str) -> None:
    git_dir = workspace / ".git"
    if git_dir.is_file():
        # A linked worktree: .git is a file pointing at the per-worktree directory, and
        # info/exclude is read from the COMMON directory that one's `commondir` names,
        # not from the per-worktree one (git's own resolution; verified on 2.x).
        try:
            line = git_dir.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not line.startswith("gitdir:"):
            return
        target = Path(line.split(":", 1)[1].strip())
        git_dir = target if target.is_absolute() else (workspace / target)
        common = git_dir / "commondir"
        if common.is_file():
            try:
                rel = common.read_text(encoding="utf-8").strip()
            except OSError:
                return
            git_dir = Path(rel) if Path(rel).is_absolute() else (git_dir / rel)
    if not git_dir.is_dir():
        return
    exclude = git_dir / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if pattern not in existing.splitlines():
        exclude.write_text(existing + ("" if existing.endswith("\n") or not existing
                                       else "\n") + pattern + "\n", encoding="utf-8")


# -- agent commands that would not load the policy ----------------------------------------

# An agent command that excludes project settings, runs Claude Code bare, overrides them from the
# command line, or changes directory before invoking it never runs under
# <cwd>/.claude/settings.json: the policy would be installed and silently not apply.
_SOURCES_FLAG = re.compile(r"--setting-sources(?:=|\s+)(['\"]?)([A-Za-z,]*)\1")
_BARE_FLAG = re.compile(r"(?<![\w-])--bare\b")
# --settings outranks project settings (CLI beats project in the precedence order), so a
# command that passes one could turn the sandbox or the hooks off.
_SETTINGS_FLAG = re.compile(r"(?<![\w-])--settings(?:=|\s)")
# `env -C dir claude` changes directory without a cd.
_ENV_CHDIR = re.compile(r"(?<![\w-])env\s[^\n;&|]*?(?<![\w-])(?:-C|--chdir)[=\s]")
# A statement-level cd, including one opening a subshell -- `(cd /x && claude -p)`.
# `$(cd "$(dirname "$0")" && pwd)` is how wrappers compute their own directory and
# changes nothing for the agent, so a cd inside a command substitution counts only when
# that substitution also runs the agent.
_CD_STMT = re.compile(
    r"(?:^|[;&|\"']|(?<!\$)\(|\bthen\b|\bdo\b|\belse\b)\s*(?:builtin\s+)?(?:cd|pushd)\s",
    re.M)
_CD_SUBST = re.compile(r"\$\(\s*(?:builtin\s+)?(?:cd|pushd)\s[^\n]*claude", re.M | re.I)
# A shell script an agent command names (one level deep): its text is scanned too.
_SCRIPT_TOKEN = re.compile(r"(?<![\w/.-])((?:~|\$HOME|/)?[\w./~$-]*\.(?:sh|bash|zsh))\b")


def _expand(value: str, env: dict[str, str]) -> str:
    """``$NAME`` / ``${NAME}`` / ``~`` expansion against ``env``; unknown names stay."""
    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        return env.get(name, m.group(0))
    out = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, value)
    if out.startswith("~"):
        out = env.get("HOME", os.path.expanduser("~")) + out[1:]
    return out


def command_findings(cmds: list[str | None], env: dict[str, str],
                     cwd: str | os.PathLike | None = None) -> list[str]:
    """Why some agent command would run the agent without loading the project settings.

    Reads the command text and the shell scripts it names (one level). Static, so it
    cannot see a script that computes its own flags, nor a directory change whose
    command is only reachable through a variable; the per-turn installation still
    happens, this only refuses the shapes known to defeat it."""

    def scripts_in(text: str) -> list[tuple[str, str]]:
        out = []
        for m in _SCRIPT_TOKEN.finditer(text):
            path = _expand(m.group(1), env)
            if not os.path.isabs(path) and cwd is not None:
                path = os.path.join(os.fspath(cwd), path)
            if os.path.isfile(path):
                try:
                    out.append((path, Path(path).read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    pass
        return out

    texts: list[tuple[str, str]] = []
    for cmd in cmds:
        if not cmd:
            continue
        texts.append(("agent command", cmd))
        texts += scripts_in(cmd)

    findings, seen = [], set()
    for source, text in texts:
        if source in seen:
            continue
        seen.add(source)
        body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        for m in _SOURCES_FLAG.finditer(body):
            if "project" not in m.group(2).split(","):
                findings.append(f"{source}: --setting-sources {m.group(2) or '(empty)'} "
                                f"excludes the project settings the agent policy is "
                                f"delivered in")
        if _BARE_FLAG.search(body):
            findings.append(f"{source}: --bare skips project settings and hooks")
        if _SETTINGS_FLAG.search(body):
            findings.append(f"{source}: --settings outranks the project settings the "
                            f"agent policy is delivered in, and could turn its sandbox "
                            f"or hooks off")
        if _ENV_CHDIR.search(body):
            findings.append(f"{source}: env -C runs the agent in another directory, so "
                            f"the policy in the working directory would not be loaded")
        if _CD_STMT.search(body) or _CD_SUBST.search(body):
            findings.append(f"{source}: changes directory before running the agent, so "
                            f"the policy in the working directory would not be loaded")
    return findings


# -- loop.toml ---------------------------------------------------------------------------

def declaration_problems(cfg: dict, oracle_mode: str) -> list[str]:
    """Why this run's reference declaration is not usable; empty when it is.

    A blackbox run must say where its reference lives, even if the answer is "nowhere
    on this machine" (``ref_source_paths = []``): an omitted key would silently deny
    nothing, and "we forgot" must not look the same as "there is nothing to deny"."""
    lc = cfg.get("loop", {})
    problems = []
    for key in (KEY_REF_SOURCE_PATHS, KEY_REF_SOURCE_URLS, KEY_DSL_SOURCE_DIRS):
        if key in lc and not (isinstance(lc[key], list)
                              and all(isinstance(x, str) for x in lc[key])):
            problems.append(f"[loop] {key} must be a list of strings, got {lc[key]!r}")
    if oracle_mode == ORACLE_BLACKBOX:
        for key, example in ((KEY_REF_SOURCE_PATHS, '["~/refs/vendor/DeepGEMM"]'),
                             (KEY_REF_SOURCE_URLS, '["github.com/deepseek-ai/DeepGEMM"]')):
            if key not in lc:
                problems.append(
                    f"blackbox run declares no [loop] {key} — list the reference's "
                    f"{'checkouts / installs on this machine' if key == KEY_REF_SOURCE_PATHS else 'repositories on the web'} "
                    f"(e.g. {key} = {example}), or {key} = [] if there are none")
    if not HOOK_SCRIPT.is_file():
        # The hook fails closed (a missing script is a non-zero exit, which blocks the
        # call), so a packaging slip would not leak — it would wedge every turn instead.
        problems.append(f"the agent-policy hook {HOOK_SCRIPT} is missing — every "
                        f"WebFetch / WebSearch / Bash call would be refused")
    return problems


def declaration_warnings(cfg: dict, dsl_source=None, language: str | None = None) -> list[str]:
    """``dsl_source``: the resolved ``loop.dsl_source.DslSource`` when the caller
    derived one. Its own notes then say what happened to the empty key (defaulted,
    or why not); without it an empty key gets the bare warning. ``language``: a
    CUDA-mode run is checked for its header trees instead (``cuda_source_dirs``)."""
    lc = cfg.get("loop", {})
    out = []
    if lang_mode.is_cuda(language):
        for key in (KEY_REF_SOURCE_PATHS, KEY_CUDA_SOURCE_DIRS, KEY_DSL_SOURCE_DIRS):
            for raw in lc.get(key, []) if isinstance(lc.get(key), list) else []:
                if not _abs(raw).exists():
                    out.append(f"[loop] {key} entry {raw!r} does not exist on this machine")
        if not (lc.get(KEY_CUDA_SOURCE_DIRS) or lc.get(KEY_DSL_SOURCE_DIRS)):
            out.append(f"[loop] {KEY_CUDA_SOURCE_DIRS} is empty — the CUDA student is offline, "
                       f"so it can read CUTLASS / CuTe C++ headers only from a local copy named here")
        return out
    for key in (KEY_REF_SOURCE_PATHS, KEY_DSL_SOURCE_DIRS):
        for raw in lc.get(key, []) if isinstance(lc.get(key), list) else []:
            if not _abs(raw).exists():
                out.append(f"[loop] {key} entry {raw!r} does not exist on this machine")
    if not lc.get(KEY_DSL_SOURCE_DIRS):
        if dsl_source is not None:
            if not dsl_source.defaulted:   # a default is news, not a warning
                out.extend(f"[loop] {n}" for n in dsl_source.notes)
        else:
            out.append(f"[loop] {KEY_DSL_SOURCE_DIRS} is empty — the student is offline, so "
                       f"it can read the DSL's API only from a local checkout named here")
    return out


LOCAL_SUBMIT_CONFIG = "submit.local.toml"


def secret_files_for(repo_root: Path, op: str | None, teacher_trees) -> tuple[Path, ...]:
    """The ``submit.local.toml`` of the operator card and of every teacher TREE (an
    oracle entry that names a file has no such config). Listed whether or not the
    file exists yet: the rules are strings, and the key is written after launch
    as often as before."""
    out = []
    if op:
        out.append(_abs(operators_dir(repo_root) / op / LOCAL_SUBMIT_CONFIG))
    for tree in teacher_trees:
        p = _abs(tree)
        if not p.suffix:
            out.append(p / LOCAL_SUBMIT_CONFIG)
    return tuple(dict.fromkeys(out))


def from_config(cfg: dict, *, oracle_mode: str, repo_root: Path,
                teacher_trees: list[Path],
                dsl_source_dirs_default: tuple[Path, ...] | list[Path] = (),
                op: str | None = None, language: str = "cutedsl") -> AgentPolicy:
    """``dsl_source_dirs_default``: what ``loop.dsl_source.resolve`` derived for an
    empty ``[loop] dsl_source_dirs`` — used only when the key is empty; a declared
    list wins as written. ``op``: the operator card, whose ``submit.local.toml`` is
    then denied to the student and a blackbox teacher (``secret_files``)."""
    lc = cfg.get("loop", {})
    declared = lc.get(KEY_DSL_SOURCE_DIRS, [])
    if lang_mode.is_cuda(language):
        declared = lc.get(KEY_CUDA_SOURCE_DIRS) or declared
    return AgentPolicy(
        oracle_mode=oracle_mode,
        repo_root=_abs(repo_root),
        ref_source_paths=tuple(_abs(p) for p in lc.get(KEY_REF_SOURCE_PATHS, [])),
        ref_source_urls=tuple(u.strip() for u in lc.get(KEY_REF_SOURCE_URLS, []) if u.strip()),
        teacher_trees=tuple(dict.fromkeys(_abs(p) for p in teacher_trees)),
        dsl_source_dirs=(tuple(_abs(p) for p in declared) if declared
                         else tuple(dict.fromkeys(_abs(p) for p in dsl_source_dirs_default))),
        secret_files=secret_files_for(repo_root, op, teacher_trees),
        language=language,
    )
