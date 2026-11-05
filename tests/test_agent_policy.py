"""The agent policy: the student offline and away from the reference source, a blackbox
teacher away from the reference source (loop/agent_policy.py, scripts/agent_policy_hook.py).

The enforcement itself is Claude Code's (deny rules, the Bash sandbox, a PreToolUse
hook) and was verified on real ``claude -p`` sessions; these tests pin what the harness
renders and when it refuses to start, so a refactor cannot quietly drop a rule."""

import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel.loop import agent_policy as ap
from forgekernel.loop import cli

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("agent_policy_hook", REPO / "scripts" / "agent_policy_hook.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _policy(tmp_path, mode="blackbox", urls=("github.com/org/RefLib",)):
    ref = tmp_path / "refs" / "RefLib"
    tree = tmp_path / "run" / "operators" / "_ref"
    ref.mkdir(parents=True)
    tree.mkdir(parents=True)
    cfg = {"loop": {"ref_source_paths": [str(ref)], "ref_source_urls": list(urls)}}
    pol = ap.from_config(cfg, oracle_mode=mode, repo_root=tmp_path / "run", teacher_trees=[tree])
    return pol, ref.resolve(), tree.resolve()


# -- rendering ---------------------------------------------------------------------------

def test_the_student_is_offline_and_cannot_read_the_reference_or_teacher_trees(tmp_path):
    pol, ref, tree = _policy(tmp_path)
    ws = tmp_path / "run" / "operators" / "op-ws"
    s = pol.settings("student", ws)
    deny = s["permissions"]["deny"]
    assert "WebFetch" in deny and "WebSearch" in deny
    # The bare names remove the tools; the domain form is also what stops sandboxed
    # commands reaching any host, and it is honoured from project settings.
    assert "WebFetch(domain:*)" in deny
    assert "Bash(curl:*)" in deny and "Bash(pip install:*)" in deny
    net = s["sandbox"]["network"]
    # allowedDomains = [] alone lets bypass mode approve any host: the deny-all entry is
    # what refuses the connection (strictAllowlist is read from user / managed / CLI
    # settings only, never from the project file this is delivered in).
    assert net["deniedDomains"] == ["*"] and net["strictAllowlist"] is True
    assert s["sandbox"]["enabled"] is True and s["sandbox"]["allowUnsandboxedCommands"] is False
    # Without this a sandbox that cannot start warns and runs the command unsandboxed.
    assert s["sandbox"]["failIfUnavailable"] is True
    assert {str(ref), str(tree)} <= set(s["sandbox"]["filesystem"]["denyRead"])
    assert f"Read(/{ref})" in deny and f"Read(/{ref}/**)" in deny
    assert all(r.startswith(("Read(//", "Edit(//", "Bash(", "Web")) for r in deny), \
        "absolute path rules need the // prefix"
    assert str(ws / ".claude") in s["sandbox"]["filesystem"]["denyWrite"]
    assert str(ap.HOOK_SCRIPT) in s["sandbox"]["filesystem"]["denyWrite"]
    # The hook is the student's tripwire too, not only the blackbox teacher's.
    assert s["hooks"]["PreToolUse"][0]["hooks"][0]["command"].count("agent_policy_hook.py")


def test_only_the_teacher_may_write_the_runs_repository(tmp_path):
    """A student turn writes one kernel file in its own working directory; the teacher
    files milestones and KB recipes across the repo. Handing the repo to the student
    would let a turn edit the prompts, the KB, or this policy's own hook."""
    pol, _, _ = _policy(tmp_path)
    ws = tmp_path / "run" / "operators" / "op-ws"
    repo = str((tmp_path / "run").resolve())
    assert repo not in pol.settings("student", ws)["sandbox"]["filesystem"]["allowWrite"]
    assert repo in pol.settings("teacher", ws)["sandbox"]["filesystem"]["allowWrite"]


def test_a_blackbox_teacher_keeps_the_network_but_not_the_reference(tmp_path):
    pol, ref, tree = _policy(tmp_path)
    s = pol.settings("teacher", tmp_path / "ws")
    deny = s["permissions"]["deny"]
    assert "WebFetch" not in deny and "WebSearch" not in deny
    assert "network" not in s["sandbox"]
    denied_read = s["sandbox"]["filesystem"]["denyRead"]
    assert str(ref) in denied_read and str(tree) not in denied_read  # its own trees stay readable
    cmd = s["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "agent_policy_hook.py" in cmd and "forgekernel-agent-policy.json" in cmd
    assert re.search("WebFetch", s["hooks"]["PreToolUse"][0]["matcher"])


def test_the_closed_reference_key_file_is_denied_to_the_student_and_the_blackbox_teacher(tmp_path):
    """closed_ref.py's key rides the card's gitignored submit.local.toml [env] into the
    student's own eval job (that is where the job env is read from), so the key file
    sits inside the run tree the student can read — and the reference tree the teacher
    drives carries the same key. Without this rule the keyed payload is plaintext to
    both: payload, key and the decoder all lie in the tree."""
    ref = tmp_path / "refs" / "RefLib"
    tree = tmp_path / "run" / "operators" / "_ref"
    ref.mkdir(parents=True)
    tree.mkdir(parents=True)
    cfg = {"loop": {"ref_source_paths": [str(ref)], "ref_source_urls": []}}
    pol = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path / "run",
                         teacher_trees=[tree], op="op")
    card_key = (tmp_path / "run" / "operators" / "op" / "submit.local.toml").resolve()
    tree_key = (tree / "submit.local.toml").resolve()
    assert pol.secret_files == (card_key, tree_key)      # listed before either exists

    s = pol.settings("student", tmp_path / "run" / "operators" / "op-ws")
    deny, denied_read = s["permissions"]["deny"], s["sandbox"]["filesystem"]["denyRead"]
    assert f"Read(/{card_key})" in deny and str(card_key) in denied_read
    assert str(card_key) in s["sandbox"]["filesystem"]["denyWrite"]

    # The blackbox teacher works IN the card: a file inside its working directory is
    # still denied (only an ancestor of the working directory would be dropped).
    s = pol.settings("teacher", tmp_path / "run" / "operators" / "op")
    deny, denied_read = s["permissions"]["deny"], s["sandbox"]["filesystem"]["denyRead"]
    assert f"Read(/{card_key})" in deny and str(card_key) in denied_read
    assert f"Read(/{tree_key})" in deny and str(tree_key) in denied_read
    assert str(tree) not in denied_read                   # the tree itself stays its own
    assert str(card_key) in pol.notice("teacher") and "key" in pol.notice("student")

    # A source-mode run's teacher distils from the reference; nothing is sealed from it.
    src = ap.from_config(cfg, oracle_mode="source", repo_root=tmp_path / "run",
                         teacher_trees=[tree], op="op")
    assert src.settings("teacher", tmp_path / "run")["sandbox"]["filesystem"]["denyRead"] == []
    assert str(card_key) in src.settings("student", tmp_path / "run" / "operators" / "op-ws")[
        "sandbox"]["filesystem"]["denyRead"]


def test_the_loop_passes_the_operator_to_the_policy():
    """cli.main is where the card name is known; a from_config call without it would
    render the rule for the teacher trees only and leave the card's key readable."""
    src = inspect.getsource(cli)
    call = src[src.index("policy = agent_policy_mod.from_config("):]
    call = call[:call.index(")\n")]
    assert "op=args.op" in call


def test_a_source_teacher_is_not_denied_the_source_it_distils_from(tmp_path):
    pol, _, _ = _policy(tmp_path, mode="source")
    s = pol.settings("teacher", tmp_path / "ws")
    assert s["sandbox"]["filesystem"]["denyRead"] == []
    assert "hooks" not in s
    assert "WebFetch" not in s["permissions"]["deny"]
    # ...while its student is still offline and still kept out of it
    assert "WebFetch" in pol.settings("student", tmp_path / "ws")["permissions"]["deny"]


def test_a_deny_path_that_contains_the_working_directory_is_dropped(tmp_path, capsys):
    """A mis-declared reference (or an oracle whose directory is the run tree itself) must
    not take the agent's own tree away from it."""
    ws = tmp_path / "run" / "operators" / "op-ws"
    cfg = {"loop": {"ref_source_paths": [str(tmp_path / "run")], "ref_source_urls": []}}
    pol = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path / "run",
                         teacher_trees=[])
    s = pol.settings("student", ws)
    assert s["sandbox"]["filesystem"]["denyRead"] == []
    assert "working directory" in capsys.readouterr().err


def test_the_notice_states_the_rules_of_the_role(tmp_path):
    pol, ref, _ = _policy(tmp_path)
    student, teacher = pol.notice("student"), pol.notice("teacher")
    assert "No network" in student and str(ref) in student
    assert "must not be copied" in teacher and "No network" not in teacher


# -- installation ------------------------------------------------------------------------

def test_install_writes_both_files_and_keeps_them_out_of_git(tmp_path):
    pol, _, _ = _policy(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    pol.install("student", ws)
    pol.install("teacher", ws)
    pol.install("student", ws)
    settings = json.loads((ws / ".claude" / "settings.json").read_text())
    assert "WebFetch" in settings["permissions"]["deny"]  # the last installed role wins
    policy = json.loads((ws / ".claude" / "forgekernel-agent-policy.json").read_text())
    assert policy["role"] == "student" and policy["offline"] is True
    exclude = (ws / ".git" / "info" / "exclude").read_text().splitlines()
    assert exclude.count(".claude/") == 1
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True,
                          text=True).stdout == ""


def test_install_excludes_the_policy_in_a_linked_worktree(tmp_path):
    """A run's tree can be a linked worktree, where .git is a file, not a directory."""
    pol, _, _ = _policy(tmp_path)
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=main, check=True)
    (main / "f").write_text("x")
    for args in (["add", "f"], ["-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=main, check=True)
    ws = tmp_path / "linked"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(ws)], cwd=main, check=True)
    assert (ws / ".git").is_file()
    pol.install("student", ws)
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True,
                          text=True).stdout == ""


def test_install_moves_a_hand_written_settings_file_aside_once(tmp_path):
    pol, _, _ = _policy(tmp_path)
    ws = tmp_path / "ws"
    (ws / ".claude").mkdir(parents=True)
    (ws / ".claude" / "settings.json").write_text('{"mine": true}')
    pol.install("teacher", ws)
    backup = ws / ".claude" / ("settings.json" + ap.FOREIGN_BACKUP_SUFFIX)
    assert json.loads(backup.read_text()) == {"mine": True}
    pol.install("student", ws)  # ours now: no second backup, no clobbering the first
    assert json.loads(backup.read_text()) == {"mine": True}


def test_every_agent_turn_runs_under_its_roles_policy(tmp_path):
    pol, _, _ = _policy(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    probe = ("python3 -c 'import json,sys; "
             "print(\"ROLE=\" + json.load(open(\".claude/forgekernel-agent-policy.json\"))[\"role\"]); "
             "sys.stdout.write(sys.stdin.read())'")
    be = cli.RealBackend("demo", ws, probe, "n", "e", teacher_agent_cmd=probe,
                         agent_retries=1, agent_retry_sleep=0.0, agent_policy=pol)
    out_s = be.run_agent("PROMPT-S", role="student")
    out_t = be.run_agent("PROMPT-T", role="teacher")
    assert out_s.startswith("ROLE=student") and "No network" in out_s and out_s.rstrip().endswith("PROMPT-S")
    assert out_t.startswith("ROLE=teacher") and "No network" not in out_t


def test_a_backend_without_a_policy_runs_the_prompt_untouched(tmp_path):
    be = cli.RealBackend("demo", tmp_path, "cat", "n", "e", agent_retries=1, agent_retry_sleep=0.0)
    assert be.run_agent("PLAIN") == "PLAIN"
    assert not (tmp_path / ".claude").exists()


# -- the hook ----------------------------------------------------------------------------

POLICY = {"role": "teacher", "oracle_mode": "blackbox", "offline": False,
          "deny_urls": ["github.com/Org/RefLib"]}


@pytest.mark.parametrize("event", [
    {"tool_name": "WebFetch", "tool_input": {"url": "https://github.com/org/reflib/blob/main/k.cuh"}},
    # the same source under the hosts a repository is actually read through
    {"tool_name": "WebFetch", "tool_input": {"url": "https://raw.githubusercontent.com/org/RefLib/main/k.cuh"}},
    {"tool_name": "WebFetch", "tool_input": {"url": "https://api.github.com/repos/org/reflib/contents/csrc"}},
    {"tool_name": "WebFetch", "tool_input": {"url": "https://cdn.jsdelivr.net/gh/org/reflib@main/k.cuh"}},
    {"tool_name": "Bash", "tool_input": {"command": "git clone git@github.com:org/RefLib.git"}},
    {"tool_name": "Bash", "tool_input": {"command": "gh repo clone org/RefLib"}},
    {"tool_name": "Bash", "tool_input": {"command": "pip download ref-lib --no-deps -d /tmp/r"}},
    {"tool_name": "Bash", "tool_input": {"command": "curl -sL https://raw.githubusercontent.com/x; git clone https://github.com/org/RefLib"}},
    {"tool_name": "WebSearch", "tool_input": {"query": "site:github.com/org/RefLib kernel"}},
])
def test_the_hook_refuses_calls_that_name_the_reference_repository(event):
    msg = hook.decide(POLICY, event)
    assert msg and "BLOCKED" in msg


@pytest.mark.parametrize("event", [
    {"tool_name": "WebFetch", "tool_input": {"url": "https://docs.nvidia.com/cutlass/"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 notes/probe.py"}},
    {"tool_name": "Bash", "tool_input": {"command": "grep -rn reflib notes/"}},  # not a fetch
    {"tool_name": "Read", "tool_input": {"file_path": "/x/github.com/org/RefLib"}},  # the sandbox's job
])
def test_the_hook_lets_everything_else_through(event):
    assert hook.decide(POLICY, event) is None


OFFLINE = {"role": "student", "oracle_mode": "blackbox", "offline": True, "deny_urls": []}


@pytest.mark.parametrize("event", [
    {"tool_name": "WebFetch", "tool_input": {"url": "https://docs.nvidia.com/cutlass/"}},
    {"tool_name": "WebSearch", "tool_input": {"query": "cute dsl tma"}},
    {"tool_name": "Bash", "tool_input": {"command": "curl -sL https://example.com/k.cuh -o k.cuh"}},
    {"tool_name": "Bash", "tool_input": {"command": "uv pip install some-kernel-lib"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 -c \"import urllib.request\" && wget https://x/y"}},
    {"tool_name": "Bash", "tool_input": {"command": "nc -z example.com 443"}},
    # a URL anywhere in the command: a script that opens the connection itself
    {"tool_name": "Bash", "tool_input": {"command": "python3 -c \"urlopen('https://example.com/k.cuh')\""}},
])
def test_the_hook_keeps_an_offline_role_off_the_network(event):
    msg = hook.decide(OFFLINE, event)
    assert msg and "offline" in msg


@pytest.mark.parametrize("event", [
    {"tool_name": "Bash", "tool_input": {"command": "git add kernel.py && git commit -m x"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 -c 'print(1)'"}},
    {"tool_name": "Bash", "tool_input": {"command": "grep -rn mbarrier kb/"}},
    # `ssh` is denied as a command, not as a substring: a local file whose name merely
    # starts with it is ordinary work.
    {"tool_name": "Bash", "tool_input": {"command": "python3 ssh_helper.py"}},
    {"tool_name": "Bash", "tool_input": {"command": "cat ssh_notes.md"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 -c \"print('ssh')\""}},
])
def test_the_hook_leaves_an_offline_roles_local_work_alone(event):
    assert hook.decide(OFFLINE, event) is None


# -- the GPU machine -----------------------------------------------------------------

@pytest.mark.parametrize("event", [
    # A student that writes its own submit script drives the GPU outside the driver's
    # ledger.
    {"tool_name": "Bash", "tool_input": {"command": "ssh gpu-box 'nvidia-smi'"}},
    {"tool_name": "Bash", "tool_input": {"command": "scp kernel.py gpu-box:/tmp/"}},
    {"tool_name": "Bash", "tool_input": {"command": "bash -lc 'cd /tmp && ssh gpu-box ls'"}},
    # Wrapped: the name directly after the wrapper's opening quote.
    {"tool_name": "Bash", "tool_input": {"command": "bash -lc 'ssh gpu-box nvidia-smi'"}},
    {"tool_name": "Bash", "tool_input": {"command": 'bash -c "rsync -a . gpu-box:/tmp/x"'}},
    {"tool_name": "Bash", "tool_input": {"command": "tmux new -d -s sub 'ssh gpu-box nvidia-smi'"}},
    {"tool_name": "Bash", "tool_input": {"command": "sh -c 'curl -sO https://x.example/k.py'"}},
])
def test_an_offline_student_may_not_drive_the_gpu_machine_itself(event):
    """A job a student starts on its own is outside the driver's ledger — unrecorded,
    invisible to the run's accounting. The sandbox refuses the connection; this is the
    layer that says which rule did it."""
    msg = hook.decide(OFFLINE, event)
    assert msg and "offline" in msg


def test_the_gpu_machine_is_denied_by_name_for_an_offline_role(tmp_path):
    """The deny rules are the layer that survives --dangerously-skip-permissions,
    so the names have to be there too, not only in the hook's pattern."""
    pol, _, _ = _policy(tmp_path)
    ws = tmp_path / "run" / "operators" / "op-ws"
    deny = pol.settings("student", ws)["permissions"]["deny"]
    assert "Bash(ssh:*)" in deny and "Bash(python -m forgekernel.submit:*)" in deny


def test_an_online_student_keeps_its_deny_list_but_not_the_gpu_machine(tmp_path):
    """student_online (the ref_from_goal session): the network is on, the reads the
    student is denied stay denied, and the GPU machine stays off."""
    import dataclasses
    pol, ref, tree = _policy(tmp_path)
    online = dataclasses.replace(pol, student_online=True)
    ws = tmp_path / "run" / "operators" / "op-ws"
    s = online.settings("student", ws)
    deny = s["permissions"]["deny"]
    assert "WebFetch" not in deny and "Bash(curl:*)" not in deny
    assert "network" not in s["sandbox"]
    assert "Bash(ssh:*)" in deny and "Bash(python3 -m forgekernel.submit:*)" in deny
    assert online.deny_read("student") == pol.deny_read("student")
    assert {str(ref), str(tree)} <= set(s["sandbox"]["filesystem"]["denyRead"])
    assert online.allow_write("student") == pol.allow_write("student")
    doc = online.policy_document("student")
    assert doc["offline"] is False and doc["no_cluster"] is True
    assert doc["deny_urls"] == ["github.com/org/RefLib"]
    assert s["hooks"]["PreToolUse"][0]["hooks"][0]["command"].count("agent_policy_hook.py")
    note = online.notice("student")
    assert "You may go online" in note and "ssh" in note and "No network" not in note
    # the teacher's policy is untouched
    assert online.settings("teacher", ws) == pol.settings("teacher", ws)


def test_the_reference_session_runs_as_an_online_student():
    src = inspect.getsource(cli)
    call = src[src.index("gen = goal_ref_mod.Generation("):]
    call = call[:call.index("make_backend=")]
    assert "student_online=True" in call and "goal_ref=False" in call
    assert "deny_write=tuple(goal_ref_mod.session_read_only_paths(op_dir))" in call


def test_deny_write_paths_are_read_only_to_the_file_tools_and_the_shell(tmp_path):
    import dataclasses
    pol, _, _ = _policy(tmp_path)
    ws = (tmp_path / "run" / "goal_ref" / "ws").resolve()
    ledger = (tmp_path / "run" / "goal_ref" / "ledger.jsonl").resolve()
    cands = (tmp_path / "run" / "goal_ref" / "candidates").resolve()
    ro = dataclasses.replace(pol, student_online=True,
                             deny_write=(ledger, cands, ws.parent))   # an ancestor: skipped
    s = ro.settings("student", ws)
    deny = s["permissions"]["deny"]
    assert f"Edit(/{ledger})" in deny and f"Edit(/{cands}/**)" in deny
    assert f"Edit(/{ws.parent})" not in deny and f"Read(/{ledger})" not in deny
    fs = s["sandbox"]["filesystem"]
    assert {str(ledger), str(cands)} <= set(fs["denyWrite"]) and str(ws.parent) not in fs["denyWrite"]
    assert str(ledger) not in fs["denyRead"]            # the check reads it from the shell
    assert "readable, not writable" in ro.notice("student")
    assert f"Edit(/{ledger})" not in pol.settings("student", ws)["permissions"]["deny"]


ONLINE_STUDENT = {"role": "student", "oracle_mode": "source", "offline": False,
                  "no_cluster": True, "deny_urls": ["github.com/Org/RefLib"]}


@pytest.mark.parametrize("event", [
    {"tool_name": "Bash", "tool_input": {"command": "ssh gpu-box nvidia-smi"}},
    {"tool_name": "Bash", "tool_input": {"command": "/usr/bin/ssh gpu-box ls"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 -m forgekernel.submit op submit"}},
    {"tool_name": "Bash", "tool_input": {"command": "tmux new -d 'scp k.py gpu-box:/tmp/'"}},
])
def test_an_online_student_may_not_drive_the_gpu_machine_either(event):
    msg = hook.decide(ONLINE_STUDENT, event)
    assert msg and "GPU machine" in msg


@pytest.mark.parametrize("event", [
    {"tool_name": "WebFetch", "tool_input": {"url": "https://docs.nvidia.com/cutlass/"}},
    {"tool_name": "WebSearch", "tool_input": {"query": "cute dsl tma"}},
    {"tool_name": "Bash", "tool_input": {"command": "curl -sL https://docs.nvidia.com/x.html"}},
    {"tool_name": "Bash", "tool_input": {"command": "python3 ssh_helper.py"}},
])
def test_an_online_student_reaches_the_web(event):
    assert hook.decide(ONLINE_STUDENT, event) is None


def test_an_online_student_still_may_not_name_the_reference_repository():
    msg = hook.decide(ONLINE_STUDENT, {"tool_name": "WebFetch", "tool_input": {
        "url": "https://raw.githubusercontent.com/org/RefLib/main/k.cuh"}})
    assert msg and "reference" in msg


def test_the_hook_blocks_when_its_policy_cannot_be_read(tmp_path):
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "agent_policy_hook.py"),
                        str(tmp_path / "missing.json")], input="{}", text=True, capture_output=True)
    assert p.returncode == 2 and "BLOCKED" in p.stderr
    good = tmp_path / "p.json"
    good.write_text(json.dumps(POLICY))
    ok = subprocess.run([sys.executable, str(REPO / "scripts" / "agent_policy_hook.py"), str(good)],
                        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}}),
                        text=True, capture_output=True)
    assert ok.returncode == 0
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "agent_policy_hook.py"), str(good)],
                         input=json.dumps({"tool_name": "WebFetch",
                                           "tool_input": {"url": "https://github.com/org/RefLib"}}),
                         text=True, capture_output=True)
    assert bad.returncode == 2 and "reference" in bad.stderr


# -- launch refusals ---------------------------------------------------------------------

def test_a_blackbox_run_must_declare_where_its_reference_lives():
    assert len(ap.declaration_problems({"loop": {}}, "blackbox")) == 2
    assert ap.declaration_problems(
        {"loop": {"ref_source_paths": [], "ref_source_urls": []}}, "blackbox") == []
    assert ap.declaration_problems({"loop": {}}, "source") == []
    assert ap.declaration_problems({"loop": {"ref_source_paths": "~/refs",
                                             "ref_source_urls": []}}, "blackbox")


def test_legs_that_would_not_load_the_project_settings_are_refused(tmp_path):
    assert ap.command_findings(["claude -p --setting-sources project --model x"], {}) == []
    assert ap.command_findings(["claude -p --setting-sources user,local --model x"], {})
    assert ap.command_findings(['claude -p --setting-sources "" --model x'], {})
    assert ap.command_findings(["claude -p --bare --model x"], {})
    assert ap.command_findings(["claude -p --settings /x/off.json"], {})
    assert ap.command_findings(["cd /somewhere && claude -p"], {})
    assert ap.command_findings(["(cd /somewhere && claude -p)"], {})
    assert ap.command_findings(['sh -c "cd /somewhere; claude -p"'], {})
    assert ap.command_findings(["env -C /somewhere claude -p"], {})
    assert ap.command_findings(['out=$(cd /somewhere && claude -p)'], {})
    wrapper = tmp_path / "agent.sh"
    wrapper.write_text('ROOT="$(cd "$(dirname "$0")" && pwd)"\nclaude -p --model "$M"\n')
    assert ap.command_findings([f"bash {wrapper}"], {}) == []  # computing a dir is not a cd

def test_the_student_is_kept_out_of_every_teacher_side_tree_but_its_own(tmp_path):
    cfg = {"loop": {"recon_op": "_recon", "ref_submit_op": "_ref", "arch_ref_submit_op": "_arch",
                    "ref_price_op": "_price", "ref_mode": "x"}}
    trees = {p.name for p in cli._teacher_only_trees(cfg, tmp_path, "op", "blackbox", None)}
    assert trees == {"_recon", "_ref", "_arch", "_price"}


def test_a_source_runs_oracle_is_denied_with_the_package_around_it(tmp_path):
    """The oracle is one file of a reference package: its directory holds the values the
    student must not copy. A list-valued oracle used to be joined into one comma-glued
    string, which denied nothing at all."""
    pkg = tmp_path / "run" / "operators" / "_oracle" / "ref_cute"
    pkg.mkdir(parents=True)
    for name in ("flash_fwd_sm90.py", "interface.py"):
        (pkg / name).write_text("x")
    cfg = {"loop": {"oracle": [str(pkg / "flash_fwd_sm90.py"), str(pkg / "interface.py")]}}

    paths = cli._oracle_paths("source", None, cfg)
    assert paths == [str(pkg / "flash_fwd_sm90.py"), str(pkg / "interface.py")]
    assert cli._resolve_oracle("source", None, cfg) == ", ".join(paths)  # prompt text unchanged

    trees = cli._teacher_only_trees(cfg, tmp_path / "run", "op", "source", paths)
    assert pkg in trees and (pkg / "flash_fwd_sm90.py") in trees

    pol = ap.from_config({"loop": {}}, oracle_mode="source", repo_root=tmp_path / "run",
                         teacher_trees=trees)
    s = pol.settings("student", tmp_path / "run" / "operators" / "op-ws")
    deny, denied_read = s["permissions"]["deny"], s["sandbox"]["filesystem"]["denyRead"]
    assert str(pkg) in denied_read and str(pkg / "flash_fwd_sm90.py") in denied_read
    # `p/**` is everything UNDER p, which for a file is nothing: the bare rule is what
    # refuses the oracle itself.
    assert f"Read(/{pkg / 'flash_fwd_sm90.py'})" in deny
    assert f"Read(/{pkg}/**)" in deny


def test_an_oracle_never_denies_the_run_tree_or_the_home_directory(tmp_path):
    """A single-file oracle sitting at a repository root would otherwise deny the root."""
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "oracle.py").write_text("x")
    trees = cli._teacher_only_trees({"loop": {}}, tmp_path / "run", "op", "source",
                                    [str(tmp_path / "run" / "oracle.py")])
    assert (tmp_path / "run" / "oracle.py") in trees
    assert (tmp_path / "run") not in trees and Path.home() not in trees


def test_the_agent_policy_exit_code_means_nothing_else():
    src = inspect.getsource(cli)
    taken = {int(m) for m in re.findall(r"^\s+return (\d+)\b", src, re.M)}
    assert cli.EXIT_AGENT_POLICY_UNDECLARED not in taken | {
        0, cli.EXIT_WINDOW_UNDECLARED, cli.EXIT_KB_ARCH_MISSING}


def test_a_source_teacher_is_denied_nothing(tmp_path):
    """A source-mode run's teacher distils FROM the reference: the declared paths and
    URLs are denied to the student only."""
    ref = tmp_path / "refs" / "Other"
    ref.mkdir(parents=True)
    base = {"ref_source_paths": [str(ref)], "ref_source_urls": ["github.com/org/Other"]}
    plain = ap.from_config({"loop": dict(base)}, oracle_mode="source",
                           repo_root=tmp_path / "run", teacher_trees=[])
    assert plain.deny_read("teacher") == [] and plain.deny_urls("teacher") == []
    assert ref.resolve() in plain.deny_read("student")


def test_the_oracle_lint_tokens_can_be_turned_off_per_run(tmp_path):
    src = tmp_path / "oracle" / "some_reference_kernel.py"
    src.parent.mkdir()
    src.write_text("def special_helper_name():\n    pass\n")
    on = cli._oracle_lint_tokens({"loop": {}}, str(src))
    assert "special_helper_name" in on
    assert cli._oracle_lint_tokens({"loop": {"kb_lint_oracle_tokens": False}}, str(src)) == []
