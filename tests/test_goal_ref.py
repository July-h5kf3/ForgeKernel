"""ref_from_goal (loop/goal_ref.py): no reference to distil from -> a /goal session writes
one on the card's own harness, the loop accepts the best correct candidate itself and wires
it in as the oracle. The agent, tmux and the GPU are stubs here: the agent is a script
that edits the kernel and calls the real evaluation client, the GPU is a backend whose
verdict is written into the kernel source."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from forgekernel.loop import goal_ref as gr
from forgekernel.loop.parse import parse_eval

SRC = Path(gr.__file__).resolve().parents[2]


# -- fixtures -----------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def _card(root: Path, op: str = "demo", *, time_reference: bool = True) -> Path:
    op_dir = root / "operators" / op
    (op_dir / "eval" / "leaderboard").mkdir(parents=True)
    (op_dir / "kernel.py").write_text('"""stub: def run(x, out) — fill out = x"""\n'
                                      "def run(x, out):\n    raise NotImplementedError\n")
    (op_dir / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(
        {"kernel_file": "kernel.py", "time_reference": time_reference}))
    (op_dir / "README.md").write_text("demo card\n")
    _git(op_dir, "init", "-q", "-b", "main")
    _git(op_dir, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(op_dir, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "card")
    return op_dir


class StubGPU:
    """submit_gpu reads the committed kernel: `# MS=<x>` is its reading, `# OK` makes it
    correct; the in-job reference reads 1.0 ms."""

    def __init__(self, tree: Path, kernel_file: str = "kernel.py"):
        self.tree, self.kernel_file = tree, kernel_file
        self.last_job_id = ""
        self.jobs: list[tuple[str, str]] = []
        self.exec_scripts: list[str] = []

    def submit_gpu(self, branch: str, mode: str) -> str:
        self.jobs.append((branch, mode))
        self.last_job_id = str(1000 + len(self.jobs))
        if mode.startswith("cmd:"):
            script = subprocess.run(["git", "-C", str(self.tree), "show",
                                     f"{branch}:{gr.EXEC_SCRIPT}"], capture_output=True,
                                    text=True).stdout
            self.exec_scripts.append(script)
            return f"[gpu_entry.sh] ran {mode[4:]}\n[gpu_entry.sh] task=cmd finished\n"
        src = subprocess.run(["git", "-C", str(self.tree), "show", f"{branch}:{self.kernel_file}"],
                             capture_output=True, text=True, check=True).stdout
        ms = float(re.search(r"MS=([\d.]+)", src).group(1))
        ok = "# OK" in src
        obj = {"correctness": {"pass": ok, **({} if ok else {"error": "max_diff 3.1 > tol"})},
               "performance": {"device_median_ms": ms, "bwd_median_ms": ms * 1.1,
                               "ref_device_ms": 1.0, "vs_ref_device": 1.0 / ms,
                               "ref_median_ms": 1.1, "vs_ref": 1.0 / ms,
                               "device_window_basis": "cupti_kernel_span"}}
        return "=== JSON ===\n" + json.dumps(obj) + "\n[gpu_entry.sh] task=all finished\n"


AGENT = textwrap.dedent('''\
    import json, os, pathlib, subprocess, sys, time
    sys.path.insert(0, {src!r})
    from forgekernel.loop import goal_ref
    ws = pathlib.Path.cwd()
    spec = json.loads((pathlib.Path(os.environ["FK_GOAL_TURN_DIR"]) / "spec.json").read_text())
    trace = pathlib.Path({trace!r})
    with open(str(trace) + ".env", "a") as f:
        f.write(json.dumps({{k: os.environ.get(k) for k in
                            ("FK_GOAL_DEADLINE",)}}) + "\\n")
    def run(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        with open(trace, "a") as f:
            f.write("$ " + cmd + "\\n" + r.stdout + r.stderr + "\\n")
        return r.stdout
    for body in {kernels!r}:
        (ws / "kernel.py").write_text(body)
        run(goal_ref.command("eval", str(ws)))
    if {exec_cmd!r}:
        (ws / "probes").mkdir(exist_ok=True)
        (ws / "probes" / "p.py").write_text("print(1)")
        run(goal_ref.command("exec", str(ws), "--", {exec_cmd!r}))
    if {tamper!r}:       # what the session's policy forbids: rewrite a judged candidate
        p = ws.parent / "candidates" / {tamper!r} / "kernel.py"
        p.write_text(p.read_text() + "# rewritten after its evaluation\\n")
    pathlib.Path(spec["reply"]).write_text("# report\\ndesign: copy\\n")
    run(spec["check"])
    time.sleep({linger})
    ''')


def _gen(tmp_path: Path, kernels: list[str], *, hours: float = 1.0, linger: float = 0,
         exec_cmd: str = "", cfg: dict | None = None, target_ms=None,
         time_reference: bool = True, kill_grace_s: float = 300.0, report_s: float = 0.0,
         tamper: str = ""):
    root = tmp_path / "repo"
    op_dir = _card(root, time_reference=time_reference)
    trace = tmp_path / "agent-trace.txt"
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT.format(src=str(SRC), trace=str(trace), kernels=kernels,
                                  exec_cmd=exec_cmd, linger=linger, tamper=tamper))
    gpus: list[StubGPU] = []

    def make_backend(op, tree):
        gpus.append(StubGPU(tree))
        return gpus[-1]

    g = gr.Generation(
        op="demo", op_dir=op_dir, root=root, cfg=cfg or {}, kernel_file="kernel.py",
        language="cutedsl", gate_window="device", target_ms=target_ms,
        agent_cmd=f"{sys.executable} {agent}", policy=None, make_backend=make_backend,
        settings=gr.Settings(hours=hours), poll_s=0.05,
        kill_grace_s=kill_grace_s, report_s=report_s, log=lambda s: None,
        extra_env={"FK_GOALREF_POLL_S": "0.05", "FK_GOALREF_WAIT_S": "60"})
    return g, gpus, trace


WRONG_FAST = "def run(x, out):\n    pass\n# MS=0.5\n"
SLOW_OK = "def run(x, out):\n    out.copy_(x)\n# MS=2.0 # OK\n"
FAST_OK = "def run(x, out):\n    out.copy_(x)\n# MS=0.8 # OK\n"


def test_uncertain_orca_launch_does_not_restart_reference_generation(tmp_path):
    g, _, _ = _gen(tmp_path, [])
    failing = tmp_path / "uncertain.py"
    failing.write_text('import sys\nsys.stderr.write("FK_AGENT_UNCERTAIN: lost receipt")\n'
                       'raise SystemExit(1)\n')
    g.agent_cmd = f"{sys.executable} {failing}"
    with pytest.raises(gr.GoalRefError, match="FK_AGENT_UNCERTAIN"):
        gr.generate(g)
    state = gr.load_state(g.op_dir)
    assert len(state["attempts"]) == 1


# -- the decision -------------------------------------------------------------------------

def test_mode_resolution_toml_then_auto():
    assert gr.resolve_mode({}) == ("auto", "default")
    assert gr.resolve_mode({"loop": {"ref_from_goal": "off"}}) == ("off", "[loop] ref_from_goal")
    assert gr.resolve_mode({"loop": {"ref_from_goal": False}})[0] == "off"
    for bad in ("sometimes", "always"):
        with pytest.raises(gr.GoalRefError):
            gr.resolve_mode({"loop": {"ref_from_goal": bad}})


@pytest.mark.parametrize("mode,oracle_mode,oracle,milestone,action", [
    ("auto", "source", [], False, "generate"),
    ("auto", "source", ["/x/oracle.py"], False, "off"),
    ("auto", "source", [], True, "off"),
    ("auto", "blackbox", [], False, "off"),
    ("off", "source", [], False, "off"),
])
def test_when_a_reference_is_generated(tmp_path, mode, oracle_mode, oracle, milestone, action):
    p = gr.plan(tmp_path, mode=mode, oracle_mode=oracle_mode, oracle_paths=oracle,
                milestone_valid=milestone)
    assert p.action == action, p.why


def test_state_on_disk_decides_reuse_resume_or_refusal(tmp_path):
    ref = tmp_path / "goal_ref" / "reference" / "kernel.py"
    ref.parent.mkdir(parents=True)
    ref.write_text("x")
    kw = dict(mode="auto", oracle_mode="source", oracle_paths=[], milestone_valid=True)
    gr.save_state(tmp_path, {"status": "accepted", "accepted": {"reference": [str(ref)]}})
    assert gr.plan(tmp_path, **kw).action == "reuse"      # even with the milestone written
    gr.save_state(tmp_path, {"status": "generating", "deadline": time.time()})
    assert gr.plan(tmp_path, **kw).action == "resume"
    gr.save_state(tmp_path, {"status": "failed", "failure": "nothing correct"})
    with pytest.raises(gr.GoalRefError, match="nothing correct"):
        gr.plan(tmp_path, **kw)
    assert gr.plan(tmp_path, **{**kw, "mode": "off"}).action == "off"


# -- the speed bar ------------------------------------------------------------------------

def test_the_bar_prefers_target_then_declared_arch_then_in_job_reference(tmp_path):
    _card(tmp_path)
    op_dir = tmp_path / "operators" / "demo"
    b = gr.resolve_bar({}, op_dir, target_ms=0.3, ratio=1.0, gate_window="device")
    assert b["kind"] == "target_ms" and gr.bar_met(b, 0.29, None) and not gr.bar_met(b, 0.31, 9)
    b = gr.resolve_bar({"loop": {"arch_ref_ms": 2.0, "arch_ref_device_ms": 1.0}}, op_dir,
                       target_ms=None, ratio=1.0, gate_window="device")
    assert b["kind"] == "arch_ref" and b["ms"] == 1.0
    b = gr.resolve_bar({}, op_dir, target_ms=None, ratio=1.05, gate_window="device")
    assert b["kind"] == "in_job_ref"
    assert gr.bar_met(b, 9.0, 1.06) and not gr.bar_met(b, 0.1, 1.0) and not gr.bar_met(b, 0.1, None)


def test_no_baseline_at_all_means_no_early_finish(tmp_path):
    """No bar: the first correct (usually naive) kernel must not end the session — it
    would become the oracle and anchor the target. The session runs to its deadline."""
    op_dir = _card(tmp_path, time_reference=False)
    b = gr.resolve_bar({}, op_dir, target_ms=None, ratio=1.0, gate_window="device")
    assert b["kind"] == "none"
    assert not gr.bar_met(b, None, None) and not gr.bar_met(b, 0.001, 9.0)
    assert "deadline" in b["text"]


# -- what a reference may not be ----------------------------------------------------------

def test_library_calls_are_found_and_the_stub_contract_text_is_not():
    clean = '"""contract: out = torch.matmul(a, b)"""\nimport torch\n@cute.jit\ndef run(a, b, out):\n    k(a, b, out)  # not torch.mm(a, b)\n'
    assert gr.library_findings(clean, "kernel.py") == []
    dirty = "import flashinfer\ndef run(a, b, out):\n    out.copy_(a @ b)\n    torch.softmax(a, -1)\n"
    found = gr.library_findings(dirty, "kernel.py")
    assert any("flashinfer" in f for f in found)
    assert any("@" in f for f in found)
    assert any("softmax" in f for f in found)
    cu = "#include <cublas_v2.h>\nvoid run() { cublasGemmEx(h); } // cudnnFoo(\n"
    assert len(gr.library_findings(cu, "kernel.cu")) == 2


# -- the condition and the check ----------------------------------------------------------

def test_the_condition_fits_and_names_the_check(tmp_path):
    ws = tmp_path / "goal_ref" / "ws"
    text = gr.condition_text(ws=ws, kernel_file="kernel.py", deadline=time.time(),
                             reply=ws / ".fk-turn" / "goalref" / "reply.md")
    assert len(text) <= gr.CONDITION_MAX
    assert gr.command("check", str(ws.parent)) in text and gr.OK_LINE in text


def test_check_wants_the_report_and_the_bar_or_the_deadline(tmp_path):
    paths = gr.Paths.of(tmp_path)
    reply = paths.turn / "reply.md"
    gr.save_state(tmp_path, {"status": "generating", "deadline": 1000.0, "reply": str(reply)})
    ok, why = gr.check(paths.root, now=10.0)
    assert not ok and any("reply.md" in w for w in why) and any("speed bar" in w for w in why)
    reply.parent.mkdir(parents=True)
    reply.write_text("report")
    assert not gr.check(paths.root, now=10.0)[0]
    gr._append(paths.ledger, {"kind": "eval", "n": 1, "eligible": True, "reading_ms": 1.0,
                              "bar_met": True})
    assert gr.check(paths.root, now=10.0)[0]
    paths.ledger.unlink()
    assert gr.check(paths.root, now=2000.0)[0]                    # past the deadline


# -- the whole generation, stubbed --------------------------------------------------------

def test_generation_accepts_the_fastest_correct_candidate(tmp_path):
    g, gpus, trace = _gen(tmp_path, [WRONG_FAST, SLOW_OK, FAST_OK], exec_cmd="python3 probes/p.py")
    state = gr.generate(g)
    assert state["status"] == "accepted", trace.read_text()
    acc = state["accepted"]
    assert acc["candidate"] == 3 and acc["runs"] == 2 and acc["bar_met"]
    ref = Path(acc["reference"][0])
    assert ref.read_text() == FAST_OK and ref.parent == gr.Paths.of(g.op_dir).reference
    assert Path(acc["reference"][1]).name == "NOTES.md"
    rows = gr.load_ledger(gr.Paths.of(g.op_dir).ledger)
    evals = [r for r in rows if r["kind"] == "eval"]
    assert [r["correct"] for r in evals] == [False, True, True]
    assert [r["bar_met"] for r in evals] == [False, False, True]
    assert [r["kind"] for r in rows].count("accept") == 2
    assert [r["kind"] for r in rows].count("exec") == 1
    # every job ran the card's tree with only the kernel (and probes) changed
    modes = [m for _, m in gpus[0].jobs]
    assert modes.count("all") == 5 and f"cmd:bash {gr.EXEC_SCRIPT}" in modes
    assert gpus[0].exec_scripts == ["#!/bin/bash\npython3 probes/p.py\n"]
    out = trace.read_text()
    assert gr.DONE_LINE in out and "NOT ELIGIBLE" not in out and gr.OK_LINE in out
    # the session's leg saw the goal environment
    assert (gr.Paths.of(g.op_dir).turn / "spec.json").is_file()
    assert state["arch"]["declare"]
    cfg, arch = gr.apply({"loop": {"ref_cmd": "x"}}, state)
    assert arch["declare"]
    assert cfg["loop"]["oracle"] == acc["reference"] and cfg["loop"]["oracle_mode"] == "source"
    assert cfg["loop"]["kb_lint_oracle_tokens"] is False
    assert cfg["loop"]["arch_ref_device_ms"] == 0.8
    assert cfg["loop"]["arch_ref_device_basis"] == "cupti_kernel_span"
    assert "ref_mode" not in cfg["loop"]          # the card's own library reference stays
    bare, _ = gr.apply({}, state)                 # none: the loop measures the generated one
    assert bare["loop"]["ref_mode"] == gr.REF_TASK
    assert bare["loop"]["ref_submit_op"] == "_demogoalref"
    tree = g.root / "operators" / "_demogoalref"
    assert (tree / "kernel.py").read_text() == FAST_OK          # its HEAD is the reference
    assert (tree / gr.MEASURE_SCRIPT).read_text() == gr.MEASURE_SCRIPT_TEXT
    assert "perf" in gr.MEASURE_SCRIPT_TEXT and "ncu" in gr.MEASURE_SCRIPT_TEXT
    import shutil
    shutil.rmtree(tree)
    gr.ensure_reference_tree(g.root, "demo", g.op_dir, state)   # a reuse rebuilds it
    assert (tree / "kernel.py").read_text() == FAST_OK
    # resume: the reference is reused, nothing runs again
    assert gr.plan(g.op_dir, mode="auto", oracle_mode="source", oracle_paths=[],
                   milestone_valid=False).action == "reuse"


def test_a_library_call_is_never_the_reference(tmp_path):
    lib = "import flashinfer\ndef run(x, out):\n    flashinfer.copy(x, out)\n# MS=0.3 # OK\n"
    g, _, trace = _gen(tmp_path, [lib, SLOW_OK], hours=3 / 3600, kill_grace_s=0.5)
    state = gr.generate(g)
    assert state["accepted"]["candidate"] == 2          # the slower, honest one
    assert "NOT ELIGIBLE" in trace.read_text()
    assert not state["accepted"]["bar_met"]
    # slower than the same-job library reference: the library keeps anchoring the target
    assert not state["arch"]["declare"]


def test_nothing_correct_fails_and_refuses_the_next_launch(tmp_path):
    g, _, _ = _gen(tmp_path, [WRONG_FAST], hours=2 / 3600, kill_grace_s=0.5)
    with pytest.raises(gr.GoalRefError, match="no eligible candidate"):
        gr.generate(g)
    assert gr.load_state(g.op_dir)["status"] == "failed"
    with pytest.raises(gr.GoalRefError):
        gr.plan(g.op_dir, mode="auto", oracle_mode="source", oracle_paths=[],
                milestone_valid=False)


def test_the_deadline_ends_a_session_that_does_not_stop(tmp_path):
    g, _, _ = _gen(tmp_path, [SLOW_OK], hours=4 / 3600, linger=600, kill_grace_s=0.5)
    t0 = time.time()
    state = gr.generate(g)
    assert time.time() - t0 < 60
    assert state["status"] == "accepted" and state["accepted"]["candidate"] == 1
    assert state["attempts"][0]["rc"] != 0          # ended from outside


def test_the_session_keeps_a_report_window_past_the_deadline(tmp_path):
    """The leg's clock runs to deadline + report_s: a session still writing its report
    when the deadline passes is not ended there, and the leg is told the same bound."""
    g, _, trace = _gen(tmp_path, [SLOW_OK], hours=5 / 3600, linger=7, report_s=600,
                       kill_grace_s=0.5)
    t0 = time.time()
    state = gr.generate(g)
    assert time.time() - t0 < 60                    # the leg ended itself, nobody waited
    assert state["status"] == "accepted"
    assert [a["rc"] for a in state["attempts"]] == [0]      # not killed at the deadline
    env = json.loads(Path(str(trace) + ".env").read_text().splitlines()[0])
    assert int(env["FK_GOAL_DEADLINE"]) == int(state["deadline"] + 600)


def test_a_candidate_rewritten_after_its_evaluation_is_not_accepted(tmp_path):
    g, gpus, _ = _gen(tmp_path, [SLOW_OK, FAST_OK], tamper="002")
    state = gr.generate(g)
    assert state["accepted"]["candidate"] == 1
    assert state["acceptance"][0] == {"candidate": 2, "ok": False,
                                      "error": "the kernel file changed since it was evaluated"}
    assert not [b for b, _ in gpus[0].jobs if b.startswith("accept-002")]


def test_the_sessions_records_are_read_only_to_it(tmp_path):
    g, _, _ = _gen(tmp_path, [FAST_OK])
    gr.generate(g)
    paths = gr.Paths.of(g.op_dir)
    ro = gr.session_read_only_paths(g.op_dir)
    assert {paths.state.resolve(), paths.ledger.resolve(), paths.candidates.resolve(),
            (paths.turn / "spec.json").resolve()} <= set(ro)
    assert all(p.exists() for p in ro)       # created before the policy names them
    ws = paths.ws.resolve()
    assert not [p for p in ro if p == ws or p in ws.parents]
    assert not (paths.turn / "reply.md").resolve() in ro      # the agent writes that one


def test_requests_after_the_deadline_get_no_gpu(tmp_path):
    g, gpus, _ = _gen(tmp_path, [SLOW_OK])
    paths = gr.Paths.of(g.op_dir)
    tree = gr.ensure_tree(g.root, g.op, g.op_dir, fresh=True)
    ws = gr.ensure_ws(paths, g.op_dir)
    (ws / "kernel.py").write_text(SLOW_OK)
    (ws / gr.SPOOL / "context.json").write_text(json.dumps({"kernel_file": "kernel.py"}))
    rid, _ = gr._new_request(ws, "eval", "kernel.py", {})
    svc = gr.Service(op="demo", op_dir=g.op_dir, cfg={}, paths=paths, tree=tree,
                     kernel_file="kernel.py", bar={"kind": "none", "text": "-"},
                     deadline=time.time() - 1, backend=StubGPU(tree), log=lambda s: None)
    assert svc.serve_pending() == 1
    text = (paths.spool / "responses" / f"{rid}.txt").read_text()
    assert gr.DEADLINE_LINE in text and not svc.backend.jobs


def test_a_foreign_directory_is_never_taken_as_the_submission_tree(tmp_path):
    root = tmp_path / "repo"
    op_dir = _card(root)
    (root / "operators" / gr.tree_op("demo")).mkdir()
    with pytest.raises(gr.GoalRefError, match="not a goal-ref submission tree"):
        gr.ensure_tree(root, "demo", op_dir, fresh=True)


def test_the_eval_record_is_the_drivers_parse(tmp_path):
    """The verdict is parse_eval's: what the session is judged by is what the student
    is judged by."""
    tree = tmp_path / "t"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    (tree / "kernel.py").write_text(FAST_OK)
    _git(tree, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(tree, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    ev = parse_eval(StubGPU(tree).submit_gpu("main", "all"))
    assert ev.correctness and gr.gate_reading(ev, "device") == 0.8
    assert gr.gate_vs_ref(ev, "device") == pytest.approx(1.25)


# -- the loop CLI's side ------------------------------------------------------------------

def test_the_student_never_reads_a_generated_reference(tmp_path):
    from forgekernel.loop.cli import _teacher_only_trees
    assert not set(gr.teacher_only_paths(tmp_path, "demo")) & set(
        _teacher_only_trees({}, tmp_path, "demo", "source", []))     # none generated here
    for p in gr.teacher_only_paths(tmp_path, "demo"):
        p.mkdir(parents=True)
    trees = _teacher_only_trees({}, tmp_path, "demo", "source", [])
    assert tmp_path / "operators" / "demo" / "goal_ref" in trees
    assert tmp_path / "operators" / "_demogoalref" in trees
    assert not set(gr.teacher_only_paths(tmp_path, "demo")) & set(
        _teacher_only_trees({}, tmp_path, "demo", "source", [], goal_ref=False))


def test_a_declared_oracle_stays_denied_when_the_generated_one_replaces_it():
    """A reuse drops the declared oracle from cfg; the student's deny list keeps it."""
    from forgekernel.loop.cli import _student_denied_oracle
    declared = ["/refs/real/kernel.py"]
    generated = {"loop": {"oracle": ["/run/goal_ref/reference/kernel.py"]}}
    assert _student_denied_oracle("source", None, generated, declared) == [
        "/run/goal_ref/reference/kernel.py", "/refs/real/kernel.py"]
    assert _student_denied_oracle("source", None, {"loop": {"oracle": declared}},
                                  declared) == declared            # no duplicate


def test_a_previous_runs_generated_reference_is_contamination(tmp_path):
    from forgekernel.loop.cli import fresh_start_contamination
    op_dir = tmp_path / "op"
    (op_dir / "student").mkdir(parents=True)
    st = op_dir / "student" / "status.jsonl"
    st.touch()
    gr.save_state(op_dir, {"status": "accepted"})
    found = fresh_start_contamination(op_dir, op_dir / "milestone.md", st,
                                      op_dir / "meta_status.jsonl", None)
    assert any("ref_from_goal" in f for f in found)


def test_the_agent_script_that_serves_the_session_exists():
    """The /goal session is served by scripts/agent-claude.sh handing the turn to
    scripts/claude-goal.py (FK_AGENT_DRIVER=goal); `forge run` always builds that command."""
    scripts = Path(gr.__file__).resolve().parents[3] / "scripts"
    assert (scripts / "agent-claude.sh").is_file() and (scripts / "claude-goal.py").is_file()
    assert "claude-goal" in (scripts / "agent-claude.sh").read_text(encoding="utf-8")


def test_an_orphaned_leg_is_ended_only_when_it_is_ours(tmp_path):
    g, _, _ = _gen(tmp_path, [SLOW_OK])
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(120)\n")
    g.agent_cmd = f"{sys.executable} {sleeper} fkgoalref-orphan-test"
    proc = subprocess.Popen(g.agent_cmd, shell=True, start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                             start_new_session=True)
    try:
        state = {"attempts": [{"n": 1, "pgid": proc.pid}, {"n": 2, "pgid": other.pid}]}
        gr.save_state(g.op_dir, {"status": "generating"})
        import forgekernel.loop.goal_ref as mod
        orig = mod.time.sleep
        mod.time.sleep = lambda s: orig(min(s, 1.0))
        try:
            gr._reap_orphan(g, state)
        finally:
            mod.time.sleep = orig
        assert proc.wait(timeout=10) != 0
        assert other.poll() is None                 # not the agent command: left alone
        assert all("end" in r for r in state["attempts"])
    finally:
        for p in (proc, other):
            if p.poll() is None:
                p.kill()


@pytest.mark.parametrize("loop,target,vs,declare", [
    ({}, None, None, True),                                          # nothing else known
    ({"arch_ref_ms": 0.03184, "arch_ref_device_ms": 0.03184}, None, None, True),  # T_b slower
    ({"arch_ref_ms": 0.01, "arch_ref_device_ms": 0.01}, None, None, False),       # faster declared
    ({"arch_ref_mode": "binref"}, None, None, False),                 # measured: never replaced
    ({}, 0.5, None, False),                                          # --target-ms wins
    ({}, None, 0.9, False),                                          # same-job library faster
    ({}, None, 1.3, True),
])
def test_the_target_anchors_the_fastest_known_reference(loop, target, vs, declare):
    acc = {"reading_ms": 0.021, "device_ms": 0.021, "e2e_ms": 0.025, "vs_ref_gate": vs,
           "reference": ["/x/kernel.py"], "candidate": 2, "jobs": ["1", "2"], "runs": 2,
           "basis": "cupti_kernel_span", "n_workloads": 1, "workloads": []}
    cfg = {"loop": dict(loop, arch_ref_name="T_b", arch_ref_workloads=["wl0"])}
    d = gr.arch_decision(cfg, target, acc)
    assert d["declare"] is declare, d["why"]
    out, d2 = gr.apply(cfg, {"accepted": acc}, target_ms=target)
    assert d2 == d and out["loop"]["oracle"] == ["/x/kernel.py"]
    if declare:
        assert out["loop"]["arch_ref_device_ms"] == 0.021 and out["loop"]["arch_ref_ms"] == 0.025
        assert "arch_ref_workloads" not in out["loop"]          # the old declaration's
        assert "goal-generated" in out["loop"]["arch_ref_name"]
    else:
        assert out["loop"].get("arch_ref_device_ms") == loop.get("arch_ref_device_ms")
