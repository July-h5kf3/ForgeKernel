"""The launch flow: the judge's files, the generated trees, loop.toml, and `forge run` end
to end with the agent, the GPU and the loop stood in for."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from forgekernel import judge as judge_mod
from forgekernel import launch, taskcard
from forgekernel.localenv import load_machine
from forgekernel.runs import list_runs, new_run, open_run
from forgekernel.task import load_task

from test_leaderboard import fib_root  # noqa: F401 -- a tiny flashinfer-trace dataset

DESCRIPTION = "y = x / sqrt(mean(x^2) + 1e-6) * w over the last axis\n"
SHAPES = [{"name": "small", "axes": {"m": 4, "n": 16}},
          {"name": "large", "axes": {"m": 64, "n": 16}}]
TASK_TOML = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
REF_TOML = TASK_TOML + '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'
REF_PY = ("import torch\n\ndef run(x, w):\n"
          "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n")

DEFINITION = {"inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                         "w": {"shape": ["n"], "dtype": "float32"}},
              "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
JUDGE = """import torch

def run(x, w):
    y = x.double()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()).float()

def make_inputs(axes, group, seed, device):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(axes["m"], axes["n"], generator=g)
    if group == "spiky":
        x[:, 0] = 1e3
    if group == "tiny":
        x = x * 1e-4
    return {"x": x.to(device), "w": torch.ones(axes["n"]).to(device)}
"""
GROUPS = {s["name"]: [{"name": "plain", "intent": "randn", "seed": 1},
                      {"name": "spiky", "intent": "one huge column", "seed": 2},
                      {"name": "tiny", "intent": "values near zero", "seed": 3}]
          for s in SHAPES}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch place for runs and locks; the harness itself stays the real repository."""
    monkeypatch.setenv("FORGEKERNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGEKERNEL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("FORGEKERNEL_OPERATORS_DIR", raising=False)
    # this machine, whatever the developer's local.env says (the environment wins over it)
    monkeypatch.setenv("FK_DELIVERY", "local")
    monkeypatch.setenv("FK_GPU", "0")
    # no GPU here: the probe job is stood in for (a test that cares overrides it)
    monkeypatch.setattr(launch, "probe_ncu", lambda task, op_dir: True)
    return tmp_path


def _task(tmp_path, toml=REF_TOML, files=None, shapes=SHAPES) -> Path:
    d = tmp_path / "tasks" / "rmsnorm"
    (d / "reference").mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "description.md").write_text(DESCRIPTION)
    (d / "shapes.json").write_text(json.dumps(shapes))
    (d / "reference" / "run.py").write_text(REF_PY)
    for rel, data in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return d


def _start(task_dir) -> int:
    """`forge new` then `forge run`, the way a user starts a run."""
    assert launch.cmd_new([str(task_dir)]) == 0
    return launch.cmd_run([str(list_runs()[-1].dir)])


def _loop_ends(hook=None, stop="target", confirm=None):
    """A stand-in for the loop: sees its argv, ends with ``stop`` in its summary (and
    ``confirm`` as its final confirmation, when given)."""
    from forgekernel.loop.cli import SUMMARY_FILE

    def main(argv):
        if hook is not None:
            hook(argv)
        d = Path(os.environ["FORGEKERNEL_OPERATORS_DIR"]) / argv[0]
        (d / SUMMARY_FILE).write_text(json.dumps({"stop": stop}))
        if confirm is not None:
            (d / "best_confirmation.json").write_text(json.dumps(confirm))
        return 0
    return main


def _write_check(d: Path, *, definition=DEFINITION, judge=JUDGE, groups=GROUPS) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / judge_mod.DEFINITION).write_text(json.dumps(definition))
    (d / judge_mod.JUDGE).write_text(judge)
    (d / judge_mod.GROUPS).write_text(json.dumps(groups))


def _st_file(tensors: dict) -> bytes:
    """A safetensors file without torch: header + raw bytes."""
    header, blobs, off = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blobs += raw
        off += len(raw)
    h = json.dumps(header).encode()
    return len(h).to_bytes(8, "little") + h + blobs


# -- the judge's files ----------------------------------------------------------------------


def test_good_judge_files_pass_the_static_check(tmp_path):
    task = load_task(_task(tmp_path))
    _write_check(tmp_path / "check")
    assert judge_mod.check_problems(tmp_path / "check", task) == []


def test_judge_files_are_checked_against_the_task(tmp_path):
    task = load_task(_task(tmp_path))
    bad_def = {"inputs": {"x": {"shape": ["m", "k"], "dtype": "float32"},
                          "w": {"shape": ["n"], "dtype": "float7"}},
               "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
    bad_judge = "def run(w, x):\n    return x\n"
    few = {"small": [{"name": "plain", "intent": "randn"}], "huge": []}
    _write_check(tmp_path / "check", definition=bad_def, judge=bad_judge, groups=few)
    text = "\n".join(judge_mod.check_problems(tmp_path / "check", task))
    for needle in ("dimension 'k'", "dtype must be one of", "must take the definition's inputs",
                   "defines no top-level make_inputs", "'huge' is not one of the task's shapes",
                   "needs at least 3"):
        assert needle in text, needle


def test_the_tasks_own_input_files_are_checked_against_the_signature(tmp_path):
    good = _st_file({"x": ("F32", [4, 16], b"\0" * 256), "w": ("F32", [16], b"\0" * 64)})
    bad = _st_file({"x": ("F16", [4, 16], b"\0" * 128), "w": ("F32", [16], b"\0" * 64)})
    shapes = [{"name": "small", "axes": {"m": 4, "n": 16},
               "inputs": [{"name": "zeros", "file": "inputs/good.safetensors"},
                          {"name": "halves", "file": "inputs/bad.safetensors"}]}]
    task = load_task(_task(tmp_path, shapes=shapes, files={"inputs/good.safetensors": good,
                                                           "inputs/bad.safetensors": bad}))
    groups = {"small": [{"name": "plain", "intent": "randn"}]}
    _write_check(tmp_path / "check", groups=groups)
    problems = judge_mod.check_problems(tmp_path / "check", task)
    assert len(problems) == 1 and "halves" in problems[0] and "float16" in problems[0]


def test_the_judge_prompt_shows_the_reference_only_in_source_mode(tmp_path):
    task = load_task(_task(tmp_path))
    with_src = judge_mod.judge_prompt(task, tmp_path / "check", reference_source=task.reference.source)
    without = judge_mod.judge_prompt(task, tmp_path / "check", reference_source=None)
    assert str(task.reference.source) in with_src and "description alone" in without
    assert "small: axes" in with_src and "add 3" in with_src


# -- the trees ------------------------------------------------------------------------------


def _git(tree, *args) -> str:
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True,
                          check=True).stdout


def _run_with_check(repo, toml=REF_TOML, **kw):
    src = load_task(_task(repo, toml=toml, **kw))
    run = new_run(src, machine=load_machine(repo, environ={}))
    task = load_task(run.task_dir, name=run.task_name)
    _write_check(run.dir / judge_mod.CHECK_DIR)
    return run, task, judge_mod.load_check(run.dir / judge_mod.CHECK_DIR)


def test_write_card_generates_a_committed_task_card(repo):
    run, task, check = _run_with_check(repo)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    assert op_dir == run.operators_dir / "rmsnorm"
    problem = json.loads((op_dir / taskcard.PROBLEM).read_text())
    assert problem["suite"] == "forgekernel" and problem["timing"] == "device"
    assert [g["name"] for g in problem["group"]] == ["small", "large"]
    assert [g["name"] for g in problem["workload"]["input_groups"]] == ["plain", "spiky", "tiny"]
    assert problem["definition"]["reference"] == JUDGE
    assert problem["time_reference"] is False
    stub = (op_dir / "kernel.py").read_text()
    assert "def run(x, w):" in stub and "small" in stub and "large" in stub
    entry = (op_dir / "gpu_entry.sh").read_text()
    assert "calibrate)" in entry and 'CUTE_DSL_ARCH:-sm_100a' in entry
    sub = tomllib.loads((op_dir / "submit.toml").read_text())
    assert sub["repo"]["delivery"] == "local" and sub["exec"]["gpu"] == "0"
    assert _git(op_dir, "status", "--porcelain") == ""
    assert _git(op_dir, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_seal_reference_rides_the_card_and_writes_the_recon_tree(repo):
    run, task, check = _run_with_check(repo)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    impl = taskcard.seal_reference(run, task, op_dir)
    assert impl["kind"] == "sealed" and impl["entry"] == "run" and impl["dps"] is False
    problem = json.loads((op_dir / taskcard.PROBLEM).read_text())
    assert problem["reference_impl"] == impl and problem["time_reference"] is True
    ref = run.operators_dir / "_rmsnormref"
    assert (ref / "sealref.py").is_file() and (ref / "ref_bin" / "ref_payload.bin").is_file()
    # the key never enters git, the plaintext never enters either tree
    for tree in (op_dir, ref):
        assert "FK_REF_KEY" in (tree / "submit.local.toml").read_text()
        assert "submit.local.toml" not in _git(tree, "ls-files").split()
        assert "rsqrt" not in "".join(p.read_text(errors="ignore") for p in tree.rglob("*.py")
                                      if ".git" not in p.parts and "leaderboard.py" != p.name
                                      and "problem.json" not in p.name)


def test_set_tolerance_freezes_it_in_both_trees(repo):
    run, task, check = _run_with_check(repo)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    taskcard.seal_reference(run, task, op_dir)
    taskcard.set_tolerance(op_dir, {"y": 3e-6}, "test")
    for tree in (op_dir, run.operators_dir / "_rmsnormref"):
        tol = json.loads((tree / taskcard.PROBLEM).read_text())["tolerance"]
        assert tol["outputs"] == {"y": 3e-6} and tol["calibrated"] == "test"


def test_loop_toml_per_mode(repo):
    run, task, check = _run_with_check(repo)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    deny = launch.deny_paths(run, task)
    cfg = tomllib.loads(taskcard.write_loop_toml(run, task, op_dir, deny=deny).read_text())["loop"]
    assert cfg["oracle_mode"] == "source" and cfg["oracle"] == str(run.task_dir / "reference")
    assert cfg["recon_op"] == "_rmsnormref" and cfg["ref_in_job"] is True
    assert str(run.task_dir / "reference") in cfg["ref_source_paths"]
    assert cfg["gate_window"] == "device" and cfg["kb_arch"] == "sm100"
    assert "loop.toml" in (op_dir / ".gitignore").read_text()

    run2, task2, check2 = _run_with_check(repo / "b", toml=TASK_TOML)
    op2 = taskcard.write_card(run2, task2, check2, load_machine(repo, environ={}))
    cfg2 = tomllib.loads(taskcard.write_loop_toml(run2, task2, op2, deny=[]).read_text())["loop"]
    assert cfg2["ref_from_goal"] == "auto" and cfg2["recon_op"] == ""
    assert cfg2["ref_from_goal_hours"] == 8.0 and "oracle" not in cfg2


def test_other_runs_are_denied(repo):
    run_a, task_a, _ = _run_with_check(repo)
    run_b = new_run(task_a, machine=load_machine(repo, environ={}))
    assert str(run_b.dir) in launch.deny_paths(run_a, task_a)


def test_source_brief_keeps_baseline_and_disables_deep_reconnaissance(repo):
    run, task, check = _run_with_check(repo, toml=REF_TOML + "reconnaissance = false\n")
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    cfg = tomllib.loads(taskcard.write_loop_toml(run, task, op_dir, deny=[]).read_text())["loop"]
    assert cfg["recon_op"] == cfg["ref_price_op"] == ""
    assert cfg["ref_submit_op"] == cfg["arch_ref_submit_op"] == "_rmsnormref"
    assert cfg["ref_in_job"] is True
    assert Path(cfg["source_brief_file"]).read_text() == task.description


# -- forge run, end to end with stand-ins ---------------------------------------------------


class _Teacher:
    """Stands in for the agent: writes the judge's files on its first turn."""

    def __init__(self):
        self.prompts = []

    def __call__(self, be, prompt, role="student"):
        self.prompts.append(prompt)
        _write_check(Path(be.workspace))
        return "done"

    @property
    def method(self):
        teacher = self

        def run_agent(be, prompt, *, role="student"):
            return teacher(be, prompt, role)
        return run_agent


def test_forge_run_drives_every_stage(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    teacher = _Teacher()
    calls = {}
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", teacher.method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(lambda argv: calls.setdefault("loop", argv)))
    rc = _start(task_dir)
    assert rc == 0
    run = list_runs()[0]
    rec = run.record()
    assert rec["status"] == "done" and rec["tolerance_frozen"] and rec["judge_written"]
    argv = calls["loop"]
    assert argv[0] == "rmsnorm" and argv[argv.index("--workspace") + 1] == str(run.ws)
    assert argv[argv.index("--equivalence-ratio") + 1] == "1.2"
    assert "scripts/agent-claude.sh" in argv[argv.index("--agent-cmd") + 1]
    assert (run.ws / ".git").exists() and (run.dir / "report.md").is_file()
    assert json.loads((run.operators_dir / "rmsnorm" / taskcard.PROBLEM).read_text())[
        "tolerance"]["outputs"] == {"y": 2e-6}
    assert len(teacher.prompts) == 1 and str(run.task_dir / "reference") in teacher.prompts[0]


def test_a_calibration_mismatch_sends_the_judge_back(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    teacher = _Teacher()
    outcomes = iter([{"match": False, "problems": ["small/plain output y: normalized error 0.9"]},
                     {"match": True, "tolerance": {"y": 1e-6}, "problems": []}])
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", teacher.method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: next(outcomes))
    monkeypatch.setattr(loop_cli, "main", _loop_ends())
    assert _start(task_dir) == 0
    assert len(teacher.prompts) == 2
    assert "normalized error 0.9" in teacher.prompts[1]


def test_a_task_without_a_reference_gets_the_floor_and_the_goal_session(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo, toml=TASK_TOML)
    calls = {}
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda *a: pytest.fail("nothing to calibrate"))
    monkeypatch.setattr(loop_cli, "main", _loop_ends(lambda argv: calls.setdefault("loop", argv)))
    assert _start(task_dir) == 0
    run = list_runs()[0]
    op_dir = run.operators_dir / "rmsnorm"
    assert not (run.operators_dir / "_rmsnormref").exists()
    tol = json.loads((op_dir / taskcard.PROBLEM).read_text())["tolerance"]
    assert tol["outputs"] == {} and "floor" in tol["calibrated"]
    cfg = tomllib.loads((op_dir / "loop.toml").read_text())["loop"]
    assert cfg["ref_from_goal"] == "auto"


def test_resume_skips_what_is_done(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    teacher = _Teacher()
    seen = []
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", teacher.method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(seen.append))
    _start(task_dir)
    run = list_runs()[0]
    assert launch.cmd_run([str(run.dir)]) == 0
    assert len(teacher.prompts) == 1                      # no second judge
    assert "--allow-preexisting-artifacts" in seen[-1]
    assert open_run(run.dir).record()["status"] == "done"


def test_dry_run_prints_the_plan_and_creates_nothing(repo, capsys):
    task_dir = _task(repo)
    assert launch.cmd_new([str(task_dir), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "task rmsnorm: cutedsl on sm100, mode source" in out and "1.2x the reference" in out
    assert not (repo / "runs").exists()


def test_forge_init_copies_the_task_template(tmp_path, monkeypatch):
    real = launch.repo_root()
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    shutil.copytree(real / "tasks" / "_template", tmp_path / "tasks" / "_template")
    for kb in ("hw_kb/cutedsl/sm100",):
        (tmp_path / kb).mkdir(parents=True)
    assert launch.cmd_init(["gemm"]) == 0
    d = tmp_path / "tasks" / "gemm"
    assert {p.name for p in d.iterdir()} == {"task.toml", "description.md", "shapes.json"}
    task = load_task(d, kb_root=tmp_path)
    assert task.target.ratio == 1.2 and task.mode == "generate"
    assert task.teacher.effort == "high" and task.teacher.timeout_s == 14400
    assert task.student.timeout_s == 7200 and task.student.provider == "native"
    assert launch.cmd_init(["gemm"]) == 2


def test_forge_new_only_creates_the_run(repo, capsys):
    task_dir = _task(repo)
    assert launch.cmd_new([str(task_dir)]) == 0
    (run,) = list_runs()
    assert run.dir.parent == repo / "runs" and run.record()["status"] == "created"
    assert not run.operators_dir.exists() and not run.log.exists()


def test_a_stop_short_of_the_target_exits_2(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop="meta_budget"))
    assert _start(task_dir) == 2
    run = list_runs()[0]
    assert run.record()["status"] == "stopped" and run.record()["stop"] == "meta_budget"
    assert "stop: meta_budget" in (run.dir / "report.md").read_text()


def _confirmed(ratio, monitor="ok"):
    return {"sha": "ab" * 20, "window": "device", "median_ms": 0.02, "median_ratio": ratio,
            "correctness_monitor": {"status": monitor}}


@pytest.mark.parametrize("confirm, rc", [
    (_confirmed(8.0), 0),                    # the budget ran out after the target was met
    (_confirmed(1.1), 2),                    # short of the global 1.2x
    (_confirmed(8.0, monitor="alarm"), 2),   # met, but its correctness is in doubt
])
def test_a_confirmed_best_at_the_target_completes_the_run(repo, monkeypatch, confirm, rc):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop="budget", confirm=confirm))
    assert _start(task_dir) == rc
    rec = list_runs()[0].record()
    assert rec["status"] == ("done" if rc == 0 else "stopped") and rec["stop"] == "budget"
    assert rec["target_met"] is (rc == 0)


# -- NCU: probing, degrading, covering every shape -------------------------------------------

NCU_ROW = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
           '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
           '"Metric Value"\n"0","4242","python3","n","mul_kernel","1","7","(128, 1, 1)",'
           '"(8192, 1, 1)","0","9.0","Command line profiler metrics","gpu__time_active.sum",'
           '"nsecond","5,312"')


def test_the_ncu_probe_reads_a_metric_row_or_a_refusal():
    assert launch.ncu_probe_ok("=== gpu_entry ===\n" + NCU_ROW + "\n[gpu_entry.sh] task=cmd finished")
    assert not launch.ncu_probe_ok("==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                                   "permission to access NVIDIA GPU Performance Counters")
    assert not launch.ncu_probe_ok("bash: ncu: command not found")


def test_without_ncu_the_loop_runs_on_timings(repo, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(launch, "probe_ncu", lambda task, op_dir: False)
    monkeypatch.setattr(loop_cli, "main", _loop_ends())
    assert _start(task_dir) == 0
    run = list_runs()[0]
    cfg = tomllib.loads((run.operators_dir / "rmsnorm" / "loop.toml").read_text())["loop"]
    assert cfg["ncu_available"] is False and cfg["timeline_probe"] == "off"
    assert cfg["stall_pricing"] is False and cfg["job_env_preflight"] == "warn"
    assert run.record()["ncu_available"] is False
    assert not loop_cli.ncu_available({"loop": cfg})


def test_an_arch_reference_without_ncu_is_accepted_only_when_ncu_is_off():
    from forgekernel.loop import cli as loop_cli

    class _Ref:
        has_ncu = False
        ref_name = "ref"

    cfg = {"loop": {"arch_ref_mode": "sealref", "arch_ref_submit_op": "_xref"}}
    assert "no NCU signature" in loop_cli.check_arch_ref_ready(cfg, _Ref(), will_distill=True)
    cfg["loop"]["ncu_available"] = False
    assert loop_cli.check_arch_ref_ready(cfg, _Ref(), will_distill=True) is None


def test_meta_tools_and_ncu_cover_every_shape_and_group(repo):
    run, task, check = _run_with_check(repo)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    problem = json.loads((op_dir / taskcard.PROBLEM).read_text())
    assert problem["meta_workloads"] == ["small", "large"]
    assert taskcard.ncu_launch_count(problem) == 48           # 2 shapes x 3 groups x 8
    assert "--launch-count 48" in (op_dir / "gpu_entry.sh").read_text()


# -- leaderboard tasks ----------------------------------------------------------------------


def _board_task(repo, fib_root, monkeypatch, op="rms") -> Path:
    monkeypatch.setattr(launch, "repo_root", lambda: repo)
    rc = launch.cmd_init([op, "--leaderboard", "flashinfer-bench", "--dataset", str(fib_root),
                         "--definition", "rmsnorm_h8", "--workload", "0"])
    assert rc == 0
    return repo / "tasks" / op


def test_forge_new_leaderboard_freezes_the_board_problem_into_a_task(repo, fib_root, monkeypatch):
    d = _board_task(repo, fib_root, monkeypatch)
    assert (d / "leaderboard" / "card" / taskcard.PROBLEM).is_file()
    assert not (d / "description.md").exists() and not (d / "shapes.json").exists()
    task = load_task(d)
    assert task.mode == "leaderboard" and task.leaderboard.suite == "flashinfer-bench"
    assert [s.name for s in task.shapes] == ["aaaa-1111"]
    assert task.shapes[0].axes == {"batch_size": 7} and task.shapes[0].groups_to_add == 0
    assert task.reference is None and task.target.ratio == 1.2


def test_a_leaderboard_task_runs_without_a_judge_or_a_calibration(repo, fib_root, monkeypatch):
    from forgekernel.loop import cli as loop_cli
    d = _board_task(repo, fib_root, monkeypatch)
    calls = {}
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent",
                        lambda *a, **k: pytest.fail("no judge to write"))
    monkeypatch.setattr(launch, "calibrate", lambda *a: pytest.fail("nothing to calibrate"))
    monkeypatch.setattr(loop_cli, "main", _loop_ends(lambda argv: calls.setdefault("loop", argv)))
    assert _start(d) == 0
    run = list_runs()[0]
    op_dir = run.operators_dir / "rms"
    assert (op_dir / ".git").exists() and not (run.dir / "check").exists()
    cfg = tomllib.loads((op_dir / "loop.toml").read_text())["loop"]
    assert cfg["kb_arch"] == "sm100" and cfg["gate_window"] == "device"
    assert str(run.task_dir / "leaderboard") in cfg["ref_source_paths"]
    sub = tomllib.loads((op_dir / "submit.toml").read_text())
    assert sub["repo"]["delivery"] == "local"
    assert calls["loop"][0] == "rms"


def test_export_writes_the_boards_solution_from_the_best_kernel(repo, fib_root, monkeypatch,
                                                                capsys):
    from forgekernel.loop import cli as loop_cli
    d = _board_task(repo, fib_root, monkeypatch)
    monkeypatch.setattr(loop_cli, "main", _loop_ends())
    _start(d)
    run = list_runs()[0]
    op_dir = run.operators_dir / "rms"
    kernel = "import torch\n\ndef run(hidden_states, weight, output):\n    output.copy_(hidden_states)\n"
    (run.ws / "kernel.py").write_text(kernel)
    subprocess.run(["git", "-C", str(run.ws), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qam", "best"], check=True)
    sha = _git(run.ws, "rev-parse", "HEAD").strip()
    (op_dir / "best_confirmation.json").write_text(json.dumps(
        {"sha": sha, "window": "device", "median_ms": 0.01, "median_ratio": 1.3}))
    capsys.readouterr()
    assert launch.cmd_export([str(run.dir)]) == 0
    sol = json.loads(capsys.readouterr().out)
    assert sol["definition"] == "rmsnorm_h8" and sol["name"] == "rms_forgekernel"
    assert sol["sources"][0]["content"] == kernel


def test_a_task_that_is_not_a_board_problem_exports_the_kernel_source(repo, monkeypatch,
                                                                      capsys):
    from forgekernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends())
    _start(task_dir)
    run = list_runs()[0]
    assert launch.cmd_export([str(run.dir)]) == 2          # nothing confirmed yet
    kernel = "import torch\n\ndef run(x, w):\n    return x\n"
    (run.ws / "kernel.py").write_text(kernel)
    subprocess.run(["git", "-C", str(run.ws), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qam", "best"], check=True)
    sha = _git(run.ws, "rev-parse", "HEAD").strip()
    (run.operators_dir / "rmsnorm" / "best_confirmation.json").write_text(json.dumps(
        {"sha": sha, "window": "device", "median_ms": 0.01, "median_ratio": 1.3}))
    capsys.readouterr()
    assert launch.cmd_export([str(run.dir)]) == 0
    assert capsys.readouterr().out == kernel
    report = launch.write_report(run, load_task(run.task_dir, name=run.task_name), 0)
    assert (run.dir / launch.BEST_DIR / "kernel.py").read_text() == kernel
    assert "- kernel: best/kernel.py" in report.read_text()


DENY_TOML = (TASK_TOML + '[reference]\nrun = "reference/run.py:run"\n'
             'deny_urls = ["github.com/org/RefLib"]\ndeny_modules = ["reflib"]\n')


def _forbid(entry: Path) -> re.Pattern:
    m = re.search(r"^BLACKBOX_FORBID='(.*)'$", entry.read_text(), re.M)
    return re.compile(m.group(1))


def test_deny_urls_and_modules_reach_the_policy_the_guards_and_the_fence(repo):
    run, task, check = _run_with_check(repo, toml=DENY_TOML)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    cfg = tomllib.loads(taskcard.write_loop_toml(run, task, op_dir, deny=[]).read_text())["loop"]
    assert cfg["oracle_mode"] == "blackbox" and cfg["ref_source_urls"] == ["github.com/org/RefLib"]

    impl = taskcard.seal_reference(run, task, op_dir)
    assert impl["deny_modules"] == ["reflib"]
    site = "cat /opt/venv/lib/python3.12/site-packages/reflib/bwd.py"
    for tree in (op_dir, run.operators_dir / "_rmsnormref"):
        forbid = _forbid(tree / "gpu_entry.sh")
        assert forbid.search(site) and forbid.search('python3 -c "import reflib.ops"')
        assert not forbid.search("python3 notes/reflib_bwd_probe.py")   # a word, not a prefix
        # the shell reads the same pattern
        assert subprocess.run(["grep", "-Eq", "--", forbid.pattern], input=site.encode()).returncode == 0

    spec = importlib.util.spec_from_file_location(
        "_fk_test_lb", taskcard.template_dir() / "eval" / "leaderboard.py")
    lbm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbm)
    problem = dict(json.loads((op_dir / taskcard.PROBLEM).read_text()),
                   _dir=str(op_dir / taskcard.PROBLEM.rsplit("/", 1)[0]))
    kfile = op_dir / problem.get("kernel_file", "kernel.py")
    kfile.write_text("import torch\n\ndef run(x, w):\n    return x\n")
    assert lbm.kernel_fence(problem) is None
    kfile.write_text("from reflib.ops import rmsnorm as run\n")
    assert "reflib" in lbm.kernel_fence(problem)
    kfile.write_text("def reflib_like(x):\n    return x\n\nrun = reflib_like\n")
    assert lbm.kernel_fence(problem) is None
