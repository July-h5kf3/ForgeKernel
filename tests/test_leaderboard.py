"""Tests for leaderboard cards (src/forgekernel/leaderboard.py + ``forge new
--leaderboard``): dataset readers, card generation, the preflight exemption and the
solution export. Pure file I/O — no torch, no GPU, no network; the datasets are tiny
fakes laid out like flashinfer-trace and SOL-ExecBench's download tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from forgekernel import leaderboard as lb
from forgekernel.scaffold import main as scaffold_main
from forgekernel.window_contract import (
    bench_workloads_findings,
    parse_window_contract,
    window_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

RMS_DEF = {
    "name": "rmsnorm_h8", "op_type": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 8}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
               "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}},
    "reference": "import torch\n\ndef run(hidden_states, weight):\n    return hidden_states\n",
}
PAGED_DEF = {
    "name": "paged_h8", "op_type": "gqa_paged",
    "axes": {"n": {"type": "var"}, "d": {"type": "const", "value": 8}},
    "inputs": {"q": {"shape": ["n", "d"], "dtype": "bfloat16"},
               "kv_indptr": {"shape": ["n"], "dtype": "int32"},
               "sm_scale": {"shape": None, "dtype": "float32"}},
    "outputs": {"out": {"shape": ["n", "d"], "dtype": "bfloat16"}},
    "reference": "def run(q, kv_indptr, sm_scale):\n    return q\n",
}


def _trace_line(defn: dict, uuid: str, axes: dict, inputs: dict | None = None) -> str:
    wl = {"uuid": uuid, "axes": axes,
          "inputs": inputs or {n: {"type": "random"} for n in defn["inputs"]}}
    return json.dumps({"definition": defn["name"], "solution": None, "workload": wl,
                       "evaluation": None})


@pytest.fixture
def fib_root(tmp_path: Path) -> Path:
    """A flashinfer-trace-shaped dataset: two definitions, a baseline, one blob."""
    root = tmp_path / "flashinfer-trace"
    for group, defn in (("rmsnorm", RMS_DEF), ("gqa_paged", PAGED_DEF)):
        (root / "definitions" / group).mkdir(parents=True)
        (root / "definitions" / group / f"{defn['name']}.json").write_text(json.dumps(defn))
        (root / "workloads" / group).mkdir(parents=True)
    (root / "workloads" / "rmsnorm" / "rmsnorm_h8.jsonl").write_text("\n".join([
        _trace_line(RMS_DEF, "aaaa-1111", {"batch_size": 7}),
        _trace_line(RMS_DEF, "bbbb-2222", {"batch_size": 64})]) + "\n")
    blob_rel = "./blob/workloads/gqa_paged/paged_h8/paged_h8_x.safetensors"
    (root / blob_rel).parent.mkdir(parents=True)
    (root / blob_rel).write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    (root / "workloads" / "gqa_paged" / "paged_h8.jsonl").write_text(_trace_line(
        PAGED_DEF, "cccc-3333", {"n": 2},
        {"q": {"type": "random"},
         "kv_indptr": {"type": "safetensors", "path": blob_rel, "tensor_key": "kv_indptr"},
         "sm_scale": {"type": "scalar", "value": 0.125}}) + "\n")
    sdir = root / "solutions" / "baseline" / "rmsnorm" / "rmsnorm_h8"
    sdir.mkdir(parents=True)
    (sdir / "flashinfer_wrapper_1.json").write_text(json.dumps({
        "name": "flashinfer_wrapper_1", "definition": "rmsnorm_h8", "author": "baseline",
        "spec": {"language": "python", "target_hardware": ["NVIDIA B200"],
                 "entry_point": "main.py::run", "dependencies": ["flashinfer"],
                 "destination_passing_style": False},
        "sources": [{"path": "main.py", "content": "def run(hidden_states, weight):\n"
                                                   "    return hidden_states\n"}]}))
    return root


@pytest.fixture
def sol_root(tmp_path: Path) -> Path:
    """SOL-ExecBench's download tree: <subset>/<problem>/{definition.json, workload.jsonl}."""
    root = tmp_path / "sol" / "benchmark"
    pdir = root / "L1" / "001_rms"
    pdir.mkdir(parents=True)
    defn = dict(RMS_DEF, name="001_rms")
    (pdir / "definition.json").write_text(json.dumps(defn))
    (pdir / "workload.jsonl").write_text("\n".join(json.dumps(w) for w in [
        {"uuid": "u-0", "axes": {"batch_size": 4},
         "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}},
         "tolerance": {"max_atol": 1e-5, "max_rtol": 0.05}},
        {"uuid": "u-1", "axes": {"batch_size": 9},
         "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}},
    ]) + "\n")
    return root


@pytest.fixture(autouse=True)
def _no_network(monkeypatch) -> None:
    """No test reaches the SOL board: a test that wants a board passes one in or
    replaces _http_json itself."""
    def refuse(url, timeout=30.0):
        raise OSError(f"network disabled in tests ({url})")
    monkeypatch.setattr(lb, "_http_json", refuse)


@pytest.fixture
def base(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "repo" / "meta")
    shutil.copytree(REPO_ROOT / "src" / "forgekernel" / "card_template",
                    tmp_path / "repo" / "src" / "forgekernel" / "card_template")
    return tmp_path / "repo"


# --- dataset readers ------------------------------------------------------------------

def test_fib_problem_picks_workload_by_index_and_uuid_prefix(fib_root: Path) -> None:
    assert lb.fib_problem(fib_root, "rmsnorm_h8", "1").workload["uuid"] == "bbbb-2222"
    p = lb.fib_problem(fib_root, "rmsnorm_h8", "aaaa")
    assert p.workload["axes"] == {"batch_size": 7}
    assert p.suite == lb.SUITE_FIB and p.source["definition_file"].endswith("rmsnorm_h8.json")


def test_fib_problem_demands_a_workload_choice(fib_root: Path) -> None:
    with pytest.raises(ValueError, match="pick one with --workload"):
        lb.fib_problem(fib_root, "rmsnorm_h8", None)
    with pytest.raises(ValueError, match="matches 0 of 2"):
        lb.fib_problem(fib_root, "rmsnorm_h8", "zzzz")


def test_fib_baseline_solution_is_the_in_job_reference(fib_root: Path) -> None:
    p = lb.fib_problem(fib_root, "rmsnorm_h8", "0")
    assert p.reference_impl == {
        "kind": "baseline_solution", "name": "flashinfer_wrapper_1", "author": "baseline",
        "path": "baseline/main.py", "entry": "run", "dps": False,
        "dependencies": ["flashinfer"]}
    assert "def run(hidden_states, weight)" in p.reference_files["baseline/main.py"]


def test_fib_without_baseline_falls_back_to_the_definition_reference(fib_root: Path) -> None:
    p = lb.fib_problem(fib_root, "paged_h8", "0")
    assert p.reference_impl == {"kind": "definition"}
    assert any("no baseline solution" in n for n in p.notes)


def test_fib_blob_is_carried_and_rewritten_relative_to_problem_json(fib_root: Path) -> None:
    p = lb.fib_problem(fib_root, "paged_h8", "0")
    spec = p.workload["inputs"]["kv_indptr"]
    assert spec["path"] == "blob/workloads/gqa_paged/paged_h8/paged_h8_x.safetensors"
    assert p.blobs[spec["path"]].is_file()


def test_fib_lfs_pointer_blob_is_refused(fib_root: Path) -> None:
    blob = next((fib_root / "blob").rglob("*.safetensors"))
    blob.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 520\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        lb.fib_problem(fib_root, "paged_h8", "0")


@pytest.mark.parametrize("defn_patch, why", [
    ({"op_type": "sampling", "name": "top_k_sampling_v8"}, "TVD"),
    ({"name": "dsa_topk_indexer_fp8_h8"}, "sorted recomputed scores"),
])
def test_fib_definitions_with_special_evaluators_are_refused(
        fib_root: Path, defn_patch: dict, why: str) -> None:
    defn = dict(RMS_DEF, **defn_patch)
    (fib_root / "definitions" / "rmsnorm" / f"{defn['name']}.json").write_text(json.dumps(defn))
    with pytest.raises(ValueError, match=why):
        lb.fib_problem(fib_root, defn["name"], "0")


def test_sol_problem_by_subset_path_and_by_bare_name(sol_root: Path) -> None:
    a = lb.sol_problem(sol_root, "L1/001_rms", "0")
    b = lb.sol_problem(sol_root, "001_rms", "u-1")
    assert a.workload["tolerance"] == {"max_atol": 1e-5, "max_rtol": 0.05}
    assert b.workload["axes"] == {"batch_size": 9}
    assert a.reference_impl["kind"] == "definition"


def test_sol_reference_py_fills_a_definition_without_one(sol_root: Path) -> None:
    pdir = sol_root / "L1" / "001_rms"
    defn = json.loads((pdir / "definition.json").read_text())
    defn["reference"] = ""
    (pdir / "definition.json").write_text(json.dumps(defn))
    (pdir / "reference.py").write_text("def run(hidden_states, weight):\n    return 1\n")
    assert "return 1" in lb.sol_problem(sol_root, "L1/001_rms", "0").definition["reference"]


# --- card generation ------------------------------------------------------------------

def test_forge_new_leaderboard_writes_a_complete_card(base: Path, fib_root: Path) -> None:
    rc = scaffold_main(["rms", "--base-dir", str(base), "--no-git",
                        "--leaderboard", "flashinfer-bench", "--dataset", str(fib_root),
                        "--definition", "rmsnorm_h8", "--workload", "0"])
    assert rc == 0
    card = base / "operators" / "rms"
    problem = json.loads((card / lb.PROBLEM_FILE).read_text())
    assert problem["suite"] == "flashinfer-bench" and problem["dps"] is True
    assert problem["definition"]["name"] == "rmsnorm_h8"
    for rel in ("eval/leaderboard.py", "eval/timing.py", "eval/bench.py", "eval/ncu_run.py",
                "gpu_entry.sh", "kernel.py", "loop.toml.example",
                "eval/leaderboard/baseline/main.py"):
        assert (card / rel).is_file(), rel
    assert (card / "eval" / "leaderboard.py").read_text() == (
        REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py").read_text()
    # the generated kernel stub carries the definition's DPS signature
    assert "def run(hidden_states, weight, output):" in (card / "kernel.py").read_text()
    assert lb.signature_findings((card / "kernel.py").read_text(), problem) == []
    assert "Leaderboard mode" in (card / "README.md").read_text()


def test_generated_bench_passes_the_multi_workload_preflight(base: Path, fib_root: Path) -> None:
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "flashinfer-bench", "--dataset", str(fib_root), "--definition",
                   "rmsnorm_h8", "--workload", "0"])
    bench = (base / "operators" / "rms" / "eval" / "bench.py").read_text()
    assert bench_workloads_findings(bench) == []
    # the exemption is the leaderboard entry point, not any bench
    assert bench_workloads_findings("def main():\n    pass\n")


def test_generated_window_contract_has_no_findings(base: Path, sol_root: Path) -> None:
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "sol-execbench", "--dataset", str(sol_root), "--definition",
                   "L1/001_rms", "--workload", "0", "--no-sol-board"])
    text = (base / "operators" / "rms" / "window.toml").read_text()
    contract = parse_window_contract(text, path="window.toml")
    assert contract.declared and window_findings(contract) == []
    assert contract.reference_inputs == contract.student_inputs == ["hidden_states", "weight"]
    assert contract.reference_outputs == contract.student_outputs == ["output"]


def test_sol_card_locks_the_published_clock(base: Path, sol_root: Path) -> None:
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "sol-execbench", "--dataset", str(sol_root), "--definition",
                   "L1/001_rms", "--workload", "0", "--no-sol-board"])
    entry = (base / "operators" / "rms" / "gpu_entry.sh").read_text()
    assert 'LB_SM_CLOCK_MHZ="${LB_SM_CLOCK_MHZ-1500}"' in entry
    assert 'LB_MEM_CLOCK_MHZ="${LB_MEM_CLOCK_MHZ-3996}"' in entry
    assert 'LB_REQUIRE_CLOCK_LOCK="${LB_REQUIRE_CLOCK_LOCK:-1}"' in entry
    assert "trap lb_clocks_reset EXIT" in entry
    assert "python3 eval/bench.py --mode ref" in entry


def test_bad_choice_does_not_leave_a_half_card(base: Path, fib_root: Path) -> None:
    with pytest.raises(SystemExit, match="matches 0 of 2"):
        scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                       "flashinfer-bench", "--dataset", str(fib_root), "--definition",
                       "rmsnorm_h8", "--workload", "nope"])
    assert not (base / "operators" / "rms").exists()


def test_list_workloads_prints_and_scaffolds_nothing(base: Path, fib_root: Path,
                                                     capsys) -> None:
    rc = scaffold_main(["rms", "--base-dir", str(base), "--leaderboard", "flashinfer-bench",
                        "--dataset", str(fib_root), "--definition", "rmsnorm_h8",
                        "--list-workloads"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "aaaa-1111" in out and "bbbb-2222" in out
    assert not (base / "operators" / "rms").exists()


# --- export ---------------------------------------------------------------------------

def _card(base: Path, root: Path, suite: str, definition: str) -> Path:
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard", suite,
                   "--dataset", str(root), "--definition", definition, "--workload", "0"]
                  + (["--no-sol-board"] if suite == "sol-execbench" else []))
    card = base / "operators" / "rms"
    (card / "kernel.py").write_text(
        "import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
        "    output.copy_(hidden_states)\n")
    return card


def test_export_fib_solution(base: Path, fib_root: Path) -> None:
    card = _card(base, fib_root, "flashinfer-bench", "rmsnorm_h8")
    sol = lb.export_solution(card, name="rms_fk", author="me")
    assert sol["definition"] == "rmsnorm_h8" and sol["author"] == "me"
    assert sol["spec"] == {"language": "python", "target_hardware": ["NVIDIA B200"],
                           "entry_point": "kernel.py::run", "dependencies": [],
                           "destination_passing_style": True}
    assert sol["sources"][0]["path"] == "kernel.py"


def test_export_sol_solution(base: Path, sol_root: Path) -> None:
    card = _card(base, sol_root, "sol-execbench", "L1/001_rms")
    sol = lb.export_solution(card, name="rms_fk", author="me")
    assert sol["spec"]["languages"] == ["cute_dsl"]
    assert sol["spec"]["dependencies"] == ["cutlass"]
    assert sol["spec"]["target_hardware"] == ["B200"]
    assert sol["spec"]["destination_passing_style"] is True


def test_export_refuses_a_drifted_signature(base: Path, fib_root: Path) -> None:
    card = _card(base, fib_root, "flashinfer-bench", "rmsnorm_h8")
    (card / "kernel.py").write_text("def run(x, w, out):\n    pass\n")
    with pytest.raises(ValueError, match=r"wants run\(hidden_states, weight, output\)"):
        lb.export_solution(card, name="n", author="a")


@pytest.mark.parametrize("src, loop_lang, want", [
    ("import cutlass.cute as cute\n", None, "cutedsl"),
    ("from triton import language as tl\n", None, "triton"),
    ("import torch\n", None, "pytorch"),
    ("import torch\n", "triton", "triton"),
])
def test_infer_language(src: str, loop_lang, want: str) -> None:
    assert lb.infer_language(src, loop_lang) == want


# --- SOL board: T_b / T_SOL, the SOL score and the loop's target -------------------------

SOL_BOARD = {"id": 7, "name": "001_rms", "tags": ["L1", "norm"], "workloads": [
    {"axes": {"batch_size": 4, "hidden_size": 8}, "baseline_latency_ms": 0.04, "sol_ms": 0.004},
    {"axes": {"batch_size": 9, "hidden_size": 8}, "baseline_latency_ms": 0.05, "sol_ms": 0.005},
]}


def _board(**changes) -> dict:
    rec = json.loads(json.dumps(SOL_BOARD))
    rec.update(changes)
    return rec


def _fake_api(monkeypatch, record=SOL_BOARD) -> list[str]:
    calls: list[str] = []

    def http(url, timeout=30.0):
        calls.append(url)
        if url.endswith("/kernels"):
            return {"data": {"kernels": [
                {"id": 7, "name": "001_rms", "tags": ["L1", "norm"]},
                {"id": 8, "name": "001_rms_fused", "tags": ["L2"]}]}}
        if url.endswith("/kernels/7"):
            return {"data": record}
        raise OSError(f"unexpected {url}")
    monkeypatch.setattr(lb, "_http_json", http)
    return calls


def test_sol_board_is_attached_by_position_with_an_axes_check(sol_root: Path) -> None:
    p = lb.sol_problem(sol_root, "L1/001_rms", "u-1", board=_board(), board_source="b.json")
    assert p.board == {"kernel_id": 7, "workload_index": 1, "t_b_ms": 0.05, "t_sol_ms": 0.005,
                       "source": "b.json", "clock_mhz": {"sm_mhz": 1500, "mem_mhz": 3996}}
    moved = _board()
    moved["workloads"][1]["axes"]["batch_size"] = 10
    for bad, why in ((moved, "axes differ"),
                     (_board(workloads=SOL_BOARD["workloads"] * 2), "lists 4 workloads"),
                     (_board(name="002_other"), "not '001_rms'")):
        q = lb.sol_problem(sol_root, "L1/001_rms", "u-1", board=bad)
        assert q.board is None and any(why in n for n in q.notes), (why, q.notes)


def test_sol_board_fetch_matches_the_name_within_the_subset(monkeypatch) -> None:
    calls = _fake_api(monkeypatch)
    assert lb.fetch_sol_board("001_rms", "L1")["workloads"][0]["baseline_latency_ms"] == 0.04
    assert calls[-1].endswith("/kernels/7")
    with pytest.raises(LookupError, match="matches 0"):
        lb.fetch_sol_board("001_rms", "Quant")


def test_sol_score_and_its_inverse_are_the_boards_formula() -> None:
    assert lb.sol_score(0.04, 0.04, 0.004) == 0.5              # at T_b
    assert lb.sol_score(0.004, 0.04, 0.004) == 1.0             # at T_SOL
    for s in (0.6, 0.8, 0.9):
        t = lb.time_for_score(s, 0.04, 0.004)
        assert lb.sol_score(t, 0.04, 0.004) == pytest.approx(s)
    assert lb.time_for_score(0.8, 0.004, 0.04) is None         # T_b not above T_SOL


def test_forge_new_sol_fetches_the_board_and_the_loop_targets_T_b(
        base: Path, sol_root: Path, monkeypatch) -> None:
    from forgekernel.loop.cli import (_declared_arch_baseline, _resolve_ref_in_job,
                                      _resolve_ref_spec, _resolve_target_ms)
    _fake_api(monkeypatch)
    assert scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                          "sol-execbench", "--dataset", str(sol_root), "--definition",
                          "L1/001_rms", "--workload", "1"]) == 0
    card = base / "operators" / "rms"
    problem = json.loads((card / lb.PROBLEM_FILE).read_text())
    assert problem["time_reference"] is False
    assert problem["sol_board"]["t_b_ms"] == 0.05 and problem["sol_board"]["kernel_id"] == 7
    example = (card / "loop.toml.example").read_text()
    cfg = tomllib.loads(example)
    loop = cfg["loop"]
    # the reference IS the board's T_b; nothing times the eager definition reference
    assert loop["arch_ref_ms"] == loop["arch_ref_device_ms"] == 0.05
    assert loop["arch_ref_device_basis"] == "cupti_kernel_span"
    assert "ref_cmd" not in loop and "ref_in_job" not in loop
    assert not _resolve_ref_spec(cfg).get("ref_cmd")
    assert _resolve_ref_in_job(cfg, _resolve_ref_spec(cfg)) is False
    arch = _declared_arch_baseline("rms", cfg)
    assert _resolve_target_ms(None, None, 1.0, arch=arch, gate_window="device") == 0.05
    assert _resolve_target_ms(None, None, 1.25, arch=arch, gate_window="device") == \
        pytest.approx(0.04)
    t80 = lb.time_for_score(0.8, 0.05, 0.005)
    assert f"S 0.8: {t80:.6g} ms" in example
    assert "T_b = 0.05 ms" in (card / "README.md").read_text()
    contract = parse_window_contract((card / "window.toml").read_text(), path="window.toml")
    assert window_findings(contract) == []


def test_forge_new_sol_without_a_board_asks_for_an_explicit_target(
        base: Path, sol_root: Path, capsys) -> None:
    # the autouse fixture makes the board unreachable
    assert scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                          "sol-execbench", "--dataset", str(sol_root), "--definition",
                          "L1/001_rms", "--workload", "0"]) == 0
    assert "NOTE SOL board not read" in capsys.readouterr().out
    card = base / "operators" / "rms"
    problem = json.loads((card / lb.PROBLEM_FILE).read_text())
    assert "sol_board" not in problem and problem["time_reference"] is False
    example = (card / "loop.toml.example").read_text()
    loop = tomllib.loads(example)["loop"]
    assert not any(k.startswith(("ref_cmd", "arch_ref", "ref_in_job")) for k in loop)
    assert "--target-ms" in example


def test_forge_new_sol_board_from_a_saved_file(base: Path, sol_root: Path,
                                               tmp_path: Path) -> None:
    saved = tmp_path / "kernels_7.json"
    saved.write_text(json.dumps({"data": SOL_BOARD}))
    assert scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                          "sol-execbench", "--dataset", str(sol_root), "--definition",
                          "L1/001_rms", "--workload", "0", "--sol-board", str(saved)]) == 0
    board = json.loads((base / "operators" / "rms" / lb.PROBLEM_FILE).read_text())["sol_board"]
    assert board["t_b_ms"] == 0.04 and board["source"] == str(saved)
    with pytest.raises(SystemExit, match="sol-execbench options"):
        scaffold_main(["rms2", "--base-dir", str(base), "--leaderboard",
                       "flashinfer-bench", "--dataset", str(sol_root), "--definition", "x",
                       "--no-sol-board"])


def test_fib_loop_example_declares_the_baseline_and_no_inert_flag(base: Path,
                                                                  fib_root: Path) -> None:
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "flashinfer-bench", "--dataset", str(fib_root), "--definition",
                   "rmsnorm_h8", "--workload", "0"])
    card = base / "operators" / "rms"
    loop = tomllib.loads((card / "loop.toml.example").read_text())["loop"]
    assert loop["ref_cmd"] == "python3 eval/bench.py --mode ref"
    assert loop["ref_name"] == "flashinfer_wrapper_1" and "ref_in_job" not in loop
    assert json.loads((card / lb.PROBLEM_FILE).read_text())["time_reference"] is True


# --- gpu_entry.sh clock lock (a fake nvidia-smi; never the real one) ---------------------

FAKE_SMI = """#!/bin/bash
echo "$*" >> "$FAKE_SMI_LOG"
case "$*" in
    *-lgc*) exit "${FAKE_LGC_RC:-0}" ;;
    *-lmc*) exit "${FAKE_LMC_RC:-0}" ;;
    *clocks.current.graphics*) printf '%b\\n' "${FAKE_READ:-1500, 3996}" ;;
    *) echo "FAKE" ;;
esac
"""


def _entry(base: Path, root: Path, suite: str, definition: str, tmp_path: Path, task: str,
           **env) -> tuple[subprocess.CompletedProcess, list[str]]:
    card = base / "operators" / "rms"
    if not card.exists():
        _card(base, root, suite, definition)
    smi = tmp_path / "fake-nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    log = tmp_path / "smi.log"
    log.write_text("")
    full = {k: v for k, v in os.environ.items()
            if k not in ("LB_SM_CLOCK_MHZ", "LB_MEM_CLOCK_MHZ", "LB_REQUIRE_CLOCK_LOCK")}
    full.update({"LB_NVIDIA_SMI": str(smi), "FAKE_SMI_LOG": str(log),
                 "LB_CLOCK_SETTLE_S": "0", "LB_INSTALL_CUPTI": "0", **env})
    proc = subprocess.run(["bash", str(card / "gpu_entry.sh"), task], cwd=card, env=full,
                          capture_output=True, text=True, timeout=60)
    calls = [line for line in log.read_text().splitlines()
             if not line.startswith("--query-gpu=name")]            # the banner's query
    return proc, calls


def test_sol_entry_locks_verifies_and_resets_on_the_way_out(base, sol_root, tmp_path) -> None:
    proc, calls = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, "help")
    assert "clocks:    locked sm=1500 mem=3996" in proc.stdout
    assert calls[:2] == ["-rgc", "-rmc"]                       # clear an inherited lock
    assert "-lgc 1500,1500" in calls and "-lmc 3996,3996" in calls
    assert calls[-2:] == ["-rgc", "-rmc"]                      # EXIT trap, even on exit 1


@pytest.mark.parametrize("env, state", [
    ({"FAKE_LGC_RC": "1"}, "lock refused"),
    ({"FAKE_LMC_RC": "1"}, "lock refused"),
    ({"FAKE_READ": "1965, 3996"}, "lock not held"),
    ({"FAKE_READ": "1500, 3996\\n1965, 3996"}, "lock not held"),   # one GPU of two
])
def test_sol_entry_refuses_to_time_without_the_boards_lock(base, sol_root, tmp_path,
                                                           env, state) -> None:
    proc, calls = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, "all", **env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stdout and state in proc.stdout
    assert "timeout=" not in proc.stdout                       # the bench never ran
    assert calls[-2:] == ["-rgc", "-rmc"]
    # nothing runs on such a node: correctness, probes (cmd) and NCU are refused too
    for task in ("correctness", "cmd", "ncu", "cardref", "ref"):
        again, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, task,
                          **env)
        assert again.returncode == 3 and "REFUSED" in again.stdout, task
    # a card told to ask for no lock (LB_SM_CLOCK_MHZ="") still never times, but its
    # other jobs run: nothing was asked of the node, so nothing failed there
    for task, code in (("correctness", None), ("all", 3)):
        again, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, task,
                          LB_SM_CLOCK_MHZ="")
        assert ("REFUSED" in again.stdout) == (code == 3), task
    # help is not a GPU job, and the refusal can be waived
    for task, extra in (("help", {}), ("all", {"LB_REQUIRE_CLOCK_LOCK": "0"})):
        again, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, task,
                          **env, **extra)
        assert "REFUSED" not in again.stdout


def test_sol_entry_refusal_is_the_line_forgekernel_reads(base, sol_root, tmp_path) -> None:
    """The node read SM 1155 MHz under an accepted -lgc 1500: the
    refusal must print the exact line loop.cli recognises, at the start of a line, and
    the banner says what the driver reported about the GPU right after."""
    from forgekernel.submit.jobs import CLOCK_LOCK_REFUSED_MARKER, clock_lock_refusal
    proc, calls = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, "all",
                         FAKE_READ="1155, 3996")
    assert proc.returncode == 3
    assert any(line.startswith(CLOCK_LOCK_REFUSED_MARKER) for line in proc.stdout.splitlines())
    assert "read 1155,3996" in clock_lock_refusal(proc.stdout)
    assert "  clockdiag: FAKE" in proc.stdout
    assert any("clocks_event_reasons.active" in c for c in calls)
    assert calls[-2:] == ["-rgc", "-rmc"]
    # a meta tool's job is refused with the same line; help still prints, with the
    # diagnosis in its banner
    tool, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path,
                     "cmd", FAKE_READ="1155, 3996")
    assert tool.returncode == 3 and clock_lock_refusal(tool.stdout) is not None
    ran, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, "help",
                    FAKE_READ="1155, 3996")
    assert clock_lock_refusal(ran.stdout) is None and "  clockdiag: " in ran.stdout
    # a lock that held prints no diagnosis
    held, _ = _entry(base, sol_root, "sol-execbench", "L1/001_rms", tmp_path, "help")
    assert "clockdiag" not in held.stdout


def test_entry_locks_and_resets_only_the_jobs_gpu(base, fib_root, tmp_path) -> None:
    """Two runs on one machine, one GPU each: a job's lock and its EXIT reset never touch
    the other run's GPU."""
    proc, calls = _entry(base, fib_root, "flashinfer-bench", "rmsnorm_h8", tmp_path, "help",
                         LB_SM_CLOCK_MHZ="1800", FAKE_READ="1800, 1234", CUDA_VISIBLE_DEVICES="2")
    assert "clocks:    locked sm=1800 mem=free" in proc.stdout
    assert calls and all(c.startswith("-i 2 ") for c in calls), calls
    assert "-i 2 -lgc 1800,1800" in calls and calls[-2:] == ["-i 2 -rgc", "-i 2 -rmc"]


def test_fib_entry_clears_an_inherited_lock_and_sets_none(base, fib_root, tmp_path) -> None:
    proc, calls = _entry(base, fib_root, "flashinfer-bench", "rmsnorm_h8", tmp_path, "help")
    assert calls == ["-rgc", "-rmc"]
    assert "clocks:    unlocked" in proc.stdout
    pinned, calls = _entry(base, fib_root, "flashinfer-bench", "rmsnorm_h8", tmp_path, "help",
                           LB_SM_CLOCK_MHZ="1800", FAKE_READ="1800, 1234")
    assert "-lgc 1800,1800" in calls and not any(c.startswith("-lmc") for c in calls)
    assert "clocks:    locked sm=1800 mem=free" in pinned.stdout
    assert calls[-2:] == ["-rgc", "-rmc"]


# --- workload groups: one card judging and timing a size class of workloads ---------------

def test_pick_workloads_reads_lists_ranges_and_uuid_prefixes(sol_root: Path) -> None:
    rows = lb.workload_rows(sol_root / "L1" / "001_rms" / "workload.jsonl")
    assert lb._pick_workloads(rows, "1", "w") == [1]
    assert lb._pick_workloads(rows, "0-1", "w") == [0, 1]
    assert lb._pick_workloads(rows, "u-1, 0, 1", "w") == [1, 0]     # order kept, repeats dropped
    for bad, why in (("0,", "empty item"), ("1-0", "not within"), ("0-2", "not within"),
                     ("zz", "matches 0")):
        with pytest.raises(ValueError, match=why):
            lb._pick_workloads(rows, bad, "w")


def test_sol_group_problem_carries_every_workload_and_its_board_row(sol_root: Path) -> None:
    p = lb.sol_problem(sol_root, "L1/001_rms", "1,0", board=_board(), board_source="b.json")
    assert [g["index"] for g in p.group] == [1, 0] and [g["name"] for g in p.group] == ["wl1", "wl0"]
    assert p.workload["uuid"] == "u-1" and p.board["workload_index"] == 1   # first = representative
    assert [g["board"]["t_b_ms"] for g in p.group] == [0.05, 0.04]
    out = lb.problem_json(p)
    assert [g["workload"]["uuid"] for g in out["group"]] == ["u-1", "u-0"]
    assert out["group"][1]["sol_board"]["t_sol_ms"] == 0.004
    assert out["workload"]["uuid"] == "u-1" and out["sol_board"]["workload_index"] == 1
    # one workload the board cannot line up: no group score at all (not a partial one)
    moved = _board()
    moved["workloads"][0]["axes"]["batch_size"] = 5
    q = lb.sol_problem(sol_root, "L1/001_rms", "0-1", board=moved)
    assert q.board is None and all(g["board"] is None for g in q.group)
    assert any("axes differ" in n for n in q.notes)
    assert "sol_board" not in lb.problem_json(q)
    # one workload stays a plain card
    assert lb.sol_problem(sol_root, "L1/001_rms", "1", board=_board()).group == []
    assert "group" not in lb.problem_json(lb.sol_problem(sol_root, "L1/001_rms", "1"))


def test_forge_new_sol_group_card_targets_the_geomean_of_T_b(base: Path, sol_root: Path,
                                                             monkeypatch) -> None:
    import math
    from forgekernel.loop.cli import _declared_arch_baseline
    from forgekernel.window_contract import bench_times_workloads
    _fake_api(monkeypatch)
    assert scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                          "sol-execbench", "--dataset", str(sol_root), "--definition",
                          "L1/001_rms", "--workload", "0,1"]) == 0
    card = base / "operators" / "rms"
    problem = json.loads((card / lb.PROBLEM_FILE).read_text())
    assert [g["name"] for g in problem["group"]] == ["wl0", "wl1"]
    cfg = tomllib.loads((card / "loop.toml.example").read_text())
    loop = cfg["loop"]
    assert loop["arch_ref_ms"] == pytest.approx(math.sqrt(0.04 * 0.05))
    assert loop["arch_ref_workloads"] == ["wl0", "wl1"]
    arch = _declared_arch_baseline("rms", cfg)
    assert arch.n_workloads == 2 and arch.workloads == ["wl0", "wl1"]
    # the loop's static check sees a geomean gate (so a 2-workload archive is not refused)
    assert bench_times_workloads(card) is True
    stub = (card / "kernel.py").read_text()
    assert "this card's 2 workloads: 4, 9" in stub
    readme = (card / "README.md").read_text()
    assert "workload group (2 workloads" in readme and "`wl1`" in readme
    assert "group of 2: wl0, wl1" in (card / "eval" / "bench.py").read_text()
    assert problem["meta_workloads"] == ["wl0", "wl1"]       # a group of two: both
    contract = parse_window_contract((card / "window.toml").read_text(), path="window.toml")
    assert window_findings(contract) == []


def test_single_workload_leaderboard_card_is_not_a_multi_workload_gate(base: Path,
                                                                       sol_root: Path) -> None:
    from forgekernel.window_contract import bench_times_workloads
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "sol-execbench", "--dataset", str(sol_root), "--definition",
                   "L1/001_rms", "--workload", "0", "--no-sol-board"])
    assert bench_times_workloads(base / "operators" / "rms") is False


def test_fib_group_card_copies_every_workloads_blobs(base: Path, fib_root: Path) -> None:
    p = lb.fib_problem(fib_root, "rmsnorm_h8", "0-1")
    assert [g["workload"]["uuid"] for g in p.group] == ["aaaa-1111", "bbbb-2222"]
    assert p.reference_impl["kind"] == "baseline_solution"
    scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                   "flashinfer-bench", "--dataset", str(fib_root), "--definition",
                   "rmsnorm_h8", "--workload", "1,0"])
    problem = json.loads((base / "operators" / "rms" / lb.PROBLEM_FILE).read_text())
    assert [g["index"] for g in problem["group"]] == [1, 0]


def _sized_group(sizes, tsol=None) -> "lb.Problem":
    p = lb.Problem(lb.SUITE_SOL, json.loads(json.dumps(RMS_DEF)), {}, source={})
    p.group = [{"index": i, "name": f"wl{i}", "board": ({"t_sol_ms": tsol[i], "t_b_ms": 1.0}
                                                        if tsol else None),
                "workload": {"uuid": f"u{i}", "axes": {"batch_size": b}}}
               for i, b in enumerate(sizes)]
    p.workload = p.group[0]["workload"]
    return p


def test_meta_workloads_are_the_representative_then_the_groups_smallest_and_largest() -> None:
    """The meta tools read 2-3 workloads of a group card: the representative, then the
    smallest and the largest -- by T_SOL when the board gives one to every workload, ties
    (a weight-bound GEMM's small-M rows share one T_SOL) broken by tensor elements."""
    assert lb.meta_workload_names(_sized_group((17, 3, 64, 9))) == ["wl0", "wl1", "wl2"]
    # one T_SOL for the whole group: element count decides; the representative is the largest
    same = _sized_group((56, 35, 48, 40, 32), tsol=[0.0158] * 5)
    # ... so the third is the group's middle size (40), not a repeat
    assert lb.meta_workload_names(same) == ["wl0", "wl4", "wl3"]
    # T_SOL first, elements only break its ties (the representative is the largest here too)
    assert lb.meta_workload_names(_sized_group((1, 100, 50), tsol=[0.3, 0.1, 0.2])) == ["wl0", "wl1", "wl2"]
    # the representative is the smallest: smallest, largest, then the middle of the rest
    assert lb.meta_workload_names(_sized_group((2, 9, 5, 7, 3, 8))) == ["wl0", "wl1", "wl2"]
    # a group of two reads both, nothing else to fill
    assert lb.meta_workload_names(_sized_group((9, 2))) == ["wl0", "wl1"]
    # a board missing on one workload: elements alone
    p = _sized_group((5, 1, 9), tsol=[0.3, 0.1, 0.2])
    p.group[2]["board"] = None
    assert lb.meta_workload_names(p) == ["wl0", "wl1", "wl2"]
    single = _sized_group((5,))
    single.group = []
    assert lb.meta_workload_names(single) == []
    out = lb.problem_json(_sized_group((17, 3, 64)))
    assert out["meta_workloads"] == ["wl0", "wl1", "wl2"]
    assert "meta_workloads" not in lb.problem_json(single)


def test_tensor_elements_resolves_const_var_and_expr_axes() -> None:
    defn = {"axes": {"n": {"type": "var"}, "d": {"type": "const", "value": 8},
                     "h": {"type": "expr", "expression": "n * 2"}},
            "inputs": {"q": {"shape": ["h", "d"]}, "s": {"shape": None}, "w": {"shape": [4]}},
            "outputs": {"o": {"shape": ["n", "d"]}}}
    assert lb._tensor_elements(defn, {"axes": {"n": 3}}) == 3 * 2 * 8 + 1 + 4 + 3 * 8  # scalar: 1
    assert lb._tensor_elements(defn, {"axes": {}}) == 0          # unresolvable: no order, no crash


def _wide_sol_root(tmp_path: Path, sizes=(4, 9, 16, 64, 128)) -> Path:
    root = tmp_path / "solw" / "benchmark"
    pdir = root / "L1" / "001_rms"
    pdir.mkdir(parents=True)
    (pdir / "definition.json").write_text(json.dumps(dict(RMS_DEF, name="001_rms")))
    (pdir / "workload.jsonl").write_text("\n".join(json.dumps(
        {"uuid": f"u-{i}", "axes": {"batch_size": b},
         "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}})
        for i, b in enumerate(sizes)) + "\n")
    return root


def test_check_workloads_ride_the_card_judged_never_timed(tmp_path: Path, base: Path) -> None:
    """--check-workload (the rest of a size group wider than the card times): problem.json
    check_only, minus what --workload already names; never in the group (so never timed);
    the meta workloads' smallest / largest span them; the student's stub and README say the
    kernel must run them."""
    root = _wide_sol_root(tmp_path)
    p = lb.sol_problem(root, "L1/001_rms", "1,2", check_workloads="0-4")
    assert [g["name"] for g in p.group] == ["wl1", "wl2"]
    assert [g["name"] for g in p.check_only] == ["wl0", "wl3", "wl4"]       # wl1/wl2 dropped
    out = lb.problem_json(p)
    assert [g["name"] for g in out["check_only"]] == ["wl0", "wl3", "wl4"]
    assert [g["name"] for g in out["group"]] == ["wl1", "wl2"]
    # the group's true extremes are correctness-only workloads: the meta tools read them
    assert out["meta_workloads"] == ["wl1", "wl0", "wl4"]
    assert "check_only" not in lb.problem_json(lb.sol_problem(root, "L1/001_rms", "1,2"))
    assert scaffold_main(["rms", "--base-dir", str(base), "--leaderboard",
                          "sol-execbench", "--dataset", str(root), "--definition",
                          "L1/001_rms", "--workload", "1,2", "--check-workload", "0-4",
                          "--no-sol-board"]) == 0
    card = base / "operators" / "rms"
    problem = json.loads((card / lb.PROBLEM_FILE).read_text())
    assert [g["name"] for g in problem["check_only"]] == ["wl0", "wl3", "wl4"]
    stub = (card / "kernel.py").read_text()
    assert "this card's 2 workloads: 9, 16" in stub and "ALSO 4, 64, 128" in stub
    readme = (card / "README.md").read_text()
    assert "Correctness-only workloads, not timed" in readme and "`wl4`" in readme
    assert "Correctness-only workloads (wl0, wl3, wl4)" in (card / "eval" / "bench.py").read_text()
    from forgekernel.loop import meta_shapes
    assert meta_shapes.card_meta_workloads(card) == ["wl1", "wl0", "wl4"]
    assert meta_shapes.card_workload_axes(card)["wl4"] == {"batch_size": 128}


def test_check_workload_needs_the_leaderboard(base: Path) -> None:
    with pytest.raises(SystemExit):
        scaffold_main(["rms", "--base-dir", str(base), "--check-workload", "3"])
