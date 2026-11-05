"""Leaderboard mode: judge and time a card the way FlashInfer-Bench or SOL-ExecBench does.
Copy into <op>/eval/ (``forge init <op> --leaderboard ...`` does), do not edit.

A leaderboard card's verdicts must be the leaderboard's verdicts. A kernel that passes
the card's own tolerance but fails the suite's element-wise check scores zero on the
board; a kernel tuned under this repo's default device window (warm L2, back-to-back
calls amortized) is optimized for a number the board never measures — both suites time
one call at a time with L2 flushed before it. So in leaderboard mode the card does not
choose its own judge or clock: this file reproduces each suite's published evaluation
protocol, pinned to an upstream commit, and the driver gates on what it reports.

Provenance: the input generators and the per-suite correctness and timing protocol are
adapted from FlashInfer-Bench (Copyright (c) 2025 by FlashInfer-Bench Contributors) and
SOL-ExecBench (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), both under the Apache
License, Version 2.0, and modified to run inside a ForgeKernel task. See NOTICE at the
repository root.

What is reproduced, per suite (upstream source in brackets):

  FlashInfer-Bench  (flashinfer-ai/flashinfer-bench @ 15618a6, bench/evaluators/default.py,
                     bench/utils.py, bench/config.py + eval_config.yaml)
    inputs       ``num_trials`` (3) input sets, drawn fresh and UNSEEDED like upstream:
                 float randn, fp8/fp4 randn clamped to +-2, ints randint, bools; scalar and
                 safetensors inputs from the workload; sampling ``probs`` softmaxed.
    correctness  reference (the definition's ``run``) on every set first, then the
                 solution; per output: shape, dtype, any inf / nan in the solution, then
                 element-wise — an element fails iff |d| > atol AND |d|/(|ref|+1e-8) > rtol;
                 the output fails when the matched fraction < required_matched_ratio
                 (default 1.0: every element). atol = rtol = 1e-2; op_type ``moe`` -> 0.95
                 (eval_config.yaml), ``moe_fp8_block_scale*`` -> 0.95 when unset (LowBit).
    timing       per set: 10 warmup + 50 timed calls, L2 flushed before each, per-call
                 GPU span (first kernel start -> last kernel end, CUPTI), median; the
                 reading is the MEAN over the sets. Speedup = baseline / solution.
    not covered  the sampling (TVD) and DSA top-k indexer evaluators — ``forge new``
                 refuses those definitions instead of judging them the wrong way.

  SOL-ExecBench     (NVIDIA/SOL-ExecBench @ a9fa080, driver/templates/eval_driver.py,
                     core/bench/{io,correctness,timing,reward_hack}.py)
    inputs       seed 200 once, then 10 rounds, each drawing fresh inputs: the name
                 heuristics (norm weights = 1, biases = 0, causal / binary masks, rope
                 cos/sin, positive stats, SSM decays, softmax outputs, fan-in scaled
                 weights) before randn; ``custom_inputs_entrypoint`` from the reference;
                 safetensors / scalar inputs from the workload.
    correctness  per round: reference first, then the solution; round 0 also checks
                 output types (exactly torch.Tensor), shape, dtype, and snapshots the
                 thread count; every round: inf / nan in EITHER tensor fails (matching
                 -inf only with ``allow_negative_inf``), an all-zero solution against a
                 non-zero reference fails, then allclose-style |d| <= atol + rtol*|ref|
                 with the WORKLOAD's own tolerance (matched ratio default 0.99, optional
                 max_error_cap).
    timing       the last round's inputs, 10 warmup + 50 timed calls, every call on
                 freshly shifted buffers (new data_ptr), L2 flushed before each, per-call
                 GPU span over kernels AND memcpy / memset activities, median. Thread
                 injection during timing fails the workload. The board times at a
                 locked SM 1500 / memory 3996 MHz (the card's gpu_entry.sh holds the
                 same lock and says so in LB_CLOCK_STATE).
    score        S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)) (sol_score.py), with the
                 board's per-workload T_b / T_SOL frozen into problem.json
                 (``sol_board``) at ``forge new``.
    not covered  the paper's LLM review.

The clock. Both suites read a per-call kernel span from CUPTI. The timer used is the
first available of (``FK_LB_TIMER`` forces one):

  upstream  the suite's own function, when its package is importable in the job image
            (flashinfer.testing.bench_gpu_time_with_cupti / sol_execbench time_runnable)
            and cupti-python is installed;
  cupti     the suites' method on cupti-python directly (SOL-ExecBench's collector,
            reduced): discover the call's activity sequence, then per timed call take
            first-start -> last-end of exactly that sequence, with each suite's own
            choices — FlashInfer-Bench: kernels only, synchronize after the flush, API
            tracing on; SOL-ExecBench: kernels + memcpy / memset, no synchronize;
  profiler  the same selection over torch.profiler's CUPTI activity (no extra package).
            Single-kernel spans match; the collector differs, which can move a
            multi-kernel call. The eval JSON carries a warning whenever it produced the
            gate number;
  events    CUDA events around each call, L2 flushed before it — the suites' own
            fallback when CUPTI is missing. NOT host-free: launch latency is inside.

``device_window_basis`` records which one produced the gate number (``cupti_kernel_span``
for the first three, ``cold_l2_events`` for the last) and ``leaderboard.timer`` names it,
so an archive or a status row always says what it is. The card's gpu_entry.sh installs
cupti-python when the image lacks it (``LB_INSTALL_CUPTI=0`` skips); bake it into the
image to save the per-job install.

The in-job reference rides the same clock where there is one worth timing:
FlashInfer-Bench cards time the dataset's ``baseline`` solution (what the board ranks
against), both sides measured mirrored after a burn-in (ref, kernel, kernel, ref —
timing.measure_paired's rule) on every input set. SOL-ExecBench cards time no
reference (problem.json ``time_reference`` is false): the only one in reach is the
definition's eager PyTorch run — many small launches, host-bound, a per-call span that
flips between readings — while the board scores against its own T_b, which the card
carries instead. The eval JSON maps the leaderboard reading into the keys the driver
gates on:

    device_median_ms / ref_device_ms / vs_ref_device   the suite reading (gate)
    bwd_median_ms / ref_median_ms / vs_ref             timing.time_e2e, for the log
    host_enqueue_ms, device_window_basis               the precondition audit
    leaderboard                                        suite, timer, per-set readings,
                                                       clock, SOL score (sol_board)

A GROUP card (problem.json ``group``: several workloads of the definition, written by
``forge new ... --workload 3,5,9``) runs the suite's protocol above on every workload in
turn -- correctness, then timing, workload by workload, so each one's thread snapshot and
burn-in are its own, as on the board -- and passes only when every workload passes (the
first failure ends the run, named). Its eval JSON keeps the multi-workload contract of
timing.workloads_performance: the gate keys above are the geometric mean over the
workloads (host-bound ratios: the worst), ``aggregate`` = "geomean", ``n_workloads``, and
one per-workload block (with its ``name``) under ``performance.workloads``; the ref_* /
vs_* keys appear only when every workload produced a reference reading. The SOL score is
reported per workload and as their mean. NCU profiles the first workload.

The loop's meta tools (lever pricing, stall / SASS pricing, the timeline probe) read a
group card one workload at a time: ``FK_LB_WORKLOAD_SCOPE`` = ``representative`` or a
group entry's name (``wl5``) narrows both the bench and ``ncu_window`` to that workload,
and the JSON then carries ``workload_scope`` (a meta reading, never a gate). The workloads
they read are problem.json ``meta_workloads`` (the representative, the group's smallest
and its largest), written by ``forge new``.

A card may also list CORRECTNESS-ONLY workloads (problem.json ``check_only``, written by
``forge new ... --check-workload 8-15``): workloads its kernel is scored on elsewhere (a
size group wider than the card times) that it must therefore run, and run right, but that
the gate does not time. Every run that judges the card (``--mode all`` / ``correctness``)
judges them first, by the same protocol, and the card fails on the first one that fails
(named, ``check_only: true`` in its verdict row) -- a kernel specialised to the timed
workloads (``unsupported M = 192``) fails here instead of at the harvest. They are never
timed and never in the geomean; a meta reading (``FK_LB_WORKLOAD_SCOPE``) may name one,
which then times it alone like any other scoped workload.

Entry points (the generated eval/bench.py and eval/ncu_run.py call these):

    run_leaderboard_bench()   --mode all|correctness|perf|ref  [--impl kernel|ref]
    ncu_window()              the kernel's calls inside a profiler start/stop window
"""

from __future__ import annotations

import argparse
import ast
import bisect
import contextlib
import importlib.machinery
import importlib.util
import inspect
import io
import json
import math
import operator
import os
import random
import re
import statistics
import sys
import threading
import time
import tokenize
from pathlib import Path

import torch

SUITE_FIB = "flashinfer-bench"
SUITE_SOL = "sol-execbench"
# A forgekernel task (``forge new tasks/<op>``): no board -- the judge is the PyTorch
# reference the teacher wrote from the task's description, the input groups are the task's,
# the clock is timing.py's. Section "the forgekernel task suite" below.
SUITE_TASK = "forgekernel"
SUITES = (SUITE_FIB, SUITE_SOL, SUITE_TASK)

UPSTREAM = {
    SUITE_FIB: "flashinfer-ai/flashinfer-bench@15618a67 (2026-09-25)",
    SUITE_SOL: "NVIDIA/SOL-ExecBench@a9fa0804 (2026-07-15)",
    SUITE_TASK: "forgekernel task protocol (docs/design/v1-launch.md)",
}

# FlashInfer-Bench bench/config.py ResolvedEvalConfig + bench/eval_config.yaml.
FIB_DEFAULTS = {"warmup_runs": 10, "iterations": 50, "num_trials": 3,
                "rtol": 1e-2, "atol": 1e-2, "required_matched_ratio": None}
FIB_OP_TYPE_CONFIG = {"moe": {"required_matched_ratio": 0.95}}
FIB_LOWBIT_MARK = "moe_fp8_block_scale"      # LowBitEvaluator.can_evaluate
FIB_LOWBIT_RATIO = 0.95

# SOL-ExecBench eval_driver.py / BenchmarkConfig / ToleranceSpec.
SOL_ROUNDS = 10
SOL_SEED = 200
SOL_WARMUP = 10
SOL_ITERS = 50
SOL_TOLERANCE_DEFAULTS = {"max_atol": 1e-2, "max_rtol": 1e-2, "required_matched_ratio": 0.99,
                          "max_error_cap": None, "allow_negative_inf": False}
SOL_POOL_ALIGNMENT = 256      # ShiftingMemoryPoolAllocator._POOL_ALIGNMENT
SOL_MAX_SHIFT_BYTES = 2048    # ShiftingMemoryPoolAllocator._MAX_SHIFT_BYTES

# device_window_basis values (forgekernel.baseline.BASIS_CUPTI is the host-free one).
BASIS_CUPTI = "cupti_kernel_span"
BASIS_COLD_EVENTS = "cold_l2_events"

TIMERS = ("upstream", "cupti", "profiler", "events")
# Packed fp4 (two E2M1 codes per byte); older torch builds lack it, and only fp4
# definitions need it.
_FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)
BURN_IN_S = 4.0
PROBLEM_FILE = Path(__file__).resolve().parent / "leaderboard" / "problem.json"


# --- problem / definition ------------------------------------------------------

def load_problem(path=None) -> dict:
    """The card's frozen problem (``eval/leaderboard/problem.json``)."""
    p = Path(path) if path else PROBLEM_FILE
    problem = json.loads(p.read_text(encoding="utf-8"))
    if problem.get("suite") not in SUITES:
        raise ValueError(f"{p}: suite must be one of {SUITES}, got {problem.get('suite')!r}")
    problem["_dir"] = str(p.parent)
    return problem


def workload_views(problem: dict) -> list[dict]:
    """The card's workloads, each as a problem dict judged and timed on its own: a group
    card's every ``group`` entry (its workload, its board row, ``_wl_name``), or the card
    itself."""
    group = problem.get("group") or []
    if len(group) <= 1:
        return [problem]
    out = []
    for g in group:
        v = dict(problem)
        v["workload"] = g["workload"]
        v.pop("sol_board", None)
        if g.get("sol_board"):
            v["sol_board"] = g["sol_board"]
        v["_wl_name"] = str(g.get("name") or f"wl{g.get('index')}")
        out.append(v)
    return out


def check_views(problem: dict) -> list[dict]:
    """The card's correctness-only workloads (problem.json ``check_only``), each as a
    problem dict judged on its own -- never timed, never in the gate -- or []."""
    out = []
    for g in problem.get("check_only") or []:
        v = {k: x for k, x in problem.items() if k not in ("group", "check_only", "sol_board")}
        v["workload"] = g["workload"]
        v["_wl_name"] = str(g.get("name") or f"wl{g.get('index')}")
        v["_check_only"] = True
        out.append(v)
    return out


def describe_card_workloads(problem: dict) -> str:
    views = workload_views(problem)
    checks = check_views(problem)
    extra = (f"; +{len(checks)} correctness-only: {','.join(v['_wl_name'] for v in checks)}"
             if checks else "")
    if len(views) == 1:
        return f"workload={problem['workload'].get('uuid')}{extra}"
    return (f"workloads={','.join(v['_wl_name'] for v in views)} (group of {len(views)}; "
            f"geomean gate){extra}")


_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def resolve_shape_expression(expr: str, env: dict) -> int:
    """An ``expr`` axis (SOL-ExecBench core/data/shapes.py): arithmetic over axis names."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise NameError(f"Unknown variable: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    value = ev(ast.parse(expr, mode="eval").body)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"expression {expr!r} is not an integer: {value}")
        value = int(value)
    return value


def resolved_axes(definition: dict, workload_axes: dict) -> dict:
    """Const axes, the workload's var axes, then every expr axis."""
    axes = definition.get("axes") or {}
    out = {n: int(a["value"]) for n, a in axes.items() if a.get("type") == "const"}
    out.update({n: int(v) for n, v in (workload_axes or {}).items()})
    missing = [n for n, a in axes.items() if a.get("type") == "var" and n not in out]
    if missing:
        raise ValueError(f"workload gives no value for var axes {missing}")
    for n, a in axes.items():
        if a.get("type") == "expr":
            out[n] = resolve_shape_expression(a["expression"], out)
    return out


def tensor_shapes(specs: dict, axes: dict) -> dict:
    """name -> concrete shape tuple (None for a python scalar)."""
    out = {}
    for name, spec in specs.items():
        shape = spec.get("shape")
        if shape is None:
            out[name] = None
            continue
        out[name] = tuple(int(a) if str(a).isdigit() else int(axes[a]) for a in shape)
    return out


def torch_dtype(name: str) -> torch.dtype:
    table = {
        "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2, "int64": torch.int64, "int32": torch.int32,
        "int16": torch.int16, "int8": torch.int8, "bool": torch.bool,
    }
    if name in ("float4_e2m1", "float4_e2m1fn_x2"):
        if _FP4X2 is None:
            raise ValueError(f"dtype {name!r} needs a torch build with float4_e2m1fn_x2")
        return _FP4X2
    if name not in table:
        raise ValueError(f"Unsupported dtype '{name}'")
    return table[name]


def _exec_source(source: str, filename: str) -> dict:
    ns: dict = {"__name__": f"leaderboard_{Path(filename).stem}", "__file__": filename}
    exec(compile(source, filename, "exec"), ns)  # noqa: S102 — the suite's own reference
    return ns


def reference_callables(problem: dict):
    """(run, custom_inputs_fn | None) from the definition's reference source."""
    defn = problem["definition"]
    ns = _exec_source(defn["reference"], f"<{defn['name']} reference>")
    custom = defn.get("custom_inputs_entrypoint")
    return ns["run"], (ns[custom] if custom else None)


def load_entry(path: Path, entry: str):
    """``entry`` from the python file at ``path``, imported as a module with its own
    directory on sys.path (a solution may import its sibling files, at call time too)."""
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        f"lb_{path.parent.name}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # under the correctness monitor, its hook watches the kernel's calls; a by-path
    # load never goes through the import hook it installs, so hand the module over
    hook = sys.modules.get("fk_monitor_hook")
    if hook is not None and hasattr(hook, "watch_loaded"):
        hook.watch_loaded(mod, str(path))
    return getattr(mod, entry)


def load_binary_entry(path: Path, module: str, entry: str):
    """``entry`` of a compiled extension — a sealed binary reference (problem.json
    reference_impl kind ``binary``, ``ref_bin/<module>.so``). Loaded under its build-time
    module name, which fixes its ``PyInit_`` symbol; torch is already imported, which the
    extension links against."""
    path = Path(path).resolve()
    loader = importlib.machinery.ExtensionFileLoader(module, str(path))
    spec = importlib.util.spec_from_loader(module, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return getattr(mod, entry)


def load_sealed_entry(path: Path, module: str, entry: str | None, key_env: str = "FK_REF_KEY"):
    """``entry`` of a sealed SOURCE reference (problem.json reference_impl kind
    ``sealed``, ``ref_bin/ref_payload.bin``): decoded in memory by the closed_ref.py next
    to this file, with the key from the job environment. The callable runs the
    reference inside its source window and withholds its exceptions' text."""
    cr_path = Path(__file__).resolve().with_name("closed_ref.py")
    spec = importlib.util.spec_from_file_location("_lb_closed_ref", cr_path)
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    fn = cr.entry(entry, str(Path(path).resolve()), name=module, key_env=key_env)
    # Decoded: the key has done its job, and the student's kernel is imported into this
    # same process next (the fence refuses a kernel that names the key or os.environ).
    os.environ.pop(key_env, None)
    return fn


_PRECISION: dict = {}


def _precision():
    """The card's eval/precision.py (the precision gate's measure), or None when the card
    predates it."""
    if "mod" not in _PRECISION:
        path = Path(__file__).resolve().with_name("precision.py")
        mod = None
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_lb_precision", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        _PRECISION["mod"] = mod
    return _PRECISION["mod"]


# A CUDA C++ kernel file (problem.json kernel_file ``*.cu``, loop language ``cuda-sol``) is
# compiled here the way SOL-ExecBench builds a cuda_cpp solution with compile_options: torch
# cpp_extension, the file's own PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) binding, CUTLASS
# headers on the include path, -O3 --use_fast_math and -lcuda (SOL's defaults), sm_100a unless
# a flag names an arch, plus the flags of any ``// fk-cuda-cflags: ...`` line. `forge <op>
# leaderboard export` writes the same list into the solution's compile_options, so the bytes
# judged and timed here are the bytes the board builds. The file is compiled alone (copied
# into its build directory): a solution is this one file.
CUDA_KERNEL_SUFFIXES = (".cu",)
CUDA_DEFAULT_CFLAGS = ["-O3", "--use_fast_math"]
CUDA_DEFAULT_LDFLAGS = ["-lcuda"]
CUDA_GENCODE = "-gencode=arch=compute_100a,code=sm_100a"
CUDA_FLAGS_LINE = re.compile(r"^[ \t]*//[ \t]*fk-cuda-cflags:(.*)$", re.M)
# "1" from the loop's stall pricing (recon/stall_price_run.py): the build adds -lineinfo.
CUDA_LINEINFO_ENV = "FK_CUDA_LINEINFO"
# What an fk-cuda-cflags line may hold: code generation and diagnostics, never a path, a
# library or a linker option (those could pull the sealed reference into the build).
CUDA_FLAG_OK = re.compile(
    r"-O[0-3]|--use_fast_math|-use_fast_math|--?ftz=(true|false)|--?prec-(div|sqrt)=(true|false)"
    r"|--?fmad=(true|false)|-lineinfo|--?extra-device-vectorization|--?maxrregcount=\d+"
    r"|-Xptxas=[-\w,=.]+|--ptxas-options=[-\w,=.]+|-D[A-Za-z_]\w*(=[-\w.]+)?|-U[A-Za-z_]\w*"
    r"|-std=c\+\+\d\d|--?expt-relaxed-constexpr|--?expt-extended-lambda|--?restrict"
    r"|-gencode=arch=compute_\w+,code=(sm|compute)_\w+|-arch=(sm|compute)_\w+"
    r"|--?default-stream=(legacy|null|per-thread)|-Xcompiler=-O[0-3]|-w|-Wno-[-\w]+")


def cuda_kernel_flags(src: str) -> list[str]:
    """nvcc flags of a CUDA kernel file: SOL's defaults, the file's fk-cuda-cflags, the arch.
    Raises ValueError on a flag outside CUDA_FLAG_OK."""
    extra = [f for line in CUDA_FLAGS_LINE.findall(src) for f in line.split()]
    if bad := [f for f in extra if not CUDA_FLAG_OK.fullmatch(f)]:
        raise ValueError(f"fk-cuda-cflags: {bad} not allowed (code generation / diagnostics "
                         f"flags only: no paths, libraries or linker options)")
    flags = CUDA_DEFAULT_CFLAGS + [f for f in extra if f not in CUDA_DEFAULT_CFLAGS]
    if not any(f.startswith(("-gencode", "-arch")) for f in flags):
        flags.append(CUDA_GENCODE)
    return flags


def load_cuda_entry(path: Path, entry: str):
    """``entry`` of a CUDA C++ kernel file, built as SOL-ExecBench builds a cuda_cpp solution
    (see CUDA_KERNEL_SUFFIXES). The build is keyed by the source and the flags, so an
    unchanged file loads the module already built in this job's build directory."""
    import hashlib
    import tempfile
    import torch.utils.cpp_extension as ext
    path = Path(path).resolve()
    src = path.read_text(encoding="utf-8")
    flags = cuda_kernel_flags(src)
    # A profiling run that maps SASS back to source lines (the stall ledger) builds with
    # -lineinfo: its own build under its own prefix, so the timed build and the census /
    # SASS-sha readers (orch_census_run.find_cuda_build: fk_kernel_*) never load it.
    lineinfo = os.environ.get(CUDA_LINEINFO_ENV) == "1" and "-lineinfo" not in flags
    if lineinfo:
        flags = flags + ["-lineinfo"]
    key = hashlib.sha256("\0".join([src, " ".join(flags), str(torch.__version__)]).encode()).hexdigest()
    module = f"{'fk_lineinfo_' if lineinfo else 'fk_kernel_'}{key[:12]}"
    root = os.environ.get("FK_KERNEL_BUILD_DIR") or os.path.join(tempfile.gettempdir(), "fk_kernel_build")
    build = Path(root) / module
    build.mkdir(parents=True, exist_ok=True)
    # Rewrite the build's source copy only when it differs: a fresh mtime marks the
    # object dirty for ninja, and every new process (each bench / pricing variant /
    # NCU target of one job) would recompile an unchanged kernel (~35 s with CUTLASS).
    staged = build / path.name
    try:
        same = staged.read_text(encoding="utf-8") == src
    except (OSError, UnicodeDecodeError):
        same = False
    if not same:
        staged.write_text(src, encoding="utf-8")
    cutlass = os.environ.get("CUTLASS_DIR", "/usr/local/cutlass")
    mod = ext.load(name=module, sources=[str(build / path.name)],
                   extra_cuda_cflags=list(flags), extra_ldflags=list(CUDA_DEFAULT_LDFLAGS),
                   extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
                   build_directory=str(build), verbose=False, is_python_module=True)
    if not hasattr(mod, entry):
        raise AttributeError(f"{path.name}: the module defines no {entry!r} "
                             f"(PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def(\"{entry}\", ...); }})")
    return getattr(mod, entry)


# A binary reference runs in the student's process, so the kernel file may not reach for
# it, for the loader machinery, or for the compiled-code toolchain (comments ignored,
# strings not: a path in a string is a reach too). The reference's module name is added
# per card.
KERNEL_FENCE = (r"ref_bin", r"ExtensionFileLoader", r"extension_file", r"cpp_extension",
                r"load_inline", r"load_library", r"\bctypes\b", r"\bcffi\b", r"cuobjdump",
                r"nvdisasm", r"\bsubprocess\b", r"os\.system", r"\bimportlib\b", r"PyInit_",
                r"sys\.modules", r"__main__", r"__loader__", r"_getframe", r"\bgc\b",
                r"\binspect\b", r"\.so\b")
# A sealed SOURCE reference (closed_ref.py) runs in the student's process as a module
# whose source the loader holds in memory, whose JIT objects keep it (Triton's .src),
# and whose compiled IR lands in the JIT's on-disk cache: the kernel file may not reach
# for the loader, the payload, its key, the interpreter's reflection hooks (a tracer
# sees the reference's frames and globals) or the JIT caches and dumps.
SEALED_FENCE = (r"closed_ref", r"ref_payload", r"_REF_KEY", r"\blinecache\b", r"/nonexistent",
                r"unseal", r"keystream", r"\bzlib\b", r"b85decode", r"settrace", r"setprofile",
                r"\bmonitoring\b", r"f_globals", r"f_locals", r"f_back", r"tb_frame", r"__globals__",
                r"__builtins__", r"__import__", r"\bbuiltins\b", r"__closed_ref", r"meta_path",
                r"\.triton\b", r"triton/cache", r"TRITON_CACHE", r"TRITON_DUMP", r"TRITON_KERNEL_DUMP",
                r"MLIR_ENABLE_DUMP", r"CUTE_DSL_KEEP", r"CUTE_DSL_DUMP", r"\.ttg?ir\b", r"\.llir\b",
                r"\.ptx\b", r"\.cubin\b", r"\benviron\b", r"/proc/")


def _code_without_comments(src: str) -> str:
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


# A CUDA C++ kernel file runs in the reference's process too, and C++ reaches further than
# Python: the dynamic loader, the interpreter's C API and pybind11's, the CUDA module and
# library loaders (a loaded fatbin's kernels), the file system, the environment, other
# processes. Comments are ignored, strings are not.
CUDA_KERNEL_FENCE = (r"ref_bin", r"PyInit_", r"\.so\b", r"\bdl(open|mopen|sym|vsym|info|addr)\b",
                     r"dl_iterate_phdr", r"link_map", r"RTLD_", r"\bPy[A-Z]\w*", r"\b_Py\w*",
                     r"\bpy::", r"\bpybind11::", r"\bimport\s*\(", r"c10::Dispatcher", r"findSchema",
                     r"torch::jit", r"\bcu(Module|Library|Kernel)\w*", r"\bcuda(Library|Kernel)\w*",
                     r"cudaGetFuncBySymbol", r"Enumerate", r"\bgetenv\b", r"\benviron\b", r"/proc/",
                     r"\bsystem\s*\(", r"\bpopen\b", r"\bexec[lv]\w*\s*\(", r"\bfork\s*\(",
                     r"\bfopen\b", r"fstream", r"\bopen\s*\(", r"\bmmap\b", r"readlink", r"opendir",
                     r"cuobjdump", r"nvdisasm", r"#\s*include\s*\"[^\"]*\.(so|py)\"",
                     r"##", r"syscall")   # token pasting would spell any of the above


def _cpp_code_without_comments(src: str) -> str:
    """C/C++ source with // and /* */ comments blanked (string and char literals kept)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def kernel_fence(problem: dict) -> str | None:
    """Why the student's kernel file may not run next to this card's sealed (binary or
    source) reference, or None (also None for any other kind of reference)."""
    ref = problem.get("reference_impl") or {}
    kind = ref.get("kind")
    if kind not in ("binary", "sealed"):
        return None
    card = Path(problem["_dir"]).parent.parent
    kfile = problem.get("kernel_file", "kernel.py")
    src = (card / kfile).read_text()
    cuda = Path(kfile).suffix in CUDA_KERNEL_SUFFIXES
    if cuda:
        try:
            cuda_kernel_flags(src)
        except ValueError as e:
            return f"the kernel's build flags reach outside code generation ({e})"
    code = _cpp_code_without_comments(src) if cuda else _code_without_comments(src)
    pats = list(CUDA_KERNEL_FENCE if cuda else KERNEL_FENCE) \
        + (list(SEALED_FENCE) if kind == "sealed" else []) \
        + ([re.escape(ref["module"])] if ref.get("module") else []) \
        + ([re.escape(ref["key_env"])] if ref.get("key_env") else []) \
        + [r"(^|[^A-Za-z0-9_])" + re.escape(n) + r"([^A-Za-z0-9_]|$)"
           for n in ref.get("deny_modules") or []]   # the reference's installed package
    m = re.search("|".join(pats), code, re.M)
    what = "binary reference or compiled-code tooling" if kind == "binary" else \
        "reference, its loader, its key or the JIT's caches"
    return (f"the kernel reaches for the sealed {what} "
            f"({m.group(0)!r} matches the card's fence)") if m else None


# --- inputs ------------------------------------------------------------------------

def _fib_rand_tensor(shape, dtype, device):
    """flashinfer_bench/bench/utils.py _rand_tensor."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, _FP4X2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        low, high = {torch.int8: (-128, 128)}.get(dtype, (-1024, 1024))
        return torch.randint(low, high, shape, device=device, dtype=dtype)
    raise ValueError(f"Unsupported random dtype: {dtype}")


def _cast_to_fp4x2(x):
    """SOL-ExecBench core/bench/io.py _cast_to_fp4x2 (E2M1 codes packed 2 per byte)."""
    r = torch.zeros_like(x, dtype=torch.uint8)
    r[(x >= 0.0) & (x <= 0.25)] = 0
    r[(x > 0.25) & (x < 0.75)] = 1
    r[(x >= 0.75) & (x <= 1.25)] = 2
    r[(x > 1.25) & (x < 1.75)] = 3
    r[(x >= 1.75) & (x <= 2.5)] = 4
    r[(x > 2.5) & (x < 3.5)] = 5
    r[(x >= 3.5) & (x <= 5.0)] = 6
    r[x > 5.0] = 7
    r[(x >= -0.25) & (x < 0.0)] = 8
    r[(x < -0.25) & (x > -0.75)] = 9
    r[(x <= -0.75) & (x >= -1.25)] = 10
    r[(x < -1.25) & (x > -1.75)] = 11
    r[(x <= -1.75) & (x >= -2.5)] = 12
    r[(x < -2.5) & (x > -3.5)] = 13
    r[(x <= -3.5) & (x >= -5.0)] = 14
    r[x < -5.0] = 15
    return (r[..., ::2] + r[..., 1::2] * 16).view(_FP4X2)


def _sol_rand_tensor(shape, dtype, device):
    """SOL-ExecBench core/bench/io.py _rand_tensor."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    if _FP4X2 is not None and dtype == _FP4X2:
        return _cast_to_fp4x2(torch.randn(shape, dtype=torch.float32, device=device))
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        low, high = {torch.int8: (-128, 128)}.get(dtype, (-1024, 1024))
        return torch.randint(low, high, shape, device=device, dtype=dtype)
    raise ValueError(f"Unsupported random dtype: {dtype}")


# SOL-ExecBench's name heuristics (core/bench/io.py), same predicates, same order.
_WEIGHT_SUFFIXES = ("_weight", "_weights", "_proj", "_projs", "_proj_weight",
                    "_proj_weights", "_weight_matrix")


def _is_weight_matrix(name, shape):
    if len(shape) < 2:
        return False
    if name.endswith(_WEIGHT_SUFFIXES) or name == "weight":
        return True
    stripped = name.rstrip("0123456789")
    return bool(stripped and stripped in ("weight",) or stripped.endswith(_WEIGHT_SUFFIXES))


def _is_norm_affine(name, whole, suffix):
    if name == whole:
        return True
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_causal_attention_mask(name, shape, description):
    if len(shape) < 2 or shape[-1] != shape[-2]:
        return False
    if name in ("attention_mask", "causal_mask"):
        return True
    d = (description or "").lower()
    return bool(description and "attention mask" in d and "causal" in d)


def _is_binary_mask(name, description):
    if name in ("x_mask", "text_mask", "aspect_ratio_mask", "drop_mask", "attention_mask"):
        return True
    if name.endswith("_mask") and description:
        d = description.lower()
        return any(k in d for k in ("binary", "{0, 1}", "0 or 1", "1.0 for valid",
                                    "0.0 for masked"))
    return False


def _is_positive_tensor(name):
    names = ("rstd", "std", "variance", "var")
    if name in names or name.endswith(("_rstd", "_std", "_variance", "_var")):
        return True
    base = name.rstrip("0123456789")
    return base in names or base.endswith(("_rstd", "_std", "_var"))


def _is_softmax_output(name, description):
    if name in ("attn_weights", "attention_weights", "routing_weights"):
        return True
    d = (description or "").lower()
    return bool(description and "softmax" in d and "output" in d)


def _sol_heuristic_tensor(name, shape, dtype, device, description):
    """SOL-ExecBench _generate_heuristic_tensor; None = fall back to _rand_tensor."""
    if not dtype.is_floating_point:
        return None
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, _FP4X2):
        return None
    if _is_norm_affine(name, "norm_weight", "_weight"):
        return torch.ones(shape, dtype=dtype, device=device)
    if _is_norm_affine(name, "norm_bias", "_bias"):
        return torch.zeros(shape, dtype=dtype, device=device)
    if _is_causal_attention_mask(name, shape, description):
        seq_len = shape[-1]
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask[..., causal] = torch.finfo(dtype).min
        return mask
    if _is_binary_mask(name, description):
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-math.pi, math.pi)
        if name in ("cos", "cos_cached", "rope_cos"):
            return torch.cos(t).to(dtype)
        return torch.sin(t).to(dtype)
    if _is_positive_tensor(name):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1
    if name in ("A", "A_log", "A_cumsum", "g"):
        if name == "A_cumsum":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-0.1, 0.0).cumsum(dim=-1)
        if name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        if name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)
    if _is_softmax_output(name, description):
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)
    if _is_weight_matrix(name, shape):
        return torch.randn(shape, dtype=dtype, device=device) / math.sqrt(shape[-1])
    return None


_SAFETENSORS_CACHE: dict = {}


def _safetensor(problem: dict, name: str, spec: dict, shape, dtype) -> torch.Tensor:
    """A workload's safetensors input, resolved against the card's problem directory
    (``forge new`` copies the blobs there and rewrites the paths)."""
    import safetensors.torch as st
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(problem["_dir"]) / path
    path = path.resolve()
    if path not in _SAFETENSORS_CACHE:
        _SAFETENSORS_CACHE[path] = st.load_file(str(path))
    tensors = _SAFETENSORS_CACHE[path]
    if spec["tensor_key"] not in tensors:
        raise ValueError(f"Missing key '{spec['tensor_key']}' in '{path}'")
    t = tensors[spec["tensor_key"]]
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"'{name}' expected {list(shape)}, got {list(t.shape)}")
    if t.dtype != dtype:
        raise ValueError(f"'{name}' expected {dtype}, got {t.dtype}")
    return t.contiguous()


def gen_inputs(problem: dict, device, custom_inputs_fn=None, stress: str | None = None,
               touched: list | None = None) -> list:
    """One input set in definition order, drawn the way the problem's suite draws it.

    ``stress`` (a precision.STRESS_PROFILES name): every PLAIN normal float draw (not a
    heuristic tensor -- norm weights, masks, rope tables, positive statistics -- not a
    workload / custom / fp8 input) is passed through precision.stress_tensor; the names
    of the inputs it moved are appended to ``touched``. Only the precision record reads
    stressed inputs: correctness and timing never do."""
    defn, wl = problem["definition"], problem["workload"]
    suite = problem["suite"]
    axes = resolved_axes(defn, wl.get("axes") or {})
    shapes = tensor_shapes(defn["inputs"], axes)
    winputs = wl.get("inputs") or {}
    dev = torch.device(device)
    sampling = defn.get("op_type") == "sampling"
    custom = None
    if suite == SUITE_SOL and custom_inputs_fn is not None:
        scalars = {n: s["value"] for n, s in winputs.items() if s.get("type") == "scalar"}
        custom = custom_inputs_fn({**axes, **scalars}, dev)
    out = []
    for name, spec in defn["inputs"].items():
        dtype = torch_dtype(spec["dtype"])
        src = winputs.get(name) or {"type": "random"}
        kind = src.get("type", "random")
        if kind == "safetensors":
            out.append(_safetensor(problem, name, src, shapes[name], dtype).to(dev))
        elif kind == "scalar":
            out.append(src["value"])
        elif kind == "custom":
            if custom is None or name not in custom:
                raise RuntimeError(f"custom input '{name}' needs the definition's "
                                   f"custom_inputs_entrypoint")
            val = custom[name]
            out.append(val.to(device=dev) if isinstance(val, torch.Tensor) else val)
        else:
            shape = shapes[name]
            plain = False
            if suite == SUITE_FIB:
                if shape is None:
                    out.append(_fib_rand_tensor((), dtype, dev).item())
                    continue
                value = _fib_rand_tensor(shape, dtype, dev)
                plain = True
            else:
                if shape is None:
                    out.append(_sol_rand_tensor((), dtype, dev).item())
                    continue
                value = _sol_heuristic_tensor(name, tuple(shape), dtype, dev,
                                              spec.get("description"))
                if value is None:
                    value = _sol_rand_tensor(shape, dtype, dev)
                    plain = True
            if stress and plain and dtype in (torch.float32, torch.float16, torch.bfloat16):
                value = _precision().stress_tensor(value, stress)
                if touched is not None:
                    touched.append(name)
            if sampling and name == "probs":
                value = torch.softmax(value, dim=-1)
            out.append(value)
    return out


def allocate_outputs(problem: dict, device) -> list:
    defn = problem["definition"]
    axes = resolved_axes(defn, problem["workload"].get("axes") or {})
    shapes = tensor_shapes(defn["outputs"], axes)
    return [torch.empty(shapes[n], dtype=torch_dtype(s["dtype"]), device=device)
            for n, s in defn["outputs"].items()]


def normalize_outputs(problem: dict, result, device) -> list:
    """A value-returning call's result as the definition's output list.

    FlashInfer-Bench accepts a tensor / scalar / tuple / list; SOL-ExecBench also a
    dict keyed by output name (its normalize_outputs)."""
    defn = problem["definition"]
    names = list(defn["outputs"])
    dtypes = {n: torch_dtype(s["dtype"]) for n, s in defn["outputs"].items()}
    dev = torch.device(device)

    def as_tensor(name, v):
        if isinstance(v, torch.Tensor):
            return v.to(dev) if v.device != dev else v
        return torch.tensor(v, dtype=dtypes[name], device=dev)

    if isinstance(result, dict) and problem["suite"] == SUITE_SOL:
        got = {k: as_tensor(k, v) for k, v in result.items() if k in dtypes}
        return [got[n] for n in names]
    if isinstance(result, (tuple, list)):
        if len(result) != len(names):
            raise ValueError(f"Tuple/list has {len(result)} elements but {len(names)} "
                             f"outputs expected")
        return [as_tensor(n, v) for n, v in zip(names, result)]
    if len(names) != 1:
        raise ValueError(f"Single value returned but {len(names)} outputs expected")
    if not isinstance(result, (torch.Tensor, int, float, bool)):
        raise ValueError(f"Unexpected return type {type(result).__name__}")
    return [as_tensor(names[0], result)]


def call_impl(problem: dict, fn, inputs: list, dps: bool, device, *, no_grad: bool) -> list:
    """Run one implementation in its calling convention; its outputs, synchronized."""
    with (torch.no_grad() if no_grad else contextlib.nullcontext()):
        if dps:
            outs = allocate_outputs(problem, device)
            fn(*inputs, *outs)
        else:
            outs = normalize_outputs(problem, fn(*inputs), device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return outs


# --- judges ------------------------------------------------------------------------

def fib_config(problem: dict) -> dict:
    """The resolved FlashInfer-Bench eval config for this definition (or the card's
    frozen one when ``forge new`` wrote ``eval_config``)."""
    frozen = problem.get("eval_config")
    if frozen:
        return dict(frozen)
    defn = problem["definition"]
    cfg = dict(FIB_DEFAULTS)
    cfg.update(FIB_OP_TYPE_CONFIG.get(defn.get("op_type"), {}))
    if FIB_LOWBIT_MARK in defn["name"] and cfg.get("required_matched_ratio") is None:
        cfg["required_matched_ratio"] = FIB_LOWBIT_RATIO
    return cfg


def fib_error_stats(output, reference, atol: float, rtol: float,
                    required_matched_ratio=None):
    """flashinfer_bench/bench/utils.py compute_error_stats:
    (max_abs, max_rel, exceeds_tol, matched_ratio)."""
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + 1e-8)
    total = abs_error.numel()
    if total == 0:
        return 0.0, 0.0, False, 1.0
    need = required_matched_ratio if required_matched_ratio is not None else 1.0
    exceeds = float(((abs_error > atol) & (rel_error > rtol)).sum().item())
    matched = max(0.0, min(1.0, 1.0 - exceeds / float(total)))
    return (float(abs_error.max().item()), float(rel_error.max().item()),
            matched < need, matched)


def fib_judge(outs: list, refs: list, cfg: dict) -> dict:
    """One input set, FlashInfer-Bench DefaultEvaluator.check_correctness semantics.

    ``fatal``: upstream returns at once (shape, dtype, a non-finite solution value);
    a numerical miss is recorded and the remaining outputs and sets still run."""
    res = {"status": "PASSED", "fatal": False, "max_abs": 0.0, "max_rel": 0.0,
           "matched_ratio": 1.0}
    failed = []
    for i, (sol, ref) in enumerate(zip(outs, refs)):
        name = f"output[{i}]"
        if tuple(sol.shape) != tuple(ref.shape):
            return {**res, "status": "INCORRECT_SHAPE", "fatal": True,
                    "error": f"{name}: shape {tuple(sol.shape)} != reference {tuple(ref.shape)}"}
        if sol.dtype != ref.dtype:
            return {**res, "status": "INCORRECT_DTYPE", "fatal": True,
                    "error": f"{name}: dtype {sol.dtype} != reference {ref.dtype}"}
        kind = ("inf" if torch.isinf(sol).any().item()
                else "nan" if torch.isnan(sol).any().item() else None)
        if kind:
            return {**res, "status": "INCORRECT_NUMERICAL", "fatal": True,
                    "max_abs": float(kind), "max_rel": float(kind),
                    "error": f"{name}: solution contains {kind}"}
        mabs, mrel, exceeds, matched = fib_error_stats(
            sol, ref, cfg["atol"], cfg["rtol"], cfg.get("required_matched_ratio"))
        res["max_abs"] = max(res["max_abs"], mabs)
        res["max_rel"] = max(res["max_rel"], mrel)
        res["matched_ratio"] = min(res["matched_ratio"], matched)
        if exceeds:
            failed.append(name)
    if failed:
        need = cfg.get("required_matched_ratio") or 1.0
        res["status"] = "INCORRECT_NUMERICAL"
        res["error"] = (f"{', '.join(failed)}: matched {res['matched_ratio']:.6f} < {need} "
                        f"(an element fails iff |d| > {cfg['atol']} and |d|/|ref| > "
                        f"{cfg['rtol']}); max_abs={res['max_abs']:.4g} "
                        f"max_rel={res['max_rel']:.4g}")
    return res


def sol_tolerance(problem: dict) -> dict:
    tol = dict(SOL_TOLERANCE_DEFAULTS)
    tol.update(problem["workload"].get("tolerance") or {})
    return tol


def sol_sanity(sol, ref, allow_negative_inf: bool):
    """SOL-ExecBench check_tensor_sanity: a failure dict, or None."""
    ref_bad = ~torch.isfinite(ref)
    sol_bad = ~torch.isfinite(sol)
    if allow_negative_inf:
        both = (ref == float("-inf")) & (sol == float("-inf"))
        ref_bad = ref_bad & ~both
        sol_bad = sol_bad & ~both
    if ref_bad.any().item() or sol_bad.any().item():
        has_nan = bool(torch.isnan(sol).any().item() or torch.isnan(ref).any().item())
        return {"has_nan": has_nan, "has_inf": not has_nan,
                "why": "nan" if has_nan else "inf"}
    ref_norm = torch.linalg.vector_norm(ref.to(torch.float32)).item()
    if ref_norm > 0 and torch.linalg.vector_norm(sol.to(torch.float32)).item() == 0:
        return {"max_abs": float(ref_norm), "max_rel": float(ref_norm),
                "why": "all-zero solution against a non-zero reference"}
    return None


def sol_error_stats(output, reference, tol: dict):
    """SOL-ExecBench compute_error_stats: (stats dict, exceeds)."""
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    bad = sol_sanity(x, y, bool(tol.get("allow_negative_inf")))
    if bad is not None:
        return {"max_abs": bad.get("max_abs", float("nan")),
                "max_rel": bad.get("max_rel", float("nan")),
                "matched_ratio": 0.0, "why": bad["why"]}, True
    if tol.get("allow_negative_inf"):
        keep = ~((x == float("-inf")) & (y == float("-inf")))
        x, y = x[keep], y[keep]
    abs_error = torch.abs(x - y)
    total = abs_error.numel()
    if total == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0}, False
    max_abs = float(abs_error.max().item())
    bound = tol["max_atol"] + tol["max_rtol"] * torch.abs(y)
    exceeds_mask = (abs_error > bound) | ~torch.isfinite(abs_error)
    matched = max(0.0, min(1.0, 1.0 - float(exceeds_mask.sum().item()) / float(total)))
    exceeds = matched < tol["required_matched_ratio"]
    cap = tol.get("max_error_cap")
    if cap is not None and max_abs > cap:
        exceeds = True
    rel = abs_error / torch.clamp(torch.abs(y), min=tol["max_atol"])
    return {"max_abs": max_abs, "max_rel": float(rel.max().item()),
            "matched_ratio": matched}, exceeds


# --- correctness protocols -----------------------------------------------------------

def _impl(problem: dict, which: str):
    """(callable, dps) for ``kernel`` (the student's), ``ref`` (the in-job reference)
    or ``definition`` (the definition's own reference run)."""
    card = Path(problem["_dir"]).parent.parent
    if which == "kernel":
        kpath = card / problem.get("kernel_file", "kernel.py")
        load = load_cuda_entry if kpath.suffix in CUDA_KERNEL_SUFFIXES else load_entry
        return load(kpath, problem.get("entry", "run")), bool(problem.get("dps", True))
    ref = problem.get("reference_impl") or {"kind": "definition"}
    if which == "definition" or ref.get("kind") == "definition":
        return reference_callables(problem)[0], False
    if ref.get("kind") == "binary":
        return (load_binary_entry(Path(problem["_dir"]) / ref["path"], ref["module"],
                                  ref.get("entry", "run")),
                bool(ref.get("dps", True)))
    if ref.get("kind") == "sealed":
        return (load_sealed_entry(Path(problem["_dir"]) / ref["path"], ref["module"],
                                  ref.get("entry"), ref.get("key_env") or "FK_REF_KEY"),
                bool(ref.get("dps", True)))
    return (load_entry(Path(problem["_dir"]) / ref["path"], ref.get("entry", "run")),
            bool(ref.get("dps", False)))


def warm_reference(problem: dict, ref_fn, ref_dps: bool, device) -> None:
    """Call an in-job reference once before correctness. SOL snapshots the thread count
    after the solution's first call and flags more threads after timing as injection;
    a reference whose first call starts threads (a compiled extension's pool, a JIT)
    would otherwise have them counted against the solution when it is first timed."""
    inp = first_inputs(problem, device)
    call_impl(problem, ref_fn, inp, ref_dps, device, no_grad=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fib_correctness(problem: dict, impl, dps: bool, device, prec=None) -> tuple[dict, list]:
    """FlashInfer-Bench: build the baseline (every set and its reference outputs)
    first, then judge the solution on each set. Returns (verdict, the timed sets —
    FlashInfer-Bench times the same sets it judged)."""
    cfg = fib_config(problem)
    ref_run, _ = reference_callables(problem)
    sets, refs, pristine = [], [], []
    for t in range(int(cfg["num_trials"])):
        inp = gen_inputs(problem, device)
        sets.append(inp)
        if t < PRECISION_SETS:
            pristine.append(_precision_snapshot(prec, inp))
        refs.append(call_impl(problem, ref_run, inp, False, device, no_grad=True))
    verdict = {"pass": True, "status": "PASSED", "max_abs": 0.0, "max_rel": 0.0,
               "matched_ratio": 1.0, "checks": [],
               "tolerance": {"atol": cfg["atol"], "rtol": cfg["rtol"],
                             "required_matched_ratio": cfg.get("required_matched_ratio") or 1.0,
                             "rule": "an element fails iff |d| > atol and "
                                     "|d|/(|ref|+1e-8) > rtol"},
               "seed": torch.initial_seed()}
    for i, inp in enumerate(sets):
        try:
            outs = call_impl(problem, impl, inp, dps, device, no_grad=True)
        except Exception as exc:  # noqa: BLE001 — the solution is foreign code
            verdict.update(pass_=False, status="RUNTIME_ERROR",
                           error=f"set {i}: {type(exc).__name__}: {exc}")
            break
        r = fib_judge(outs, refs[i], cfg)
        if prec is not None and i < PRECISION_SETS:
            _precision_measure(problem, prec, f"set{i}", outs, refs[i], ref_run, pristine[i],
                               device)
        verdict["checks"].append({"set": i, **{k: v for k, v in r.items() if k != "fatal"}})
        verdict["max_abs"] = max(verdict["max_abs"], r["max_abs"])
        verdict["max_rel"] = max(verdict["max_rel"], r["max_rel"])
        verdict["matched_ratio"] = min(verdict["matched_ratio"], r["matched_ratio"])
        if r["status"] != "PASSED" and verdict["status"] == "PASSED":
            verdict.update(status=r["status"], error=f"set {i}: {r['error']}")
        if r["fatal"]:
            verdict["status"] = r["status"]
            verdict["error"] = f"set {i}: {r['error']}"
            break
    verdict["pass"] = verdict.pop("pass_", True) and verdict["status"] == "PASSED"
    return verdict, sets


def _lazy_output(outs: list):
    for t in outs:
        if type(t) is not torch.Tensor:
            return f"output is {type(t).__name__}, not torch.Tensor (lazy evaluation)"
    return None


def sol_correctness(problem: dict, impl, dps: bool, device, prec=None) -> tuple[dict, list]:
    """SOL-ExecBench: seed 200, ten rounds of fresh inputs, reference then solution.
    Returns (verdict, the timed sets — SOL times the last round's inputs only).
    ``verdict["threads_after_first_call"]`` is the thread count SOL snapshots after
    round 0 (JIT compilers start their threads on the first call); more threads after
    timing is SOL's thread-injection reward hack."""
    tol = sol_tolerance(problem)
    ref_run, custom = reference_callables(problem)
    random.seed(SOL_SEED)
    torch.manual_seed(SOL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SOL_SEED)
    verdict = {"pass": False, "status": "PASSED", "max_abs": 0.0, "max_rel": 0.0,
               "matched_ratio": 1.0, "checks": [], "tolerance": tol, "seed": SOL_SEED,
               "threads_after_first_call": None}
    inp: list = []

    def fail(status, error):
        verdict.update(status=status, error=error)
        return verdict, [inp]

    for rnd in range(SOL_ROUNDS):
        inp = gen_inputs(problem, device, custom_inputs_fn=custom)
        pristine = _precision_snapshot(prec, inp) if rnd < PRECISION_SETS else None
        refs = call_impl(problem, ref_run, inp, False, device, no_grad=False)
        try:
            outs = call_impl(problem, impl, inp, dps, device, no_grad=False)
        except Exception as exc:  # noqa: BLE001 — the solution is foreign code
            return fail("RUNTIME_ERROR", f"round {rnd}: {type(exc).__name__}: {exc}")
        if rnd == 0:
            verdict["threads_after_first_call"] = threading.active_count()
            if (why := _lazy_output(outs)):
                return fail("REWARD_HACK", why)
            for i, (sol, ref) in enumerate(zip(outs, refs)):
                if tuple(sol.shape) != tuple(ref.shape):
                    return fail("INCORRECT_SHAPE", f"output[{i}]: shape {tuple(sol.shape)} "
                                                   f"!= reference {tuple(ref.shape)}")
                if sol.dtype != ref.dtype:
                    return fail("INCORRECT_DTYPE",
                                f"output[{i}]: dtype {sol.dtype} != reference {ref.dtype}")
        if prec is not None and rnd < PRECISION_SETS:
            _precision_measure(problem, prec, f"round{rnd}", outs, refs, ref_run, pristine,
                               device)
        for i, (sol, ref) in enumerate(zip(outs, refs)):
            stats, exceeds = sol_error_stats(sol, ref, tol)
            verdict["checks"].append({"round": rnd, "output": i, **stats, "exceeds": exceeds})
            for k in ("max_abs", "max_rel"):
                if isinstance(stats.get(k), float) and not math.isnan(stats[k]):
                    verdict[k] = max(verdict[k], stats[k])
            verdict["matched_ratio"] = min(verdict["matched_ratio"], stats["matched_ratio"])
            if exceeds:
                why = stats.get("why") or (
                    f"matched {stats['matched_ratio']:.6f} < {tol['required_matched_ratio']}"
                    + (f" or max_abs {stats['max_abs']:.4g} > cap {tol['max_error_cap']}"
                       if tol.get("max_error_cap") is not None else ""))
                return fail("INCORRECT_NUMERICAL",
                            f"round {rnd} output[{i}]: {why} (|d| <= {tol['max_atol']} + "
                            f"{tol['max_rtol']}*|ref|)")
    verdict["pass"] = True
    return verdict, [inp]


# --- clocks --------------------------------------------------------------------------

def _flush_buffer(device):
    size = int(torch.cuda.get_device_properties(torch.device(device)).L2_cache_size) * 2
    return torch.empty(max(size, 1 << 20), dtype=torch.int8, device=device)


def _reset_persisting_l2():
    try:
        torch.cuda.cudart().cudaCtxResetPersistingL2Cache()
    except Exception:  # noqa: BLE001 — best effort, like the suites' own reset
        pass


class _Shifted:
    """SOL-ExecBench's ShiftingMemoryPoolAllocator, reduced: every timed call gets its
    tensors at a fresh data_ptr (cumulative random shifts of 256 B multiples, <= 2 KiB
    per step, seeded), refilled from the originals OUTSIDE the timed call."""

    def __init__(self, args: list, total_calls: int, seed: int):
        rng = random.Random(seed)
        mult = SOL_MAX_SHIFT_BYTES // SOL_POOL_ALIGNMENT
        self.offsets = [0]
        for _ in range(max(0, total_calls - 1)):
            self.offsets.append(self.offsets[-1] + rng.randint(1, mult))
        self.entries = []
        for a in args:
            if not isinstance(a, torch.Tensor):
                self.entries.append(("scalar", a))
                continue
            src = a.contiguous()
            nbytes = src.numel() * src.element_size()
            span = nbytes + self.offsets[-1] * SOL_POOL_ALIGNMENT + SOL_POOL_ALIGNMENT
            pool = torch.empty(span, dtype=torch.uint8, device=src.device)
            self.entries.append(("tensor", (src, pool, nbytes)))
        self.i = 0

    def next(self) -> list:
        off = self.offsets[min(self.i, len(self.offsets) - 1)] * SOL_POOL_ALIGNMENT
        self.i += 1
        out = []
        for kind, e in self.entries:
            if kind == "scalar":
                out.append(e)
                continue
            src, pool, nbytes = e
            view = pool[off:off + nbytes].view(src.dtype).view(src.shape)
            view.copy_(src)
            out.append(view)
        return out


def _kernel_events(prof, include_memory: bool = False) -> list:
    """(name, start_ns, end_ns) of the device activities in a torch.profiler trace,
    ordered by start. Kernels always; memcpy / memset only with ``include_memory``
    (SOL-ExecBench times CONCURRENT_KERNEL + MEMCPY + MEMSET, FlashInfer-Bench's
    flashinfer timer CONCURRENT_KERNEL only)."""
    out = []
    try:
        for ev in prof.profiler.kineto_results.events():
            if ev.device_type() != torch.autograd.DeviceType.CUDA:
                continue
            name = ev.name()
            if not include_memory and name.startswith(("Memcpy", "Memset")):
                continue
            if hasattr(ev, "start_ns"):
                start, dur = ev.start_ns(), ev.duration_ns()
            else:
                start, dur = ev.start_us() * 1000, ev.duration_us() * 1000
            out.append((name, start, start + dur))
    except AttributeError:
        out = []
        for ev in prof.events():
            if ev.device_type != torch.autograd.DeviceType.CUDA:
                continue
            if not include_memory and ev.name.startswith(("Memcpy", "Memset")):
                continue
            out.append((ev.name, int(ev.time_range.start * 1e3), int(ev.time_range.end * 1e3)))
    out.sort(key=lambda k: (k[1], k[2]))
    return out


def _profile(fn, include_memory: bool = False):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return _kernel_events(prof, include_memory)


def _select_sequence(window: list, expected: list):
    """The call's activities inside one iteration's window: among the activities whose
    identity occurs in the discovered sequence, the first run equal to it
    (SOL-ExecBench select_activity_sequence; an unrelated launch in between — the
    flush, another stream — is skipped rather than breaking the match)."""
    n = len(expected)
    wanted = set(expected)
    cand = [k for k in window if k[0] in wanted]
    names = [k[0] for k in cand]
    for s in range(len(cand) - n + 1):
        if names[s:s + n] == expected:
            return cand[s:s + n]
    return None


def time_profiler_spans(call, setup, flush, *, warmup: int, iters: int,
                        include_memory: bool = False, sync_after_flush: bool = True) -> list:
    """Per-call GPU span from CUPTI activity (torch.profiler): first activity start ->
    last activity end of the call's own sequence (kernels; plus memcpy / memset with
    ``include_memory``); setup and the L2 flush run outside it. Per-call spans in ms."""
    for _ in range(warmup):
        args = setup()
        flush()
        call(args)
    torch.cuda.synchronize()
    args = setup()
    flush()
    torch.cuda.synchronize()
    expected = [k[0] for k in _profile(lambda: call(args), include_memory)]
    if not expected:
        raise RuntimeError("the call launched no kernel the profiler could see")

    def timed():
        for _ in range(iters):
            a = setup()
            flush()
            if sync_after_flush:
                torch.cuda.synchronize()
            call(a)
            torch.cuda.synchronize()

    kernels = _profile(timed, include_memory)
    # One iteration's activities all start after its own flush and before the next
    # iteration's setup; with a synchronize between iterations the call's sequence is
    # the LAST len(expected) activities of each setup/flush/call group.
    per = len(kernels) // iters if iters else 0
    spans = []
    if per * iters == len(kernels) and per >= len(expected):
        for i in range(iters):
            group = kernels[i * per:(i + 1) * per]
            seq = _select_sequence(group[per - len(expected):], expected) or \
                _select_sequence(group, expected)
            if seq is None:
                break
            spans.append((max(k[2] for k in seq) - min(k[1] for k in seq)) / 1e6)
    if len(spans) != iters:
        # uneven groups (a launch the discovery call did not make): scan in order
        spans, rest = [], kernels
        for _ in range(iters):
            seq = _select_sequence(rest, expected)
            if seq is None:
                raise RuntimeError(f"could not find the call's kernel sequence "
                                   f"{expected[:4]}... in iteration {len(spans)}")
            spans.append((max(k[2] for k in seq) - min(k[1] for k in seq)) / 1e6)
            rest = rest[rest.index(seq[-1]) + 1:]
    return spans


def _cupti_key(act, cupti) -> str:
    """An activity's identity across iterations (SOL-ExecBench kernel_string)."""
    kind = act.kind
    if kind == cupti.ActivityKind.CONCURRENT_KERNEL:
        return f"K:{act.name}"
    nbytes = getattr(act, "bytes", 0)
    if kind == cupti.ActivityKind.MEMCPY:
        return f"MEMCPY:{getattr(act, 'copy_kind', 0)}:{nbytes}"
    return f"MEMSET:{getattr(act, 'value', 0)}:{nbytes}"


@contextlib.contextmanager
def _cupti_collect(cupti, kinds, enable=None):
    """(key, start_ns, end_ns) of every activity of ``kinds`` recorded inside the
    block (SOL-ExecBench collect_cupti_activities, cupti-python). ``enable``: the
    activity kinds switched on while collecting (default ``kinds``) — a superset
    reproduces a harness that traces more than it keeps."""
    records: list = []

    def request():
        return 8 * 1024 * 1024, 0

    def complete(activities):
        for act in activities:
            if act.kind in kinds:
                records.append((_cupti_key(act, cupti), act.start, act.end))

    enabled = []
    try:
        for k in (enable or kinds):
            cupti.activity_enable(k)
            enabled.append(k)
        cupti.activity_register_callbacks(request, complete)
        yield records
    finally:
        if enabled:
            try:
                cupti.activity_flush_all(0)
            finally:
                for k in enabled:
                    cupti.activity_disable(k)
                cupti.finalize()


def time_cupti_spans(call, setup, flush, *, warmup: int, iters: int,
                     include_memory: bool = False, sync_after_flush: bool = False,
                     trace_api: bool = False) -> list:
    """Per-call GPU span read with cupti-python, the suites' own way: discover the
    call's activity sequence, then per timed call take the activities that start
    inside its CPU timestamp window, select that sequence, and read first start ->
    last end. The two harnesses differ in three places, each a flag:

      include_memory    SOL-ExecBench counts memcpy / memset into the span;
                        flashinfer counts kernels only.
      sync_after_flush  flashinfer synchronizes between the L2 flush and the timed
                        call; SOL-ExecBench does not, so the call queues behind the
                        flush and part of its launch gaps hides under it.
      trace_api         flashinfer also traces RUNTIME / DRIVER / MEMCPY / MEMSET
                        activity while timing (it keeps only the kernels); API
                        tracing costs host time per launch, which a multi-kernel
                        call pays in its gaps.

    A call that is host-bound (many small launches) reads unstably under ANY span
    clock — measured on B200, the same PyTorch reference read 125 and 207 us in one
    mirrored pair through SOL-ExecBench's own timer — so its ratio is not a kernel
    fact; a GPU-bound call reads the same under every clock. Per-call spans in ms."""
    from cupti import cupti
    kinds = [cupti.ActivityKind.CONCURRENT_KERNEL]
    if include_memory:
        kinds += [cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET]
    enable = list(kinds)
    if trace_api:
        enable = [cupti.ActivityKind.RUNTIME, cupti.ActivityKind.CONCURRENT_KERNEL,
                  cupti.ActivityKind.DRIVER, cupti.ActivityKind.MEMCPY,
                  cupti.ActivityKind.MEMSET]
    for _ in range(warmup):
        args = setup()
        flush()
        call(args)
    torch.cuda.synchronize()
    args = setup()
    flush()
    torch.cuda.synchronize()
    with _cupti_collect(cupti, kinds) as found:
        call(args)
        torch.cuda.synchronize()
    expected = [r[0] for r in sorted(found, key=lambda r: (r[1], r[2]))]
    if not expected:
        raise RuntimeError("the call launched no activity CUPTI could see")
    stamps = []
    with _cupti_collect(cupti, kinds, enable) as found:
        torch.cuda.synchronize()
        for _ in range(iters):
            a = setup()
            flush()
            if sync_after_flush:
                torch.cuda.synchronize()
            t0 = cupti.get_timestamp()
            call(a)
            torch.cuda.synchronize()
            stamps.append((t0, cupti.get_timestamp()))
        torch.cuda.synchronize()
    acts = sorted(found, key=lambda r: (r[1], r[2]))
    starts = [r[1] for r in acts]
    spans = []
    for i, (t0, t1) in enumerate(stamps):
        window = acts[bisect.bisect_left(starts, t0):bisect.bisect_right(starts, t1)]
        seq = _select_sequence(window, expected)
        if seq is None:
            raise RuntimeError(f"the call's activity sequence {expected[:4]}... is not in "
                               f"timed call {i}'s window")
        spans.append((max(r[2] for r in seq) - min(r[1] for r in seq)) / 1e6)
    return spans


def time_cold_events(call, setup, flush, *, warmup: int, iters: int) -> list:
    """CUDA events around each call, L2 flushed before it (the suites' fallback)."""
    for _ in range(warmup):
        args = setup()
        flush()
        call(args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        args = setup()
        flush()
        torch.cuda.synchronize()
        starts[i].record()
        call(args)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _cupti_python() -> bool:
    return importlib.util.find_spec("cupti") is not None


_UPSTREAM_PARAMS = {
    SUITE_FIB: ("fn", "dry_run_iters", "repeat_iters", "input_args", "cold_l2_cache",
                "use_cuda_graph"),
    SUITE_SOL: ("fn", "inputs", "outputs", "device", "warmup", "rep", "seed"),
}


def _upstream_timer(suite: str):
    """The suite's own timing function, when its package is in the image AND takes the
    arguments the suite's harness passes (an older flashinfer without ``cold_l2_cache``
    would time a warm L2 — not the board's number, so it does not qualify)."""
    try:
        if suite == SUITE_FIB:
            from flashinfer.testing import bench_gpu_time_with_cupti as fn
        else:
            from sol_execbench.core.bench.timing import time_runnable as fn
        params = inspect.signature(fn).parameters
    except Exception:  # noqa: BLE001 — absent or broken: fall through to our own
        return None
    if any(p not in params for p in _UPSTREAM_PARAMS[suite]):
        return None
    return f"{fn.__module__}.{fn.__name__}"


def choose_timer(suite: str) -> tuple[str, str]:
    """(timer, basis): the first available of TIMERS; FK_LB_TIMER forces one."""
    forced = os.environ.get("FK_LB_TIMER", "").strip()
    if forced and forced not in TIMERS:
        raise ValueError(f"FK_LB_TIMER must be one of {TIMERS}, got {forced!r}")
    if forced in ("", "upstream") and _upstream_timer(suite) and _cupti_python():
        return "upstream", BASIS_CUPTI
    if forced == "upstream":
        raise RuntimeError(f"FK_LB_TIMER=upstream but the {suite} timer (or cupti-python) "
                           f"is not importable in this image")
    if forced in ("", "cupti") and _cupti_python():
        return "cupti", BASIS_CUPTI
    if forced == "cupti":
        raise RuntimeError("FK_LB_TIMER=cupti but cupti-python is not importable")
    if forced in ("", "profiler"):
        return "profiler", BASIS_CUPTI
    return "events", BASIS_COLD_EVENTS


# Printed and carried in the eval JSON whenever the profiler clock produced the gate
# reading. On B200 its single-kernel spans matched the boards' own timers; what it
# cannot reproduce is the boards' collector (activity kinds traced, per-op callbacks
# riding along), which only moves multi-kernel calls — and those are unstable under
# every span clock anyway (time_cupti_spans).
PROFILER_NOTE = ("torch.profiler clock: not the board's CUPTI collector — single-kernel "
                 "calls read the same, multi-kernel calls may not; install cupti-python")


def _time_once(suite: str, timer: str, fn, inputs: list, outputs: list, device, *,
               warmup: int, iters: int, seed: int) -> float:
    """Median per-call ms of ``fn`` on one input set, the suite's way."""
    args = list(inputs) + list(outputs)
    if timer == "upstream":
        if suite == SUITE_FIB:
            from flashinfer.testing import bench_gpu_time_with_cupti
            times = bench_gpu_time_with_cupti(fn=fn, dry_run_iters=warmup,
                                              repeat_iters=iters, input_args=tuple(args),
                                              cold_l2_cache=True, use_cuda_graph=False)
            return statistics.median(times)
        from sol_execbench.core.bench.timing import time_runnable
        return float(time_runnable(fn, list(inputs), list(outputs), device, warmup=warmup,
                                   rep=iters, seed=seed))
    flush_buf = _flush_buffer(device)

    def flush():
        _reset_persisting_l2()
        flush_buf.zero_()

    if suite == SUITE_SOL:
        pool = _Shifted(args, warmup + iters + 1, seed)
        setup = pool.next
    else:
        def setup():
            return args

    def call(a):
        fn(*a)

    if timer == "cupti":
        spans = time_cupti_spans(call, setup, flush, warmup=warmup, iters=iters,
                                 include_memory=(suite == SUITE_SOL),
                                 sync_after_flush=(suite == SUITE_FIB),
                                 trace_api=(suite == SUITE_FIB))
    elif timer == "profiler":
        spans = time_profiler_spans(call, setup, flush, warmup=warmup, iters=iters,
                                    include_memory=(suite == SUITE_SOL),
                                    sync_after_flush=(suite == SUITE_FIB))
    else:
        spans = time_cold_events(call, setup, flush, warmup=warmup, iters=iters)
    return statistics.median(spans)


def _protocol(problem: dict) -> dict:
    if problem["suite"] == SUITE_FIB:
        cfg = fib_config(problem)
        return {"warmup": int(cfg["warmup_runs"]), "iters": int(cfg["iterations"]),
                "aggregate": "mean", "seed": 0}
    return {"warmup": SOL_WARMUP, "iters": SOL_ITERS, "aggregate": "single", "seed": SOL_SEED}


def measure(problem: dict, sets: list, kernel, kernel_dps: bool, ref=None,
            ref_dps: bool = False, device="cuda", burn_in_s: float = BURN_IN_S) -> dict:
    """The leaderboard reading of the kernel (and the in-job reference) on ``sets``:
    per set, after one burn-in, ref / kernel / kernel / ref, each side the mean of its
    two readings; the suite's aggregate over sets (FlashInfer-Bench: mean; SOL: the one
    set it times)."""
    suite = problem["suite"]
    timer, basis = choose_timer(suite)
    proto = _protocol(problem)

    def outs_for(dps):
        return allocate_outputs(problem, device) if dps else []

    def reading(fn, dps, inp):
        return _time_once(suite, timer, fn, inp, outs_for(dps), device,
                          warmup=proto["warmup"], iters=proto["iters"], seed=proto["seed"])

    first = sets[0]
    kout = outs_for(kernel_dps)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < burn_in_s:
        for _ in range(20):
            kernel(*first, *kout)
        torch.cuda.synchronize()
    k_sets, r_sets, ref_error = [], [], None
    for inp in sets:
        r1 = r2 = None
        if ref is not None and ref_error is None:
            try:
                r1 = reading(ref, ref_dps, inp)
            except Exception as exc:  # noqa: BLE001 — the reference is foreign code
                ref_error = f"{type(exc).__name__}: {exc}"
        k1 = reading(kernel, kernel_dps, inp)
        k2 = reading(kernel, kernel_dps, inp)
        if r1 is not None and ref_error is None:
            try:
                r2 = reading(ref, ref_dps, inp)
            except Exception as exc:  # noqa: BLE001
                ref_error = f"{type(exc).__name__}: {exc}"
        k_sets.append([k1, k2])
        if r1 is not None and r2 is not None:
            r_sets.append([r1, r2])

    def agg(pairs):
        per = [sum(p) / 2.0 for p in pairs]
        return (statistics.mean(per) if proto["aggregate"] == "mean" else per[-1]), per

    k_ms, k_per = agg(k_sets)
    out = {"suite": suite, "timer": timer, "basis": basis, "protocol": proto,
           "kernel_ms": k_ms, "kernel_per_set_ms": k_per, "kernel_readings_ms": k_sets,
           "ref_ms": None, "ref_per_set_ms": [], "ref_readings_ms": [],
           "ref_error": ref_error}
    if ref is not None and ref_error is None and len(r_sets) == len(sets):
        r_ms, r_per = agg(r_sets)
        out.update(ref_ms=r_ms, ref_per_set_ms=r_per, ref_readings_ms=r_sets)
    elif ref is not None and ref_error is None:
        out["ref_error"] = "reference produced no reading"
    out["unstable"] = {"kernel": unstable_sets(k_sets),
                       "ref": unstable_sets(out["ref_readings_ms"])}
    return out


# Two mirrored readings of the SAME callable on the SAME inputs further apart than this
# are not a kernel fact. It is how a host-bound call (many small launches) shows up
# under a per-call span clock: whether its launch gaps hide behind the L2 flush is a
# race, and whole readings flip — on B200 one PyTorch reference read 125 and 207 us in
# a pair through SOL-ExecBench's own timer, and 44 vs 111 us through the cupti clock,
# while single-kernel calls repeat to ~0.1 us.
UNSTABLE_PAIR_RATIO = 1.2


def unstable_sets(pairs: list) -> list:
    """Indices of the input sets whose two mirrored readings disagree past
    UNSTABLE_PAIR_RATIO."""
    return [i for i, p in enumerate(pairs)
            if len(p) == 2 and min(p) > 0 and max(p) / min(p) > UNSTABLE_PAIR_RATIO]


# --- the eval JSON ---------------------------------------------------------------------

def _round(v, n=4):
    return None if v is None else round(float(v), n)


def sol_score(t_k: float, t_b: float, t_sol: float) -> float:
    """SOL-ExecBench sol_score.py: S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)) — 0.5 at
    T_b, 1 at T_SOL (above 1 past it; the board does not clamp)."""
    gap = t_b - t_sol
    if gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / gap)


# What gpu_entry.sh exports about the clock lock; its first word is "locked" only when
# the lock it asked for was read back.
CLOCK_ENV = "LB_CLOCK_STATE"


def clock_state() -> str:
    return os.environ.get(CLOCK_ENV) or "unknown (not launched through gpu_entry.sh)"


def board_block(problem: dict, kernel_ms: float, clock: str) -> dict | None:
    """The SOL score of ``kernel_ms`` against the board's frozen T_b / T_SOL, or None
    when the card carries none. Carries a warning when the reading was not taken
    under the board's lock: T_b / T_SOL are readings at that lock, and an unlocked
    kernel reads faster than the board would time it."""
    board = problem.get("sol_board")
    if not board or not kernel_ms:
        return None
    t_b, t_sol = float(board["t_b_ms"]), float(board["t_sol_ms"])
    out = {"t_b_ms": t_b, "t_sol_ms": t_sol,
           "sol_score": _round(sol_score(kernel_ms, t_b, t_sol)),
           "vs_t_b": _round(t_b / kernel_ms), "kernel_id": board.get("kernel_id"),
           "workload_index": board.get("workload_index")}
    if clock.split(" ", 1)[0] != "locked":
        mhz = board.get("clock_mhz") or {}
        out["clock_warning"] = (f"T_b / T_SOL are the board's readings at SM "
                                f"{mhz.get('sm_mhz', '?')} / memory {mhz.get('mem_mhz', '?')} "
                                f"MHz; this reading was taken with the clock {clock} — the "
                                f"score compares two clocks")
    return out


def _side_windows(fn, dps: bool, problem: dict, inp: list, device) -> dict:
    """timing.py's e2e window and host-enqueue clock on one set (log / audit only)."""
    import timing
    outs = allocate_outputs(problem, device) if dps else []

    def call():
        fn(*inp, *outs)

    e2e = timing.time_e2e(call)
    return {"median_ms": statistics.median(e2e), "min_ms": min(e2e),
            "host_enqueue_ms": timing.time_host_enqueue(call)}


def performance_block(problem: dict, m: dict, kwin: dict, rwin: dict | None) -> dict:
    perf = {
        "bwd_median_ms": _round(kwin["median_ms"]),
        "min_ms": _round(kwin["min_ms"]),
        "device_median_ms": _round(m["kernel_ms"]),
        "device_window_basis": m["basis"],
        "host_enqueue_ms": _round(kwin["host_enqueue_ms"]),
        "host_bound_ratio": (round(kwin["host_enqueue_ms"] / m["kernel_ms"], 3)
                             if m["basis"] != BASIS_CUPTI and m["kernel_ms"] else None),
        "in_job_ref": ("paired" if m["ref_ms"] is not None
                       else "not timed (problem.json time_reference)"
                       if not problem.get("time_reference", True)
                       else (m["ref_error"] or "absent")),
    }
    if m["ref_ms"] is not None and rwin is not None:
        perf.update({
            "ref_median_ms": _round(rwin["median_ms"]),
            "vs_ref": _round(rwin["median_ms"] / kwin["median_ms"]),
            "ref_device_ms": _round(m["ref_ms"]),
            "vs_ref_device": _round(m["ref_ms"] / m["kernel_ms"]),
            "ref_host_enqueue_ms": _round(rwin["host_enqueue_ms"]),
            "ref_host_bound_ratio": (round(rwin["host_enqueue_ms"] / m["ref_ms"], 3)
                                     if m["basis"] != BASIS_CUPTI and m["ref_ms"] else None),
        })
    perf["leaderboard"] = {
        "suite": m["suite"], "upstream": UPSTREAM[m["suite"]], "timer": m["timer"],
        "protocol": m["protocol"],
        "reading": ("mean over input sets of the per-set median per-call span"
                    if m["protocol"]["aggregate"] == "mean"
                    else "median per-call span on the last correctness round's inputs"),
        "kernel_per_set_ms": [_round(v) for v in m["kernel_per_set_ms"]],
        "ref_per_set_ms": [_round(v) for v in m["ref_per_set_ms"]],
        "definition": problem["definition"]["name"],
        "workload": problem["workload"].get("uuid"),
        "reference": (problem.get("reference_impl") or {}).get("name", "definition reference"),
        "clock": clock_state(),
    }
    board = board_block(problem, m["kernel_ms"], perf["leaderboard"]["clock"])
    if board is not None:
        perf["leaderboard"]["sol_board"] = board
    if m["timer"] == "profiler":
        perf["leaderboard"]["timer_note"] = PROFILER_NOTE
    unstable = {side: sets for side, sets in (m.get("unstable") or {}).items() if sets}
    if unstable:
        perf["leaderboard"]["unstable_sets"] = unstable
        perf["leaderboard"]["unstable_note"] = (
            f"mirrored readings of one side differ by more than {UNSTABLE_PAIR_RATIO}x on "
            f"set(s) {unstable} — a host-bound call under a per-call span clock; the "
            f"reading is not a kernel fact")
    return perf


# A group card timed on ONE of its workloads: set by the seeded meta tools that time many
# variants in one job (lever pricing, the timeline probe, stall / SASS pricing), where the
# whole group x rounds x variants outlives the job clock -- ``representative`` (the first
# `group` entry) or a group entry's name (``wl5``: the tools read each of problem.json
# ``meta_workloads`` in turn, one job or one run each). The eval JSON then carries
# ``workload_scope`` and the loop's gate parser refuses it: a meta reading, never a gate.
# ncu_window honours it too, so a tool's NCU reads the workload its bench timed.
WORKLOAD_SCOPE_ENV = "FK_LB_WORKLOAD_SCOPE"
_SCOPE: dict = {}
_WHOLE_SCOPES = ("", "all")


def apply_workload_scope(problem: dict, env=None) -> dict | None:
    """Narrow a group card to one workload when ``FK_LB_WORKLOAD_SCOPE`` names one --
    ``representative`` (the first ``group`` entry) or an entry's name; returns the scope
    record (None: the card is timed whole -- unset / ``all``, or a one-workload card). A name
    the group does not have raises: a typo must not time the whole group."""
    raw = ((os.environ if env is None else env).get(WORKLOAD_SCOPE_ENV) or "").strip()
    group = problem.get("group") or []
    if raw.lower() in _WHOLE_SCOPES or len(group) <= 1:
        return None
    # a correctness-only workload may be read too (it is timed alone, as a meta reading)
    every = group + list(problem.get("check_only") or [])
    names = [str(g.get("name") or f"wl{g.get('index')}") for g in every]
    if raw.lower() == "representative":
        pick = 0
    elif raw in names:
        pick = names.index(raw)
    else:
        raise ValueError(f"{WORKLOAD_SCOPE_ENV}={raw!r} names no workload of this group card "
                         f"({', '.join(names)}; or 'representative')")
    one = every[pick]
    problem["group"] = [one]
    problem.pop("check_only", None)
    problem["workload"] = one["workload"]
    problem.pop("sol_board", None)
    if one.get("sol_board"):
        problem["sol_board"] = one["sol_board"]
    return {"workloads": "representative" if raw.lower() == "representative" else names[pick],
            "rep": names[pick], "of": len(group)}


def _emit(result: dict) -> None:
    if _SCOPE:
        result = {**result, "workload_scope": dict(_SCOPE)}
    print("\n=== JSON ===")
    print(json.dumps(result, indent=2, allow_nan=True))


def precision_on(problem: dict) -> bool:
    """Whether this card measures the precision record: only when problem.json asks for
    it (``precision_contract`` = "strict" / "official"). The default is off -- a
    leaderboard card judges exactly as its board does: no float64 re-run, no stressed
    inputs, no ``precision`` block. ``FK_PRECISION=0`` counts only as read at import
    (``PRECISION_ENV_OFF``): the kernel module, imported later, cannot switch it off."""
    return (str(problem.get("precision_contract") or "off").strip().lower()
            in ("strict", "official")
            and not PRECISION_ENV_OFF)


def _suite_judge(problem: dict):
    """The suite's correctness protocol: (problem, impl, dps, device, prec=None) ->
    (verdict, the input sets it judged)."""
    return {SUITE_FIB: fib_correctness, SUITE_SOL: sol_correctness,
            SUITE_TASK: task_correctness}[problem["suite"]]


def _correctness(problem: dict, impl, dps: bool, device):
    """The suite's correctness protocol. On a card whose problem.json turns the precision
    contract on (``precision_on``), a timed workload also gets the precision record
    (eval/precision.py) on the same input sets plus the stress profiles -- attached to
    the verdict as ``precision``, never changing ``pass``. Not on a correctness-only
    workload, nor on the reference timed alone (``_no_precision``)."""
    if not precision_on(problem) or problem.get("_no_precision") or problem.get("_check_only"):
        return _suite_judge(problem)(
            problem, impl, dps, device)
    P = _precision()
    if P is None:
        out = _suite_judge(problem)(
            problem, impl, dps, device)
        out[0]["precision"] = {"status": "unavailable",
                               "why": "eval/precision.py is missing (the card predates the "
                                      "precision gate: copy it from src/forgekernel/card_template/eval/)"}
        return out
    prec = P.PrecisionRecorder()
    verdict, sets = _suite_judge(problem)(
        problem, impl, dps, device, prec=prec)
    if verdict.get("pass"):
        _precision_stress(problem, impl, dps, device, prec)
    verdict["precision"] = prec.block()
    return verdict, sets


# The precision record (eval/precision.py): the first PRECISION_SETS input sets the suite
# judges, then one input set per stress profile (seeded, the RNG state restored around
# it, so the suite's own draws -- SOL's seed-200 sequence -- never move).
PRECISION_SETS = 3
PRECISION_STRESS_SEED = 7100
PRECISION_ENV = "FK_PRECISION"
# Read once, at import -- before any kernel module runs in this process.
PRECISION_ENV_OFF = os.environ.get(PRECISION_ENV, "").strip().lower() in ("0", "off")


def _precision_snapshot(prec, inp: list):
    """A host copy of ``inp`` taken before the reference or the kernel touches it (None
    when nothing is measured): the float64 truth runs on the inputs as drawn, not on what
    an in-place definition (a residual or cache written back) left in them."""
    return _precision().snapshot(inp) if prec is not None else None


def _precision_measure(problem: dict, prec, case: str, outs: list, refs: list, ref_run,
                       pristine, device, stress: str | None = None) -> None:
    P = _precision()
    names = list(problem["definition"]["outputs"])
    prec.measure(case, outs, refs,
                 lambda: P.truth_of(ref_run, P.restore(pristine),
                                    normalize=lambda r: normalize_outputs(problem, r, device)),
                 names=names, stress=stress)


def _precision_stress(problem: dict, impl, dps: bool, device, prec) -> None:
    P = _precision()
    ref_run, custom = reference_callables(problem)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    for j, profile in enumerate(P.STRESS_PROFILES):
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(PRECISION_STRESS_SEED + j)
            touched: list = []
            try:
                inp = gen_inputs(problem, device, custom_inputs_fn=custom, stress=profile,
                                 touched=touched)
            except Exception as e:  # noqa: BLE001
                prec.unavailable("stress", f"inputs: {type(e).__name__}: {e}", stress=profile)
                continue
            if not touched:
                prec.unavailable("stress", "no plain random float input to stress",
                                 stress=profile)
                continue
            pristine = _precision_snapshot(prec, inp)
            try:
                refs = call_impl(problem, ref_run, inp, False, device, no_grad=True)
            except Exception as e:  # noqa: BLE001
                prec.unavailable("stress", f"the reference raised: {type(e).__name__}: {e}",
                                 stress=profile)
                continue
            try:
                outs = call_impl(problem, impl, inp, dps, device, no_grad=True)
            except Exception as e:  # noqa: BLE001 — the solution is foreign code
                prec.unavailable("stress", f"the kernel raised on the stressed input: "
                                           f"{type(e).__name__}: {str(e)[:200]}", stress=profile)
                continue
            _precision_measure(problem, prec, "stress", outs, refs, ref_run, pristine, device,
                               stress=profile)


def _correctness_json(verdict: dict) -> dict:
    out = {k: v for k, v in verdict.items() if k not in ("checks", "precision")}
    out["max_diff"] = verdict.get("max_abs")
    out["checks"] = verdict.get("checks", [])[:40]
    if verdict.get("pass"):
        out.pop("error", None)
    return out


def judge_check_only(problem: dict, impl, dps: bool, device) -> tuple[list, tuple | None]:
    """The suite's correctness protocol on the card's correctness-only workloads, in turn,
    stopping at the first failure: ([(view, verdict), ...], (view, verdict) of that failure
    or None). [] / None on a card without them."""
    per, first_bad = [], None
    for v in check_views(problem):
        verdict, _ = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        print(f"[leaderboard] {v['_wl_name']} {json.dumps(v['workload'].get('axes'))} "
              f"(correctness only): {verdict['status']}"
              + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
    return per, first_bad


def judge_workloads(problem: dict, impl, dps: bool, device) -> tuple[dict, list]:
    """The suite's correctness protocol on every workload of the card (its correctness-only
    ones first, then a group card's in turn, stopping at the first failure); (verdict,
    [(view, its timed sets), ...] -- the timed workloads only). A one-workload card without
    correctness-only workloads: ``_correctness`` itself."""
    views = workload_views(problem)
    checks = check_views(problem)
    if len(views) == 1 and not checks:
        verdict, sets = _correctness(problem, impl, dps, device)
        return verdict, [(problem, sets)]
    per, runs = [], []
    cper, first_bad = judge_check_only(problem, impl, dps, device)
    if first_bad is not None:
        return group_verdict(cper, len(views) + len(checks), first_bad), runs
    if len(views) == 1:
        verdict, sets = _correctness(problem, impl, dps, device)
        v = dict(problem, _wl_name=_view_name(problem))
        out = group_verdict(cper + [(v, verdict)], 1 + len(checks),
                            None if verdict["pass"] else (v, verdict))
        return out, [(problem, sets)]
    per = list(cper)
    for v in views:
        verdict, sets = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        runs.append((v, sets))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
    return group_verdict(per, len(views) + len(checks), first_bad), runs


def _view_name(problem: dict) -> str:
    """A one-workload card's own workload, named for the verdict rows."""
    g = (problem.get("group") or [None])[0]
    return str(g.get("name")) if g else str(problem["workload"].get("uuid"))


def group_verdict(per: list, n_views: int, first_bad=None) -> dict:
    """One verdict over a group's per-workload verdicts [(view, verdict), ...]: the first
    failure's status and error (prefixed with its workload), the worst error stats, and a
    compact per-workload list."""
    out = {"pass": first_bad is None and len(per) == n_views, "status": "PASSED",
           "max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0, "checks": [],
           "n_workloads": n_views, "workloads": []}
    for v, verdict in per:
        name = v["_wl_name"]
        out["workloads"].append({"name": name, "status": verdict.get("status"),
                                 "max_abs": verdict.get("max_abs"),
                                 "matched_ratio": verdict.get("matched_ratio"),
                                 **({"check_only": True} if v.get("_check_only") else {})})
        for k in ("max_abs", "max_rel"):
            x = verdict.get(k)
            if isinstance(x, (int, float)) and not math.isnan(x):
                out[k] = max(out[k], float(x))
        if isinstance(verdict.get("matched_ratio"), (int, float)):
            out["matched_ratio"] = min(out["matched_ratio"], verdict["matched_ratio"])
        out["checks"] += [{"workload": name, **c} for c in verdict.get("checks", [])
                          if isinstance(c, dict)]
        for k in ("tolerance", "seed"):
            out.setdefault(k, verdict.get(k))
    if (prec := precision_result(per)) is not None:
        out["precision"] = prec
    if first_bad is not None:
        v, verdict = first_bad
        axes = json.dumps(v["workload"].get("axes"))
        out["status"] = verdict.get("status")
        out["error"] = (f"workload {v['_wl_name']} {axes}"
                        + (" (correctness-only: scored outside this card, never timed here)"
                           if v.get("_check_only") else "")
                        + f": {verdict.get('error')}")
        if len(per) < n_views:
            out["error"] += f" ({n_views - len(per)} later workload(s) not run)"
    return out


def precision_result(per: list) -> dict | None:
    """The eval JSON's top-level ``precision`` block over [(view, verdict), ...]: the
    timed workloads' records (a group card's merged, rows named by workload)."""
    named = [(v.get("_wl_name") or _view_name(v), x.get("precision")) for v, x in per
             if isinstance(x.get("precision"), dict) and not v.get("_check_only")]
    if not named:
        return None
    P = _precision()
    if len(named) == 1 or P is None or any(b.get("schema") is None for _, b in named):
        return named[0][1]
    return P.merge_blocks(named)


def _say_precision(block: dict | None) -> None:
    if not block:
        return
    w = block.get("worst") or {}
    worst = (f"; worst {w.get('case')}{' [' + w['stress'] + ']' if w.get('stress') else ''} "
             f"{w.get('output')}: e_cand {w.get('e_cand')} vs e_ref {w.get('e_ref')} "
             f"(limit {w.get('limit')})") if w else ""
    status = str(block.get("status", "?")).upper()
    print(f"[leaderboard] precision: {status} ({block.get('n_checks', 0)} checks, truth "
          f"{block.get('truth')}){worst}"
          + (f" — {block.get('why')}" if block.get("why") and status != "PASS" else "")
          + (" (recorded only: [loop] precision_contract decides what it costs)"
             if status == "FAIL" else ""))


def _geomean(values) -> float:
    vals = [float(x) for x in values]
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


# Gate keys a group card reports as the geomean over its workloads (host-bound ratios as
# the worst workload's, so the precondition audit is never softened by averaging).
_GEO_KEYS = ("bwd_median_ms", "min_ms", "device_median_ms", "host_enqueue_ms")
_REF_GEO_KEYS = ("ref_median_ms", "ref_device_ms", "ref_host_enqueue_ms")


def group_performance_block(problem: dict, blocks: list) -> dict:
    """A group card's ``performance``: the per-workload ``performance_block``s (each with
    its ``name``) under ``workloads``, and the gate keys at the top level as the geometric
    mean over them (timing.workloads_performance's contract). ``vs_ref_device`` = geomean
    reference / geomean kernel = the geomean of the per-workload ratios."""
    out: dict = {}
    for k in _GEO_KEYS:
        vals = [b.get(k) for b in blocks]
        out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                               for x in vals) else None
    bases = list(dict.fromkeys(b.get("device_window_basis") for b in blocks))
    out["device_window_basis"] = bases[0] if len(bases) == 1 else "mixed: " + ", ".join(map(str, bases))
    hb = [b["host_bound_ratio"] for b in blocks if b.get("host_bound_ratio") is not None]
    out["host_bound_ratio"] = max(hb) if hb else None
    paired = [b for b in blocks if b.get("ref_device_ms")]
    if len(paired) == len(blocks):
        out["in_job_ref"] = "paired"
        for k in _REF_GEO_KEYS:
            vals = [b.get(k) for b in blocks]
            out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                                   for x in vals) else None
        # the geomean of the per-workload ratios (each from unrounded readings): dividing
        # the two rounded geomeans would move a us-scale ratio by tenths of a percent
        for k in ("vs_ref", "vs_ref_device"):
            vals = [b.get(k) for b in blocks]
            out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                                   for x in vals) else None
        rhb = [b["ref_host_bound_ratio"] for b in blocks if b.get("ref_host_bound_ratio") is not None]
        out["ref_host_bound_ratio"] = max(rhb) if rhb else None
    else:
        missing = [b["name"] for b in blocks if not b.get("ref_device_ms")]
        out["in_job_ref"] = (blocks[0].get("in_job_ref") if len(missing) == len(blocks)
                             else f"partial: no reference reading on {', '.join(missing)}")
    out["aggregate"] = "geomean"
    out["n_workloads"] = len(blocks)
    lbs = [b["leaderboard"] for b in blocks]
    lb = {k: lbs[0][k] for k in ("suite", "upstream", "timer", "protocol", "definition",
                                 "reference", "clock") if k in lbs[0]}
    lb["reading"] = (f"geometric mean over {len(blocks)} workloads of: {lbs[0].get('reading')}")
    lb["workloads"] = [b["name"] for b in blocks]
    boards = [x.get("sol_board") for x in lbs]
    if all(boards):
        lb["sol_board"] = {
            "sol_score_mean": _round(statistics.mean(x["sol_score"] for x in boards)),
            "sol_score": {b["name"]: x["sol_score"] for b, x in zip(blocks, boards)},
            "vs_t_b_geomean": _round(_geomean(x["vs_t_b"] for x in boards)),
            "kernel_id": boards[0].get("kernel_id"),
            "workload_index": [x.get("workload_index") for x in boards]}
        warns = [x["clock_warning"] for x in boards if x.get("clock_warning")]
        if warns:
            lb["sol_board"]["clock_warning"] = warns[0]
    notes = {b["name"]: b["leaderboard"]["unstable_sets"] for b in blocks
             if b["leaderboard"].get("unstable_sets")}
    if notes:
        lb["unstable_sets"] = notes
        lb["unstable_note"] = (f"mirrored readings of one side differ by more than "
                               f"{UNSTABLE_PAIR_RATIO}x on {notes} — a host-bound call under a "
                               f"per-call span clock; those readings are not kernel facts")
    if lbs[0].get("timer_note"):
        lb["timer_note"] = lbs[0]["timer_note"]
    out["leaderboard"] = lb
    out["workloads"] = blocks
    return out


# --- the forgekernel task suite ------------------------------------------------------
#
# A task has no board. Its problem.json (written by `forge run`) carries:
#
#   definition   the signature (inputs / outputs: shape over the axes, dtype) and, as
#                ``reference``, the JUDGE: the PyTorch semantics the teacher wrote from the
#                task's description, with ``make_inputs`` (custom_inputs_entrypoint)
#   workload / group
#                the task's shapes (a group card when there are several); each workload's
#                ``input_groups``: {"name", "kind": "file", "path"} -- every input from a
#                safetensors file -- or {"name", "kind": "generated", "seed", "intent"} --
#                make_inputs(axes, group=<name>, seed=<seed>, device=<device>)
#   tolerance    per output: TASK_TOL_MARGIN x the largest normalized error the reference
#                showed against the judge on every input group (``--mode calibrate``),
#                floored per dtype
#   timing       "device" (timing.py's pipelined device window) or "e2e" (per call)
#   reference_impl
#                the task's reference (sealed: decoded in memory, timed beside the kernel)
#
# Correctness: every input group of every shape, the judge on a copy of the inputs, then the
# implementation; per output, shape and dtype must match, non-finite values must sit where
# the judge has the same ones, and the normalized error max|y - j| / max|j| (finite
# positions) must not exceed the output's tolerance. Integer and bool outputs match exactly.
# Timing: per shape, timing.measure_windows_workloads over its input groups (burn-in, then
# reference / kernel / kernel / reference on each), the geomean over the groups; across
# shapes, the geomean over the shapes. Every input group of every shape takes part.

# Two units of the output dtype's rounding, relative to the output's scale: the tolerance
# never falls below it, so a reference that matches the judge bit for bit does not demand
# the same of a kernel that only reorders a sum.
TASK_FLOOR = {"float64": 2 * 2.0 ** -53, "float32": 2 * 2.0 ** -24, "float16": 2 * 2.0 ** -11,
              "bfloat16": 2 * 2.0 ** -8, "float8_e4m3fn": 2 * 2.0 ** -4,
              "float8_e5m2": 2 * 2.0 ** -3}
# The tolerance is this many times the reference's largest error. The reference's error is
# one draw from its own summation order on the card's inputs: another correct kernel sums
# in another order and lands as often above that draw as below it, and on a fresh seed the
# reference itself can land above it. Twice the error absorbs that spread and stays far
# below what a lower-precision path costs.
TASK_TOL_MARGIN = 2.0
# A reference further than this from the judge on any output does not compute what the
# judge computes: calibration reports no match instead of turning the gap into a tolerance.
TASK_MATCH_CAP = 0.05
TASK_EXACT_DTYPES = ("int64", "int32", "int16", "int8", "bool")
BASIS_PLAIN = "plain_batch"


def task_groups(problem: dict) -> list[dict]:
    return list(problem["workload"].get("input_groups") or [])


def first_inputs(problem: dict, device) -> list:
    """One input set of the card's first workload, drawn the suite's way (a task: its first
    shape's first input group)."""
    if problem["suite"] == SUITE_TASK:
        view = _task_shapes(problem)[0]
        return task_inputs(view, task_groups(view)[0], device)
    _, custom = reference_callables(problem)
    return gen_inputs(problem, device, custom_inputs_fn=custom)


def _task_custom(problem: dict):
    defn = problem["definition"]
    ns = _exec_source(defn["reference"], f"<{defn['name']} judge>")
    return ns["run"], ns[defn.get("custom_inputs_entrypoint") or "make_inputs"]


def task_inputs(problem: dict, group: dict, device, make_inputs=None) -> list:
    """One input group of the problem's workload, in definition order, checked against
    the signature (a wrong shape or dtype is the judge's or the file's fault, raised as
    ValueError -- never charged to the kernel)."""
    defn = problem["definition"]
    axes = resolved_axes(defn, problem["workload"].get("axes") or {})
    shapes = tensor_shapes(defn["inputs"], axes)
    dev = torch.device(device)
    if group.get("kind") == "file":
        path = Path(group["path"])
        if not path.is_absolute():
            path = Path(problem["_dir"]) / path
        got = read_safetensors(path.resolve())
    else:
        if make_inputs is None:
            make_inputs = _task_custom(problem)[1]
        got = make_inputs(dict(axes), group=group["name"], seed=int(group.get("seed", 0)),
                          device=dev)
        if not isinstance(got, dict):
            raise ValueError(f"make_inputs returned {type(got).__name__}, not a dict of inputs")
    out = []
    for name, spec in defn["inputs"].items():
        if name not in got:
            raise ValueError(f"input group {group['name']!r}: no input {name!r}")
        v = got[name]
        if shapes[name] is None:
            out.append(v.item() if isinstance(v, torch.Tensor) else v)
            continue
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"input group {group['name']!r}: {name} is "
                             f"{type(v).__name__}, not a tensor")
        want = torch_dtype(spec["dtype"])
        if tuple(v.shape) != tuple(shapes[name]) or v.dtype != want:
            raise ValueError(f"input group {group['name']!r}: {name} is {tuple(v.shape)} "
                             f"{v.dtype}, the signature says {tuple(shapes[name])} {want}")
        out.append(v.to(dev).contiguous())
    return out


_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
              "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
              "F8_E5M2": torch.float8_e5m2, "I64": torch.int64, "I32": torch.int32,
              "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def read_safetensors(path: Path) -> dict:
    """name -> tensor (CPU) of a safetensors file, read with torch alone: an image without
    the safetensors package still reads a task's input files."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    body = raw[8 + n:]
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPES.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: {name} has unsupported dtype {spec['dtype']}")
        start, end = spec["data_offsets"]
        flat = torch.frombuffer(bytearray(body[start:end]), dtype=dtype) if end > start \
            else torch.empty(0, dtype=dtype)
        out[name] = flat.reshape(spec["shape"])
    return out


def _clone(inp: list) -> list:
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inp]


def task_error(out, ref, dtype: str) -> tuple[float, str | None]:
    """(normalized error, why it fails regardless of tolerance or None)."""
    if tuple(out.shape) != tuple(ref.shape):
        return math.inf, f"shape {tuple(out.shape)} != judge {tuple(ref.shape)}"
    if out.dtype != ref.dtype:
        return math.inf, f"dtype {out.dtype} != judge {ref.dtype}"
    if dtype in TASK_EXACT_DTYPES:
        bad = int((out != ref).sum().item())
        return (0.0, None) if bad == 0 else (math.inf, f"{bad} element(s) differ (exact dtype)")
    x, y = out.to(torch.float64), ref.to(torch.float64)
    yf, xf = torch.isfinite(y), torch.isfinite(x)
    if bool((yf & ~xf).any()):
        return math.inf, f"{int((yf & ~xf).sum())} non-finite value(s) where the judge is finite"
    nonfin = ~yf
    if bool(nonfin.any()):
        same = (torch.isnan(x) & torch.isnan(y)) | (x == y)
        if not bool(same[nonfin].all()):
            return math.inf, "non-finite values differ from the judge's"
    if not bool(yf.any()):
        return 0.0, None
    scale = float(y[yf].abs().max().item())
    err = float((x[yf] - y[yf]).abs().max().item())
    if err == 0.0:
        return 0.0, None
    return (err / scale if scale > 0 else math.inf), None


def task_tolerance(problem: dict) -> dict:
    """Per output: the calibrated tolerance, never below the dtype floor."""
    cal = (problem.get("tolerance") or {}).get("outputs") or {}
    out = {}
    for name, spec in problem["definition"]["outputs"].items():
        floor = TASK_FLOOR.get(spec["dtype"], 0.0)
        out[name] = max(float(cal.get(name, 0.0)), floor)
    return out


def task_correctness(problem: dict, impl, dps: bool, device, prec=None) -> tuple[dict, list]:
    """Every input group of the problem's workload: the judge, then ``impl``. Returns
    (verdict, the input sets in group order -- every one is timed)."""
    judge, make_inputs = _task_custom(problem)
    tol = task_tolerance(problem)
    names = list(problem["definition"]["outputs"])
    dtypes = {n: s["dtype"] for n, s in problem["definition"]["outputs"].items()}
    verdict = {"pass": False, "status": "PASSED", "max_abs": 0.0, "max_rel": 0.0,
               "matched_ratio": 1.0, "checks": [],
               "tolerance": {"outputs": tol, "rule": "max|y - judge| / max|judge| <= tolerance "
                             "per output (finite positions); non-finite values where the judge "
                             "has them; integer / bool outputs exact"}}
    sets: list = []

    def fail(status, error):
        verdict.update(status=status, error=error)
        return verdict, sets

    for gi, group in enumerate(task_groups(problem)):
        try:
            inp = task_inputs(problem, group, device, make_inputs)
        except Exception as exc:  # noqa: BLE001 -- the judge's / the file's, not the kernel's
            return fail("INPUT_ERROR", f"input group {group['name']}: {type(exc).__name__}: {exc}")
        sets.append(inp)
        pristine = _precision_snapshot(prec, inp) if prec is not None and gi < PRECISION_SETS else None
        refs = call_impl(problem, judge, _clone(inp), False, device, no_grad=True)
        try:
            outs = call_impl(problem, impl, inp, dps, device, no_grad=True)
        except Exception as exc:  # noqa: BLE001 — the implementation is foreign code
            return fail("RUNTIME_ERROR", f"input group {group['name']}: {type(exc).__name__}: {exc}")
        if (why := _lazy_output(outs)):
            return fail("REWARD_HACK", why)
        if pristine is not None:
            _precision_measure(problem, prec, f"group_{group['name']}", outs, refs, judge,
                               pristine, device)
        for name, o, r in zip(names, outs, refs):
            e, why = task_error(o, r, dtypes[name])
            verdict["checks"].append({"group": group["name"], "output": name,
                                      "error": None if math.isinf(e) else e,
                                      "tolerance": tol[name], **({"why": why} if why else {})})
            if not math.isinf(e):
                verdict["max_rel"] = max(verdict["max_rel"], e)
            if why or e > tol[name]:
                return fail("INCORRECT_NUMERICAL",
                            f"input group {group['name']} output {name}: "
                            + (why or f"normalized error {e:.4g} > tolerance {tol[name]:.4g}"))
    verdict["pass"] = True
    return verdict, sets


def _task_side(problem: dict, sets: list, kernel, kernel_dps: bool, ref=None,
               ref_dps: bool = False, device="cuda") -> dict:
    """One shape's performance block: timing.py over its input groups, each paired with
    the reference when one is timed."""
    import timing

    def calls(fn, dps, inp):
        if fn is None:
            return None
        outs = allocate_outputs(problem, device) if dps else []
        return [lambda: fn(*inp, *outs)]

    groups = [g["name"] for g in task_groups(problem)]
    work = [(name, calls(kernel, kernel_dps, inp), calls(ref, ref_dps, inp))
            for name, inp in zip(groups, sets)]
    result = timing.measure_windows_workloads(work, burn_in_s=BURN_IN_S)
    block = timing.workloads_performance(result)
    block["device_window_basis"] = BASIS_PLAIN
    block["leaderboard"] = {
        "suite": SUITE_TASK, "upstream": UPSTREAM[SUITE_TASK], "timer": "timing.py",
        "protocol": {"aggregate": "geomean over input groups", "window": problem.get("timing")},
        "reading": "geometric mean over the shape's input groups of timing.py's windows",
        "kernel_per_set_ms": [_round(w.get("device_median_ms")) for w in block["workloads"]],
        "ref_per_set_ms": [_round(w.get("ref_device_ms")) for w in block["workloads"]
                           if w.get("ref_device_ms") is not None],
        "definition": problem["definition"]["name"],
        "workload": problem["workload"].get("uuid"),
        "reference": (problem.get("reference_impl") or {}).get("name", "none"),
        "clock": clock_state(),
    }
    return block


def _task_shapes(problem: dict) -> list[dict]:
    """Every shape of the task as a problem view (a one-shape task is its own view)."""
    views = workload_views(problem)
    if len(views) == 1:
        views = [dict(problem, _wl_name=str(problem["workload"].get("uuid")))]
    return views


def _task_aggregate(problem: dict, blocks: list) -> dict:
    """One shape: its own block (geomean over its input groups). Several: the geomean over
    the shapes, each shape's block under ``workloads``."""
    if len(blocks) == 1:
        return {k: v for k, v in blocks[0].items() if k not in ("name", "axes")}
    return group_performance_block(problem, blocks)


def _run_task(problem: dict, args, impl, dps: bool, ref_fn, ref_dps: bool, timed_ref,
              device) -> int:
    """A task's correctness + timing, shape by shape, stopping at the first shape that
    fails."""
    views = _task_shapes(problem)
    per, blocks, first_bad = [], [], None
    for v in views:
        verdict, sets = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        print(f"[task] {v['_wl_name']} {json.dumps(v['workload'].get('axes'))}: correctness "
              f"{verdict['status']}" + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
        if args.mode == "correctness":
            continue
        block = _task_side(v, sets, impl, dps, timed_ref, ref_dps, device)
        blocks.append({"name": v["_wl_name"], "axes": v["workload"].get("axes"), **block})
        print(f"[task] {v['_wl_name']}: kernel device {block['device_median_ms']} ms, e2e "
              f"{block['bwd_median_ms']} ms"
              + (f"; reference device {block['ref_device_ms']} ms, x{block['vs_ref_device']}"
                 if block.get("ref_device_ms") else f" [reference: {block.get('in_job_ref')}]"))
    verdict = group_verdict(per, len(views), first_bad)
    result = {"correctness": _correctness_json(verdict), "performance": None}
    print(f"[task] correctness: {verdict['status']} ({len(per)}/{len(views)} shapes judged)"
          + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
    if not verdict["pass"] or args.mode == "correctness":
        _emit(result)
        return 0
    perf = _task_aggregate(problem, blocks)
    result["performance"] = perf
    print(f"[task] {len(blocks)} shape(s): kernel device {perf['device_median_ms']} ms, e2e "
          f"{perf['bwd_median_ms']} ms"
          + (f"; reference device {perf['ref_device_ms']} ms, x{perf['vs_ref_device']}"
             if perf.get("vs_ref_device") else f" [reference: {perf.get('in_job_ref')}]")
          + f"; clock {clock_state()}")
    _emit(result)
    return 0


def _task_ref_alone(problem: dict, ref_fn, ref_dps: bool, device) -> int:
    """--mode ref: the reference alone, every input group of every shape, in the same
    geomean structure the bench reports (so the archive and the gate name the same
    workloads)."""
    blocks = []
    for v in _task_shapes(problem):
        verdict, sets = _correctness(dict(v, _no_precision=True), ref_fn, ref_dps, device)
        if not verdict["pass"]:
            _emit({"error": f"the reference fails the judge on {v['_wl_name']}: "
                            f"{verdict.get('error')}"})
            return 1
        block = _task_side(v, sets, ref_fn, ref_dps, None, False, device)
        blocks.append({"name": v["_wl_name"], "axes": v["workload"].get("axes"), **block})
    perf = _task_aggregate(problem, blocks)
    out = {"median_ms": perf["bwd_median_ms"], "device_median_ms": perf["device_median_ms"],
           "device_window_basis": BASIS_PLAIN, "host_enqueue_ms": perf.get("host_enqueue_ms"),
           "leaderboard_timer": "timing.py", "clock": clock_state(),
           "aggregate": "geomean", "n_workloads": perf.get("n_workloads", len(blocks)),
           "workloads": [{"name": w.get("name"), "median_ms": w.get("bwd_median_ms"),
                          "device_median_ms": w.get("device_median_ms")}
                         for w in perf.get("workloads", [])]}
    _emit(out)
    return 0


def _task_calibrate(problem: dict, ref_fn, ref_dps: bool, device) -> int:
    """--mode calibrate: the reference against the judge on every input group of every
    shape; per output the largest normalized error, whether it is close enough to call a
    match (TASK_MATCH_CAP), and the tolerance it implies (TASK_TOL_MARGIN x that error,
    never below the dtype floor)."""
    outputs = problem["definition"]["outputs"]
    worst = {n: 0.0 for n in outputs}
    rows, problems = [], []
    for v in _task_shapes(problem):
        judge, make_inputs = _task_custom(v)
        for group in task_groups(v):
            where = f"{v['_wl_name']}/{group['name']}"
            try:
                inp = task_inputs(v, group, device, make_inputs)
                refs = call_impl(v, judge, _clone(inp), False, device, no_grad=True)
                outs = call_impl(v, ref_fn, inp, ref_dps, device, no_grad=True)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{where}: {type(exc).__name__}: {exc}")
                continue
            for name, o, r in zip(outputs, outs, refs):
                e, why = task_error(o, r, outputs[name]["dtype"])
                rows.append({"shape": v["_wl_name"], "group": group["name"], "output": name,
                             "error": None if math.isinf(e) else e, **({"why": why} if why else {})})
                if why:
                    problems.append(f"{where} output {name}: {why}")
                elif e > TASK_MATCH_CAP:
                    problems.append(f"{where} output {name}: normalized error {e:.4g} > "
                                    f"{TASK_MATCH_CAP} -- the judge and the reference disagree")
                else:
                    worst[name] = max(worst[name], e)
    tol = {n: max(TASK_TOL_MARGIN * worst[n], TASK_FLOOR.get(outputs[n]["dtype"], 0.0))
           for n in outputs}
    _emit({"calibration": {"match": not problems, "problems": problems[:40],
                           "max_error": worst, "margin": TASK_TOL_MARGIN, "tolerance": tol,
                           "floor": {n: TASK_FLOOR.get(s["dtype"], 0.0) for n, s in outputs.items()},
                           "cap": TASK_MATCH_CAP, "rows": rows[:400]}})
    print(f"[task] calibrate: {'MATCH' if not problems else 'NO MATCH'}; tolerance {tol}")
    return 0


def _ref_alone(problem: dict, ref_fn, ref_dps: bool, device) -> int:
    """--mode ref: the in-job reference alone on the suite's clock, on every workload --
    through the one callable the bench loaded. A sealed source reference serves its
    source only inside the windows of the closed_ref instance that imported it (a JIT
    compiling at its first call, CuTe DSL, reads it there); loaded again, it would come
    back through another instance whose windows do not open that loader."""
    views = workload_views(problem)
    rows = []
    for v in views:
        _, sets = _correctness(dict(v, _no_precision=True), ref_fn, ref_dps, device)
        m = measure(v, sets, ref_fn, ref_dps, device=device)
        win = _side_windows(ref_fn, ref_dps, v, sets[0], device)
        rows.append({"median_ms": _round(win["median_ms"]),
                     "device_median_ms": _round(m["kernel_ms"]),
                     "device_window_basis": m["basis"],
                     "host_enqueue_ms": _round(win["host_enqueue_ms"]),
                     "leaderboard_timer": m["timer"],
                     "clock": clock_state(),
                     "per_set_device_ms": [_round(x) for x in m["kernel_per_set_ms"]]})
    if len(views) == 1:
        _emit(rows[0])
        return 0
    # One aggregate block: forgekernel.baseline_cli.parse_ref_stdout reads the geomean
    # block (aggregate = "geomean") and the workloads it names, never a nested one.
    out = {k: _round(_geomean(r[k] for r in rows))
           for k in ("median_ms", "device_median_ms", "host_enqueue_ms")}
    out.update({"device_window_basis": rows[0]["device_window_basis"],
                "leaderboard_timer": rows[0]["leaderboard_timer"], "clock": clock_state(),
                "aggregate": "geomean", "n_workloads": len(rows),
                "workloads": [{"name": v["_wl_name"], **r} for v, r in zip(views, rows)]})
    _emit(out)
    return 0


def run_leaderboard_bench(argv=None, problem_path=None) -> int:
    """The card's eval/bench.py: correctness then timing, the problem's suite's way."""
    p = argparse.ArgumentParser(prog="bench.py (leaderboard mode)")
    p.add_argument("--mode", choices=("all", "correctness", "perf", "ref", "calibrate"),
                   default="all",
                   help="calibrate (task suite only): the reference against the judge")
    p.add_argument("--impl", choices=("kernel", "ref"), default="kernel",
                   help="ref: the in-job reference judged and timed AS the implementation "
                        "(card self-test; vs_ref_device must read ~1.0)")
    args = p.parse_args(argv)
    problem = load_problem(problem_path)
    _SCOPE.clear()
    if scope := apply_workload_scope(problem):
        _SCOPE.update(scope)
        which = "representative workload" if scope["workloads"] == "representative" else "workload"
        print(f"[leaderboard] WORKLOAD SCOPE: {which} {scope['rep']} only "
              f"(1 of {scope['of']}; {WORKLOAD_SCOPE_ENV}) — a meta reading, never a gate")
    device = "cuda"
    suite = problem["suite"]
    print(f"[leaderboard] suite={suite} ({UPSTREAM[suite]}) definition="
          f"{problem['definition']['name']} {describe_card_workloads(problem)}")

    ref_fn, ref_dps = _impl(problem, "ref")
    if suite == SUITE_TASK and args.mode == "calibrate":
        return _task_calibrate(problem, ref_fn, ref_dps, device)
    if args.mode == "calibrate":
        p.error("--mode calibrate is the forgekernel task suite's")
    if suite == SUITE_TASK and args.mode == "ref":
        return _task_ref_alone(problem, ref_fn, ref_dps, device)
    if args.mode == "ref" and len(workload_views(problem)) > 1:
        return _ref_alone(problem, ref_fn, ref_dps, device)
    if args.mode == "ref":
        # [loop] ref_cmd: the in-job reference alone on the suite's clock, for
        # the loop's reference measurement. One flat block — nested per-set blocks with an ms
        # key would be read as rival backends by the archive's fastest-block rule.
        _, sets = _correctness(dict(problem, _no_precision=True), ref_fn, ref_dps, device)
        m = measure(problem, sets, ref_fn, ref_dps, device=device)
        win = _side_windows(ref_fn, ref_dps, problem, sets[0], device)
        _emit({"median_ms": _round(win["median_ms"]),
               "device_median_ms": _round(m["kernel_ms"]),
               "device_window_basis": m["basis"],
               "host_enqueue_ms": _round(win["host_enqueue_ms"]),
               "leaderboard_timer": m["timer"],
               "clock": clock_state(),
               "per_set_device_ms": [_round(v) for v in m["kernel_per_set_ms"]]})
        return 0

    if args.impl == "kernel" and (why := kernel_fence(problem)):
        print(f"[leaderboard] correctness: FENCE — {why}")
        _emit({"correctness": {"pass": False, "status": "FENCE", "error": why},
               "performance": None})
        return 0
    timed_ref = ref_fn if problem.get("time_reference", True) else None
    if timed_ref is not None and args.mode != "correctness" and \
            (problem.get("reference_impl") or {}).get("kind") not in (None, "definition"):
        warm_reference(problem, ref_fn, ref_dps, device)
    try:
        impl, dps = (ref_fn, ref_dps) if args.impl == "ref" else _impl(problem, "kernel")
    except Exception as e:  # noqa: BLE001
        if Path(problem.get("kernel_file", "kernel.py")).suffix not in CUDA_KERNEL_SUFFIXES:
            raise
        why = f"{type(e).__name__}: {e}"
        print(f"[leaderboard] correctness: COMPILE_ERROR — {why[-6000:]}")
        _emit({"correctness": {"pass": False, "status": "COMPILE_ERROR", "error": why[-6000:]},
               "performance": None})
        return 0
    if suite == SUITE_TASK:
        return _run_task(problem, args, impl, dps, ref_fn, ref_dps, timed_ref, device)
    if len(workload_views(problem)) > 1:
        return _run_group(problem, args, impl, dps, ref_fn, ref_dps, timed_ref, device)
    cper, cbad = judge_check_only(problem, impl, dps, device)
    if cbad is not None:
        verdict = group_verdict(cper, 1 + len(check_views(problem)), cbad)
        print(f"[leaderboard] correctness: {verdict['status']} — {verdict.get('error')}")
        _emit({"correctness": _correctness_json(verdict), "performance": None})
        return 0
    verdict, sets = _correctness(problem, impl, dps, device)
    result = {"correctness": _correctness_json(verdict), "performance": None}
    if verdict.get("precision"):
        result["precision"] = verdict["precision"]
    if cper:
        result["correctness"]["check_only"] = [{"name": v["_wl_name"], "status": x.get("status")}
                                               for v, x in cper]
    print(f"[leaderboard] correctness: {verdict['status']}"
          + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
    _say_precision(result.get("precision"))
    if not verdict["pass"] or args.mode == "correctness":
        _emit(result)
        return 0
    timed_sets = sets
    threads_mark = verdict.get("threads_after_first_call")
    m = measure(problem, timed_sets, impl, dps, timed_ref, ref_dps, device=device)
    if threads_mark is not None and threading.active_count() > threads_mark:
        result["correctness"] = {"pass": False, "status": "REWARD_HACK",
                                 "error": f"thread injection: {threading.active_count()} "
                                          f"threads after timing vs {threads_mark} before"}
        _emit(result)
        return 0
    kwin = _side_windows(impl, dps, problem, timed_sets[0], device)
    rwin = (_side_windows(ref_fn, ref_dps, problem, timed_sets[0], device)
            if m["ref_ms"] is not None else None)
    result["performance"] = performance_block(problem, m, kwin, rwin)
    lb = result["performance"]["leaderboard"]
    print(f"[leaderboard] {m['timer']} ({m['basis']}): kernel {m['kernel_ms']:.4f} ms"
          + (f", ref {m['ref_ms']:.4f} ms, x{m['ref_ms'] / m['kernel_ms']:.4f}"
             if m["ref_ms"] else f" [reference: {result['performance']['in_job_ref']}]")
          + f"; per set {lb['kernel_per_set_ms']}; clock {lb['clock']}")
    if board := lb.get("sol_board"):
        print(f"[leaderboard] SOL score {board['sol_score']} (T_b {board['t_b_ms']} ms, "
              f"T_SOL {board['t_sol_ms']} ms; x{board['vs_t_b']} vs T_b)")
    for note in ("timer_note", "unstable_note"):
        if lb.get(note):
            print(f"[leaderboard] WARNING {lb[note]}")
    if board and board.get("clock_warning"):
        print(f"[leaderboard] WARNING {board['clock_warning']}")
    _emit(result)
    return 0


def _run_group(problem: dict, args, impl, dps: bool, ref_fn, ref_dps: bool, timed_ref,
               device) -> int:
    """A group card's correctness + timing: workload by workload (each its own correctness
    rounds, thread snapshot and timing), stopping at the first workload that fails."""
    views = workload_views(problem)
    n_checks = len(check_views(problem))
    # the correctness-only workloads first: a kernel that cannot run them is out before a
    # single timed workload is paid for
    per, first_bad = judge_check_only(problem, impl, dps, device)
    blocks = []
    for v in views if first_bad is None else []:
        verdict, sets = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        print(f"[leaderboard] {v['_wl_name']} {json.dumps(v['workload'].get('axes'))}: "
              f"correctness {verdict['status']}"
              + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
        if args.mode == "correctness":
            continue
        threads_mark = verdict.get("threads_after_first_call")
        m = measure(v, sets, impl, dps, timed_ref, ref_dps, device=device)
        if threads_mark is not None and threading.active_count() > threads_mark:
            first_bad = (v, {"pass": False, "status": "REWARD_HACK",
                             "error": f"thread injection: {threading.active_count()} threads "
                                      f"after timing vs {threads_mark} before"})
            per[-1] = first_bad
            break
        kwin = _side_windows(impl, dps, v, sets[0], device)
        rwin = (_side_windows(ref_fn, ref_dps, v, sets[0], device)
                if m["ref_ms"] is not None else None)
        block = performance_block(v, m, kwin, rwin)
        blocks.append({"name": v["_wl_name"], "axes": v["workload"].get("axes"), **block})
        print(f"[leaderboard] {v['_wl_name']} {m['timer']} ({m['basis']}): kernel "
              f"{m['kernel_ms']:.4f} ms"
              + (f", ref {m['ref_ms']:.4f} ms, x{m['ref_ms'] / m['kernel_ms']:.4f}"
                 if m["ref_ms"] else f" [reference: {block['in_job_ref']}]")
              + (f"; SOL {block['leaderboard']['sol_board']['sol_score']}"
                 if block["leaderboard"].get("sol_board") else ""))
    verdict = group_verdict(per, len(views) + n_checks, first_bad)
    result = {"correctness": _correctness_json(verdict), "performance": None}
    if verdict.get("precision"):
        result["precision"] = verdict["precision"]
    _say_precision(result.get("precision"))
    print(f"[leaderboard] correctness: {verdict['status']} ({len(per)}/{len(views) + n_checks} workloads judged"
          + (f", {n_checks} of them correctness-only" if n_checks else "") + ")"
          + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
    if not verdict["pass"] or args.mode == "correctness":
        _emit(result)
        return 0
    perf = group_performance_block(problem, blocks)
    result["performance"] = perf
    lb = perf["leaderboard"]
    print(f"[leaderboard] geomean over {len(blocks)} workloads: kernel "
          f"{perf['device_median_ms']} ms"
          + (f", ref {perf['ref_device_ms']} ms, x{perf['vs_ref_device']}"
             if perf.get("vs_ref_device") else f" [reference: {perf['in_job_ref']}]")
          + f"; clock {lb.get('clock')}")
    if board := lb.get("sol_board"):
        print(f"[leaderboard] SOL score mean {board['sol_score_mean']} over the group "
              f"(per workload {board['sol_score']})")
    for note in ("timer_note", "unstable_note"):
        if lb.get(note):
            print(f"[leaderboard] WARNING {lb[note]}")
    if board and board.get("clock_warning"):
        print(f"[leaderboard] WARNING {board['clock_warning']}")
    _emit(result)
    return 0


def ncu_window(problem_path=None, calls: int = 8, which: str = "kernel") -> None:
    """eval/ncu_run.py: the kernel's calls on one input set inside profiler start/stop
    (input construction and JIT warm-up stay outside the NCU window). ``which="ref"``
    profiles the in-job reference instead (a sealed reference's recon tree). A group card
    profiles its representative, or the workload ``FK_LB_WORKLOAD_SCOPE`` names (a meta
    tool reading the card one workload at a time)."""
    problem = load_problem(problem_path)
    if scope := apply_workload_scope(problem):
        print(f"[leaderboard] ncu: workload {scope['rep']} (1 of {scope['of']}; "
              f"{WORKLOAD_SCOPE_ENV})", flush=True)
    if which == "kernel" and (why := kernel_fence(problem)):
        # the bench's fence, on the profiler's path into the kernel file too
        print(f"[leaderboard] ncu: FENCE — {why}")
        raise SystemExit(3)
    kernel, dps = _impl(problem, which)
    if problem["suite"] == SUITE_TASK:
        # every input group of every shape (the one shape a meta tool scoped to, if any),
        # the calls spread over them: a profile of one group is a profile tuned to it
        pairs = [(v, task_inputs(v, g, "cuda")) for v in _task_shapes(problem)
                 for g in task_groups(v)]
        per = max(1, calls // len(pairs))
        outs = {id(v): allocate_outputs(v, "cuda") if dps else [] for v, _ in pairs}
        for v, inp in pairs:
            for _ in range(3):
                kernel(*inp, *outs[id(v)])
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for v, inp in pairs:
            for _ in range(per):
                kernel(*inp, *outs[id(v)])
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return
    if problem["suite"] == SUITE_SOL:
        torch.manual_seed(SOL_SEED)
        random.seed(SOL_SEED)
        _, custom = reference_callables(problem)
        inp = gen_inputs(problem, "cuda", custom_inputs_fn=custom)
    else:
        inp = gen_inputs(problem, "cuda")
    outs = allocate_outputs(problem, "cuda") if dps else []
    for _ in range(3):
        kernel(*inp, *outs)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(calls):
        kernel(*inp, *outs)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    raise SystemExit(run_leaderboard_bench())
