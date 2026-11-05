# Leaderboard tasks: FlashInfer-Bench / SOL-ExecBench

A **leaderboard task** is one problem of a leaderboard (one workload of one definition, or a group of
workloads), judged and timed **by that board's evaluation protocol**. The design rule in one sentence:
**the task's verdict must equal the board's verdict** -- the tolerance, the inputs and the stopwatch
are not the task author's choice; they replicate, item by item, the board's public evaluation code
(pinned to an upstream commit).

Why an ordinary task cannot simply be used to climb a board:

| | Ordinary task | FlashInfer-Bench | SOL-ExecBench |
|---|---|---|---|
| Correctness | the judge the teacher writes from `description.md`, tolerance calibrated against the reference (2 × its normalised max error, never below the dtype floor) | elementwise: an element is wrong only when `|d|>atol and |d|/|ref|>rtol`; by default 100% of the elements must pass | elementwise allclose, tolerance calibrated per workload, 99% by default |
| Inputs | the input groups the user gives, completed by the teacher to at least 3 per shape | 3 unseeded randn groups / the dataset's safetensors | seed 200, 10 rounds, parameter-name heuristics + the problem's own generator |
| Timing | warm L2, amortised over consecutive calls (device window), paired with the reference in the same job | one call at a time, L2 flushed before each call, CUPTI kernel span, median, averaged over 3 groups; clocks not locked | as left (one group, a new address every call); SM 1500 / memory 3996 MHz locked |
| Ranking | -- | speedup over the dataset's baseline solution | SOL score `S = 1/(1+(T_k−T_SOL)/(T_b−T_SOL))`, with T_b / T_SOL published per workload by the board |

A kernel can pass its own judge and fail the board's; an optimization tuned under a warm-L2 amortised
window may not survive the board's cold-L2 single-call reading -- either way the work is wasted.

## Usage

```bash
# 0) the datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
#    (when a problem uses safetensors inputs, `git lfs pull --include=<that blob>`; forge init names it)
#    SOL-ExecBench: run scripts/download_solexecbench.py from its repository to get data/benchmark/<subset>/<problem>/

# 1) list the problem's workloads and pick one (or a group)
forge init gqa_decode --leaderboard flashinfer-bench --dataset ~/flashinfer-trace \
    --definition gqa_paged_decode_h32_kv8_d128_ps1 --list-workloads

# 2) create the task tasks/<op>/
forge init gqa_decode --leaderboard flashinfer-bench --dataset ~/flashinfer-trace \
    --definition gqa_paged_decode_h32_kv8_d128_ps1 --workload e2142798
forge init attn_bwd --leaderboard sol-execbench --dataset ~/SOL-ExecBench/data/benchmark \
    --definition L1/001_attention_softmax_dropout_value_matmul_backward --workload 0
#    an SOL task fetches this workload's T_b / T_SOL from the board's public API and freezes them into
#    the task (offline: --sol-board <a saved kernels/<id> JSON>; without: --no-sol-board, and then
#    task.toml must set [target] ms)

# 3) review tasks/<op>/task.toml (arch, target, agents, budget), then create and run as usual
forge new tasks/gqa_decode --dry-run
forge new tasks/gqa_decode                          # -> .artifacts/gqa_decode_<id>
forge run .artifacts/gqa_decode_<id>

# 4) export the board's solution JSON
forge export .artifacts/gqa_decode_<id> --out solution.json --name <name> --author <author>
```

`--workload` can also be a group of workloads (comma-separated indices, inclusive `lo-hi` ranges or
uuid prefixes, e.g. `3,5,9` or `0-6`): one task judges all of them and gates on the geometric mean of
their readings, and each workload is one shape of the task. `--check-workload` (same syntax) names
workloads that are judged for correctness only, never timed.

`forge init` writes `task.toml`, pointing at the frozen card:

```toml
[task]
language = "cutedsl"                 # "cuda" when generated with --kernel-file kernel.cu
arch     = "sm100"

[leaderboard]
suite = "flashinfer-bench"
card  = "leaderboard/card"
# ref = "leaderboard/ref"            # only when generated with --sealed-ref / --binary-ref

# [target] ratio / ms, [agent], [budget] as in an ordinary task; what is not written comes from global_config/
```

For a leaderboard task, `forge run` writes no judge and calibrates nothing (the judge, the inputs and
the stopwatch come with the card), and never has /goal write a reference first.

The generated card is frozen under `tasks/<op>/leaderboard/card/` and copied into
`.artifacts/<op>_<id>/operators/<op>/` for every run:

```
leaderboard/card/
  eval/leaderboard/problem.json   <- the frozen problem: definition + workload (verbatim) + reference + provenance
  eval/leaderboard/blob/…         <- the workload's safetensors inputs (paths rewritten relative to problem.json)
  eval/leaderboard/baseline/…     <- FlashInfer-Bench: the dataset's baseline solution (the in-job reference)
  eval/leaderboard.py             <- copied from the template, do not edit: both boards' judges + inputs + stopwatches
  eval/timing.py                  <- likewise (the e2e window and the host enqueue clock, for logs and audits)
  eval/bench.py  eval/ncu_run.py  <- generated thin entry points (they only call leaderboard.py)
  gpu_entry.sh                    <- correctness / all / cardref / ref / ncu / cmd
  kernel.py                       <- signature frozen by the definition: run(<inputs…>, <outputs…>), DPS
  window.toml                     <- filled in: both sides are the definition's inputs / outputs, no asymmetry
  loop.toml.example               <- the suggested loop configuration (gate_window + reference: FlashInfer-Bench = ref_cmd,
                                     SOL = the declared T_b); forge run generates the run's loop.toml from it,
                                     do not edit by hand
```

## What is replicated (pinned upstream)

- FlashInfer-Bench: `flashinfer-ai/flashinfer-bench@15618a67` (`bench/evaluators/default.py`,
  `lowbit.py`, `bench/utils.py`, `bench/config.py` + `eval_config.yaml`)
- SOL-ExecBench: `NVIDIA/SOL-ExecBench@a9fa0804` (`driver/templates/eval_driver.py`,
  `core/bench/{io,correctness,timing,reward_hack}.py`)

| | FlashInfer-Bench | SOL-ExecBench |
|---|---|---|
| Input groups | `num_trials`=3 groups, **unseeded** (as upstream: every job draws different ones) | `seed=200` once, one group drawn per round for 10 rounds; timing uses only the last round's group |
| Input generation | float randn; fp8/fp4 randn clipped to ±2; int randint (±1024, int8 ±128); bool; scalar / safetensors from the workload; sampling's `probs` through softmax | parameter-name heuristics (norm weight=1, bias=0, causal/binary mask, rope cos/sin, positive statistics, SSM decay, softmax output, `randn/√fan_in` weights) → otherwise randn; `custom_inputs_entrypoint`; safetensors / scalar |
| Order of checks | the reference on every group first, then the solution group by group; shape → dtype → inf/NaN in the solution (wrong at once) → elementwise | reference → solution in every round; round 0 checks the output type is strictly `torch.Tensor`, the shape, the dtype, and records the thread count; every round: inf/NaN on either side is wrong (only `allow_negative_inf` lets a −inf at the same position through), an all-zero solution where the reference is non-zero is wrong, `|d| ≤ atol + rtol·|ref|` |
| Tolerance | `atol=rtol=1e-2`, match ratio 1.0 by default; `op_type=moe` → 0.95; a name containing `moe_fp8_block_scale` with nothing set → 0.95 | from the workload (default 1e-2 / 1e-2 / 0.99, optional `max_error_cap`) |
| Timing | per group: 10 warm-up, 50 timed, L2 flushed every call, median of the single-call kernel span; **the reading = the mean over the groups** | last group: 10 warm-up, 50 timed, L2 flushed every call, a new address every call (random offset in multiples of 256 B, seed 200), median |
| Anti-cheating | -- (none upstream) | lazy outputs, thread injection during timing |

The reference:

- A FlashInfer-Bench task's in-job reference = the solution in the dataset's `solutions/baseline/`
  (what the board compares against), on **the same clock and the same input groups as the student,
  mirror-paired** (after burn-in: ref / kernel / kernel / ref, each side the mean of two readings -- the
  rule of `timing.measure_paired`); the GPU machine needs the libraries it depends on (mostly
  `flashinfer`).
- An SOL-ExecBench task **does not time a reference in the job** (problem.json
  `time_reference = false`): the only one within reach is the definition's own eager PyTorch
  reference -- a dozen small ops, host-bound, whose single-call span is a whole-call outlier under any
  stopwatch; as a reference it yields only noise, and any fused kernel beats it from the start. The
  board ranks against its own scoring baseline T_b (an optimised PyTorch implementation whose source is
  not public, but **whose per-workload latency is**) and the hardware floor T_SOL: `forge init` takes
  both numbers for this problem's workload from the board's API
  (`research.nvidia.com/benchmarks/sol-execbench/api`, `workloads[i].baseline_latency_ms` / `sol_ms` of
  `kernels/<id>`; the list order = the dataset's workload.jsonl order, checked against the axes at
  generation time) and freezes them into `sol_board` in `problem.json`.
- An existing solution can also serve as the reference (blackbox mode, see
  [blackbox-mode.md](blackbox-mode.md)): `--sealed-ref <solution.json>` (a Python solution: PyTorch /
  Triton / CuTe DSL) or `--binary-ref <solution.json>` (a compiled cuda_cpp solution) seals it into the
  card and a recon tree (`tasks/<op>/leaderboard/ref/`), where it is timed beside the kernel in the job.

How the eval JSON maps: `device_median_ms` / `ref_device_ms` / `vs_ref_device` = the board's reading
(what the gate reads); `bwd_median_ms` / `vs_ref` = `timing.time_e2e` (logs only);
`device_window_basis` names the stopwatch; the `leaderboard` block carries the suite, the upstream,
the timer, the per-group readings and `clock` (the clock-lock state gpu_entry.sh exports); an SOL task
also has `sol_board`: `sol_score`, `vs_t_b`, T_b / T_SOL.

### The stopping target

The loop's target = the reading of the reference the card declares ÷ `[target] ratio` (or the absolute
`[target] ms` when `task.toml` sets one); a KEEP that reads below it stops the loop. So the reference
must be the number the board ranks by:

- FlashInfer-Bench: the reference = the dataset's baseline solution (the card's
  `ref_cmd = "python3 eval/bench.py --mode ref"`), target = its board reading / ratio (1.0 = match it,
  1.2 = 20% faster, which is also the `global_config/` default). For a definition with no baseline
  solution the reference falls back to the eager reference; set `[target] ms` in `task.toml` then.
- SOL-ExecBench: the card declares T_b as the architecture reference (`arch_ref_ms` /
  `arch_ref_device_ms`, basis `cupti_kernel_span`), target = T_b / ratio, so ratio 1.0 stops at
  S = 0.5; the card's `loop.toml.example` also lists the kernel latencies for S = 0.6–0.9 -- to aim
  higher, write one of those as `[target] ms`. A task without `sol_board` (`--no-sol-board`, the API was
  unreachable, the problem did not match) has no reference and must set `[target] ms`.

### Clock locking

A clock lock is a state of the GPU, not of a process: a lock that is not reset stays for every later
job on that GPU. So gpu_entry.sh, on every entry, first clears any lock a previous job left
(`nvidia-smi -rgc` / `-rmc`; a process killed outright runs no trap, so only the next one in can clear
it), locks only what this card needs, reads back `clocks.current.graphics / memory` 3 s later (±50 MHz,
as upstream's `verify_clocks`), and resets again on exit through an EXIT trap (including failures and
timeout kills). Locking clocks needs the corresponding permission on the GPU machine.

An SOL task locks SM 1500 / memory 3996 MHz as the board does (upstream `device_config.py`'s B200
preset). When the lock does not hold, or the read-back is off, the card refuses every task except
`help` (it prints a `REFUSED` line and exits 3) -- T_b / T_SOL are readings at that clock, an unlocked
reading is faster, and scoring it or comparing it with the target would pass kernels that fail on the
board. The loop treats such a refusal as a job that did not run (nothing is charged to the kernel),
resubmits, and stops the run after 3 refusals in a row. On a machine that cannot lock clocks, putting
`LB_REQUIRE_CLOCK_LOCK=0` into the job environment (local delivery: the environment `forge` runs in;
ssh delivery: the `REMOTE_ENV_FILE` script) times anyway, but the readings are then not comparable to
the board's. A FlashInfer-Bench task does not lock (the board does not); `LB_SM_CLOCK_MHZ` / `LB_MEM_CLOCK_MHZ` pin
one if you want.

### Stopwatches: four tiers, picked by availability (`FK_LB_TIMER` forces one)

| Tier | What it is | `device_window_basis` |
|---|---|---|
| `upstream` | the board's own function is installed on the GPU machine with a matching signature: `flashinfer.testing.bench_gpu_time_with_cupti` / `sol_execbench…time_runnable` (and `cupti-python` is there) | `cupti_kernel_span` |
| `cupti` | the board's method run directly on cupti-python (a trimmed port of SOL's collector): first discover this call's activity sequence, then for every call take "first start → last end" of the same sequence within that call's CPU timestamp window. Three details follow each board: FlashInfer-Bench counts only kernels, synchronizes after the L2 flush before timing, and traces RUNTIME/DRIVER during timing (as flashinfer does); SOL-ExecBench also counts memcpy/memset and does not synchronize after the flush | `cupti_kernel_span` |
| `profiler` | the same sequence logic on torch.profiler's CUPTI activities (no extra package). Single-kernel calls read the same as the board's timer; the collector differs, so multi-kernel calls may not. The eval JSON carries a warning when it is used | `cupti_kernel_span` |
| `events` | L2 flushed every call + CUDA events around one call (both boards' fallback without CUPTI) | `cold_l2_events` (**not** host-free) |

cupti-python is on PyPI; the card's `gpu_entry.sh` installs it when the GPU machine lacks it (bounded,
a failure only warns, `LB_INSTALL_CUPTI=0` turns it off) -- **better to install it into the GPU
machine's Python environment** and save every job the install. `cupti_kernel_span` is already a
host-free basis in the driver (`baseline.HOST_FREE_BASES`), so the host-bound check exempts it
automatically.

## Fidelity checks (offline, CPU, item by item against the pinned upstream sources)

The check scripts are not in the repository (they need the upstream sources + torch). The results:

- FlashInfer-Bench `compute_error_stats` vs this implementation: 1200 cases (fp32/bf16/fp16 × four
  magnitudes × four noise levels × three match ratios, with zeroed elements) agree value for value.
- SOL-ExecBench `compute_error_stats` vs this implementation: 1600 cases (NaN, inf in the reference,
  −inf at the same position ± `allow_negative_inf`, all zeros, `max_error_cap`, four workload
  tolerances) agree in verdict and value.
- Input generation (same seed, CPU): all 30 purely random-input definitions in flashinfer-trace, and
  174 of all 235 SOL-ExecBench problems (the 61 skipped have safetensors or a smallest workload over
  256 MiB), agree element for element.

## Measured on GPU (B200, sglang v0.5.19 image, flashinfer 0.6.18, cupti-python 13.4)

Four leaderboard tasks ran through the real GPU job path: FlashInfer-Bench rmsnorm_h4096, FlashInfer-Bench gqa_paged_decode
(with safetensors), SOL rmsnorm_h4096, SOL L1 rope (`custom_inputs_entrypoint`) -- correctness PASS
on all of them, the eval JSON read by `parse_eval`, the ref-mode output archived by
`parse_ref_stdout`, NCU metrics collected.

Stopwatches compared (µs; each row comes from one job, the two numbers = two passes in that job;
`upstream` = the board's own timing function):

| Call | upstream | cupti | profiler |
|---|---|---|---|
| FlashInfer-Bench baseline `flashinfer.norm.rmsnorm` (one kernel) | 3.5 / 3.5 | 3.5 / 3.5 | 3.5 / 3.5 |
| FlashInfer-Bench baseline paged decode (one kernel) | 9.3 / 9.3 | 9.3 / 9.3 | -- |
| flashinfer rmsnorm under SOL's mode (one kernel) | 3.71 / 3.74 | 3.71 / 3.74 | 3.73 / 3.74 |
| SOL rmsnorm's PyTorch reference (9 small kernels, host-bound) | 49.7 / 48.2 | 49.7 / 49.7 | 47.9 / 47.9 |
| an 8-op PyTorch kernel under FlashInfer-Bench (host-bound) | 110 / 107 | 98 / 83 | 96 / 83 |

The `events` tier includes launch overhead and is not a kernel span: in another job the same rmsnorm
baseline spanned 2.8 µs against 24 µs by events, and paged decode 7.6 vs 29 µs.

- Single-kernel calls: all tiers agree bit for bit (which is the shape of a student kernel).
- Host-bound multi-kernel calls: a diagnostic job's per-window dumps confirmed the sequence selection
  (the L2-flush launch excluded, all 9 kernels of the call selected), and within one job cupti agrees
  with upstream; but **whole-call readings turn into outliers** -- a pair of mirrored readings of the
  same call came out 125 vs 207 under SOL's own timer, and 44 vs 111 under the cupti tier. Under FlashInfer-Bench,
  cupti reads 10–25% below flashinfer; the cause was not traced item by item (flashinfer also assigns
  windows by the correlation of RUNTIME launches). The runtime flags "two mirrored readings of the same
  side differ by > 1.2×" as `unstable_sets` and warns.
- "Synchronize after the L2 flush or not" matters enormously for a host-bound call: the same call reads
  44 µs without the sync and 70–76 µs with it -- and the two boards do exactly the opposite (FlashInfer-Bench syncs,
  SOL does not); each card follows its own board.

## Not covered / known differences

- **FlashInfer-Bench's two special evaluators, sampling (TVD over 500k samples) and the DSA top-k
  indexer (re-scoring and sorting by the returned indices), are not replicated**: `forge init` refuses
  those definitions outright rather than misjudging them with the default judge. The DSA sparse
  attention special evaluator is equivalent to the default path under DPS.
- **The SOL score uses the board's frozen values**: T_b / T_SOL were measured by the board on its
  machine (Modal B200, the same clock lock), the task's readings on your own B200 at the same clock, and
  the residual between the two machines remains in the score. The LLM static review and the
  quantisation / sampling special evaluators described in the paper are not in the open-source code and
  are not replicated.
- **Fine differences between the profiler tier and upstream CUPTI**: the signature that identifies a
  sequence (upstream SOL = name + copy direction + byte count + memset value + category; here only the
  activity name); `cudaCtxResetPersistingL2Cache` goes through torch's cudart binding and is skipped on
  failure (as upstream's best effort). How far the tiers read apart on one call is what the three-tier
  comparison on GPU above shows.
- **A host-bound multi-kernel call is unstable under any "single-call kernel span" stopwatch**: the span
  includes CPU launch gaps, and how much of them the L2-flush launch covers depends on timing (SOL does
  not synchronize after the flush, so the call queues behind it and part of the gaps is hidden). Measured
  on B200: one pair of mirrored readings of the same PyTorch reference under SOL's own timer was 125 vs
  207 µs. GPU-bound calls (one heavy kernel -- what a student usually writes) read the same under every
  stopwatch. Hence **an SOL task does not time the definition's PyTorch reference in the card** (a dozen
  small ops; the ratio would be noise) and uses the board's published T_b instead (see "The reference"
  above).
- FlashInfer-Bench sets no seed: every job sees different inputs -- that is the board's behaviour, and
  a kernel that fails occasionally gets caught in some round instead of being hidden by a fixed seed.
- safetensors inputs travel into the job with the card; some SOL problems (MoE-like) have workload
  inputs of tens of GB, which only affects GPU memory.
- Language: the export infers it from the card's language / the kernel's imports -- `cutedsl` (FlashInfer-Bench
  `python`, SOL `cute_dsl`), `triton`, otherwise a PyTorch host (FlashInfer-Bench `python`, SOL `pytorch`).

## How it differs from an ordinary task (harness side)

- The multi-workload pre-check at loop start (`window_contract.bench_workloads_findings`) lets a bench
  that calls `run_leaderboard_bench` through: which input groups are timed is the board protocol's call
  (FlashInfer-Bench times the 3 groups it judged, SOL the round-10 group). The driver's one-time per-round hint for
  `n_workloads < 3` is still printed.
- window.toml is filled in by the generator (both sides have the same boundary; a DPS student's output
  buffers are allocated outside the window, the reference allocates inside the call -- the caching
  allocator, no kernel).
- Shapes: one workload is one shape; a workload not named in `--workload` needs a task of its own.

## CUDA mode (`--kernel-file kernel.cu`, task.toml `language = "cuda"`, loop language `cuda-sol`)

The student writes the board's cuda_cpp solution directly: a single CUDA C++ file with its own
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`, which the card compiles with nvcc + the CUTLASS headers the
way SOL-ExecBench does (`-O3 --use_fast_math`, sm_100a, `-lcuda`; extra options only on a
`// fk-cuda-cflags:` line); the export emits it verbatim as `cuda_cpp`, and it can be combined with
C++ solutions.

```bash
forge init <op> --leaderboard sol-execbench ... --kernel-file kernel.cu
# the generated task.toml already says language = "cuda"
```

It and the CuTe DSL mode are **two independent sets of inputs**, decided in one place,
`src/forgekernel/loop/lang_mode.py`:

| | CuTe DSL mode (`cutedsl` etc.) | CUDA mode (`cuda-sol`) |
|---|---|---|
| Prompts | `meta/<prompt>.md` | `meta/cuda/<prompt>.md` (the ones that have an override), the rest shared |
| Knowledge base | `hw_kb/<language>/<kb_arch>/` | `hw_kb/cuda/<kb_arch>/` (starts with methodology entries only; mechanism entries grow as CUDA runs certify them) |
| KB provenance archive | `docs/kb-provenance/` | `docs/kb-provenance/cuda/` |
| Student offline reading | `dsl_source_dirs` (the DSL package source) | `cuda_source_dirs` (CUDA / CUTLASS headers; falls back to `dsl_source_dirs` when empty) |

Tool differences between the two modes:

- **timeline**: the kernel keeps `#define FK_SEG(...) ((void)0)` and marks with
  `FK_SEG("<role>.<seg>.<event>", i);`. The timeline probe replaces it with `%clock64` timestamps (the
  same PTX and the same slab as the DSL version), adds a read-out function to the student's
  `PYBIND11_MODULE`, runs the bench / launcher under `timeline_probe.py --cuda-run`, and fetches the
  slab on exit.
- **lever-price**: all variants are precompiled in parallel before timing (`PRICE PREBUILD`), and the
  timing rounds are cut to the job's remaining time (`PRICE ROUNDS cut`); the no-op check reads the
  SASS of the built `.so` (`cuobjdump -sass`).
- **SASS census**: finds this build's `.so` from the source, extracts the cubin with
  `cuobjdump -xelf`, then runs the same census.
- **Build cache**: the card's `load_cuda_entry` rewrites the source copy in the build directory only
  when the source changed, so an unchanged kernel is not recompiled in every new process (the CUTLASS
  headers take about half a minute).
- **Job environment pre-check**: also checks nvcc, ninja and `$CUTLASS_DIR/include`.
- **Mismatch guard**: a language that does not match the kernel file's suffix (a CUDA language with
  `kernel.py`, or `.cu` with a DSL language) is refused at start, exit 18.
