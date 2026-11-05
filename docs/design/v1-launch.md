# ForgeKernel V1 launch design

> This is the design record of V1's launch surface: what the user writes, what the harness generates,
> where it runs, and how a run is resumed. §0–§12 are the design decisions, §13 the changes to the
> code, and §14 records the trade-offs made in implementation and where the result departs from the
> design.
>
> The two-layer loop itself (the teacher distilling a milestone, the student iterating, the teacher
> diagnosing and patching) is out of scope here. Deeper simplification is left to V2.

---

## 0. Principles

1. **The user writes the problem, not the process.** The user answers "what to compute, on which
   shapes, how fast, learning from what, written in what, by whom, at what cost"; the judge, the timed
   window, the job entry point and the deny list are generated or derived by the harness.
2. **The problem and the run are separate.** The problem definition is in git (`tasks/<op>/`); every
   run gets its own directory (`.artifacts/<op>_<id>/`), not in git. However many times a problem is
   run, the runs do not affect each other, and their results are comparable.
3. **A run cannot change the problem.** The command line takes no argument that overrides the problem;
   to change the configuration, edit the task definition or create another task.
4. **There is exactly one reference.** It is both the source the teacher reads (when readable) and
   the source of the target speed.
5. **The core depends on no particular infrastructure.** Only two ways to reach a GPU are supported:
   this machine and ssh.

---

## 1. Overview

```
tasks/<op>/                      written by the user, in git
  task.toml  description.md  shapes.json  [reference/]
        │
        │  forge new tasks/<op>; forge run .artifacts/<op>_<id>
        ▼
.artifacts/<op>_<id>/                  written by the harness, not in git
  ① snapshot the task definition
  ② the teacher writes the PyTorch judge and the input generation, completing every shape to 3 input groups
  ③ with a reference: align it with the judge on the GPU → freeze the tolerance; without: the dtype floor
  ④ without a reference, /goal writes one first (it must pass the judge); measure the reference
  ⑤ the teacher does reconnaissance and distils milestone v1 (source mode reads the reference's
     source; blackbox mode only runs, times and profiles it)
  ⑥ the two-layer loop: the student iterates ↔ the teacher diagnoses and patches
  ⑦ the finish: confirm the best version over repeated measurements, the correctness monitor,
     the new knowledge-base entries, the report
```

---

## 2. The task definition: `tasks/<op>/`

```
tasks/<op>/
  task.toml        configuration (§2.1)
  description.md   required: the operator's description (§2.2)
  shapes.json      required: the shapes, optionally with input groups per shape (§2.3)
  inputs/          optional: data files of user-provided input groups
  reference/       optional: the reference's source or runnable implementation (§3)
```

### 2.1 `task.toml`

```toml
[task]
language = "cutedsl"                # cutedsl | cuda
arch     = "sm100"                  # the target hardware; also picks the hardware KB (hw_kb/<language>/<arch>/)

[reference]                         # leave the table out = no reference; /goal writes one first (§3.3)
source = "reference/src"            # given → source mode; omitted → blackbox mode
run    = "reference/run.py:run"     # the reference's callable (<file>.py:<function>), same signature as the judge
# blackbox = true                   # source exists but the teacher may not read it (for controlled comparisons)
# deny = ["/path/to/other/impl"]    # extra paths no agent may read; what the harness derives need not be listed (§3.5)

[target]
ratio  = 1.2                        # multiple of the reference's speed (>1 = faster than the reference); or ms
# ms   = 0.85                       # an absolute target (milliseconds)
timing = "device"                   # the timing basis: device | e2e

[agent.teacher]
backend = "claude"
model   = "claude-opus-5-5"
effort  = "high"

[agent.student]
backend = "claude"
model   = "claude-opus-5-5"

[budget]
hours    = 4                        # the inner loop's wall clock
max_meta = 3                        # at most this many outer (teacher patch) rounds
# reference_hours = 8               # without a reference: /goal's wall clock to write one (not counted in hours)
```

`task.toml` writes only what differs from `global_config/`, the bottom layer every task is merged onto
(`models.toml` sets `[agent]`, `loop.toml` sets `[target]` and `[budget]`, `backends.toml` lists API
endpoints). The "Default" column below is what `global_config/` holds.

| Table | Field | Required | Default | Meaning |
|---|---|---|---|---|
| `[task]` | `language` | yes | -- | the language the student writes the kernel in |
| | `arch` | yes | -- | the target hardware: the kernel's compile target and the KB directory |
| `[reference]` | `source` | no | -- | the reference's source path; decides the mode (§3.1) |
| | `run` | yes, with a reference | -- | the reference's callable `<file>.py:<function>`: paired timing calls it repeatedly in the same process |
| | `blackbox` | no | `false` | force blackbox even though source exists |
| | `deny` | no | `[]` | extra paths no agent may read |
| `[target]` | `ratio` / `ms` | no | `ratio = 1.2` | one or the other; by default 20% faster than the reference; without a reference and without `ms`, see §3.3 |
| | `timing` | no | `device` | the timing basis |
| `[agent]` (both agents), `[agent.teacher]`, `[agent.student]` (one) | `backend` | no | `claude` | the agent CLI; V1 supports only `claude` |
| | `provider` | no | `native` | where the tokens go: `native` = the CLI's own login state and environment; or a name from `backends.toml` |
| | `model` | no | `claude-opus-5-5` | the model actually used is written into the run's snapshot |
| | `effort` | no | teacher `high`; student left to the CLI | reasoning effort |
| | `timeout_s` | no | student 7200; teacher 14400 | one turn's wall-clock limit |
| `[budget]` | `hours` | no | `4` | the inner loop's wall clock |
| | `max_meta` | no | `3` | the cap on outer rounds |
| | `reference_hours` | no | `8` | the budget for writing a reference when there is none; not counted in `hours` |

Credentials (API keys, tokens) never go in `task.toml`; a run reads them from the environment or from
`local.env` (§9).

### 2.2 `description.md`

The operator's complete description, from which the teacher writes the PyTorch judge:

- the function signature; each input's / output's shape, dtype and layout;
- the semantics: a formula or prose, including edge conventions (masks, padding, out-of-range values,
  ...);
- the legal range of the inputs (indices must be in bounds, some input is monotonic, ...) -- random
  input generation must respect it.

Write only "what to compute", never "how to compute it" (the same writing rule as the milestone
template's A.2 Math Spec).

### 2.3 `shapes.json`

There are only two concepts:

- **shape**: one concrete set of axis values, which fixes the shape of every tensor. A task may have
  one or more shapes; the student writes **one** kernel that serves all of them, and all tuning is for
  all shapes, never for one of them.
- **input group**: one concrete set of input data for a shape. Each shape has its own input groups.

```json
[
  {"name": "decode_b64", "axes": {"batch": 64, "seq": 1},
   "inputs": [
     {"name": "real_trace", "file": "inputs/decode_b64_trace.safetensors"},
     {"name": "half_empty", "file": "inputs/decode_b64_half_empty.safetensors"}
   ]},
  {"name": "prefill_4k", "axes": {"batch": 1, "seq": 4096}}
]
```

- The user may give input groups per shape (data files under `inputs/`); every group given is used.
- Every shape has at least 3 input groups. When fewer are given, the teacher **completes them to 3**
  while writing the judge (§4), shape by shape; the added groups must differ from the existing ones,
  cover different data situations of that shape (a typical distribution, extremes within the legal
  range, operator-specific structure such as sparsity / masks / index distributions), and state what
  each group is for.
- **Every shape and every input group counts the same**: correctness checks, timing, NCU profiling,
  pricing and diagnosis are all done on every input group of every shape; there is no
  "representative shape" or "representative input group". A single-shape task is no exception.

---

## 3. The reference

### 3.1 The mode is inferred from what is given

| What the user gives | Mode | Teacher | Student |
|---|---|---|---|
| `[reference].source` + `run` | source | reads the source | cannot read it |
| only `run` | blackbox | may only run, time and profile it | cannot read it |
| `source` + `run` + `blackbox = true` | blackbox | as above | cannot read it |
| no `[reference]` | generated first (§3.3), then as source | reads the generated source | cannot read it |

There is no separate "generate a reference with /goal" mode, and no "no distillation" mode. The details
of blackbox mode are in [docs/blackbox-mode.md](../blackbox-mode.md).

### 3.2 One reference, two uses

- **Read**: in source mode the teacher reads its source to distil the milestone.
- **Run**: measure its time and the NCU signature of each of its kernels; the target speed = the
  reference's time ÷ `ratio`, and the milestone's mechanism lines are anchored on these signatures.

There is no longer an "architecture reference" separate from a "library baseline". A comparison with an
existing library can come back later as an optional part of the report, if needed.

### 3.3 Without a reference

1. A /goal session writes an implementation from `description.md` and `shapes.json`; its budget is
   `[budget].reference_hours` (8 hours by default, not counted in `hours`).
2. With `[target].ms` set, it ends as soon as that is reached; without it, it runs until the budget
   is spent.
3. The fastest correct version becomes the reference, treated from then on exactly like a
   user-provided one.

When /goal generates the reference, the §4 judge **must** be written from `description.md` alone, and
the teacher does not read the source /goal wrote; both start from the description independently, and
the generated reference must pass the judge -- that is the check that the two "agree".

### 3.4 Measurement

- Every run measures afresh; the results are stored in the run's operator tree, never cached across
  runs.
- NCU signatures of each kernel are collected when possible. When NCU is unavailable on the machine
  (no permission to collect performance counters, for example), the run **degrades**: it advances on
  timings alone, and the steps that depend on NCU (the signature anchors of the milestone's mechanism
  lines, per-round NCU profiling, stall / timeline pricing) are skipped and noted in the report (see
  §14, item 5).
- When the reference is to be timed beside the student's kernel in the same job, the harness seals it
  automatically (`eval/closed_ref.py`), so no plaintext appears anywhere the student can reach.

### 3.5 The deny list is derived

The student (and, in blackbox mode, the teacher) may not read: the source under `reference/`, the
reference's sealed artifacts and key, other run directories. The user can only add to this through
`[reference].deny`, never remove from it.

---

## 4. Judging correctness

1. **Who writes it**: the teacher, in a session of its own before distillation. It writes three things:
   the PyTorch judge (a direct implementation of the reference's semantics), a per-shape input
   generator (respecting the legal ranges in `description.md`), and the input groups that bring every
   shape with fewer than 3 up to 3 (§2.3).
2. **How it is checked**: with a reference, the judge is compared with the reference's outputs on every
   input group of every shape; on a mismatch the teacher fixes it, given the numbers, and after 3
   failed attempts the run stops with an error.
3. **Frozen**: written into `.artifacts/<op>_<id>/check/`; the student may read it but not write it;
   the harness's tamper check covers it.
4. **Tolerance**: for each output, **twice** the **maximum** of the reference's error against the judge
   over all checked inputs (another correct kernel sums in another order and lands as often above the
   reference's error as below it; on a fresh seed the reference itself can land above it), with a
   floor set by the output dtype at twice that type's rounding error (relative):
   bf16 ≈ 7.8e-3, fp16 ≈ 9.8e-4, fp32 ≈ 1.2e-7. When the reference matches the judge bit for bit, the
   student is not held to bit-for-bit agreement.
5. **The judge and the reference are independent**: generating the reference (§3.3) and writing the
   judge are two separate sessions.
6. **The correctness monitor at the finish**: on the confirmed best kernel, rerun the judge with new
   random seeds and stressed inputs (scaled up, scaled down, spikes, zero blocks, identical rows,
   integer edges), and check the outputs for NaN / inf and whether two runs on the same inputs agree.
   It only records and never changes a verdict; the result goes into the confirmation record (see
   [docs/correctness-monitor.md](../correctness-monitor.md)).

### 4.1 Leaderboard tasks

`forge init <op> --leaderboard flashinfer-bench|sol-execbench …` turns one problem of a public
leaderboard into `tasks/<op>/`. The judge, the input generation and the timing basis of such a task
**come from the board itself** (pinned to an upstream commit) instead of steps 1–4 above, and after the
run `forge export` exports a solution JSON in the board's format. Each workload on the board is one
shape here, and the input groups **follow the board's evaluation protocol exactly** (the number of
groups, the data, the tolerance all as the board has them, not completed to 3); choosing several
workloads makes a multi-shape task. Everything else (the reference, the target, the agents, the budget,
the run directory, the delivery) is as for an ordinary task. Details are in
[docs/leaderboards.md](../leaderboards.md).

---

## 5. Target and timing

- The target: `ratio` (1.2 by default, i.e. 20% faster than the reference) or `ms`, one or the other.
- The timing basis: `device` (pipelined batches, close to device time) or `e2e` (one call at a time
  plus a synchronize, host overhead included).
- Which inputs and outputs the timed window contains is derived by the harness from `[reference].run`
  and the judge's signature; the user does not write `window.toml`.
- Correctness: every input group of every shape must pass; the stressed inputs of the finish's monitor
  are judged only, never timed.
- Performance: within each shape, the geometric mean over its input groups, then the geometric mean
  across the shapes; the target is judged on that overall reading. At least 3 input groups per shape
  keep a kernel from being tuned to one set of data (leaderboard tasks excepted: they follow the
  board, see §4.1).

---

## 6. Agents

- One table each for the teacher and the student, with structured fields: `backend`, `provider`,
  `model`, `effort`, `timeout_s`. The harness builds the command itself, so it can handle /goal
  sessions, session continuation, usage accounting and permission settings uniformly.
- V1 supports only `backend = "claude"` (the Claude Code CLI); the field stays so that adding a backend
  later does not change the format.
- With `provider = "native"`, the CLI's own login state and environment decide where the tokens go;
  any other provider is an endpoint in `global_config/backends.toml`, whose key is read from the
  variable its `key_env` names.
- Credentials are read only from the environment or from `local.env`; they never appear in any file
  under version control.
- Without a `model`, `claude-opus-5-5` is used; the model actually used is written into the run's
  snapshot.

---

## 7. Budget and stopping

The whole budget is in `task.toml`: `hours` (the inner loop's wall clock) and `max_meta` (outer rounds).
The stop conditions are the loop's own: the target is reached, the budget is spent, the outer rounds are
used up, or the teacher judges the target unreachable and the reachability check confirms it.

---

## 8. The run directory: `.artifacts/<op>_<id>/`

```
.artifacts/<op>_<id>/
  run.json        what was launched: the task, its snapshot digest, the harness version, the machine, the run's status
  task/           the task definition as it was at launch (every later step reads this copy)
  check/          the judge the teacher wrote: definition.json, judge.py, groups.json
  operators/<op>/        the operator tree (a git repository): the judge and the clock, the kernel stub,
                         gpu_entry.sh, the milestone, the round records (student/status.jsonl,
                         meta_status.jsonl), the reference's measurements
  operators/_<op>ref/    with a reference: the recon tree (the sealed reference), the teacher's
  ws/             the student's workspace (a clone of the operator tree, the only place the student may write)
  best/           the best kernel
  run.log         the run's output
  report.md       the final report
```

- A new run = a new directory; it inherits nothing from any earlier run.
- Resuming: run `forge run .artifacts/<op>_<id>` again; the steps already done are skipped, and the loop
  resumes from its own records; the task definition is the snapshot's.

---

## 9. Where it runs

| Delivery | What it does |
|---|---|
| `local` | the GPU is on this machine; jobs run directly |
| `ssh` | each job's code tree (`git archive`) is sent over ssh to a GPU machine, unpacked and run there, and the results are brought back |

The machine is described in `local.env` at the repository root (not in git), never in the task
definition:

```
FK_DELIVERY=ssh             # local | ssh
FK_GPU=0                    # which GPU
REMOTE_HOST=gpu-box         # an ssh alias that logs in non-interactively
REMOTE_WORKDIR=/data/fk     # a directory on the host used only by forgekernel
REMOTE_ENV_FILE=            # optional: a script sourced before every command
ANTHROPIC_API_KEY=...
```

- **Pre-start check**: before the loop, a very short NCU probe job runs on the target machine to see
  whether performance counters can be collected (§14, item 5).
- **GPU lock**: one job per GPU at a time; both deliveries take a lock on the machine that holds the
  GPU, so jobs never time each other.
- **Clock locking is optional**: locked when possible (it needs permission; `LB_SM_CLOCK_MHZ` /
  `LB_MEM_CLOCK_MHZ`), and whether it held is recorded with every evaluation.

---

## 10. The command line

```
forge init <op>                    # copy tasks/_template/ to tasks/<op>/
forge init <op> --leaderboard …    # a task from one problem of a public leaderboard (§4.1)
forge new tasks/<op> [--dry-run]   # create a run (.artifacts/<op>_<id>/); --dry-run only checks and prints the plan
forge run .artifacts/<op>_<id>     # run it; a run that has started continues where it stopped
forge status [.artifacts/<op>_<id>] # progress (every run when omitted)
forge harvest .artifacts/<op>_<id> # the end-of-run knowledge-base write-up of a stopped run
forge export .artifacts/<op>_<id>  # the best kernel: a leaderboard task's as the board's solution JSON, any other task's as its source
```

`forge new` takes only `--dry-run`, and `forge run` only the run directory; everything about the
problem comes from `task.toml`. `forge run` exits 0 when the run ends at its target and 2 on any other
stop.

---

## 11. The self-evolution script

A script outside the core that uses only the public commands above:

```
python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]
```

Each generation:

1. `forge new` + `forge run`, then read the confirmed best kernel and its speedup over the reference;
2. stop when there is no confirmed best kernel, or its gain over the reference is within the noise
   floor: the spread of the generation's same-job confirmation ratios ((max - min) / min), never
   below 1.5% (the loop's plateau margin);
3. otherwise write the next generation's task `tasks/<op>-g<k>/`: copy `task.toml`, `description.md`,
   `shapes.json` (and `inputs/`); **copy** the previous generation's best kernel into the new task's
   `reference/` (the new task is self-contained, can be committed and rerun elsewhere); target
   `ratio = --step` (1.05 by default);
4. no generation cap by default; `--max-gen` sets one.

The new reference has source, so by §3.1 it runs in source mode; sealing it for same-job pairing is the
harness's job (§3.4), not the script's.

---

## 12. What was taken off the launch surface

| Category | What | Where it went |
|---|---|---|
| tuning knobs (about 40) | pricing rounds, verdict experiments, reopen, repricing, watermark confirmation, the re-anchoring band, the timeline probe, round caps, ... | fixed inside the harness, invisible and not configurable; values set to the configurations that actually ran well |
| switches that bypass checks | `--allow-*`, `--skip-*`, `--no-m0-audit` | off the launch surface; the harness satisfies the checks it needs |
| hand-written things that can be derived | `window.toml`, the deny list, the job entry script, `--kernel-file`, the branch prefix, the git identity, the reference's description fields | generated or derived by the harness |
| a second reference | the library baseline (`ref_cmd` / `ref_baseline.json`) | merged into the one reference |
| reference-generation options | `ref_from_goal = always/off`, `ref_from_goal_*` | off the launch surface; a reference is generated automatically when there is none, and only the budget `[budget].reference_hours` remains |
| run-time overrides | `--hours`, `--equivalence-ratio`, `--target-ms`, `--language`, ... | only in `task.toml` |
| internal infrastructure | cluster job submission and related machinery | removed |
| automatic KB publishing | publishing certified entries automatically at the finish | removed; the harvest stays: the report lists newly certified entries, and the user reviews and commits them |
| composite operators | chaining several operators | not in V1 |
| separate development / acceptance shapes | developing on one shape and accepting on another | removed (the geometric mean over several shapes covers it) |
| the word "workload" | in an ordinary task it meant one set of inputs of a shape, in a leaderboard task one shape | unified as shape and input group (§2.3); an ordinary task has no correctness-only shapes (for a leaderboard task's `--check-workload`, see §14, item 11) |

---

## 13. Changes to the code

| Module | Before | Now |
|---|---|---|
| `scaffold.py` / `forge init` | built a standalone git repository under `operators/<op>/` with copies of the eval template, the job entry and the prompts | `forge init` creates the three template files of `tasks/<op>/`; eval, entry and prompts belong to the harness (`taskcard.py` generates the operator trees inside the run directory); `scaffold.py` remains as the leaderboard card generator only (§15) |
| `loop/cli.py` | 43 arguments + about 50 `loop.toml` keys, filled in by the user | `forge run` reads `task.toml`, creates the run directory, generates `loop.toml` and calls the loop with fixed arguments |
| `baseline.py` / `baseline_cli.py` | two references, measured once and cached forever in the operator repository | one reference, measured on every run, stored in the run directory |
| `loop/goal_ref.py` | three settings: auto / always / off | only auto for ordinary tasks; its products land in the run directory |
| new: the judge step (`judge.py`) | -- | the teacher writes the PyTorch judge and the input generation; compared with the reference, frozen, tolerance set |
| `window_contract.py` | read the `window.toml` the user wrote | derived and generated from the reference's call and the judge's signature |
| `loop/agent_policy.py` | read the deny list from hand-written `ref_source_paths` etc. | derived automatically + `[reference].deny` |
| `submit/` | job submission to an internal cluster | two executors, `local` and `ssh`, + the GPU lock |
| agent access | raw command strings | an adapter that builds the command from `backend` / `provider` (`agents.py`, `scripts/agent-claude.sh`) |
| resuming (`launch.py`, `loop/cli.py`) | assumed "the operator repository is one git repository" | relies on the run directory (`run.json` flags, the loop's own records). `resume/` itself is the milestone-DAG checkpoint invalidation of prefix reuse and is unchanged |
| `loop/driver.py` | refused to start without NCU signatures | degrades when NCU is unavailable (§3.4); otherwise largely unchanged, only where the `LoopContext` fields come from changed |
| KB publishing | certified entries published automatically | removed; new entries are only listed in the report and the terminal |
| composite operators | supported | out of V1 |
| NCU profiling, pricing, the timeline (`eval/ncu_run.py`, `loop/meta_shapes.py`, ...) | profiled only representatives; with several shapes, pricing used only the representative + the smallest + the largest | every shape × every input group |
| new: `scripts/self_evolve.py` | generations scheduled outside the repository | a standalone script in the repository |

---

## 14. Implementation record

Trade-offs made in implementation, and where the result departs from the text above:

1. **Judging reuses the leaderboard path**: a third suite, `forgekernel`, sits beside FlashInfer-Bench
   and SOL-ExecBench (`src/forgekernel/card_template/eval/leaderboard.py`); bench generation,
   `gpu_entry.sh`, the timed-window contract, reference sealing and tamper checks are all reused. The
   error is the "normalised max error" `max|y − j| / max|j|` (over finite positions); NaN / Inf must sit
   at the same positions as the judge's, and integer and bool outputs must match exactly.
2. **When the judge and the reference "agree"**: every output's normalised error is at most 5%
   (`TASK_MATCH_CAP`). Above that they do not agree, and the judge goes back to the teacher with the
   numbers to rewrite, at most 3 times; without a cap, a wrong judge would be absorbed into the
   tolerance.
3. **The order without a reference**: the teacher writes the judge from the description first, then
   /goal writes the reference, which must pass the judge; the two come from different sessions, each
   starting from the description, and passing the judge is their "agreement". The tolerance is the
   dtype floor. The generated reference serves as a constant target, as `goal_ref` does it (it is not
   timed beside the kernel in the same job).
4. **The run directory's actual layout**: see §8. The milestone, the round records and the reference's
   measurements are in the operator tree, where the loop has always kept them.
5. **NCU degradation**: every launch first runs a very short probe job on the target machine
   (`eval/ncu_probe.py`); when no counters can be collected, the generated loop configuration says
   `ncu_available = false`, the reference no longer needs NCU signatures, the timeline / stall pricing /
   orchestration census are off, and the job-environment pre-check only warns.
6. **Every shape × every input group**: each round's NCU window spreads its calls over every input
   group of every shape (the sampling cap grows accordingly); tools that read shape by shape, such as
   pricing, stall and the timeline, read `meta_workloads`, which now lists every shape.
7. **Agent commands**: built in one place, `scripts/agent-claude.sh`; an ordinary turn is `claude -p`
   under a wall-clock timeout, and when /goal is needed (`FK_AGENT_DRIVER=goal`) it hands over to
   `scripts/claude-goal.py`. A /goal session needs tmux.
8. **Self-evolution**: `scripts/self_evolve.py`. Only `cutedsl` tasks for now (the next generation's
   reference is sealed as Python source).
9. **Leaderboard tasks**: `forge init <op> --leaderboard …` generates the card (and the recon tree of a
   sealed reference) with the existing generator, freezes it under `tasks/<op>/leaderboard/`, and
   `task.toml` points at it with a `[leaderboard]` table; `forge run` skips writing and calibrating a
   judge, and takes the loop configuration from the card's own suggestion (`loop.toml.example`), filled
   in with this run's hardware, deny list and NCU availability; a leaderboard task never has /goal write
   a reference first.
10. **`[task].arch` is not checked against the machine**: it decides the compile target
    (`CUTE_DSL_ARCH`) and the KB directory, but the harness does not check that the GPU machine's GPU is
    of that architecture.
11. **`--check-workload` remains for leaderboard tasks**: `forge init --leaderboard` passes it through
    to the card generator, and those workloads are judged for correctness only, never timed. Ordinary
    tasks have no such concept.
12. **Measured on B200** (three GPU jobs): calibration, the reference self-check, judging and paired
    timing of a student kernel through the local delivery executor, NCU, the NCU probe and the recon
    tree's reference baseline all ran, and the outputs were checked by the driver's own parsers. Two
    fixes came out of it: task input files are parsed by torch itself (the GPU machine may not have the
    safetensors package); local delivery marks the directory as safe when calling git inside a
    container.
13. **A real-agent end-to-end run** (RMSNorm, prefill 8192×4096 + decode 64×4096, bf16, source mode,
    `hours = 1`, a remote B200, ssh delivery): one `forge run` went through writing the judge (static
    checks passed on the first round) → calibration (the tolerance landed on the bf16 floor) → the NCU
    probe → the reference baseline → 9 teacher reconnaissance jobs → milestone v1 → a single fused kernel
    in the student's round 0 → KEEP → confirmation on the same GPU at 5.24× (the 1.2× target reached,
    stop) → the correctness monitor → the report, in 84 minutes in all. It led to: the report writes the
    best kernel into the run directory's `best/`, and `forge export` emits the kernel source for an
    ordinary task. Also from that run: `hours` counts from the end of the reference baseline, so it
    includes the teacher's reconnaissance (44 minutes this time) — now stated where the budget is
    configured; the reference was measured 3 times as the architecture reference and 3 more as the ref
    baseline — the loop now reuses the first reading; the correctness monitor's stress profiles had no
    effect on task trees (the judge's `make_inputs` was not counted as bench code, and the kernel loaded
    by file path never reached the hook) — both fixed. Still open: without CuTe DSL installed on the
    machine that runs `forge`, the student cannot read the DSL source (this did not stop it from writing
    a working kernel here).
14. **The execution layer**: jobs are recorded by `forgekernel.submit.jobs` (one detached runner per
    job, the log and the verdict under `operators/<op>/.jobs/<id>/`); a lost runner is resubmitted, and
    a transport failure or a clock lock that does not hold is reported as "did not run" and never
    charged to the kernel; stopping a job ends its whole process tree on this machine or the ssh host
    (on the ssh host a job runs in its own session). The agent policy forbids the student to ssh to the
    GPU machine or submit jobs itself.

---

## 15. Simplification record (branch `simplify-harness`)

The code behind the settings §12 and Appendix A declare gone was still in the tree; this pass
removed it, so that the surface and the code agree:

| Removed | Was |
|---|---|
| `forge init <op> --kernel-file ...` and `scaffold.new_operator` | the pre-V1 standalone operator repository; `scaffold.py` is now the leaderboard card generator only, and `taskcard.copy_template` copies the four template files every tree starts from |
| `card_template/window.toml`, `card_template/submit.toml`, the `agent/*.md` prompt copies | overwritten by the generator before any job (window.toml, submit.toml) or read by nothing (agent/); the old 660-line card README that documented them is replaced by a description of a generated tree |
| `forge <op> baseline`, `forge <op> window`, `forge <op> monitor`, `forge <op> leaderboard show\|export`, the `forge <op> submit\|status\|logs\|stop` fall-through | human-only wrappers around functions the loop calls directly; `forge export` and `python -m forgekernel.submit` remain |
| 29 of the 40 `forge <op> loop` arguments, `--dry-run`, and the loop.toml reads of the fixed knobs (`watermark_margin`, `reanchor_band`, `reopen_*`, `reprice_*`, `lever_pricing_*`, `declared_gate_close`, `reuse_unchanged_reading`, `confirm_jobs`, `correctness_monitor*`, `infeasible_min_diagnoses`, `ref_price_pm_interval`, `simulator_dir`) | A.1 / A.2 "fixed defaults": now constants at the top of `loop/cli.py`; the loop reads from loop.toml only what `forge run` writes (`language`, `kb_arch`, `gate_window`, the reference keys, the NCU-availability switches) |
| `--no-oracle` / `oracle_mode = "none"`, `ref_from_goal = "always"`, `ref_from_goal_ratio / _accept_runs / _as_target` | modes and knobs no task can express |
| `loop/agentctx.py` (the 200k context-window gate) and the static scan for a goal-capable agent command | read the command the harness itself builds (`agents.agent_command`); `agent_policy.command_findings` keeps the one scan that still protects something (the project settings must load) |
| `[loop] agent_allow_write`, `source_teacher_denies_ref`, `dsl_python`, the `DSL_PIN` version gate | keys nothing writes (`write_submit` never wrote `DSL_PIN`) |
| `inner/results.py`, `inner/workspace.py`, the `cublas_*` legacy key remap, `milestone.is_valid`, `loop/__init__.py` re-exports, duplicate `_repo_root` / `_toml_value` helpers | dead or duplicated |

Left in place on purpose, each a behaviour change the loop would feel and that a CPU test suite
cannot certify (candidates for a later pass, in order of expected gain): the two-reference
machinery (`arch_baseline.json` and its `ref_baseline.json` copy, `{ARCH_REF}` / `{REF_BASELINE}`
bindings); the `window.toml` asymmetry / prep / pricing code, inert on generated contracts (both
sides always share one boundary); the absolute-target and cross-machine machinery, inert while the
reference rides the job (`ref_in_job = true`), live for generate-mode tasks; the timeline probe,
off for ordinary tasks and on for leaderboard cards; the precision gate, off on every generated
tree; the `meta/cuda/` prompt overlays, ~95% identical to the shared prompts; `binref` (the sealed
binary reference of `--binary-ref` leaderboard cards).

---

## Appendix A: where the old settings went

The tables below describe the user surface at the level of `forge run`. `forge <op> loop` takes
only what `forge run` passes it (the workspace, the budget, the target, the two agent commands,
the resume flags); the rows below that say "fixed defaults" are constants in `loop/cli.py`.

### A.1 `forge <op> loop` arguments

| Argument | Where it went |
|---|---|
| `op`, `--workspace`, `--milestone` | decided by `forge new tasks/<op>` and the run directory |
| `--oracle`, `--oracle-mode`, `--no-oracle` | `[reference].source` / `blackbox`; ordinary tasks have no "no distillation" mode |
| `--ref-from-goal` | automatic |
| `--ref-from-goal-hours` | `[budget].reference_hours` (default 8) |
| `--language`, `--kb-arch` | `[task].language` / `arch` |
| `--hours`, `--max-meta` | `[budget]` |
| `--target-ms`, `--equivalence-ratio` | `[target].ms` / `ratio` |
| `--gate-window` | `[target].timing` |
| `--agent-cmd`, `--teacher-agent-cmd` | `[agent.student]` / `[agent.teacher]` |
| `--kernel-file`, `--branch-prefix`, `--git-name`, `--git-email` | derived or fixed by the harness |
| `--ref-repeat`, `--agent-heartbeat-s`, `--kb-tripwire-rounds`, `--confirm-jobs`, `--max-probe-rounds`, `--max-verdict-probe-rounds`, `--min-meta-before-infeasible`, `--lever-pricing-rounds`, `--max-consecutive-reverts`, `--no-m0-audit` | fixed defaults |
| `--allow-host-bound-ref`, `--allow-200k-window`, `--allow-preexisting-artifacts`, `--allow-dirty-kb`, `--skip-kb-lint`, `--allow-undeclared-window`, `--allow-single-workload` | off the launch surface (on resume the harness passes `--allow-preexisting-artifacts` and `--allow-dirty-kb` itself) |
| `--skip-chain-contract-check` | removed (composite operators are out) |
| arguments for internal infrastructure | removed |
| `--harvest-only` | became `forge harvest .artifacts/<op>_<id>`: the end-of-run knowledge-base write-up of a stopped run |
| `--dry-run` | kept, on `forge new` |

### A.2 `loop.toml [loop]`

| Key | Where it went |
|---|---|
| `language`, `kb_arch`, `gate_window`, `oracle`, `oracle_mode` | `task.toml` |
| `ref_cmd`, `ref_ncu_cmd`, `ref_mode`, `arch_ref_mode`, `ref_in_job` | merged into `[reference].run` |
| `ref_submit_op`, `arch_ref_submit_op`, `recon_op`, `ref_price_op`, `ref_price_app` | the harness builds the reference tree inside the run directory |
| `arch_ref_ms`, `arch_ref_device_ms` | measured on every run for an ordinary task (a leaderboard card declares them, see [docs/leaderboards.md](../leaderboards.md)) |
| `ref_name`, `arch_ref_name`, `ref_shape`, `ref_hardware` | recorded automatically |
| `ref_source_paths`, `ref_source_urls`, `source_teacher_denies_ref` | derived automatically + `[reference].deny` |
| `dsl_source_dirs`, `cuda_source_dirs`, `dsl_python`, `job_env_files`, `simulator_dir` | machine-specific: into `local.env`, or detected automatically |
| `ref_from_goal_hours` | `[budget].reference_hours` (default 8) |
| `ref_from_goal`, `ref_from_goal_ratio`, `ref_from_goal_accept_runs`, `ref_from_goal_as_target` | fixed internally |
| `watermark_margin`, `reanchor_band`, `timeline_probe`, `stall_pricing`, `orch_census`, `reuse_unchanged_reading`, `reprice_on_new_best`, `reprice_min_gain_pct`, `reopen_requires_price`, `reopen_max_fruitless`, `reopen_in_place`, `ref_price_pm_interval`, `lever_pricing_rounds`, `lever_pricing_max_levers`, `infeasible_min_diagnoses`, `declared_gate_close`, `confirm_jobs`, `correctness_monitor`, `correctness_monitor_profiles`, `kb_lint_oracle_tokens`, `job_env_preflight`, `harness_guard` | fixed defaults |
| `agent_allow_write`, keys for internal infrastructure | removed |

`loop.toml` itself still exists: `forge run` generates it for every run; the user neither writes nor
edits it.

### A.3 `submit.toml` and `window.toml`

| File / table | Where it went |
|---|---|
| `submit.toml [job]` (an internal cluster's job parameters) | removed; only the job timeout remains, set by the harness |
| `submit.toml [repo]` (delivery, branch, entry script) | not written by the user; the delivery comes from `FK_DELIVERY`, and the code tree and entry are managed by the harness |
| `submit.toml [env]` | derived from `[task].arch`, or written into `REMOTE_ENV_FILE` |
| `submit.toml` as a whole | generated by `forge run` from `local.env` on every launch; the user neither writes nor edits it |
| `window.toml` | derived and generated by the harness; the user neither writes nor edits it |
| environment variables of internal infrastructure | removed |
