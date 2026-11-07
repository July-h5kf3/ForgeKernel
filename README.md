<div align="center">

# ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents

**English** | [中文](README_zh.md)

[![Quick Start](https://img.shields.io/badge/Docs-Quick%20Start-blue.svg)](#quick-start)
[![Design](https://img.shields.io/badge/Docs-Design-blue.svg)](docs/design/v1-launch.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

<img src="docs/assets/figure1-self-evolution.png" width="90%" alt="The two mechanisms of self-evolution: promote each accepted kernel to the next reference, and forge certified findings into a knowledge base">

</div>

ForgeKernel is a **self-evolving GPU kernel agent** that starts from a **reference kernel**.
It reads the reference's architecture by running, timing and profiling it, turns that into an
architecture-first milestone curriculum that a coding agent implements, and **promotes every
accepted kernel to the next reference**, so optimization keeps going past the best known
implementation. Findings certified by a hardware probe accumulate in a
**knowledge base shared across operators**. In the paper, across 27 operators on Hopper and
Blackwell, ForgeKernel matches or beats **18 of 22 expert-library kernels** (FlashAttention-4,
cuDNN, DeepGEMM and others) and **all 7 published kernels** of three kernel-generation systems,
by up to **+170.6%** (+39.4% geometric mean). The kernels are released in [`kernels/`](kernels/).

ForgeKernel is part of the **Forge family**, together with
[ForgeTrain](https://github.com/OpenBMB/ForgeTrain) and
[ForgeStencil](https://github.com/OpenBMB/ForgeStencil).

## News

- 📄 **[2026-10]** JustGoal paper release: [arXiv:2610.36434](https://arxiv.org/abs/2610.36434).
- 🏆 **[2026-10]** We reached **#1** on the FlashInfer-Bench leaderboard of NVIDIA
  [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench).
- 🏆 **[2026-10]** We reached **#1** on Huawei CANN-Bench (Ascend A3).
- 🏆 **[2026-10]** We also reached **#1** on Ascend 950DT on the
  [CANN-Bench leaderboard](https://cannbench.com/leaderboard).

## TODO

Planned for **ForgeKernel v1.5**:

- [ ] **Shape clustering**: group the shapes of a task into clusters and optimize per cluster,
  so one run can cover a wide range of shapes.
- [ ] **A much simpler harness**: fewer moving parts in the loop and a smaller configuration
  surface.
- [ ] **One repository with [ForgeTrain](https://github.com/OpenBMB/ForgeTrain) and
  ForgeMegaKernel** (in progress).

## Highlights

- 🔁 **Self-evolution: always one step past the reference.** Every accepted kernel becomes the
  next run's reference, so each generation has to beat the one before it, and the chain stops
  only when the hardware gives no more (a gain below the noise floor). In the paper
  every promoted reference was beaten again, by up to **+9.9%** a generation, including `/goal`
  agent kernels already far ahead of the expert library: GDN prefill ends **+100%** over
  FlashInfer, GQA attention forward **+29%** over FlashAttention-4 (`scripts/self_evolve.py`).
- 🗺️ **Architecture-first curriculum: the error-prone decisions are settled first.** The decisions
  that are easiest to get wrong and most expensive to change later, the architecture (kernel
  split, tile shapes, warp roles, register and shared-memory budget) and the orchestration (what
  stays resident, the issue order, which work hides under which), are all made in the first
  milestone, **M0**, and frozen once its gate passes. Later milestones `M1 → Mk` only tune inside
  that structure, each closed by a gate of correctness, performance and mechanism lines. When
  measurement shows a limit only a new structure can lift, the arch agent goes back and revises
  M0 instead of letting later milestones work around it.
- 📚 **A knowledge base that outlives the run.** Hardware mechanisms certified by a GPU probe are
  stored per GPU architecture (`hw_kb/`) and prescribed to later operators.

## 🤖 Set it up with a coding agent

Clone this repository, then paste this into Claude Code (or any coding agent that can run shell
commands) started in the clone:

```text
Read README.md in this directory and set up ForgeKernel on this machine:
install it in a virtual environment, write local.env for my GPU (this
machine, or the ssh host I name) and run `forge new tasks/rmsnorm --dry-run`.
Show me the plan it prints.
```

## Quick Start

**Requirements.** The machine you run `forge` on: Linux or macOS, Python ≥ 3.11, git, and the
[Claude Code](https://docs.claude.com/en/docs/claude-code/overview) CLI (logged in, or an
`ANTHROPIC_API_KEY`); tmux if a task has no reference. The GPU machine (the same one, or an ssh
host): an NVIDIA Hopper (`sm90`) or Blackwell (`sm100`) GPU, `python3` with a CUDA build of
PyTorch and, for CuTe DSL tasks, `nvidia-cutlass-dsl`; Nsight Compute (`ncu`) is recommended,
and without it a run proceeds on timings alone.

**1. Install**

```bash
cd ForgeKernel                   # a clone of this repository
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install nvidia-cutlass-dsl   # recommended, see below
forge --version                  # forgekernel 0.1.0
```

The coding agent works offline; installing `nvidia-cutlass-dsl` next to ForgeKernel is how it
reads the CuTe DSL API it writes against.

**2. Say where the GPU is**

```bash
cp local.env.example local.env
# FK_DELIVERY=local -> the GPU on this machine (FK_GPU picks the device)
# FK_DELIVERY=ssh   -> REMOTE_HOST (an ssh alias that logs in without a
#                      prompt) and REMOTE_WORKDIR (a directory on it)
```

For **local agents with an SSH workspace**, set `FK_WORKSPACE_HOST` and `FK_WORKSPACE_DIR`
in the local `local.env`. ForgeKernel, Orca, Codex and Grok all run and authenticate locally.
The server needs only Bash, stdlib Python and the GPU evaluation environment, without agent
installations or logins. `FK_WORKSPACE_ENV_FILE` optionally sources that remote GPU environment;
`FK_WORKSPACE_PYTHON` selects the file-transfer interpreter (default `python3`). GPU delivery
automatically uses the workspace host; `FK_GPU` selects its device.

CLI commands and paths, including `export --out`, stay local. Working copies live under
`.artifacts/<operator>/<run>/`; remote files are grouped as `<FK_WORKSPACE_DIR>/<operator>/<run>/`
with `state/` for snapshots/reports, `ws/` for the agent's code mirror, and `jobs/<tree>/` for
isolated GPU jobs. Successful turns synchronize code over SSH; run entry and exit synchronize
state. Logs and verdicts return to the local controller. Git metadata, agent configuration,
`local.env` and `*.local.toml` are excluded. Only controller-marked mirrors may be replaced.
A sync failure preserves local files and stops without repeating an already successful turn.

Teacher and student choose their harness and model independently. Each role supports `claude`,
`codex`, or `grok`, including the same harness with different models or reversed roles.
`backend` selects the harness; `model` is forwarded unchanged to its CLI without a ForgeKernel
model allowlist. Model availability depends on that CLI and account. One example is:

```toml
[agent.teacher]
backend = "codex"
model = "default"
effort = "high"

[agent.student]
backend = "grok"
model = "default"
```

`default` uses the CLI's model default for all three harnesses. Changing `backend` resets an
unspecified `model`/`provider` to `default`/`native`, avoiding another harness's inherited model
or endpoint. `effort` and `timeout_s` can also differ by role; task role settings override task
common settings, which override global settings. Codex and Grok currently require
`provider = "native"`. Install and authenticate only the selected CLIs locally. Codex and
Grok require a local Orca runtime (`orca-ide` on Linux, or `ORCA_CLI_COMMAND`); Claude keeps
its existing direct CLI launcher.
`FK_ORCA_WORKSPACE` optionally selects an existing Orca workspace; otherwise the checkout is
registered as a folder. Orca owns each headless turn's terminal; ForgeKernel passes prompts and
collects final responses through private files, enforcing timeouts and closing only its own terminal.
The existing Claude backend remains available. Codex uses a named permission profile; Grok uses
a custom sandbox (Landlock and bubblewrap on Linux). Restricted Grok turns on macOS expose only
`read_file`, `search_replace`, `grep` and `list_dir`; shell, web and subagents are unavailable,
and the controller handles compilation, tests and SSH. Blackbox URL restrictions
disable network and web search in the native adapters. Uncertain launch/cleanup outcomes stop retries
and preserve the private turn directory for inspection.

**3. Run the example task**: fuse a PyTorch RMSNorm (nine kernel launches a call) into one CuTe DSL
kernel on a Blackwell GPU (set `arch = "sm90"` in `task.toml` for Hopper).

```bash
forge new tasks/rmsnorm --dry-run                       # check the task, print the plan
forge new tasks/rmsnorm                                 # -> .artifacts/rmsnorm_<id>
tmux new -s forge 'forge run .artifacts/rmsnorm_<id>'   # a run takes hours
forge status .artifacts/rmsnorm_<id>                    # progress
forge run .artifacts/rmsnorm_<id>                       # continue after an interruption
```

`forge run` exits 0 when the run reaches its target -- the loop stopped on it, or the final
confirmation of the best kernel meets it -- and 2 otherwise.

**Expected output.** The run writes the correctness judge, calibrates it against the reference,
measures the reference, distils a milestone, and iterates until the kernel is 1.2× the
reference or the budget ends. It finishes with `.artifacts/rmsnorm_<id>/report.md` and the best
kernel in `.artifacts/rmsnorm_<id>/best/kernel.py`. On a B200 our run of this example stopped at **5.2×** the
PyTorch reference after about 1.5 hours:

```text
- task: rmsnorm (source mode)
- best: 77856a9d9561 -- device median 0.0307 ms, 5.2363x the reference
- kernel: best/kernel.py
```

> Full launch reference → [docs/design/v1-launch.md](docs/design/v1-launch.md)

## How It Works

<div align="center">
<img src="docs/assets/figure2-architecture.png" width="90%" alt="ForgeKernel architecture: the arch agent analyses the reference and writes milestones; the coding agent implements one decision row at a time; gates retry, re-analyse or ship">
</div>

A run has two nested loops. In the **inner loop** the coding agent implements one decision row of
the milestone at a time, and one GPU job judges correctness, times the candidate against the
reference and profiles it; a correctness failure is reverted. In the **outer loop** the arch
agent re-enters after iterations without a new best, at an exhausted decision graph or at a gate
impasse: it prices levers, runs verdict probes and revises the curriculum. A shipped kernel can
become the next run's reference, and what probes certify feeds the knowledge base.

Three artifacts carry the work, and their boundary is what keeps the reference's code out of the
kernel: the per-operator **milestone** (decisions, parameters and contracts, never an
implementation body), the **hardware knowledge base** (operator-agnostic mechanisms, the only
artifact with code), and the **meta-milestone** ([`meta_milestone.md`](meta_milestone.md)), the
distillation policy. The paper's terms map onto the code as follows:

| Paper | In this repository |
|---|---|
| arch agent | the *teacher* (`[agent.teacher]` in `task.toml`) |
| coding agent | the *student* (`[agent.student]`) |
| campaign | a run, `.artifacts/<op>_<id>/` |
| knowledge base | `hw_kb/cutedsl/` (CuTe DSL), `hw_kb/cuda/` (CUDA C++) |

> Design notes → [docs/design/milestone-distillation.md](docs/design/milestone-distillation.md),
> milestone format → [docs/milestones-template.md](docs/milestones-template.md)

## Writing a Task

`forge init <op>` copies `tasks/_template/` to `tasks/<op>/`, three files to fill in:

- `description.md`: the operator, completely (signature, semantics, legal inputs). The teacher
  writes the correctness judge from this file alone.
- `shapes.json`: the shapes to optimize for; every shape and every input group is judged and timed.
- `task.toml`: language (`cutedsl` or `cuda`), GPU architecture, the reference, the target, the
  models, the budget.

The reference decides the mode:

| `[reference]` | Mode | What the teacher sees |
|---|---|---|
| `run` + `source` | `source` | the reference's source and its hardware behavior |
| `run` only, or `blackbox = true` | `blackbox` | hardware behavior only |
| (none) | generate | a `/goal` session writes a reference first |
| `[leaderboard]` instead | leaderboard | the board's problem, judge and clock |

A task writes only what differs from `global_config/`, the bottom layer every task is merged
onto, as in ForgeTrain: `models.toml` sets `[agent]` (`backend`, `provider`, `model`, `effort`,
`timeout_s`; `[agent.teacher]` / `[agent.student]` for one agent), `loop.toml` sets `[target]`
and `[budget]`, and `backends.toml` lists the API endpoints a `provider` other than `native`
can name, with the key read from `local.env`. The defaults are a 1.2× target in device time,
4 hours of loop, 3 teacher rounds, and `claude-opus-5-5` for both agents (the paper's runs used
Claude Fable 5.1 as the arch agent and Claude Opus 5 as the coding agent).

## Self-Evolution

```bash
python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]
```

Each generation is an ordinary run (`forge new`, then `forge run`). When it ends with a confirmed
best kernel faster than its reference by more than the noise floor, the script writes the next
generation as a new task `tasks/<op>-g<k>/`: the same `task.toml`, `description.md` and
`shapes.json` (and `inputs/`), the best kernel copied into its `reference/`, and a target of `--step`
(default 1.05) times the new reference's speed. Each generation's task is self-contained, so it can be
committed and rerun elsewhere. The chain stops only when the hardware gives no more: at a generation
whose gain is within the noise floor (the spread of that generation's same-job confirmation ratios,
at least 1.5%), or at one that ends without a confirmed best kernel. There is no generation cap
unless you set `--max-gen`. The
promoted kernel carries its source, so every generation after the first runs in `source` mode: the
teacher reads the previous best, the student never does. (CuTe DSL tasks only.)

## Leaderboard Tasks

```bash
forge init rms --leaderboard flashinfer-bench --dataset <flashinfer-trace> \
    --definition <definition> --workload <workload>
forge new tasks/rms                                  # -> .artifacts/rms_<id>
forge run .artifacts/rms_<id>
forge export .artifacts/rms_<id> --out solution.json
```

The task freezes the board's problem: inputs, tolerance and stopwatch follow the board's own
evaluation code. → [docs/leaderboards.md](docs/leaderboards.md)

## Kernels

[`kernels/`](kernels/) releases the kernels ForgeKernel generated in the paper (24 kernels in CuTe
DSL, on H100 and B200) and one generated after it (the DeepSeek-V4.1 fused sparse-attention
prefill in CUDA C++, 1.10× FlashMLA on B200). They are grouped by operator type, then by operator
and GPU:

| Directory | Operators |
|---|---|
| [`attention/`](kernels/attention/) | attention forward (with its self-evolution generations) and backward, GQA and MLA paged decode, MLA paged prefill |
| [`sparse_attention/`](kernels/sparse_attention/) | DeepSeek sparse attention (DSA) backward, score recompute and indexer backward; DeepSeek-V4.1 fused prefill |
| [`linear_attention/`](kernels/linear_attention/) | Gated DeltaNet prefill forward and backward, varlen prefill; recurrent Kimi Delta Attention prefill |
| [`gemm/`](kernels/gemm/) | dense 16-bit GEMM, FP8 GEMM, FP8 × FP4 grouped GEMM (MoE) |
| [`conv/`](kernels/conv/) | Conv2d weight and data gradient, Conv3d forward |

The speedup of each kernel, what it was compared against, and the environment it ran in are listed
in [kernels/README.md](kernels/README.md).

## Environment

The paper's experiments ran on the following GPUs and software. Most campaigns used the main
image of each GPU:

| | Hopper | Blackwell |
|---|---|---|
| GPU | NVIDIA H100 80 GB HBM3 SXM, 132 SMs (`sm_90a`) | NVIDIA B200, 180 GB HBM3e, 148 SMs (`sm_100a`), 1000 W, max SM clock 1965 MHz |
| Driver | 550.90.07 | 580.105.08 |
| CUDA toolkit | 12.8 | 13.1 (`nvcc` 13.1.115) |
| PyTorch | 2.6.0a0 (NGC 25.01) | 2.10.0+cu130 |
| CuTe DSL (`nvidia-cutlass-dsl`) | 4.4.2 | 4.4.2 |
| Nsight Compute (`ncu`) | 2025.1 | 2025.4.1 |
| cuDNN | 9.7 | 9.19.1 |

Some campaigns ran on a second image: on H100, CUDA 13.1, PyTorch 2.10 (NGC 25.12) and CuTe DSL
4.6.1; on B200, CUDA 13.0, PyTorch 2.13 and CuTe DSL 4.6.2 with Nsight Compute 2025.3.1.
Every reported gain times the reference and the candidate in the same GPU job, so the
comparison does not depend on which image a campaign used.

## Repository Layout

The layout follows ForgeTrain's:

```text
src/forgekernel/     the harness: CLI, tasks and runs, the loop, GPU jobs
global_config/       the bottom config layer: models.toml, loop.toml, backends.toml
tasks/               <op>/: task.toml, description.md, shapes.json, reference/;
                     _template/ (what `forge init` copies), rmsnorm/ (ready to run)
.artifacts/          runs, gitignored: <op>_<id>/{run.json, ws/, report.md, best/, ...}
skills/              the agents' tools, one SKILL.md + scripts each, mirrored into
                     the agent's .claude/skills/ before every turn
meta/                teacher and student prompts (meta/cuda/: CUDA C++)
hw_kb/               the hardware knowledge base (cutedsl/, cuda/), by GPU architecture
docs/                design notes, milestone template, leaderboards, KB
scripts/             agent launcher, /goal driver, self-evolution
tests/               pytest suite (CPU only)
kernels/             kernels generated by ForgeKernel, by operator type
```

## Contributing

Contributions are welcome: bug reports, feature requests, pull requests, documentation and
translations. Run the tests before sending a change (`pip install pytest torch safetensors`,
then `pytest`; the suite needs no GPU). A knowledge-base
entry needs a GPU probe that certifies it; see [docs/kb-harvest.md](docs/kb-harvest.md).

## Limitations

- **A reference helps most.** Given only a PyTorch implementation and a device-time number, the
  paper's ablation stalls on all three operators tested; ForgeKernel surpasses an implementation
  that exists more reliably than it writes the first one.
- **A plateau is an estimate, not a proof.** Self-evolution stops at the same-job noise floor:
  the loop found nothing more, which does not mean nothing more exists.
- **Coverage.** Four operator families on two NVIDIA GPU generations; sparsity patterns and memory
  hierarchies unlike these are untested.
- **Cost.** A run is hours of GPU time and frontier-model usage, and the agents run with
  permissions bypassed inside a policy the driver writes; run it on a machine meant for it.

## License

Apache License 2.0. See [LICENSE](LICENSE). The leaderboard evaluator adapts code from
FlashInfer-Bench and SOL-ExecBench (both Apache-2.0); see [NOTICE](NOTICE).

## Acknowledgments

ForgeKernel builds on [CUTLASS and CuTe DSL](https://github.com/NVIDIA/cutlass),
[Nsight Compute](https://developer.nvidia.com/nsight-compute),
[Claude Code](https://docs.claude.com/en/docs/claude-code/overview),
[FlashInfer-Bench](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) and
[SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench), and measures itself against the
expert kernels of FlashAttention, DeepGEMM, FlashMLA, FlashInfer, cuDNN and cuBLAS.
