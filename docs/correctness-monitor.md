# Correctness monitor

Every operator tree decides correctness with its own judge: the trees `forge run` generates from a
task use the judge the teacher wrote and calibrated against the reference, a leaderboard task uses the
board's judge, and a hand-made tree uses its own `eval/bench.py` -- each with its own reference, its
own tolerance and its own inputs; the driver reads only `correctness.pass`. Trees differ in how strict
they are, and the more common problem is a **narrow input surface**: judged on a single randn input,
the distribution-dependent fast paths and the NaNs at the edges never run once.

The monitor does not replace a tree's judge; it widens, from outside the tree, the range of questions
the judge is asked, **without changing the tree**:

1. **Rerun the tree's own correctness on other inputs.** The bench draws its inputs from torch's RNG,
   hands the same tensors to the kernel and to its own reference, and judges with its own tolerance.
   The monitor intervenes where the inputs are drawn: both sides see the same new inputs, and the
   tree's judge still decides right or wrong.
2. **Watch the kernel's outputs.** On every call from outside to a public function of the `kernel`
   module, it records the shape, the NaN / ±inf counts and a finite-values-only checksum of the tensors
   the call produced, and whether the inputs held non-finite values. "Produced" covers the return value
   and the arguments it wrote: tensor arguments are read **before** and after the call, an argument
   that changed during the call (a DPS kernel writing into the caller's `out`, or an in-place update)
   counts as an output, and what it held before the call (left-overs in an output buffer) does not
   count as an input. NaNs the judge let through, and two calls on the same inputs that disagree, show
   up here.

## Input profiles

| profile | what it does |
|---|---|
| `base` | nothing (the tree's own inputs) |
| `repeat` | nothing, run again -- the output checksums and the verdict are compared with `base` |
| `seed1` / `seed2` | every seed the bench sets (torch global, `torch.Generator`, numpy, random) is offset: a fresh draw from the same distribution |
| `amp` / `tiny` | every normal draw ×4 / ×1/16 |
| `spikes` | 0.1% of the elements of normal draws set to ±12 |
| `zero_block` | normal draws of rank ≥2: the first 1/8 of the last dimension zeroed, the first 1/16 of the rows zeroed |
| `const_rows` | normal draws of rank ≥2: the first 1/16 of the rows all equal to row 0 |
| `int_edges` | integer tensors pushed to the edges of the range the bench itself gives: 1/16 of a `torch.randint` set to `low`, 1/16 to `high-1`; the first 1/16 of a non-zero integer `torch.full` set to 0 (a length / count of 0 = an empty row) |

The floating-point profiles change only **normal** draws (`torch.randn` / `randn_like` / `normal` /
`Tensor.normal_`); uniform draws are left alone. Every value `int_edges` picks lies in the range the
bench itself could produce, so the inputs stay legal.

**Only the bench's own draws change**: only files whose call sites are inside the job tree (the bench
and its helpers) are affected; buffers the kernel file allocates itself (an int buffer built with
`torch.full`, say) and draws inside a sealed reference are left alone.

A profile that changes none of the kernel's inputs (the inputs do not come from the torch RNG /
`torch.full`), or that turns the inputs non-finite, is recorded as "not applicable" and does not count
toward the verdict.

## Verdict (reported, never acted on)

| level | when |
|---|---|
| **alarm** | the tree fails on its own inputs; fails with another seed; fails under a stressed input on which **the reference passes**; under some profile the kernel produces NaNs that `base` did not; the correctness task crashes under some profile (`base` is fine); two runs on the same inputs flip the verdict |
| **warn** | fails under a stressed input, but the tree has no `cardref` task (the reference cannot run on the same inputs); two runs on the same inputs differ in output checksum by a relative > 1e-3 (a race); `base` outputs hold NaNs and the judge passed them; a task timed out; the monitor job was cut short |
| **info** | the kernel and the reference **both** fail under a stressed input (the judge's tolerance does not fit this distribution; not the kernel's fault); the output is not bitwise reproducible but differs by ≤ 1e-3; a profile is not applicable; no kernel call was observed |

Comparing against the reference relies on the tree's `cardref` task (the reference run as the
implementation through the same bench). With it, a failure under a stressed input can be attributed to
the kernel or to the tolerance; without it, only warn can be reported.

The result is only recorded: in the operator tree's `correctness_monitor.jsonl` (in a run,
`.artifacts/<op>_<id>/operators/<op>/correctness_monitor.jsonl`), and at the finish also in the
`correctness_monitor` field of `best_confirmation.json`. **It is not fed to any agent, does not change
KEEP/REVERT, and does not affect when the loop stops.** The one exception is outside the loop: when the
loop did not stop at its target, `forge run` counts the best kernel's final confirmation as meeting the
target only if the monitor raised no alarm on that kernel (this decides `forge run`'s exit code).

## How to run it

**At the finish of every run**: after the confirmation step, `forge run` runs one monitor job on the
confirmed sha (or, without a confirmation, on the best green KEEP) with all default profiles
(`forgekernel.monitor.DEFAULT_PROFILES`); this is not configurable. The result lands in the
confirmation record (`best_confirmation.json` → `correctness_monitor`) and in
`correctness_monitor.jsonl`; an alarm on the confirmed kernel makes `forge run` exit 2 even when
the loop stopped on its target.

## Mechanism

- The runtime (`src/forgekernel/monitor/runtime/`) rides into the job tree's `.fk_monitor/` as an
  overlay commit (the tree under test is unchanged); the job is
  `gpu_entry.sh cmd python3 .fk_monitor/run.py ...`.
- `run.py` uses only the standard library (the `python3` on the job's PATH may have no torch). For
  each profile it runs `bash gpu_entry.sh correctness` once with `.fk_monitor/hook/` first on
  PYTHONPATH; when the bench starts under the tree's own interpreter it loads the `sitecustomize.py`
  there, which installs the hooks and then hands over to the image's own `sitecustomize`.
- The hooks act only when `FK_MON_PROFILE` is set. They give the `kernel` module another module class:
  a public function taken from **outside** the module comes back wrapped, while calls inside the module
  through its globals, and objects handed to the JIT compiler, are untouched. Nothing is recorded during
  CUDA graph capture, and at most the first 64 calls are recorded. The wrapper summarises the tensor
  arguments before and after each call; an argument position that any call of a function wrote is
  treated as an output in all its calls (when the same memory is reallocated to the next call and the
  kernel writes back an identical result, that call looks as if it wrote nothing).
- After each profile a `=== FK MONITOR PROFILE ===` block is printed, and at the end the complete
  `=== FK MONITOR JSON ===`; when the job is killed midway, the harness assembles a partial result from
  the blocks already printed (marked as truncated). The budget defaults to 85% of `TASK_TIMEOUT`; when
  what is left is not enough, the remaining profiles are skipped and noted.

## Limitations

- The tree's own judge still decides right or wrong: when its tolerance is too loose (only a global
  cosine, say), the monitor widens the input surface but cannot tighten the judge. The NaN check and
  the reproducibility check are the exceptions; they do not depend on the judge.
- On trees whose inputs do not come from the torch RNG / `torch.full` (real workload files, numpy
  draws), the stress profiles do not apply.
- On the trees `forge run` generates from a task, the profiles reshape the input groups the judge's
  `make_inputs` draws (the hook treats the judge, which the bench compiles from source, as bench code),
  and the kernel the bench loads by file path is handed to the hook, so its outputs are watched too.
  Input groups read from files are left as they are.
- Structural edges reach only the integers the bench builds with `randint` / `torch.full`; indices
  produced by operations such as `argsort` or `topk` (out-of-range sentinels, shuffled order,
  duplicates) are out of reach, and those edges still need the tree's own cases.
- When the kernel and the reference both fail under a stressed input, only info is reported: it may be
  the tolerance, or both sides may be wrong.
- When the kernel is not imported as the `kernel` module (a non-Python kernel the bench loads itself),
  its outputs cannot be seen; only the judge's verdict under each profile remains.
- "Wrote" is judged by whether the before/after summaries (NaN/inf counts, the checksums) changed: a
  write that only moves values between positions, leaving the summaries unchanged, goes unseen.
