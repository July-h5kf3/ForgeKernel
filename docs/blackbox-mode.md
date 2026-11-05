# Blackbox mode: the reference as a binary

In blackbox mode the reference can be **run, timed, profiled and disassembled**, but its **source is
readable by neither the teacher nor the student**. A `task.toml` whose `[reference]` gives only `run`
(or gives `source` and also `blackbox = true`) is in this mode, and the loop configuration that
`forge run` generates declares `oracle_mode = "blackbox"`. The reference travels encrypted with two
trees -- the operator tree (the student's, which holds the reference timed beside the kernel in the
same job) and the reference / recon tree (the teacher's) -- and is unsealed only inside a GPU job,
with the key (see "How the reference reaches the job" below).

## Where the boundary is

| Open | Forbidden |
|---|---|
| SASS (`--print-source sass`, `cuobjdump -sass` on the JIT cubin) | NCU line-level source (`--print-source cuda` / `--import-source`) |
| NCU metrics, timings, CUPTI spans | the sealed blob, the key, the unsealing machinery, the temp tree the JIT compiles from |
| public algorithms and public definitions (including their torch reference implementation) | the reference's upstream repository and any mirror or paraphrase of it |
| introspecting the CuTe DSL package itself (`cutlass.__file__`, `importlib.metadata`, its .py files) | `getsource` / `linecache` / `sys.modules[...]` on the reference module |

Two defenses each cover one side: the in-job tripwire (next section) covers the commands run on the
GPU machine; on the machine that runs `forge`, the agent policy hook (`scripts/agent_policy_hook.py`,
which reads `ref_source_paths` / `ref_source_urls` from the generated loop configuration) refuses
agent Read and Bash access to the reference's source. The tripwire is a tripwire, not the boundary
itself.

A reference that is a **library** -- public code, possibly installed on the GPU machine -- has two
more ways to its source than its files in `reference/`, and `task.toml` closes both:

```toml
[reference]
run          = "reference/run.py:run"           # calls the installed library
deny_urls    = ["github.com/org/RefLib", "pypi.org/project/ref-lib"]
deny_modules = ["reflib", "ref-lib"]
```

- `deny_urls`: URL fragments of its repositories. The policy hook refuses any agent tool call that
  names one, and any download that asks for the repository by name (`pip download ref-lib`).
- `deny_modules`: its import and distribution names. The in-job tripwire of both trees refuses a
  `cmd` (and a notes/ script) that names one as a whole word -- `import reflib`,
  `.../site-packages/reflib/...` -- and the judge refuses a kernel that does, so the student can
  neither read the installed copy nor forward to it.

## The in-job tripwire (blackbox guard)

In both trees `forge run` generates, the `cmd` task of `gpu_entry.sh` (the channel agents run probes
through) is guarded: when the command string, or a `*.py` / `*.sh` under `notes/` (and C/C++ probes,
and the operator tree's kernel file), reaches for the sealed reference (the payload, the key, the
loader, reflection hooks, JIT caches, ...), the job prints `REFUSED` and exits 3. The harness
generates this guard; the user writes nothing.

### The shared guard: `scripts/blackbox-guard.sh`

The guard's word list lives in one place, **`scripts/blackbox-guard.sh`**. A job only ever sees
`git archive` of the reference tree itself; this repository's `scripts/` is not there. So on a
blackbox run the driver seeds the file into the reference tree's `notes/` during reconnaissance
(beside `sass_tools.py` and the other tool copies) and commits it. A `gpu_entry.sh` that sources
**that copy** carries no guard logic of its own:

```bash
BLACKBOX_EXTRA='kda_gdn|gdn_runtime'          # this reference's module / blob names, optional
source "$W/notes/blackbox-guard.sh"           # W = the reference tree's root (where gpu_entry.sh is)
...
cmd) blackbox_guard "$*"
     echo "[task] blackbox guard: passed"
     run_bounded bash -c "$*" ;;
```

`blackbox_guard <command string>` scans the command string and every `*.py` / `*.sh` under
`$W/notes/` (where the teacher's probe scripts live); on a hit it prints `BLACKBOX-GUARD REFUSED`
(with the matched tokens) and exits 3.

**`BLACKBOX_EXTRA` must not contain a substring of the reference kernel's symbol name.** That symbol is
public: NCU prints it, the harness writes it into `ref_baseline.json` and the recon prompt, and a
legitimate probe filters on it with `-k`. An observed failure mode: EXTRA listed a word that also
occurs in the main kernel's mangled name, so the teacher's first probe that filtered on the kernel
name was refused; and because `notes/` is scanned **cumulatively**, that one resident file refused
every later round as well. Module names and blob names are the right EXTRA tokens; anything that also
occurs in a mangled kernel name is not.

**`BLACKBOX_EXTRA` must not name the reference tree's own measurement wrapper either** -- the
`<tree>/<name>.py` that the tree's README tells probes to import for `make_inputs()` /
`ref_runner()`. It is the **sanctioned handle** on the reference, not the reference. Listing it refuses
every probe that does the sanctioned thing and pushes the teacher to move the wrapper outside
`notes/`, where the scan cannot see it -- a workaround forced by a contradictory configuration. The
right EXTRA tokens are only: the sealed blob's name, the unsealed entry's name, and implementation
module names (`fkref_*` is already in the base list). The guard checks this **when it is sourced**: an
EXTRA token that matches any `*.py` module name at the reference tree's root prints
`BLACKBOX-GUARD WARNING` at once (in every job log, so the baseline job already shows it).

**The guard does not scan itself.** Once seeded into `notes/`, it sits in the very directory it scans,
and its own comments and forbid list contain every forbidden word -- without an exemption it would
refuse **every** `cmd` before a single probe ran (unit tests that source it from `scripts/` never see
the self-hit, so they cannot catch this). Only a file that is **byte-for-byte identical to the running
guard** is skipped; a file with the same name but different content is scanned as usual, so a probe
cannot hide under the guard's file name.

**A refusal must be loud.** With an unguarded `notes/*.sh` glob and no `.sh` files in the directory,
the glob stays literal, `cat` on it returns non-zero, and under the caller's `set -euo pipefail` **the
whole job dies at the aggregation line before the refusal is printed**; the teacher sees a bare
`rc=1` it cannot tell from its own probe crashing, and retries again and again. The file list is now
built by testing each candidate with `-f`, and every aggregation pipeline ends in `|| true`.

**`tempfile` / `mkdtemp` / `tarfile` are not on the forbid list.** The unsealer uses them, but they
are standard library, and an NCU probe legitimately needs a temp file for its `.ncu-rep`; banning them
only refuses ordinary probes for one `import tempfile` line while stopping no real leak. What guards
the unsealed tree is the `fkref` prefix + the blob + the key + the module object + `getsource` /
`linecache` / `sys.modules[...]`, all of which are on the list.

`tests/test_blackbox_guard.py` pins all of the above as regression tests, in the production shape
(seeded into `notes/`, then sourced from there).

## Same-job pairing: both sides on the same warm plateau

When the reference is timed beside the kernel inside the job (`ref_in_job = true` in the generated
loop configuration), the verdict is the **same-job ratio** ref / kernel. An absolute time measured for
the reference alone, in another tree and another job, cannot serve as the target: the same code was
measured 7–10% apart on different GPU nodes, and a 5% equivalence bar drowns entirely in that noise
(every KEEP / REVERT would be a guess).

But "measure twice, back to back" is still unfair. The device window drifts with the GPU's clock and
temperature, and the drift systematically favours whichever side is measured first: a job's first
timed window is the fastest (a cold GPU at full boost; the same kernel's first window was 3–5% faster
than the next three), and readings then slow down as the part heats (the same kernel measured first
as the kernel and then as the reference read 2.7% apart).

So the template `eval/timing.py` provides `measure_paired`, and a bench must use it rather than
calling `measure_windows` twice:

1. **burn-in**: first heat the part with the kernel under test for a few seconds, so every timed
   window lands on the warm plateau;
2. **mirroring**: measure ref, kernel, kernel, ref, and give each side the mean of its two device
   readings -- both sides then occupy the same average position in the remaining drift.

The same kernel on both sides should read ≈1.00; `paired_performance` maps the result into the perf
JSON (the `ref_*` / `vs_*` keys appear only when the reference was really paired), and
`describe_paired` gives the four raw readings for checking in the job log.

## How the reference reaches the job (sealing)

Pairing in the same job means the reference has to travel into the job, and putting it in the
operator tree in plaintext means the student can read it. This holds for every run that has a
reference (in source mode the teacher reads the source, but the student still must not; a
self-evolution generation's reference is the previous generation's `kernel.py`). Both wrong
directions have actually been taken: putting a plaintext `ref_kernel.py` into the operator tree and
relying on a regex that forbids importing it (the student can still read it), or moving the reference
back into a separate tree and using absolute times (back to the noise of the previous section).

`forge run` seals the reference with the template's `eval/closed_ref.py` when it writes the trees;
the user does nothing:

- **Seal**: every source file of the reference is sealed into one keyed payload
  (`ref_bin/ref_payload.bin`), committed into the operator tree and the recon tree (byte-identical in
  both). The key goes only into the gitignored `submit.local.toml` `[env] FK_REF_KEY` of the two trees
  and is never committed. **The key file is inside the run directory, where an agent could otherwise
  read it**: the harness's agent policy denies the student and a blackbox teacher every tree's
  `submit.local.toml` (a Read rule plus the Bash sandbox's denyRead, the same mechanism that denies the
  reference's source) -- on the machine that runs `forge`, what stops unpacking is this denial, not
  the encryption; a key put in any other file has no such rule.
- **The key does not stay in the job environment**: every process of a job inherits the environment
  (the student's `cmd` probes, the kernel that runs in the reference's process, the launch environment
  in a profiler report, `/proc/<pid>/environ`), so the first thing a tree's `gpu_entry.sh` does is move
  the key into `ref_bin/ref_payload.key` beside the payload (mode 600, gitignored) and re-exec itself
  without it (`forgekernel.sealref.key_preamble`); `closed_ref.load` reads that file when the
  environment has no key. The transport never puts the key on a command line either: ssh delivery
  sends `*_REF_KEY` over ssh's stdin and reads it into the environment on the host before starting
  `gpu_entry.sh`.
- **Load** (in the job): the bench gets the callable with `closed_ref.entry("<entry name>")`; the
  module is materialised in memory, with no source file on disk. CuTe DSL's `inspect.getsource` still
  gets the source through the loader's `get_source`, so DSL decorators work as usual.
- **Check**: the payload header carries the plaintext's sha256, so a verbatim copy is recognised
  without the key; with the key, a copy with a few lines changed is recognised too (refused at ≥40%
  overlap of distinctive lines). `forge run` runs this check on both trees before handing them over.
- **Block**: the guard's base list already contains `closed_ref` / `ref_payload` / `_REF_KEY` /
  `keystream` / `unseal`; the operator tree's kernel file has a fence of its own (the loader, the
  payload, the key, the environment, reflection hooks, JIT caches) so a student kernel cannot simply
  forward to the reference.

The same tool can be run by hand to inspect a run's trees:

```bash
python eval/closed_ref.py check-tree <operator tree / workspace / reference tree> eval/ref_bin/ref_payload.bin
python eval/closed_ref.py digest eval/ref_bin/ref_payload.bin
```

## Reconnaissance: schedule transcription per MMA family

The `sched_live_tools` module seeded into the recon tree (`skills/sass/`) first takes a census of the
MMA instruction families on the SASS page, then picks the transcription by the census:

- **wgmma** (Hopper HGMMA): chains by accumulator base register; the decisive reading is the
  `WARPGROUP.DEPBAR.LE` immediate (the wait depth) in force when each elementwise run starts.
- **tcgen05** (Blackwell UTC*MMA): the accumulator is in TMEM; chains by the TMEM accumulator operand;
  the handoffs are `UTCBAR` (tcgen05.commit → mbarrier) and `SYNCS` waits; accumulator traffic is
  `LDTM` / `STTM`.
- **mma_sync** (per-warp HMMA / IMMA / DMMA / BMMA): chains by accumulator base register; the handoffs
  are `DEPBAR.LE SB` copy-group waits, named / cluster barriers and mbarrier.

The LIVE half (register liveness) does not depend on the architecture and is delivered as usual. **A
page with no MMA at all** is not "the wrong instrument" but "the wrong page" (the wrong kernel was
disassembled, or only a prologue) -- it is still `SCHED-FAIL`, and the gate stays closed. When the
schedule genuinely cannot be obtained, the teacher writes `SCHED-WAIVED: <reason>` in the recon report
with the architecture's own synchronisation topology, or writes an architecture-specific probe that
prints a self-checking verdict line.
