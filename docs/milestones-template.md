<!--
  ForgeKernel's milestones meta-template.

  It started from an external blueprint specification and has since evolved in place for
  this repo (the reader is now the teacher, the free-optimization phase was removed, the L0
  boundary was aligned), **and is no longer kept verbatim in sync with its source** — change it
  here directly when needed. The mapping to the §9 machine-read skeleton and the mechanisms
  specific to this repo (decision five-tuple / DAG prefix reuse / trajectory harvest) live in
  `meta_milestone.md`; where it conflicts with this file, `meta_milestone.md` wins.
-->

# Milestones meta-template — how to write a kernel-optimization milestone

> This document defines how each `operators/<op>/milestone.md` (below, "the instance file") should
> be written. The instance file is the student loop's **only construction blueprint + read-only
> rules**; this template is its writing specification, and its reader is the **teacher (meta
> agent)**; the student does not consume this file.

---

## A. Template structure specification

The instance file **must** contain the following sections, **in order**:

### A.1 Problem statement

- **Operator definition**: which operator is computed, including the upstream contract (e.g. which
  tensors fwd saves for bwd).
- **Fixed shapes**: one table of benchmark shapes (B/H/N/D/dtype/mask etc.), unchanged throughout.
- **Reference (ref)**: what the ref is and how it runs (bench script, NCU script paths), and the
  correctness golden reference (usually FP32 autograd / torch matmul).
- **Hardware**: GPU model, peak FLOPS / bandwidth, target ISA (e.g. sm_90a WGMMA+TMA).
- **Target**: the relation to the ref (parity / some ratio); only the qualitative target is written
  here, the quantitative number is converted from the ref baseline (see A.4).

### A.2 Math Spec (correctness contract)

One section per kernel, stating:

- Input / output tensors: shape, dtype, layout (including layout conventions such as flat / 4D).
- Mathematical definition: the elementwise / blockwise formula of `output = f(input)`, and the
  derivation of key mathematical identities (e.g. the rowsum identity of softmax bwd).
- A diagnosis checklist of common correctness bugs (optional, but strongly recommended).

**Writing rule (important)**: the Math Spec is a contract of **what is computed**, **not** a hint
of how to compute it. The instance author should avoid smuggling in algorithmic priors — GEMM
numbering, loop traversal order (KV-outer vs Q-outer), and formulas written in the shape of an
optimization trick (e.g. writing the formula directly as exp2 + a pre-folded constexpr) are all
algorithmic priors. Principle: **what is not in this run's prior sources does not go into the
exam** — when there is a measurable ref, a representation choice already confirmed by the ref's
behavior (e.g. the log2 representation of LSE) may be written into the contract; without a ref,
write only the mathematical definition and leave the representation choice to the loop to explore.

### A.3 M0 — architecture fixed + first runnable version

M0 delivers **the final architecture itself**. It contains two groups of decisions:

1. **Algorithm-contract decisions** — kernel split / data flow and stationarity (what stays
   resident, what streams) / intermediate-tensor ABI / precision contract / redundant-dimension
   (broadcast, grouping) strategy. Criterion: **they can be decided without knowing the hardware
   budget**.
2. **Hardware-architecture decisions** — tile shape / warp roles and thread counts / register
   budget / smem layout and the **access atoms** of each exchange buffer / MMA atom layout /
   **issue sequence and latency-hiding relations** (which elementwise segment runs while which MMA
   chain is in flight) and its **segmented live-set budget**. Criterion: **they decide the global
   ceiling; everything downstream only adapts inside them**. Levers such as the stage count and the
   wg_wait depth setting are given only as starting values and are not fixed here (see A.3.2; for
   where swizzle and wg_wait belong see the same section — the path / the latency-hiding relation
   belong to the contract, choosing a mode within the path / tuning the depth within the relation
   are the levers). **Granularity-coupling exception**: a stage count counts as a lever only if the
   supply stage granularity = the consumer atom granularity; when recon measures the reference's
   stage granularity as finer than the candidate consumer atom, sweeping it alone is forbidden (a
   single-variable reading has zero evidential weight), and a joint verdict experiment of
   {coarse + coarse} vs {fine + fine} is required: wins → enter the contract as a companion pair;
   loses → set by the student's measurement, and "the reference chose a fine granularity" is
   recorded as an unexplained structure of the reference (most likely in the producer's supply
   organization) and handed to recon; it must not be re-swept over and over (the clause and two
   rounds of evidence are in the clause of the same name in meta_milestone.md).

By **change radius**, the two groups of decisions split further into two delivery forms — the
**resource-topology contract** (must be fixed in M0 in contract form) and **levers** (starting
values + left to M1+ to measure round by round); the dividing criterion and the contract format are
in A.3.2.

Other rules:

3. **Sub-steps inside a group have no gate**, only a **build self-check** (compiles + a probe
   verifies the local contract. Whole-machine correctness cannot be measured during the build
   phase — the driver skips the whole-machine eval, and the correctness account is settled by the
   gate row in one go, see POLICY §3; so this is not called "coarse hygiene" here, which refers
   only to "compiles + correctness PASS" in the post-green context). M0 is one large delivery;
   without intermediate self-checks, when it gets stuck you cannot even see where. On top of the
   build self-check, the build phase has three **failure-path disciplines**:
   - **Whole-machine smoke checkpoint (the gate row's entry gate)**: before the M0 gate row is
     submitted, the lineage must already hold a passing **whole-machine smoke probe** — every
     kernel in the delivery list lowers successfully at its real shape and **completes its launch**
     (it only proves "can launch", values are not compared; the numeric account still belongs to
     the gate). Basis: compiling ≠ launching without a crash; a runtime crash (the launch failed /
     illegal memory access family) shows up in a full eval as one unlocated error line, while a
     smoke is a minutes-long probe. With the smoke certificate in hand, the gate's failure space is
     reduced to the numeric layer — and the numeric layer has a per-tensor diff; it is the only one
     of the three layers that can be debugged directly.
   - **High-risk runtime mechanisms mounted one at a time**: mechanisms that are "invisible to
     trace, blow up only at launch", such as TMA copy / write-back, the mbarrier pipeline and
     register reallocation (setmaxregister), must not go onto the machine for the first time two or
     more at once without smoke coverage. For each one mounted, run one whole-machine smoke; if the
     launch crashes, the suspect is the last one mounted — bisection is built into the assembly
     order, so no bisection is needed afterwards. When the skeleton is incomplete and cannot launch,
     fill in with stubs (the consumer computes fake data, results are not written back): make it
     launchable first, then grow the real thing.
   - **Two failures → unbundle**: after the same multi-decision bundle (the same decision_id
     combination) fails whole-machine twice (trace error, launch crash, gate correctness FAIL —
     regardless of layer), resubmitting it a third time unchanged is illegal — it must be
     downgraded: take the bundle's high-risk mechanisms down to the conservative form (cooperative
     scalar copies, compiler-default registers, a naive exchange face), get first green, then after
     green swap them back one at a time under A.6's free granularity, measuring round by round.
     Basis: a bundle's failure evidence cannot point to a member (A.6 discipline 3 corollary a), and
     the build phase has no per-step measurement — resubmitting the same bundle over and over is
     repeating the same bet on a channel with no information.
4. **M0 Gate (closing)**: correctness + machine-verifiable architecture-conformance lines (see
   below; the topology-contract predicates of A.3.2 are their core) + mechanism-verification lines.
   The prerequisite of submitting the gate row = the whole-machine smoke certificate of the previous
   item (the milestone's M0 §3 should write it as the first row of the checklist).
   **There is no absolute performance line** — how far the first version gets cannot be known in
   advance; set high it is unreachable, set low it is meaningless; whether the architecture really
   took effect is judged by the mechanism lines (ratios, anchored to the ref), which are more robust
   to cross-job fluctuation than absolute ms.
5. **Signature freeze + topology freeze**: the kernel API is frozen once committed, additive-only
   afterwards; **M0's resource-topology contract (A.3.2) is frozen as well, and M1+ must not modify
   it**. Levers are outside the freeze — tuning levers round by round in M1+ is planned
   optimization, not touching the foundation. Without this rule, M1+ goes back and changes the
   topology, and "incremental optimization" degenerates into a rewrite.
6. M0 cannot be deleted or trimmed.

### A.3.1 Verifiability marks for architecture decisions `[VERIFY:]`

Every M0 decision must be marked with **what establishes that it was really carried out**. The
reason shares its root with `[MEASURED]/[ESTIMATE]`: that one marks "where the number came from",
this one marks "how to verify it landed". A statement the agent wrote itself is not evidence.

| Mark | Meaning | Typical observables |
|---|---|---|
| `[VERIFY: ncu-direct]` | launch footprint, read directly | number of kernel groups and their time shares, block/grid size, register count, smem bytes |
| `[VERIFY: ncu-derived]` | traffic footprint, threshold-type inference | a residency failure amplifies DRAM reads by an order of magnitude (e.g. an upper bound on `dram_pct`) |
| `[VERIFY: correctness]` | a violation is an error; the correctness gate already covers it | intermediate-tensor ABI, precision contract |
| `[VERIFY: sass-derived]` | dossier footprint, read statically (the driver runs a census over the student's SASS once per evaluated iteration) | cache-policy bits (`cache_ops`), DEPBAR in-flight depth (`commit`), supply issue order (`supply`), NamedBarrier ruling / ring form, MMA operand source (`mma_a_reg` / `mma_a_desc`) |
| `[VERIFY: none]` | **no machine-read verification** (the dossier axes cannot read it either); relies on later performance lines to expose it | violations of a convention that are self-consistent on both ends (e.g. the ABI uses row-major instead of fragment order) |
| `[VERIFY: inferred]` | a structural reading for which recon's discriminating observation **produced no positive reading** (several readings are identical in bytes / counts; a script printing 0 rows is not a ruling) | the segment axis of a KV ring (32 keys × full width vs 64 keys × half width); list each reading as an `[ESTIMATE]` candidate, and the student's first probe decides the axis |

**Only the first four may enter the M0 Gate.** A decision marked `none` is still written into the
milestone as build guidance, but not as an acceptance line — writing something that cannot be
verified as a gate line amounts to asking the agent to check itself against itself.

**Two hygiene rules for gate rows (driver machine check, `validate_milestone`)**:

1. **Compiler allocation values are not acceptance lines.** The immediates of `USETMAXREG` /
   `setmaxnreg` and an exact `regs_per_thread == N` are allocation results **observed** on the
   reference, not requirements. A gate writes only `regs_per_thread <= <cap>` and
   `spill_ld_sectors == 0`; the split numbers are the student's choice.
   (An observed failure mode: a v1 pinned one set of split numbers while also requiring the
   producer to prefetch a whole trip's indices, and first green spilled heavily; another milestone
   for the same operator wrote "the split numbers are nvcc's allocation result, not an acceptance
   line", the student chose a different split, and spilled nothing.)
2. **A SASS shape can be a gate row only once it has been priced.** `[VERIFY: sass-derived]` says
   what the reference **looks like**; every such line in a gate block must carry
   `[PRICED: <how much slower without it>, <job N | KB:<entry> | ref-price>]` — a probe has
   measured "what this shape is worth". An unpriced one goes into the decision body as a
   description, not into the gate.
   (The cp.async BYPASS bit measured -22% in one row and may enter the gate; the pinned register
   split was never priced and was in fact slower than the alternative. An observed failure mode: a
   v1's M0 gate carried a dozen or so unpriced shape lines, and the student spent a whole round
   proving with probes that "I look like it".) A.5's operand-source line (`mma_a_reg` /
   `mma_a_desc`) is that lint's own obligation and is exempt. The number of
`none` rows in a milestone is itself a measure of its verifiability; the driver collects the
`none` rows of the recon table into the **out-of-run gap queue** (`evidence/gap_queue.md` +
`docs/kb-provenance/gaps/`, for humans to read); they are unverified readings of the reference's
structure and **do not enter hw_kb**.

**Expressibility ladder `[EXPR:]` for the modifier / instruction-variant axis**: when the difference
of a decision lies in a modifier or an instruction variant of the same family (cache bits, prefetch
width, wait encoding, reduction variant…), the five-tuple must carry
`[EXPR: A=<DSL spelling> | none(<verbatim op signature>); B=<verbatim inline PTX> | none(<footprint absent/compile error>); C=toolchain-limit | ->]`:
rung A = a constructor argument / enum of the DSL op (the evidence is the verbatim signature
printed in a job, not memory); rung B = issuing PTX directly per `hw_kb/cutedsl/sm90/ptx_escape.md` (the
evidence is a footprint count); rung C = the toolchain limit, **which may be written only after both
A and B were measured as none**. The acceptance line of such a decision is anchored to that axis of
`[VERIFY: sass-derived]`, not to a derived proxy (a modifier changes latency, not volume; a proxy
such as `dram_bytes` would judge an already-delivered structure red).

### A.3.2 Resource-topology contract and levers (commit early)

M0's decisions split by **change radius** into two delivery forms:

**Resource-topology contract** — every decision that defines the **shared-resource topology**.
Criterion: it declares the **existence, ownership and paths** of shared resources; deciding it late
or changing it is a global restructuring. Inventory:

- **smem buffer inventory**: which buffers exist, who owns each, their lifetimes.
- **Access atoms of exchange buffers** (one column each for the write and the read side): matrix
  copy atom `STSM/LDSM` / vectorized / TMA-bulk / MMA smem-descriptor direct read. The values are
  **transcribed buffer by buffer** from the architecture reference's SASS memory-access footprint,
  marked `[MEASURED, job N]`; only a buffer recon did not cover may be marked `[INFERRED]`, and then
  it must be backed by a wavefront mechanism line. An atom is **a cliff, not a knob**: a scalar
  elementwise path costs several times more wavefronts than a matrix path — **pinning swizzle
  without pinning the atom is a void contract** (a scalar path tuned to zero conflicts still pays
  the full volume tax; with a matrix atom pinned, swizzle can simply be copied from that instruction
  family's manual standard and the conflicts disappear with it).
- **Issue sequence and latency-hiding relations**: which elementwise / handshake segment runs while
  which MMA chain is in flight. Copied entry by entry from recon's schedule-order transcription,
  marked `[MEASURED]` (the `DEPBAR.LE` immediate is static plain text). Writing only "must overlap"
  without naming the relation is a void contract — there is a legal, easy-to-get-right
  implementation that drains every chain fully and reduces it to zero.
- **Segmented live-set budget** (**delivered as a pair** with the previous item): cut the iteration
  into segments and specify the fragment set each segment allows to coexist and the
  **non-accumulator working-set cap**; including a "conversion must end the upstream live range"
  clause and the spill yield ladder. Transcribed from recon's live-range bill, marked `[MEASURED]`.
  Why the pair: the order decides which lifetimes are forced to overlap, and the segment ledger
  prices that overlap — pinning the order without pinning the segment ledger ends with the schedule
  copied, the registers overflowing, and slower than not overlapping at all.
- **Register-residency policy**: who stays resident in registers across GEMMs.
- **Accumulator precision**; **algorithm skeleton**.

Decisions on **global shape** such as tile shape / warp roles / grid are not in this inventory —
not because they are not frozen (they are M0 hardware-architecture decisions too and are frozen as
usual), but because they are **naturally machine-read scalars** (`block_size` / `grid_size` /
`smem_dynamic`) and go straight into the M0 Gate's architecture-conformance lines as predicates;
this inventory collects the kind that a metric-by-metric scan cannot reveal, hidden in paths and
timing, which is what needs "transcription from recon + contract text".
**Instance form** of the issue-sequence and segmented-live-set-budget pair (a skeleton example; the
values are copied from recon's schedule-order transcription and live-range bill; the
non-accumulator working-set cap is written `<W>`):

> - **Segment A (chain i / chain j in flight)**: issue sequence = chain i and chain j issued back to
>   back, then `wait(1)`; this segment's elementwise work runs while chain j is in flight; allowed to
>   coexist = {acc_i, acc_j, resident accumulators, row scalars, addressing temporaries}; this is the
>   peak segment, **non-accumulator working set ≤ W words** `[MEASURED]`.
> - **Segment B (elementwise → pack)**: the pack / precision conversion **must end acc_i's live
>   range** — no high-precision copy of it may be kept after the conversion (keeping one = the first
>   suspect of a peak over budget).
> - **Over-budget handling (yield ladder)**: r1 <cheapest yield action> → r2 <next choice> → still
>   `spill != 0` → the coexistence budget is falsified; report back to meta to change this contract;
>   the student must not change the order on its own.

Such decisions **must be written in contract form into the earliest structural milestone** (in
this template = M0), with four elements per contract item:

- **Form = a few lines of machine-checked predicates**, not an implementation workload: the
  predicates sit on the observables of A.3.1 (smem bytes, regs/thread, kernel_count, dram traffic
  thresholds, correctness…); the cost of verifying them is "measure it", not "build it first".
- **Traceability**: every predicate traces to a measured signature of the architecture reference
  (`[MEASURED, job N]`), not to deduction or symmetry aesthetics.
- **Expected cost**: state what this commitment occupies (smem bytes / registers / synchronization
  / bandwidth); the budget ledgers of later milestones cite it and do not re-declare it.
- **Closable, revocable**: a predicate that misses goes through A.5's four-state ruling — DEFERRED
  must name the receiving milestone (which planned change is the lever that delivers it) and attach
  a **falsification clause**: if the predicate still does not hold after the receiving side lands →
  the contract is falsified and gets a final ruling on the spot (meta goes back and amends M0 to
  change the contract, or INFEASIBLE), and must not be passed down further.
- **Point of effect = the M0 Gate (a one-time acceptance, not resident monitoring)**: the contract
  predicates are ruled once at the M0 closing gate; after PASS the contract's binding force becomes
  **the freeze of the topology itself** (A.3 item 5), and the predicates' exact values are not
  re-checked. Predicate quantities often depend physically on levers (e.g. smem bytes ≈ buffer size
  × stage count), so in M1+ **numeric drift caused by planned lever tuning is not a breach and does
  not trigger the falsification clause** — the resource account after the drift is taken over by
  that lever decision's own measurement and by the closing gate of its milestone. The falsification
  clause is in effect only in the M0 Gate context (when a predicate cannot be closed). Writing the
  predicates at the values measured under the M0 starting configuration is enough; no margin needs
  to be reserved for future tuning.

**Levers** — **scalar / enum values inside an established topology**, such as the stage count and
the `wg_wait` depth. Two boundaries (the full argument is in the contract inventory above; only the
criteria are given here): swizzle is "choosing a layout mode inside the atom contract" (e.g. 128B vs
64B), while the atom / path itself belongs to the contract; `wg_wait` is "tuning the depth setting
inside the latency-hiding relation" (0/1/2), while the relation itself belongs to the contract —
search cannot converge on the relation, whereas in the reference's schedule-order transcription it
is a free, established fact. M0 gives only starting values (cite the architecture reference or mark
`[ESTIMATE]`), **explicitly left to later milestones to measure round by round**; tuning levers does
not touch A.3's freeze clause. **"Left to later" does not mean free play**: lever tuning itself must
be written as planned decisions of M1+ (five-tuple: which lever, candidate value domain, trigger
condition, trade-off criterion), and the student still executes only actions inside the decision
graph — levers and topology differ only in "whether the value is frozen", not in "whether it is in
the decision graph".

**Basis**: committing the topology late = retrofitting, which is a dead end — by retrofit time the
overlap graph is saturated, the resident quantities no longer fit → spill explodes → it can never
land; the main component of the structural-phase gap to the architecture reference is already fixed
in the earliest structural milestone. Levers are the opposite: their optimum depends on measured
feedback, and locking them early only locks them wrong.

### A.4 ref baseline (an input prerequisite for writing the instance, not a milestone)

**The ref baseline is measured before the instance file is written** (the loop measures the
reference at its start and archives it in `operators/<op>/arch_baseline.json`) and injected as an
input at writing time. It is
**not** a milestone handed to the loop to execute — writing it as M0.x would require the instance
file to write gates anchored to hardware ground truth before it has any hardware ground truth,
which cannot be done.

The baseline contains:

- **ref-perf**: the ref's end-to-end performance (bench script, production shape, median + MFU),
  with the job number; the author writes `[MEASURED, job N]` from it.
- **ref-NCU (conditional)**: **only when the workload has a usable NCU collection path**, the ref's
  NCU per-kernel signature (time_active, tensor pipe %, regs, smem, spill, bank conflicts, stall
  breakdown, waves, etc.), same shape, same seed, as the comparison baseline throughout. There are
  two collection paths; if either is usable it must be collected:
  - **The harness's built-in NCU hook**: needs `src/*.cu` (used to grab kernel names for
    filtering); without a `.cu` the hook reports `no .cu kernel sources` and skips — this is an
    implementation constraint of the hook, **not a limit of NCU's capability**.
  - **A workload-defined ncu suite**: run `ncu` directly inside the job. JIT / Python-DSL kernels
    (CuTe DSL, Triton, etc.) go this way: per-kernel metrics do not depend on a `.cu` existing (CuTe
    DSL has been measured to give the full per-kernel signature and a SASS-level stall table; with
    `CUTE_DSL_LINEINFO=1` source-line mapping is also possible).
  **Only when neither path is usable** (the target toolchain / platform really has no NCU
  equivalent, e.g. an NPU): **the author does not write the NCU anchor or any NCU-anchored gate line
  into the instance** — an inapplicable line never enters the exam, and no runtime ruling is
  needed. Whether a collection path is usable can usually be decided at authoring time; only in the
  rare case where this surfaces only at runtime does A.5's N/A state remove it as a backstop.
- **Every gate number must be anchored from the baseline's `[MEASURED, job]`**; unsourced
  performance bands must not appear in the instance file. Intermediate targets derived from the
  baseline are marked `[ESTIMATE]` with the basis of the derivation written out — `[MEASURED]` is a
  measured fact about the ref, `[ESTIMATE]` is an expectation derived from facts; they must not be
  mixed.
- **There are only three legitimate sources**: measured values of the reference, hardware
  specifications (SM count / smem limit / register limit / peak compute…), or analytic formulas
  derived from those two. **Using the trajectory of the system under test from a past run as an
  anchor is forbidden** — even if it is marked `[MEASURED]`, even for the same operator on the same
  base. That would leak the previous experiment's difficulty curve ("how far the n-th milestone
  should get") into the exam and make this run's score incomparable. Note the distinction:
  **measurements of the reference** are a legitimate source, **scores of a past DUT** are not.
- **There are two references; do not conflate them**:
  - **Architecture reference** = the implementation this milestone teaches. It answers "did you
    reproduce the machine I taught you". **`target_ms` and each milestone's performance ladder are
    anchored to it.**
  - **Library baseline** = the fastest ready-to-call implementation (cuDNN / cuBLAS…). It answers
    "is it better than the off-the-shelf library" — a product metric, **not a stop condition**.
  The two can differ a lot (20% has been observed); anchoring the target to the library baseline
  makes the loop declare success after reproducing only part of the performance. If for some
  operator the two are the same thing, state explicitly that they coincide.
- **Mechanism lines are preferably anchored to the architecture reference's NCU signature**; when
  that is not available they may fall back to the library baseline, but it **must be stated** that
  they anchor to the library baseline, and they are used only as **lower-bound sanity lines** (the
  library baseline is a black box, not necessarily the same algorithm, and its microarchitectural
  profile is not "what you should look like"). When neither is available, use an analytic formula
  (traffic / resource accounting + hardware specification), marked `[ESTIMATE]` with the derivation
  written out.
- **The target is converted from the baseline**: equivalence threshold = `ref_ms / equivalence
  ratio`. The equivalence ratio is **the value configured for this run** (the loop command line's
  `--equivalence-ratio`; default 1.0 = fully reproduce the ref's throughput; self-evolution runs use
  >1, e.g. 1.05 = 5% faster than the ref), **not** a default — the driver has already converted
  `target_ms` with this run's ratio, and that number is what §0 writes; it must not be changed back
  to the 1.0 convention, nor downgraded to a "stretch line". This is the loop's stop threshold, not
  an estimate.

### A.5 M1..Mn — structural milestones

M0 has already fixed and frozen the resource topology, so M1..Mn are **neither foundation nor the
source of the global ceiling** — they are **incremental optimizations** on the frozen topology:
levers measured round by round (stage count / swizzle mode / wg_wait depth setting…, see A.3.2),
**depth tuning** of schedule overlap (the latency-hiding relation is already pinned by the M0
contract; what is tuned here is the wait depth and stage values within the established relation),
layout refinement, numeric fast paths, cross-kernel scheduling, resource fine-tuning. Each item
should be independently A/B-able, with performance acceptance at this milestone's closing gate.

If while writing it turns out a ceiling can only be broken by changing the **resource topology**
(the scope of the A.3.2 contract), **M0 was written wrong**, and the meta loop should go back and
add the M0 decision — **not** quietly retile in M1+, which would turn incremental optimization back
into a rewrite, exactly what A.3's freeze clause guards against. Lever tuning is not in this
category. **Mandatory trigger**: the lever's candidate domain fully swept + the gate line still
short + the value the reference actually uses is exactly the candidate that lost the sweep → meta
must open a verdict experiment to verify the cross-layer coupling (granularity-coupling exception),
and if it holds, amend M0 with a joint decision; "the lever's setting was already measured" is no
reason to keep the original ruling — when a coupling exists, the single-variable measurement is
exactly the wrong evidence.

Each milestone contains:

1. **Goal**: one sentence + why these sub-decisions must be bundled and fixed together in this
   milestone (the dependency argument).
2. **Sub-milestones (construction blueprint, not acceptance lines)**: decompose the milestone into
   sub-steps (goal / change / commit message), marking `⊗` atomic companions and `⇐/⇒` dependencies.
   Sub-steps **have no acceptance line** — delivery granularity is the agent's choice (A.6: anything
   from a single slice to the whole bundle), and the sub-step decomposition is a construction
   reference, not a commit boundary. **Companion group = a set of sub-steps whose performance gains
   are coupled** (typical: a step that enlarges the tile / unrolls a dimension, before the companion
   split / parallelization step lands, temporarily regresses because per-thread resources
   [registers / smem] exceed budget and spill, or occupancy drops) is still circled by the author
   from the dependency graph — it serves two purposes: the natural packing unit when the agent lands
   a bundle; and when the agent lands it split up, a performance regression of the intermediate
   commits inside the group is expected and triggers no ruling, with the performance account
   declared in the ledger and settled at the group's end / the milestone gate (the exemption channel
   of A.6 discipline 2). **Performance acceptance happens only at the milestone's closing gate**;
   sub-steps and companion groups never have a performance gate.
3. **Mandatory Gate**: checklist form; numbers of performance lines must have a source (the
   [MEASURED]/[ESTIMATE] discipline), and relaxed lines of early milestones follow section B. Each
   gate line takes one of the four states of the "Gate three-state ruling" below: a clean PASS
   records one line only (`PASS: <mechanism> [MEASURED, job N] <number>`); only DEFERRED /
   INFEASIBLE expand into a full ruling record, and N/A is removed — details are in that section and
   not repeated here.

   **Mechanism-verification line (mandatory at milestone granularity)**: besides the end-to-end
   performance line, every structural milestone's gate must contain **at least one
   mechanism-verification line** — a metric that separates "the target mechanism has taken effect"
   from "the mechanism did not take effect but the end-to-end number happened to land", compared with
   A.4's ref NCU baseline, with the expected direction and magnitude stated (typical: pipeline
   overlap looks at tensor pipe active %, removing spill looks at local memory accesses ≈ 0, raising
   occupancy looks at warps_active %, smem layout looks at the bank conflict count). **Ruling
   coupling**: end-to-end PASS but the mechanism line misses → this milestone must not be closed as a
   whole; first re-attribute where the end-to-end pass came from; the mechanism line itself is ruled
   by the four states as usual. The mechanism lines are the 1–2 lines distilled in advance from the
   full scan of the advisory NCU Analysis below and promoted to mandatory. For a workload whose NCU
   collection path is unavailable (see A.4): use a substitute metric observable by that toolchain
   (compiler resource report / DSL profiler etc.); with no means of observation at all, do not write
   one. It applies only at milestone granularity; sub-steps have no acceptance line (see A.6
   delivery granularity).
4. **NCU Analysis (advisory)**: a metric-by-metric comparison with A.4's ref NCU baseline, for
   diagnosis only; it does not constrain the choice of the next step.

#### Gate three-state ruling

**Trigger boundary (on demand)**: the DEFERRED / INFEASIBLE rulings + closure record of this section
are an **exception mechanism**, entered only when a gate line objectively cannot be closed as PASS
(the perf / correctness target still missed after N [MEASURED] attempts). A line that can pass
cleanly records one line only and does not go through this section. The trigger is decided by an
objective criterion, not by the agent's choice — a missed line must not be reported as PASS to save
the bookkeeping.

**PASS**: the line meets its target, with [MEASURED, job] evidence attached; settled in one line.

**N/A (structurally inapplicable → removed; a runtime backstop)**: a line that is structurally
inapplicable to this workload **should normally not be written by the author in the first place**
when writing the instance (see the collection-path condition in A.4). **Only when** the
inapplicability of a preset line surfaces only at runtime (e.g. the built-in hook reports
`no .cu kernel sources` and the workload cannot define its own ncu suite either) is it marked
`N/A (workload-structural)` and **removed from the gate list** — not captured, not passed down, no
three-condition analysis, not absorbed in the final ruling. The boundary with DEFERRED: DEFERRED is
"applicable but missed at this stage", N/A is "not applicable to this workload at all".

**DEFERRED**: the line moves to a named later gate and becomes a **mandatory additional line** of
that gate. Admission requires all three of the following to hold **at once** (AND); none may be
missing:

1. **Reachability evidence**: the ref meets this metric within the same resource envelope
   (regs/thread, thread count, grid/waves, smem magnitude the same or worse) [MEASURED]. Without
   this → it must not be deferred; the way out is to keep fixing, or to argue per the Rules-section
   INFEASIBLE clause.
2. **Evidence that it cannot move locally**: [MEASURED] attempts within this milestone's structural
   scope show the metric **does not respond** to this stage's levers (a byte-identical metric or a
   change in the opposite direction both count as not responding; if the metric responds but still
   misses → the lever is still effective, keep fixing, do not defer), with a mechanism explanation.
3. **Named receiving mechanism**: name which **planned** structural change of which later milestone
   is the lever that controls this metric, and explain the mechanism (usually readable directly from
   the ref comparison of condition 1). Without this → either argue per the INFEASIBLE clause, or
   state that the line is attached to the wrong milestone (an author error: fix the instance file
   rather than defer).

Additional disciplines:

- **Every hop of a chained hand-down must re-issue the three conditions**: the same analysis may
  not be passed down two levels unchanged; the mechanism attribution must be updated with new
  evidence.
- **The receiving gate is a falsification experiment**: condition 3 is a falsifiable hypothesis. If
  the metric still does not move after the receiving milestone's structural change lands, the
  hypothesis is falsified, and the receiving gate gives a final ruling directly (PASS or
  INFEASIBLE); it must not be passed down further.
- **Final ruling**: the gate of the last structural milestone must give a final ruling on every
  deferred line — close it, or after a metric-by-metric ref comparison declare INFEASIBLE and stop
  (the line then becomes the diagnosis input of meta's next step). **No work may end with an
  unruled line**. At the final ruling a higher-level end-to-end metric may absorb the ruling of its
  proxy lines (e.g. per-kernel time_active covered by end-to-end wall time); the absorption must be
  noted in the gate closure record.
- **Scope that cannot be deferred**: correctness lines and structural completion items (section B:
  rigid throughout); metrics whose lever this milestone itself owns (deferring with the lever in
  hand = this milestone is not finished).
- **Degraded form without a ref (or with a ref of a different resource shape)**: condition 1
  degrades to a roofline / first-principles argument ([ESTIMATE], weak evidential force), and in that
  case the "receiving gate is a falsification experiment" clause applies mandatorily.

**INFEASIBLE**: see the Rules-section INFEASIBLE clause; its boundary with DEFERRED is drawn by
condition 1 above (a metric the ref meets with the same resources is no basis for INFEASIBLE).

### A.6 Rules

Contains at least:

- **Free delivery granularity**: there is no acceptance line for sub-decisions inside a milestone;
  how many changes one commit carries is the agent's choice — anything from a single slice to the
  whole bundle (a `⊗` companion group can naturally land as one commit). The only real acceptance
  is the milestone's closing gate (A.5: per-line four-state ruling + mechanism-verification line +
  defer), unchanged. **The only exception: the gate row must be committed on its own and must not
  be bundled with any decision** — a gate row is an acceptance action, not a delivery action, and
  mixing in new changes entangles the acceptance reading with an unverified delivery (the driver
  recognizes a gate row by a decision_id containing the substring `gate`; a bundle like
  `D0.12+M0-gate` is illegal: first deliver the final sub-steps, then submit the gate row alone).
  After green (the lineage has passed correctness) two cases do not cost an extra round: when a
  delivery's change writes `[CLOSES: M<k>-gate]` and its whole-machine eval KEEPs, the driver files
  the gate row as a separate row with the same tree and the same reading (still a row of its own;
  what is accepted is the tree just measured, with no new changes); a commit that only touches rows
  in `notes/` (a gate row submitted alone, a decision re-reported unchanged) is not re-measured and
  reuses that tree's last KEEP reading (re-measuring the same tree only adds one more noisy
  reading). Free granularity is bounded by three minimal disciplines:
  1. **Every commit is measured once and its source is marked**: every commit records one
     measurement (except NEUTRAL rounds of the build phase, semantics in POLICY §3), and every
     number is marked with its source per the [MEASURED]/[ESTIMATE] discipline.
  2. **A clear regression beyond noise must not become foundation**: anything that is not an
     improvement is reverted; the committed wall time never gets worse (a watermark ratchet);
     differences within measurement noise (±~1%) are not computed finely and not chased (no
     per-workload thresholds, no multi-sample confirmation below 1% — that would be operating below
     the noise floor). The only exemption: the intermediate states of a companion group declared in
     the ledger (A.5 item 2), whose performance account is settled at the group's end / the
     milestone gate.
  3. **Bundle ledger**: when one commit carries several changes, the commit message / record block
     states the **budgets and knobs shared** by the changes (smem / registers / synchronization /
     tile…), so that reverts and attribution have an account to check; a multi-decision bundle joins
     all its decision_ids with `+` (e.g. `D2.3+D2.4`). **The ruling and bookkeeping granularity of a
     bundle = the bundle**: a commit is naturally atomic, and a REVERT undoes the whole bundle; a
     bundle has one measured reading, and its member decisions have no readings of their own. Two
     corollaries — (a) when a bundle is REVERTed, the dead-end table records **one entry under the
     composite id** (combination + symptom) and **does not ban member decisions individually**: the
     failure evidence points only to this combination, not to a member; (b) the retry precondition
     defaults to **unbundle** — a member retried alone or in a smaller bundle is not blocked by that
     dead-end entry, while retrying the original bundle unchanged requires the precondition to have
     changed. When member-level attribution is needed, the evidence can only come from later rows of
     unbundled retries, not be inferred backwards from the bundle row.
- **[MEASURED]/[ESTIMATE] marking discipline**: every number is marked with its source;
  unsourced expected numbers are forbidden.
- **Anti-plagiarism clause slot**: an explicit list of paths allowed to read / strictly forbidden to
  read + the detection and reject rules.
- **INFEASIBLE clause**: when a gate is proven unreachable, a metric-by-metric ref comparison must
  be given before declaring it and stopping; unlimited retries are not allowed. Note the section B
  boundary: a metric the ref meets with the same resources is no basis for INFEASIBLE; for the
  three-state ruling and DEFERRED admission see A.5 "Gate three-state ruling". **There is only one
  channel for stopping**: meta's diagnosis `cause_class: INFEASIBLE` — the driver stops the whole
  loop on it, and this run finishes with its current best (the stop is final); a declaration written
  only into the milestone is not read by the driver, and the loop will not stop. The driver accepts
  only a request whose reachability counter-proof holds: `reachability` judged BLOCKED and
  `floor_opt > target × 1.20` (see diagnosis flow 1c); when OPEN, the request is refused and the loop
  continues.
- **Tightness rules for performance lines** (section B, verbatim into the Rules section).

### A.7 Decision tree quick reference (optional, recommended)

A troubleshooting table "symptom → which milestone to check", routing gate failures back to the
milestone that has the means to close them.

---

## B. Tightness rules for performance metric lines

Performance metric lines may appear in early milestones in a relaxed form as health checks, but
**the comment must state what exactly is being caught and what is let through**; the tightened
version sits in the milestone that has the means to close it. If a line unexpectedly fails beyond
the relaxed line, issue the three-condition analysis per A.5 "Gate three-state ruling" and rule
DEFERRED or INFEASIBLE; evidence that the ref meets it with the same resources **does not support**
INFEASIBLE.

Performance relaxation during the rebuild phase is distinguished by companion group, not applied
across the board: **intermediate commits inside a companion group** (declared in the ledger) are
expected to regress (let through, not refined; the performance account is settled at the group's
end / the milestone gate); an **independent commit outside a group** (whose conceptual gain can be
estimated as positive) that regresses is a red flag of a dubious implementation → refine it within
this step before freezing, rather than letting it through and freezing.
**Boundary**: for a step whose conceptual gain cannot be estimated in magnitude, "rough
implementation" cannot be told apart from "the concept is useless at this shape", so no forced
judgment — just A/B and accept the result.

Correctness lines and structural completion items are not subject to tightness tiers — rigid /
binary throughout, with no relaxed version and no deferral.
