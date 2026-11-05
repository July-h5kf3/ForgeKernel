# Meta-Milestone — how to distill an operator into a milestone

> This is the **operator-agnostic** distillation policy: the method and vocabulary the meta-agent
> follows when it writes each `operators/<op>/milestone.md`.
> It is the core of cross-operator reuse — for a new operator, reuse this file and rewrite only the
> per-op milestone. This file contains nothing specific to any operator.
>
> **Normative source**: how a milestone is written is governed by `docs/milestones-template.md`
> (the milestones meta-template). **Read that template first, then this file.** This file projects
> the template's A.1–A.7 onto ForgeKernel's machine-read skeleton (§9) and adds the mechanisms
> specific to this repo that the template lacks (decision five-tuple / DAG prefix reuse /
> trajectory harvest / L0 disclosure boundary). Where the two conflict, this file wins.

---

## 0. What a milestone is

A milestone is **the structure of the optimization path to an operator**, not "the operator's code
written as prose". It decomposes "build this high-performance operator" into a **smooth
curriculum**: a sequence of locally easy decisions, ordered by dependency, that a
**low-capability agent** can climb step by step.

What it distills: which decisions matter / in what order / when each one triggers / what it costs /
by what evidence each step counts as done.
What it does **not** distill: what the optimized kernel looks like (the student composes that
itself; hardware interfaces are looked up in `hw_kb/`).

---

## 1. Two buckets (every piece of information belongs to exactly one)

| Bucket | Mark | Meaning | Where it goes |
|----|------|------|------|
| **Decision** | `D` | a design knob (tile, staging, layout, scheduling…), described by the five-tuple | milestone § Decision details |
| **Tactic / interface** | — | "how to use the hardware" (issue an MMA, write a DMA, barriers…) | **`hw_kb/` (operator-agnostic, with example code)** |

> **Hardware / math facts are not a bucket of their own.** They are written **where they are
> used**: the hardware envelope (MMA atom sizes, register / smem limits, target ISA) goes into the
> hardware subsection of §0, or directly into the "Trigger condition / Cost" of the decision it
> constrains; mathematical identities go into §1 Math Spec.

> The bucket boundary is an **ownership** boundary, not a **code** boundary: operator-agnostic
> interface usage belongs to `hw_kb/` (that is its home, and only there is it reused across
> operators), not because a milestone may not contain code — for the code boundary see §4.

---

## 2. Decision five-tuple format

```
Action             —— what to do (one sentence, shape-free)
Purpose            —— which bottleneck it removes / which constraint it satisfies
Trigger condition  —— when it should be done (profiler symptom / predicate / shape range). A "decision" without a trigger condition is empty talk
Cost               —— side effects (more smem / registers / synchronization / cross-kernel contract)
Depends on         —— derived from whom (⇐) | must be committed atomically together with whom (⊗) | is a prerequisite of whom (⇒)
```

- The five labels are machine-checked **exactly as written above** (`Action` / `Purpose` /
  `Trigger condition` / `Cost` / `Depends on`, same capitalization).
- `⊗ atomic co-commit`: changes marked this way **must land in the same commit**; enabling one of
  them alone regresses.
- `⇐ / ⇒`: a derived quantity is a "computation", not a "choice"; a successor must not be done
  before its prerequisite.
- **The `Depends on` line is machine-read, and direction is given by the arrow**:
  `parse_milestone_dag` takes only the `D<x>.<y>` after `⇐` as DAG edges (an id without an arrow
  counts as `⇐`); `⊗` is recorded as a companion group and `⇒` is for readability only — neither
  enters the DAG. A patch's `changed_entries` compute prefix invalidation along the `⇐` edges: a
  missing dependency = wrong prefix reuse; a `⇒` written as `⇐` = a cyclic graph, where one patch
  voids the whole graph. When distilling v1 the validator requires every decision except the first
  three of §3 to have at least one `⇐`, and the `⇐` graph to be acyclic; on patch / resume these
  two, like "`A ⇒ B` but B does not write `⇐ A`", are only logged as warnings (a milestone written
  before the arrow grammar must not be rewritten as "not distilled" because of them).

**Companion group (template A.5)**: a set of decisions whose performance gains are coupled; landing
one of them alone temporarily regresses (typical: enlarge the tile first, while the split /
parallelization step has not caught up → spill). The author marks the group boundary with `⊗`.
Delivery granularity is the student's choice (template A.6): the whole group may land in one
commit; when it lands split up, the intermediate commits inside the group only need coarse hygiene
(**coarse hygiene = compiles + correctness PASS; this is the only definition in the whole
document**; a post-green context, where every commit has a whole-machine eval), a regression is
expected (it must be declared in the ledger), and **performance acceptance always happens at the
milestone's closing gate** — neither groups nor sub-steps have a performance gate.

---

## 3. Decision dependency graph (DAG) + phase order

A milestone is organized as the phase order `M0 → M1 → … → Mk`, **strictly top-down, no
skipping**. The decisions inside one milestone need not be serial, as long as they respect `⊗` and
`⇐/⇒`.

- **M0 = architecture fixed + first runnable version** (template A.3): it delivers **the final
  architecture itself** in one go — the algorithm-contract decisions (kernel split / data flow and
  stationarity / intermediate-tensor ABI / precision contract / redundant-dimension strategy) and
  the hardware-architecture decisions (tile shape / warp roles and thread counts / register budget
  / smem layout / MMA atom layout).
  Within them, the **resource-topology contract** (the smem buffer inventory and each buffer's
  access atoms / issue sequence and latency-hiding relations / segmented live-set budget /
  register-residency policy / accumulator precision / algorithm skeleton, template A.3.2) is fixed
  and frozen in the form of machine-checkable predicates — and the values of two of its items, the
  "buffer inventory" and "register residency", are not given by feel: they come out of the
  row-by-row rulings of the **intermediate consumption-path ledger** below;
  **levers** (stage count / swizzle mode / wg_wait depth setting) are given only as starting
  values, and their tuning is written as **planned decisions** of M1+ (five-tuple), measured round
  by round — tunable, but not outside the decision graph.
  **Granularity-coupling exception (mandatory)**: a stage count counts as a lever only if **the
  supply stage granularity = the consumer atom granularity** (one stage is exactly what the consumer
  side eats in one bite). When recon measures the reference's stage granularity as **finer than**
  the consumer atom (e.g. the reference uses 4 stages × N/2 while the candidate consumer atom eats N
  in one bite), the stage count **must not** be opened as a lever on its own: a single-variable
  sweep to finer stages (consumer granularity unchanged) = twice the handshakes per trip with the
  same total lookahead (total keys in the ring), a pure loss `[MEASURED]` — and that reading has
  **zero evidential weight** on "why the reference chose fine stages"; no setting may be decided on
  it. The correct move is **one joint verdict experiment**: {coarse stage + coarse consumer} (the
  student's current form) against {fine stage + fine consumer} (atom split in half + wait points
  inserted), same-job A/B, with two exits:
  the joint form **wins** → both enter the M0 contract together as a companion group (or the joint
  candidate domain goes through the companion-group channel);
  the joint form **also loses** `[MEASURED: the fine consumer doubles the MMA instruction count and
  the fence/commit overhead; if the wait bottleneck is supply-side arrival rather than consumer
  granularity, the overlap gain is zero and it is worse than finer synchronization alone]` → that
  parameter is set by the student's measurement, **and "the reference chose a fine granularity" is
  recorded as an unexplained structure of the reference** (most likely backed by the supply
  organization on the reference's producer side), handed to recon / §2 as an open question — it must
  not be re-swept over and over in M1+, and an unexplained structure must not be treated as ruled
  out. Splitting across layers is forbidden. General rule: whenever recon finds that the reference
  and the candidate architecture differ on some structural parameter, the transcription must answer
  "at what granularity does the consumer side consume this parameter" — if it cannot, that is a
  recon gap: open a case instead of defaulting to a single-variable lever.
  Sub-steps inside a group have no gate, only a **build self-check** (compiles + a probe verifies
  the local contract; not "coarse hygiene" — whole-machine correctness cannot be measured during the
  build phase, see below). **Build-phase semantics (enforced by the driver)**: the build phase = no
  whole-machine correctness PASS yet since the last workspace reset (judged by whether the lineage
  has turned green, not by milestone number). Build-phase sub-step commits get no whole-machine GPU
  eval and are recorded as `verdict: NEUTRAL` (empty correctness, no performance number); the sha is
  kept as a resume checkpoint; intermediate verification goes through the GPU layout-probe. The
  gate row (a decision_id containing `gate`) **always** has a whole-machine reading, and **must be a
  row of its own, never bundled with any decision** (an acceptance action is not mixed with a
  delivery) — after green, the reading may come from an existing whole-machine eval of the same
  tree: a gate row that only touches `notes/` reuses that tree's last KEEP reading, and when a
  delivery closes with `[CLOSES: M<k>-gate]` the driver files the gate row as a separate row with
  that delivery's reading; the verification deferred during the build phase is settled here in one
  go; after the first PASS the lineage turns green, per-commit eval resumes, and a correctness break
  after green is still a hard REVERT. **The three build-phase failure-path disciplines** (template
  A.3 other rules item 3; the milestone's M0 §3 must write the first one as the first row of the gate
  checklist): ① the prerequisite of the gate row is a **whole-machine smoke certificate** — the
  lineage already holds a passing smoke probe: every kernel in the delivery list lowers successfully
  at its real shape and completes its launch (values are not compared); ② high-risk mechanisms that
  are "invisible to trace, blow up only at launch", such as TMA / mbarrier / register reallocation,
  are **mounted one at a time, with one smoke per mount**; two or more of them going onto the
  machine for the first time without smoke coverage is forbidden (the suspect of a launch crash is
  the last one mounted); ③ **two failures → unbundle** — after the same bundle fails whole-machine
  twice, resubmitting it unchanged is forbidden; it must be downgraded to the conservative form to
  get first green, then swapped back one at a time after green. Failure evidence cannot point to a
  member of the bundle (the corollary of A.6), so resubmitting the same bundle during the build phase
  = repeating a bet with no new information.
  **③'s conservative form is a failure response, not a construction step (downgrades must not be
  pre-scheduled)**: the construction order and a decision's trigger condition must not say "first
  use X (scalar inverse / placeholder / conservative form) to get first green, then switch to this
  decision after green" — once a placeholder is in, the build phase has no performance reading, the
  gate sets no performance line, and "after first green" does not exist when every sub-step is a
  SCAFFOLD (the first green is the gate row), so the placeholder crosses the gate untouched and
  becomes this phase's "best". The validator (`downgrade_schedule_violations`) rejects a downgrade
  clause without a failure trigger; the only allowed form is an entry in a fallback table / the
  Rules section that carries its trigger ("If … FAILs twice in a row, you may temporarily use … to
  get first green, then swap back after green").
  **Placeholders must be recorded (enforced by the driver)**: when the student lands a decision in a
  conservative form, the first sentence of `change` is `[PLACEHOLDER: D<x>.<y> → <real form>]`; on
  swapping back it writes `[RESTORED: D<x>.<y>]`; while the ledger is not cleared, the gate row is
  not accepted (no commit, no eval, re-prompted in place). **The M0 Gate's closing exam is
  correctness + architecture-conformance lines (including the topology-contract predicates) +
  mechanism lines; there is no absolute performance line**. The signature and the topology contract
  are frozen once committed, additive-only afterwards. M0 cannot be deleted or trimmed.
  **M0 gives no implementation-body code, see §4.**
- **Contract candidates are adjudicated on the spot (no defer option)**: representation-choice
  candidates — a cross-kernel numeric domain (e.g. the log base of the softmax statistics), the
  intermediate-tensor ABI, constants pre-folded on the host — as long as reference / recon evidence
  is at hand, must be decided one way or the other while M0 is written: **write it into the
  contract, or explicitly reject it in the decision details with the reason**. They must not be
  scheduled into M1+ as performance optimizations and must not hang on a trigger condition: freezing
  the signature and contract makes "postpone" in effect "give up" (M1+ has no right to change the
  contract), while such choices cost ≈0 at build time and, as a retrofit, cost cross-kernel contract
  risk + queueing — they do not need to be "worth an A/B of their own" to qualify for M0. (Pricing
  them by individual gain, scheduling them late and locking them behind a trigger ends with the
  budget running out before their turn, and afterwards they show up as components of the main gap.)
- **Intermediate consumption-path ledger (a required part of the topology contract)**: for every
  intermediate tensor that is "produced by an MMA and consumed again by an MMA as an operand",
  **one row per consumer**: the accumulator's actual orientation (as given by the M/N ownership
  actually computed, **not the orientation of the mathematical notation** — a transpose sign in the
  formula can cancel against the orientation the registers already carry) / the A-operand
  orientation this consumer needs / ruling / basis. The ruling rule is **asymmetric**:
  **orientations differ → smem (mandatory; the only path for a real transpose; no argument
  needed); orientations agree → RS is the default candidate** — choosing RS requires checking the
  bijection condition (the accumulator fragment aliases into the A-fragment through a pure layout
  view, with no cross-thread reshuffle) and entering it into the register ledger (an RS operand must
  not be overwritten before the corresponding MMA retires); not choosing RS requires stating the
  reason (register budget / scheduling-decoupling trade-off). Both directions must carry a
  falsification clause. An operator with no such intermediate writes "none".
  **Buffer endorsement**: every buffer in the smem buffer inventory must be endorsed by at least
  one "smem" ruling row in the ledger; a buffer without endorsement must not be opened; an
  **exclusionary assertion** such as "can only come from smem" **must carry [MEASURED] or an hw_kb
  certification source**. (An unsourced exclusionary assertion opens a staging buffer that should
  not exist, and it takes a whole round of rework to remove it.)
- **Exchange-face access-atom transcription (a required part of the topology contract)**: for every
  row the ledger rules "smem", that buffer's contract must carry a column with **the access atoms of
  both the write and the read side** — which instruction family each side uses (matrix copy atom
  `STSM/LDSM` / vectorized / TMA-bulk / MMA smem-descriptor direct read), **transcribed entry by
  entry from recon's SASS memory-access footprint**, marked `[MEASURED, job N]`; only when recon did
  not cover that buffer may it be left blank and marked `[INFERRED]`, and then it must be backed by a
  wavefront mechanism line (total `smem_wavefronts_st+ld` ≤ 1.5× the reference).
  Swizzle is an M1+ lever only in the sense of "choosing a mode within the layout family the atom
  gives"; the argument for "why pinning swizzle without pinning the atom is ineffective" is in
  template A.3.2 and is not repeated here.
- **Issue sequence and segmented live-set budget (a required part of the topology contract,
  transcribed as a pair)**: from recon's **schedule-order transcription** (§2d), copy "which
  elementwise / handshake segment runs while which MMA chain is in flight" into the M0 contract,
  marked `[MEASURED]` (the `DEPBAR.LE` immediate is static plain text); at the same time copy the
  segmented live-set budget from the **live-range bill** (§2e) — the fragment set each segment
  allows to coexist, the **non-accumulator working-set cap**, clauses such as "a conversion must
  end the upstream live range", and the spill yield ladder. **Either one without the other is
  void** (the argument about both sides' failure modes is in template A.3.2 and is not repeated
  here). The wait-depth **value** (setting 0/1/2) is still an M1 lever — what is frozen is the
  latency-hiding relation, not the setting.
  **The wait-target relation must be transcribed (beyond the position table)**: every consumer
  wait must state **which tile it waits for, and whether that tile was issued in this round or the
  previous one** (read out from the §2d cross-iteration alignment), and how many beats the prologue
  must pre-warm. If only the segment order is copied without the wait targets, the student places
  the wait at the top of the loop body waiting for the current tile — the order table is right
  segment by segment, correctness / spill are all green, the architecture-conformance lines all
  PASS, and the latency-hiding window is empty (an observed failure mode: `tensor_pct` collapses by
  5–6pp while every memory metric is normal).
  **The backstop mechanism line = a lower bound on `tensor_pct` (a hard gate line)**, which catches
  "no overlap at all".
  **The barrier-stall share is only an advisory attribution signal and must not be written as a hard
  gate line**: the absolute barrier share does not transfer across implementations — a correct
  implementation at the reference's exact configuration can read ~2× the reference (a difference in
  how the toolchain / DSL encodes synchronization is enough), so a band anchored at "reference ×N"
  is unreachable from the start, and its only outcome is REVERTing a correct kernel or forcing meta
  to loosen the line, wasting a step. Its correct use is on the **attribution side**: write it into
  the symptom table and the diagnosis order (barrier share clearly above the reference → first check
  whether the wait-target relation is established, and whether any instruction is actually issued
  between an async operation's issue point and its wait point), for audit / diagnose to cite, not
  for acceptance.
- **Work → CTA dispatch / rasterization order (inter-CTA; its phase is decided dynamically by the
  recon §3b bottleneck ruling)**: this and the **issue sequence** of the previous item (intra-CTA
  pipeline latency hiding, DEPBAR / wait_group, transcribable from SASS, fixed phase) are **two
  orthogonal levels** — this item governs the **grid level**: which CTAs run concurrently, and
  therefore which working set must co-reside in some cache level (it decides DRAM re-read
  amplification / L2 hits; it is about **bandwidth**, not latency). It is **essentially invisible in
  SASS** (decided jointly by the grid→work index arithmetic and the hardware CTA scheduler, not by a
  single instruction), so it is adjudicated on the NCU side (recon §3b).
  **Phase assignment**: recon §3b rules **residency load-bearing** → this item enters the **M0
  topology contract**, and the M0 Gate adds a residency hard line (the student's `dram_pct` / DRAM
  read amplification ≤ within the reference band, `[VERIFY: ncu-derived]`); ruled **not
  load-bearing** (compute-bound, or the working set fits in cache) → it stays a tail lever of M1+,
  and memory metrics are advisory only.
  **This is the only decision in the phase order whose assignment is decided dynamically by shape ×
  hardware** — the phase of every other M0 decision is fixed. Hard-coding "compute pipeline first,
  locality later" holds for compute-bound operators, but for a memory-bound operator "whose dominant
  working set falls out of cache" it pushes the number-one lever behind the budget wall, where it
  never gets its turn before the budget runs out.
- **Inter-kernel intermediate ledger (conditional, only when the reference is split into several
  kernels)**: the correctness of a multi-kernel operator depends first of all on whether the
  intermediates are **handed over across kernels successfully**, and the handover protocol is
  invisible from a single-kernel point of view (NCU per-kernel, SASS disassembly of one kernel). The
  M0 topology contract must contain an intermediate ledger — one row per cross-kernel intermediate:
  producing kernel / storage location and layout / **who allocates, who zeroes** / consuming kernel
  / write-back and read primitives (TMA reduce-add / atomic / plain store) / ordering guarantee
  (blackbox: transcribed from recon §3c and marked `[MEASURED]`; source mode: transcribed from the
  ORACLE source; rows that cannot be measured / read are marked `[INFERRED]` with their basis). And
  **the first step of M0's construction order is fixed as data-flow continuity**: get the whole
  producer–consumer chain through with the simplest implementation, then deepen each component. An
  observed failure mode: with the ledger missing, the student copied the mechanism names but missed
  "who zeroes", and the pipeline's end-to-end output stayed at the initialization constant — the
  max_diff of many consecutive correctness FAILs was bit-identical, and even with every component
  probe PASSing the fault could not be located. This item does not apply to single-kernel operators.
- **The ref baseline is not a milestone** (template A.4): it is measured **before** distillation
  (by the loop at its start, archived as `operators/<op>/arch_baseline.json`) and injected into the
  distill prompt as an authoring input. For how the gate numbers are anchored, see the next item.
  The phase order therefore goes straight from M0 to M1.
- **Two references, split anchoring (template A.4)**: `arch_baseline.json` = **architecture
  reference** (the implementation the milestone teaches), `ref_baseline.json` = **library
  baseline**. **`target_ms` and the performance ladder are anchored to the architecture
  reference**; the library baseline only answers "is it better than the off-the-shelf library" — a
  product metric, not a stop condition. The two references can differ by tens of percent —
  anchoring to the wrong one makes the loop stop and report success after reproducing only part of
  the performance. Mechanism lines are preferably anchored to the architecture reference's NCU
  signature; when that is not available they may be anchored to the library baseline, but this must
  be stated, and they are used only as lower-bound sanity lines.
- **A gate number has only three legitimate sources**: measured values of the reference (the
  architecture reference or the library baseline), hardware specifications, or analytic formulas
  derived from those two. **Using the trajectory of the system under test from a past run as an
  anchor is forbidden** — even if marked `[MEASURED]`, even for the same operator on the same base.
  The distinction: **measurements of the reference** are legitimate, **scores of a past DUT** are
  not (the latter would leak the previous experiment's difficulty curve into the exam). Only a
  first-principles estimate is marked `[ESTIMATE]` with its derivation written out.
  **Single-ref contract**: the whole milestone recognizes only the one reference entity declared in
  loop.toml. In a run where the reference travels with the eval job, same-job ratio lines
  (`vs_ref*`) are legitimate; in a run where the reference does not travel with the job
  (blackbox), the bench measures only the student and **there is no same-job ratio line to write**
  — using any side-by-side comparison in the bench (the deprecated "ruler" concept) as a gate anchor
  is forbidden; the driver discards such readings, and if written they can never take a value.
- **Foundation = the one M0 phase**: the resource topology is fixed and frozen in M0; **this is the
  only source of the global ceiling**; lever values are not foundation (template A.3.2).
- **M1..Mn = incremental optimization on the frozen topology** (levers measured round by round /
  staging / layout / numeric fast paths / cross-kernel scheduling / resource fine-tuning); each item
  can be A/B'd independently, with performance acceptance at its own closing gate. **M1+ must not
  modify M0's topology contract** — if a ceiling can only be broken by changing the topology, M0 was
  written wrong, and meta goes back to add the M0 decision; it is not to be retiled quietly in M1+;
  tuning levers is planned optimization and is not in this category. **The mandatory trigger for "M0
  was written wrong"**: a lever's candidate domain has been fully swept, its gate line is still
  short, and **the value the reference actually uses is exactly the candidate that lost the
  sweep** — this shape almost certainly means the lever has a coupled dual outside the candidate
  domain (the cross-layer split the granularity-coupling exception refers to). Meta must then open a
  verdict experiment (the PROBE_REQUEST channel) to verify the joint form; if it holds, M0 is
  amended with a joint decision; keeping the original ruling on the grounds that "the lever's
  setting was already measured" is not allowed — when a coupling exists, the single-variable
  measurement is exactly the wrong evidence.
- **The phase order stops at Mk; there is no free-play phase outside the decision graph**: once the
  phase order is walked, it is done. Running out of the decision graph still below target = **a
  defect of the milestone, not room for the student to improvise** — the meta loop diagnoses it and
  adds decisions (the driver routes `exhausted` into the meta step just like plateau / stuck).
  Stopping is the driver's job (target reached / plateau / max_iters / meta budget); the milestone
  does not write stop conditions.

---

## 4. Disclosure boundary L0 (what may be given, what may not)

> This experiment measures "can a natural-language optimization decision graph let an agent that
> cannot see the original code reproduce the **performance**" — **whether the starting point is
> correct is not under test**, so a milestone is not strictly code-free. This boundary decides the
> table below.

| Category | May it be written into a milestone? |
|---|---|
| target (measured performance numbers of the ref), equivalence judge | ✅ may be given |
| architecture decisions (kernel split, warp roles, scheduling structure) | ✅ may be given |
| tiling / pipeline parameters (tile_m, stage count, reg budget…) | ✅ may be given (topology given explicitly; levers given as starting values marked tunable, template A.3.2) |
| hardware / math facts (written into the §0 hardware subsection or into the trigger condition / cost of the relevant decision) | ✅ may be given |
| **the kernel implementation body of any milestone** (including M0) | ❌ **forbidden** — M0 now delivers the final architecture itself; giving it would give away the answer |
| short snippets in later milestones that show a **contract / interface shape** (signature, struct, enum, call shape) | ✅ may be given |
| hardware-interface usage code (how to issue an MMA / write a DMA) | ⚠️ belongs in `hw_kb/` (only operator-agnostic code is reused across operators), not in the milestone |
| **the implementation body of the ORACLE's optimized kernel** (copied verbatim, copied with renames, or transcribed line by line into equivalent code) | ❌ **forbidden — red line** |

The rule in one sentence: **give "contracts" and "decisions", not "implementation bodies".**
When unsure, ask yourself: is this code an interface shape the student must have before anything
else, or part of the structure this run wants it to earn by itself? Give the former, not the
latter.

The role of `hw_kb/` does not change: an **operator-agnostic** repository of hardware-interface
examples, with code, admitted after certification by a GPU probe on the G3 branch.

---

## 5. Scope and shapes (hard-coding is allowed)

- A milestone **may target specific shapes** — hard-coding shape parameters is not a leak, it is an
  honest statement of scope. The only leak red line is the ORACLE implementation body of §4.
- It is still recommended to gather the shape-specific values into one `## Worked Example`
  section — it is **this milestone's scope label**, making "which shapes this one covers" obvious
  at a glance and easy to compare when tiling.
- **Scope = the workloads given to this run**: performance is judged by the geometric mean over
  these workloads. A target shape outside this range → build a separate task for that regime and
  write a separate milestone.
- **Generalization = a family of milestones tiling the shape space**. Distillation must be
  faithful (one correct narrow milestone); coverage comes from tiling; being shape-free is a
  side bonus, not a hard goal.

---

## 6. Harvesting the dead-end table from student trajectories (knowledge the source does not have)

The ORACLE is a **final-state snapshot**; it does not contain "what was tried, and why not this
way". The dead-end table can only come from **experiment**:

- Failures the inner student produces while executing (a correctness error, or a decision verified
  to be harmful) → harvested by meta into the milestone's `## 4. Known dead ends` table:
  `decision / symptom / trigger iteration / retry precondition`.
- **The granularity of a dead-end entry = the unit of delivery** (template A.6 discipline 3): a
  single-decision row records the decision id; a bundle row (joined with `+`, e.g. `D2.3+D2.4`)
  records the **composite id** — the collective penalty applies only to this combination, **member
  decisions are not banned individually** (a bundle's failure evidence cannot point to a member),
  and its retry precondition defaults to "unbundle" (a member retried alone or in a smaller bundle
  counts as a changed precondition). When harvesting, the failure of a bundle row must not be split
  and attributed to one member — that would be fabricating evidence; member-level evidence can only
  come from later rows of unbundled retries.
- This is the **most valuable and most compressed** part of a milestone: it keeps later students
  from stepping into the same holes and lets them converge fast.

> **Why there is no symmetric "positive harvest"**: positive knowledge has two existing entries —
> `R2` (present in the ORACLE but missing from the milestone: add that decision) and `G1/G2`
> (decision underdetermined / too coarse: refine it). As for "gains the student discovered by itself
> that are not in the ORACLE either", this repo **has no mechanism that produces them**: there is no
> free-play phase outside the phase order (§3) — delivery granularity is free (template A.6), but
> the student still implements only the decisions inside the decision graph, so a rule-abiding
> student produces no change outside the graph. The real residual leak is **decision
> underdetermination** (a decision's measured gain comes from an implementation choice it did not
> spell out), which belongs to G1 and can only be judged if meta can see the implementation diff —
> which it cannot now. Do not build a harvest channel with no supply for this leak.

---

## 7. Diagnosis taxonomy (meta judging where the student is stuck)

When the student's progress is blocked, it falls into two broad classes (**the operational version
is in `meta/diagnose_prompt.md`**; the meta agent reads trajectories by it):

**G — guidance failure** (the milestone did not guide well; the student cannot make the right move):

| Symptom | Root cause | Fix |
|---|---|---|
| blind search / no direction | G1 underdetermined | add an abstract decision / add a trigger condition (milestone) |
| repeatedly gets the same step wrong, steps too big | G2 too coarse | split the decision finer (milestone) |
| cannot use some hardware interface | G3 hardware interface unknown (operator-agnostic) | add an example to `hw_kb/` (not the milestone) |
| knows the interface but correctness keeps flickering | G4 student noise | tolerate / retry, do not change the milestone |

**R — reach failure** (the guidance is fine; the student follows correctly but tops out below target):

| Symptom | Root cause | Fix |
|---|---|---|
| tops out early, levers within the phase exhausted, some foundation value pinned | R1 foundation ceiling | loosen the foundation decision / parameter (structural patch, foundation first) |
| follows to the end but falls short; present in the ORACLE, missing from the milestone | R2 missing optimization decision | add that upper-level decision |

> G is "cannot move", R is "moved but topped out" — do not mix them up. The FA validation case = R1
> (tile pinned → spill ceiling).
> Finer granularity = each step easier but more steps; calibrate the optimal granularity to the
> capability of the target student.

---

## 8. Smoothness principle (the core signal of milestone quality)

A milestone is a smooth curriculum; **there should be no large-scale restructuring or repeated
trial and error**. Meta reads the inner perf curve:

- **Smooth monotonic rise** = a good milestone (decision order right, foundation right).
- **Deep dip, then recovery** = the foundation milestone is in the wrong order or lacks a
  prerequisite decision → fix the foundation first (see the design doc §4.3, foundation first).
- **Exception**: an intermediate regression inside a `⊗` companion group is **expected** (§2); it
  is not a smoothness defect, do not change the foundation because of it.

---

## 9. Standard skeleton of a per-op milestone (machine-checked)

Every `operators/<op>/milestone.md` should contain the following sections (fixed order;
`validate_milestone` checks them one by one):

```
# <op> milestone
## 0. Problem statement and target          # template A.1: op / shape / ref / hardware + machine-read target_ms + equivalence judge
## 1. Problem contract (Math Spec)          # template A.2: what is computed (contract, not decisions; no smuggled algorithmic priors)
## 2. Decision DAG + phase order            # template A.3/A.5: M0 → M1…Mk (the ref baseline is an authoring input, not a phase)
## 3. Decision details                      # template A.5: five-tuple for every D<x>.<y> + each milestone's mandatory Gate
## 4. Known dead ends                       # trajectory harvest (may be empty in the first version)
## 5. Global correctness / performance gate # correctness tolerance + health criteria for each perf signal
## 6. Rules                                 # template A.6: delivery granularity + three disciplines / marking discipline / anti-plagiarism / INFEASIBLE
## 7. Decision tree quick reference         # template A.7, optional (not validated)
## Worked Example                           # the only section allowed to hold shape-specific values
```

**The mandatory Gate of each milestone in §3** (template A.5, follow it item by item):

- Checklist form; the numbers of performance lines **must have a source** (`[MEASURED, job N]` /
  `[ESTIMATE]`); an unsourced performance band must not appear.
- **At least one mechanism-verification line**: a metric that separates "the target mechanism has
  taken effect" from "the mechanism did not take effect but the end-to-end number happened to land",
  compared with the ref NCU baseline of §2, with the expected direction and magnitude stated.
  Ruling coupling: end-to-end PASS but the mechanism line misses → this milestone must not be closed
  as a whole; first re-attribute where the end-to-end pass came from. For a workload whose NCU
  collection path is unavailable, use a substitute metric observable by that toolchain; with no means
  of observation at all, do not write one.
- **The `[VERIFY:]` mark of M0 decisions** (template A.3.1): every M0 decision is marked with what
  establishes that it was really carried out —
  `ncu-direct` (launch footprint, read directly) / `ncu-derived` (traffic footprint, threshold
  inference) / `sass-derived` (inferred from a static SASS census, see the "operand-source
  verification line" below) / `correctness` (a violation is an error) /
  `none` (no machine-read verification). **Only the first four may be written into the M0 Gate**; a
  decision marked `none` is still written into §3 as build guidance, but not as an acceptance line.
  The reason shares its root with `[MEASURED]/[ESTIMATE]`: **a statement the agent wrote itself is
  not evidence** — writing something that cannot be verified as a gate line amounts to asking the
  agent to check itself against itself (a probe verdict was once self-certified exactly this way).
  The machine-read quantities the current harness offers (`profiler` fields, see `inner/ncu.py`):
  `kernel_count` / `dominant_time_pct` (kernel split), `block_size` (warp roles and thread count),
  `grid_size` (work decomposition; with a known shape the tile can be solved back),
  `regs_per_thread` (setmaxregister), `smem_static` / `smem_dynamic` (stage budget), `tensor_pct` /
  `dram_pct` (mechanism lines, anchored to the ref), `mma_total` / `mma_a_reg` / `mma_a_desc` (MMA
  A-operand source census, `inner/sass_census.py`, reserved for `[VERIFY: sass-derived]`; produced by
  running the census on the SASS dump of the dominant kernel in the operator's ncu task; it is
  trusted only if the digest's `mma_census_kernel` matches the dominant kernel),
  `stall_barrier` / `stall_long_sb` / `stall_mio` (stall-attribution shares — the sync-topology /
  indirect-feed mechanism lines anchor to the ref signature; see the two verification lines below).
  The predicates of the **resource-topology contract** (template A.3.2) are written with these
  machine-read quantities and enter the M0 Gate's architecture-conformance lines; every contract
  item must trace to a measured signature of the architecture reference and state its expected cost;
  a miss goes through DEFERRED with a named receiver + a falsification clause; the topology contract
  must include §3's **intermediate consumption-path ledger** (per-consumer RS/smem ruling + buffer
  endorsement: every block in the smem buffer inventory corresponds to a ledger row; a buffer without
  endorsement must not appear in the contract). **The predicates' force ends at the M0 Gate**: after
  PASS the binding force becomes the topology freeze itself; numeric drift caused by planned lever
  tuning in M1+ (predicate quantities physically depend on levers, e.g. smem bytes change with the
  stage count) is not a breach and does not trigger falsification. Levers (stage count / swizzle /
  wg_wait) are written only as starting values — not frozen, not hard gate lines.
- **Operand-source verification line (conditional, only when the ledger has an RS ruling row)**:
  every ruling row of the intermediate consumption-path ledger must carry the machine tag
  `[SOURCE: RS]` / `[SOURCE: smem]`; wherever there is a `[SOURCE: RS]` row, the M0 Gate must add a
  `[VERIFY: sass-derived]` line (anchored to the `mma_a_reg`/`mma_a_desc` census, stating the
  expected direction for that RS chain: e.g. "the dominant kernel has MMAs with a REG A operand,
  `mma_a_reg ≥ <chain beat count>`"), **paired with a main-kernel `spill ≈ 0` line** — RS trades
  register pressure for smem traffic, and "RS built + spill" is worse than not building it (an
  observed failure mode: the schedule was transplanted without the register economy, spill went from
  0 to the tens of millions, and the whole kernel got slower).
  **Why the register count is not enough** (an observed failure mode): `regs_per_thread` only proves
  the budget was reserved, not that the operand is really fed from the RF — a kernel that met the
  budget while still staging the operand through smem once passed the gate unchanged and was
  finalized, and it took two audits afterwards to locate. **Timing follows DEFERRED semantics**:
  this line may miss at first green (the continuity line comes first; get through with the simplest
  implementation), but before M0 is finalized (closed) it must turn green or be explicitly disposed
  of by the four-state ruling. **Re-ruling exit**: census green but spill red = this operator's
  register ledger does not close; the teacher should re-rule that row as `[SOURCE: smem]` and record
  the basis for the change in the ledger — this template takes no position on "whether RS should be
  used"; it only guarantees that what was ruled is what gets verified.
- **Two-branch health criteria for smem-traffic metrics (§5 discipline)**: whenever an
  `smem_wavefronts` / `bankconf` criterion is set, the cause table must **give both branches at
  once** — the conflict type (`bankconf` over the line → check swizzle / member choice) and the volume
  type (`wavefronts` over the line while `bankconf` is clean → check scalarization / repeated
  landing / redundant staging / a `[SOURCE: RS]` row that was not built), each with its lever. A
  single-branch cause table misdirects a volume disease to swizzle (an observed failure mode: write
  wavefronts ≈2× the reference while bankconf was far below the reference; re-tuning per "check
  swizzle" gained nothing, and the right lever was in the feed path).
- **Residency hard line (conditional, only when recon §3b rules it load-bearing)**: for a
  load-bearing operator the M0 Gate adds a memory line for the inter-CTA dispatch order — the
  student's `dram_pct` (or DRAM read amplification) falls within the reference band,
  `[VERIFY: ncu-derived]`, anchored to the architecture reference's measurement. **An operator ruled
  not load-bearing does not write this line** (gating a compute-bound operator on a non-bottleneck
  would wrongly REVERT a qualified kernel). This gate uses `dram_pct`, which the harness already
  exposes, and does not depend on the L2 hit rate that only the recon §3b side has.
- **Data-flow continuity line (conditional, only when the reference is split into several
  kernels)**: the M0 Gate adds — the end-to-end output responds to input changes (not the
  initialization constant), `[VERIFY: correctness]`. Observable criterion: one eval's per-tensor diff
  is enough to falsify it — **bit-identical `max_diff` across consecutive failed evals = the
  continuity line has not passed** (the digest raises a `constant_output` flag); then fix the
  handover (check the ledger column by column: address / zeroing / timing), not the numerics.
  Single-kernel operators do not write this line.
- **Gate four-state ruling** (template A.5 "Gate three-state ruling" subsection, four states
  including N/A): a clean PASS records one line only; `DEFERRED` must satisfy three conditions at
  once (the ref meets the metric within the same resource envelope, measured / the local levers are
  measured not to respond + a mechanism explanation / a named receiving milestone); `INFEASIBLE` is
  argued per the Rules-section clause; `N/A` is only for removing structurally inapplicable lines as
  a runtime backstop. **The last structural milestone must give a final ruling on every deferred
  line** — the final ruling has only two exits: close it, or declare INFEASIBLE and stop (the line
  then becomes the diagnosis input of meta's next step; the stop takes effect through that step's
  diagnosis `cause_class: INFEASIBLE`, see the template A.6 INFEASIBLE clause).
- **Mechanism-line band discipline**: whenever a stall-class mechanism line (`stall_barrier` /
  `stall_long_sb` etc.) says "within the architecture reference band", **the band width defaults to
  the reference's measured value ×1.25** (one notch above the cross-job noise floor); a wider band
  must have its reason written in the ledger together with the basis for its upper limit. **Why**
  (evidence): a stall line anchored to a reference value of 1.6 was once written as ≤6.0 — the
  student sat comfortably inside the band with twice the reference's waiting and passed the gate
  round after round, and the lesion was exposed only by reading the source side by side. A band
  several times wider than the reference leaves the acceptance line with only ritual value.
- **Sync-topology verification line (mandatory tag + conditional gate line)**: §3 must give a
  machine-tag ruling on the **cross-role supply handoff protocol** — `[SYNC: mbarrier-pipeline]`
  (per-stage mbarrier pipeline, producer N stages ahead) / `[SYNC: named-barrier]` (all-hands
  rendezvous at a named barrier) / `[SYNC: none]` (an operator with no cross-role handoff, recorded
  explicitly as "none"). Ruling discipline: **streaming supply** (the producer keeps feeding the
  consumer's operand stream stage by stage) defaults to `mbarrier-pipeline`; ruling `named-barrier`
  is limited to one-off / low-frequency handoffs and needs a stated reason — an all-hands rendezvous
  puts the supply latency onto the critical path, and the symptom is a **triple**: `tensor_pct` low
  + `dram_pct` low + `stall_barrier` high (both sides waiting on each other, compute and bandwidth
  idle at once — looking at any one alone misdiagnoses it as "compute is weak" or "bandwidth is
  weak").
  Wherever there is a non-`none` `[SYNC:]` ruling, the M0 Gate must add a `[VERIFY: ncu-direct]`
  sync mechanism line: `stall_barrier` within the architecture reference band (anchored to the
  measured signature, marked `[MEASURED, job N]`). The ruling coupling is the same as "end-to-end
  PASS with the mechanism line missing must not close": mechanism line red → first attribute it to
  the handoff protocol, then talk about closing. **Why the end-to-end number is not enough**
  (evidence): a kernel that ruled a streaming KV handoff as an all-hands rendezvous once passed M0
  with the triple of `stall_barrier` 16%, `tensor_pct` 15%, `dram_pct` 9% and was tuned round after
  round — comparing the sync signature should have raised the alarm in the first green round; the
  cost was a whole stretch of budget spent on a register ledger that was not the lesion.
  **Re-ruling exit**: mechanism line red while the handoff really is low-frequency (a constant number
  of times per kernel) → the teacher records the basis for the change in the ledger, re-tags it
  `[SYNC: none]` and withdraws the line — this template takes no position on protocol choice; it only
  guarantees that what was ruled is what gets verified.
- **Indirect-feed verification line (mandatory tag + conditional gate line)**: §3 must give a
  machine-tag ruling on every **gmem feed stream whose addresses depend on data** (gather type: the
  index array must be read first to know what to fetch — of the form `x[idx[i]]`; purely affine /
  contiguous loads are not included) — `[GATHER: prefetched]` (indices prefetched in batches, rows
  decoupled: a row's LDGSTS issue does not depend on "this row's index just came back from gmem") /
  `[GATHER: serial]` (row by row "read index → compute address → fetch" serially; it must state where
  that chain's latency is hidden) / `[GATHER: none]` (an operator with no indirect feed stream,
  recorded explicitly as "none").
  Wherever the ruling is not `none`, the M0 Gate must add a `[VERIFY: ncu-direct]` feed mechanism
  line: **`stall_long_sb` within the architecture reference band** (anchored to the measured
  signature, marked `[MEASURED, job N]`).
  **Wherever `prefetched` is ruled, the contract must also transcribe the issue order of the index
  reads** (index-class LDGs relative to "wait for an empty buffer": batch prefetch = the index loads
  are issued **before** the empty-wait, so their latency overlaps the wait), paired with a
  `[VERIFY: sass-derived]` issue-order line that checks in the produced SASS that the index loads
  are clustered before the fill segment.
  **Why** (evidence): a contract that ruled `prefetched` once delivered only the early publication
  of the mask face, while the indices stayed inline row by row after the empty-wait — stall lines in
  band, tag on file, and the build was the serial form; hoisting the index loads into one batch,
  that single variable, later measured +9%. A `prefetched` tag without an issue-order criterion
  constrains only the vocabulary, not the structure.
  The symptom is a third triple: `stall_long_sb` many times over the reference + `tensor_pct` low +
  `stall_barrier` / DRAM bytes **within band** — together with the sync-topology triple (barrier
  dominant) it forms a differential diagnosis: barrier high = protocol disease, long_sb high = serial
  feed-chain disease; both show low tensor, so tensor alone cannot tell them apart.
  **Why the end-to-end number is not enough** (evidence): a kernel with a row-serial gather once
  passed M0 with `stall_long_sb` at 12.7× the reference (8.91 vs 0.7) and was tuned round after round
  — sync lines, registers and byte volumes all in band; the only order-of-magnitude anomaly had no
  gate line anchoring it, and a whole stretch of budget was spent on innocent suspects.
  Ruling coupling and re-ruling exit as in the previous item: mechanism line red → first attribute
  it to the feed form, then talk about closing; if it really is structural (e.g. the index stream
  itself can be fully hidden) → record the basis in the ledger, re-rule and withdraw the line.
- **Consumer-phase verification line (mandatory tag + conditional transcription duty)**: §3 must
  give a machine-tag ruling on **the relative timing of multiple same-role consumer warpgroups** —
  `[PHASE: ping-pong]` (staggered mutual cover: each WG's scalar / elementwise segment lies under the
  other side's in-flight MMA chain, maintained by one-way signals with **asymmetric placement** —
  each signal states who is arrive-only, who blocks, and whether it waits for this trip's operand or
  the previous trip's) / `[PHASE: convoy]` (lockstep; it must state why the scalar segments need no
  cover from the other side — e.g. the scalar share is negligible) / `[PHASE: none]` (a single
  consumer WG, or an operator with no scalar segment, recorded explicitly as "none"). Wherever
  `ping-pong` is ruled, the M0 contract's "issue sequence and latency-hiding relations"
  transcription must contain a **phase contract** entry (per-signal table: direction / arrive side
  and blocking side / this trip or the previous one; how the prologue staggers the two WGs).
  **Why the ruling must be explicit** (evidence): this is the fourth lesion form, and with the two
  triples above it completes the differential — `tensor_pct` low while `stall_barrier`,
  `stall_long_sb` and DRAM bytes are **all in band** (both WGs are waiting "normally", no single
  anomaly) = a phase disease. A kernel that transcribed the reference's four asymmetric one-way
  signals into a symmetric rendezvous once passed every stall line in this form and was tuned round
  after round, until meta located it by elimination — a static SASS transcription can copy the
  existence and count of the barriers but not the asymmetry of their placement, so the
  transcription duty must be forced by the ruling tag.
  The coverage and accounting duties of the phase contract (two, both lessons from evidence):
  ① **The contract must cover every cross-WG signal**, including the rendezvous of statistic
  reductions (row max / row sum exchange) — the intuition that a reduction rendezvous is not a
  "phase signal" is exactly why it escapes transcription; and it must give an explicit ruling on
  **where the statistics are reduced** (reduced every trip vs reduced once, deferred to the
  epilogue): a statistic that only enters the epilogue but is reduced every trip in the loop is pure
  overhead plus one extra all-hands rendezvous.
  ② **Per-trip blocking rendezvous count line**: the contract writes down one number — the total
  number of blocking rendezvous (arrive-and-wait) a single consumer WG executes within one trip —
  anchored to the reference value transcribed by recon (also one number). A per-signal phase
  contract can be "asymmetric" on every line while the total number of blocking points is still
  twice the reference — compliant line by line, over budget in total; this number is what catches
  the total. It is a sass-derived static quantity and adds no NCU line.

**What the machine check rejects** (`src/forgekernel/milestone.py`): a missing section, a §3
decision missing its five-tuple, no `D<x>.<y>`, no `[MEASURED]`/`[ESTIMATE]` mark, §3 without a
"mechanism-verification line", a `[SOURCE: RS]` row missing the `[VERIFY: sass-derived]` line or the
paired spill line, an RS ruling that exists only in prose without `[SOURCE:]` tags,
§3 without a `[SYNC:]` handoff-protocol ruling, a non-`none` `[SYNC:]` ruling missing a
`stall_barrier` **`[VERIFY:]` mechanism line** (naming the metric in prose does not count),
§3 without a `[GATHER:]` feed-form ruling, a non-`none` `[GATHER:]` ruling missing a `stall_long_sb`
**`[VERIFY:]` mechanism line**, a `[GATHER: prefetched]` ruling whose **own decision block** lacks
the index **issue order** transcription,
§3 without a `[PHASE:]` consumer-phase ruling, a `[PHASE: ping-pong]` ruling whose **own decision
block** lacks the per-signal **phase contract** transcription or a **numeric blocking rendezvous
count** accounting line (the two are reported independently, in no particular order),
a ruling value outside the vocabulary (values are exact lowercase words: `mbarrier-pipeline` /
`named-barrier` / `prefetched` / `serial` / `ping-pong` / `convoy` / `none`; case is normalized, and
an out-of-vocabulary spelling is reported separately as an "unknown value" rather than misreported as
"no ruling").
Rulings are a **per-decision duty**: every decision involving a cross-role handoff / indirect feed /
multi-consumer phase carries its own tag, and a tag's transcription duty is checked only **inside
its own decision block** — another decision's prose does not do its homework for it.
The validator can only audit to this depth: a mixed milestone that tags an affine decision `none`
while the real gather decision is left untagged cannot be caught by the machine check; audit and
human review backstop it.
The three ruling lines are mandatory for **fresh distillation** (authoring validation); resume /
patch validation exempts an old milestone that carries **no tags at all** from the presence duty
(otherwise a template upgrade would judge every in-flight run "not distilled" and trigger
re-distillation, destroying the accumulated meta patches and violating resume's contract of not
re-measuring baselines); tags that are present carry their full duties in every mode.
The violation list is sent back to you with a request to rewrite, until it conforms or the rounds
run out.

---

## 10. Template A.x → this skeleton

| `docs/milestones-template.md` | This skeleton | Notes |
|---|---|---|
| A.1 Problem statement | §0 | additionally requires the machine-read `target_ms:` (the driver's stop SSOT) |
| A.2 Math Spec | §1 | the "what is computed" contract, no smuggled algorithmic priors |
| A.3 M0 | head of the §2 phase order / §3 M0 decisions | architecture fixed (two groups of decisions: algorithm contract + hardware architecture); **no implementation body** (see §4) |
| A.3.1 `[VERIFY:]` marks | §3 every M0 decision | only ncu-direct / ncu-derived / correctness / sass-derived may enter the M0 Gate |
| A.3.2 Topology contract and levers | §3 M0 decisions + M0 Gate | contract predicates go into the architecture-conformance lines (traceability + cost + falsification clause); levers get only starting values, not frozen |
| A.4 ref baseline | (not in the skeleton) | measured before distillation and injected as an authoring input; not a milestone |
| A.5 M1..Mn + Gate + mechanism-verification line | §2 DAG / §3 details | decisions are carried by the five-tuple |
| A.5 Gate four-state ruling | §3 each milestone's gate | follow it |
| A.5 correctness gate | §5 | global tolerances and health criteria |
| A.6 Rules | §6 | the anti-plagiarism clause is written per the L0 of §4 |
| A.7 Decision tree quick reference | §7 | optional |
| A.5/B tightness rules | inside §6 | verbatim into the Rules section |
| — | §4 Known dead ends | no corresponding section in the template: its input is the harvest of student trajectories |
