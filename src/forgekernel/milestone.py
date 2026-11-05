"""Per-operator milestone format validation.

A milestone is a structured decision graph, not free prose. This enforces the
fixed schema (meta_milestone.md §9, the ForgeKernel projection of the
milestones template in docs/milestones-template.md): the required sections in
order, the five-tuple format for the decisions in §3, and the two template
disciplines that are cheap to check mechanically — every number carries an
evidence marker, and every milestone gate carries a mechanism-verification line.
Used to gate what the meta agent writes so a patched milestone stays
machine-navigable rather than free-form.
"""

from __future__ import annotations

import re

from forgekernel.resume import Decision

# Required section markers, in order (meta_milestone.md §9). "## 7. Decision tree
# quick reference" is optional (template A.7 is "recommended", not mandatory) and so
# is not listed. The phase
# order ends at Mk: ForgeKernel measures whether the decision graph carries the
# student, so a milestone that runs out of decisions below target is the meta loop's
# problem to fix, not the student's to improvise around (meta_milestone.md §3).
REQUIRED_SECTIONS = [
    "## 0.",   # Problem statement and target
    "## 1.",   # Problem contract (Math Spec)
    "## 2.",   # Decision DAG + phase order
    "## 3.",   # Decision details (incl. each milestone's mandatory Gate)
    "## 4.",   # Known dead ends
    "## 5.",   # Global correctness / performance gate
    "## 6.",   # Rules
    "## Worked Example",
]

# The decisions live in §3; everything that can follow it bounds the section body.
# Matched as whole heading lines, not substrings: "### 4. M2 pipeline" contains
# "## 4." and would otherwise cut §3 in half, dropping every decision after it from
# both the DAG and the gate checks — silently, since fewer decisions just means
# fewer things to complain about.
DECISIONS_SECTION = "## 3."
DECISIONS_ENDS = ("## 4.", "## 5.", "## 6.", "## 7.", "## Worked Example")


def _heading_at(text: str, marker: str, start: int = 0) -> int:
    """Offset of ``marker`` used as a heading (at line start), or -1."""
    m = re.compile(rf"(?m)^{re.escape(marker)}").search(text, start)
    return m.start() if m else -1

# Every decision in §3 must be described by this five-tuple. Exact-case labels
# (the template shows them as `Action / Purpose / Trigger condition / Cost /
# Depends on`); "Trigger" is a substring of "Trigger condition" on purpose.
FIVE_TUPLE = ["Action", "Purpose", "Trigger", "Cost", "Depends on"]

# Template A.4/A.6: every number carries its provenance. "[MEASURED, job 12]" and
# "[ESTIMATE]" both match on the opening token, so the job ref stays free-form.
_EVIDENCE_MARKER = re.compile(r"\[(?:MEASURED|ESTIMATE)")

# Template A.5: each structural milestone's gate needs a mechanism-verification line —
# a metric that separates "the mechanism engaged" from "the end-to-end number happened
# to land". Required per milestone, not once per file. Matched case-insensitively and
# with either a hyphen or a space: authors capitalise it as a label
# ("- Mechanism-verification line: ...").
MECHANISM_LINE_TERM = "mechanism-verification line"
_MECHANISM_LINE = re.compile(r"mechanism[- ]verification line", re.I)

# A performance claim: a number carrying a latency/throughput unit. Percentages are
# deliberately excluded — they are as often a target ("occupancy >=80%") as a
# measurement, and flagging them would put the patch agent in a re-prompt loop it
# cannot exit (a gate that cannot be satisfied rolls every patch back forever).
_PERF_NUMBER = re.compile(r"\d[\d.,]*\s*(?:ms|us|µs|TFLOPs?|TFLOPS|TF/s|GB/s)\b", re.IGNORECASE)

# entry-id tokens that a decision can depend on (decision ids).
_ENTRY_ID = re.compile(r"D\d+\.\d+")
# A §3 decision header in any markdown shape the teacher might reach for:
# "### D1.1 tile choice", "**D1.1 …**", "- **D1.1 …**", "| D1.1 |", bare "D1.1 …".
# Pinning one shape is not an option — nothing in the template or the prompts says
# which to use, and the one real distillation artifact in the repo wrote all 21 of
# its decisions as "### D0.1". Missing them is silent and total: no decision blocks
# means the five-tuple, dependency and mechanism-line checks all have nothing to
# iterate over, so a prose milestone passes clean.
_DECISION_HEADER = re.compile(r"(?m)^[#*\-|>\s]*\**\s*(D\d+\.\d+)\b")
# a decision's dependency line, the SSOT for the DAG. Accepts every shape the
# teachers have reached for: "- Depends on: D1.1", "Depends on: ⇐ D0.1 | ⊗ D0.15 |
# ⇒ D0.11", "**Depends on** —— ⇒ D0.2 / D0.3". The bold form was invisible to the
# colon-only pattern (one milestone had all 13 M0 decisions parsed as
# dependency-free). Case-sensitive, and only followed by a separator, so prose
# such as "depends on the tile" is never read as a dependency line.
_DEP_LINE = re.compile(r"Depends on\**\s*[:：—–-]+\s*(.+)")
# The template's three arrows (meta_milestone §9): ⇐ derives-from, ⊗ atomic
# companion, ⇒ is-prerequisite-of. Only ⇐ is a DAG edge. Reading every id on
# the line as depends_on turned ⇒ into a reverse edge and ⊗ into a symmetric
# one, so the "DAG" was cyclic and a closure over it covered the whole file
# (a grouped-gemm patch of two entries closed over 28/28 decisions).
_DEP_TOKEN = re.compile(r"(⇐|⊗|⇒|D\d+\.\d+)")
_ARROW_UPSTREAM, _ARROW_COMPANION, _ARROW_DOWNSTREAM = "⇐", "⊗", "⇒"


def parse_dependency_line(line: str) -> tuple[set[str], set[str], set[str]]:
    """(upstream ⇐, companions ⊗, downstream ⇒) decision ids named on a
    `Depends on` line.

    Ids before any arrow are legacy shorthand for ⇐ (the pre-arrow milestones
    wrote "Depends on: D1.1"). An id is attributed to the most recent arrow on the line,
    so "⇐ D0.1。⇒ D0.7" and "⇐ D0.1 | ⇒ D0.7" read the same."""
    up, comp, down = set(), set(), set()
    bucket = up
    for tok in _DEP_TOKEN.findall(line):
        if tok == _ARROW_UPSTREAM:
            bucket = up
        elif tok == _ARROW_COMPANION:
            bucket = comp
        elif tok == _ARROW_DOWNSTREAM:
            bucket = down
        else:
            bucket.add(tok)
    return up, comp, down


def dependency_warnings(text: str) -> list[str]:
    """Advisory (not violations): the DAG-shape findings of the resume/patch mode.

    ⇒ claims the other side does not echo as ⇐, ⇐ holes on non-root decisions,
    ⇐ cycles. In ``validate_milestone(authoring=True)`` the last two are
    violations; here they are logged, never re-prompted and never read as "not
    distilled": both resume gates translate a violation into re-authoring, and
    the milestones written before the arrow grammar (three of twelve checked: a
    dependency-free D0.11, a D0.13, a D0.13 ⇐ D0.5 ⇐ D0.13 cycle) would have had
    v1 re-distilled over their accumulated patches."""
    warns = []
    sec = _section_body(text, DECISIONS_SECTION, DECISIONS_ENDS)
    blocks = _decision_blocks(sec)
    lines = {}
    for did, block in blocks:
        m = _DEP_LINE.search(block)
        lines[did] = parse_dependency_line(m.group(1)) if m else (set(), set(), set())
    for did, (_up, _comp, down) in lines.items():
        for other in sorted(down):
            if other in lines and did not in lines[other][0]:
                warns.append(f"§3 {did} declares ⇒ {other} but {other} does not list ⇐ {did}")
    warns += _dag_shape_findings(blocks, {d: {r for r in up if r != d}
                                          for d, (up, _c, _d) in lines.items()})
    return warns


def _dag_shape_findings(blocks, upstream: dict[str, set[str]]) -> list[str]:
    """⇐ holes on non-root decisions and ⇐ cycles, as message lines.

    A non-root decision with no ⇐ edge is a DAG hole: prefix reuse would keep
    its checkpoint through any patch of what it actually builds on. The first
    three decisions of §3 are the foundation and may derive from nothing."""
    out = []
    for did, _ in blocks[3:]:
        if did in upstream and not upstream[did]:
            out.append(f"§3 {did} names no upstream (⇐) dependency — a non-root decision "
                       f"must derive from something; ⊗/⇒ alone do not build the DAG")
    for cyc in _dependency_cycles(upstream):
        out.append(f"§3 dependency cycle: {' ⇐ '.join(cyc)} — ⇐ must point upstream only")
    return out


def _section_body(text: str, start: str, ends: tuple[str, ...]) -> str:
    i = _heading_at(text, start)
    if i < 0:
        return ""
    rest = text[i + len(start):]
    cut = len(rest)
    for e in ends:
        j = _heading_at(rest, e)
        if j >= 0:
            cut = min(cut, j)
    return rest[:cut]


def _decision_blocks(sec: str) -> list[tuple[str, str]]:
    """[(decision_id, block_text)] for each decision in the decisions section."""
    headers = list(_DECISION_HEADER.finditer(sec))
    out = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(sec)
        out.append((m.group(1), sec[m.end():end]))
    return out


def _milestone_spans(blocks: list[tuple[str, str]]) -> dict[str, str]:
    """Milestone prefix ("D2") -> the concatenated text of its decisions.

    Decision ids encode their milestone (``D<milestone>.<step>``) and §2 orders them
    by phase, so contiguous same-prefix blocks are one milestone's span — which is
    where that milestone's gate has to live.
    """
    spans: dict[str, list[str]] = {}
    for did, block in blocks:
        spans.setdefault(did.split(".")[0], []).append(block)
    return {k: "\n".join(vs) for k, vs in spans.items()}


# Orchestration adjudication tags (meta_milestone's sync-topology / indirect-feed /
# consumer-phase verification lines). The value class deliberately admits underscores and any case: a typo
# like "mbarrier_pipeline" or "Prefetched" must be CAPTURED and reported as an
# out-of-vocabulary value — a regex that refuses to match it would misreport
# "no adjudication at all", which is fail-closed but misdiagnosed (the author
# wrote one), and a case-sensitive equality compare would silently skip the
# very obligations the tag exists to trigger. Values are normalized to
# lowercase before any comparison; the vocabulary is exact words.
_SYNC_TAG = re.compile(r"\[SYNC:\s*([A-Za-z_-]+)\s*\]")
_GATHER_TAG = re.compile(r"\[GATHER:\s*([A-Za-z_-]+)\s*\]")
_PHASE_TAG = re.compile(r"\[PHASE:\s*([A-Za-z_-]+)\s*\]")

_SYNC_VOCAB = ("mbarrier-pipeline", "named-barrier", "none")
_GATHER_VOCAB = ("prefetched", "serial", "none")
_PHASE_VOCAB = ("ping-pong", "convoy", "none")


def _tag_scopes(sec: str, pat: "re.Pattern[str]") -> list:
    """[(lowercased_value, scope_text)] for each tag occurrence in the section.

    The scope is the enclosing decision block when the tag sits inside one, else
    the whole section. Obligations attached to a tag (phase contract, blocking
    rendezvous count, issue order) are checked INSIDE its scope: a mixed milestone must not satisfy one
    decision's transcription duty with prose that belongs to another decision —
    the whole point of per-decision adjudication.
    """
    headers = list(_DECISION_HEADER.finditer(sec))
    spans = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(sec)
        spans.append((m.start(), end))
    out = []
    for tm in pat.finditer(sec):
        scope = sec
        for a, b in spans:
            if a <= tm.start() < b:
                scope = sec[a:b]
                break
        out.append((tm.group(1).lower(), scope))
    return out


def _verify_line(sec: str, term: str) -> bool:
    """True when `term` appears on a line that is a real verification line
    (carries a [VERIFY: channel), not merely mentioned in prose. "may push up
    stall_barrier" or "stall_barrier does not apply to this op" must not satisfy a
    gate-line obligation — the repo's own lesson is "written down but not enforced
    by a gate line = not written"."""
    return any(term in ln and "[VERIFY:" in ln for ln in sec.splitlines())


def validate_milestone(text: str, *, authoring: bool = False,
                       baseline: str | None = None) -> list[str]:
    """Return a list of format violations (empty list == conforms to the schema).

    ``authoring=True`` is the strict mode for freshly distilled milestones: the
    orchestration adjudications ([SYNC:]/[GATHER:]/[PHASE:]) are mandatory.
    The default is the resume/patch mode, where a milestone carrying NONE of the
    three tag kinds is grandfathered on the presence checks: milestones written
    before the tags existed must not read as "not distilled yet" — both resume
    gates translate any violation into re-authoring, which would overwrite the
    accumulated meta patches and re-measure baselines a resume contract forbids.
    Tags that ARE present carry their full obligations in every mode (so a patch
    that introduces one is held to it). Residual hole, accepted: a patch that
    deletes every tag from a new-era milestone passes this mode — the patch
    landing rules, not this validator, are the place to hold deletions.

    ``baseline`` (patch validation only): the milestone text before the patch.
    The gate-line hygiene rules then apply to gate lines the patch ADDED, never
    to lines the baseline already carried — a rule that reached back into
    existing lines would reject a patch that priced one of fifteen shape lines
    because the other fourteen are still unpriced, and would make every resume
    gate read a pre-rule milestone as "not distilled yet".
    """
    v: list[str] = []
    if not text.lstrip().startswith("# "):
        v.append("missing title header (# <op> milestone)")

    # presence AND order: a milestone with the right headings in the wrong sequence is
    # not the construction blueprint the phase order describes
    found: list[tuple[str, int]] = []
    for sec in REQUIRED_SECTIONS:
        i = text.find(sec)
        if i < 0:
            v.append(f"missing required section: {sec}")
        else:
            found.append((sec, i))
    misordered = [b for (a, ia), (b, ib) in zip(found, found[1:]) if ib < ia]
    if misordered:
        v.append(f"sections out of order (must follow meta_milestone.md §9): {misordered}")

    if re.search(r"(?m)^#+\s.*YOLO", text):
        v.append("free-optimization section present — the phase order ends at Mk "
                 "(meta_milestone.md §3); the decision graph must carry the whole run")

    sec3 = _section_body(text, DECISIONS_SECTION, DECISIONS_ENDS)
    blocks = _decision_blocks(sec3) if sec3 else []
    defined = {did for did, _ in blocks}
    upstream: dict[str, set[str]] = {}
    # No parsed decisions is a failure, not a clean bill: every per-decision check
    # below iterates over `blocks`, so an empty list silently satisfies all of them.
    # It also means parse_milestone_dag sees no DAG, and prefix reuse invalidates
    # nothing. A milestone with no decisions is not a decision graph.
    if not blocks:
        v.append("§3 has no parseable decisions (expected headers like 'D<x>.<y> <name>') "
                 "— without them the five-tuple, dependency and mechanism-line checks "
                 "cannot run and the DAG is empty")
    for did, block in blocks:
        missing = [k for k in FIVE_TUPLE if k not in block]
        if missing:
            v.append(f"§3 {did} missing five-tuple keys: {missing}")
        dep_line = _DEP_LINE.search(block)
        up, comp, down = (parse_dependency_line(dep_line.group(1)) if dep_line
                          else (set(), set(), set()))
        for ref in sorted(up | comp | down):
            # a dangling dependency silently reads as "no dependency", so prefix-reuse
            # under-invalidates and the loop resumes from a checkpoint that is stale
            if ref != did and ref not in defined:
                v.append(f"§3 {did} depends on undefined decision {ref}")
        upstream[did] = {r for r in up if r != did}
    if authoring:
        # DAG shape (⇐ holes, ⇐ cycles) is held strictly on a fresh v1 only. In
        # the resume/patch mode it is advisory (dependency_warnings): the same
        # grandfathering as the tag presence checks above — a pre-arrow
        # milestone must not read as "not distilled yet".
        v += _dag_shape_findings(blocks, upstream)

    for milestone, span in _milestone_spans(blocks).items():
        # M0 used to be exempt, back when it delivered a naive implementation and
        # had nothing structural to verify. It now freezes the architecture, which
        # makes it the one milestone whose mechanism line matters most: everything
        # downstream is fitted inside the choices M0 pins.
        if not _MECHANISM_LINE.search(span):
            v.append(f"§3 milestone {milestone} has no \"{MECHANISM_LINE_TERM}\" "
                     f"(template A.5: every structural milestone's gate needs one)")

    # Sync-topology adjudication (meta_milestone "sync-topology verification line"): the handoff
    # protocol must be tagged, and a non-none tag must be verifiable. A streaming
    # handoff adjudicated as an all-hands rendezvous once shipped with the full
    # triple symptom (tensor low + dram low + stall_barrier high) and passed M0
    # on end-to-end numbers alone — the sync signature line is what fires there.
    if sec3:
        # Out-of-vocabulary values are their own violation, for every kind: the
        # machine vocabulary is exact words, and a near-miss ("pipeline",
        # "mbarrier_pipeline") must be named as such — not misreported as a
        # missing adjudication, and never silently accepted.
        for kind, pat, vocab in (("SYNC", _SYNC_TAG, _SYNC_VOCAB),
                                 ("GATHER", _GATHER_TAG, _GATHER_VOCAB),
                                 ("PHASE", _PHASE_TAG, _PHASE_VOCAB)):
            for val, _scope in _tag_scopes(sec3, pat):
                if val not in vocab:
                    v.append(f"§3 [{kind}: {val}] is outside the machine "
                             f"vocabulary ({' / '.join(vocab)}) — adjudication "
                             f"values are exact lowercase words")

        sync = _tag_scopes(sec3, _SYNC_TAG)
        if not sync and authoring:
            v.append("§3 has no [SYNC:] handoff adjudication (mbarrier-pipeline / "
                     "named-barrier / none) — the cross-role supply protocol must be "
                     "adjudicated explicitly (meta_milestone: sync-topology "
                     "verification line)")
        elif any(t != "none" for t, _ in sync) and not _verify_line(sec3, "stall_barrier"):
            v.append("§3 adjudicates a cross-role sync topology ([SYNC:] != none) but "
                     "no [VERIFY:] gate line anchors stall_barrier (a prose mention "
                     "is not a gate line) — an unverified handoff protocol can "
                     "serialize the supply path and still pass on end-to-end numbers")
        # Indirect-feed adjudication (meta_milestone "indirect-feed verification
        # line"): a gather
        # whose row issue serializes on its own index loads once shipped with
        # stall_long_sb at 12.7x the reference while every other meter stayed
        # in-band — the long_sb line is what fires there.
        gather = _tag_scopes(sec3, _GATHER_TAG)
        if not gather and authoring:
            v.append("§3 has no [GATHER:] feed adjudication (prefetched / serial / "
                     "none) — every data-dependent gmem feed must be adjudicated "
                     "explicitly (meta_milestone: indirect-feed verification line)")
        elif any(t != "none" for t, _ in gather) and not _verify_line(sec3, "stall_long_sb"):
            v.append("§3 adjudicates an indirect gmem feed ([GATHER:] != none) but "
                     "no [VERIFY:] gate line anchors stall_long_sb (a prose mention "
                     "is not a gate line) — a row-serial index chain can expose "
                     "full memory latency per row and still pass on end-to-end "
                     "numbers")
        # A prefetched verdict without the index-load ordering transcribed is
        # how the serial form ships under the prefetched label: one contract
        # delivered only the mask face early while every index load stayed
        # inline after the empty-wait — stall lines in band, label on file,
        # structure serial. The ordering (index loads issued BEFORE the
        # empty-wait, latency overlapped) is a static, sass-checkable fact.
        # Checked inside the tag's own decision block: another decision's prose
        # must not satisfy this decision's transcription duty.
        for t, scope in gather:
            if t == "prefetched" and "issue order" not in scope.lower():
                v.append("§3 adjudicates [GATHER: prefetched] but its decision "
                         "block transcribes no issue order (index loads relative to the "
                         "empty-wait, with a sass-derived check) — without it the "
                         "row-serial form ships under the prefetched label with "
                         "every stall line in band")
        # Consumer-phase adjudication (meta_milestone "consumer-phase verification
        # line"): two
        # same-role consumer warpgroups whose scalar segments coincide idle the
        # tensor pipe with EVERY stall meter in band — the disease is invisible
        # to stall_barrier/stall_long_sb lines and shows only as tensor_pct
        # that will not move.  The enforceable half is the transcription: a
        # ping-pong verdict without the asymmetric signal contract written down
        # is how the symmetric-rendezvous form ships.
        phase = _tag_scopes(sec3, _PHASE_TAG)
        if not phase and authoring:
            v.append("§3 has no [PHASE:] consumer-phase adjudication (ping-pong / "
                     "convoy / none) — how same-role consumer warpgroups cover each "
                     "other's scalar segments must be adjudicated explicitly "
                     "(meta_milestone: consumer-phase verification line)")
        # The two ping-pong obligations are independent checks on purpose: a
        # rewrite budget is finite, and reporting them one-per-round (the elif
        # form) burns a round teaching the author the second duty only after
        # the first is fixed.
        for t, scope in phase:
            if t != "ping-pong":
                continue
            if "phase contract" not in scope.lower():
                v.append("§3 adjudicates [PHASE: ping-pong] but its decision block "
                         "transcribes no phase contract (per-signal table: direction / "
                         "arrive-only vs blocking side / this-pass vs last-pass "
                         "operand) — without it the symmetric-rendezvous form "
                         "passes every stall line with the cover windows empty")
            # A per-signal contract can be individually asymmetric on every line
            # and still carry twice the reference's blocking total (one shipped
            # with 4 blocking rendezvous per trip against the reference's 2 —
            # each signal documented, the sum never audited). The count line
            # must carry an actual number.
            count_lines = [ln for ln in scope.splitlines()
                           if "blocking rendezvous count" in ln.lower()]
            if not any(re.search(r"\d", ln) for ln in count_lines):
                v.append("§3 adjudicates [PHASE: ping-pong] but its decision block "
                         "carries no numeric blocking rendezvous count line (blocking "
                         "arrive-and-waits per trip per consumer WG, anchored to "
                         "the reference's count) — a contract can be asymmetric "
                         "on every signal and still double the blocking total")

    for line in sec3.splitlines():
        if _PERF_NUMBER.search(line) and not _EVIDENCE_MARKER.search(line):
            v.append(f"§3 performance number without [MEASURED]/[ESTIMATE]: {line.strip()[:60]}")

    v.extend(_operand_source_violations(sec3))

    v.extend(_gate_hygiene_violations(text, authoring=authoring, baseline=baseline))

    if authoring or baseline is not None:
        # A fresh v1 is held to it whole; a patch only for the clauses it adds.
        # The default resume mode skips it: a milestone already on disk that
        # carries such a clause must not read as "not distilled yet".
        v.extend(downgrade_schedule_violations(text, baseline=baseline))

    if not re.search(r"D\d+\.\d+", text):
        v.append("no decision id (D<x>.<y>) found")
    if not _EVIDENCE_MARKER.search(text):
        v.append("no [MEASURED]/[ESTIMATE] evidence marker found "
                 "(template A.4: every number carries its provenance)")
    return v


def _dependency_cycles(upstream: dict[str, set[str]]) -> list[list[str]]:
    """Cycles in the ⇐ graph (each reported once, from its smallest id)."""
    found, seen_cycles = [], set()

    def walk(start, node, path, visiting):
        for nxt in sorted(upstream.get(node, ())):
            if nxt == start:
                key = frozenset(path)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    found.append(path + [start])
            elif nxt not in visiting and nxt in upstream:
                walk(start, nxt, path + [nxt], visiting | {nxt})

    for did in sorted(upstream):
        walk(did, did, [did], {did})
    return found


# --- operand-source adjudication lint (template A.5 "operand-source verification
# line") ---------------------------------------------------------------------------
#
# Why (an observed failure mode on a blackbox run): the ledger adjudicated one
# operand path as RS, the M0 gate checked only the register *budget* (regs_per_thread), and a
# kernel that reserved the budget while still round-tripping the operand through
# smem passed the gate and was sealed as the definitive architecture. Recovering
# that cost two audit interventions and outlived the run's budget. The rule this
# lint enforces is architecture-neutral — it never says RS is right; it says
# "however the ledger adjudicated, the gate must be able to catch a build that
# quietly did otherwise":
#
#   * adjudication rows carry a machine tag: [SOURCE: RS] / [SOURCE: smem];
#   * any [SOURCE: RS] requires a [VERIFY: sass-derived] gate line (anchored to
#     the sass_census digest fields mma_a_reg / mma_a_desc);
#   * and a spill guard line — RS trades smem traffic for register pressure, and
#     "RS built + spill" is strictly worse than not building it (a measured A/B:
#     schedule transplanted without register economy -> spill 0 -> 10.8M).
#
# RS adjudication prose WITHOUT any [SOURCE:] tag is also a violation: an
# untagged ledger is invisible to this lint, which is exactly how a past gap
# happened ("written down but not enforced by a gate line = not written").
_SOURCE_TAG = re.compile(r"\[SOURCE:\s*(RS|smem)\s*\]", re.I)
_RS_PROSE = re.compile(r"\bRS\s*[AB]?\s*operands?\b|register-?sourced|register-?fed|"
                       r"fed\s+(?:directly\s+)?from\s+registers", re.I)
_SASS_DERIVED_TAG = re.compile(r"\[VERIFY:\s*sass-derived\s*\]")
_SPILL_TERM = re.compile(r"spill", re.I)


def _operand_source_violations(sec3: str) -> list[str]:
    tags = _SOURCE_TAG.findall(sec3)
    if any(t.upper() == "RS" for t in tags):
        out = []
        if not _SASS_DERIVED_TAG.search(sec3):
            out.append("§3 has [SOURCE: RS] adjudication(s) but no [VERIFY: sass-derived] "
                       "gate line (template A.5 operand-source verification line: every "
                       "RS row needs a "
                       "SASS-static check anchored to mma_a_reg/mma_a_desc — a register "
                       "budget alone cannot tell RS-built from smem-round-tripped)")
        if not _SPILL_TERM.search(sec3):
            out.append("§3 has [SOURCE: RS] adjudication(s) but no spill guard line "
                       "(template A.5: RS pays in register pressure; pair the census "
                       "line with a main-kernel spill≈0 line so 'RS built + spilling' "
                       "cannot pass as success)")
        return out
    if not tags and _RS_PROSE.search(sec3):
        return ["§3 adjudicates an operand source as RS in prose but carries no "
                "[SOURCE: RS|smem] tags (template A.5: tag every operand-source "
                "adjudication row so the gate lint can see it — an untagged ledger "
                "is unenforceable)"]
    return []


# Gate-line hygiene, two rules learned on a blackbox run of a sparse-attention op.
#
# (1) Compiler allocation values are observations, never gate lines. The v1
#     pinned nvcc's register split as `USETMAXREG ∋ DEALLOC 0x48 ∧ TRY_ALLOC
#     0xd8` and `regs_per_thread == 168`; the student obeyed, its producer
#     could not hold the index prefetch the same milestone demanded, and the
#     kernel spilled 59.8M sectors at first green. A sibling run's teacher
#     wrote "72/216 is nvcc's allocation result, not an acceptance bar for the
#     student" and gated on regs <= cap + spill == 0; its student chose 88/208 and spilled nothing.
# (2) A SASS shape earns a gate line only with a measured price. sass-derived
#     lines state what the reference LOOKS like; 35 of them (11 in the M0 gate)
#     turned "looks like the ref" into contracts the student had to prove with
#     probes (eight rounds, two hours, no kernel work). The one shape
#     that WAS priced (cp.async BYPASS, -22%) is a legitimate gate line; the
#     split above, never priced, was in fact slower than the alternative. So a
#     `[VERIFY: sass-derived]` line inside a gate block must carry
#     `[PRICED: <delta>, <job N | KB:<entry> | ref-price>]`; without it the
#     line belongs in §3 as a description. The A.5 operand-source line
#     (mma_a_reg / mma_a_desc) is exempt: it is that lint's own obligation.
# A gate block starts at any of the three spellings milestones use for a gate
# — a heading (`### M1 Gate`), a bold line (`**M1 Gate**：`), a plain line
# (`M1 Gate:`) — and runs to the next heading of any level or the next gate
# line. No template pins one spelling, so the lint must not either: a rule
# that only saw headings was silently vacuous on the other two.
_GATE_HEADING = re.compile(
    r"^(?:#{2,5}\s*M\d+[ -]?[Gg]ate\b.*|\*\*M\d+[ -]?[Gg]ate\b[^*\n]*\*\*.*|M\d+[ -]?[Gg]ate\s*[:：].*)$",
    re.M)
_GATE_TOKEN = re.compile(r"M\d+[ -]?[Gg]ate")
_NEXT_HEADING = re.compile(r"^#{2,5}\s", re.M)
_ALLOC_PIN = re.compile(r"USETMAXREG|setmaxnreg|DEALLOC\s+0x|TRY_ALLOC\s+0x|"
                        r"regs_per_thread\s*={1,2}\s*\d", re.I)
_PRICED_TAG = re.compile(r"\[PRICED:\s*[^\]]*\S[^\]]*\]")
_OPERAND_SOURCE_LINE = re.compile(r"mma_a_reg|mma_a_desc")
_GATE_CLASSES = ("ncu-direct", "ncu-derived", "correctness", "sass-derived")
_VERIFY_CLASS = re.compile(r"\[VERIFY:\s*([A-Za-z-]+)\s*\]")


def gate_blocks(text: str) -> list[tuple[str, str]]:
    """(gate title, body) for every gate line (`### M<k> Gate` / `**M<k> Gate**：`
    / `M<k> Gate:`), body running to the next heading of any level or the next
    gate line, whichever comes first."""
    out = []
    heads = list(_GATE_HEADING.finditer(text))
    for k, h in enumerate(heads):
        nxt = _NEXT_HEADING.search(text, h.end())
        end = nxt.start() if nxt else len(text)
        if k + 1 < len(heads):
            end = min(end, heads[k + 1].start())
        tm = _GATE_TOKEN.search(h.group(0))
        title = tm.group(0) if tm else h.group(0).strip("# ").strip()
        out.append((title, text[h.end():end]))
    return out


def _gate_line_shape(ln: str) -> str:
    """A gate line's shape across a threshold re-tune: the PRICED tag stripped,
    every number and comparison operator normalised. Two lines with one shape
    are the same claim at different values — what a patch does when it
    re-tunes a pre-rule gate line, which must stay grandfathered.

    The VERIFY class is PART of the shape: a line that keeps its numbers but
    changes class (`sass-derived` -> `inferred` / `none`) is a different claim
    wearing the old line's words, not a re-tune, and the class rule must see
    it. The PRICED tag is not part of the shape (a re-priced line is the same
    claim at a new price); whether the baseline line WAS priced is tracked
    separately by the caller."""
    s = _VERIFY_CLASS.sub(lambda m: f"[verify:{m.group(1).lower()}]", ln)
    s = _PRICED_TAG.sub("", s)
    s = re.sub(r"[0-9][0-9.,]*", "#", s)
    s = re.sub(r"[≤≥<>=]+", "~", s)
    return re.sub(r"\s+", " ", s.strip().lstrip("-*•").strip()).lower()


def _gate_hygiene_violations(text: str, *, authoring: bool,
                             baseline: str | None = None) -> list[str]:
    """Authoring: every gate line is held. Patch mode (baseline given): only
    gate lines the patch added — matched by their stripped text — are held.
    Resume / bare mode (no baseline): nothing, by design (see validate_milestone)."""
    if not authoring and baseline is None:
        return []
    known: set = set()
    known_shapes: set = set()
    known_pin_shapes: set = set()
    known_priced_shapes: set = set()
    if baseline is not None:
        for _, body in gate_blocks(baseline):
            for ln in body.splitlines():
                if "[VERIFY:" in ln:
                    known.add(ln.strip())
                    shape = _gate_line_shape(ln)
                    known_shapes.add(shape)
                    if _ALLOC_PIN.search(ln):
                        known_pin_shapes.add(shape)
                    if _PRICED_TAG.search(ln) is not None:
                        known_priced_shapes.add(shape)
    out = []
    blocks = gate_blocks(text)
    if authoring and not blocks:
        out.append("no gate block found: write each phase's acceptance as a "
                   "`### M<k> Gate` block (a `**M<k> Gate**：` or `M<k> Gate:` line also "
                   "counts) — the gate-line hygiene rules can only hold lines they can see")
    for title, body in blocks:
        for ln in body.splitlines():
            if "[VERIFY:" not in ln:
                continue
            if ln.strip() in known:
                continue
            # a pre-rule line re-tuned (numbers / operator only, same VERIFY
            # class) makes no new claim: the class and price rules hold new
            # claims, not old ones re-thresholded. The pin rule still applies
            # to it unless the line it re-tunes already pinned — turning a
            # bound (`<= cap`) into an allocation pin (`== N`) IS a new claim.
            # The price rule is only waived for a re-tune of a line the
            # baseline itself carried UNPRICED: dropping a [PRICED] tag while
            # moving the threshold is a priced claim re-issued without its
            # price, not a grandfathered one.
            shape = _gate_line_shape(ln)
            retuned = shape in known_shapes
            cm = _VERIFY_CLASS.search(ln)
            if not retuned and cm and cm.group(1).lower() not in _GATE_CLASSES:
                out.append(f"{title}: `{ln.strip()[:90]}` has [VERIFY: {cm.group(1)}] — only "
                           f"{'/'.join(_GATE_CLASSES)} may be gate lines (template A.3.1); "
                           f"`inferred` / `none` rows stay in the decision body as "
                           f"[ESTIMATE] candidates")
            if _ALLOC_PIN.search(ln) and not (retuned and shape in known_pin_shapes):
                out.append(f"{title}: `{ln.strip()[:90]}` pins a compiler allocation value "
                           f"(template A.3.1: USETMAXREG / setmaxnreg immediates and an "
                           f"exact regs_per_thread are observations of the reference, not "
                           f"acceptance lines — gate on `regs_per_thread <= <cap>` and "
                           f"`spill_ld_sectors == 0`, and leave the split to the student)")
            if ((not retuned or shape in known_priced_shapes)
                    and _SASS_DERIVED_TAG.search(ln)
                    and _PRICED_TAG.search(ln) is None
                    and _OPERAND_SOURCE_LINE.search(ln) is None):
                out.append(f"{title}: `{ln.strip()[:90]}` is a [VERIFY: sass-derived] gate "
                           f"line with no [PRICED: <delta>, <job N | KB:<entry> | "
                           f"ref-price>] tag (template A.3.1: a SASS shape says what the "
                           f"reference LOOKS like; it becomes a gate line only once a "
                           f"probe measured what the shape is WORTH — otherwise move it "
                           f"to the decision body as a description"
                           + ("; the line it re-tunes carried a price, so the re-tune "
                              "must too" if retuned else "") + ")")
    return out


# machine-readable target the driver reads as the stop threshold (SSOT). The meta
# agent writes/re-baselines this in §0; the CLI --target-ms only seeds it.
# --- pre-scheduled downgrades -------------------------------------------------
#
# POLICY §3 ③ makes the conservative form (scalar solve, cooperative copies,
# compiler-default registers, a naive exchange face) a FAILURE RESPONSE: the
# same package failed whole-machine twice, so the student downgrades to get a
# first green and swaps the real form back afterwards. A milestone that writes
# the downgrade INTO the construction order or a decision's trigger clause
# ("first use an f32 scalar forward substitution … correctness green first … after
# first green, switch to …") schedules the
# placeholder with zero failures behind it. That is the shape that lets a
# placeholder cross the gate untouched: scaffold rows carry no timing, the M0
# gate sets no performance line, and "after first green" names no iteration
# — the first green IS the gate row when every sub-step was scaffolded. The
# clause is refused unless it reads as a conditional fallback (a failure
# trigger in the same clause) or a prohibition of one.
_DOWNGRADE_WORD = re.compile(
    r"\b(?:placeholder|interim|stand-in|transitional|temporar(?:y|ily)\s+(?:use|form|solve)|"
    r"first\s+use|start\s+with|conservative\s+form|downgrade[sd]?|"
    r"fall(?:s|ing)?\s+back\s+to|stopgap)", re.I)
_SWAP_BACK_WORD = re.compile(
    r"first[ -]?green|(?:after|once)\s+green|swap(?:ped)?\s+(?:back|in)\b|"
    r"switch(?:ed)?\s+(?:back|to)\b|then\s+swap|replace(?:d)?\s+(?:it\s+)?with", re.I)
_FAILURE_TRIGGER = re.compile(
    r"REVERT|iter\s*\d+|twice|two\s+(?:consecutive\s+|whole-machine\s+)*(?:failures|FAILs?|fails|crashes)|"
    r"(?:FAIL|fail)s?\s+(?:twice|two\s+times)|consecutive\s+FAILs?|after\s+\S+\s+fail|"
    r"after\s+(?:a\s+|the\s+)?fail(?:ure|ed)", re.I)
_DOWNGRADE_NEGATION = re.compile(
    r"must\s+not|may\s+not|do\s+not|don't|never|not\s+allowed|forbidden|prohibited|"
    r"banned|no\s+longer", re.I)
# How far before a downgrade word a prohibition may sit and still govern it.
# Sized for English clauses: "Do not schedule a placeholder" already puts the
# negation 18 characters back.
_NEGATION_LOOKBACK = 40


def _downgrade_is_negated(blob: str) -> bool:
    """Every downgrade word in ``blob`` sits right after a prohibition
    ("it is not allowed to 'first use plain STS/LDS to get it running, then swap'")
    — a ban on placeholders is not one."""
    hits = list(_DOWNGRADE_WORD.finditer(blob))
    return bool(hits) and all(
        _DOWNGRADE_NEGATION.search(blob[max(0, m.start() - _NEGATION_LOOKBACK):m.start()])
        for m in hits)


def downgrade_schedule_violations(text: str, *, baseline: str | None = None) -> list[str]:
    """Clauses that schedule a conservative / placeholder form ahead of any failure.

    A clause is a line carrying both a downgrade word and a swap-back word,
    read together with up to two preceding lines of its paragraph (the rule's
    own wording spreads the failure trigger and the downgrade over a line
    break). It conforms when that context names a failure trigger (fails
    twice / two consecutive FAILs / REVERT …) or prohibits the downgrade. With
    ``baseline`` (patch mode) only clauses absent from the baseline text are
    reported.
    """
    base_lines = set(ln.strip() for ln in baseline.splitlines()) if baseline is not None else None
    lines = text.splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        if not (_DOWNGRADE_WORD.search(ln) and _SWAP_BACK_WORD.search(ln)):
            continue
        if base_lines is not None and ln.strip() in base_lines:
            continue
        ctx = [ln]
        j = i - 1
        while j >= 0 and len(ctx) < 3 and lines[j].strip():
            ctx.insert(0, lines[j])
            j -= 1
        blob = " ".join(ctx)
        if _FAILURE_TRIGGER.search(blob) or _downgrade_is_negated(blob):
            continue
        out.append(
            f"Pre-scheduled downgrade / placeholder form in the construction order "
            f"(line {i + 1}): \"{ln.strip()[:100]}\" — POLICY §3 ③: the conservative "
            f"form may only be a failure response after the same bundle failed "
            f"whole-machine twice; it must not be written as a construction step or a "
            f"trigger condition (once a placeholder is in, the build phase has no "
            f"performance reading and the gate sets no performance line, so it crosses "
            f"the gate untouched). Delete this clause, or rewrite it as a fallback with "
            f"a failure trigger: \"If … FAILs twice in a row, you may temporarily use … "
            f"to get first green, then swap back with [RESTORED: D<x>.<y>]\"")
    return out


_TARGET_RE = re.compile(r"target_ms\s*[:：]\s*([0-9]*\.?[0-9]+)")


def parse_target_ms(text: str) -> float | None:
    """The machine-readable ``target_ms: <value>`` from the milestone, or None.

    This is the target SSOT: putting it in the milestone (not just the CLI flag)
    is what makes the meta agent's §0 target edit actually move the stop threshold.
    """
    m = _TARGET_RE.search(text)
    return float(m.group(1)) if m else None


# Which timing window §0's target_ms is anchored in. A bare number cannot say, and
# the two windows differ by the ref's host tax (~12% on attn_fwd) — a device-gated
# loop comparing device readings against an e2e-anchored target would declare
# success that far early. The driver seeds this line next to target_ms and refuses
# to use a §0 target whose annotation does not match the run's gate window.
_TARGET_WINDOW_RE = re.compile(r"target_window\s*[:：]\s*(e2e|device)")


def parse_target_window(text: str) -> str | None:
    """The machine-readable ``target_window: e2e|device`` from §0, or None.

    None on a pre-annotation milestone: every milestone written before the
    gate_window knob existed is e2e-anchored by construction, so the reader
    (driver._resolve_target) treats absence as e2e rather than refusing history.
    """
    m = _TARGET_WINDOW_RE.search(text)
    return m.group(1) if m else None


# A decision's machine-readable verification declaration: "[VERIFY: none]" or
# "[VERIFY: ncu-direct] grid_size = 5120". The tag names what evidence can catch the
# decision — "none" means nothing per-step-measurable stands behind it (it stays in the
# milestone as guidance and out of the gate), the ncu-* kinds are launch/traffic facts a
# single profile at the gate row harvests in one shot.
_VERIFY_TAG = re.compile(r"\[VERIFY:\s*([A-Za-z-]+)\s*\]")


def parse_verify_tags(text: str) -> dict[str, str]:
    """Decision id -> its ``[VERIFY:]`` tag kind ("none", "ncu-direct", ...).

    This is the reader for the declaration the schema asks each M0 decision to carry:
    the driver uses it to say WHY a scaffold-phase commit gets no per-step eval
    (nothing a full eval of an under-construction kernel could check the declaration
    against). Decisions without a tag are simply absent from the dict.
    """
    out: dict[str, str] = {}
    for did, block in _decision_blocks(_section_body(text, DECISIONS_SECTION, DECISIONS_ENDS)):
        m = _VERIFY_TAG.search(block)
        if m:
            out[did] = m.group(1)
    return out


def parse_milestone_dag(text: str) -> list[Decision]:
    """Build the decision DAG from §3 by reading each decision's `Depends on` field.

    Each ``D<x>.<y>`` decision in §3 becomes a Decision whose depends_on is the set
    of decision ids (``D<x>.<y>``) named in its `Depends on` line. This is the SSOT
    for prefix-reuse invalidation: a milestone patch's changed_entries are matched
    against these dependencies to decide which trajectory checkpoints go stale.
    """
    out: list[Decision] = []
    for did, block in _decision_blocks(_section_body(text, DECISIONS_SECTION, DECISIONS_ENDS)):
        dep_line = _DEP_LINE.search(block)
        up, comp, _down = (parse_dependency_line(dep_line.group(1)) if dep_line
                           else (set(), set(), set()))
        up.discard(did)  # a decision never depends on itself
        comp.discard(did)
        out.append(Decision(id=did, depends_on=frozenset(up), companions=frozenset(comp)))
    return out
