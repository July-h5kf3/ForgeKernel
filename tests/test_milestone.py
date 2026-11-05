"""Tests for milestone schema validation (formatted, not free-form)."""

from forgekernel.milestone import parse_milestone_dag, validate_milestone

VALID = """\
# demo_op milestone

## 0. Problem statement and target
target_ms: 0.7
0.7 ms / 45% MFU [MEASURED, job 42]; reproduction threshold ≥95%.

## 1. Problem contract (Math Spec)
C = A @ B, fp16.

## 2. Decision DAG + phase order
```
M0 → M0.1 → M1 → M2
```

## 3. Decision details
D1.1 tile choice
- Action: choose BM=128, BN=256
- Purpose: keep the tensor cores fed
- Trigger condition: foundation, first step
- Cost: register budget
- Depends on: —
- Handoff protocol: [SYNC: none] (single-role demo, no cross-role supply)
- Feed form: [GATHER: none] (pure affine loads)
- Consumer phase: [PHASE: none] (single consumer WG demo)

M1 Gate:
- End-to-end performance line: ≤1.2 ms [ESTIMATE]
- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%

## 4. Known dead ends
| Decision | Symptom | Trigger iteration | Retry precondition |
|------|------|-----------|----------|

## 5. Global correctness / performance gate
atol=5, rtol=0.05; MFU ≥95% target.

## 6. Rules
One variable per round; anti-plagiarism clause; INFEASIBLE clause.

## Worked Example
BM=128, BN=256, BK=64.
"""


def test_valid_milestone_conforms():
    assert validate_milestone(VALID) == []


def test_missing_section_flagged():
    broken = VALID.replace("## 5. Global correctness / performance gate", "## 5b. something else")
    v = validate_milestone(broken)
    assert any("## 5." in x for x in v)


def test_missing_worked_example_flagged():
    broken = VALID.replace("## Worked Example", "## Appendix")
    assert any("Worked Example" in x for x in validate_milestone(broken))


def test_decisions_without_five_tuple_flagged():
    # §3 present but free-form (no five-tuple keys)
    broken = VALID.replace(
        "D1.1 tile choice\n- Action: choose BM=128, BN=256\n- Purpose: keep the tensor cores fed\n"
        "- Trigger condition: foundation, first step\n- Cost: register budget\n- Depends on: —",
        "D1.1: just pick some tile, use your judgement.",
    )
    v = validate_milestone(broken)
    assert any("five-tuple" in x for x in v)


def test_gate_without_mechanism_line_flagged():
    # template A.5: an end-to-end perf line alone cannot separate "the mechanism
    # engaged" from "the number happened to land", so every gate needs a mechanism line
    broken = VALID.replace("- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n", "")
    assert any("mechanism-verification line" in x for x in validate_milestone(broken))


def test_number_without_evidence_marker_flagged():
    # template A.4: no unsourced performance bands
    broken = VALID.replace("[MEASURED, job 42]", "").replace("[ESTIMATE]", "")
    assert any("evidence marker" in x for x in validate_milestone(broken))


def test_no_decision_id_flagged():
    broken = VALID.replace("D1.1", "the first decision")
    assert any("decision id" in x for x in validate_milestone(broken))


_DAG_MS = """\
# demo milestone
## 3. Decision details
**D1.0 base**
- Action: a
- Depends on: —

**D2.3 next**
- Action: b
- Depends on: D1.0

**D5.2 late**
- Action: c
- Depends on: D2.3
## 4. Known dead ends
"""


def _with_decisions(body: str) -> str:
    """VALID's frame with §3 replaced, so only the decision shapes vary."""
    head, _, rest = VALID.partition("## 3. Decision details\n")
    _, _, tail = rest.partition("## 4. Known dead ends")
    return f"{head}## 3. Decision details\n{body}\n## 4. Known dead ends{tail}"


def test_decision_headers_are_found_under_any_markdown_shape():
    """Nothing in the template or the prompts pins one shape, and the only real
    distillation artifact in the repo wrote all 21 of its decisions as '### D0.1'.
    Recognising just '**D1.1**' made that whole file parse as zero decisions."""
    for header in ("### D1.1 tile", "**D1.1 tile**", "- **D1.1** tile",
                   "| D1.1 | tile |", "D1.1 tile", "#### D1.1 tile"):
        ms = _with_decisions(
            f"{header}\n- Action: x\n- Purpose: x\n- Trigger condition: x\n- Cost: x\n- Depends on: —\n"
            "- Handoff protocol: [SYNC: none]\n- Feed form: [GATHER: none]\n"
            "- Consumer phase: [PHASE: none]\n"
            "- Mechanism-verification line: tensor pipe active %\n")
        assert [d.id for d in parse_milestone_dag(ms)] == ["D1.1"], header
        assert validate_milestone(ms) == [], header


def test_a_milestone_with_no_parseable_decisions_is_a_violation():
    """Zero decisions used to mean zero complaints: every per-decision check
    iterates over the parsed blocks, so an empty list satisfied all of them."""
    ms = _with_decisions("This section is all prose, with no decision id at all.\n")
    v = validate_milestone(ms)
    assert any("no parseable decisions" in x for x in v), v


def test_nested_numbered_heading_does_not_truncate_the_decisions_section():
    """'### 4. M2 pipeline' contains the substring '## 4.'. Cutting §3 there dropped
    every later decision from the DAG *and* from the gate checks, both silently."""
    ms = _with_decisions(
        "### 3. M1 tiling\n**D1.1 tile**\n- Action: x\n- Purpose: x\n- Trigger condition: x\n"
        "- Cost: x\n- Depends on: —\n- Mechanism-verification line: tensor pipe active %\n\n"
        "### 4. M2 pipeline\n**D2.1 staging**\n- Action: x\n- Purpose: x\n- Trigger condition: x\n"
        "- Cost: x\n- Depends on: D1.1\n")
    assert [d.id for d in parse_milestone_dag(ms)] == ["D1.1", "D2.1"]
    # D2 really is missing its mechanism line — the point is that it gets checked
    assert any("D2" in x and "mechanism-verification line" in x for x in validate_milestone(ms))


def test_m0_is_no_longer_exempt_from_the_mechanism_line():
    """M0 used to be skipped here: back when it delivered a naive implementation
    there was nothing structural to verify. It now freezes the architecture that
    every later milestone is fitted inside, so an M0 gate with nothing mechanical
    in it is the hole that let a scalar kernel pass as a foundation."""
    ms = _with_decisions(
        "**D0.1 three-kernel split**\n- Action: x\n- Purpose: x\n- Trigger condition: x\n"
        "- Cost: x\n- Depends on: —\n")
    assert any("D0" in x and "mechanism-verification line" in x for x in validate_milestone(ms))
    # ...and it passes once the gate carries one
    ok = _with_decisions(
        "**D0.1 three-kernel split**\n- Action: x\n- Purpose: x\n- Trigger condition: x\n"
        "- Cost: x\n- Depends on: —\n- Handoff protocol: [SYNC: none]\n"
        "- Feed form: [GATHER: none]\n- Consumer phase: [PHASE: none]\n"
        "- Mechanism-verification line: tensor pipe active % vs ref ≥50%\n")
    assert validate_milestone(ok) == []


def test_parse_milestone_dag_extracts_dependencies():
    dag = parse_milestone_dag(_DAG_MS)
    assert [d.id for d in dag] == ["D1.0", "D2.3", "D5.2"]
    assert dag[0].depends_on == frozenset()
    assert dag[1].depends_on == frozenset({"D1.0"})
    assert dag[2].depends_on == frozenset({"D2.3"})


def test_parse_milestone_dag_feeds_compute_resume():
    # changing D1.0 (the foundation decision) invalidates everything downstream of it
    from forgekernel.resume import compute_resume
    dag = parse_milestone_dag(_DAG_MS)
    plan = compute_resume(dag, ["D1.0"])
    assert plan.resume_after is None  # D1.0 is the first decision -> full restart
    assert plan.invalidated == ["D1.0", "D2.3", "D5.2"]


def test_parse_milestone_dag_empty_section():
    assert parse_milestone_dag("# no section here") == []


def test_parse_target_ms_reads_machine_readable_marker():
    from forgekernel.milestone import parse_target_ms
    ms = "# op milestone\n## 0. target and equivalence judge\ntarget_ms: 0.1852\nequivalence judge: ...\n## 1. x\n"
    assert parse_target_ms(ms) == 0.1852


def test_parse_target_ms_none_without_marker():
    from forgekernel.milestone import parse_target_ms
    assert parse_target_ms("# op\n## 0. target\nmeasured performance of the original op.\n") is None


def test_parse_target_ms_fullwidth_colon():
    from forgekernel.milestone import parse_target_ms
    assert parse_target_ms("target_ms：0.40") == 0.40


def test_parse_target_window_reads_the_annotation():
    from forgekernel.milestone import parse_target_window
    ms = "## 0. target\ntarget_ms: 0.26\ntarget_window: device\n"
    assert parse_target_window(ms) == "device"
    assert parse_target_window("target_ms: 0.26\ntarget_window: e2e\n") == "e2e"


def test_parse_target_window_none_on_pre_annotation_milestones():
    # every milestone written before the gate_window knob is e2e-anchored by
    # construction; the reader treats absence as e2e instead of refusing history
    from forgekernel.milestone import parse_target_window
    assert parse_target_window("target_ms: 0.26\n") is None
    assert parse_target_window("target_window: wallclock\n") is None



def test_parse_verify_tags_reads_the_per_decision_declaration():
    from forgekernel.milestone import parse_verify_tags
    text = """# demo milestone

## 3. Decision details

### D0.1 three-kernel split
- Action: a
- Depends on: —
- [VERIFY: ncu-direct] `kernel_count = 3`

### D0.13 barrier pairing
- Action: b
- Depends on: D0.1
- [VERIFY: none] — whether it took effect is backstopped by the mechanism line.

### D0.14 scheduling
- Action: c
- Depends on: D0.1

## 4. Known dead ends
x
"""
    tags = parse_verify_tags(text)
    assert tags["D0.1"] == "ncu-direct"
    assert tags["D0.13"] == "none"
    assert "D0.14" not in tags  # undeclared stays absent, not defaulted


# --- operand-source adjudication lint (template A.5 operand-source verification line) --

RS_LEDGER_ROW = """\
D1.2 operand-source adjudication
- Action: chain4's A operand is register-fed [SOURCE: RS]; the rest [SOURCE: smem]
- Purpose: cut the smem round trip through the exchange face
- Trigger condition: M0 architecture
- Cost: consumer registers squeezed to 240
- Depends on: D1.1
"""

RS_GATE_LINES = """\
- Mechanism-verification line: chain4 RS built, mma_a_reg reaches the chain's beat count [VERIFY: sass-derived]
- Main-kernel spill line: spill_ld_sectors + spill_st_sectors == 0
"""


def _with_rs(ledger: str, gate_extra: str = "") -> str:
    return VALID.replace("M1 Gate:", ledger + "\nM1 Gate:\n" + gate_extra)


def test_rs_source_tag_without_sass_line_flagged():
    v = validate_milestone(_with_rs(RS_LEDGER_ROW))
    assert any("[VERIFY: sass-derived]" in x for x in v)
    assert any("spill" in x for x in v)


def test_rs_source_tag_with_sass_and_spill_lines_conforms():
    assert validate_milestone(_with_rs(RS_LEDGER_ROW, RS_GATE_LINES)) == []


def test_rs_prose_without_source_tags_flagged():
    untagged = RS_LEDGER_ROW.replace(" [SOURCE: RS]", "").replace(" [SOURCE: smem]", "")
    v = validate_milestone(_with_rs(untagged, RS_GATE_LINES))
    assert any("[SOURCE: RS|smem]" in x for x in v)


def test_smem_only_adjudication_needs_no_sass_line():
    smem_only = RS_LEDGER_ROW.replace("is register-fed [SOURCE: RS]", "goes through the smem transpose face [SOURCE: smem]")
    assert validate_milestone(_with_rs(smem_only)) == []


def test_missing_sync_adjudication_is_flagged():
    broken = VALID.replace(
        "- Handoff protocol: [SYNC: none] (single-role demo, no cross-role supply)\n", "")
    assert any("[SYNC:]" in x for x in validate_milestone(broken, authoring=True))


def test_nonnone_sync_without_stall_barrier_line_is_flagged():
    # adjudicating a real cross-role handoff obliges a stall_barrier gate line:
    # an unverified protocol can serialize the supply path and still pass on
    # end-to-end numbers (the triple symptom ships silently).
    broken = VALID.replace("[SYNC: none] (single-role demo, no cross-role supply)",
                           "[SYNC: named-barrier] (low-frequency handoff)")
    assert any("stall_barrier" in x for x in validate_milestone(broken))


def test_nonnone_sync_with_stall_barrier_line_conforms():
    ok = VALID.replace("[SYNC: none] (single-role demo, no cross-role supply)",
                       "[SYNC: mbarrier-pipeline] (streaming KV supply)")
    ok = ok.replace(
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%",
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n"
        "- Sync mechanism line: [VERIFY: ncu-direct] stall_barrier within the reference band [MEASURED, job 42]")
    assert validate_milestone(ok) == []


def test_missing_gather_adjudication_is_flagged():
    broken = VALID.replace("- Feed form: [GATHER: none] (pure affine loads)\n", "")
    assert any("[GATHER:]" in x for x in validate_milestone(broken, authoring=True))


def test_nonnone_gather_without_long_sb_line_is_flagged():
    # a row-serial index chain exposes full memory latency per row while every
    # other meter stays in-band — only a stall_long_sb anchor catches it.
    broken = VALID.replace("[GATHER: none] (pure affine loads)",
                           "[GATHER: serial] (the index stream can be hidden — to be proven)")
    assert any("stall_long_sb" in x for x in validate_milestone(broken))


def test_nonnone_gather_with_long_sb_line_conforms():
    ok = VALID.replace("[GATHER: none] (pure affine loads)",
                       "[GATHER: prefetched] (indices prefetched in batches; issue order: index LDGs "
                       "issued in a batch before the empty-wait, sass-derived check of the cluster)")
    ok = ok.replace(
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%",
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n"
        "- Feed mechanism line: [VERIFY: ncu-direct] stall_long_sb within the reference band [MEASURED, job 42]")
    assert validate_milestone(ok) == []


def test_prefetched_gather_without_ordering_transcription_is_flagged():
    # one contract shipped "prefetched" while every index load stayed inline
    # after the empty-wait — the label constrained vocabulary, not structure.
    # The ordering transcription is the structural half of the verdict.
    broken = VALID.replace("[GATHER: none] (pure affine loads)",
                           "[GATHER: prefetched] (indices prefetched in batches)")
    broken = broken.replace(
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%",
        "- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n"
        "- Feed mechanism line: [VERIFY: ncu-direct] stall_long_sb within the reference band [MEASURED, job 42]")
    assert any("issue order" in x for x in validate_milestone(broken))
    # serial adjudication carries no ordering obligation
    ok_serial = broken.replace("[GATHER: prefetched] (indices prefetched in batches)",
                               "[GATHER: serial] (latency hidden by the ring depth)")
    assert not any("issue order" in x for x in validate_milestone(ok_serial))


def test_missing_phase_adjudication_is_flagged():
    broken = VALID.replace("- Consumer phase: [PHASE: none] (single consumer WG demo)\n", "")
    assert any("[PHASE:]" in x for x in validate_milestone(broken, authoring=True))


def test_pingpong_phase_without_contract_transcription_is_flagged():
    # two same-role consumer WGs marching in phase idle the tensor pipe with
    # every stall meter in band — the enforceable half is the transcription.
    broken = VALID.replace("[PHASE: none] (single consumer WG demo)",
                           "[PHASE: ping-pong] (two consumers, staggered, covering each other)")
    assert any("phase contract" in x for x in validate_milestone(broken))


def test_pingpong_phase_with_contract_transcription_conforms():
    ok = VALID.replace(
        "[PHASE: none] (single consumer WG demo)",
        "[PHASE: ping-pong] (two consumers, staggered, covering each other); phase contract: "
        "0x8 WG0→WG1 arrive-only/blocking=WG1, waits for the previous trip's P; 0x9 WG1→WG0 "
        "symmetric mirror; prologue: WG0 takes QK(0) first; "
        "blocking rendezvous count: 2/trip/WG, anchored to the reference's 2 [MEASURED, job 42]")
    assert validate_milestone(ok) == []


def test_pingpong_phase_without_blocking_count_ledger_is_flagged():
    # a per-signal contract can be asymmetric on every line and still carry
    # twice the reference's blocking total — each signal documented, the sum
    # never audited. The count line is the ledger that catches it.
    broken = VALID.replace(
        "[PHASE: none] (single consumer WG demo)",
        "[PHASE: ping-pong] (two consumers, staggered, covering each other); phase contract: "
        "0x8 WG0→WG1 arrive-only/blocking=WG1, waits for the previous trip's P; 0x9 WG1→WG0 "
        "symmetric mirror; prologue: WG0 takes QK(0) first")
    assert any("blocking rendezvous count" in x for x in validate_milestone(broken))


def test_convoy_phase_needs_no_contract_transcription():
    ok = VALID.replace("[PHASE: none] (single consumer WG demo)",
                       "[PHASE: convoy] (scalar segments <2% of the time, no cover needed)")
    assert validate_milestone(ok) == []


# --- review-fix regressions (orchestration adjudication hardening) -------------


def test_legacy_milestone_without_tags_is_grandfathered_outside_authoring():
    # A schema addition must not turn every pre-existing milestone into "not
    # distilled yet": both resume gates translate violations into re-authoring,
    # which overwrites accumulated meta patches and re-measures baselines the
    # resume contract promises to reuse.
    legacy = VALID.replace(
        "- Handoff protocol: [SYNC: none] (single-role demo, no cross-role supply)\n", "").replace(
        "- Feed form: [GATHER: none] (pure affine loads)\n", "").replace(
        "- Consumer phase: [PHASE: none] (single consumer WG demo)\n", "")
    assert validate_milestone(legacy) == []          # resume/patch mode: clean
    strict = validate_milestone(legacy, authoring=True)
    assert any("[SYNC:]" in x for x in strict)
    assert any("[GATHER:]" in x for x in strict)
    assert any("[PHASE:]" in x for x in strict)


def test_tag_values_compare_case_insensitively():
    # [GATHER: Prefetched] must trigger the ordering obligation, not silently
    # skip it; [SYNC: NONE] must not be misread as a non-none adjudication.
    broken = VALID.replace("[GATHER: none] (pure affine loads)",
                           "[GATHER: Prefetched] (indices prefetched in batches)")
    assert any("issue order" in x for x in validate_milestone(broken))
    ok = VALID.replace("[SYNC: none] (single-role demo, no cross-role supply)",
                       "[SYNC: NONE] (single-role demo)")
    assert not any("stall_barrier" in x for x in validate_milestone(ok))


def test_prose_mention_of_stall_metric_is_not_a_gate_line():
    # "may push up stall_barrier" satisfied the old bare-substring check; the
    # obligation is a [VERIFY:] line, and prose must not stand in for it.
    broken = VALID.replace("[SYNC: none] (single-role demo, no cross-role supply)",
                           "[SYNC: mbarrier-pipeline] (streaming supply; if this protocol is wrong it "
                           "may push up stall_barrier)")
    assert any("stall_barrier" in x for x in validate_milestone(broken))


def test_out_of_vocabulary_tag_value_is_named_not_misreported():
    # An underscore typo used to fall outside the value regex entirely and get
    # reported as "no adjudication at all" — fail-closed but misdiagnosed.
    broken = VALID.replace("[SYNC: none] (single-role demo, no cross-role supply)",
                           "[SYNC: mbarrier_pipeline] (typo)")
    v = validate_milestone(broken)
    assert any("vocabulary" in x and "mbarrier_pipeline" in x for x in v)
    assert not any("has no [SYNC:]" in x for x in v)
    loose = VALID.replace("[GATHER: none] (pure affine loads)",
                          "[GATHER: anything-here] (out of vocabulary)")
    assert any("vocabulary" in x for x in validate_milestone(loose))


def test_pingpong_missing_both_obligations_reports_both_in_one_pass():
    # elif-chained checks report one violation per rewrite round and burn the
    # finite patch budget teaching the second duty after the first is fixed.
    broken = VALID.replace("[PHASE: none] (single consumer WG demo)",
                           "[PHASE: ping-pong] (two consumers, staggered, covering each other)")
    v = validate_milestone(broken)
    assert any("phase contract" in x for x in v)
    assert any("blocking rendezvous count" in x for x in v)


def test_pingpong_blocking_count_must_be_numeric():
    broken = VALID.replace(
        "[PHASE: none] (single consumer WG demo)",
        "[PHASE: ping-pong] (staggered cover); phase contract: 0x8 WG0→WG1 arrive-only/"
        "blocking=WG1, waits for the previous trip's P\n- Blocking rendezvous count: aligned with "
        "the reference (exact number still to be filled in)")
    assert any("blocking rendezvous count" in x for x in validate_milestone(broken))


def test_tag_obligations_are_scoped_to_their_decision_block():
    # A mixed milestone must not satisfy one decision's transcription duty with
    # prose that belongs to another decision.
    two_blocks = VALID.replace(
        "M1 Gate:",
        "D1.2 gather feed\n"
        "- Action: fetch KV indirectly\n"
        "- Purpose: feed PV\n"
        "- Trigger condition: foundation\n"
        "- Cost: smem ring\n"
        "- Depends on: —\n"
        "- Feed form: [GATHER: prefetched] (indices prefetched in batches)\n\n"
        "M1 Gate:\n"
        "- Feed mechanism line: [VERIFY: ncu-direct] stall_long_sb within the reference band "
        "[MEASURED, job 42]",
    ).replace("- Feed form: [GATHER: none] (pure affine loads)",
              "- Feed form: [GATHER: none] (pure affine loads); issue order: not applicable to this decision")
    # an issue order exists in D1.1's block, but D1.2's prefetched tag has none in ITS
    # block — the obligation must fire.
    assert any("issue order" in x for x in validate_milestone(two_blocks))



# --- dependency arrows: only ⇐ builds the DAG ----------------------------------

_ARROW_MS = """# op milestone

## 0. Problem statement and target
x

## 1. Problem contract (Math Spec)
x

## 2. Decision DAG + phase order
x

## 3. Decision details
### D0.1 root
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: ⇒ D0.2, D0.3

#### D0.2 producer
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on：⇐ D0.1 | ⊗ D0.3 | ⇒ D0.4

**D0.3 consumer**
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
**Depends on** —— ⇐ D0.1. ⇒ D0.4 (`grad` resident in smem).

M0 Gate:
- Mechanism-verification line: tensor pipe active % ≥60%

### D0.4 epilogue
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: D0.2, D0.3

M1 Gate:
- Mechanism-verification line: x

## 4. Known dead ends
x

## 5. Global correctness / performance gate
x

## 6. Rules
x

## Worked Example
x
"""


def test_parse_dependency_line_attributes_ids_to_the_nearest_arrow():
    from forgekernel.milestone import parse_dependency_line
    assert parse_dependency_line("⇐ D0.1 | ⊗ D0.15 | ⇒ D0.11, D0.18") == (
        {"D0.1"}, {"D0.15"}, {"D0.11", "D0.18"})
    assert parse_dependency_line("⇐ D0.1. ⇒ D0.7 (resident in smem)") == ({"D0.1"}, set(), {"D0.7"})
    assert parse_dependency_line("D1.1, D1.2") == ({"D1.1", "D1.2"}, set(), set())
    assert parse_dependency_line("—") == (set(), set(), set())


def test_parse_milestone_dag_only_follows_upstream_arrows():
    dag = {d.id: d for d in parse_milestone_dag(_ARROW_MS)}
    assert dag["D0.1"].depends_on == frozenset()           # ⇒ is not a reverse edge
    assert dag["D0.2"].depends_on == frozenset({"D0.1"})   # ⊗ D0.3 is not an edge
    assert dag["D0.2"].companions == frozenset({"D0.3"})
    assert dag["D0.3"].depends_on == frozenset({"D0.1"})   # the bold "**Depends on** ——" form parses
    assert dag["D0.4"].depends_on == frozenset({"D0.2", "D0.3"})  # legacy arrow-less = ⇐


def test_parse_milestone_dag_no_longer_closes_over_the_whole_file():
    # regression (grouped-gemm): every id on the line was an edge, so a patch of D0.2
    # reached D0.1 through its own ⇒ and the "DAG" invalidated everything
    from forgekernel.resume import compute_resume
    plan = compute_resume(parse_milestone_dag(_ARROW_MS), ["D0.3"])
    assert plan.resume_after == "D0.2" and plan.invalidated == ["D0.3", "D0.4"]


def test_validator_requires_an_upstream_edge_on_non_root_decisions_when_authoring():
    from forgekernel.milestone import dependency_warnings, validate_milestone
    assert not [v for v in validate_milestone(_ARROW_MS, authoring=True) if "upstream" in v]
    broken = _ARROW_MS.replace("- Depends on: D0.2, D0.3", "- Depends on: ⇒ D0.5")
    v = validate_milestone(broken, authoring=True)
    assert any("D0.4 names no upstream" in x for x in v)
    assert any("undefined decision D0.5" in x for x in v)
    # resume/patch mode: the hole is advisory. Three of twelve in-flight
    # milestones carried one; as a violation it read as "not distilled"
    # and re-authored v1 over their patches on the next --resume.
    assert not [x for x in validate_milestone(broken) if "upstream" in x]
    assert any("D0.4 names no upstream" in x for x in dependency_warnings(broken))


def test_validator_rejects_a_dependency_cycle_when_authoring():
    from forgekernel.milestone import dependency_warnings, validate_milestone
    cyclic = _ARROW_MS.replace("- Depends on: ⇒ D0.2, D0.3", "- Depends on: ⇐ D0.4")
    assert any("dependency cycle" in x for x in validate_milestone(cyclic, authoring=True))
    assert not any("dependency cycle" in x for x in validate_milestone(cyclic))
    assert any("dependency cycle" in x for x in dependency_warnings(cyclic))


def test_dependency_warnings_flag_one_sided_downstream_claims():
    from forgekernel.milestone import dependency_warnings
    # D0.1 ⇒ D0.2 and D0.2 ⇐ D0.1 agree; D0.2 ⇒ D0.4 is echoed by the legacy line;
    # add a ⇒ nobody echoes
    ms = _ARROW_MS.replace("- Depends on: ⇒ D0.2, D0.3", "- Depends on: ⇒ D0.2, D0.3, D0.4")
    w = dependency_warnings(ms)
    assert w == ["§3 D0.1 declares ⇒ D0.4 but D0.4 does not list ⇐ D0.1"]


def _with_gate(lines: str) -> str:
    """A milestone whose M1 gate is a real `### M1 Gate` heading block."""
    return VALID.replace(
        "M1 Gate:\n- End-to-end performance line: ≤1.2 ms [ESTIMATE]\n- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n",
        "### M1 Gate\n- End-to-end performance line: ≤1.2 ms [ESTIMATE]\n- Mechanism-verification line: tensor pipe active % vs the ref baseline ≥80%\n"
        + lines + "\n")


def test_gate_hygiene_rejects_compiler_allocation_pins_at_authoring():
    """A dsa_sparse v1 milestone pinned nvcc's split (USETMAXREG 0x48/0xd8, regs == 168)
    as gate lines; the student obeyed and spilled 59.8M sectors. Observations
    of the reference, not acceptance lines — authoring mode refuses them,
    bounded forms pass, and a running task's patches are not held to it."""
    pinned = _with_gate("- [VERIFY: sass-derived] USETMAXREG ∋ DEALLOC 0x48 ∧ TRY_ALLOC 0xd8 "
                        "[PRICED: 0%, job 1]\n- [VERIFY: ncu-derived] regs_per_thread == 168")
    v = validate_milestone(pinned, authoring=True)
    assert sum("compiler allocation value" in x for x in v) == 2
    assert validate_milestone(pinned) == []                     # resume mode: never held
    assert validate_milestone(pinned, baseline=pinned) == []    # patch mode: pre-existing lines
    assert sum("compiler allocation value" in x
               for x in validate_milestone(pinned, baseline=VALID)) == 2   # patch added them
    bounded = _with_gate("- [VERIFY: ncu-direct] regs_per_thread <= 200\n"
                         "- [VERIFY: ncu-direct] spill_ld_sectors == 0")
    assert validate_milestone(bounded, authoring=True) == []


def test_sass_derived_gate_lines_need_a_measured_price():
    """A SASS shape says what the ref LOOKS like; it is a gate line only once a
    probe measured what the shape is WORTH ([PRICED:]). A dsa_sparse v1 carried 11
    unpriced shape lines in its M0 gate and the student spent a whole iteration
    proving it looked like the ref. The A.5 operand-source line is that lint's own
    obligation and stays exempt."""
    unpriced = _with_gate("- [VERIFY: sass-derived] supply ∋ \"interleaved\"")
    v = validate_milestone(unpriced, authoring=True)
    assert len(v) == 1 and "[PRICED:" in v[0] and "supply" in v[0]
    assert validate_milestone(unpriced) == []                   # no PRICED era yet: grandfathered
    priced = _with_gate("- [VERIFY: sass-derived] cache_ops ∋ LDGSTS.E.BYPASS "
                        "[PRICED: -22.2%, job 42]")
    assert validate_milestone(priced, authoring=True) == []
    # patch mode: only gate lines the patch ADDED are held (baseline = pre-patch
    # text); pricing one of many shape lines must not turn the rest into
    # violations, and a resume gate (no baseline) never re-authors over it
    mixed = _with_gate("- [VERIFY: sass-derived] cache_ops ∋ LDGSTS.E.BYPASS [PRICED: -22.2%, job 42]\n"
                       "- [VERIFY: sass-derived] tight-chained copies == 0")
    assert validate_milestone(mixed) == []
    v = validate_milestone(mixed, baseline=priced)
    assert len(v) == 1 and "tight-chained" in v[0]
    assert validate_milestone(unpriced, baseline=unpriced) == []
    assert validate_milestone(priced.replace("job 42", "job 1") , baseline=priced) == []
    exempt = _with_gate("- [VERIFY: sass-derived] mma_a_reg >= 4 and mma_a_desc >= 8")
    assert validate_milestone(exempt, authoring=True) == []
    # a sass-derived line OUTSIDE a gate block is a description and is never held
    body = VALID.replace("- Cost: register budget", "- Cost: register budget\n- [VERIFY: sass-derived] supply ∋ \"interleaved\"")
    assert validate_milestone(body, authoring=True) == []


def test_gate_lines_admit_only_the_four_verifiable_classes_at_authoring():
    inferred = _with_gate("- [VERIFY: inferred] KV ring segment = 32 key × 512 d")
    v = validate_milestone(inferred, authoring=True)
    assert len(v) == 1 and "only ncu-direct/ncu-derived/correctness/sass-derived" in v[0]
    assert validate_milestone(inferred) == []
    none_line = _with_gate("- [VERIFY: none] prologue warm-up depth = 4")
    assert any("[VERIFY: none]" in x for x in validate_milestone(none_line, authoring=True))


def test_gate_blocks_see_the_three_gate_spellings():
    from forgekernel.milestone import gate_blocks
    text = ("**M0 Gate**：correctness\n- a [VERIFY: sass-derived] foo\n\n"
            "M1 Gate:\n- b [VERIFY: sass-derived] bar\n\n"
            "### M2 Gate (final ruling)\n- c [VERIFY: sass-derived] baz\n\n## 4. Known dead ends\nx\n")
    blocks = gate_blocks(text)
    assert [t for t, _ in blocks] == ["M0 Gate", "M1 Gate", "M2 Gate"]
    assert all(body.count("[VERIFY:") == 1 for _, body in blocks)   # bodies end at the next gate


def test_gate_hygiene_holds_plain_and_bold_gate_lines_at_authoring():
    plain = VALID.replace("M1 Gate:\n", "M1 Gate:\n- [VERIFY: sass-derived] supply ∋ \"interleaved\"\n")
    assert any("[PRICED:" in x for x in validate_milestone(plain, authoring=True))
    bold = VALID.replace("M1 Gate:\n", "**M1 Gate**：\n- [VERIFY: sass-derived] supply ∋ \"interleaved\"\n")
    assert any("[PRICED:" in x for x in validate_milestone(bold, authoring=True))
    # a milestone with no gate block at all cannot be held: authoring says so
    none = VALID.replace("M1 Gate:", "Acceptance:")
    assert any("no gate block" in x for x in validate_milestone(none, authoring=True))
    assert not any("no gate block" in x for x in validate_milestone(none))


def test_gate_patch_retuning_a_pre_rule_line_stays_grandfathered():
    """A patch that re-thresholds a pre-existing unpriced shape line (number /
    operator only) makes no new claim; only a genuinely new gate line is held."""
    base = _with_gate("- [VERIFY: sass-derived] tight-chained copies == 0")
    retuned = _with_gate("- [VERIFY: sass-derived] tight-chained copies <= 2")
    assert validate_milestone(retuned, baseline=base) == []
    added = _with_gate("- [VERIFY: sass-derived] tight-chained copies == 0\n"
                       "- [VERIFY: sass-derived] supply ∋ \"interleaved\"")
    v = validate_milestone(added, baseline=base)
    assert len(v) == 1 and "supply" in v[0]


def test_retuning_a_bound_into_a_compiler_pin_is_a_new_claim():
    """Shape grandfathering covers a re-threshold, not a change of kind: a
    bounded gate line re-tuned into an allocation pin is held, while a pin
    the baseline already carried stays grandfathered (verbatim or re-tuned)."""
    bounded = _with_gate("- [VERIFY: ncu-direct] regs_per_thread <= 200")
    pinned = _with_gate("- [VERIFY: ncu-direct] regs_per_thread == 168")
    v = validate_milestone(pinned, baseline=bounded)
    assert len(v) == 1 and "compiler allocation value" in v[0]
    assert validate_milestone(pinned, baseline=pinned) == []
    repinned = _with_gate("- [VERIFY: ncu-direct] regs_per_thread == 176")
    assert validate_milestone(repinned, baseline=pinned) == []


def test_shape_grandfathering_does_not_cover_a_class_change_or_a_dropped_price():
    """The re-tune waiver is for the SAME claim at a new threshold. A line that
    keeps its words but changes its VERIFY class (sass-derived -> inferred /
    none) is a new claim and the class rule holds it; a priced sass-derived
    line re-tuned WITHOUT its [PRICED] tag is a priced claim re-issued without
    its price and the price rule holds it. Only a re-tune of a line the
    baseline itself carried unpriced stays waived."""
    priced = _with_gate("- [VERIFY: sass-derived] tight-chained copies == 0 [PRICED: -3%, job 123]")
    downgraded = _with_gate("- [VERIFY: inferred] tight-chained copies == 0")
    v = validate_milestone(downgraded, baseline=priced)
    assert len(v) == 1 and "[VERIFY: inferred]" in v[0]
    none_retuned = _with_gate("- [VERIFY: none] tight-chained copies <= 2")
    v = validate_milestone(none_retuned, baseline=priced)
    assert len(v) == 1 and "[VERIFY: none]" in v[0]
    unpriced_retune = _with_gate("- [VERIFY: sass-derived] tight-chained copies <= 1")
    v = validate_milestone(unpriced_retune, baseline=priced)
    assert len(v) == 1 and "[PRICED:" in v[0] and "re-tunes carried a price" in v[0]
    # re-priced re-tune: the same claim, a new threshold, a new price — waived
    repriced = _with_gate("- [VERIFY: sass-derived] tight-chained copies <= 1 [PRICED: -1.5%, job 124]")
    assert validate_milestone(repriced, baseline=priced) == []
    # a baseline line that was never priced may still be re-tuned unpriced
    unpriced_base = _with_gate("- [VERIFY: sass-derived] tight-chained copies == 0")
    assert validate_milestone(unpriced_retune, baseline=unpriced_base) == []


# --- pre-scheduled downgrades (POLICY §3 ③ is a failure response, not a step) ---

from forgekernel.milestone import downgrade_schedule_violations  # noqa: E402

PLACEHOLDER_ORDER = (
    "Construction order: ① launchable skeleton → whole-machine smoke;\n"
    "② mount the high-risk mechanisms one at a time: TMA load ring → setmaxnreg →\n"
    "solver (first use an f32 scalar forward substitution for the whole-block inverse to get first green) → epilogue;\n"
    "③ after first green, switch the solver to D1.1's fp16 block-doubling form.\n")


def test_prescheduled_downgrade_is_flagged_when_authoring():
    """The construction order schedules a scalar stand-in with zero failures
    behind it. That stand-in crosses a gate that sets no performance line."""
    ms = VALID.replace("M1 Gate:", PLACEHOLDER_ORDER + "M1 Gate:")
    v = validate_milestone(ms, authoring=True)
    hits = [x for x in v if "Pre-scheduled downgrade" in x]
    assert len(hits) == 1 and "first use an f32 scalar forward substitution" in hits[0]
    assert "(line " in hits[0]
    # the resume mode leaves a milestone already on disk alone
    assert not any("Pre-scheduled downgrade" in x for x in validate_milestone(ms))


def test_downgrade_conditioned_on_two_failures_conforms():
    row = ("| fallback (a) | if (b)'s smem-transpose implementation FAILs correctness twice "
           "in a row, you may temporarily use (a) to get first green, then swap back to (b) |\n")
    ms = VALID.replace("M1 Gate:", row + "M1 Gate:")
    assert not any("Pre-scheduled downgrade" in x for x in validate_milestone(ms, authoring=True))


def test_failure_trigger_on_the_previous_line_counts():
    """The rule's own wording spreads the trigger and the downgrade over a
    line break — the clause is read with its paragraph."""
    rule = ("- **Two failures, unbundle**: after the same bundle fails whole-machine twice, resubmitting "
            "it as-is is forbidden; the high-risk mechanisms in it must be taken down to\n"
            "  the conservative form (cooperative scalar copies, compiler-default registers), get first "
            "green, then swap them back one at a time after green.\n")
    ms = VALID.replace("M1 Gate:", rule + "M1 Gate:")
    assert not any("Pre-scheduled downgrade" in x for x in validate_milestone(ms, authoring=True))


def test_a_ban_on_placeholders_is_not_a_downgrade():
    ban = ("- Trigger condition: build it this way from the first M0 version; it is not allowed to "
           "\"first use plain STS/LDS to get it running, then swap\" — swapping the atom is a rewrite.\n")
    ms = VALID.replace("M1 Gate:", ban + "M1 Gate:")
    assert not any("Pre-scheduled downgrade" in x for x in validate_milestone(ms, authoring=True))


def test_patch_mode_flags_only_the_downgrade_clauses_the_patch_added():
    before = VALID.replace("M1 Gate:", PLACEHOLDER_ORDER + "M1 Gate:")
    assert downgrade_schedule_violations(before, baseline=before) == []
    after = before + ("\n- Trigger condition: first use a scalar placeholder to get first green, "
                      "then switch to this decision.\n")
    hits = [x for x in validate_milestone(after, baseline=before) if "Pre-scheduled downgrade" in x]
    assert len(hits) == 1 and "scalar placeholder" in hits[0]
