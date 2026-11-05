"""The timed-window CONTENT contract: what each side's stopwatch actually covers.

The two-window timing contract (``src/forgekernel/card_template/eval/timing.py``,
``forgekernel.timing_audit``) settles *how* a reading is taken — per-call vs
pipelined, host tax in or out. It says nothing about *what work sits inside the
window*. The reference's timed call and the student's timed call are two
different functions with two different boundaries, and wherever those
boundaries differ one side is quietly paying for work the other gets for free:

* a backward whose reference is called the way its own autograd Function calls
  it — with the tensors the forward saved (a chunk cumsum, an intra-chunk
  solve, an LSE) — while the student rebuilds them from the raw inputs inside
  its window (the reference is favoured);
* a student signature that receives a derived tensor the reference computes
  inside its window, or skips an output the reference must produce (the
  student is favoured);
* a library whose second call finds a weight / workspace cache warm;
* a reference window that carries an auxiliary reduction or copy the student
  contract never mentions.

Every one of those is a fixed tax on one side that no kernel-quality lever can
remove, and the ratio the run is judged on is loose or tight by exactly that
tax. None of it is visible in the numbers: a student that has matched the
reference kernel-for-kernel still reads 0.8x, and the diagnosis goes looking
for the missing 20% in register ledgers.

So the boundary is declared once per op card, in ``window.toml`` next to
``README.md``, machine-checked at ``loop start`` in BOTH directions (an
undeclared difference on either side's inputs or outputs refuses to start the
way an unmeasured device window does), banked as a per-kernel table on both
sides, and PRICED: the launches that exist only to do the extra work are named
by a regex on the side that pays, and the confirmation reports the credited
ratio next to the raw one. The raw ratio stays the gate — the credit is a
diagnostic, not a discount.

File layout (``operators/<op>/window.toml``)::

    schema_version = 1

    [reference]
    window  = "bwd call only (autograd path)"
    inputs  = ["q", "k", "v", "g_cumsum", "A", "beta", "do"]
    outputs = ["dq", "dk", "dv", "dbeta", "dg"]

    [student]
    window  = "bwd call from raw inputs"
    inputs  = ["q", "k", "v", "g", "beta", "do"]
    outputs = ["dq", "dk", "dv", "dg", "dbeta"]

    [[asymmetry]]                 # keyed by EXACTLY ONE of ref_input / student_input /
    ref_input = "A"               #   ref_output / student_output
    class = "price"               # symmetrize | price | accept
    produced_by = "forward"       # forward | prep | cache | other
    derived_from = "k, beta"      # the OTHER side's raw inputs this is derived from
    student_kernel_regex = "k1_"  # price, student pays: its launches that only do this work
    note = "per-chunk triangular solve"

Who pays, and therefore which regex an entry may carry:

* ``ref_input`` (the student does not receive it) and ``student_output`` (the
  reference does not produce it): the STUDENT pays — ``student_kernel_regex``.
* ``student_input`` (the reference does not receive it, so it computes it in
  its window) and ``ref_output`` (the student does not produce it): the
  REFERENCE pays — ``ref_kernel_regex``.

Classes:

* ``symmetrize`` (inputs only) — both sides receive it; the entry documents
  a derived tensor that costs neither window.
* ``price`` — the paying side does the work inside its window. With a regex
  the work is priced off that side's NCU per-kernel table and the
  confirmation reports a credited ratio.
* ``accept`` — the paying side does the work and the run knowingly pays;
  recorded, never credited.

A student-side extra input named in some ``ref_input`` entry's
``derived_from`` (the raw form of a derived reference input) is explained by
that entry and needs none of its own; symmetrically for the other direction.

The prep step (symmetrize, mechanically). A symmetrized input reaches the
student's timed call from a PREP step the bench runs outside every window
(``eval/timing.py`` ``measure_windows(fn, prep=...)``), the way the reference's
forward produced it. That step is the one place work legitimately leaves the
window, so it is fenced from the harness side: each symmetrize entry declares
the tensor's ``shape`` and ``dtype``; the bench reports what prep actually
produced (name / shape / dtype, plus the step's own timing) in the eval JSON;
the driver holds the report to the declaration — an undeclared extra, a
missing tensor, a wrong shape or dtype fails the iteration like a correctness
FAIL (``check_prep_report``); and ``loop start`` refuses a card whose bench
never passes ``prep=`` while the contract says it must (``bench_prep_findings``).
The prep cost is banked per iteration and shown to the teacher next to both
tables, so moving work outside the window is visible, never free.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

WINDOW_FILENAME = "window.toml"
SCHEMA_VERSION = 1

CLASS_SYMMETRIZE = "symmetrize"
CLASS_PRICE = "price"
CLASS_ACCEPT = "accept"
CLASSES = (CLASS_SYMMETRIZE, CLASS_PRICE, CLASS_ACCEPT)

PRODUCED_BY = ("forward", "prep", "cache", "other")

KIND_REF_INPUT = "ref_input"
KIND_STUDENT_INPUT = "student_input"
KIND_REF_OUTPUT = "ref_output"
KIND_STUDENT_OUTPUT = "student_output"
KINDS = (KIND_REF_INPUT, KIND_STUDENT_INPUT, KIND_REF_OUTPUT, KIND_STUDENT_OUTPUT)

PAYER_STUDENT = "student"
PAYER_REFERENCE = "reference"
_PAYER = {KIND_REF_INPUT: PAYER_STUDENT, KIND_STUDENT_OUTPUT: PAYER_STUDENT,
          KIND_STUDENT_INPUT: PAYER_REFERENCE, KIND_REF_OUTPUT: PAYER_REFERENCE}
_REGEX_KEY = {PAYER_STUDENT: "student_kernel_regex", PAYER_REFERENCE: "ref_kernel_regex"}

# ``loop start`` exit code for an undeclared / inconsistent window contract.
# NOT 12: that already means "chain-member contract check failed".
EXIT_WINDOW_UNDECLARED = 13

# How far the profiled per-call kernel total may sit from the timed window
# reading before a credit read off that table is refused as unreconciled. Same
# arithmetic as the mis-aimed-profile alarm (timing_audit.ncu_coverage), same
# loose band: a table that also holds the bench's setup launches dilutes every
# share, and a window the table does not cover inflates them.
COVERAGE_BAND = 0.20


def _names(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,\s]+", text or "") if x.strip()]


@dataclass
class Asymmetry:
    kind: str                       # one of KINDS: which list the name is extra on
    name: str
    cls: str
    produced_by: str = ""
    derived_from: str = ""          # the other side's raw inputs this one derives from
    student_kernel_regex: str = ""
    ref_kernel_regex: str = ""
    note: str = ""
    # symmetrize only: what the prep step must hand the student's timed call.
    shape: list[int] | None = None
    dtype: str = ""

    @property
    def prep_checked(self) -> bool:
        """Is this a tensor the bench's prep step must produce (and is held to)?
        Every symmetrized input except a warm cache, which is not a tensor."""
        return self.cls == CLASS_SYMMETRIZE and self.produced_by != "cache"

    @property
    def payer(self) -> str:
        """Which side does the extra work inside its window."""
        return _PAYER[self.kind]

    @property
    def regex(self) -> str:
        return self.student_kernel_regex if self.payer == PAYER_STUDENT else self.ref_kernel_regex

    @property
    def priced(self) -> bool:
        """Can this entry be credited off the paying side's per-kernel table?"""
        return self.cls == CLASS_PRICE and bool(self.regex)

    @property
    def derived_names(self) -> list[str]:
        return _names(self.derived_from)

    # kept for callers written against the first schema draft
    @property
    def ref_input(self) -> str:
        return self.name if self.kind == KIND_REF_INPUT else ""


@dataclass
class WindowContract:
    reference_inputs: list[str] = field(default_factory=list)
    reference_outputs: list[str] = field(default_factory=list)
    student_inputs: list[str] = field(default_factory=list)
    student_outputs: list[str] = field(default_factory=list)
    asymmetries: list[Asymmetry] = field(default_factory=list)
    reference_window: str = ""
    student_window: str = ""
    schema_version: int = SCHEMA_VERSION
    path: str = ""

    @property
    def declared(self) -> bool:
        """Has the author filled the file in (vs the scaffolded placeholder)?
        All four lists: an output list left empty would hide a skipped output."""
        return bool(self.reference_inputs and self.student_inputs
                    and self.reference_outputs and self.student_outputs)

    @property
    def missing_on_student(self) -> list[str]:
        """Reference inputs the student's timed call does not receive."""
        have = set(self.student_inputs)
        return [x for x in self.reference_inputs if x not in have]

    @property
    def extra_on_student(self) -> list[str]:
        """Student inputs the reference's timed call does not receive — the
        reference computes them inside its window (or never needs them)."""
        have = set(self.reference_inputs)
        return [x for x in self.student_inputs if x not in have]

    @property
    def outputs_missing_on_student(self) -> list[str]:
        have = set(self.student_outputs)
        return [x for x in self.reference_outputs if x not in have]

    @property
    def outputs_extra_on_student(self) -> list[str]:
        have = set(self.reference_outputs)
        return [x for x in self.student_outputs if x not in have]

    def entry(self, kind: str, name: str) -> Asymmetry | None:
        return next((a for a in self.asymmetries if a.kind == kind and a.name == name), None)

    def asymmetry_for(self, ref_input: str) -> Asymmetry | None:
        return self.entry(KIND_REF_INPUT, ref_input)

    @property
    def priced_entries(self) -> list[Asymmetry]:
        return [a for a in self.asymmetries if a.priced]

    @property
    def unpriced_entries(self) -> list[Asymmetry]:
        """``price``-class entries nobody has named a regex for yet."""
        return [a for a in self.asymmetries if a.cls == CLASS_PRICE and not a.priced]

    def paid_by(self, payer: str) -> list[Asymmetry]:
        return [a for a in self.asymmetries if a.cls != CLASS_SYMMETRIZE and a.payer == payer]

    @property
    def prep_entries(self) -> list[Asymmetry]:
        """Symmetrized inputs the bench's prep step must produce, in file order."""
        return [a for a in self.asymmetries if a.prep_checked]


def window_path(op_dir: Path) -> Path:
    return Path(op_dir) / WINDOW_FILENAME


def _str_list(table: dict, key: str, where: str) -> list[str]:
    value = table.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where}.{key} must be a list of strings")
    out: list[str] = []
    for x in value:
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def parse_window_contract(text: str, *, path: str = "") -> WindowContract:
    """Parse ``window.toml`` text. Raises ValueError on a malformed file.

    Malformed (not TOML, wrong types, unknown schema, an entry keyed by none or
    several of the four kinds) is distinct from *undeclared* (a placeholder
    with empty lists) — the latter parses fine and is reported by
    ``window_findings``.
    """
    label = path or WINDOW_FILENAME
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{label} is not valid TOML: {e}") from e
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{label} has schema_version={version}, expected {SCHEMA_VERSION}")
    ref = data.get("reference", {}) or {}
    stu = data.get("student", {}) or {}
    if not isinstance(ref, dict) or not isinstance(stu, dict):
        raise ValueError(f"{label}: [reference] and [student] must be tables")
    asym_raw = data.get("asymmetry", []) or []
    if not isinstance(asym_raw, list) or not all(isinstance(a, dict) for a in asym_raw):
        raise ValueError(f"{label}: [[asymmetry]] must be an array of tables")
    asymmetries = []
    for i, a in enumerate(asym_raw, 1):
        keyed = [(k, str(a.get(k, "")).strip()) for k in KINDS if str(a.get(k, "")).strip()]
        if len(keyed) != 1:
            raise ValueError(f"{label}: [[asymmetry]] #{i} must be keyed by exactly one of "
                             f"{', '.join(KINDS)} (got {len(keyed)})")
        kind, name = keyed[0]
        shape = a.get("shape")
        if shape is not None and not (isinstance(shape, list)
                                      and all(isinstance(s, int) and not isinstance(s, bool)
                                              for s in shape)):
            raise ValueError(f"{label}: [[asymmetry]] #{i} shape must be a list of ints")
        asymmetries.append(Asymmetry(
            kind=kind, name=name,
            cls=str(a.get("class", "")).strip(),
            produced_by=str(a.get("produced_by", "")).strip(),
            derived_from=str(a.get("derived_from", "")).strip(),
            student_kernel_regex=str(a.get("student_kernel_regex", "")).strip(),
            ref_kernel_regex=str(a.get("ref_kernel_regex", "")).strip(),
            note=str(a.get("note", "")).strip(),
            shape=list(shape) if shape is not None else None,
            dtype=str(a.get("dtype", "")).strip(),
        ))
    return WindowContract(
        reference_inputs=_str_list(ref, "inputs", "[reference]"),
        reference_outputs=_str_list(ref, "outputs", "[reference]"),
        student_inputs=_str_list(stu, "inputs", "[student]"),
        student_outputs=_str_list(stu, "outputs", "[student]"),
        asymmetries=asymmetries,
        reference_window=str(ref.get("window", "")).strip(),
        student_window=str(stu.get("window", "")).strip(),
        schema_version=int(version),
        path=path,
    )


def load_window_contract(op_dir: Path) -> WindowContract | None:
    """The op card's window contract, or None when the file is absent.

    Absence is reported by the caller (a run with a reference must declare);
    a file that exists but cannot be read raises, the way a corrupt baseline
    archive does — proceeding on a half-read contract is the failure mode this
    module exists to close.
    """
    p = window_path(op_dir)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"{p} is unreadable: {e}") from e
    return parse_window_contract(text, path=str(p))


def _kind_findings(contract: WindowContract, a: Asymmetry) -> list[str]:
    where = f"[[asymmetry]] {a.kind}={a.name!r}"
    out: list[str] = []
    ref_in, stu_in = contract.reference_inputs, contract.student_inputs
    ref_out, stu_out = contract.reference_outputs, contract.student_outputs
    own = {KIND_REF_INPUT: ref_in, KIND_STUDENT_INPUT: stu_in,
           KIND_REF_OUTPUT: ref_out, KIND_STUDENT_OUTPUT: stu_out}[a.kind]
    other = {KIND_REF_INPUT: stu_in, KIND_STUDENT_INPUT: ref_in,
             KIND_REF_OUTPUT: stu_out, KIND_STUDENT_OUTPUT: ref_out}[a.kind]
    own_list, other_list = {
        KIND_REF_INPUT: ("[reference].inputs", "[student].inputs"),
        KIND_STUDENT_INPUT: ("[student].inputs", "[reference].inputs"),
        KIND_REF_OUTPUT: ("[reference].outputs", "[student].outputs"),
        KIND_STUDENT_OUTPUT: ("[student].outputs", "[reference].outputs"),
    }[a.kind]
    if a.name not in own:
        out.append(f"{where} names something that is not in {own_list}")
    if a.cls not in CLASSES:
        out.append(f"{where} has class={a.cls!r}; must be one of {', '.join(CLASSES)}")
        return out
    on_other = a.name in other
    if a.kind in (KIND_REF_OUTPUT, KIND_STUDENT_OUTPUT):
        if a.cls == CLASS_SYMMETRIZE:
            out.append(f"{where}: outputs cannot be symmetrize — an output both sides "
                       f"produce needs no entry; class it price/accept")
        elif on_other:
            out.append(f"{where} is class={a.cls} but both sides produce it — remove "
                       f"the entry")
    else:
        if a.cls == CLASS_SYMMETRIZE and not on_other:
            out.append(f"{where} is class=symmetrize but only one side receives it — add it "
                       f"to {other_list}, or class it price/accept")
        if a.cls != CLASS_SYMMETRIZE and on_other:
            out.append(f"{where} is class={a.cls} but both sides receive it — class it "
                       f"symmetrize, or remove it from {other_list}")
    if a.produced_by and a.produced_by not in PRODUCED_BY:
        out.append(f"{where} has produced_by={a.produced_by!r}; must be one of "
                   f"{', '.join(PRODUCED_BY)}")
    if a.prep_checked:
        # The prep step is fenced by this declaration: without it nothing bounds
        # what the student may receive from outside the window.
        if not a.shape:
            out.append(f"{where} is symmetrize but declares no shape — the bench's prep "
                       f"step is held to it (shape = [..] as the student's timed call "
                       f"receives it)")
        if not a.dtype:
            out.append(f"{where} is symmetrize but declares no dtype — the bench's prep "
                       f"step is held to it (dtype = \"bf16\" / \"fp32\" / ...)")
    elif a.shape or a.dtype:
        out.append(f"{where} carries shape/dtype but is not a prep-produced symmetrize "
                   f"entry — only symmetrized inputs are held to a prep declaration")
    wrong_key = _REGEX_KEY[PAYER_REFERENCE if a.payer == PAYER_STUDENT else PAYER_STUDENT]
    if getattr(a, wrong_key):
        out.append(f"{where} carries {wrong_key} but the {a.payer} side pays for it — "
                   f"use {_REGEX_KEY[a.payer]}")
    if a.regex:
        if a.cls != CLASS_PRICE:
            out.append(f"{where} carries {_REGEX_KEY[a.payer]} but is class={a.cls} — only "
                       f"price-class entries are credited")
        else:
            try:
                re.compile(a.regex)
            except re.error as e:
                out.append(f"{where} {_REGEX_KEY[a.payer]} is not a valid regex: {e}")
    return out


def window_findings(contract: WindowContract | None) -> list[str]:
    """Why this contract cannot anchor a run. Empty = consistent and declared.

    Both directions are held to the same rule: every reference input the
    student does not receive, every student input the reference does not
    receive, and every output only one side produces needs an entry. Each
    finding names the fix. The list is what the loop prints before refusing to
    start (exit ``EXIT_WINDOW_UNDECLARED``).
    """
    if contract is None:
        return [f"{WINDOW_FILENAME} is absent — `forge run` generates it from the judge's "
                f"signature (taskcard.render_window_toml); a hand-made tree declares the "
                f"reference's and the student's timed-call boundaries in it"]
    out: list[str] = []
    for lst, where, what in (
            (contract.reference_inputs, "[reference].inputs",
             "every tensor the reference's timed call receives, by the reference API's own names"),
            (contract.student_inputs, "[student].inputs",
             "every tensor the student's timed call receives (the frozen signature in README)"),
            (contract.reference_outputs, "[reference].outputs",
             "every tensor the reference's timed call produces"),
            (contract.student_outputs, "[student].outputs",
             "every tensor the student's timed call produces")):
        if not lst:
            out.append(f"{where} is empty — list {what}")
    if out:
        return out
    seen: set[tuple[str, str]] = set()
    explained_ref_inputs: set[str] = set()
    explained_student_inputs: set[str] = set()
    for a in contract.asymmetries:
        key = (a.kind, a.name)
        if key in seen:
            out.append(f"[[asymmetry]] {a.kind}={a.name!r} is declared twice")
            continue
        seen.add(key)
        out.extend(_kind_findings(contract, a))
        if a.kind == KIND_REF_INPUT:
            explained_student_inputs.update(a.derived_names)
        elif a.kind == KIND_STUDENT_INPUT:
            explained_ref_inputs.update(a.derived_names)
    for name in contract.missing_on_student:
        if (KIND_REF_INPUT, name) not in seen and name not in explained_ref_inputs:
            out.append(f"reference input {name!r} is not a student input and has no "
                       f"[[asymmetry]] ref_input entry — declare it (class symmetrize / "
                       f"price / accept), or name it in a student_input entry's derived_from")
    for name in contract.extra_on_student:
        if (KIND_STUDENT_INPUT, name) not in seen and name not in explained_student_inputs:
            out.append(f"student input {name!r} is not a reference input and has no "
                       f"[[asymmetry]] student_input entry — the reference computes it "
                       f"inside its window: declare it (class price / accept), or name it "
                       f"in a ref_input entry's derived_from")
    for name in contract.outputs_missing_on_student:
        if (KIND_REF_OUTPUT, name) not in seen:
            out.append(f"reference output {name!r} is not produced by the student and has no "
                       f"[[asymmetry]] ref_output entry — declare it (class price / accept)")
    for name in contract.outputs_extra_on_student:
        if (KIND_STUDENT_OUTPUT, name) not in seen:
            out.append(f"student output {name!r} is not produced by the reference and has no "
                       f"[[asymmetry]] student_output entry — declare it (class price / accept)")
    return out


def window_warnings(contract: WindowContract | None) -> list[str]:
    """Non-refusing notes: what the contract cannot yet price."""
    if contract is None or not contract.declared:
        return []
    out = []
    for a in contract.unpriced_entries:
        out.append(f"asymmetry {a.kind}={a.name!r} is class=price but has no "
                   f"{_REGEX_KEY[a.payer]} — the credited ratio will not be computed until "
                   f"the {a.payer} side's launches doing that work are named (see its "
                   f"per-kernel table)")
    return out


# --- the prep step: held to the declaration ----------------------------------

_DTYPE_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "half": "fp16", "fp16": "fp16",
    "float32": "fp32", "float": "fp32", "fp32": "fp32",
    "float64": "fp64", "double": "fp64", "fp64": "fp64",
    "float8_e4m3fn": "e4m3", "e4m3": "e4m3", "float8_e5m2": "e5m2", "e5m2": "e5m2",
    "int64": "i64", "long": "i64", "i64": "i64",
    "int32": "i32", "int": "i32", "i32": "i32",
    "int16": "i16", "i16": "i16", "int8": "i8", "i8": "i8",
    "uint8": "u8", "u8": "u8", "bool": "bool",
}


PREP_UNREPORTED = "the bench reported no prep block"


def prep_report_missing(report) -> bool:
    """No usable prep block at all: a structural bench defect (the card's
    frozen bench never ran a prep step), not something a student edit heals."""
    return not isinstance(report, dict) or not isinstance(report.get("outputs"), dict)


def normalize_dtype(dtype) -> str:
    """``torch.bfloat16`` / ``bfloat16`` / ``bf16`` -> ``bf16``; unknown names pass
    through lower-cased, so a declaration and a report written differently still
    compare equal when they mean the same thing."""
    s = str(dtype or "").strip().lower()
    if s.startswith("torch."):
        s = s[len("torch."):]
    return _DTYPE_ALIASES.get(s, s)


def check_prep_report(contract: WindowContract | None, report: dict | None) -> list[str]:
    """Hold what the bench's prep step produced to the contract's symmetrize entries.

    ``report`` is the eval JSON's ``prep`` block (``timing.measure_windows``:
    ``outputs`` = {name: {shape, dtype}}). Empty list = nothing to hold or all
    held. Each mismatch names the tensor and the fix; the driver fails the
    iteration on any of them, so a student cannot widen what its timed call
    receives from outside the window without changing the declaration. A
    multi-workload block (``timing.workloads_performance``) is held per workload.
    """
    if contract is None or not contract.declared:
        return []
    entries = contract.prep_entries
    if not entries:
        return []
    names = ", ".join(a.name for a in entries)
    subs = report.get("workloads") if isinstance(report, dict) else None
    if isinstance(subs, list) and subs:
        # A multi-workload bench (timing.measure_windows_workloads) runs the prep
        # step once per workload: hold EVERY workload's block, not only the
        # campaign one the top level mirrors. The same problem on several
        # workloads is said once, naming them.
        by_problem: dict[str, list[str]] = {}
        for i, sub in enumerate(subs):
            if not isinstance(sub, dict):
                continue
            tag = str(sub.get("name", i))
            for problem in _check_prep_block(entries, sub, names):
                by_problem.setdefault(problem, []).append(tag)
        return [f"{p} (workload{'s' if len(t) > 1 else ''} {', '.join(t)})"
                for p, t in by_problem.items()]
    return _check_prep_block(entries, report, names)


def _check_prep_block(entries, report: dict, names: str) -> list[str]:
    """``check_prep_report`` on one prep block (one workload's)."""
    if prep_report_missing(report):
        return [f"{PREP_UNREPORTED} — the symmetrized inputs ({names}) must be produced "
                f"outside the window by measure_windows(fn, prep=...) and the 'prep' block "
                f"placed in the eval JSON's performance"]
    outputs = report["outputs"]
    out: list[str] = []
    mutated = report.get("mutated")
    if isinstance(mutated, list) and mutated:
        out.append(f"the timed call mutated the prep tensor(s) {', '.join(map(str, mutated))} "
                   f"— they are handed to every timed call and must stay read-only; state "
                   f"kept there makes every call after the first a copy")
    for a in entries:
        got = outputs.get(a.name)
        if not isinstance(got, dict):
            out.append(f"prep did not produce {a.name!r} (declared symmetrize; got "
                       f"{', '.join(sorted(outputs)) or 'nothing'})")
            continue
        shape = [int(s) for s in (got.get("shape") or [])]
        if a.shape and shape != list(a.shape):
            out.append(f"prep produced {a.name!r} with shape {shape}, declared {list(a.shape)}")
        if a.dtype and normalize_dtype(got.get("dtype")) != normalize_dtype(a.dtype):
            out.append(f"prep produced {a.name!r} with dtype {got.get('dtype')!r}, "
                       f"declared {a.dtype!r}")
    declared = {a.name for a in entries}
    for name in outputs:
        if name not in declared:
            got = outputs[name] if isinstance(outputs[name], dict) else {}
            out.append(f"prep produced an undeclared tensor {name!r} (shape "
                       f"{got.get('shape')}, dtype {got.get('dtype')}) — the timed call may "
                       f"only receive what window.toml symmetrizes: declare it on both "
                       f"sides, or stop producing it")
    return out


KERNEL_MODULE = "kernel"
_MEASURE_PREFIX = "measure_windows"


def _callee_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _prep_keyword(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "prep":
            return kw.value
    return None


def _kernel_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound by ``from kernel import ...``, aliases bound by ``import
    kernel``) anywhere in the module — the student's namespace."""
    names: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == KERNEL_MODULE:
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == KERNEL_MODULE:
                    modules.add(a.asname or a.name.split(".")[0])
    return names, modules


def _references(body: ast.AST, names: set[str], modules: set[str]) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(body):
        if isinstance(node, ast.Name) and node.id in (names | modules):
            hits.append(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (getattr(node, "module", "") or "").split(".")[0]
            if mod == KERNEL_MODULE or any(a.name.split(".")[0] == KERNEL_MODULE
                                           for a in node.names):
                hits.append(KERNEL_MODULE)
    return sorted(set(hits))


def bench_prep_analysis(bench_src: str) -> dict:
    """What the bench does about its prep step, statically.

    Returns ``{"prep": <name or None>, "findings": [...]}``. The rules, each a
    finding when broken:

    * a ``measure_windows*`` call passes ``prep=`` and the value is not the
      ``None`` literal (``prep=None`` is the no-prep form spelled out);
    * the value is a NAME bound to a module-level ``def`` in bench.py — not a
      lambda, an attribute, or an import — so ``ncu_run.py`` can share it;
    * that def references nothing bound from the ``kernel`` module (no
      ``from kernel import ...`` name, no ``kernel.`` attribute, no import of
      it inside the body). The prep step is the card author's, written with
      torch; a student that owns it owns the work outside the window.
    """
    try:
        tree = ast.parse(bench_src)
    except SyntaxError as e:
        return {"prep": None, "findings": [f"eval/bench.py does not parse: {e}"]}
    prep_value = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node).startswith(_MEASURE_PREFIX):
            value = _prep_keyword(node)
            if value is not None and not (isinstance(value, ast.Constant) and value.value is None):
                prep_value = value
                break
    if prep_value is None:
        return {"prep": None, "findings": [
            "eval/bench.py never passes prep= (other than None) to measure_windows"]}
    if not isinstance(prep_value, ast.Name):
        return {"prep": None, "findings": [
            "eval/bench.py passes prep= something other than a plain name — the prep step "
            "must be a module-level def in bench.py (not a lambda / attribute / call), so "
            "eval/ncu_run.py can import and run the same function"]}
    name = prep_value.id
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == name]
    if not defs:
        return {"prep": name, "findings": [
            f"eval/bench.py passes prep={name} but {name!r} is not a module-level def in "
            f"bench.py — the prep step must be defined there (an import from kernel would "
            f"hand the student the work outside the window)"]}
    names, modules = _kernel_bindings(tree)
    hits = _references(defs[0], names, modules)
    if hits:
        return {"prep": name, "findings": [
            f"eval/bench.py prep function {name!r} references the kernel module "
            f"({', '.join(hits)}) — the prep step is the card author's, written with torch; "
            f"nothing from kernel.py may run outside the timed window"]}
    return {"prep": name, "findings": []}


def ncu_run_prep_findings(ncu_src: str, prep_name: str) -> list[str]:
    """``eval/ncu_run.py`` must run the SAME prep function as bench.py, and run
    it before ``torch.cuda.profiler.start()``: a prep launched inside the
    profiler window enters the NCU table, where it dilutes the credit's shares,
    can steal the dominant-kernel slot, and pollutes the wgmma census."""
    try:
        tree = ast.parse(ncu_src)
    except SyntaxError as e:
        return [f"eval/ncu_run.py does not parse: {e}"]
    imported = False
    aliases = {prep_name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "bench":
            for a in node.names:
                if a.name == prep_name:
                    imported = True
                    aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import) and any(a.name == "bench" for a in node.names):
            imported = True   # bench.<prep_name>(...) form
    calls = []
    starts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in aliases:
            calls.append(node.lineno)
        elif (isinstance(f, ast.Attribute) and f.attr == prep_name
              and isinstance(f.value, ast.Name) and f.value.id == "bench"):
            calls.append(node.lineno)
        elif (isinstance(f, ast.Attribute) and f.attr == "start"
              and isinstance(f.value, ast.Attribute) and f.value.attr == "profiler"):
            starts.append(node.lineno)
    out = []
    if not imported or not calls:
        out.append(f"eval/ncu_run.py does not import and call bench.py's prep function "
                   f"{prep_name!r} (from bench import {prep_name}) — the NCU job must feed "
                   f"the timed call the same prep tensors the bench does")
        return out
    if not starts:
        out.append("eval/ncu_run.py never calls torch.cuda.profiler.start() — without a "
                   "confined profiler window the prep launches enter the NCU table")
        return out
    first_start = min(starts)
    late = [ln for ln in calls if ln > first_start]
    if late:
        out.append(f"eval/ncu_run.py calls the prep function {prep_name!r} at line "
                   f"{late[0]}, after torch.cuda.profiler.start() at line {first_start} — "
                   f"prep must run before the profiler window opens, or its launches enter "
                   f"the NCU table (diluted credit, wrong dominant kernel, polluted census)")
    return out


MULTI_WORKLOAD_CALL = "measure_windows_workloads"
PERF_WORKLOADS = 3   # mirrors src/forgekernel/card_template/eval/timing.py PERF_WORKLOADS
# A leaderboard card's bench (src/forgekernel/card_template/eval/leaderboard.py) takes its input
# sets from the board's own protocol — FlashInfer-Bench times its three judged draws,
# SOL-ExecBench the last of its ten seeded rounds. The card's verdict must be the
# board's, so the board decides what gets timed, not this rule.
LEADERBOARD_CALL = "run_leaderboard_bench"


def bench_workloads_findings(bench_src: str) -> list[str]:
    """Does the bench time several workloads of its shape? (static, ``loop start``)

    The perf verdict of a card is the geometric mean over ``PERF_WORKLOADS``
    workloads (timing.py ``measure_windows_workloads``): a kernel timed on one
    input set fits that input set. Findings when the bench never calls
    ``measure_windows_workloads``, or calls it with a literal list/tuple of fewer
    than ``PERF_WORKLOADS`` entries. A list built at runtime cannot be counted
    here; the driver checks ``n_workloads`` in the eval JSON on every iteration.
    """
    try:
        tree = ast.parse(bench_src)
    except SyntaxError as e:
        return [f"eval/bench.py does not parse: {e}"]
    if any(isinstance(n, ast.Call) and _callee_name(n) == LEADERBOARD_CALL
           for n in ast.walk(tree)):
        return []
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and _callee_name(n) == MULTI_WORKLOAD_CALL]
    if not calls:
        return [f"eval/bench.py never calls timing.{MULTI_WORKLOAD_CALL} — the perf "
                f"verdict must be the geometric mean over {PERF_WORKLOADS} representative "
                f"workloads of the card's shape, not one input set (template README, "
                f"\"Performance is judged on the geomean of 3 workloads\")"]
    out = []
    for call in calls:
        arg = call.args[0] if call.args else next(
            (kw.value for kw in call.keywords if kw.arg == "workloads"), None)
        if isinstance(arg, (ast.List, ast.Tuple)) and len(arg.elts) < PERF_WORKLOADS:
            out.append(f"eval/bench.py line {call.lineno}: {MULTI_WORKLOAD_CALL} gets "
                       f"{len(arg.elts)} workload(s), fewer than {PERF_WORKLOADS}")
    return out


def bench_times_workloads(op_dir: Path) -> bool | None:
    """Does ``<op_dir>/eval/bench.py`` report the geomean over several workloads
    (call ``measure_windows_workloads``)? None when there is no parsable bench."""
    bench = Path(op_dir) / "eval" / "bench.py"
    try:
        tree = ast.parse(bench.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    if any(isinstance(n, ast.Call) and _callee_name(n) == LEADERBOARD_CALL for n in ast.walk(tree)):
        # a leaderboard card: a workload GROUP (problem.json "group") reports the geomean
        try:
            problem = json.loads((Path(op_dir) / "eval" / "leaderboard" / "problem.json")
                                 .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return len(problem.get("group") or []) > 1
    return any(isinstance(n, ast.Call) and _callee_name(n) == MULTI_WORKLOAD_CALL
               for n in ast.walk(tree))


def bench_workloads_preflight(op_dir: Path) -> list[str]:
    """``bench_workloads_findings`` on ``<op_dir>/eval/bench.py``; [] when absent."""
    bench = Path(op_dir) / "eval" / "bench.py"
    if not bench.exists():
        return []
    try:
        return bench_workloads_findings(bench.read_text(encoding="utf-8"))
    except OSError as e:
        return [f"eval/bench.py is unreadable: {e}"]


def bench_uses_prep(bench_src: str) -> bool:
    """Does the bench run a conforming prep step? (``bench_prep_analysis``.)"""
    return not bench_prep_analysis(bench_src)["findings"]


def bench_prep_findings(op_dir: Path, contract: WindowContract | None) -> list[str]:
    """Static ``loop start`` check: a contract with prep-produced symmetrize
    entries needs a bench that runs a conforming prep step
    (``bench_prep_analysis``) and an NCU launch target that runs the same one
    outside the profiler window (``ncu_run_prep_findings``). Reads
    ``eval/bench.py`` and ``eval/ncu_run.py``; an absent file is nothing to say
    here (other preflights own that)."""
    if contract is None or not contract.declared or not contract.prep_entries:
        return []
    op_dir = Path(op_dir)
    bench = op_dir / "eval" / "bench.py"
    if not bench.exists():
        return []
    try:
        src = bench.read_text(encoding="utf-8")
    except OSError as e:
        return [f"eval/bench.py is unreadable: {e}"]
    names = ", ".join(a.name for a in contract.prep_entries)
    analysis = bench_prep_analysis(src)
    if analysis["findings"]:
        return [f"{f} — the symmetrized inputs ({names}) need a window-outside producer "
                f"the card author owns (timing.py module doc)" for f in analysis["findings"]]
    ncu = op_dir / "eval" / "ncu_run.py"
    if not ncu.exists():
        return []
    try:
        ncu_src = ncu.read_text(encoding="utf-8")
    except OSError as e:
        return [f"eval/ncu_run.py is unreadable: {e}"]
    return ncu_run_prep_findings(ncu_src, analysis["prep"])


# --- pricing off the per-kernel tables ----------------------------------------

def _kernel_time(k: dict) -> float:
    try:
        return float(k.get("time_ns") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _side_credit(entries: list[Asymmetry], kernels: list[dict] | None) -> dict:
    """Price one side's entries off that side's table."""
    out = {"priced": [a.name for a in entries], "fraction": 0.0, "matched": [],
           "matched_ns": 0.0, "total_ns": 0.0, "unmatched": [], "table": bool(kernels)}
    if not entries:
        return out
    if not kernels:
        out["unmatched"] = [a.name for a in entries]
        return out
    total = sum(_kernel_time(k) for k in kernels)
    out["total_ns"] = total
    matched: dict[str, float] = {}
    for a in entries:
        pat = re.compile(a.regex)
        found = False
        for k in kernels:
            name = str(k.get("name", ""))
            if pat.search(name):
                found = True
                matched[name] = _kernel_time(k)
        if not found:
            out["unmatched"].append(a.name)
    out["matched"] = sorted(matched)
    out["matched_ns"] = sum(matched.values())
    out["fraction"] = (out["matched_ns"] / total) if total > 0 else 0.0
    return out


def _coverage(profiler: dict | None, window_ms: float | None) -> float | None:
    if not profiler or not window_ms:
        return None
    from forgekernel.timing_audit import ncu_coverage
    return ncu_coverage(profiler, window_ms)


def window_credit(contract: WindowContract | None, student_kernels: list[dict] | None,
                  ref_kernels: list[dict] | None = None, *,
                  student_profiler: dict | None = None, student_ms: float | None = None,
                  ref_profiler: dict | None = None, ref_ms: float | None = None,
                  ) -> dict | None:
    """Price the declared ``price`` asymmetries off the per-kernel tables.

    Each side's entries are priced against that side's table (``inner.ncu.
    kernel_table`` rows: name / launches / time_ns / pct): the student's
    rebuild launches against the student table, the reference's in-window
    extras against the reference table. Shares are window totals over window
    totals, so they are launch-count independent.

    Returns None when nothing is priced (no contract, no priced entry). Else::

        student / reference: {fraction, matched, matched_ns, total_ns, priced,
                              unmatched, table}
        usable: bool          # every priced entry matched, tables present,
                              # and the tables reconcile with the windows
        why: [reasons]        # when not usable
        unpriced: [...]       # price-class entries without a regex
        reconciliation: {student, reference}  # profiled per-call share of the
                              # timed reading, when the digest and reading are given

    **A credit that cannot be trusted is NOT a credit of 0.** A regex that
    matches nothing (the rebuild was fused into the main kernel, or renamed)
    would otherwise read as "no tax" and hand out a credited ratio equal to the
    raw one; a table that also holds the bench's setup launches dilutes every
    share; a table that does not cover the timed window inflates them. All of
    those set ``usable=False`` and ``credited_ratio`` then refuses.
    """
    if contract is None or not contract.declared:
        return None
    priced = contract.priced_entries
    if not priced:
        return None
    stu = _side_credit([a for a in priced if a.payer == PAYER_STUDENT], student_kernels)
    ref = _side_credit([a for a in priced if a.payer == PAYER_REFERENCE], ref_kernels)
    why: list[str] = []
    for side, part in (("student", stu), ("reference", ref)):
        if part["priced"] and not part["table"]:
            why.append(f"no {side} per-kernel table on record")
        elif part["unmatched"]:
            why.append(f"{side} regex matched nothing for "
                       f"{', '.join(part['unmatched'])} (fused / renamed?)")
    recon = {}
    for side, part, prof, ms in (("student", stu, student_profiler, student_ms),
                                 ("reference", ref, ref_profiler, ref_ms)):
        if not part["priced"]:
            continue
        cov = _coverage(prof, ms)
        if cov is None:
            continue
        recon[side] = round(cov, 3)
        if abs(cov - 1.0) > COVERAGE_BAND:
            why.append(f"{side} NCU table does not reconcile with its timed window "
                       f"(profiled per-call kernel time = {cov:.2f}x the reading; band "
                       f"±{COVERAGE_BAND:.0%}) — setup launches in the table, or the "
                       f"window not covered")
    out = {"student": stu, "reference": ref, "usable": not why, "why": why,
           "unpriced": [a.name for a in contract.unpriced_entries]}
    if recon:
        out["reconciliation"] = recon
    return out


def credited_ratio(credit: dict | None, student_ms: float | None,
                   ref_ms: float | None) -> dict | None:
    """The ratio with the priced extra work taken out of BOTH readings.

    ``ratio = ref / student`` is the run's currency (``>= 1`` = at parity or
    better). credited student = ``student_ms * (1 - student fraction)``,
    credited ref = ``ref_ms * (1 - reference fraction)``. None when any input
    is missing or the credit is not usable. Never used for a verdict — see
    module doc.
    """
    if not credit or not credit.get("usable") or not student_ms or not ref_ms:
        return None
    sf = float(credit["student"]["fraction"])
    rf = float(credit["reference"]["fraction"])
    credited_student = student_ms * (1.0 - sf)
    credited_ref = ref_ms * (1.0 - rf)
    if credited_student <= 0 or credited_ref <= 0:
        return None
    return {"credited_student_ms": credited_student, "credited_ref_ms": credited_ref,
            "student_ms": student_ms, "ref_ms": ref_ms,
            "credited_ratio": credited_ref / credited_student,
            "raw_ratio": ref_ms / student_ms,
            "student_fraction": sf, "reference_fraction": rf}


# --- rendering ---------------------------------------------------------------

def _short(name: str, width: int = 64) -> str:
    if len(name) <= width:
        return name
    head = width // 2 - 1
    return name[:head] + "…" + name[-(width - head - 1):]


def baseline_label(baseline) -> str:
    """``role name job N`` for a RefBaseline, with a long ref_name cut short —
    the archive's name field is a sentence on some cards, and the ledger row
    has to stay a row."""
    if baseline is None:
        return ""
    name = _short(str(baseline.ref_name or ""), 48)
    return f"{baseline.role} {name} job {baseline.job or '?'}".strip()


def render_kernel_rows(kernels: list[dict] | None, window_ms: float | None = None,
                       indent: str = "    ") -> list[str]:
    """One line per kernel: launches, share, and — when the window reading is
    known — the per-call estimate (share x window reading). Sorted by time."""
    if not kernels:
        return [f"{indent}(no per-kernel table banked)"]
    rows = sorted(kernels, key=_kernel_time, reverse=True)
    total = sum(_kernel_time(k) for k in rows) or 0.0
    out = []
    for k in rows:
        t = _kernel_time(k)
        pct = (100.0 * t / total) if total else 0.0
        est = f"  ≈{pct / 100.0 * window_ms * 1000:.1f} µs/call" if window_ms else ""
        out.append(f"{indent}{_short(str(k.get('name', '?')))}  launches={k.get('launches', '?')}"
                   f"  {pct:.1f}%{est}")
    return out


_KIND_LABEL = {
    KIND_REF_INPUT: "Inputs the reference window gets for free and the student must rebuild inside its window",
    KIND_STUDENT_INPUT: "Inputs the student gets for free and the reference must compute inside its window",
    KIND_REF_OUTPUT: "Outputs the reference window must produce and the student does not",
    KIND_STUDENT_OUTPUT: "Extra outputs the student window produces and the reference does not",
}


def _entry_line(a: Asymmetry) -> str:
    how = f", derived from {a.derived_from}" if a.derived_from else ""
    price = ""
    if a.cls == CLASS_PRICE:
        price = (f", priced by {_REGEX_KEY[a.payer]}=`{a.regex}`" if a.priced
                 else f", **unpriced** (no {_REGEX_KEY[a.payer]})")
    note = f" — {a.note}" if a.note else ""
    return (f"    - {a.name}: class={a.cls}, {a.payer} pays"
            f"{f', produced by {a.produced_by}' if a.produced_by else ''}{how}{price}{note}")


def _sym_label(a: Asymmetry) -> str:
    spec = ""
    if a.shape or a.dtype:
        spec = f" {tuple(a.shape) if a.shape else '?'} {a.dtype or '?'}"
    return f"{a.name}({a.produced_by or '?'}{spec})"


def render_prep_lines(contract: WindowContract, student_prep: dict | None,
                      indent: str = "    ") -> list[str]:
    """The student's window-outside prep step as the teacher reads it: what it
    must produce, what the latest banked report says it produced, and what it
    cost. Empty when the contract symmetrizes nothing."""
    entries = contract.prep_entries
    if not entries:
        return []
    lines = ["- Student prep step outside the window (produces the symmetrize tensors, symmetric "
             "with the reference's forward; **not counted in the ratio**; the driver checks it "
             "every round by shape/dtype, and an extra / missing / mis-shaped output = REVERT for "
             "that round):"]
    if not isinstance(student_prep, dict) or not student_prep:
        lines.append(f"{indent}(no prep reading banked yet — the bench does not call "
                     f"measure_windows(prep=...), or there is no KEEP yet)")
        return lines
    outputs = student_prep.get("outputs") if isinstance(student_prep.get("outputs"), dict) else {}
    for a in entries:
        got = outputs.get(a.name) if isinstance(outputs.get(a.name), dict) else None
        seen = (f"{tuple(got.get('shape') or ())} {got.get('dtype')}" if got
                else "not produced")
        lines.append(f"{indent}{a.name}: declared {tuple(a.shape) if a.shape else '?'} "
                     f"{a.dtype or '?'}, produced {seen}")
    ms = []
    if student_prep.get("device_median_ms") is not None:
        ms.append(f"device {float(student_prep['device_median_ms']):.4f} ms/call")
    if student_prep.get("median_ms") is not None:
        ms.append(f"e2e {float(student_prep['median_ms']):.4f} ms/call")
    if ms:
        lines.append(f"{indent}prep cost: {', '.join(ms)} (the reference-side equivalent work = "
                     f"its forward, not banked; a prep step clearly heavier than \"computing these "
                     f"tensors\" itself means in-window work has been moved outside the window)")
    return lines


def render_window_ledger(contract: WindowContract | None, *,
                         ref_kernels: list[dict] | None = None,
                         ref_ms: float | None = None, ref_label: str = "",
                         ref_profiler: dict | None = None,
                         student_kernels: list[dict] | None = None,
                         student_ms: float | None = None, student_label: str = "",
                         student_profiler: dict | None = None,
                         student_prep: dict | None = None,
                         ) -> str:
    """The block the teacher reads: declared boundary in both directions, both
    per-kernel tables, the student's prep-step reading, and the credit (or why
    there is none). Prose + indented rows, like ``RefBaseline.render``."""
    lines = ["### Timed-window ledger (WINDOW)"]
    if contract is None:
        lines.append(f"- **Not declared**: the operator tree has no `{WINDOW_FILENAME}`. Nobody "
                     f"records whether the boundaries of the reference's timed call and the "
                     f"student's timed call agree — any estimate of \"work outside the window\" in "
                     f"the milestone has no basis, and the difference between the two readings "
                     f"may contain a fixed contract tax.")
        return "\n".join(lines)
    findings = window_findings(contract)
    if findings:
        lines.append(f"- **Declaration incomplete** (`{contract.path or WINDOW_FILENAME}`):")
        lines.extend(f"    - {f}" for f in findings)
        return "\n".join(lines)
    lines.append(f"- Reference window: {contract.reference_window or '(not described)'}; inputs "
                 f"{', '.join(contract.reference_inputs)}; outputs "
                 f"{', '.join(contract.reference_outputs)}")
    lines.append(f"- Student window: {contract.student_window or '(not described)'}; inputs "
                 f"{', '.join(contract.student_inputs)}; outputs "
                 f"{', '.join(contract.student_outputs)}")
    sym = [a for a in contract.asymmetries if a.cls == CLASS_SYMMETRIZE]
    if sym:
        lines.append("- Derived inputs both sides get outside the window (symmetrize, counted "
                     "on neither side): "
                     + ", ".join(_sym_label(a) for a in sym))
    lines.extend(render_prep_lines(contract, student_prep))
    any_tax = False
    for kind in KINDS:
        items = [a for a in contract.asymmetries if a.kind == kind and a.cls != CLASS_SYMMETRIZE]
        if not items:
            continue
        any_tax = True
        lines.append(f"- **{_KIND_LABEL[kind]}**:")
        lines.extend(_entry_line(a) for a in items)
    if not any_tax:
        lines.append("- Asymmetries: none (the two timed calls have the same boundary)")
    lines.append(f"- Launches inside the reference window (NCU{f', {ref_label}' if ref_label else ''}"
                 f"{f'; device {ref_ms} ms' if ref_ms else ''}):")
    lines.extend(render_kernel_rows(ref_kernels, ref_ms))
    lines.append(f"- Launches inside the student window (NCU{f', {student_label}' if student_label else ''}"
                 f"{f'; device {student_ms} ms' if student_ms else ''}):")
    lines.extend(render_kernel_rows(student_kernels, student_ms))
    credit = window_credit(contract, student_kernels, ref_kernels,
                           student_profiler=student_profiler, student_ms=student_ms,
                           ref_profiler=ref_profiler, ref_ms=ref_ms)
    if credit:
        for side, who in (("student", "Student"), ("reference", "Reference")):
            part = credit[side]
            if part["priced"] and part["table"]:
                lines.append(f"- {who}-side pricing: the launches matching the regex take "
                             f"{part['fraction'] * 100:.1f}% of the {who.lower()}'s device time"
                             f" ({', '.join(_short(m, 40) for m in part['matched']) or 'no match'})")
        if credit.get("reconciliation"):
            lines.append("- NCU table reconciled against the timed window (per-call total kernel "
                         "time / window reading): "
                         + ", ".join(f"{s} {v:.2f}" for s, v in credit["reconciliation"].items()))
        cr = credited_ratio(credit, student_ms, ref_ms)
        if cr:
            lines.append(f"    raw ratio {cr['raw_ratio']:.4f} → tax-credited ratio "
                         f"{cr['credited_ratio']:.4f}"
                         f" (student {student_ms} → {cr['credited_student_ms']:.4f} ms, "
                         f"reference {ref_ms} → {cr['credited_ref_ms']:.4f} ms)")
        elif not credit["usable"]:
            lines.append("- **Pricing unavailable**: " + "; ".join(credit["why"]))
        if credit["unpriced"]:
            lines.append(f"- Unpriced price items (no regex): {', '.join(credit['unpriced'])}")
    elif any_tax and any(a.cls == CLASS_PRICE for a in contract.asymmetries):
        lines.append("- Pricing: none yet (the price-class items have no regex)")
    lines.append("- How to read this: the asymmetry tax explains the part of the two readings that "
                 "**kernel quality cannot remove**; stop, KEEP and target still use the raw ratio, "
                 "and the tax-credited ratio only goes into diagnosis and the finish report.")
    return "\n".join(lines)
