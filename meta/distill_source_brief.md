# Teacher — Initial distillation authoring step (source and measured baseline)

Write a concise optimization milestone to `{MILESTONE_PATH}`. The student implements it in
`{LANGUAGE}` on `{KB_ARCH}`. Give decisions and their rationale in prose. Keep implementation
code and private reference details out of the milestone.

## Operator

{SOURCE_BRIEF}

Read the relevant reference source at `{ORACLE}`. Use the supplied measurements and task
contract to choose the first useful implementation. The brief below has the required schema.
Limit file reads to the reference and any task facts needed for the decision.

## Measured baseline

{ARCH_REF}

Use `target_ms = {TARGET_MS}` for the `{GATE_WINDOW}` timing window. Preserve the timed
workloads and the harness correctness checks.

{WINDOW_CONTRACT}

## Milestone schema

Start with `# <operator> milestone`, then use these headings in order:

```text
## 0. Problem statement and target
## 1. Problem contract (Math Spec)
## 2. Decision DAG + phase order
## 3. Decision details
## 4. Known dead ends
## 5. Global correctness / performance gate
## 6. Rules
## Worked Example
```

- Include the target, shapes, mathematical contract and numerical risks. Tie any claimed
  bottleneck to the source or the measured baseline. Mark performance numbers `[MEASURED]`
  with their job or `[ESTIMATE]` with their basis wherever they appear in decisions.
- Start with M0 and use only the phases needed. Give each decision a `### D0.1 Name` heading
  (increment the ID) and `Action`, `Purpose`, `Trigger condition`, `Cost` and `Depends on`
  fields. Use `⇐ D0.1` for dependencies and `—` for a root.
- Use `[SYNC: none]`, `[GATHER: none]` and `[PHASE: none]` for absent features. The other
  values are `mbarrier-pipeline` or `named-barrier` for cross-role synchronization,
  `prefetched` or `serial` for indirect gathers and `ping-pong` or `convoy` for consumer
  phases. Anchor each chosen mechanism to observed evidence.
- After each phase's decisions add a `### M0 Gate` block (using the phase's ID). Include a
  `Mechanism-verification line:` stating what observable would show that the intended mechanism
  is active. Require correctness on every workload. Use measured evidence for hard resource
  thresholds and leave unmeasured tuning choices as estimates in the decision body.
- Keep the milestone compact. Write the file before ending the turn. If the controller
  returns schema feedback, read the existing milestone and revise the cited lines.
