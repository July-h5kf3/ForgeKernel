# ForgeKernel 代码梳理（code map）

> 面向读代码的人：harness 由哪几层组成、一次 `forge run` 走过哪些函数、每个模块归谁管、
> 哪些地方是设计上已经定死但代码里仍然"可配"的旋钮。本文对应 `simplify-harness` 分支的状态；
> 该分支砍掉了什么、留了什么，见 [`design/v1-launch.md` §15](design/v1-launch.md)。
> 行数以该分支为准，仅作量级参考。

## 1. 一句话模型

ForgeKernel 是一个**两层循环**：**teacher**（论文里的 arch agent）读参考实现、做侦察、蒸馏出
milestone（决策图 + 门槛），**student**（coding agent）按 milestone 一行决策一行决策地写 kernel，
每一轮由一个 GPU job 判对错、计时、做 NCU profile。student 卡住或用完决策图时 teacher 回来诊断、
给 lever 定价、补 milestone。跑完后把最优 kernel 重测确认、跑正确性监视、收割知识库条目。

三个产物的边界是整个设计的核心：**milestone**（每算子一份，只有决策和参数，不许有实现体）、
**hw_kb/**（跨算子的硬件机制知识库，唯一允许带代码的产物）、**meta_milestone.md**（蒸馏策略）。
相关设计文档：`design/milestone-distillation.md`、`meta_milestone.md`、`milestones-template.md`。

## 2. 分层

```
用户面      tasks/<op>/{task.toml, description.md, shapes.json, reference/}   global_config/*.toml   local.env
             │
启动层      src/forgekernel/{__main__, launch, task, runs, localenv, judge, taskcard, scaffold, agents, provider, paths}.py
             │  生成 .artifacts/<op>_<id>/{run.json, task/, check/, operators/<op>/, operators/_<op>ref/, ws/}
循环层      src/forgekernel/loop/{cli, driver, parse, agent_policy, goal_ref, harness_guard, precision_gate,
             │                     meta_shapes, kb_lint, lang_mode, dsl_source, harvest, job_env}.py
             │  + milestone.py, meta_trajectory.py, inner/{trajectory, ncu}.py, resume/, baseline.py, baseline_cli.py
执行层      src/forgekernel/submit/{jobs, direct, config, cli}.py      一台机器 / 一个 ssh 主机，一卡一锁
GPU 侧      src/forgekernel/card_template/eval/{leaderboard, timing, precision, closed_ref, ncu_probe}.py
             │  被复制进每个 operator 树的 eval/，随 `git archive` 进 job；加 sealref/binref 的封装工具
代理资产    meta/*.md（teacher/student 提示词，meta/cuda/ 为 CUDA 覆盖）  skills/（agent 工具）  hw_kb/
```

## 3. 一次 `forge run` 的流程

`__main__.main` → `launch.COMMANDS["run"]` → `launch.cmd_run` → `launch.drive` →
`launch._drive`（普通任务）或 `launch._drive_board`（leaderboard 任务）：

1. **judge**（`launch.write_judge`）：teacher 用 `judge.judge_prompt` 从 `description.md` 写出
   `check/{definition.json, judge.py, groups.json}`，`judge.check_problems` 静态检查，最多 3 轮。
2. **trees**（`taskcard.write_card`）：复制模板 4 个文件（`taskcard.copy_template`）+ `eval/*.py`，生成
   `problem.json`（`taskcard.card_problem`，suite = "forgekernel"）、`eval/bench.py`、`eval/ncu_run.py`、
   `gpu_entry.sh`（模板来自 `leaderboard.py` 的 `*_TEMPLATE`）、kernel 桩、`window.toml`、`submit.toml`，
   `git init`。有参考实现时 `taskcard.seal_reference` 用 `sealref.write_trees` 把参考源码封进
   `ref_bin/ref_payload.bin`，密钥写入 gitignored 的 `submit.local.toml`，并生成 teacher 专用的侦察树
   `operators/_<op>ref/`。
3. **calibration**（`launch.calibrate` → GPU job `calibrate` 模式）：参考实现 vs judge，在每个 shape
   的每组输入上比对，容差 = 2 × 最大误差（有 dtype 下限），不一致则回到第 1 步重写 judge。
4. **NCU probe**（`launch.probe_ncu`）：一个短 job 看能否读计数器；不能则 loop.toml 写入
   `ncu_available = false` 等开关，运行退化为只计时。
5. **loop.toml / workspace**：`taskcard.write_loop_toml` 写循环配置（语言、arch、计时窗口、参考如何到达、
   拒读路径），`taskcard.clone_workspace` 单分支克隆出 `ws/`。
6. **loop**（`launch.run_loop` → 进程内调用 `loop.cli.main`）：参数只有 workspace、小时数、
   max_meta、两个 agent 命令（`agents.agent_command` → `scripts/agent-claude.sh`）、目标（ms 或比例）、
   恢复标志。其余旋钮是 `loop/cli.py` 顶部的常量。
7. **report**（`launch.write_report`）：读 `loop_summary.json` 与 `best_confirmation.json`，写
   `report.md`、`best/kernel.*`，列出待人工审阅的新 KB 条目。

`loop.cli._main` 在交给 driver 前的预检顺序：oracle 模式 → goal_ref 计划（无参考时先用 /goal 会话写一个，
`loop/goal_ref.py` + `scripts/claude-goal.py`）→ 语言 / KB arch → agent policy 声明与命令扫描
（`loop/agent_policy.py`）→ window.toml 契约 → 多 workload 预检 → 新开始污染检查
（`fresh_start_contamination`）→ 测参考基线（`_ensure_arch_baseline` / `_ensure_ref_baseline` →
`baseline_cli.measure_and_archive`）→ 推导 target_ms → 构造 `RealBackend` 与 `LoopContext` →
harness guard / precision 预检 → KB 脏检查与泄漏 lint → job 环境预检 → `driver.run_two_layer` →
最优 sha 重测确认（`driver.run_best_confirmation`）→ 正确性监视（`monitor.run`）→ KB 收割
（`driver.run_kb_harvest`）→ `loop_summary.json`。

## 4. 循环核心：`loop/driver.py`（约 1.2 万行）

纯逻辑、用 `DriverBackend` 协议与外界隔离，测试用 `FakeBackend` 驱动。入口 `run_two_layer(ctx, max_meta)`：

| 阶段 | 函数 | 做什么 |
|---|---|---|
| 侦察 | `run_recon_step` | teacher 在参考树上跑 SASS/NCU 探针（`skills/` 里的工具被播种到 `notes/`），产出 report / candidates / 各类台账；完成门槛检查证据齐全 |
| 蒸馏 | `run_distill_step` | teacher 按 `meta/distill_prompt.md` 写 milestone v1，`milestone.validate_milestone` 机器校验 §9 骨架，`_seed_target_ms` 写入目标 |
| 内环 | `run_inner_loop` → `run_inner_iteration` | 每轮：student 一个 turn（可先跑布局探针）→ 提交 → GPU `all` job 判对错计时 → NCU → KEEP/REVERT 记录到 `student/status.jsonl`；停止条件：到达目标、同 job 平价、连续 REVERT、决策图耗尽、平台期 |
| M0 审计 | `run_meta_step(audit=True)` | 第一次绿灯后强制 teacher 对照硬件审计 milestone |
| 外环 | `run_meta_step` | 收集证据（NCU 报告、stall/role/timeline 台账、lever 定价 `_run_lever_pricing`、dossier diff）→ teacher 诊断（G/R 分类，可做 verdict probe）→ patch milestone → `_judge_patch` 判 KEEP_PATCH / HOLD_PATCH / REVERT_PATCH → `resume/` 算前缀复用的恢复点 |
| 收尾 | `run_best_confirmation`、`run_kb_harvest`、`apply_window_credit` | 多 job 重测最优 sha；把探针证明过的机制写进 hw_kb 待审 |

driver 里按功能分区大致是：`LoopContext`（约 80 个字段）、提示词绑定（`_bindings`、`_student_prompt`）、
占位符台账、best/ratio 选择与跨机器重锚、探针反馈、NCU 解析与 host-bound 审计、harness guard、
内环、外环、lever 定价与价格陈旧、timeline 探针、相位耗尽、reopen 门、证据台账的存取、蒸馏、确认。
约 650 行是写给 teacher 看的说明文字常量（`_LEVER_DESIGN_BLOCK`、各类 `*_GATE_MSG`），放在
`meta/` 更合适。

## 5. 模块归属速查

| 模块 | 职责 | 主要调用方 |
|---|---|---|
| `task.py` | 读 `tasks/<op>/` 并合并 `global_config/`，推导模式（source / blackbox / generate / leaderboard） | launch、taskcard、judge、self_evolve |
| `runs.py` / `localenv.py` / `paths.py` | 运行目录生命周期、`local.env`、`FORGEKERNEL_OPERATORS_DIR` | launch、submit |
| `judge.py` | judge 提示词与静态检查 | launch |
| `taskcard.py` | 从任务生成 operator 树与 loop.toml / submit.toml / window.toml | launch、scaffold |
| `scaffold.py` | `forge init --leaderboard` 的卡片生成器 | launch |
| `leaderboard.py` | FlashInfer-Bench / SOL-ExecBench 数据集读取、卡片渲染、解答导出 | scaffold、taskcard、launch.cmd_export |
| `sealref.py` / `sealref_tool.py` | 封装 Python 源码参考（源码模式、blackbox、自进化）与侦察树 | taskcard、scaffold |
| `binref.py` / `binref_tool.py` | 封装编译后的 cuda_cpp 参考（仅 `--binary-ref` 卡片） | scaffold；sealref 复用其 guard 片段 |
| `window_contract.py` | 读/校验 `window.toml`，渲染两侧 per-kernel 台账，不对称项定价 | loop.cli 预检、driver |
| `baseline.py` / `baseline_cli.py` | 参考基线数据结构、测量与归档、收尾时的重测 | loop.cli、driver |
| `timing_audit.py` | host-bound 判定与 NCU 覆盖率核对 | driver、baseline |
| `inner/trajectory.py` | 内环记录 `IterationRecord` 与机械摘要 `build_digest` | driver、cli |
| `inner/ncu.py` | NCU CSV 解析、按 launch 聚合、报告渲染；`forge <op> ncu-trace\|ncu-report` 由 teacher 技能调用 | driver |
| `meta_trajectory.py` | 外环记录 `MetaRecord` | driver |
| `milestone.py` | milestone 文本校验（§9 骨架、五元组、门槛行、标签）与 DAG 解析 | driver、cli |
| `resume/invalidation.py` | 按 DAG 算 patch 后的恢复点与 reopen 闭包 | driver |
| `loop/parse.py` | 解析 student 回合、eval JSON、探针判决、平台期检测 | driver、goal_ref |
| `loop/agent_policy.py` | 每个 turn 前写 `.claude/settings.json`（拒读路径、离线沙箱、PreToolUse 钩子 `scripts/agent_policy_hook.py`） | cli、launch、goal_ref |
| `loop/harness_guard.py` | 评测器属于卡片：job 树覆盖卡片的 eval/、gpu_entry.sh，student 改动即 REVERT | cli、driver |
| `loop/precision_gate.py` | 误差比精度门（生成的树上默认 off） | cli、driver |
| `loop/meta_shapes.py` | 多 shape 时 lever 定价 / stall / timeline 的逐 shape 读数合并 | driver |
| `loop/kb_lint.py` | hw_kb 泄漏 lint（参考实现的符号名不得进 KB） | cli、driver |
| `loop/lang_mode.py` | cutedsl 与 cuda 两种模式：提示词覆盖目录、KB 根、provenance 目录 | 多处 |
| `loop/dsl_source.py` | 离线 student 可读的 CuTe DSL 源码目录（从本机安装推导） | cli |
| `loop/goal_ref.py` + `scripts/claude-goal.py` | 无参考时用 /goal 会话写参考：请求假脱机、评测台账、验收重测 | cli |
| `submit/` | job 记录（`.jobs/<id>/`）、本机/ssh 执行器、GPU 锁、`submit.toml` 读取 | cli.RealBackend |
| `monitor/` | 收尾正确性监视：带新种子与压力输入重跑 judge，观察 NaN / 不确定性 | cli |
| `skills/__init__.py` + `skills/` | 7 个技能目录，每个 turn 镜像进 workspace 的 `.claude/skills/`；脚本被 driver 播种进 `notes/` 在 GPU 上运行 | agent_policy、driver |
| `scripts/` | `agent-claude.sh`（agent 命令）、`timeout-run.py`、`agent_policy_hook.py`、`blackbox-guard.sh`、`self_evolve.py` | agents、driver、用户 |

## 6. 配置面：谁写、谁读

- 用户只写 `tasks/<op>/task.toml`（叠在 `global_config/` 之上）和 `local.env`。
- `forge run` 生成 `loop.toml`（`taskcard.write_loop_toml` / `write_board_loop_toml`）和 `submit.toml`
  （`taskcard.write_submit`），每次启动重写。
- `loop/cli.py` 从 loop.toml 读的键：`language`、`kb_arch`、`gate_window`、`oracle*`、`recon_op`、
  `ref_*` / `arch_ref_*`（参考如何到达与声明值）、`ref_source_paths/urls`、`ref_from_goal*`、
  `ncu_available`、`job_env_preflight`、`timeline_probe`、`stall_pricing`、`orch_census`、
  `harness_guard`、`precision_contract`、`kb_lint_oracle_tokens`、`job_env_files`。其余旋钮是常量。
- 测试用 `LoopContext` 的字段（`m0_audit`、`stall_pricing`、`lever_pricing_rounds`、
  `reopen_requires_price`、`watermark_margin` 等）当作开关构造场景，所以这些字段保留在 dataclass 里，
  只是不再从配置读。

## 7. 仍存在的结构性冗余（本分支未动，按预期收益排序）

1. **双参考机制**：`arch_baseline.json` 与 `ref_baseline.json` 对普通任务是同一读数的两份归档
   （`loop/cli._ensure_ref_baseline` 复制 arch 读数），driver / 提示词仍分别绑定 `{ARCH_REF}` 与
   `{REF_BASELINE}`。合并涉及 target 推导与约 60 个测试，且 leaderboard 卡片与 goal_ref 依赖
   "声明的 arch_ref" 路径。约 500 行。
2. **`window.toml` 不对称项**：生成的契约两侧永远同边界，`window_contract.py` 的 asymmetry /
   prep / pricing 校验与 driver 的 prep 契约、window credit 在生成的树上永不触发。约 900 行。
3. **绝对目标与跨机器机制**：`ref_in_job = true` 时 `_confirm_apparent_target`、`_cluster_scale`、
   `_phantom_gap_reanchor` 的绝对分支不可达，但 generate 模式（无参考）仍走它们。约 550 行。
4. **timeline 探针**（driver 约 660 行）：普通任务 off，leaderboard 卡片 on。
5. **精度门**：`precision_contract` 在生成的树上默认 off（taskcard 不写），`precision.py` +
   `precision_gate.py` + bench 接线约 600 行处于休眠；要么在 `card_problem` 里打开，要么删。
6. **`meta/cuda/*.md`**：8 份覆盖提示词与共享版约 95% 相同，差异可归为 5 个按模式替换的片段。
   约 1350 行提示词。
7. **`binref`**：仅服务 `--binary-ref` 的 leaderboard 卡片；`sealref_tool` / `binref_tool` 有约
   45 行重复命令。
8. **driver 内部重复**：8 份同形的证据文件读取、7 份 jsonl 读、5 份 jsonl 追加、7 种"最优 KEEP"选择、
   5 处"提交→推送→cmd job→解析判决"的探针回合，约 250 行可合并；650 行提示词常量可移到 `meta/`。
9. **执行层的三跳**：本机 job 走 `loop → python -m forgekernel.submit → 独立 runner → bash gpu_entry.sh`，
   第一跳可以进程内调用 `jobs.spawn_runner`；`cluster` 命名是内部集群时代的残留。

## 8. 测试怎么看

`tests/` 全部 CPU 运行；`tests/test_loop_driver.py` 是事实上的夹具库（`FakeBackend`、`_ctx`、
脚本化的 `REC` / `EVAL` / `NCU` 日志），19 个文件从它导入。结构性不变量在 `test_skills.py`
（技能目录与 `skills/README.md` 对齐、播种模块可导入）、`test_kb_arch.py` / `test_kb_lint.py`
（真实 hw_kb 树可启动且 lint 干净）、`test_examples.py`（示例任务可加载、`forge` 用法文案）、
`test_repo_hygiene.py`（不得提交 `operators/` 或 `.artifacts/`）。套件依赖 Linux（`fcntl`、bash、
tmux），Windows 上需要桩模块并与干净 `main` 的失败集合做差分对比。
