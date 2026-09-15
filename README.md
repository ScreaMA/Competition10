# 《未来战争》编程大赛客户端（V2）

> 云核心网第十届编程大赛 · 客户端实现
> **V2 设计文档：[设计文档V2.md](./设计文档V2.md)**（V1 的 [设计文档.md](./设计文档.md) 与
> `CoreGeek/src/agent/brain.py` 的 7114 行单文件实现已整体废弃，仅作历史存档）

---

## 这是什么

一个跑在判题系统里的 HTTP 客户端：判题系统 POST 一回合的地图状态过来，
客户端返回这一回合所有角色的调度指令。整场比赛打两半场、每半场 1300 回合。

设计目标是**在没有人类干预的情况下把每一步都做对**，尤其是任务书 §5.3 的
自进化类任务——那一项是我们与对手分差最大、也最容易做砸的地方。

---

## 快速开始

```bash
cd CoreGeek

# 启动客户端（判题系统的调用方式）
bash run.sh 8000

# 单元测试（236 个用例）
python -m pytest tests/ -q

# 端到端自检：真实报文 + 边界场景 + 1300 回合性能
python tools/local_check.py

# 本地对局模拟：不用判题系统也能看出行为是否正常
python tools/simulate.py --rounds 400 --log sim.log
python tools/analyze_log.py --log sim.log            # 回合统计
python tools/analyze_log.py --log sim.log --task     # 自进化任务链路
python tools/analyze_log.py --log sim.log --template # 复盘填空稿

# 一键联调（启动客户端→发请求→打印响应）
bash tools/test_client.sh 8000
```

**约束**：Python ≥ 3.11，**零第三方依赖**（判题与沙盒环境都不能假设有 pip）。

---

## 模块地图

```
CoreGeek/
├── main3.py                  入口：端口校验、日志双通道、启动服务
├── run.sh                    判题系统调用方式（bash run.sh <port>）
├── src/agent/
│   ├── protocol.py           规则常量表 + 报文解析 + 指令构造（无策略）
│   ├── grid.py               几何与寻路：切比雪夫距离、A*、可达性、锥形判定
│   ├── world.py              世界模型：基地原点、敌方来向、可建造区在线学习
│   ├── telemetry.py          逐回合增量（击杀/掉血/损失/空转），只服务日志
│   ├── brain.py              编排器：decide(payload) -> response
│   ├── server.py             HTTP 服务 + 决策超时保护
│   └── strategy/
│       ├── economy.py        采集、贩卖、建造执行、金币分配
│       ├── defense.py        塔位与墙线规划、夜晚火力控制
│       └── task/             自进化任务子系统
│           ├── solver.py       状态机：调度 / 阶梯 / 反馈 / 终态
│           ├── skills.py       任务族识别 + SOP 阶梯
│           ├── scripts.py      注入沙盒的 shell/python 脚本模板
│           ├── sandbox.py      标记解析、输出指纹
│           ├── answer.py       答案闸门（唯一允许产出 submitAnswer 的地方）
│           └── memory.py       跨回合持久层：事实 / 技能 / 当前任务
├── tests/                    236 个用例（含 11 个真实故障回归）
└── tools/                    自检、模拟、复盘工具
```

## 改代码时该看哪个文件

复盘报告里的问题定位到模块，比定位到函数更有用——V2 每个模块的职责边界就是
测试边界：

| 症状 | 文件 | 典型改法 |
|------|------|---------|
| 任务：读题成功却从不调 API / 反复读同一份文档 | `strategy/task/skills.py`、`solver.py` | 调整阶梯顺序或 `STEP_MAX_ATTEMPTS` |
| 任务：沙盒命令本身写错了（取数/鉴权/分页） | `strategy/task/scripts.py` | 改 `_QUERY` 脚本模板 |
| 任务：把文档原文/错误 JSON 当答案交上去 | `strategy/task/answer.py` | 调 `gate()` 的判据 |
| 任务：技能没有跨任务复用 | `strategy/task/memory.py`、`skills.py` | 改 `signature` / `build_skill` |
| 金币冻结、采集中断、开局不建塔 | `strategy/economy.py` | 调优先级队列与卖矿触发条件 |
| 塔位堵路、围墙留口、夜晚不攻击 | `strategy/defense.py` | 调 `tower_sites` / `fire_targets` |
| 可建造区学错了 / 换边后塔位失效 | `world.py` | 改 `BuildableZone` 的消歧判据 |
| 指令非法（吃异常预算） | `protocol.py` | 改指令构造器（宁可不发，不发半成品） |
| 日志缺字段、复盘填不出表 | `brain.py` 的 `_log_turn`、`telemetry.py` | 加字段并同步 `tools/analyze_log.py` |
| 移动绕远、走位抖动 | `grid.py` | 改 A* 的平局打破项 |

---

## 自进化任务：V2 的核心

任务书 §5.3 的要求是：

> 玩家需要根据任务 1 探索的内容，形成固定 SOP 或者 SKILL，实现 Agent 自进化，
> 进而快速做出后续任务。

V2 把它落成一套**可执行的机制**，而不是一句口号：

```
观测 → 族识别 → 阶梯(或缓存技能) → 执行 executeCmd → 反馈 → 提交/放弃
   ↑                                                          │
   └──────────────── 下一回合 lastCmdResult ──────────────────┘
```

- **技能库**（`memory.py`）：第一次成功跑通的任务，把实际走通的步骤序列与
  学到的接口事实（地址/鉴权头/参数名/字段映射）沉淀成 `Skill`。
  第二个同族任务**首回合就直接下发 query**，不再重新探索。
  技能连续失败 2 次自动降级停用。
- **输出指纹**（`sandbox.py`）：沙盒输出做归一化哈希。同一份输出第二次出现
  就说明这一步卡死了，立刻跳下一步——这是对 V1 "读题成功却反复重读同一份
  文档 6 个回合"的直接对策。
- **答案闸门**（`answer.py`）：唯一允许产出 `submitAnswer` 的地方。拒绝文档
  回声、错误体、空壳；接受部分正确（任务书 §6：部分完成按通过率给分，
  交一半对严格优于不交）。
- **必有终态**（`solver.py`）：任何一个任务最终都会走到 `DONE` 或 `ABANDON`，
  且**放弃前必定先提交手上的答案**。

### 实测到的任务族

| 族 | 沙盒路径 | 形态 |
|----|---------|------|
| `api-query` | `/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/` | 读文档 → 调本地接口 → 聚合统计 → 提交 JSON（文档故意过时，要以实测为准） |
| `engineering-fix` | `/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/` | 修工作区（CRLF/权限/配置）→ 跑 `./check` → 提取 `TOKEN` → 提交 |
| 未知族 | — | `recon → generic → LLM 兜底` |

任务期间 LLM 调用**不计入每日限额**（接口文档 §1.7 的注），所以 LLM 只用在
任务里，且只在确定性阶梯全部失败后作为兜底顾问。

---

## 测试

```bash
cd CoreGeek && python -m pytest tests/ -q
```

236 个用例，其中 `tests/test_solver.py` 的 11 个用例是**真实故障回归**——
每一个都对应 Issues #1–#199 里反复出现的某一类问题：

| 用例 | 对应故障 |
|------|---------|
| `test_recon_advances_to_query_not_repeat` | T1 读题成功但 `api=0` 死循环 |
| `test_repeated_output_advances_step` | T4 反复重读同一份文档 |
| `test_success_output_triggers_submit` | T2 接任务后从未提交 |
| `test_abandon_still_submits` | T2 放弃时也不交（部分分白丢） |
| `test_rejected_answer_is_not_resubmitted` | 复读同一个被判错的答案直到额度烧完 |
| `test_auth_failure_leads_to_retry_with_auth` | T3 缺鉴权头吃 401 |
| `test_task_two_is_accepted_after_first` | T5 任务 2 全程可接未接 |
| `test_skill_reuse_skips_recon` | 自进化：第二个同族任务必须更快 |
| `test_night_keeps_pioneer_on_defense` | 开拓者为了赶路放弃守夜 |
| `test_doc_echo_is_rejected` | T7 把任务文档原文当答案 |
| `test_type_mismatch_rejected` | 把数字 `0` 写成 `"0"` |

---

## 日志与复盘

`debug.log` 是复盘流水线唯一的输入，每回合固定四行：

```
request_decoded round=… gold=… hp=… towers=… walls=… robots=…(s m l b)
                tasks=[…] phase=… task=… zone=… bag=…
strategy_done   round=… commands=… actions=… sandbox=… note=… learn=… fail=[…]
round_end       round=… kills=…(…) score=… station_damage=… idle=…
day_summary     day=… rounds=… kills=… gold_in=… submits=… sandbox=…
```

比 V1 多出来的部分是**逐回合增量**（击杀/掉血/建筑损失/空转人·回合）与
**任务链路事件**（步骤推进、交卷、放弃）——V1 的复盘报告里大量"日志未覆盖"
的结论就是因为日志只有状态快照、没有增量。

`tools/analyze_log.py --template` 能直接把这些渲染成
`战术参考/对战分析模板.md` 的填空稿。

---

## 自动化流水线（`automation/`）

> 这部分与客户端**没有代码耦合**，V2 重写客户端时未做任何改动。

### 功能

- 轮询或 webhook 监听 GitHub Issues
- 根据 Issue 标签过滤需要处理的任务
- **自动处理对战分析报告（`battle-analysis` 标签）**
- 调用本地 Claude Code 进行代码修改
- 自动创建 PR 并推送到远程仓库
- **PR 自动合并**（可关闭）：开启后无需人工审批，创建 PR 即自动合并
- **PR 合并后自动同步本地代码**（每轮轮询先检查自己创建的 PR 状态）

### 对战分析自动化

系统会自动识别带 `battle-analysis` 标签的 Issue，提取优化建议并应用：

**Issue 格式**：

```markdown
Title: 对战分析报告 (PK 580611)
Labels: auto-fix, battle-analysis

## 5. 优化建议
**S1｜围墙提前介入（改动最小、收益最大）**
- 涉及：CoreGeek/src/agent/strategy/economy.py
- 做法：STONE_BATCH 由 6 降到 2-3
- 预期：第 5-6 回合立起第一段围墙
```

**处理流程**：解析建议 → 提取"涉及/做法"生成提示词 → 改代码并跑测试 →
创建 PR 并关联原 Issue。

> 建议里的"涉及"文件请按上面的**模块地图**填——V2 已经把 `brain.py` 这一个
> 7114 行的文件拆成了 10 个模块，写 `brain.py` 的建议会让自动修复找不到落点。

### 使用流程

1. **创建 Issue**（带 `auto-fix` 标签）
2. 自动化系统处理：检测 Issue → 创建分支 → 调用 Claude 修改 → 提交推送 → 创建 PR
3. 人工审核合并（或 `auto_merge: true` 时自动合并）
4. 合并后自动同步：下一轮轮询把本地主干同步到远端最新，并清理已合并的本地分支

### 配置

`automation/automation_config.yaml`，关键项：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `poll_interval` | `60` | 轮询间隔（秒），每次轮询仅 1 次 GitHub API 调用 |
| `max_issues_per_round` | `3` | 每轮最多处理几个 Issue（只看最新的几个） |
| `auto_merge` | `true` | 设为 `false` 即恢复"人工审核后合并" |
| `auto_merge_method` | `squash` | `squash` / `merge` / `rebase` |
| `auto_sync` | `true` | 每轮先检查自己创建的 PR 是否已合并，合并后同步主干 |
| `sync_strategy` | `rebase` | 本地与远端分叉时的处理方式 |

**安全约束**：优先快进；需要 rebase/merge 时一旦冲突立即 `--abort` 回滚；
工作区有未提交改动时直接跳过；绝不用远端覆盖本地提交。

### 部署

```bash
cd automation/
pip install -r requirements.txt

# 配置 Token（默认读项目根目录的 githubtoken.txt，也可用环境变量 GITHUB_TOKEN）
echo "ghp_your_token" > ../githubtoken.txt

# 运行（--once 只处理一轮）
python automation_main.py
```

---

## 常见问题

### 本地 push 弹出 GitHub 账号选择框

```bash
git config --local credential.helper manager
git config --local user.name  "你的名字"
git config --local user.email "你的邮箱"
```

### 自动化处理 Issue 后"没改代码"

先看 `automation/automation.log`。常见原因：Issue 没有 `auto-fix` 标签、
带 `wontfix`/`manual-review` 被忽略、或者建议里点名的文件路径在仓库里不存在
（V2 之后请按模块地图填写）。

### 客户端在判题器里"什么都不做"

`brain.decide` 的兜底是**返回空指令**（空指令不算异常，任务书 §8），所以
内部异常在判题器侧只表现为"这一回合没动作"。查 `debug.log` 里的
`ERROR` / `Traceback`，或者直接跑 `python -m pytest tests/test_brain.py::test_decide_internals_never_raise`
——那个用例专门绕开兜底去暴露这类被吞掉的 bug。

### 沙盒命令"没有输出"

沙盒命令限时 15 秒、输出超 64KB 截断。生成的脚本必须以 `[DONE]` 收尾；
没有 `[DONE]` 就是被掐断了，日志里的 `round_end` / `note=` 能看出这一点。
`tests/test_scripts.py` 会对每一条生成的脚本做静态语法检查。

---

## 相关文档

- **[设计文档V2.md](./设计文档V2.md)** — 当前实现的完整设计（含 V1 失败清单与逐条对策）
- [docs/任务书.md](./docs/任务书.md) — 比赛规则
- [docs/接口文档.md](./docs/接口文档.md) — 报文规范
- [战术参考/](./战术参考/) — 聊天记录、布局图、对战分析模板
- [设计文档.md](./设计文档.md) — V1 设计（历史存档）
