# Issue 总结模板 V2（对战分析 → 自动修复）

> 配套：[日志分析模板V2.md](./日志分析模板V2.md)（读日志）
> 生成器：`cd CoreGeek && python tools/analyze_log.py --log debug.log --issue --out issue.md`
>
> 两份模板是**成对**的：分析模板负责「往下读」（从日志挖问题），
> 本模板负责「往上写」（把问题写成自动化系统能吃的 Issue）。
> 字段对应见文末对照表。

---

## 一、自动化系统怎么读 Issue

`automation/issue_monitor.py` 按标签筛选，`automation/dispatcher.py` 提取提示词。

对战分析类 Issue 必须同时带这两个标签：

```
Labels: auto-fix, battle-analysis
```

`dispatcher.extract_claude_prompt` 的提取规则：

- 正文里有 ` ```claude-prompt ` 围栏时，**只取围栏内的内容**作为提示词
- 没有围栏时整个正文当提示词（但会加上「请分析需求」的引导语）
- 提示词里**必须包含文件路径**，否则 Claude 不知道改哪

> ⚠️ V2 之后文件路径变了：`CoreGeek/src/agent/brain.py` 已拆成 10 个模块。
> 见 §四的模块表——写 `brain.py` 会让这一轮自动修复空转。

---

## 二、模板正文

复制下面内容替换 `{}`。**保留 `claude-prompt` 围栏**，那是精确提取的关键。

```markdown
## 对战分析报告 (PK {pkId})

**结果**: teamAName={队名A}; teamBName={队名B}; scoreA={分A}; scoreB={分B}

**我方**: {队名}（teamId={id}，{challenger/defender}）
**日志范围**: 第{起}回合 - 第{止}回合（{n}回合，其中夜战{m}回合）

---

### 1. 战况概览

| 指标 | 我方 | 敌方 |
|-----|-----|-----|
| 最终积分 | {n} | 报文不提供 |
| 基地血量 | {hp}/{full} | 报文不提供 |
| 武器塔 | {n}座（{类型@坐标 L等级}） | 可见{enemy_visible}个单位 |
| 围墙 | {n}段（{等级分布}） | {enemy_walls}段 |
| 任务 | 交卷{n}/领{n} | 报文不提供 |
| 金币 | 峰值{n}，终值{n} | 报文不提供 |
| 机器人击杀 | {n}（战斗分{n}） | 报文不提供 |
| 角色空转 | {n} 人·回合 | 报文不提供 |

<!-- 以上每一格都来自 debug.log：request_decoded / round_end / day_summary -->

### 2. 任务链路

- R{n} 领取任务（point={x},{y}，超时{n}回合）
- R{n} 任务开始 family={api-query/engineering-fix} step=recon
- R{n} 推进步骤 step=query（reason=ok）
- R{n} 提交答案（answer_len={n}）
- R{n} 技能入库（steps={n}）
- R{n} 任务结束（submitted={n}）

<!-- 逐条来自 task_event；这一节是判断"任务为什么没做完"的唯一依据 -->

### 3. 日志原文（分析依据，**必填**）

> 不要删这一节。下面是相关回合的 `debug.log` 原文——**下面每一条结论都必须
> 能在这些原文里指到具体字段**，指不到的结论不要写进 §4。
>
> 取更多原文：
> ```bash
> cd CoreGeek && python tools/analyze_log.py --log debug.log --rounds {起}-{止} --issue
> ```

```log
{粘贴 request_decoded / strategy_done / round_end / task_event 原文}
```

**关键片段摘录**（一个结论一行，方便核对）：

| 结论要点 | 回合 | 日志字段 | 原文片段 |
|---------|-----|---------|---------|
| 例：任务接取后没提交 | R11-R16 | `task_event` | `event=start` 之后没有 `event=submit` |
| {待人工} | {待人工} | {待人工} | `{待人工：从上面拷一行}` |

### 4. 发现的问题

**P1｜{问题标题}**
- **现象**: {回合号 + 具体字段，例如 `R11-R16 round_end idle_weapon=3`}
- **根因**: {模块 + 函数/常量}
- **影响**: {失多少分 / 是否导致失败}

**P2｜{问题标题}**
- ...

### 5. 优化建议

**S1｜{建议标题}**
- 涉及：`CoreGeek/src/agent/{模块}.py`
- 做法：{具体改动，写出常量名/函数名，例如 `STEP_MAX_ATTEMPTS["query"] = 3 改成 2`}
- 预期：{可验证的效果，例如"任务在 2 回合内闭环，`task_event` 里 3 回合内出现 submit"}

**S2｜{建议标题}**
- ...

---

```claude-prompt
根据对战分析报告的以下建议修改代码：

S1: {建议标题}
- 文件：CoreGeek/src/agent/{模块}.py
- 改动：{具体做法}
- 预期：{效果}

S2: {建议标题}
- 文件：CoreGeek/src/agent/{模块}.py
- 改动：{具体做法}

约束：
1. 保持无状态决策设计（每回合从 Turn 重新解析，不引入跨回合战场缓存；
   任务链路的学习成果只放 strategy/task/memory.py）
2. 修改后必须通过 `cd CoreGeek && python -m pytest tests/ -q`
3. 新增行为要补单测，按功能放进对应的测试文件：
   protocol / grid / world / telemetry(日志) / economy / defense /
   sandbox / answer / skills / solver / brain(端到端)
4. 不要改 protocol.py 里已有的报文字段名与动作码
5. 不要改 main3.py / run.sh（判题系统入口）
6. 改日志字段时同步更新 tools/analyze_log.py 与 战术参考/日志分析模板V2.md
```

---

*本 issue 由对战分析生成。§3 的日志原文是全部结论的依据——*
*如果建议与原文冲突，以原文为准。*
```

---

## 三、为什么 §3 必须留原始日志

V1 的自动修复循环里最容易出错的一环是：报告只给"反推的常量名"和结论
（`TOWER_LOADOUT`、`STONE_BATCH`、`TASK_NO_API_MAX_WATCH` …），修代码的人
拿不到原始证据，只能照着描述猜。结果就是：

- 建议指向的常量名在仓库里**根本不存在**（grep 无匹配）；
- 现象其实已经被**几个提交之前**的改动修掉了，照着改等于重做一遍；
- 同一份日志被反复分析成不同的结论。

所以 V2 的 Issue 模板把**原文摘录位**钉死在 §3：

1. **取原文有现成命令**（`--rounds A-B --issue`），不是手工活；
2. 生成了"关键片段摘录"表，**每条结论必须填一行**，填不出就是从原文里
   找不到依据，那这条结论不该进 §4；
3. 附了 `附：日志字段速查`（见下），不熟悉字段的人也能读。

### 3.1 贴原文的长度建议

| 场景 | 贴多少 |
|-----|-------|
| 单个现象（如"任务卡住"） | 现象发生的回合 ±2，连同 `task_event` 全部 |
| 经济/建造类 | 首个异常回合起 20 回合 |
| 夜战类 | 那一夜的全部 `round_end` 行 |
| 整场复盘 | 不要全贴，用 `day_summary` + `--template` 的曲线代替 |

> 判题系统与日志文件都在本地，原文**不需要脱敏**——但别把整份 `debug.log`
> 贴进 Issue（几十万字符），用 `--rounds` 切。

---

## 四、对应模块怎么填

| 建议涉及 | 文件 |
|---------|------|
| 任务：阶梯 / 止损 / 调度 / 提交与放弃 | `CoreGeek/src/agent/strategy/task/solver.py` |
| 任务：沙盒命令（取数、鉴权、分页、聚合、工程修复） | `CoreGeek/src/agent/strategy/task/scripts.py` |
| 任务：答案校验闸门 | `CoreGeek/src/agent/strategy/task/answer.py` |
| 任务：族识别 / 技能签名 / 技能生成 | `CoreGeek/src/agent/strategy/task/skills.py` |
| 任务：跨任务记忆 | `CoreGeek/src/agent/strategy/task/memory.py` |
| 任务：沙盒输出解析 / 指纹 | `CoreGeek/src/agent/strategy/task/sandbox.py` |
| 经济：采集 / 贩卖 / 建造执行 / 金币分配 | `CoreGeek/src/agent/strategy/economy.py` |
| 防御：塔位 / 墙线 / 夜战火力 | `CoreGeek/src/agent/strategy/defense.py` |
| 可建造区学习 / 敌方来向 | `CoreGeek/src/agent/world.py` |
| 寻路 / 可达性 / 锥形判定 | `CoreGeek/src/agent/grid.py` |
| 报文字段 / 指令构造 | `CoreGeek/src/agent/protocol.py` |
| 日志字段 / 逐回合增量 | `CoreGeek/src/agent/brain.py`（`_log_turn`）、`telemetry.py` |
| HTTP / 决策超时 | `CoreGeek/src/agent/server.py` |

**一条建议一个文件。** S1 改 `solver.py`、S2 改 `economy.py` 就分开写——
混在一起 Claude 容易只改一半。

---

## 五、写 Issue 的注意事项

### 5.1 必须带的字段

| 字段 | 为什么必须 |
|-----|-----------|
| `Labels: auto-fix, battle-analysis` | 少了标签系统不会处理 |
| `claude-prompt` 围栏 | 没有围栏时提示词会掺进整篇报告，Claude 容易跑偏 |
| §3 的**日志原文** | 结论的可信度全靠它；V1 反复改错地方就是因为缺这个 |
| §5 的**文件路径** | 没路径 Claude 要全库找，容易改错文件 |
| §4 的**回合号 + 字段** | 让 Claude 能回原文核对，而不是凭空改 |
| §5 的**预期效果** | 自动化跑完测试后靠这个判断改对了没 |

### 5.2 建议这么写

- **常量名写全**：`STEP_MAX_ATTEMPTS["query"] 由 3 改成 2`，不要写"把重试次数调小"
- **带上"别改什么"**：明显的边界（如"不要动 `protocol.py` 的字段名"）能避免顺手重构
- **预期要可验证**：写"`task_event` 里 3 回合内出现 `submit`"，
  而不是"任务完成率提升"
- **一次不超过 3 条**：`automation_config.yaml` 的 `max_issues_per_round` 会截断

### 5.3 不要这么写

- **不要写"优化策略"**这种没有落地点的建议——Claude 只能瞎改
- **不要建议改 `main3.py` / `run.sh`**——判题系统入口，动了可能整个跑不起来
- **不要按报告自报的"疑似常量名"直接改**：那些名字多半是反推出来的
  （V1 的报告里 `TASK_NO_API_MAX_WATCH`、`WALL_LOADOUT` 之类全仓 grep 无匹配）。
  先在对应模块里 `grep` 到真实的名字，再写进建议
- **不要把"日志未覆盖"当成事实**：报文里根本没有的字段（敌方经济、敌方塔型），
  写进建议等于让 Claude 去实现一个不存在的数据源

---

## 六、两个模板的字段对照

| 日志分析模板V2（读） | Issue 模板（写） | 日志来源 |
|--------------------|----------------|---------|
| §1.1 关键数据对比 | §1 战况概览 | `request_decoded` 的 `hp/towers/walls/robots`、`round_end` 的 `kills`、`day_summary` |
| §1.2 时间线 | §2 任务链路 + §4 问题 | `task_event`、`strategy_done` 的 `actions` |
| §六 日志原文 | §3 日志原文 | 直接拷 `debug.log` 行 |
| §二 症状→字段对照 | §4 发现的问题（P1/P2） | `fail=`、`freeze_alert`、`round_end` |
| §四 模块表 | §5 建议的「涉及」 | — |
| §七 优化建议 | §5 建议（S1/S2/S3） | — |
| （分析结论） | `claude-prompt` 围栏 | — |

**优先级映射**：分析模板的 P0/P1 必须进 `claude-prompt` 围栏，P2 可以只留在
正文里。围栏里最多 3 条。

---

## 七、最小示例（填好长什么样）

```markdown
## 对战分析报告 (PK 592173)

**结果**: teamAName=泥头车; teamBName=trailblazer; scoreA=3; scoreB=0
**我方**: trailblazer（teamId=4334，defender）
**日志范围**: 第1回合 - 第130回合（130回合，其中夜战60回合）

### 1. 战况概览

| 指标 | 我方 | 敌方 |
|-----|-----|-----|
| 最终积分 | 80 | 报文不提供 |
| 基地血量 | 1200/1500 | 报文不提供 |
| 武器塔 | 3座（火箭29,8L1 / 电磁29,9L1 / 加特林29,10L1） | 可见1个单位 |
| 围墙 | 2段（L1×2） | 0段 |
| 任务 | 交卷1/领2 | 报文不提供 |
| 金币 | 峰值 476，终值 0 | 报文不提供 |
| 机器人击杀 | 7（战斗分 10） | 报文不提供 |
| 角色空转 | 49 人·回合 | 报文不提供 |

### 2. 任务链路

- R7 领取任务（point=23,14，超时15回合）
- R8 任务开始 family=api-query step=recon
- R9 推进步骤 step=query（reason=ok）
- R10 提交答案（answer_len=98）
- R11 技能入库（steps=2）
- R11 任务结束（submitted=1）

### 3. 日志原文（分析依据）

```log
request_decoded round=7 day=1 tod=day round_in_day=7 … gold=0 gold_delta=-25 …
strategy_done round=7 commands=3 elapsed=0.57ms gold_spent=0 actions=10010:move→(27,10)
              10011:acceptTask 10012:move→(18,13) fail=[] sandbox=空闲 note=accept@23,14
round_end round=7 kills=0(s=0 m=0 l=0 b=0) kill_score=0 … idle_units=0
task_event round=7 event=accept detail=point=23,14 timeout=15 score=80
```

| 结论要点 | 回合 | 日志字段 | 原文片段 |
|---------|-----|---------|---------|
| 任务闭环正常 | R7-R11 | `task_event` | `event=submit` 在 `event=start` 后 2 回合 |

### 4. 发现的问题

**P1｜第二座任务点 30 回合冷却期间开拓者整段空转**
- **现象**: `R12-R39 round_end idle_units=1 idle_ids=10011`，其间 `task=idle plan=cooling`
- **根因**: `strategy/task/solver.py` 的 `_schedule` 在冷却期直接返回 `IDLE`
- **影响**: 每天约 30 人·回合浪费（不影响得分，但影响夜间操控人数）

### 5. 优化建议

**S1｜冷却期让开拓者回防而不是原地等**
- 涉及：`CoreGeek/src/agent/strategy/task/solver.py`
- 做法：`_schedule` 的 cooling 分支返回 `hold=False`，由 `brain._fill_idle`
  把它拉去武器旁站岗
- 预期：`round_end idle_units` 在冷却期降为 0，夜里 `manned` 由 2 升到 3
```

> 上面这个例子里，**P1 的证据（`idle_ids=10011`）是直接从 §3 的原文里读出来的**，
> 而不是从报告结论反推的——这正是 V2 要的写法。

---

## 八、和历史版本的差异

| 项 | V1（[Issue模板.md](./Issue模板.md)） | V2 |
|----|------|-----|
| 日志原文 | 没有专门的位置，结论靠反推 | §3 固定留位，生成器直接灌好相关回合 |
| 字段出处 | 笼统写"debug.log" | 每条结论标到具体字段（`round_end idle_weapon`） |
| 文件路径 | `brain.py` | 按 §四 的模块表填 |
| 约束条目 | 4 条 | 6 条（加了"日志字段改动要同步复盘工具"） |
| 反推常量名 | 报告里常见且会照改 | §5.3 明确禁止：先 grep 到真名再写 |

> V1 的那份模板保留作历史存档；**新 Issue 一律用本模板**。
