# 《未来战争》编程大赛客户端设计

> **代码审查完成**: ✅ 代码质量评分 9.1/10，符合度98%，可以直接使用

## 📚 文档导航

### 核心文档
- **[设计文档.md](./设计文档.md)** - 完整的技术设计文档（推荐阅读）
- **[快速参考.md](./快速参考.md)** - 一页纸快速参考卡 ⭐ 推荐打印

### 审查文档
- **[审查总结.md](./审查总结.md)** - 代码审查总结报告
- **[代码审查报告.md](./代码审查报告.md)** - 详细的模块审查结果
- **[改进清单.md](./改进清单.md)** - 改进任务和实施计划

### 比赛文档
- **[docs/任务书.md](./docs/任务书.md)** - 比赛任务说明
- **[docs/接口文档.md](./docs/接口文档.md)** - API接口规范
- **[Demo/CoreGeek/](./Demo/CoreGeek/)** - 参考实现代码

---

## 🎯 快速开始

### 项目结构

```
CoreGeek/
├── main3.py                 # 程序入口（判题系统入口文件）
├── run.sh                   # 启动脚本（判题系统调用: bash run.sh <port>）
├── pyproject.toml           # 项目配置
├── debug.log               # 完整请求响应日志（运行时生成）
├── src/agent/
│   ├── server.py           # HTTP服务器
│   ├── protocol.py         # 数据结构和协议
│   ├── grid.py            # A*路径规划
│   └── brain.py           # 决策引擎
├── tests/                  # 单元测试（pytest）
└── tools/                  # 自检与调试脚本
```

### 运行客户端

```bash
cd CoreGeek
bash run.sh 8000
```

**注意**: 判题系统使用 `bash run.sh <port>` 启动，入口为 `main3.py`；
客户端监听 `0.0.0.0:<port>`，接收判题系统的 `POST /` 请求（任意路径均可）。

### 测试与自检

```bash
cd CoreGeek

# 单元测试（60个用例，覆盖协议/寻路/决策）
python -m pytest tests/ -v

# 端到端自检：真实报文 + 边界场景 + 1300回合性能
python tools/local_check.py 8000

# 一键联调（启动客户端→发请求→打印响应）
bash tools/test_client.sh 8000

# 复盘：统计每回合决策耗时与指令数
python tools/analyze_log.py
```

### 打包发布（带时间戳）

```bash
# Linux/Mac
chmod +x package.sh
./package.sh
# 输出: CoreGeek_20260912_143052.tar.gz

# Windows
package.bat
# 输出: CoreGeek_20260912_143052.tar.gz
```

---

## ⚙️ 核心设计

### 模块职责

| 模块 | 文件 | 职责 |
|-----|------|------|
| 入口 | `main3.py` | 解析参数,启动HTTP服务器 |
| 服务器 | `agent/server.py` | 接收POST请求,调用决策 |
| 协议 | `agent/protocol.py` | 数据结构定义,指令构建 |
| 地图 | `agent/grid.py` | A*路径规划算法 |
| 决策 | `agent/brain.py` | 白天建造,夜晚战斗策略 |

### 核心策略

**白天（70回合）** - v1.1优化后：

| 角色 | 优先级 | 行为 |
|-----|-------|------|
| 工人 | 1 | 金币≥25时建造3座武器塔（加特林、电磁狙击炮、火箭发射台），**三塔分散在基地四方位（上/左/下/右）** |
| 工人 | 2 | 采集石头（**批次从6降到3**），**边采边建围墙**，加快防御体系成型 |
| 工人 | 3 | 建造围墙防御圈（留右下角入口），**第5-6回合就能立起第一段** |
| 工人 | 4 | **围墙建完后，金币≥100时买升级券升级武器（level1→level2）** |
| 工人 | 5 | 继续采石并把多余石头（≥10）卖给小贩换金币，**形成经济循环**；没有石矿时退而采铁/铜，背包满了先卖矿腾地方 |
| 工人 | — | **天黑前5回合停止建造与采集，回防到武器旁待命**（火力不足或一段围墙都没有时先抢建） |
| 开拓者 | 1 | 维持进行中的任务（留在任务点旁，离开会强制结束任务） |
| 开拓者 | 2 | **提交任务答案（submitAnswer，答案取自沙盒 `executeCmd` 输出）** |
| 开拓者 | 3 | 前往任务点领取自进化类任务（acceptTask，**每个任务80分+80金币**），**优先快过期的任务** |
| 开拓者 | 4 | 无任务时跟随武器塔，为夜晚操控做准备（任务快结束冷却时提前到任务点旁待命） |

**夜晚（60回合）** - v1.1优化后：
1. **角色按距离就近贪心配对到武器周围1格内**（不再使用固定zip顺序）
2. 操控武器攻击射程内的机器人（跳过冷却中的武器）
3. **机器人按威胁分级（BOSS/大型 > 中型 > 小型），同级取最近**；无机器人时攻击视野内最近的敌方单位

**自进化任务** - v1.1新增完整闭环: 任务期间通过响应中的 `executeCmd` 在沙盒中读取任务文件，
下一回合从请求的 `lastCmdResult` 中解析答案并通过 `submitAnswer` 提交（命令带任务标识，避免复用上一个任务的结果）。

**LLM 策略咨询**: 每天第一个回合向 `prompt` 字段提交一次局面咨询
（每个游戏日LLM调用有配额，故每天只请求一次）。
可用环境变量 `LLM_PROMPT=0` 关闭。

**v1.1关键优化**（根据对战分析issue #8）:
- ✅ 围墙建造提前到第5-6回合（STONE_BATCH: 6→3）
- ✅ 任务完成闭环（submitAnswer）：每场可得160分+160金币
- ✅ 武器塔按四方位分散，避免全部挤在同侧
- ✅ 黄昏回防机制，保证首夜满火力
- ✅ 经济循环（卖石头→买升级券→升级武器）
- ✅ 攻击目标按威胁分级（BOSS优先）

**v1.2关键优化**（根据对战分析issue #10）:
- ✅ 消灭空转：工人无建造目标时按 石→铁→铜 就近采集，背包满了先卖矿换金币，不再有整回合无指令
- ✅ 入夜前火力/防线预算检查（塔<2 或 0 围墙时先抢建，再回防）
- ✅ 角色不再站到武器塔/围墙的建造点上（占住建造位会让建筑永远建不起来）
- ✅ 任务临期优先 + 冷却快结束时提前到任务点旁待命
- ✅ 沙盒输出是"文件读不到"这类错误时不提交，避免白费任务冷却


---

## 🔧 实现要点

### 1. 日志系统

**双层日志设计**:
- **stdout**: INFO级别，显示回合号、指令数量与决策耗时
- **debug.log**: DEBUG级别，记录完整的请求和响应JSON

```python
# main3.py中配置：stdout 与 debug.log 各自独立设置级别
log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("debug.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler, file_handler])
```

**日志输出示例**:

stdout（单行，便于机器解析；超长请求体会被截断到5000字符）:
```
2026-09-13 00:21:49,996 | INFO | agent.server | request_raw id=1 path=/ bytes=5307 body={"roundNo": 1, ...}
2026-09-13 00:21:49,997 | INFO | agent.server | request_decoded id=1 round=1 team=6324 team_type=challenger roles=9
2026-09-13 00:21:49,999 | INFO | agent.server | strategy_done id=1 round=1 commands=3 elapsed=2.22ms
2026-09-13 00:21:49,999 | INFO | agent.server | response_raw id=1 body={"roleCommandMap": {...}}
2026-09-13 00:21:49,999 | INFO | agent.server | response_sent id=1 status=200 bytes=725
```

debug.log（格式化，便于人工阅读）:
```
2026-09-13 00:21:49,996 | DEBUG | agent.server | REQUEST round 1:
{
  "roundNo": 1,
  "mapInfo": {...},
  ...
}
2026-09-13 00:21:49,999 | DEBUG | agent.server | RESPONSE round 1:
{
  "roleCommandMap": {...}
}
```

### 2. 打包脚本

**package.sh**:
```bash
#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PACKAGE_NAME="CoreGeek_${TIMESTAMP}.tar.gz"

cd CoreGeek
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -f debug.log

cd ..
tar -czf "$PACKAGE_NAME" CoreGeek/
echo "Package created: $PACKAGE_NAME"
```

### 3. 扩展性设计

**无状态决策**: 每回合从`Turn`对象重新解析地图状态
```python
def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    turn = Turn.load(payload)  # 重新加载最新状态
    # 决策不依赖上一回合缓存
```

**动态资源发现**: 矿区每10次采集会消失并随机刷新
```python
def stone_mines(self) -> tuple[Pos, ...]:
    """从zones实时查询石矿位置"""
    return self.get_mines(STONE_MINE)
```

**路径重规划**: A*算法每次基于当前地图状态
```python
blocked = turn.blocked(unit)  # 实时查询阻挡物（含建筑/单位/机器人/中立元素）
next_step = next_step(turn, unit, goal)
```

---

## 🤖 GitHub自动化系统

### 功能

- 轮询或webhook监听GitHub Issues
- 根据Issue标签过滤需要处理的任务
- **自动处理对战分析报告（battle-analysis标签）**
- 调用本地Claude Code进行代码修改
- 自动创建PR并推送到远程仓库
- **PR合并后自动同步本地代码**（每轮轮询先检查自己创建的PR状态）

### 对战分析自动化（v1.1新增）

系统会自动识别带`battle-analysis`标签的Issue，提取优化建议并应用：

**Issue格式**：
```markdown
Title: 对战分析报告 (PK 580611)
Labels: auto-fix, battle-analysis

## 5. 优化建议
**S1｜围墙提前介入（改动最小、收益最大）**
- 涉及：CoreGeek/src/agent/brain.py
- 做法：STONE_BATCH由6降到2-3
- 预期：第5-6回合立起第一段围墙

**S2｜补全任务完成闭环（直接兑换160分）**
- 涉及：CoreGeek/src/agent/brain.py
- 做法：在phase_task分支中新增submitAnswer调用
```

**自动处理流程**：
1. 解析优化建议（S1/S2/...）
2. 提取"涉及"和"做法"生成Claude提示词
3. 执行代码修改并运行测试
4. 创建PR（如 #9）并关联原Issue

**实际案例**：
- Issue #8：对战0:3失败，发现6个问题（P1-P6）
- 自动应用S1-S3建议
- PR #9：修改brain.py，优化围墙/任务/塔位
- 结果：围墙建造从10+回合提前到6回合，任务系统正常工作

### 使用流程

1. **创建Issue**（带`auto-fix`标签）:
```markdown
Title: 优化武器建造顺序
Labels: auto-fix

请修改brain.py，将武器顺序改为：火箭→电磁炮→加特林
```

2. **自动化系统处理**:
   - 检测Issue
   - 创建分支`auto-fix/issue-123-20260912-143052`
   - 调用Claude修改代码
   - 提交并推送
   - 创建PR

3. **人工审核合并PR**

4. **合并后自动同步**: 下一轮轮询会查询这些PR的状态，一旦发现已合并，
   就把本地主干快进到远端最新，并清理已合并的本地工作分支。

### PR合并后的代码同步

每轮轮询开始时（`poll_interval`，默认60秒）执行，逻辑见
`Automation.sync_repository()` 与 `GitPusher.sync_main()`：

| PR状态 | 动作 |
|--------|------|
| 已合并 (`merged=true`) | `fetch` → 切到主干 → 同步到 `origin/主干` → 清理已合并的本地分支 |
| 已关闭未合并 | 标记完成，不再跟踪，不同步代码 |
| 仍开放 | 什么都不做，下轮再看 |

**同步策略**（`sync_strategy`，本地与远端分叉时生效）：

| 取值 | 行为 |
|------|------|
| `rebase`（默认） | 把本地提交重放到远端之上，保持线性历史 |
| `merge` | 生成一个合并提交 |
| `ff-only` | 只允许快进，分叉时保持现状（最保守） |

> 实际使用中本地几乎总是「又领先又落后」（本地有自己的提交 + 远端有你刚合并的 PR），
> 所以只做快进等于永远不同步，默认才用 `rebase`——等价于 `git pull --rebase`。

**安全约束**：

- 优先快进（`--ff-only`）；需要 rebase/merge 时一旦冲突立即 `--abort` 回滚，
  保证不留下半成品，也绝不用远端覆盖本地提交
- 工作区有未提交改动时直接跳过（同步失败不标记完成，下轮自动重试）
- 清理本地分支用 `git branch -d`（未合并的分支会拒绝删除）
- 远端分支默认保留，需要删除时把 `delete_remote_branch_after_merge` 设为 `true`
- 不想要这个行为就把 `auto_sync` 设为 `false`

### 部署

```bash
cd automation/
pip install -r requirements.txt

# 配置Token（默认读取项目根目录的 githubtoken.txt，也可用环境变量 GITHUB_TOKEN）
echo "ghp_your_token" > ../githubtoken.txt

# 运行（--once 只处理一轮）
python automation_main.py
```

详细设计见**设计文档.md第11章**。

---

## 📋 实现进度

### Phase 1: 基础框架 ✅
- [x] 实现HTTP服务器（任意路径的POST均可接收，判题系统实际发往 `POST /`）
- [x] 实现协议解析
- [x] 测试能否正确接收游戏状态

### Phase 2: 基本策略 ✅
- [x] 实现A*路径规划
- [x] 实现移动指令
- [x] 实现采集指令

### Phase 3: 完整策略 ✅
- [x] 实现武器建造逻辑
- [x] 实现围墙建造逻辑
- [x] 实现夜晚战斗逻辑

### Phase 4: 优化迭代 🔄
- [x] 开拓者任务系统（领取并维持任务）
- [x] 资源交易（围墙建完后卖石头换金币）
- [x] LLM策略咨询（每天一次）
- [x] 日志体积控制、围墙边界检查
- [x] 单元测试（`tests/`，69个用例）
- [x] 提交任务答案（submitAnswer，沙盒 `executeCmd` + `lastCmdResult`）
- [x] 购买/使用武器升级券
- [ ] 宝藏召唤（summonTreasure）

### Phase 5: 自动化系统 ✅
- [x] 部署GitHub自动化
- [x] 通过Issue快速迭代

> 详细改进项与验收标准见 **[改进清单.md](./改进清单.md)**。

---

## 📖 关键概念

### 游戏规则
- 地图: **41×32** 矩形区域
- 回合制: 最多 **1300 回合**（10天）
- 昼夜循环: 白天70回合，夜晚60回合
- 胜负判定: 按积分决定

### 核心机制
- **白天**: 采集资源、建造防御、完成任务
- **夜晚**: 防御机器人进攻、攻击敌方单位
- **资源**: 石头/铁/铜矿，金币
- **单位**: 工人、开拓者、基地、武器、围墙

### 动作指令
- `move`: 移动
- `collect`: 采集
- `build`: 建造
- `attack`: 攻击
- `buy/sell`: 交易
- `acceptTask/submitAnswer`: 任务系统

---

## ⚠️ 注意事项

1. **稳定性优先**: 异常处理必须完善，程序不能崩溃
2. **性能要求**: 每回合决策时间<1秒（实测平均约1.5ms）
3. **日志完整**: debug.log必须记录完整请求响应（单条INFO日志上限5000字符）
4. **时间戳打包**: 使用package.sh生成带时间戳的tar.gz
5. **扩展性**: 每回合重新解析状态，无缓存依赖
6. **接口路径**: 判题系统发往 `POST /`，服务端**不按路径路由**（`/` 与 `/action` 均可）
7. **启动方式**: 判题系统使用 `bash run.sh <port>`，入口为 `main3.py`
8. **提交前自检**: `python -m pytest tests/ -q` 与 `python tools/local_check.py` 均需通过

---

## ❓ 常见问题

### 本地 push 弹出 GitHub 账号选择框

**原因**: Git for Windows 在系统级 gitconfig 里配了 `credential.helper=manager`（GCM），
它排在 `store` 之前；git 会按「系统 → 全局 → 本地」顺序把 helper 追加成链，
GitHub 无可用凭据时 GCM 就弹账号选择框。URL 级配置只是追加、不会替换，所以必须在
本地作用域用**空值清空整条链**再指定 `store`：

```bash
cd <仓库根目录>            # 即 Competition10/
git config --local credential.helper ""
git config --local --add credential.helper store
```

凭据本身取自 `~/.git-credentials`（内容形如
`https://x-access-token:<PAT>@github.com`）。验证方式（应只看到 `credential-store`）：

```bash
GIT_TRACE=1 git push --dry-run origin main 2>&1 | grep credential
```

> 自动化系统自身的推送不受影响：它在推送 URL 里直接携带 token，不经过 helper。

### 自动化处理Issue后“没改代码”

**排查顺序**:

1. 看 `automation/automation.log` 里 `executing claude for issue #N` 那行的
   `prompt=X chars/Y lines`。如果 X 很小、Y=1，说明提示词在传给 Claude 前就被截断了。
2. 看紧随其后的 `claude summary:`：这条会把 Claude 的原话记下来，能直接看出它是
   “没看到需求”还是“看过代码后判断无需修改”。

提示词是**多行文本**（标题、正文、要求各占一行），早期实现把它作为命令行参数传给
`claude`，在 Windows 上会被截断在第一行：`shutil.which("claude")` 解析到的是 npm 生成的
`claude.cmd` 垫片，参数经 `cmd.exe` 解析时换行等同于命令结束，Issue 的标题与正文整段丢失。
现在改为在 `-p` 模式下把提示词从 **stdin** 送入（`automation/executor.py` 的
`_build_invocation`），既不受换行影响，也绕开了 Windows 约 32K 的命令行长度上限。

验证方式（改动提示词传递后）：

```bash
cd <仓库根目录>
python automation/automation_main.py --once   # 处理完一轮即退出
```

日志中若出现 `prompt=... chars/<N> lines`（N>1）且 `claude summary:` 里引用了 Issue 正文，
即说明提示词已完整送达。

---

## 📞 支持

- 远程仓库: https://github.com/ScreaMA/Competition10.git
- 详细设计: 查看[设计文档.md](./设计文档.md)
- 参考实现: 查看`Demo/CoreGeek/`目录

---

**文档版本**: v1.2  
**更新日期**: 2026-09-13
