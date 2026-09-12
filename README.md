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

**白天（70回合）**:

| 角色 | 优先级 | 行为 |
|-----|-------|------|
| 工人 | 1 | 金币≥25时建造3座武器塔（加特林、电磁狙击炮、火箭发射台） |
| 工人 | 2 | 采集石头（每批6个），为围墙备料 |
| 工人 | 3 | 建造围墙防御圈（留右下角入口） |
| 工人 | 4 | 围墙建完后，把多余石头（≥10）卖给小贩换金币 |
| 开拓者 | 1 | 维持进行中的任务（留在任务点旁，离开会强制结束任务） |
| 开拓者 | 2 | 前往任务点领取自进化类任务（acceptTask） |
| 开拓者 | 3 | 无任务时跟随武器塔，为夜晚操控做准备 |

**夜晚（60回合）**:
1. 角色移动到武器周围1格内
2. 操控武器攻击最近的机器人（跳过冷却中的武器）
3. 机器人优先，其次攻击视野内的敌方单位

**LLM 策略咨询**: 每天第一个回合向 `prompt` 字段提交一次局面咨询
（每个游戏日LLM调用有配额，故每天只请求一次）。
可用环境变量 `LLM_PROMPT=0` 关闭。

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
- 调用本地Claude Code进行代码修改
- 自动创建PR并推送到远程仓库

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
- [x] 单元测试（`tests/`，60个用例）
- [ ] 购买/使用升级券
- [ ] 宝藏召唤（summonTreasure）
- [ ] 提交任务答案（submitAnswer，需要沙盒配合）

### Phase 5: 自动化系统 ✅
- [x] 部署GitHub自动化
- [x] 通过Issue快速迭代

> 详细改进项与验收标准见 **[改进清单.md](./改进清单.md)**。

---

## 📖 关键概念

### 游戏规则
- 地图: 41×32矩形区域
- 回合制: 最多1300回合（10天）
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

## 📞 支持

- 远程仓库: https://github.com/ScreaMA/Competition10.git
- 详细设计: 查看[设计文档.md](./设计文档.md)
- 参考实现: 查看`Demo/CoreGeek/`目录

---

**文档版本**: v1.1  
**更新日期**: 2026-09-12
