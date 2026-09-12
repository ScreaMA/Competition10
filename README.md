# 《未来战争》编程大赛客户端设计

## 📚 文档导航

- **[设计文档.md](./设计文档.md)** - 完整的技术设计文档（推荐阅读）
- **[docs/任务书.md](./docs/任务书.md)** - 比赛任务说明
- **[docs/接口文档.md](./docs/接口文档.md)** - API接口规范
- **[Demo/CoreGeek/](./Demo/CoreGeek/)** - 参考实现代码

---

## 🎯 快速开始

### 项目结构

```
CoreGeek/
├── main3.py                 # 程序入口
├── pyproject.toml           # 项目配置
├── debug.log               # 完整请求响应日志
└── src/agent/
    ├── server.py           # HTTP服务器
    ├── protocol.py         # 数据结构和协议
    ├── grid.py            # A*路径规划
    └── brain.py           # 决策引擎
```

### 运行客户端

```bash
cd CoreGeek
python main3.py 8000
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
1. 工人建造3座武器塔（加特林、电磁炮、火箭）
2. 采集石头资源
3. 建造围墙防御体系

**夜晚（60回合）**:
1. 角色移动到武器周围
2. 操控武器攻击最近的机器人

---

## 🔧 实现要点

### 1. 日志系统

**双层日志设计**:
- **stdout**: INFO级别，显示回合号和指令数量
- **debug.log**: DEBUG级别，记录完整的请求和响应JSON

```python
# main3.py中配置
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# 同时配置文件日志
file_handler = logging.FileHandler("debug.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
```

**日志输出示例**:

stdout:
```
2026-09-12 14:30:05,123 | INFO | round 1 -> 3 commands
```

debug.log:
```
2026-09-12 14:30:05,124 | DEBUG | REQUEST round 1:
{
  "roundNo": 1,
  "mapInfo": {...},
  ...
}
2026-09-12 14:30:05,200 | DEBUG | RESPONSE round 1:
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
blocked = turn.blocked()  # 实时查询阻挡物
next_step = find_path(start, goal, blocked)
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
pip install pyyaml requests

# 配置Token
echo "ghp_your_token" > github_token.txt

# 运行
python automation_main.py
```

详细设计见**设计文档.md第11章**。

---

## 📋 实现建议

### Phase 1: 基础框架（1-2天）
- [ ] 实现HTTP服务器
- [ ] 实现协议解析
- [ ] 测试能否正确接收游戏状态

### Phase 2: 基本策略（2-3天）
- [ ] 实现A*路径规划
- [ ] 实现移动指令
- [ ] 实现采集指令

### Phase 3: 完整策略（3-5天）
- [ ] 实现武器建造逻辑
- [ ] 实现围墙建造逻辑
- [ ] 实现夜晚战斗逻辑

### Phase 4: 优化迭代（持续）
- [ ] 调整策略参数
- [ ] 添加任务系统
- [ ] 性能优化

### Phase 5: 自动化系统（可选）
- [ ] 部署GitHub自动化
- [ ] 通过Issue快速迭代

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
2. **性能要求**: 每回合决策时间<1秒
3. **日志完整**: debug.log必须记录完整请求响应
4. **时间戳打包**: 使用package.sh生成带时间戳的tar.gz
5. **扩展性**: 每回合重新解析状态，无缓存依赖

---

## 📞 支持

- 远程仓库: https://github.com/ScreaMA/Competition10.git
- 详细设计: 查看[设计文档.md](./设计文档.md)
- 参考实现: 查看`Demo/CoreGeek/`目录

---

**文档版本**: v1.1  
**更新日期**: 2026-09-12
