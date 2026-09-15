"""pytest 公共配置与测试数据工厂。

对应设计文档V2 第 10 章。

把 `CoreGeek/src` 加入 `sys.path`，使测试可以直接 `import agent.*`；
并提供一套构造报文的最小工厂——所有用例都用它拼地图，不依赖真实报文文件，
这样每个用例只声明"它在测的那几个单位"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BUILDING_KINDS = ("station", "wall", "gatling", "railgun", "rocket")

# 基地放在地图中间偏右下，四周留够 3 圈空白，方便测塔位/墙位
DEFAULT_BASE = (20, 10)


@pytest.fixture(autouse=True)
def _reset_learned_state():
    """每个用例前后清空**跨回合学习成果**

    `agent.world` 的可建造区、`agent.strategy.task.memory.MEMORY` 的技能库与
    当前任务状态都是模块级单例（它们本来就该活在一局比赛里）。跨用例残留会让
    上一个用例学到的格子/技能影响下一个用例的判断——比如某个用例用一份
    `[ANSWER]` 建了技能，后面同签名的用例就会跳过 recon 直接 query。
    """
    from agent import world
    from agent.strategy.task import memory

    world.reset_learning()
    memory.reset()
    yield
    world.reset_learning()
    memory.reset()


@pytest.fixture
def role_factory():
    """构造 Role JSON 的工厂（接口文档 §1.3.1）"""

    def _make(unit_id: int, kind: str, x: int, y: int, **overrides):
        role = {
            "id": unit_id,
            "pos": {"x": x, "y": y},
            "roleType": kind,
            "health": 100,
            "attackPower": 0,
            "attackRange": 0,
            "backPackCapability": 0,
            "backpack": [],
        }
        if kind in BUILDING_KINDS:
            role["level"] = 1
        if kind == "worker":
            role.update({"health": 220, "backPackCapability": 100})
        elif kind == "pioneer":
            role.update({"health": 200, "backPackCapability": 40})
        elif kind == "station":
            role.update({"health": 1500})
        elif kind in ("gatling", "railgun", "rocket"):
            role.update({"health": 1000, "attackPower": 10, "attackRange": 6})
        elif kind == "wall":
            role.update({"health": 1000})
        role.update(overrides)
        return role

    return _make


@pytest.fixture
def robot_factory():
    """构造机器人 JSON 的工厂（接口文档 §1.5.1）"""

    def _make(robot_id: int, x: int, y: int, kind: str = "smallRobot", **overrides):
        robot = {
            "id": robot_id,
            "pos": {"x": x, "y": y},
            "roleType": kind,
            "health": {"smallRobot": 40, "middleRobot": 60,
                       "largeRobot": 500, "bossRobot": 800}.get(kind, 40),
            "abnormalState": "",
            "targetTeam": "challenger",
        }
        robot.update(overrides)
        return robot

    return _make


@pytest.fixture
def task_factory():
    """构造任务点 JSON 的工厂（接口文档 §1.3.2）"""

    def _make(
        task_type: str,
        x: int,
        y: int,
        *,
        cooldown: int = 0,
        score: int = 80,
        gold: int = 80,
        valid: bool = True,
        timeout: int = 15,
    ):
        return {
            "taskType": task_type,
            "taskPosition": {"x": x, "y": y},
            "coldDownRounds": cooldown,
            "scoreReward": score,
            "goldReward": gold,
            "isValid": valid,
            "timeoutRounds": timeout,
        }

    return _make


@pytest.fixture
def payload_factory():
    """构造完整请求报文（接口文档 §1.1）"""

    def _make(
        *,
        round_no: int = 1,
        team: str = "challenger",
        gold: int = 75,
        roles: list | None = None,
        zones: list | None = None,
        player_tasks: list | None = None,
        robots: list | None = None,
        enemies: list | None = None,
        phase_task: str = "",
        last_cmd_result: str = "",
        last_action_results: dict | None = None,
        llm_resp: str = "",
        errors: list | None = None,
        weapon_shop: list | None = None,
        vendor_shop: list | None = None,
        width: int = 41,
        height: int = 32,
    ):
        return {
            "roundNo": round_no,
            "mapInfo": {
                "width": width,
                "height": height,
                "zones": zones if zones is not None else [],
            },
            "teamOur": {
                "type": team,
                "teamId": "1",
                "teamName": "T",
                "goldNum": gold,
                "totalScore": 0,
                "playerTasks": player_tasks if player_tasks is not None else [],
                "roles": roles if roles is not None else [],
            },
            "teamEnemy": {"roles": enemies if enemies is not None else []},
            "robot": {"roles": robots if robots is not None else []},
            "phaseTask": phase_task,
            "lastRoundRoleActionResults": last_action_results or {},
            "lastSummonTreasureResult": 0,
            "llmResp": llm_resp,
            "worldNews": {"officialNews": "", "folkLegends": ""},
            "lastCmdResult": last_cmd_result,
            "vendorShopList": vendor_shop
            if vendor_shop is not None
            else [{"name": "stone", "price": 1}, {"name": "iron", "price": 3},
                  {"name": "copper", "price": 5}],
            "weaponShopList": weapon_shop if weapon_shop is not None else [],
            "errors": errors if errors is not None else [],
        }

    return _make


@pytest.fixture
def base_roles(role_factory):
    """一套标准的我方单位：基地 + 两个工人 + 一个开拓者（任务书 §4.5.3）"""
    bx, by = DEFAULT_BASE
    return [
        role_factory(10013, "station", bx, by),
        role_factory(10010, "worker", 5, 23),
        role_factory(10011, "pioneer", 25, 3),
        role_factory(10012, "worker", 10, 16),
    ]


@pytest.fixture
def zone_factory():
    """构造 mapInfo.zones 条目的工厂（接口文档 §1.2.1）"""

    def _make(neutral_type: str, x: int, y: int):
        return {"neutralType": neutral_type, "pos": {"x": x, "y": y}}

    return _make


@pytest.fixture
def build_turn(payload_factory, role_factory):
    """把 payload 直接变成 `Turn`（需要强类型快照的用例用）"""
    from agent.protocol import Turn

    def _make(**kwargs):
        return Turn.load(payload_factory(**kwargs))

    return _make


# ==========================================================================
# 真实对战日志片段（用于回归用例）
# ==========================================================================

# PK592172 / PK592173 的 R11–R13 沙盒输出：读题成功、`docs=2`、`key=yes`，
# 但 `api=0` —— 执行器从头到尾没发出过一条接口请求。这是 V1 最致命的故障
# （任务接取后 6–9 个回合从未提交，任务分 0）。
V1_DEAD_LOOP_OUTPUT = """\
[exitCode:0]
[TASK]请阅读task_1_beijing.md
[SCAN] files=9 dirs=2 py=yes sh=no api_calls=0 hits=0
[DOC] path=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md
[DONE] step=read elapsed=1.20s
"""

# 同一份任务的**正确**沙盒输出：调通了接口并聚合出答案。
SUCCESS_OUTPUT = """\
[exitCode:0]
[QUERY] root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api has_task=yes chars=4200
[PARSE] target=北京 bases=2 paths=3 params=2 auths=3
[API] url=http://localhost:8899/api/v1/heritage/search?location=%E5%8C%97%E4%BA%AC&limit=100 status=200 auth=yes n=100
[SCAN] api_calls=1 hits=100 tried=200/api/v1/heritage/search
[DATA] n=137 total=137 pages=2
[MAPPING] city= total_count= world_heritage_count=protected_level types=type oldest_era=name
[ANSWER] {"city":"北京","total_count":137,"world_heritage_count":12,"types":["古遗址","古建筑"],"oldest_era":"周口店遗址"}
[PROFILE] base=http://localhost:8899 path=/api/v1/heritage/search param=location auth="Authorization: Bearer heritage-api-key-2024" target=北京 fields=5
[DONE] step=query elapsed=3.10s calls=2 records=137 fields=5
"""

# 401 输出：命令发出去了，但缺鉴权头（PK592173 的 R14）
AUTH_FAIL_OUTPUT = """\
[exitCode:0]
[PARSE] target=北京 bases=1 paths=2 params=2 auths=0
[APIFAIL] url=http://localhost:8899/api/v1/heritage/search?location=%E5%8C%97%E4%BA%AC status=401 reason=missing_auth
[SCAN] api_calls=1 hits=0 tried=401/api/v1/heritage/search
[DATA] n=0 total=0 pages=0
[DONE] step=query elapsed=1.40s calls=1 records=0 fields=0
"""

# 任务原文（实测报文里的 phaseTask 与 lastCmdResult 原文）
BEIJING_TASK_TEXT = """\
请阅读task_1_beijing.md，获取任务信息

# 自进化任务 A-1：查询北京文化遗产

系统提供了一个 API 服务（运行在 `http://localhost:8899`），API 文档在
`API_DOCS.md` 中。注意：由于该系统经过了长期迭代，文档中的部分字段内容已经
发生变化，描述不再准确（其它内容可以认为是准确的）。

## 任务要求

从 API 查询北京市的**全部**文化遗产记录，然后通过 `submitAnswer` 接口提交
以下统计信息：

```json
{
  "city": "北京",
  "total_count": <总记录条数>,
  "world_heritage_count": <保护级别为"世界遗产"的数量>,
  "types": ["<所有不重复的遗产类型，顺序不限>"],
  "oldest_era": "<年代最早的遗产名称>"
}
```

## 提交形式

```json
{"city":"北京","total_count":0,"world_heritage_count":0,"types":["a","b"],"oldest_era":"c"}
```

- 提交答案为数字/字符串敏感型，不能将数字0写成"0"，否则算错
"""
