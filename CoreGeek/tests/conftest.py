"""pytest 公共配置与测试数据工厂。

把 CoreGeek/src 加入 sys.path，使测试可以直接 import agent.*。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 建筑类型（接口文档规定只有建筑带 level 字段）
BUILDING_KINDS = ("station", "wall", "gatling", "railgun", "rocket")


def _build_task(task, task_factory):
    """任务点支持三种写法：dict / (x, y) / (x, y, overrides)"""
    if isinstance(task, dict):
        return task
    if len(task) == 3:
        return task_factory(task[0], task[1], **task[2])
    return task_factory(*task)


@pytest.fixture(autouse=True)
def _clear_task_answer_cache():
    """每个用例前后清空自进化任务的答案缓存与任务看门狗

    `agent.brain._TASK_ANSWER_CACHE` 与 `agent.brain._TASK_WATCH` 都是模块级
    状态（沙盒里读回来的任务答案、上一回合的任务观察值），跨用例残留会让
    上一个用例的答案/观察值影响到下一个用例。
    """
    from agent import brain

    brain._TASK_ANSWER_CACHE.clear()
    brain._TASK_WATCH = None
    yield
    brain._TASK_ANSWER_CACHE.clear()
    brain._TASK_WATCH = None


@pytest.fixture
def role_factory():
    """构造 Role（角色/建筑）JSON 的工厂"""

    def _make(unit_id: int, kind: str, x: int, y: int, **overrides):
        role = {
            "id": unit_id,
            "pos": {"x": x, "y": y},
            "roleType": kind,
            "health": 1000,
            "attackPower": 0,
            "attackRange": 0,
            "backPackCapability": 0,
            "backpack": [],
        }
        if kind in BUILDING_KINDS:
            role["level"] = 1
        role.update(overrides)
        return role

    return _make


@pytest.fixture
def robot_factory():
    """构造机器人 JSON 的工厂"""

    def _make(robot_id: int, x: int, y: int, **overrides):
        robot = {
            "id": robot_id,
            "pos": {"x": x, "y": y},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }
        robot.update(overrides)
        return robot

    return _make


@pytest.fixture
def task_factory():
    """构造任务点 JSON 的工厂"""

    def _make(x: int, y: int, **overrides):
        task = {
            "taskType": "自进化类1",
            "taskPosition": {"x": x, "y": y},
            "coldDownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "isValid": True,
            "timeoutRounds": 15,
        }
        task.update(overrides)
        return task

    return _make


@pytest.fixture
def payload_factory(role_factory, task_factory):
    """构造判题系统请求 payload 的工厂

    默认给一个位于 (10, 24) 的挑战者基地，可自由追加角色、矿区、
    任务点、机器人等元素。
    """

    def _build(
        round_no: int = 1,
        gold: int = 75,
        station=(10, 24),
        roles=(),
        zones=(),
        tasks=(),
        robots=(),
        enemies=(),
        team_type: str = "challenger",
        phase_task: str = "",
        width: int = 41,
        height: int = 32,
    ) -> dict:
        all_roles = []
        if station is not None:
            all_roles.append(
                role_factory(10013, "station", station[0], station[1], health=1500)
            )
        all_roles.extend(roles)

        return {
            "roundNo": round_no,
            "mapInfo": {
                "width": width,
                "height": height,
                "zones": [
                    {"neutralType": kind, "pos": {"x": x, "y": y}}
                    for kind, x, y in zones
                ],
            },
            "teamOur": {
                "type": team_type,
                "teamId": "6324",
                "teamName": "Challenger",
                "goldNum": gold,
                "totalScore": 0,
                "playerTasks": [
                    _build_task(task, task_factory) for task in tasks
                ],
                "roles": all_roles,
            },
            "teamEnemy": {"roles": list(enemies)},
            "robot": {"roles": list(robots)},
            "phaseTask": phase_task,
            "lastRoundRoleActionResults": {},
            "lastSummonTreasureResult": 0,
            "llmResp": "",
            "worldNews": {"officialNews": "", "folkLegends": ""},
            "lastCmdResult": "",
            "vendorShopList": [{"name": "stone", "price": 1}],
            "weaponShopList": [{"name": "WeaponUpgradeVoucher1", "price": 100}],
            "errors": [],
        }

    return _build
