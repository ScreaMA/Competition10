"""集成用例：`decide(payload)` 的端到端行为与不变量。

对应设计文档V2 §3.2 与第 10 章。

这是唯一一个把协议层、世界模型、经济、防御、任务链路全部串起来的测试文件。
它守住的是**接口契约**（接口文档 §2）与**异常预算**（任务书 §8）这两件
一旦破坏就整场报废的事。
"""

from __future__ import annotations

import json

import pytest

from agent import protocol
from agent.brain import decide
from agent.protocol import Turn
from agent.world import World

from conftest import BEIJING_TASK_TEXT, SUCCESS_OUTPUT

VALID_ACTIONS = {
    "move", "attack", "sell", "buy", "build", "remove",
    "acceptTask", "submitAnswer", "summonTreasure", "use", "drop", "collect",
}

# 每种动作必需/禁用的字段（接口文档 §2.2 + §2.3）
NEEDS_TARGET = {
    "move", "collect", "remove", "build", "attack", "summonTreasure", "submitAnswer",
}
FORBIDDEN_FIELD = {"action", "controllerId", "targetPos", "name", "num",
                   "taskAnswer", "item"}


def _assert_valid_response(payload, response):
    """响应必须满足接口文档 §2 的全部硬约束"""
    assert set(response) == {"roleCommandMap", "prompt", "executeCmd"}
    assert isinstance(response["roleCommandMap"], dict)
    assert isinstance(response["prompt"], str)
    assert isinstance(response["executeCmd"], str)

    turn = Turn.load(payload)
    alive = {unit.unit_id: unit for unit in turn.ours if unit.is_alive}

    for key, command in response["roleCommandMap"].items():
        assert isinstance(key, str), "roleCommandMap 的 key 必须是字符串"
        unit_id = int(key)
        assert unit_id in alive, f"给不存在/阵亡的单位 {unit_id} 发了指令"
        unit = alive[unit_id]

        action = command.get("action")
        assert action in VALID_ACTIONS, f"非法动作码 {action!r}"
        assert set(command) <= FORBIDDEN_FIELD, "出现了接口文档没有的字段"

        # 值域：坐标必须在图内
        for spot in command.get("targetPos") or []:
            assert 0 <= spot["x"] < turn.width
            assert 0 <= spot["y"] < turn.height

        # 动作与角色的匹配（接口文档 §2.3 的"可用角色"）
        if action in ("build", "collect", "remove"):
            assert unit.kind == "worker", f"{unit.kind} 不能执行 {action}"
        if action == "attack":
            assert command.get("controllerId"), "attack 必须有 controllerId"
            assert turn.is_day is False, "白天不能攻击（任务书 §4.4）"
        if action in ("acceptTask", "submitAnswer", "summonTreasure"):
            assert unit.kind == "pioneer", f"{unit.kind} 不能执行 {action}"
        if action == "use":
            name = command.get("name")
            if name in protocol.USE_NEEDS_POS:
                assert command.get("targetPos"), f"use {name} 必须带 targetPos"
        if action in NEEDS_TARGET and action != "submitAnswer":
            assert command.get("targetPos") or action == "attack"

    if response["executeCmd"]:
        assert turn.phase_task, "没有任务在身时不能下发 executeCmd（接口文档 §2.1）"


# ==========================================================================
# 基础契约
# ==========================================================================


def test_response_contract_on_sample_payload(payload_factory, base_roles):
    payload = payload_factory(round_no=85, roles=base_roles)
    response = decide(payload)
    _assert_valid_response(payload, response)


def test_empty_payload_returns_empty_response():
    """报文完全不合法时也不能崩（返回空指令不是异常）"""
    response = decide({})
    assert response == protocol.EMPTY_RESPONSE


def test_exception_is_swallowed(payload_factory, monkeypatch):
    """决策内部抛异常时退化成空指令，而不是让服务挂掉"""
    from agent import brain

    monkeypatch.setattr(brain, "_decide", lambda payload: 1 / 0)
    assert brain.decide({}) == protocol.EMPTY_RESPONSE


def test_decide_internals_never_raise(payload_factory, role_factory, task_factory):
    """直接调不带兜底的 `_decide`

    `brain.decide` 的 `try/except` 是给判题器的保险，但它同时会把
    `NameError` 这类真 bug 吞掉——表现只是"这一回合什么都没做"，
    在对战日志里几乎查不出来。所以这里绕开兜底直接调内部实现：
    任何异常都说明有 bug 被吞掉了。
    """
    from agent.brain import _decide

    sparse_task = task_factory("自进化类1", 14, 14)
    sparse_task.pop("timeoutRounds")
    variants = [
        {},                                                        # 全空
        {"roundNo": 1},                                            # 只有回合号
        payload_factory(round_no=1),                               # 无单位
        payload_factory(round_no=1, roles=[role_factory(10013, "station", 20, 10)]),
        payload_factory(round_no=1, player_tasks=[sparse_task]),   # 任务点缺字段
        payload_factory(round_no=1, last_action_results={"10010": False}),
        payload_factory(round_no=1, phase_task="phase", last_cmd_result="[TIMEOUT]\n"),
        payload_factory(round_no=1, errors=[{"errorCode": 9, "description": "?"}]),
        payload_factory(round_no=1, roles=[role_factory(10013, "station", 0, 0)]),
        payload_factory(round_no=1, roles=[role_factory(10013, "station", 40, 31)]),
    ]
    for payload in variants:
        _decide(payload)  # 抛异常即失败


def test_at_most_one_command_per_unit(payload_factory, base_roles):
    """一个角色一回合只能有一条指令（接口文档 §2.3 注）"""
    payload = payload_factory(round_no=30, roles=base_roles)
    response = decide(payload)
    keys = list(response["roleCommandMap"])
    assert len(keys) == len(set(keys))


def test_commands_only_for_own_units(payload_factory, base_roles):
    payload = payload_factory(round_no=5, roles=base_roles)
    response = decide(payload)
    own = {str(u.unit_id) for u in Turn.load(payload).ours}
    assert set(response["roleCommandMap"]) <= own


# ==========================================================================
# 开局（E5：R1 就要建塔）
# ==========================================================================


def test_first_round_issues_tower_build(payload_factory, role_factory, zone_factory):
    """R1、金币 75 ⇒ 指令里必须出现 build（敌人 R1 就连建 2 座塔）"""
    from agent.protocol import Pos
    from agent.strategy import defense

    probe = World.load(Turn.load(payload_factory(
        round_no=1,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10010, "worker", 30, 30),
               role_factory(10011, "pioneer", 30, 30),
               role_factory(10012, "worker", 30, 30)],
    )))
    site = defense.tower_sites(probe)[0]

    payload = payload_factory(
        round_no=1,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10010, "worker", site.x - 1, site.y),
               role_factory(10011, "pioneer", 30, 30),
               role_factory(10012, "worker", 30, 30)],
    )
    response = decide(payload)
    actions = [
        command["action"] for command in response["roleCommandMap"].values()
    ]
    assert "build" in actions
    _assert_valid_response(payload, response)


def test_no_command_leaves_unit_idle_but_never_invalid(payload_factory, base_roles):
    payload = payload_factory(round_no=1, gold=0, roles=base_roles)
    response = decide(payload)
    _assert_valid_response(payload, response)


# ==========================================================================
# 金币冻结（E1）
# ==========================================================================


def test_gold_zero_with_ore_triggers_sell(payload_factory, role_factory, zone_factory):
    """金币为 0 且背包有矿石 ⇒ 该回合必须有 sell 指令

    故障 E1：PK592172 的 R8–R16 金币恒 0，石材/铜材攒着不卖。
    """
    payload = payload_factory(
        round_no=40,
        gold=0,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10010, "worker", 20, 15, backpack=["copper", "copper"]),
            role_factory(10012, "worker", 20, 17, backpack=["iron"]),
        ],
        zones=[{"neutralType": "vendor", "pos": {"x": 20, "y": 16}}],
    )
    response = decide(payload)
    actions = [
        command["action"] for command in response["roleCommandMap"].values()
    ]
    assert "sell" in actions, response["roleCommandMap"]
    _assert_valid_response(payload, response)


# ==========================================================================
# 夜晚（第 8 章）
# ==========================================================================


def test_night_issues_attack_when_positioned(
    payload_factory, role_factory, robot_factory
):
    """夜晚、角色已就位、射程内有机器人 ⇒ 必须攻击"""
    payload = payload_factory(
        round_no=75,  # 夜晚
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10010, "worker", 19, 13),   # 站在加特林旁边
            role_factory(10012, "worker", 21, 13),
            role_factory(10020, "gatling", 20, 13, attackRange=5),
        ],
        robots=[robot_factory(30001, 24, 13, "smallRobot")],
    )
    response = decide(payload)
    attacks = [
        command for command in response["roleCommandMap"].values()
        if command["action"] == "attack"
    ]
    assert attacks
    assert attacks[0]["controllerId"] in {"10010", "10012"}
    _assert_valid_response(payload, response)


def test_day_never_attacks(payload_factory, role_factory, robot_factory):
    """白天不能攻击（任务书 §4.4），发了就是指令错误"""
    payload = payload_factory(
        round_no=10,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10010, "worker", 19, 10),
            role_factory(10020, "gatling", 20, 13, attackRange=5),
        ],
        robots=[robot_factory(30001, 24, 13, "smallRobot")],
    )
    response = decide(payload)
    assert all(
        command["action"] != "attack"
        for command in response["roleCommandMap"].values()
    )


# ==========================================================================
# 自进化任务链路（第 6 章）
# ==========================================================================


def _task_payload(payload_factory, role_factory, task_factory, round_no, **kw):
    return payload_factory(
        round_no=round_no,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10011, "pioneer", 24, 13),
        ],
        player_tasks=[task_factory("自进化类1", 24, 12)],
        **kw,
    )


def test_accept_task_when_adjacent(payload_factory, role_factory, task_factory):
    payload = _task_payload(payload_factory, role_factory, task_factory, 5)
    response = decide(payload)
    assert response["roleCommandMap"]["10011"]["action"] == "acceptTask"


def test_task_execute_sends_sandbox_command(
    payload_factory, role_factory, task_factory
):
    payload = _task_payload(
        payload_factory, role_factory, task_factory, 6, phase_task=BEIJING_TASK_TEXT
    )
    response = decide(payload)
    assert response["executeCmd"], "任务进行中必须下发 executeCmd"
    assert "$P -u -" in response["executeCmd"]
    _assert_valid_response(payload, response)


def test_no_sandbox_command_without_task(payload_factory, base_roles):
    payload = payload_factory(round_no=5, roles=base_roles)
    response = decide(payload)
    assert response["executeCmd"] == ""


def test_submit_answer_on_answer_ready(payload_factory, role_factory, task_factory):
    """沙盒给出 `[ANSWER]` ⇒ 本回合必须提交（T2：从未提交）"""
    payload = _task_payload(
        payload_factory, role_factory, task_factory, 20,
        phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    )
    # 先让它认领任务（上一回合下发过 recon）
    decide(_task_payload(
        payload_factory, role_factory, task_factory, 19, phase_task=BEIJING_TASK_TEXT
    ))
    response = decide(payload)
    assert response["roleCommandMap"]["10011"]["action"] == "submitAnswer"
    _assert_valid_response(payload, response)


def test_task_never_seen_does_not_send_sandbox_command(
    payload_factory, base_roles
):
    payload = payload_factory(round_no=50, roles=base_roles)
    response = decide(payload)
    assert response["executeCmd"] == ""


# ==========================================================================
# 稳健性
# ==========================================================================


def test_survives_missing_station(payload_factory, role_factory):
    """基地被摧毁后仍要能出指令（不能因为 `station() is None` 就崩）"""
    payload = payload_factory(
        round_no=100,
        roles=[role_factory(10010, "worker", 5, 5, backpack=["stone"])],
    )
    response = decide(payload)
    _assert_valid_response(payload, response)


def test_survives_map_with_no_land(payload_factory, role_factory, zone_factory):
    """极端报文（到处是矿区）也不能崩"""
    zones = [
        zone_factory("stone", x, y) for x in range(0, 41) for y in range(0, 32)
    ]
    payload = payload_factory(
        round_no=1,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10010, "worker", 5, 5)],
        zones=zones,
    )
    response = decide(payload)
    assert set(response) == {"roleCommandMap", "prompt", "executeCmd"}


def test_full_game_never_raises(payload_factory, role_factory, robot_factory,
                                task_factory, zone_factory):
    """连跑 130 个回合（一整天）不抛异常、不产生非法指令"""
    roles = [
        role_factory(10013, "station", 30, 10),
        role_factory(10010, "worker", 5, 23),
        role_factory(10011, "pioneer", 25, 3),
        role_factory(10012, "worker", 10, 16),
    ]
    zones = [
        zone_factory("stone", 4, 24), zone_factory("iron", 25, 10),
        zone_factory("copper", 22, 26), zone_factory("vendor", 20, 16),
        zone_factory("weaponShop", 25, 20),
        zone_factory("challengerTaskPoint1", 14, 14),
        zone_factory("challengerTaskPoint2", 17, 17),
    ]
    robots = [robot_factory(30000 + i, 20 + i % 5, 20 + i // 5)
              for i in range(12)]

    for round_no in range(1, 131):
        payload = payload_factory(
            round_no=round_no,
            gold=75,
            roles=roles,
            zones=zones,
            robots=robots if round_no > 70 else [],
            player_tasks=[task_factory("自进化类1", 14, 14)],
            last_action_results={u["id"]: True for u in roles},
        )
        response = decide(payload)
        _assert_valid_response(payload, response)
