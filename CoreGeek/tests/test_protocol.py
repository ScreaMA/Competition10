"""协议层用例：报文解析与指令构造。

对应设计文档V2 §10.1。

这一层是**异常预算**（任务书 §8：累计 5 次异常即整场停调度）的唯一防线，
所以用例集中在两件事上：

1. 报文里每个字段都真的被解析了（V1 漏了 `errors`/`llmResp`/
   `lastSummonTreasureResult`，导致"答案错误"这类关键信号读不到）。
2. 构造器在参数不合法时**返回 None**，而不是发一条字段不全的指令。
"""

from __future__ import annotations

import json

import pytest

from agent import protocol
from agent.protocol import (
    Pos,
    Turn,
    Unit,
    attack_command,
    build_command,
    distance,
    neighbours,
    station_footprint,
    use_command,
)


# ==========================================================================
# 报文解析
# ==========================================================================


def test_load_full_payload(payload_factory, role_factory, task_factory, robot_factory,
                           zone_factory):
    payload = payload_factory(
        round_no=85,
        gold=20,
        roles=[role_factory(10013, "station", 10, 24)],
        zones=[zone_factory("stone", 4, 24), zone_factory("vendor", 20, 16)],
        player_tasks=[task_factory("自进化类1", 14, 14)],
        robots=[robot_factory(30001, 4, 4)],
        enemies=[role_factory(20013, "station", 30, 10)],
        phase_task="请阅读task_1_beijing.md，获取任务信息",
        llm_resp="ANSWER: 42",
        errors=[{"errorCode": 2, "description": "答案错误"}],
    )
    turn = Turn.load(payload)

    assert turn.round_no == 85
    assert turn.gold == 20
    assert turn.station() is not None
    assert turn.player_tasks[0].pos == Pos(14, 14)
    assert turn.player_tasks[0].timeout_rounds == 15
    assert turn.mines("stone") == (Pos(4, 24),)
    assert turn.enemies[0].kind == "station"
    assert turn.phase_task.startswith("请阅读")
    # V1 漏解析的三个字段
    assert turn.llm_resp == "ANSWER: 42"
    assert turn.errors[0].code == 2
    assert turn.answer_wrong is True
    assert turn.last_summon_result == 0


def test_day_night_boundary(payload_factory):
    """白天 70 回合、夜晚 60 回合（任务书 §4.2）"""
    def is_day(round_no):
        return Turn.load(payload_factory(round_no=round_no)).is_day

    assert is_day(1) is True
    assert is_day(70) is True
    assert is_day(71) is False
    assert is_day(130) is False
    assert is_day(131) is True  # 第 2 天
    # 第 10 天 = R1171..R1300，其中 R1171..R1240 是白天、R1241..R1300 是夜晚
    assert is_day(1171) is True
    assert is_day(1240) is True
    assert is_day(1241) is False
    assert is_day(1300) is False  # 最后 60 回合是夜战


def test_missing_optional_fields_do_not_crash():
    """只给必需字段也要能解析（判题器早期回合可能省略可选字段）"""
    turn = Turn.load({
        "roundNo": 1,
        "mapInfo": {"width": 41, "height": 32},
        "teamOur": {"type": "challenger", "roles": []},
    })
    assert turn.round_no == 1
    assert turn.robots == ()
    assert turn.player_tasks == ()
    assert turn.last_action_results == {}


def test_task_without_timeout_rounds(payload_factory, task_factory):
    """任务点缺 `timeoutRounds` 时要走兜底值，而不是让整回合决策崩掉

    报文里的可选字段随时可能缺席，而 `decide` 的兜底是"返回空指令"——
    这种情况在日志里只表现为"这一回合什么都没做"，很难查。所以这里直接
    对着 `PlayerTask.load` 断言。
    """
    task = task_factory("自进化类1", 14, 14)
    task.pop("timeoutRounds")
    turn = Turn.load(payload_factory(player_tasks=[task]))
    assert turn.player_tasks[0].timeout_rounds == protocol.TASK_DEFAULT_TIMEOUT


def test_station_footprint_is_2x2():
    """基地 pos 是左上角，占 2×2（接口文档 §1.3.1 注）"""
    cells = station_footprint(Pos(10, 24))
    assert set(cells) == {Pos(10, 24), Pos(11, 24), Pos(10, 23), Pos(11, 23)}
    assert protocol.footprint_origin(cells) == Pos(10, 23)


def test_distance_is_chebyshev():
    """切比雪夫距离（任务书 §4.5.4）"""
    assert distance(Pos(0, 0), Pos(3, 1)) == 3
    assert distance(Pos(0, 0), Pos(2, 2)) == 2
    assert distance(Pos(5, 5), Pos(5, 5)) == 0


def test_blocked_includes_all_obstacles(payload_factory, role_factory, zone_factory,
                                        robot_factory):
    """任务书 §4.1：建筑、角色、机器人、中立单位、任务点、矿区**全部**阻挡移动"""
    worker = role_factory(10010, "worker", 5, 5)
    turn = Turn.load(payload_factory(
        roles=[worker, role_factory(10013, "station", 20, 10)],
        zones=[zone_factory("stone", 4, 4), zone_factory("vendor", 20, 16)],
        robots=[robot_factory(30001, 6, 6)],
    ))
    blocked = turn.blocked_for(Unit.load(worker))
    assert Pos(4, 4) in blocked       # 矿区
    assert Pos(20, 16) in blocked     # 小贩
    assert Pos(20, 10) in blocked     # 己方基地
    assert Pos(6, 6) in blocked       # 机器人
    assert Pos(5, 5) not in blocked   # 自己所在格不算阻挡
    assert turn.is_land(Pos(7, 7)) is True


# ==========================================================================
# 指令构造（异常预算防线）
# ==========================================================================


def test_attack_target_count_must_equal_level():
    """加特林/火箭的 `targetPos` 个数必须等于武器等级（接口文档 §2.2）

    个数不对会让**整次攻击非法**，直接吃掉一次异常预算。
    """
    for level, expected in ((1, 1), (2, 2), (3, 3)):
        gatling = Unit.load({
            "id": 10020, "pos": {"x": 10, "y": 10}, "roleType": "gatling",
            "health": 1000, "attackPower": 10, "attackRange": 3, "level": level,
        })
        command = attack_command(10010, gatling, [Pos(12, 12)])
        assert command is not None
        assert len(command["targetPos"]) == expected
        # 落点不够时用最后一个补齐（重复落点 = 指令执行失败，不是异常）
        assert all(pos == {"x": 12, "y": 12} for pos in command["targetPos"])


def test_railgun_always_single_target():
    """电磁狙击炮只能攻击一个目标（任务书 §4.5.4）"""
    railgun = Unit.load({
        "id": 10030, "pos": {"x": 10, "y": 10}, "roleType": "railgun",
        "health": 1000, "attackPower": 10, "attackRange": 6, "level": 3,
    })
    command = attack_command(10010, railgun, [Pos(12, 12), Pos(13, 13)])
    assert command is not None
    assert len(command["targetPos"]) == 1


def test_attack_requires_target():
    gatling = Unit.load({
        "id": 10020, "pos": {"x": 10, "y": 10}, "roleType": "gatling",
        "health": 1000, "attackPower": 10, "attackRange": 3, "level": 1,
    })
    assert attack_command(10010, gatling, []) is None


def test_use_with_position_required_for_bomb():
    """眩晕法宝/范围炸弹必须带 `targetPos`（接口文档 §2.2 的指令错误口径）"""
    assert use_command("Bomb") is None
    assert use_command("DizzyWeapon") is None
    assert use_command("Bomb", Pos(3, 3)) is not None
    # 生命药剂不需要坐标
    assert use_command("Medicine") is not None


def test_submit_answer_rejects_empty():
    assert protocol.submit_answer_command("") is None
    assert protocol.submit_answer_command("   ") is None
    assert protocol.submit_answer_command('{"a":1}') is not None


def test_sell_buy_reject_bad_num():
    assert protocol.sell_command("stone", 0) is None
    assert protocol.sell_command("stone", -1) is None
    assert protocol.buy_command("Medicine", 0) is None
    assert protocol.sell_command("stone") == {
        "action": "sell", "name": "stone", "num": 1
    }


def test_build_command_shape():
    assert build_command(Pos(3, 4), "wall") == {
        "action": "build",
        "targetPos": [{"x": 3, "y": 4}],
        "name": "wall",
    }


def test_response_keys_are_strings():
    """`roleCommandMap` 的 key 必须是字符串（接口文档 §2.1 是 Map<int,…>）"""
    response = protocol.build_response({10010: protocol.move_command(Pos(1, 1))})
    assert set(response["roleCommandMap"]) == {"10010"}
    assert set(response) == {"roleCommandMap", "prompt", "executeCmd"}
    assert json.loads(protocol.dumps(response))["roleCommandMap"]["10010"]["action"] == "move"


def test_use_needs_pos_covers_all_vouchers():
    """所有"需要目标位置"的道具白名单要完整（漏一个就会产生一次异常）"""
    for name in ("WeaponUpgradeVoucher1", "WeaponUpgradeVoucher2",
                 "WallUpgradeVoucher1", "StationUpgradeVoucher1",
                 "WallFixer", "Bomb", "DizzyWeapon"):
        assert name in protocol.USE_NEEDS_POS


def test_weapon_range_by_level():
    """武器射程表（任务书 §4.5.1）"""
    rocket = Unit.load({
        "id": 10040, "pos": {"x": 1, "y": 1}, "roleType": "rocket",
        "health": 1000, "attackPower": 20, "attackRange": 0, "level": 3,
    })
    assert rocket.range_of_attack() == 10**9  # level3 全图
    gatling = Unit.load({
        "id": 10020, "pos": {"x": 1, "y": 1}, "roleType": "gatling",
        "health": 1000, "attackPower": 0, "attackRange": 0, "level": 2,
    })
    assert gatling.range_of_attack() == 5


def test_max_health_table():
    """满血表（任务书 §4.5.1）"""
    wall = Unit.load({
        "id": 40000, "pos": {"x": 1, "y": 1}, "roleType": "wall",
        "health": 500, "level": 2,
    })
    assert wall.max_health() == 1500
    assert wall.health_ratio() == pytest.approx(500 / 1500)


def test_neighbours_are_eight_directions():
    """八方向邻居（任务书 §4.5.4 第 2 条）"""
    assert len(neighbours(Pos(5, 5))) == 8
    assert Pos(4, 4) in neighbours(Pos(5, 5))
    assert Pos(6, 6) in neighbours(Pos(5, 5))
