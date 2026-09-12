"""协议模块测试（设计文档 3.3 节）"""

from __future__ import annotations

import pytest

from agent.protocol import (
    DAY_ROUNDS,
    GATLING,
    ROCKET,
    STONE_MINE,
    VENDOR,
    WALL,
    WORKER,
    Pos,
    Turn,
    Unit,
    accept_task_command,
    attack_command,
    build_command,
    buy_command,
    collect_command,
    distance,
    drop_command,
    manhattan_distance,
    move_command,
    remove_command,
    sell_command,
    station_footprint,
    submit_answer_command,
    summon_treasure_command,
    use_command,
)


# === 基础几何 ===


def test_pos_distance():
    """切比雪夫距离"""
    assert distance(Pos(0, 0), Pos(3, 4)) == 4
    assert distance(Pos(0, 0), Pos(3, 3)) == 3
    assert distance(Pos(5, 5), Pos(5, 5)) == 0


def test_manhattan_distance():
    """曼哈顿距离"""
    assert manhattan_distance(Pos(0, 0), Pos(3, 4)) == 7
    assert manhattan_distance(Pos(0, 0), Pos(3, 3)) == 6


def test_pos_add_and_dump():
    """坐标加法与序列化"""
    assert Pos(5, 10) + (2, -3) == Pos(7, 7)
    assert Pos(5, 10).dump() == {"x": 5, "y": 10}
    assert Pos.load({"x": "5", "y": 10}) == Pos(5, 10)


def test_station_footprint():
    """基地占2x2，pos为左上角"""
    footprint = station_footprint(Pos(10, 24))
    assert len(footprint) == 4
    assert Pos(10, 24) in footprint
    assert Pos(11, 24) in footprint
    assert Pos(10, 23) in footprint
    assert Pos(11, 23) in footprint


# === 单位解析 ===


def test_unit_load_defaults(role_factory):
    """角色缺少可选字段时应使用默认值"""
    unit = Unit.load(role_factory(10010, WORKER, 5, 23))
    assert unit.unit_id == 10010
    assert unit.kind == WORKER
    assert unit.pos == Pos(5, 23)
    assert unit.level == 1
    assert unit.cooldown == 0
    assert unit.backpack == ()
    assert unit.is_alive


def test_unit_backpack_full(role_factory):
    """背包容量判断"""
    raw = role_factory(10010, WORKER, 5, 23, backPackCapability=2,
                       backpack=["stone", "iron"])
    assert Unit.load(raw).backpack_full

    raw["backpack"] = ["stone"]
    assert not Unit.load(raw).backpack_full

    # 报文未给出背包容量的单位（capacity 为 None）不视为“已满”
    raw = role_factory(10013, "station", 10, 24)
    del raw["backPackCapability"]
    assert not Unit.load(raw).backpack_full

    # 容量为0的建筑无法携带物品，视为已满（不会派去采矿）
    assert Unit.load(role_factory(10013, "station", 10, 24)).backpack_full


def test_unit_dead_is_not_alive(role_factory):
    """血量归零即阵亡"""
    unit = Unit.load(role_factory(10010, WORKER, 5, 23, health=0))
    assert not unit.is_alive


def test_range_of_attack_prefers_payload_value(role_factory):
    """攻击距离优先取报文中的实际值（火箭 level1 即为全图）"""
    rocket = Unit.load(role_factory(10040, ROCKET, 9, 25, attackRange=2147483647))
    assert rocket.range_of_attack() == 2147483647

    # 报文未给出时回退到等级表
    gatling = Unit.load(role_factory(10020, GATLING, 9, 24, attackRange=0, level=2))
    assert gatling.range_of_attack() == 5

    # 非武器单位无攻击距离
    assert Unit.load(role_factory(10010, WORKER, 5, 23)).range_of_attack() == 0


# === 回合解析 ===


def test_player_task_load_without_timeout_rounds(task_factory):
    """回归：真实请求的 playerTasks 可能缺少 timeoutRounds，不能抛异常"""
    raw = task_factory(14, 14)
    del raw["timeoutRounds"]

    turn = Turn.load({
        "roundNo": 1,
        "mapInfo": {"width": 41, "height": 32, "zones": []},
        "teamOur": {"type": "challenger", "roles": [], "playerTasks": [raw]},
    })
    assert turn.player_tasks[0].timeout_rounds == 0
    assert turn.player_tasks[0].is_valid


@pytest.mark.parametrize(
    "round_no,expected_day",
    [
        (1, True),          # 第1天白天
        (DAY_ROUNDS, True),  # 第1天最后一个白天回合
        (DAY_ROUNDS + 1, False),  # 第1天第一个夜晚回合
        (130, False),       # 第1天最后一个夜晚回合
        (131, True),        # 第2天白天
    ],
)
def test_turn_day_night(payload_factory, round_no, expected_day):
    """昼夜判定：白天70回合，夜晚60回合"""
    turn = Turn.load(payload_factory(round_no=round_no))
    assert turn.is_day is expected_day


def test_turn_queries(payload_factory, role_factory, robot_factory):
    """单位与地图查询"""
    payload = payload_factory(
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(10011, "pioneer", 10, 12, backPackCapability=40),
            role_factory(10020, GATLING, 9, 24, level=1),
            role_factory(40000, WALL, 5, 20),
        ],
        zones=[(STONE_MINE, 4, 24), (VENDOR, 20, 16)],
        robots=[robot_factory(30001, 4, 4)],
    )
    turn = Turn.load(payload)

    assert turn.station() is not None
    assert [unit.unit_id for unit in turn.workers()] == [10010]
    assert [unit.unit_id for unit in turn.pioneers()] == [10011]
    assert [unit.unit_id for unit in turn.weapons()] == [10020]
    assert len(turn.walls()) == 1
    assert turn.controllable() == (turn.workers()[0], turn.pioneers()[0])

    assert turn.stone_mines() == (Pos(4, 24),)
    assert turn.iron_mines() == ()
    assert turn.copper_mines() == ()

    # 可通行判断：越界/矿区/中立单位都不可通行
    assert turn.land(Pos(0, 0))
    assert not turn.land(Pos(-1, 0))
    assert not turn.land(Pos(41, 0))
    assert not turn.land(Pos(4, 24))


def test_turn_occupied_and_blocked(payload_factory, role_factory, robot_factory):
    """阻挡格计算：含己方单位、矿区、机器人与敌方单位"""
    payload = payload_factory(
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
        zones=[(STONE_MINE, 4, 24)],
        robots=[robot_factory(30001, 4, 4)],
        enemies=[role_factory(20013, "station", 30, 10)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]

    # 基地2x2 + 工人1格
    assert len(turn.occupied_cells()) == 5
    assert Pos(10, 23) in turn.occupied_cells()

    blocked = turn.blocked(worker)
    assert Pos(10, 24) in blocked  # 己方基地
    assert Pos(4, 24) in blocked  # 矿区
    assert Pos(4, 4) in blocked  # 机器人
    assert Pos(30, 10) in blocked  # 敌方基地（2x2）
    assert worker.pos not in blocked  # 自身位置不算阻挡


def test_alive_robots_targeting_me(payload_factory, robot_factory):
    """只统计攻击我方且存活的机器人"""
    payload = payload_factory(
        robots=[
            robot_factory(30001, 4, 4, targetTeam="challenger"),
            robot_factory(30002, 5, 4, targetTeam="defender"),
            robot_factory(30003, 6, 4, targetTeam="challenger", health=0),
        ],
    )
    turn = Turn.load(payload)
    assert [robot.robot_id for robot in turn.alive_robots_targeting_me()] == [30001]


# === 指令构建 ===


def test_move_and_collect_commands():
    assert move_command(Pos(1, 2)) == {
        "action": "move", "targetPos": [{"x": 1, "y": 2}],
    }
    assert collect_command(Pos(6, 13)) == {
        "action": "collect", "targetPos": [{"x": 6, "y": 13}],
    }


def test_build_and_remove_commands():
    assert build_command(Pos(9, 24), WALL) == {
        "action": "build", "targetPos": [{"x": 9, "y": 24}], "name": "wall",
    }
    assert remove_command(Pos(9, 24)) == {
        "action": "remove", "targetPos": [{"x": 9, "y": 24}],
    }


def test_attack_command_multi_target():
    """加特林/火箭可多目标，controllerId 必须是字符串"""
    command = attack_command(10010, [Pos(29, 7), Pos(29, 8)])
    assert command["action"] == "attack"
    assert command["controllerId"] == "10010"
    assert command["targetPos"] == [{"x": 29, "y": 7}, {"x": 29, "y": 8}]


def test_sell_and_buy_commands():
    assert sell_command("stone") == {"action": "sell", "name": "stone", "num": 1}
    assert sell_command("stone", 10) == {
        "action": "sell", "name": "stone", "num": 10,
    }
    assert buy_command("Medicine", 2) == {
        "action": "buy", "name": "Medicine", "num": 2,
    }


def test_use_command_with_and_without_target():
    assert use_command("Medicine") == {"action": "use", "name": "Medicine"}
    assert use_command("WallFixer", Pos(9, 24)) == {
        "action": "use", "name": "WallFixer", "targetPos": [{"x": 9, "y": 24}],
    }


def test_drop_and_task_commands():
    assert drop_command("stone") == {"action": "drop", "name": "stone"}
    assert accept_task_command() == {"action": "acceptTask"}
    assert submit_answer_command("xxx") == {
        "action": "submitAnswer", "taskAnswer": "xxx",
    }


def test_summon_treasure_command():
    command = summon_treasure_command(Pos(29, 7), ["AcientTablet", "StarSand"])
    assert command == {
        "action": "summonTreasure",
        "targetPos": [{"x": 29, "y": 7}],
        "item": ["AcientTablet", "StarSand"],
    }
