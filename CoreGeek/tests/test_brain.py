"""决策模块测试（设计文档 3.5 节）"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent.brain as brain
from agent.brain import (
    SELL_BATCH,
    STONE_BATCH,
    TASK_MARKER,
    TOWER_LOADOUT,
    UPGRADE_GOLD,
    WEAPON_UPGRADE_VOUCHER,
    _calc_tower_sites,
    _calc_wall_order,
    _generate_strategy_prompt,
    _pair_controllers_and_weapons,
    _task_token,
    _valid_stand_cells,
    decide,
    sandbox_command,
)
from agent.grid import get_neighbors
from agent.protocol import (
    CHALLENGER_TASK_1,
    COPPER_MINE,
    DAY_ROUNDS,
    GATLING,
    IRON_MINE,
    PIONEER,
    RAILGUN,
    ROCKET,
    STONE_MINE,
    VENDOR,
    WALL,
    WALL_MATERIAL,
    WEAPON_BUILD_COST,
    WEAPON_SHOP,
    WORKER,
    Pos,
    Turn,
    distance,
    station_footprint,
)

SAMPLE = Path(__file__).resolve().parents[2] / "docs" / "request.txt"

# 默认基地坐标（conftest 中的 payload_factory 默认值）
STATION = Pos(10, 24)


def _footprint_distance(pos: Pos, footprint) -> int:
    return min(distance(pos, cell) for cell in footprint)


def _sandbox_result(phase_task: str, output: str, exit_code: int = 0) -> str:
    """按接口文档格式构造一条沙盒命令执行结果（lastCmdResult）"""
    return f"[exitCode:{exit_code}]\n{TASK_MARKER}{_task_token(phase_task)}\n{output}"


# === 建造位置规划 ===


def test_tower_sites_layout(payload_factory):
    """武器塔位置：基地周围一圈内、分散在三个不同方位"""
    turn = Turn.load(payload_factory())
    sites = _calc_tower_sites(turn)
    footprint = station_footprint(STATION)

    assert len(sites) == 3
    assert sites == (Pos(12, 23), Pos(10, 22), Pos(9, 23))
    for site in sites:
        assert turn.land(site)
        assert site not in footprint
        assert _footprint_distance(site, footprint) == 1
    assert len(set(sites)) == 3
    # 回归：三座塔曾经全部挤在基地左边同一列(x=9)，现在必须覆盖不同方位
    assert len({site.x for site in sites}) == 3


def test_tower_loadout_covers_all_weapon_types():
    """建造顺序包含全部三种武器"""
    assert TOWER_LOADOUT == (GATLING, RAILGUN, ROCKET)


def test_tower_sites_without_station(payload_factory):
    """没有基地时不规划武器塔（不崩溃）"""
    turn = Turn.load(payload_factory(station=None))
    assert _calc_tower_sites(turn) == ()


def test_wall_order_shape(payload_factory):
    """围墙：环绕基地第二圈，无重复，且留出右下角入口"""
    turn = Turn.load(payload_factory())
    order = _calc_wall_order(turn)
    footprint = station_footprint(STATION)

    # 6(上) + 4(左) + 6(下) + 4(右) - 1(入口) = 19
    assert len(order) == 19
    assert len(set(order)) == 19
    assert Pos(13, 22) not in order  # 入口
    for pos in order:
        assert _footprint_distance(pos, footprint) == 2
        assert turn.land(pos)
        assert pos != STATION


def test_wall_order_within_map_bounds(payload_factory):
    """基地贴地图角落时，围墙坐标不得越界"""
    turn = Turn.load(payload_factory(station=(0, 0)))
    order = _calc_wall_order(turn)

    assert order  # 仍能规划出可用围墙
    for pos in order:
        assert 0 <= pos.x < turn.width
        assert 0 <= pos.y < turn.height


def test_wall_order_without_station(payload_factory):
    turn = Turn.load(payload_factory(station=None))
    assert _calc_wall_order(turn) == ()


# === 白天决策 ===


def test_decide_day_builds_weapon(payload_factory, role_factory):
    """金币充足且工人站在建造点旁时，优先建造武器工事"""
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[role_factory(10010, WORKER, 11, 22, backPackCapability=100)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": GATLING,
    }


def test_workers_take_different_tower_sites(payload_factory, role_factory):
    """多名工人分头施工：塔位一经认领，后面的工人改去下一座

    回归：以前只有"真正把塔建起来"才会占用 claimed，于是两个工人会一起奔向
    同一座塔，另一个塔位整局没人管（对战复盘里"金币闲置、武器只建成一座"
    就是这个现象）。
    """
    before = Pos(15, 30)
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST * 3,
        roles=[
            role_factory(10010, WORKER, before.x, before.y, backPackCapability=100),
            role_factory(10012, WORKER, 11, 22, backPackCapability=100),
        ],
    )
    commands, _ = decide(payload)

    # 第一个工人离得远，认领第1座塔位(12,23)并向它移动
    command = commands["10010"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(12, 23)) < distance(before, Pos(12, 23))
    # 第二个工人已经站在第2座塔位(10,22)旁边，直接开工而不是跟着挤第1座
    assert commands["10012"] == {
        "action": "build",
        "targetPos": [{"x": 10, "y": 22}],
        "name": RAILGUN,
    }


def test_worker_retries_next_site_after_failed_action(
    payload_factory, role_factory,
):
    """上一回合动作失败时换一座塔重试，不再原地重复同一条失败指令"""
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST * 3,
        roles=[role_factory(10010, WORKER, 11, 22, backPackCapability=100)],
    )

    # 正常情况：就站在第1座塔位(12,23)旁，直接建造
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": GATLING,
    }

    # 上一回合被判为失败（目标点被夺取等）：换第2座塔位(10,22)重试
    payload["lastRoundRoleActionResults"] = {"10010": False}
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 10, "y": 22}],
        "name": RAILGUN,
    }

    # 只剩一座塔位可建时不换位，否则只会白白空转
    payload["teamOur"]["roles"] += [
        role_factory(10020, RAILGUN, 10, 22, attackRange=6),
        role_factory(10030, ROCKET, 9, 23, attackRange=10),
    ]
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": GATLING,
    }


def test_action_failed_reads_last_round_results(payload_factory):
    """上一回合的动作结果按角色ID解析，报文没给的角色视为成功"""
    payload = payload_factory()
    payload["lastRoundRoleActionResults"] = {"10010": False, "10012": True}
    turn = Turn.load(payload)

    assert turn.action_failed(10010) is True
    assert turn.action_failed(10012) is False
    assert turn.action_failed(99999) is False


def test_decide_day_collects_stone(payload_factory, role_factory):
    """工人紧邻石矿时执行采集"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "collect",
        "targetPos": [{"x": 4, "y": 24}],
    }


def test_decide_day_builds_wall_when_has_stone(payload_factory, role_factory):
    """背包里有石头且站在建造点旁时建造围墙"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(
                10012, WORKER, 12, 26, backPackCapability=100,
                backpack=[WALL_MATERIAL],
            ),
        ],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }


def test_worker_stone_batch_makes_it_keep_collecting(payload_factory, role_factory):
    """石头不足一批时继续采集，不急着去建围墙"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, 5, 23, backPackCapability=100,
                backpack=[WALL_MATERIAL] * (STONE_BATCH - 1),
            ),
        ],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)
    assert commands["10010"]["action"] == "collect"


def test_worker_gathers_iron_when_no_stone(payload_factory, role_factory):
    """附近没有石矿时退而采集铁矿，避免整回合没有任何产出（空转）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
        zones=[(IRON_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "collect",
        "targetPos": [{"x": 4, "y": 24}],
    }


def test_worker_keeps_building_when_defense_not_ready_before_night(
    payload_factory, role_factory,
):
    """天黑前火力不足且买得起塔时先补塔，而不是回防待命"""
    payload = payload_factory(
        round_no=DAY_ROUNDS - 3,  # 距天黑还有4回合
        gold=WEAPON_BUILD_COST,
        roles=[role_factory(10010, WORKER, 11, 22, backPackCapability=100)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": GATLING,
    }


def test_units_avoid_standing_on_build_sites(payload_factory, role_factory):
    """白天角色不站到武器塔/围墙的建造点上（占住建造位会让建筑建不起来）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10010, WORKER, 20, 20, backPackCapability=100)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]

    # (12,23) 是规划中的武器塔位置，(11,22) 的相邻格里包含它
    assert Pos(12, 23) in _calc_tower_sites(turn)
    assert Pos(12, 23) not in _valid_stand_cells(turn, worker, Pos(11, 22), set())
    # 目标本身就是建造点时（走去施工）不过滤，否则相邻格全是建造位就无处落脚
    assert Pos(13, 23) in _calc_wall_order(turn)
    assert Pos(13, 23) in _valid_stand_cells(turn, worker, Pos(12, 23), set())

    # 夜晚不施工，建造点可以正常站人（操控武器时站位更自由）
    night = Turn.load(payload_factory(
        round_no=71,
        gold=0,
        roles=[role_factory(10010, WORKER, 20, 20, backPackCapability=100)],
    ))
    assert Pos(12, 23) in _valid_stand_cells(
        night, night.workers()[0], Pos(11, 22), set(),
    )


def test_worker_falls_back_to_weapon_before_night(payload_factory, role_factory):
    """天黑前工人停止采集，回到武器旁待命，保证夜晚火力不空转"""
    before = Pos(5, 23)
    weapon_pos = Pos(9, 24)
    payload = payload_factory(
        round_no=DAY_ROUNDS - 3,  # 距天黑还有4回合
        gold=0,
        roles=[
            role_factory(10010, WORKER, before.x, before.y, backPackCapability=100),
            role_factory(10020, GATLING, weapon_pos.x, weapon_pos.y, attackRange=4),
        ],
        zones=[(STONE_MINE, 4, 24)],  # 紧邻石矿，但回防优先
    )
    commands, _ = decide(payload)

    command = commands["10010"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, weapon_pos) < distance(before, weapon_pos)


# === 资源交易 ===


def _payload_with_full_walls(payload_factory, role_factory, worker_pos: Pos,
                             zones=(), backpack=(), capacity: int = 100) -> dict:
    """构造“围墙已建完”的局面"""
    order = _calc_wall_order(Turn.load(payload_factory()))
    walls = [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(order)
    ]
    walls.append(
        role_factory(
            10010, WORKER, worker_pos.x, worker_pos.y,
            backPackCapability=capacity, backpack=list(backpack),
        ),
    )
    return payload_factory(gold=0, roles=walls, zones=list(zones))


def test_trade_sells_stone_when_adjacent_to_vendor(payload_factory, role_factory):
    """围墙建完且站在小贩旁时，卖出多余石头换金币"""
    payload = _payload_with_full_walls(
        payload_factory, role_factory,
        worker_pos=Pos(20, 17),
        zones=[(VENDOR, 20, 16)],
        backpack=[WALL_MATERIAL] * SELL_BATCH,
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "sell",
        "name": WALL_MATERIAL,
        "num": SELL_BATCH,
    }


def test_trade_moves_toward_vendor_when_far(payload_factory, role_factory):
    """石头够一批但不在小贩旁时，先走向小贩"""
    before = Pos(20, 21)
    payload = _payload_with_full_walls(
        payload_factory, role_factory,
        worker_pos=before,
        zones=[(VENDOR, 20, 16)],
        backpack=[WALL_MATERIAL] * SELL_BATCH,
    )
    commands, _ = decide(payload)

    command = commands["10010"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(20, 16)) < distance(before, Pos(20, 16))


def test_trade_keeps_stone_when_backpack_not_full(payload_factory, role_factory):
    """石头不足一批时不卖（留给后续围墙/升级）"""
    payload = _payload_with_full_walls(
        payload_factory, role_factory,
        worker_pos=Pos(20, 17),
        zones=[(VENDOR, 20, 16)],
        backpack=[WALL_MATERIAL] * (SELL_BATCH - 1),
    )
    commands, _ = decide(payload)

    assert "10010" not in commands


def test_trade_sells_ore_when_backpack_full(payload_factory, role_factory):
    """背包被矿石塞满时先卖一批腾地方（铁/铜同样能换金币）"""
    payload = _payload_with_full_walls(
        payload_factory, role_factory,
        worker_pos=Pos(20, 17),
        zones=[(VENDOR, 20, 16)],
        backpack=[COPPER_MINE] * 3,
        capacity=3,
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": 3,
    }


def test_worker_keeps_mining_for_gold_after_walls_done(
    payload_factory, role_factory,
):
    """围墙建完后工人继续采石换金币，避免经济在第1天就冻结"""
    payload = _payload_with_full_walls(
        payload_factory, role_factory,
        worker_pos=Pos(5, 23),
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "collect",
        "targetPos": [{"x": 4, "y": 24}],
    }


# === 金币消费（武器升级） ===


def _payload_with_full_defense(payload_factory, role_factory, gold: int,
                               worker_pos: Pos, backpack=(), zones=()) -> dict:
    """构造“围墙与三座武器都已建完”的局面"""
    base = Turn.load(payload_factory())
    roles = [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(_calc_wall_order(base))
    ]
    roles += [
        role_factory(10020 + index, kind, pos.x, pos.y, attackRange=4)
        for index, (kind, pos) in enumerate(
            zip(TOWER_LOADOUT, _calc_tower_sites(base))
        )
    ]
    roles.append(
        role_factory(
            10010, WORKER, worker_pos.x, worker_pos.y,
            backPackCapability=100, backpack=list(backpack),
        ),
    )
    return payload_factory(gold=gold, roles=roles, zones=list(zones))


def test_worker_buys_upgrade_voucher_when_gold_spare(
    payload_factory, role_factory,
):
    """防线建完后金币不再闲置：去武器商店买武器升级券"""
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=UPGRADE_GOLD,
        worker_pos=Pos(20, 17),
        zones=[(WEAPON_SHOP, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "buy",
        "name": WEAPON_UPGRADE_VOUCHER,
        "num": 1,
    }


def test_worker_uses_upgrade_voucher_on_weapon(payload_factory, role_factory):
    """背包里有升级券时，走到武器旁使用（level1 -> level2）"""
    site = _calc_tower_sites(Turn.load(payload_factory()))[0]
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=0,
        worker_pos=Pos(site.x, site.y - 1),
        backpack=[WEAPON_UPGRADE_VOUCHER],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "use",
        "name": WEAPON_UPGRADE_VOUCHER,
        "targetPos": [{"x": site.x, "y": site.y}],
    }


# === 任务系统 ===


def test_pioneer_accepts_task_when_adjacent(payload_factory, role_factory):
    """开拓者在任务点旁领取任务"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 13, backPackCapability=40)],
        tasks=[(14, 14)],
        zones=[(CHALLENGER_TASK_1, 14, 14)],
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {"action": "acceptTask"}


def test_pioneer_moves_to_task_point(payload_factory, role_factory):
    """开拓者远离任务点时向任务点移动"""
    before = Pos(10, 12)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        tasks=[(14, 14)],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(14, 14)) < distance(before, Pos(14, 14))


def test_pioneer_stays_while_task_running(payload_factory, role_factory):
    """任务进行中且仍在任务点旁时原地不动（离开会导致任务结束）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="自进化类任务：请查询北京天气",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands


def test_pioneer_returns_to_task_point_when_task_running(payload_factory, role_factory):
    """任务进行中但被挤开时，回到任务点旁"""
    before = Pos(18, 14)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="自进化类任务：请查询北京天气",
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(14, 14)) < distance(before, Pos(14, 14))


def test_pioneer_falls_back_to_weapon(payload_factory, role_factory):
    """任务都在冷却中且短期不会开放时跟随武器塔"""
    before = Pos(11, 22)
    weapon_pos = Pos(9, 24)
    task_pos = Pos(14, 14)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40),
            role_factory(10020, GATLING, weapon_pos.x, weapon_pos.y, attackRange=4),
        ],
        tasks=[(task_pos.x, task_pos.y, {"isValid": False, "coldDownRounds": 20})],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    # 走向武器塔备战，而不是跑去冷却中的任务点
    assert distance(step, weapon_pos) <= distance(before, weapon_pos)
    assert distance(step, task_pos) >= distance(before, task_pos)


def test_pioneer_prefers_task_closer_to_timeout(payload_factory, role_factory):
    """两个任务都可接时先去快过期的那个（临期优先，避免任务白白过期）"""
    before = Pos(10, 12)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        tasks=[(14, 14, {"timeoutRounds": 30}), (10, 28, {"timeoutRounds": 5})],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(10, 28)) < distance(before, Pos(10, 28))


def test_pioneer_falls_back_to_next_task_when_one_is_unreachable(
    payload_factory, role_factory,
):
    """最优先的任务点被完全挡住时，开拓者退而去领另一个任务点

    回归：以前只试最优先的那个任务点，走不通就整回合放弃任务去跟随武器塔，
    160分+160金币的两个任务点会一起过期。
    """
    before = Pos(20, 20)
    blocked_task = Pos(30, 28)
    reachable_task = Pos(14, 14)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        # 任务点周围一格全是矿区（不可通行的中立元素），开拓者无法靠近领取
        zones=[(COPPER_MINE, pos.x, pos.y) for pos in get_neighbors(blocked_task)],
        tasks=[
            (blocked_task.x, blocked_task.y, {"timeoutRounds": 3}),
            (reachable_task.x, reachable_task.y, {"timeoutRounds": 30}),
        ],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    # 走向另一个可接的任务点，而不是干等被挡住的这个任务点过期
    assert step.x < before.x  # 朝任务点(14,14)所在方向移动
    assert distance(step, blocked_task) >= distance(before, blocked_task)


def test_pioneer_waits_near_task_point_when_on_cooldown(
    payload_factory, role_factory,
):
    """任务点还在冷却但快开放时，开拓者提前到任务点旁待命"""
    before = Pos(10, 12)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        tasks=[(14, 14, {"isValid": False, "coldDownRounds": 2})],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(14, 14)) < distance(before, Pos(14, 14))


def test_pioneer_uses_map_task_point_without_task_data(
    payload_factory, role_factory,
):
    """报文没有 playerTasks 时，按地图上的己方任务点前往，而不是原地游走"""
    before = Pos(10, 12)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        zones=[(CHALLENGER_TASK_1, 14, 14)],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, Pos(14, 14)) < distance(before, Pos(14, 14))


# === 自进化任务（沙盒作答） ===


def test_sandbox_command_reads_task_file(payload_factory, role_factory):
    """任务进行中：提交读取任务文件的沙盒命令"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读task_1_beijing.md",
    )
    command = sandbox_command(payload)

    assert 'cat -- "task_1_beijing.md"' in command
    # 命令必须带上任务标识，才能确认下一回合的输出属于本任务
    assert TASK_MARKER in command


def test_sandbox_command_idle_without_task(payload_factory):
    """没有进行中的任务时不使用沙盒"""
    assert sandbox_command(payload_factory(round_no=1)) == ""


def test_pioneer_submits_answer_from_sandbox(payload_factory, role_factory):
    """开拓者拿到沙盒输出后在任务点旁提交答案"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    # 本回合还没有沙盒输出，先申请命令
    assert sandbox_command(payload) != ""

    # 下一回合带回沙盒输出，开拓者直接作答
    payload["lastCmdResult"] = _sandbox_result(phase_task, "北京 晴 25摄氏度")
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": "北京 晴 25摄氏度",
    }
    # 已有答案后不再重复执行沙盒命令
    assert sandbox_command(payload) == ""


def test_pioneer_ignores_stale_or_failed_sandbox_output(
    payload_factory, role_factory,
):
    """沙盒输出不属于当前任务或执行失败时，不能拿来作答"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )

    # 上一个任务残留的输出
    payload["lastCmdResult"] = _sandbox_result("请阅读task_2_shanghai.md", "上海 多云")
    commands, _ = decide(payload)
    assert "10011" not in commands

    # 本任务但命令执行失败
    payload["lastCmdResult"] = _sandbox_result(
        phase_task, "No such file or directory", exit_code=1,
    )
    commands, _ = decide(payload)
    assert "10011" not in commands


def test_pioneer_skips_sandbox_error_output(payload_factory, role_factory):
    """沙盒输出是"文件读不到"这类错误时不提交：提交错误答案只会白费任务冷却"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _sandbox_result(
        phase_task, "cat: task_1_beijing.md: No such file or directory",
    )
    commands, _ = decide(payload)
    assert "10011" not in commands


# === 夜晚决策 ===


def test_decide_night_attacks_robot(payload_factory, role_factory, robot_factory):
    """夜晚：角色在武器旁时操控武器攻击最近的机器人"""
    payload = payload_factory(
        round_no=71,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        robots=[
            robot_factory(30001, 12, 24, targetTeam="challenger"),
            robot_factory(30002, 40, 1, targetTeam="challenger"),  # 超出射程
        ],
    )
    commands, _ = decide(payload)

    assert commands["10020"] == {
        "action": "attack",
        "targetPos": [{"x": 12, "y": 24}],
        "controllerId": "10010",
    }


def test_decide_night_prioritizes_dangerous_robot(
    payload_factory, role_factory, robot_factory,
):
    """射程内同时有小型机器人与BOSS时，优先攻击威胁更高的BOSS"""
    payload = payload_factory(
        round_no=71,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=5),
        ],
        robots=[
            robot_factory(30001, 12, 24, targetTeam="challenger"),  # 小型, 距离3
            robot_factory(
                30002, 13, 26, roleType="bossRobot", health=800,
                targetTeam="challenger",
            ),  # BOSS, 距离4
        ],
    )
    commands, _ = decide(payload)

    assert commands["10020"] == {
        "action": "attack",
        "targetPos": [{"x": 13, "y": 26}],
        "controllerId": "10010",
    }


def test_decide_night_ignores_robots_targeting_enemy(
    payload_factory, role_factory, robot_factory,
):
    """不攻击打向敌方阵营的机器人"""
    payload = payload_factory(
        round_no=71,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        robots=[robot_factory(30001, 12, 24, targetTeam="defender")],
    )
    commands, _ = decide(payload)

    assert "10020" not in commands


def test_decide_night_skips_weapon_on_cooldown(
    payload_factory, role_factory, robot_factory,
):
    """火箭发射台冷却中不攻击"""
    payload = payload_factory(
        round_no=71,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10040, ROCKET, 9, 24, attackRange=10, cooldown=3),
        ],
        robots=[robot_factory(30001, 12, 24, targetTeam="challenger")],
    )
    commands, _ = decide(payload)

    assert "10040" not in commands


def test_decide_night_moves_controller_to_weapon(payload_factory, role_factory):
    """夜晚：角色不在武器旁时逐回合走向武器，最终到位可操控攻击"""
    weapon_pos = Pos(9, 24)
    current = Pos(5, 20)

    for _ in range(10):
        commands, _ = decide(payload_factory(
            round_no=71,
            roles=[
                role_factory(
                    10010, WORKER, current.x, current.y, backPackCapability=100,
                ),
                role_factory(10020, GATLING, weapon_pos.x, weapon_pos.y, attackRange=4),
            ],
        ))
        if "10010" not in commands:  # 已到位，本回合由武器发起攻击
            break
        command = commands["10010"]
        assert command["action"] == "move"
        current = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
        assert distance(current, weapon_pos) <= distance(Pos(5, 20), weapon_pos)

    assert distance(current, weapon_pos) <= 1


def test_pair_controllers_and_weapons(payload_factory, role_factory):
    """角色与武器按距离就近配对，数量不匹配时取较少的那个"""
    payload = payload_factory(
        roles=[
            role_factory(10010, WORKER, 5, 23),
            role_factory(10011, PIONEER, 10, 12),
            role_factory(10012, WORKER, 10, 16),
            role_factory(10020, GATLING, 9, 24),
            role_factory(10030, RAILGUN, 10, 25),
        ],
    )
    pairs = _pair_controllers_and_weapons(Turn.load(payload))

    assert len(pairs) == 2
    # 回归：不再按ID顺序硬配对（开拓者在(10,12)离两座武器都最远，不应占坑）
    assert [controller.unit_id for controller, _ in pairs] == [10010, 10012]
    assert [weapon.unit_id for _, weapon in pairs] == [10020, 10030]
    assert distance(pairs[0][0].pos, pairs[0][1].pos) == 4


# === LLM 策略咨询 ===


def test_strategy_prompt_only_on_first_round_of_day(payload_factory, monkeypatch):
    """每天第一个回合生成一次prompt，其余回合为空（节省LLM额度）"""
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)

    prompt = _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {})
    assert "回合 1" in prompt
    assert "白天" in prompt

    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=2)), {}) == ""
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=131)), {})
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=132)), {}) == ""


def test_strategy_prompt_can_be_disabled(payload_factory, monkeypatch):
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", False)
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {}) == ""


def test_strategy_prompt_uses_previous_llm_reply(payload_factory, monkeypatch):
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    turn = Turn.load(payload_factory(round_no=131))
    prompt = _generate_strategy_prompt(turn, {"llmResp": "优先升级火箭发射台"})
    assert "优先升级火箭发射台" in prompt


def test_decide_returns_commands_and_prompt(payload_factory):
    """decide 返回 (指令字典, prompt)"""
    commands, prompt = decide(payload_factory(round_no=1))
    assert isinstance(commands, dict)
    assert isinstance(prompt, str)


# === 鲁棒性 ===


def test_decide_without_units(payload_factory):
    """没有任何可控制角色时返回空指令"""
    commands, _ = decide(payload_factory(station=None, round_no=71))
    assert commands == {}


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少 docs/request.txt 真实报文样例")
def test_decide_with_real_sample():
    """用真实报文跑一遍，校验指令格式合法"""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    commands, prompt = decide(payload)

    assert isinstance(prompt, str)
    for role_id, command in commands.items():
        assert role_id.isdigit()
        assert "action" in command
        if "targetPos" in command:
            assert isinstance(command["targetPos"], list)
            for pos in command["targetPos"]:
                assert set(pos) == {"x", "y"}
        if command["action"] == "attack":
            assert isinstance(command["controllerId"], str)
