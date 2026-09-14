"""决策模块测试（设计文档 3.5 节）"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent.brain as brain
from agent.brain import (
    GOLD_FLUSH_TOWERS,
    LLM_MAX_WALLS,
    LLM_MIN_TOWERS,
    LLM_PLAN_DEFAULT,
    LLM_PLAN_TEMPLATE,
    LLM_PROMPT_PER_DAY,
    SELL_BATCH,
    STONE_BATCH,
    TASK_END_MARKER,
    TASK_FILE_END,
    TASK_FILE_MARKER,
    TASK_MARKER,
    TASK_PROBE_MARKER,
    TOWER_LOADOUT,
    UPGRADE_GOLD,
    WEAPON_UPGRADE_VOUCHER,
    LlmPlan,
    _calc_tower_sites,
    _calc_wall_order,
    _generate_strategy_prompt,
    _llm_plan,
    _pair_controllers_and_weapons,
    _reserved_build_sites,
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
    DEFENDER_TASK_2,
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


def test_tower_sites_prefer_enemy_side(payload_factory, role_factory):
    """塔位优先罩住敌方来路

    回归：以前塔位只按"朝地图内侧的空间大小"排，敌人在哪个方向完全没参考
    （复盘里"塔位未按敌方路径规划，是否覆盖来路无法确认"）。
    """
    payload = payload_factory(
        roles=[role_factory(10010, WORKER, 9, 25, backPackCapability=100)],
        enemies=[role_factory(20010, WORKER, 3, 25)],  # 敌方单位在基地左侧
    )
    sites = _calc_tower_sites(Turn.load(payload))

    # 左侧是敌方来路，第一座塔建在左侧；看不到敌人时第一座塔在右侧
    assert sites[0] == Pos(9, 23)
    assert _calc_tower_sites(Turn.load(payload_factory()))[0] == Pos(12, 23)


def test_tower_sites_accept_llm_preferred_side(payload_factory):
    """塔位接受 LLM 计划指定的布防方位（建议与指令生成器共用同一决策函数）"""
    turn = Turn.load(payload_factory())

    assert _calc_tower_sites(turn, "left")[0] == Pos(9, 23)
    assert _calc_tower_sites(turn, "up")[0] == Pos(10, 25)
    # 不带偏好时保持原顺序
    assert _calc_tower_sites(turn, None) == _calc_tower_sites(turn)


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


def test_wall_order_follows_enemy_side(payload_factory, role_factory):
    """围墙按敌方来路排序：先封来路，把进攻路线压进炮塔射程

    回归：围墙顺序固定为"上->左->下->右"，敌人在哪一侧都先砌背面，正面一直
    空着（复盘里"防守方 0 段围墙、正面无任何阻挡"，以及"用墙把来路压缩进
    塔射程"）。
    """
    payload = payload_factory(
        enemies=[role_factory(20010, WORKER, 3, 24)],  # 敌方单位在基地正左方
    )
    order = _calc_wall_order(Turn.load(payload))

    # 来路在左侧：第一段围墙砌在基地左边（x = xmin-2 = 8）
    assert order[0] == Pos(8, 25)
    assert Pos(8, 22) in order  # 左边整条都在队列里
    assert order.index(Pos(8, 25)) < order.index(Pos(13, 26))

    # 看不到敌方单位时顺序不变（上边先砌），策略与改造前完全一致
    assert _calc_wall_order(Turn.load(payload_factory()))[0] == Pos(13, 26)


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
    # 目标本身就是建造点时（走去施工）也不许站在别的建造点上：站上去会把
    # 那一格占住，另一名工人这一回合就建不成（issue #24 的"建造指令空转"）；
    # 建造点之外一处也站不下时才退回旧行为（见
    # `test_walk_to_tower_skips_other_build_sites`）
    assert Pos(13, 23) in _calc_wall_order(turn)
    assert Pos(13, 23) not in _valid_stand_cells(turn, worker, Pos(12, 23), set())

    # 夜晚不施工，建造点可以正常站人（操控武器时站位更自由）
    night = Turn.load(payload_factory(
        round_no=71,
        gold=0,
        roles=[role_factory(10010, WORKER, 20, 20, backPackCapability=100)],
    ))
    assert Pos(12, 23) in _valid_stand_cells(
        night, night.workers()[0], Pos(11, 22), set(),
    )


def test_walk_to_tower_skips_other_build_sites(payload_factory, role_factory):
    """走去施工时不在别的建造点上落脚，实在无处落脚才放行

    回归：复盘里"R3 下了 rocket(30,8) 的建造指令，R4 金币仍是 50、塔也没出现"
    ——目标本身是建造点时旧实现会整体放行建造点，角色于是顺路站到别的塔位/
    墙位上，另一名工人这一回合就建不成。
    """
    tower = Pos(12, 23)
    payload = payload_factory(
        gold=0,
        roles=[role_factory(10010, WORKER, 14, 25, backPackCapability=100)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]

    stands = _valid_stand_cells(turn, worker, tower, set())
    assert stands  # 建造点之外还有落脚点，不会因为过滤建造点而无处可去
    assert all(pos not in _reserved_build_sites(turn) for pos in stands)
    assert Pos(13, 23) in _calc_wall_order(turn)  # 右侧围墙的建造位
    assert Pos(13, 23) not in stands

    # 建造点之外一处也站不下时退回旧行为：允许站在建造点上，
    # 否则角色会因为"相邻格全是建造位"而永远建不起来
    crowded = payload_factory(
        gold=0,
        roles=[
            role_factory(10010, WORKER, 14, 25, backPackCapability=100),
            role_factory(40001, WALL, 11, 22),
            role_factory(40002, WALL, 12, 22),
            role_factory(40003, WALL, 12, 24),
            role_factory(40004, WALL, 13, 22),  # 围墙圈的入口，也不是建造点
        ],
    )
    turn = Turn.load(crowded)
    assert _valid_stand_cells(turn, turn.workers()[0], tower, set()) == [
        Pos(13, 23), Pos(13, 24),
    ]


def test_second_worker_skips_build_when_gold_runs_out(
    payload_factory, role_factory,
):
    """手里只够一座塔的钱时，第二个工人不再下注定失败的建造指令

    回归：同一回合的金币要等结算才扣，`turn.gold` 一直是回合开始时的余额，
    两名工人会各下一条 build，后一条白下（复盘里的"下了建造指令、下回合
    金币没扣、塔也没出现"）。
    """
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,  # 只够一座塔
        roles=[
            role_factory(10010, WORKER, 11, 22, backPackCapability=100),
            role_factory(10012, WORKER, 9, 22, backPackCapability=100),
        ],
        zones=[(STONE_MINE, 8, 21), (IRON_MINE, 10, 21)],
    )
    commands, _ = decide(payload)

    # 第一名工人把唯一一座塔建起来，金币被这一条指令占满
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": GATLING,
    }
    # 第二名工人改去采集（经济分工里它负责铁/铜），不再空下一条 build
    assert commands["10012"] == {
        "action": "collect",
        "targetPos": [{"x": 10, "y": 21}],
    }


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


# === 工人分工与经济循环（issue #15） ===


def test_worker_falls_through_to_gather_when_tower_unreachable(
    payload_factory, role_factory,
):
    """塔位走不通时工人转去采集，而不是空手过一回合

    回归：对战复盘里工人连续多回合只下 move、金币零增长——塔位走不通时
    决策在建造分支直接返回，采集/建墙/交易一条都排不上。
    """
    worker_pos = Pos(5, 23)
    stone_pos = Pos(4, 24)
    # 把工人围住（只留石矿那一格），三座塔位它一座也够不着
    enclosure = [pos for pos in get_neighbors(worker_pos) if pos != stone_pos]
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[
            role_factory(
                10010, WORKER, worker_pos.x, worker_pos.y, backPackCapability=100,
            ),
            *[
                role_factory(40000 + index, WALL, pos.x, pos.y)
                for index, pos in enumerate(enclosure)
            ],
        ],
        zones=[(STONE_MINE, stone_pos.x, stone_pos.y)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "collect",
        "targetPos": [{"x": stone_pos.x, "y": stone_pos.y}],
    }


def test_workers_split_mine_types(payload_factory, role_factory):
    """两名工人按矿种分工：一名采石材（围墙），一名采铁/铜（换金币）

    回归：以前两名工人只按"石矿 -> 铁矿 -> 铜矿"的顺序就近采，只要地图上
    还剩别的石矿，负责变现的那名工人就会被石矿带着跑，铁/铜永远排不上，
    金币整局没有产出（对战复盘里的"金币零增长"）。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(10012, WORKER, 20, 23, backPackCapability=100),
        ],
        # (35,5) 是远处那座没被第1名工人认领的石矿
        zones=[(STONE_MINE, 4, 24), (STONE_MINE, 35, 5), (IRON_MINE, 20, 22)],
    )
    commands, _ = decide(payload)

    # 第1名工人就地采石材
    assert commands["10010"] == {
        "action": "collect",
        "targetPos": [{"x": 4, "y": 24}],
    }
    # 第2名工人不去抢石矿，就地采铁（矿石换金币这条收入线）
    assert commands["10012"] == {
        "action": "collect",
        "targetPos": [{"x": 20, "y": 22}],
    }


def test_economy_worker_sells_ore_before_walls_done(
    payload_factory, role_factory,
):
    """围墙还没建完，负责经济的工人也会把矿石卖给小贩换金币

    回归：以前只有"围墙圈建完"或"没矿可采"时才会走到卖矿分支——这里地图上
    还有石矿，占着经济分工的工人却被派去采石，手头那批矿石一直变不成金币
    （对战复盘里的"金币连续多回合零增长"）。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(
                10012, WORKER, 20, 17, backPackCapability=100,
                backpack=[IRON_MINE] * SELL_BATCH,
            ),
        ],
        # 石矿在远处，负责采石的工人有活干；卖矿的工人不该被它带着走
        zones=[(STONE_MINE, 35, 5), (VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "sell",
        "name": IRON_MINE,
        "num": SELL_BATCH,
    }


def test_economy_worker_buys_upgrade_before_walls_done(
    payload_factory, role_factory,
):
    """三座武器都已建成、围墙还在施工时，富余金币就用来买武器升级券

    回归：以前买券也卡在"围墙建完"这个条件上，围墙施工期间金币只进不出。
    """
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=1,
        gold=UPGRADE_GOLD,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(10012, WORKER, 20, 17, backPackCapability=100),
            *[
                role_factory(10020 + index, kind, pos.x, pos.y, attackRange=4)
                for index, (kind, pos) in enumerate(zip(TOWER_LOADOUT, sites))
            ],
        ],
        zones=[(WEAPON_SHOP, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "buy",
        "name": WEAPON_UPGRADE_VOUCHER,
        "num": 1,
    }


def test_economy_worker_skips_vendor_trip_close_to_night(
    payload_factory, role_factory,
):
    """天黑前跑不到小贩就别出门：经济动作不能把角色拖在地图另一头

    夜晚的武器要有角色操控才会开火（任务书4.4节），所以只在天黑前还剩
    "往返路费"时才出发去卖矿。
    """
    payload = payload_factory(
        round_no=DAY_ROUNDS - 10,  # 距天黑还有11回合，来不及跑一趟(20,16)
        gold=0,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(
                10012, WORKER, 30, 20, backPackCapability=100,
                backpack=[IRON_MINE] * SELL_BATCH,
            ),
        ],
        zones=[(VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert "10012" not in commands


# === 空转兜底与首夜防线（issue #13） ===


def test_pioneer_without_task_gathers_instead_of_idling(
    payload_factory, role_factory,
):
    """既没有任务也没有武器时，开拓者就近采矿，而不是原地待着

    回归：复盘里"三个单位原地小步挪动、金币连续多回合冻结"——决策的每条
    分支都没目标时角色会整回合没有任何指令。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 5, 23, backPackCapability=40)],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "collect",
        "targetPos": [{"x": 4, "y": 24}],
    }


def test_idle_gather_keeps_pioneer_on_running_task(
    payload_factory, role_factory,
):
    """任务进行中的开拓者不被兜底支走（离开任务点周围一格会强制结束任务）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        zones=[(STONE_MINE, 15, 15)],
        phase_task="自进化类任务：请查询北京天气",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands


def test_idle_gather_waits_for_night_instead_of_mining(
    payload_factory, role_factory,
):
    """天黑前回防待命的角色不被兜底支去采矿（夜晚武器要有人操控才会开火）"""
    payload = payload_factory(
        round_no=DAY_ROUNDS - 3,  # 距天黑还有4回合
        gold=0,
        roles=[
            role_factory(10010, WORKER, 9, 25, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        zones=[(STONE_MINE, 9, 26)],  # 紧邻石矿，但回防优先
    )
    commands, _ = decide(payload)

    assert "10010" not in commands


def test_worker_builds_second_wall_before_night(
    payload_factory, role_factory,
):
    """天黑前不足两段围墙时，手里有石头就先补墙，而不是提前回防

    回归：复盘里首夜只有一座光塔、零段围墙，防线没有纵深；以前只要立起
    一段围墙就算"防线达标"，工人会被提前召回武器旁。
    """
    order = _calc_wall_order(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=DAY_ROUNDS - 3,  # 距天黑还有4回合
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, 13, 27, backPackCapability=100,
                backpack=[WALL_MATERIAL],
            ),
            # 已经立起的一段围墙（不在施工顺位首位，工人仍该去补更靠前的一段）
            role_factory(10020, WALL, order[-1].x, order[-1].y),
            role_factory(10030, GATLING, 9, 24, attackRange=4),
        ],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
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


def test_pioneer_waits_when_task_point_is_crowded(payload_factory, role_factory):
    """有任务可领却这一回合走不动时，开拓者原地等，不退回去跟随武器塔

    回归：走向任务点走不通时旧实现会掉到"跟随武器塔"，把开拓者带回基地，
    下一回合再往外走——来回打转，复盘里"开拓者整局在基地附近徘徊、
    从未靠近任务点"，两处任务点合计160分+160金币一直没人领。
    """
    task_pos = Pos(14, 14)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10011, PIONEER, 10, 12, backPackCapability=40),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        tasks=[(task_pos.x, task_pos.y)],
        # 任务点周围一格全是矿区（不可通行），这一回合开拓者挤不进去
        zones=[(COPPER_MINE, pos.x, pos.y) for pos in get_neighbors(task_pos)],
    )
    commands, _ = decide(payload)

    # 原地等下一个回合，而不是朝基地方向的武器塔移动
    assert "10011" not in commands


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


def test_pioneer_accepts_task_from_second_cell(payload_factory, role_factory):
    """任务点2占据两格（任务书4.6.2）：站在另一格旁边同样能领任务

    回归：以前只按报文给出的那一格算距离，开拓者站在任务点另一格旁边
    （距离 2）会继续绕路，白白多跑几个回合。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        team_type="defender",
        roles=[role_factory(10011, PIONEER, 28, 18, backPackCapability=40)],
        tasks=[(26, 17)],
        zones=[(DEFENDER_TASK_2, 26, 17), (DEFENDER_TASK_2, 27, 17)],
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {"action": "acceptTask"}


def test_pioneer_holds_task_from_second_cell(payload_factory, role_factory):
    """任务点在另一格旁边时原地待命，不再来回挪位"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        team_type="defender",
        roles=[role_factory(10011, PIONEER, 28, 18, backPackCapability=40)],
        tasks=[(26, 17)],
        zones=[(DEFENDER_TASK_2, 26, 17), (DEFENDER_TASK_2, 27, 17)],
        phase_task="自进化类任务：请阅读task_1_beijing.md",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands


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


def test_sandbox_command_gathers_clues_in_one_shot(
    payload_factory, role_factory,
):
    """一条沙盒命令同时带上兜底搜索与目录诊断，减少逐次试错回合

    复盘里敌方"用错鉴权头→401、补参数又缺字段→400"逐次试错，白丢好几个
    回合；这里把"读任务文件 + 路径不对时按文件名再找 + 列出沙盒目录"合成
    一条命令，一次就能拿到更多线索。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读tasks/task_1_beijing.md",
    )
    command = sandbox_command(payload)

    assert 'cat -- "tasks/task_1_beijing.md"' in command
    # 描述里的路径读不到时按文件名在沙盒里再找一次（只按文件名，不带目录）
    assert 'find . -maxdepth 3 -type f -name "task_1_beijing.md"' in command
    # 诊断信息排在答案结束标记之后，不会被当成答案提交
    assert command.index(TASK_MARKER) < command.index(TASK_END_MARKER)
    assert command.index(TASK_END_MARKER) < command.index("ls -a")


def test_sandbox_probes_task_dir_when_description_has_no_file(
    payload_factory, role_factory,
):
    """描述里没有文件名时：先探测沙盒任务目录，认出文件再读它作答

    回归：任务描述只写"按沙盒里的任务说明作答"这类话时，客户端不知道该读
    哪个文件，开拓者会卡在任务点拿到一堆无关输出，整个任务周期（15 回合）
    空转（复盘里的"接取任务后反复答非所问、最后放弃"）。
    """
    phase_task = "请按沙盒里的任务说明作答"
    token = _task_token(phase_task)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 20, 20, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )

    # 第一回合：描述里没有文件名，下发探测命令列出沙盒里的任务文件
    command = sandbox_command(payload)
    assert TASK_PROBE_MARKER in command
    assert 'find "/tmp/selfEvolutionTask"' in command
    # 探测输出带的是探测标记，不会被 `_task_answer` 当成答案
    assert TASK_MARKER not in command

    # 探测结果里认出了任务文件：改用完整路径读它，并带上本任务标识
    payload["lastCmdResult"] = (
        f"[exitCode:0]\n{TASK_PROBE_MARKER}{token}\n"
        "/tmp/selfEvolutionTask/task_1_alpha.md\n"
    )
    command = sandbox_command(payload)
    assert 'cat -- "/tmp/selfEvolutionTask/task_1_alpha.md"' in command
    assert TASK_MARKER in command

    # 目录清单本身不是答案，不能被提交上去
    commands, _ = decide(payload)
    assert commands.get("10011", {}).get("action") != "submitAnswer"


def test_sandbox_answer_ignores_diagnostics_after_end_marker(
    payload_factory, role_factory,
):
    """沙盒输出里结束标记之后的诊断信息不会被当成答案"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = (
        _sandbox_result(phase_task, "北京 晴 25摄氏度")
        + f"\n{TASK_END_MARKER}\ntask_1_beijing.md\ntask_2_shanghai.md\n"
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": "北京 晴 25摄氏度",
    }


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


def test_sandbox_command_dumps_task_files_for_cache(
    payload_factory, role_factory,
):
    """读任务文件时顺带把任务目录里的文件都读回来，供后续任务直接作答

    任务书5.3节要求把探索结果做成 SOP/SKILL（自进化），积分又按"完成回合 -
    接取回合"倒扣：一次读回来缓存住，下一个任务点领到同一个任务就能立刻交卷。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读task_1_beijing.md",
    )
    command = sandbox_command(payload)

    assert TASK_FILE_MARKER in command
    assert TASK_FILE_END in command
    assert 'find "/tmp/selfEvolutionTask"' in command
    # 整段 dump 排在答案结束标记之后，不会被当成当前任务的答案提交
    assert command.index(TASK_END_MARKER) < command.index(TASK_FILE_MARKER)


def test_task_cache_skips_failed_reads_and_keeps_first_answer():
    """缓存不记"文件读不到"的输出，也不会被后来的输出覆盖"""
    brain._remember_task_files(
        "[exitCode:1]\n[TASK_FILE]/tmp/x/task_1_alpha.md\n"
        "cat: /tmp/x/task_1_alpha.md: No such file or directory\n[TASK_EOF]\n"
    )
    assert brain._TASK_ANSWER_CACHE == {}

    brain._remember_task_files(
        "[TASK_FILE]/tmp/x/task_1_alpha.md\nalpha-answer\n[TASK_EOF]\n"
    )
    brain._remember_task_files(
        "[TASK_FILE]/tmp/x/task_1_alpha.md\n乱码\n[TASK_EOF]\n"
    )
    assert brain._TASK_ANSWER_CACHE == {"task_1_alpha.md": "alpha-answer"}


def test_pioneer_answers_from_cached_task_file(payload_factory, role_factory):
    """沙盒里提前读回来的任务文件命中时，接取后立刻交卷

    回归：复盘里敌方靠答案缓存两次提交各拿 155 分（任务奖励 80 +
    5×标准回合数15/(完成回合-接取回合)1），我们每次都重新读一遍沙盒，
    完成回合差至少 2 回合，分数被白白扣掉。
    """
    phase_task = "请阅读task_2_beijing.md"
    payload = payload_factory(
        round_no=2,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    # 上一个任务期间沙盒把任务目录里的文件都读回来过（含本次要用的这份）
    payload["lastCmdResult"] = (
        "[exitCode:0]\n"
        "[TASK]上一个任务\n"
        "[TASK_FILE]/tmp/selfEvolutionTask/task_1_alpha.md\n"
        "alpha-answer\n"
        "[TASK_EOF]\n"
        "[TASK_FILE]/tmp/selfEvolutionTask/task_2_beijing.md\n"
        "北京 晴 25摄氏度\n"
        "[TASK_EOF]\n"
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": "北京 晴 25摄氏度",
    }
    # 已经有缓存答案，不必再花一个来回执行沙盒命令
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


def test_strategy_prompt_uses_daily_quota(payload_factory, monkeypatch):
    """每个游戏日把 LLM 额度用满：当日前3个回合各咨询一次，之后为空

    回归：以前每天只在第 1 回合咨询一次，复盘里"R2、R3 请求中 prompt 为空，
    决策循环仅在 R1 调用了 LLM、指令退化为无目标移动"。
    """
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    assert LLM_PROMPT_PER_DAY == 3  # 接口文档：每队每个游戏日 3 次额度

    for round_no in range(1, LLM_PROMPT_PER_DAY + 1):
        prompt = _generate_strategy_prompt(
            Turn.load(payload_factory(round_no=round_no)), {},
        )
        assert f"回合 {round_no}" in prompt
        assert "白天" in prompt

    # 当日额度用完后不再请求（errorCode=5 是额度超限）
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=4)), {}) == ""

    # 次日首回合额度重置
    for round_no in range(131, 131 + LLM_PROMPT_PER_DAY):
        turn = Turn.load(payload_factory(round_no=round_no))
        assert _generate_strategy_prompt(turn, {})
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=134)), {}) == ""


def test_strategy_prompt_can_be_disabled(payload_factory, monkeypatch):
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", False)
    assert _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {}) == ""


def test_strategy_prompt_uses_previous_llm_reply(payload_factory, monkeypatch):
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    turn = Turn.load(payload_factory(round_no=131))
    prompt = _generate_strategy_prompt(turn, {"llmResp": "优先升级火箭发射台"})
    assert "优先升级火箭发射台" in prompt


def test_strategy_prompt_asks_for_plan_line(payload_factory, monkeypatch):
    """prompt 给出本回合既定计划并要求回一行可解析的 PLAN（建议与指令合一）"""
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    prompt = _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {})

    assert "本回合既定计划" in prompt
    assert LLM_PLAN_TEMPLATE in prompt


def test_strategy_prompt_includes_task_points(payload_factory, monkeypatch):
    """prompt 带上任务点坐标/奖励，LLM 的建议才能落到具体任务上"""
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    turn = Turn.load(payload_factory(round_no=1, tasks=[(23, 14)]))
    prompt = _generate_strategy_prompt(turn, {})

    assert "(23,14)" in prompt


# === LLM 计划落地（issue #14） ===


def test_llm_plan_defaults_without_plan_line():
    """没有 LLM 回复、或回复里没有 PLAN 行时退回默认计划（等价改造前策略）"""
    assert _llm_plan({}) == LLM_PLAN_DEFAULT
    assert _llm_plan({"llmResp": "优先升级火箭发射台"}) == LLM_PLAN_DEFAULT


def test_llm_plan_parses_known_fields():
    """PLAN 行里的已知字段被解析成计划，未知字段忽略"""
    plan = _llm_plan({
        "llmResp": "先补炮塔再铺墙\nPLAN: tower=2 wall=2 upgrade=off defend=left",
    })

    assert plan == LlmPlan(tower=2, wall=2, upgrade=False, defend="left")


def test_llm_plan_clamps_out_of_range_values():
    """越界与非法值被钳制：LLM 不能把塔数调成 0 让白天完全不设防"""
    plan = _llm_plan({"llmResp": "PLAN: tower=0 wall=99 upgrade=maybe defend=左上"})

    assert plan.tower == LLM_MIN_TOWERS
    assert plan.wall == LLM_MAX_WALLS
    assert plan.upgrade is True
    assert plan.defend is None


def test_llm_plan_ignores_unfilled_template():
    """LLM 照抄模板占位符时不会被解析成 tower=1 这类误读（尖括号挡住）"""
    assert _llm_plan({"llmResp": LLM_PLAN_TEMPLATE}) == LLM_PLAN_DEFAULT


def test_llm_plan_skips_echoed_template_before_real_plan():
    """建议里先复述模板、再给真正的计划时，以真正的那一行为准"""
    reply = f"请按 {LLM_PLAN_TEMPLATE} 输出\nPLAN: tower=2 upgrade=off"
    plan = _llm_plan({"llmResp": reply})

    assert plan.tower == 2
    assert plan.upgrade is False


def test_llm_plan_tower_cap_stops_extra_tower(payload_factory, role_factory):
    """计划里 tower=2 时不再开工第 3 座塔，金币不被继续压在塔上"""
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[
            role_factory(10010, WORKER, 9, 22, backPackCapability=100),
            role_factory(10020, GATLING, sites[0].x, sites[0].y, attackRange=4),
            role_factory(10030, RAILGUN, sites[1].x, sites[1].y, attackRange=6),
        ],
    )

    # 默认计划: 金币够就直接开工第 3 座塔（火箭发射台）
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": sites[2].x, "y": sites[2].y}],
        "name": ROCKET,
    }

    # 计划只要求 2 座塔: 不再开工第 3 座
    payload["llmResp"] = "PLAN: tower=2"
    commands, _ = decide(payload)
    assert "10010" not in commands


def test_llm_plan_wall_quota_builds_walls_before_trading(
    payload_factory, role_factory,
):
    """计划里 wall=2 时经济工人先铺围墙，而不是先把矿石卖掉"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(
                10012, WORKER, 13, 27, backPackCapability=100,
                backpack=[WALL_MATERIAL] * SELL_BATCH,
            ),
        ],
        zones=[(VENDOR, 13, 28)],
    )

    # 默认计划: 围墙没建完也先把矿石变现（金币滚动起来）
    commands, _ = decide(payload)
    assert commands["10012"] == {
        "action": "sell",
        "name": WALL_MATERIAL,
        "num": SELL_BATCH,
    }

    # 计划要求先铺 2 段围墙: 手里的石材先用于施工
    payload["llmResp"] = "PLAN: wall=2"
    commands, _ = decide(payload)
    assert commands["10012"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }


def test_llm_plan_upgrade_off_keeps_gold(payload_factory, role_factory):
    """计划里 upgrade=off 时当天不买武器升级券，金币留作他用"""
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

    payload["llmResp"] = "PLAN: upgrade=off"
    commands, _ = decide(payload)
    assert "10010" not in commands


def test_llm_plan_defend_side_builds_tower_on_that_side(
    payload_factory, role_factory,
):
    """计划里的 defend 方位决定第一座塔建在哪一侧"""
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[role_factory(10010, WORKER, 8, 23, backPackCapability=100)],
    )
    payload["llmResp"] = "PLAN: defend=left"
    commands, _ = decide(payload)

    # 工人站在左侧塔位旁，计划指定左侧布防时直接开工
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 9, "y": 23}],
        "name": GATLING,
    }


def test_decide_returns_commands_and_prompt(payload_factory):
    """decide 返回 (指令字典, prompt)"""
    commands, prompt = decide(payload_factory(round_no=1))
    assert isinstance(commands, dict)
    assert isinstance(prompt, str)


# === 金币闲置熔断（issue #12） ===


def _two_towers_payload(payload_factory, role_factory, gold: int) -> dict:
    """构造"两座塔已建完、金币闲置"的局面，工人站在第 3 座塔位旁"""
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    return payload_factory(
        round_no=1,
        gold=gold,
        roles=[
            role_factory(10010, WORKER, 9, 22, backPackCapability=100),
            role_factory(10020, GATLING, sites[0].x, sites[0].y, attackRange=4),
            role_factory(10030, RAILGUN, sites[1].x, sites[1].y, attackRange=6),
        ],
    )


def test_gold_flush_overrides_tower_cap(payload_factory, role_factory):
    """金币够再建两座塔时把塔数配额提到满编，不再持币空转

    回归：复盘里"金币 75 只花 25 建 1 座塔，余下 50 连续三个回合冻结"。
    """
    payload = _two_towers_payload(payload_factory, role_factory, GOLD_FLUSH_TOWERS)
    sites = _calc_tower_sites(Turn.load(payload))

    # 计划把塔数压到 2 座，但手里还攥着两座塔的钱：第 3 座照建
    payload["llmResp"] = "PLAN: tower=2"
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": sites[2].x, "y": sites[2].y}],
        "name": ROCKET,
    }


def test_gold_flush_keeps_plan_cap_below_threshold(
    payload_factory, role_factory,
):
    """金币不到熔断线时仍然听计划的，留钱买升级券"""
    payload = _two_towers_payload(
        payload_factory, role_factory, GOLD_FLUSH_TOWERS - 1,
    )
    payload["llmResp"] = "PLAN: tower=2"

    commands, _ = decide(payload)
    assert "10010" not in commands


# === 防守方围墙配额（issue #22） ===


def test_defender_plan_keeps_wall_quota(payload_factory, role_factory):
    """计划把围墙压到 0 段时，防守方仍按下限先铺墙

    回归：防守方整天 0 段围墙、正面毫无阻挡，机器人直接贴脸打基地，
    而同局的进攻方反倒把来路封得严严实实。
    """

    def _payload(team_type: str) -> dict:
        payload = payload_factory(
            round_no=1,
            gold=0,
            team_type=team_type,
            roles=[
                role_factory(10010, WORKER, 5, 23, backPackCapability=100),
                role_factory(
                    10012, WORKER, 13, 27, backPackCapability=100,
                    backpack=[WALL_MATERIAL] * SELL_BATCH,
                ),
            ],
            zones=[(VENDOR, 13, 28)],
        )
        payload["llmResp"] = "PLAN: wall=0"
        return payload

    # 防守方：手里的石材先砌成围墙，矿石不急着变现
    commands, prompt = decide(_payload("defender"))
    assert commands["10012"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }
    # LLM 看到的配额也是执行层真正要铺的段数（建议与指令同源）
    assert "优先铺围墙 1 段" in prompt

    # 进攻方不受下限影响：仍按计划把矿石卖掉换金币
    commands, _ = decide(_payload("challenger"))
    assert commands["10012"] == {
        "action": "sell",
        "name": WALL_MATERIAL,
        "num": SELL_BATCH,
    }

    # 地图上没有石材来源时下限失效：矿石照卖，经济线不被"还差一段墙"扣住
    no_stone = _payload("defender")
    no_stone["teamOur"]["roles"][1]["backpack"] = [COPPER_MINE] * SELL_BATCH
    commands, _ = decide(no_stone)
    assert commands["10012"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": SELL_BATCH,
    }


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
