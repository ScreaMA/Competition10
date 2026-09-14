"""决策模块测试（设计文档 3.5 节）"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import agent.brain as brain
from agent.brain import (
    DAY_PLAN_LIMIT,
    ECONOMY_MINE_ORDER,
    FIXER_STOCK,
    FIXER_STOCK_FIRST,
    GOLD_FLUSH_TOWERS,
    LLM_MAX_WALLS,
    LLM_MIN_TOWERS,
    LLM_PLAN_DEFAULT,
    LLM_PLAN_TEMPLATE,
    LLM_PROMPT_PER_DAY,
    LOW_GOLD_THRESHOLD,
    MEDICINE,
    MEDICINE_HP,
    MEDICINE_HP_NIGHT,
    MINERAL_SELL_THRESHOLD,
    PIONEER_WEAPON_VOUCHERS,
    QUEUE_TARGETS,
    SELL_BATCH,
    STATION_UPGRADE_VOUCHER,
    STONE_BATCH,
    STONE_PLAN_MAX,
    STONE_RESERVE_MIN,
    TASK_API_FAIL_LIMIT,
    TASK_API_FAIL_MARKER,
    TASK_API_KEEP,
    TASK_DATA_MARKER,
    TASK_DOC_MARKER,
    TASK_END_MARKER,
    TASK_FILE_END,
    TASK_FILE_EXTS,
    TASK_FILE_MARKER,
    TASK_LOOP_LIMIT,
    TASK_MARKER,
    TASK_PROBE_LIMIT,
    TASK_PROBE_MARKER,
    TASK_ROOTS,
    TASK_SCAN_MARKER,
    TASK_SOLUTION_END,
    TASK_SOLUTION_MARKER,
    TASK_SUBMIT_LIMIT,
    TASK_TIMEOUT,
    TOWER_LOADOUT,
    UPGRADE_GOLD,
    WALL_BUILD_ROUNDS,
    WALL_FIRST_ROUND,
    WALL_FIXER,
    WALL_FIXER_GOLD,
    WALL_STONE_COST,
    WALL_UPGRADE_GOLD,
    WALL_UPGRADE_VOUCHER,
    WEAPON_UPGRADE_VOUCHER,
    WEAPON_UPGRADE_VOUCHER2,
    LlmPlan,
    _base_layout,
    _calc_tower_sites,
    _calc_wall_order,
    _closest_step,
    _day_plan,
    _generate_strategy_prompt,
    _gold_left,
    _llm_plan,
    _mine_order,
    _pair_controllers_and_weapons,
    _plan_extras,
    _plan_summary,
    _repair_plan,
    _nearest_zone,
    _shop_is_last_stop,
    _reserved_build_sites,
    _step_toward,
    _stone_demand,
    _task_executor,
    _task_token,
    _tower_site_brief,
    _tower_sites_reachable,
    _valid_stand_cells,
    _wall_holes,
    _wall_ring,
    _work_queue,
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
    ROUNDS_PER_DAY,
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


def _solution_result(
    phase_task: str,
    task_file: str,
    body: str,
    url: str = "http://localhost:8899/heritage?city=beijing",
) -> str:
    """构造一条"执行器取数成功"的沙盒输出（答案区只有取到的数据）

    答案区里必须先有取数证据（`TASK_DATA_MARKER`），`[SOLUTION]` 段里才是
    要提交的答案；诊断信息排在 `TASK_END_MARKER` 之后。
    """
    return _sandbox_result(
        phase_task,
        f"{TASK_DATA_MARKER} {url} => {len(body)}\n"
        f"{TASK_SOLUTION_MARKER}{task_file}\n{body}\n{TASK_SOLUTION_END}\n"
        f"{TASK_END_MARKER}\n"
        "pwd\n/\ntask_1_beijing.md\n",
    )


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
    """建造顺序包含全部三种武器，且射程最远的火箭发射台先落地

    回归：复盘里敌方开局就建射程 10 的火箭发射台，我方先建的却是射程 3 的
    加特林，机器人走到基地跟前才开始挨打（"武器优先级表（rocket 优先）"）。
    """
    assert TOWER_LOADOUT == (ROCKET, RAILGUN, GATLING)


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
    assert Pos(13, 24) not in order  # 入口开在威胁最小一侧的中间格（V4）
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
        "name": ROCKET,
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
        "name": ROCKET,
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
        role_factory(10030, GATLING, 9, 23, attackRange=3),
    ]
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": ROCKET,
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
    """背包里有石头且站在建造点旁时建造围墙（一段墙只要石头*1）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(
                10012, WORKER, 12, 26, backPackCapability=100,
                backpack=[WALL_MATERIAL] * WALL_STONE_COST,
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
        "name": ROCKET,
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
    assert Pos(13, 24) in stands  # 唯一可落脚的那格仍然可用

    # 建造点之外一处也站不下时退回旧行为：允许站在建造点上，
    # 否则角色会因为"相邻格全是建造位"而永远建不起来
    crowded = payload_factory(
        gold=0,
        roles=[
            role_factory(10010, WORKER, 14, 25, backPackCapability=100),
            role_factory(40001, WALL, 11, 22),
            role_factory(40002, WALL, 12, 22),
            role_factory(40003, WALL, 12, 24),
            role_factory(40004, WALL, 13, 22),
            # V4 的入口在 (13,24)：把它也堵上，非建造点的落脚位就一个不剩
            role_factory(40005, WALL, 13, 24),
        ],
    )
    turn = Turn.load(crowded)
    stands = _valid_stand_cells(turn, turn.workers()[0], tower, set())
    assert stands == [Pos(13, 23)]  # 退回旧行为：允许站在建造点上


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
        "name": ROCKET,
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
                             zones=(), backpack=(), capacity: int = 100,
                             gold: int = 0) -> dict:
    """构造“围墙已建完”的局面（默认金币见底，用来观察经济回路）"""
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
    return payload_factory(gold=gold, roles=walls, zones=list(zones))


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
    """防线与塔都齐了时，零散石材也卖掉（攒批没有意义）

    V4/S2：围墙配额铺满、武器塔也建满之后，背包里的石材再攒批只是让金币继续
    躺着（复盘里 stone 从 1 块堆到 3 块、金币从 R6 起恒 0 到 R17）。
    手里还有建造需求时才有攒批的意义（见 `_build_backlog`）。
    """
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=LOW_GOLD_THRESHOLD,
        worker_pos=Pos(20, 17),
        backpack=[WALL_MATERIAL] * (SELL_BATCH - 1),
        zones=[(VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "sell",
        "name": WALL_MATERIAL,
        "num": SELL_BATCH - 1,
    }


def test_trade_sells_ore_as_soon_as_gold_runs_out(
    payload_factory, role_factory,
):
    """金币见底时手里有几块卖几块，不再等凑够一批

    回归：复盘里 R6 建完第三座塔后 gold=0 冻结 13 个回合，背包里的 stone:6
    因为凑不满 SELL_BATCH 一直没卖出去，建造/升级/买券跟着一起停摆。
    """
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=LOW_GOLD_THRESHOLD - 1,
        worker_pos=Pos(20, 17),
        backpack=[WALL_MATERIAL] * (SELL_BATCH - 1),
        zones=[(VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "sell",
        "name": WALL_MATERIAL,
        "num": SELL_BATCH - 1,
    }


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


def test_worker_sells_ore_before_gathering_when_gold_runs_out(
    payload_factory, role_factory,
):
    """金币见底时先把手头的矿石变现，而不是继续往背包里堆

    回归：复盘里 R6 建完第三座塔后 gold=0 冻结 13 个回合，工人每回合都在
    采矿，背包里的矿石却一块都没卖出去过（卖矿被"攒够一批"和采集分支堵住），
    建造/升级/买券跟着一起停摆。
    """
    payload = payload_factory(
        round_no=1,
        gold=LOW_GOLD_THRESHOLD - 1,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(
                10012, WORKER, 20, 17, backPackCapability=100,
                backpack=[IRON_MINE] * 2,
            ),
        ],
        # 身边就有铁矿可采，也仍然该先把背包里那两块卖掉
        zones=[(STONE_MINE, 35, 5), (IRON_MINE, 21, 18), (VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "sell",
        "name": IRON_MINE,
        "num": 2,
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


def test_pioneer_without_task_never_gets_worker_only_collect(
    payload_factory, role_factory,
):
    """既没有任务也没有武器时，开拓者不会被派去采矿（collect 只有工人能用）

    回归：PK590881 的 R11–R15 连续 5 个回合
    `[COMMAND_ERROR] role 20011 (pioneer) wants collect, but only worker can
    do this action`——任务书 4.4 的指令表里 collect 的可用角色只有工人，
    旧实现把"就近采一铲"当成空转兜底发给了开拓者，矿一块没采到（那条指令
    整条被驳回），这个回合也一起白搭。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 5, 23, backPackCapability=40)],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    command = commands.get("10011")
    assert command is None or command["action"] != "collect"


def test_task_pioneer_does_not_mine_beside_task_point(
    payload_factory, role_factory,
):
    """任务点旁的开拓者不会被派去采石（PK590881 的 R11–R15 就是这个形态）

    任务点 (23,14) 旁边正好压着一座石矿 (24,14)，旧实现认为"采集不移动、
    任务照旧有效"，于是每回合都发一条 collect→(24,14)：判题系统整条驳回，
    开拓者白等 5 个回合，任务窗口被耗尽、答案始终没交上去。
    """
    payload = payload_factory(
        round_no=11,
        gold=0,
        roles=[role_factory(20011, PIONEER, 23, 14, backPackCapability=40)],
        tasks=[(23, 14)],
        zones=[(STONE_MINE, 24, 14)],
        phase_task="请阅读task_1_beijing.md，获取任务信息",
    )
    commands, _ = decide(payload)

    command = commands.get("20011")
    assert command is None or command["action"] != "collect"
    if command is not None:
        # 等答案期间只能留在任务点周围一格内（离开会强制结束任务）
        step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
        assert distance(step, Pos(23, 14)) <= 1


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

    # 任务期间可以在任务点周围一格内挪步（S1），但既不能被支走、也不能采矿
    # （collect 是工人专属动作，见 `_go_mine`）
    command = commands.get("10011")
    assert command is None or command["action"] == "move"
    if command is not None:
        step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
        assert distance(step, Pos(14, 14)) <= 1


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


def test_pioneer_waits_at_cooling_task_point(payload_factory, role_factory):
    """任务冷却期间白天守在任务点旁，不再回基地再折返

    回归：旧实现只在冷却剩 3 回合时才往外走，其余回合掉头回基地跟随武器塔
    （`test_pioneer_returns_to_weapon_before_night` 覆盖天黑前那一段）。
    任务点离基地十几格，一天来回一趟就是二十多个回合，两处任务点因此常常
    只赶得上一个——复盘里敌方 r14 交完第一个任务、r17 立刻接上第二个。
    """
    before = Pos(11, 22)
    task_pos = Pos(14, 14)
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        tasks=[(task_pos.x, task_pos.y, {"isValid": False, "coldDownRounds": 20})],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    # 走向冷却中的任务点守候，而不是朝基地里的武器塔移动
    assert distance(step, task_pos) < distance(before, task_pos)


def test_pioneer_returns_to_weapon_before_night(payload_factory, role_factory):
    """天黑前按返程路费提前退回基地：守任务点不能把夜晚的火力搭进去

    任务书4.4节：武器要有角色操控才会开火。任务点离基地 10 格时，
    最后 15 回合（路费 + 提前回防量）就不再往外守，直接回防。
    """
    before = Pos(14, 14)
    weapon_pos = Pos(9, 24)
    task_pos = Pos(14, 14)
    payload = payload_factory(
        round_no=DAY_ROUNDS - 2,  # 距天黑 3 回合，不够从任务点走回基地
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
    # 走向武器塔备战，而不是继续守在冷却中的任务点旁
    assert distance(step, weapon_pos) <= distance(before, weapon_pos)
    assert distance(step, task_pos) >= distance(before, task_pos)


def test_pioneer_waits_at_task_point_opening_soonest(payload_factory, role_factory):
    """两个任务点都在冷却时，守在更早开放的那一个旁边

    守错任务点会白白错过另一个先开放的任务点（复盘里两个任务点合计
    160分+160金币，只守一个等于把另一个让给对手）。
    """
    before = Pos(10, 12)
    late_task = Pos(10, 14)  # 近，但要 30 回合后才开放
    soon_task = Pos(20, 14)  # 远，2 回合后就开放
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, before.x, before.y, backPackCapability=40)],
        tasks=[
            (late_task.x, late_task.y, {"isValid": False, "coldDownRounds": 30}),
            (soon_task.x, soon_task.y, {"isValid": False, "coldDownRounds": 2}),
        ],
    )
    commands, _ = decide(payload)

    command = commands["10011"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert distance(step, soon_task) < distance(before, soon_task)


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


def test_sandbox_command_executes_task_in_sandbox(payload_factory, role_factory):
    """任务进行中：提交执行器命令，而不是把任务文件打印出来当答案

    任务文件里写的是任务要求（"查询北京文化遗产"），沙盒里能打印出来的只有
    它自己；答案要按描述去调本地接口取。旧实现把 `cat 任务文件` 的输出直接
    当答案提交，复盘里 4 次 submitAnswer 交的全是任务原文、任务分恒为 0。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读task_1_beijing.md",
    )
    command = sandbox_command(payload)

    # 答案区里跑的是执行器：任务文件的路径交给它，由它在沙盒里读文件并取数
    assert "task_1_beijing.md" in command
    assert "python3 python" in command
    assert TASK_SOLUTION_MARKER in command
    assert TASK_DATA_MARKER in command
    # 答案区不再直接打印任务文件：那正是"提交任务原文"的来源
    assert 'cat -- "task_1_beijing.md"' not in command
    # 命令必须带上任务标识，才能确认下一回合的输出属于本任务
    assert TASK_MARKER in command


def test_sandbox_command_idle_without_task(payload_factory):
    """没有进行中的任务时不使用沙盒"""
    assert sandbox_command(payload_factory(round_no=1)) == ""


def test_sandbox_command_gathers_clues_in_one_shot(
    payload_factory, role_factory,
):
    """一条沙盒命令同时带上接口取数、兜底搜索与目录诊断，减少逐次试错回合

    复盘里敌方"用错鉴权头→401、补参数又缺字段→400"逐次试错，白丢好几个
    回合；这里把"读文件取数 + 全盘找任务文件 + 列出沙盒工作目录"合成一条
    命令，一次就能拿到更多线索。搜索必须覆盖整个沙盒：实测沙盒的工作目录
    就是 `/`，而任务文件不在 `/` 的前三层里。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读tasks/task_1_beijing.md",
    )
    command = sandbox_command(payload)

    # 描述里给的路径原样交给执行器（它自己会做"读不到就按文件名全盘找"的兜底）
    assert "tasks/task_1_beijing.md" in command
    assert "find / " in command
    assert "-maxdepth" not in command
    # 诊断信息排在答案结束标记之后，不会被当成答案提交
    assert command.index(TASK_MARKER) < command.index(TASK_END_MARKER)
    assert command.index(TASK_END_MARKER) < command.index("ls -a")
    # 整条命令以 `:` 收尾，退出码为 0 才会被 `_task_answer` 采纳
    assert command.rstrip().endswith(":")


def test_sandbox_probes_task_dir_when_description_has_no_file(
    payload_factory, role_factory,
):
    """描述里没有文件名时：先全盘探测沙盒，认出文件再读它作答

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
    assert "find / " in command
    assert '-name "task*"' in command
    # 探测输出带的是探测标记，不会被 `_task_answer` 当成答案
    assert TASK_MARKER not in command

    # 探测结果里认出了任务文件：改用完整路径读它，并带上本任务标识
    payload["lastCmdResult"] = (
        f"[exitCode:0]\n{TASK_PROBE_MARKER}{token}\n"
        "/tmp/selfEvolutionTask/task_1_alpha.md\n"
    )
    command = sandbox_command(payload)
    assert "/tmp/selfEvolutionTask/task_1_alpha.md" in command
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
    payload["lastCmdResult"] = _solution_result(
        phase_task, "task_1_beijing.md", '{"city": "北京", "count": 7}',
    )
    commands, _ = decide(payload)

    # 提交的是执行器取到的数据，不是任务文本，也不含标记之后的诊断信息
    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": '{"city": "北京", "count": 7}',
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
    payload["lastCmdResult"] = _solution_result(
        phase_task, "task_1_beijing.md", '{"city": "北京", "count": 7}',
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": '{"city": "北京", "count": 7}',
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
    assert '-name "task*"' in command
    assert '-name "spec*"' in command
    # 整段 dump 排在答案结束标记之后，不会被当成当前任务的答案提交
    assert command.index(TASK_END_MARKER) < command.index(TASK_FILE_MARKER)


def test_task_dump_widens_search_after_empty_result(
    payload_factory, role_factory,
):
    """上一回合什么任务文件都没回读到时，回读放宽到 *.md

    沙盒里任务文件的实际命名未必和任务描述里写的一致（描述写
    task_1_beijing.md、沙盒里却是别的名字），卡在一个文件名上反复空转
    不如把候选文件都摊开——回读出来的文件名同样会被 `_task_file` 认出来，
    下一回合就能直接读中意的那一份。
    """
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=2,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    # 上一回合的命令跑完了，但输出里一个任务文件分段都没有（全盘也没找到）
    payload["lastCmdResult"] = (
        f"[exitCode:0]\n{TASK_MARKER}{_task_token(phase_task)}\n"
        "find: './etc/docker': Permission denied\n"
        f"{TASK_END_MARKER}\n"
    )
    command = sandbox_command(payload)

    assert '-name "*.md"' in command


def test_task_cache_keeps_first_answer_per_task_file():
    """缓存按任务文件名记执行器解出来的答案，且不会被后来的输出覆盖"""
    brain._remember_task_answers(
        "[exitCode:0]\n[TASK]上一个任务\n"
        "[API] http://localhost:8899/heritage?city=alpha => 12\n"
        "[SOLUTION]task_1_alpha.md\nalpha-answer\n[/SOLUTION]\n"
    )
    # 没有取数证据的 `[SOLUTION]` 段不入缓存（那只是任务原文，不是答案）
    brain._remember_task_answers(
        "[SOLUTION]task_1_alpha.md\n乱码\n[/SOLUTION]\n"
    )
    assert brain._TASK_ANSWER_CACHE == {"task_1_alpha.md": "alpha-answer"}

    # 没有答案段的输出（执行器什么都没取到）不入缓存
    brain._remember_task_answers("[exitCode:0]\n[TASK]上一个任务\n[TASK_END]\n")
    assert brain._TASK_ANSWER_CACHE == {"task_1_alpha.md": "alpha-answer"}


def test_pioneer_answers_from_cached_solution(payload_factory, role_factory):
    """沙盒里提前解出来的答案命中时，接取后立刻交卷

    回归：复盘里敌方靠答案缓存两次提交各拿 155 分（任务奖励 80 +
    5×标准回合数15/(完成回合-接取回合)1），我们每次都重新跑一遍沙盒，
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
    # 上一个任务期间执行器把沙盒里的任务文件都试着解了一遍（含本次要用的这份）
    payload["lastCmdResult"] = (
        "[exitCode:0]\n"
        "[TASK]上一个任务\n"
        "[API] http://localhost:8899/heritage?city=alpha => 12\n"
        f"{TASK_SOLUTION_MARKER}/tmp/selfEvolutionTask/task_1_alpha.md\n"
        '{"city": "Alpha", "count": 3}\n'
        f"{TASK_SOLUTION_END}\n"
        f"{TASK_SOLUTION_MARKER}/tmp/selfEvolutionTask/task_2_beijing.md\n"
        '{"city": "北京", "count": 7}\n'
        f"{TASK_SOLUTION_END}\n"
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": '{"city": "北京", "count": 7}',
    }
    # 已经有缓存答案，不必再花一个来回执行沙盒命令
    assert sandbox_command(payload) == ""


def test_sandbox_answer_rejects_task_text_echo(payload_factory, role_factory):
    """答案区里只有任务原文时绝不提交（复盘里 4 次 0 分提交就是这个形态）"""
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    # 旧实现把 `cat 任务文件` 的输出当答案：取数证据一个也没有
    payload["lastCmdResult"] = _sandbox_result(
        phase_task,
        "# 自进化任务 A-1：查询北京文化遗产\n## 任务背景\n"
        "请阅读task_1_beijing.md，获取任务信息\n",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands
    # 答案区里没有取数证据，沙盒命令继续下发（下一回合再执行一次）
    assert sandbox_command(payload) != ""


def test_sandbox_answer_rejects_solution_without_api_data(
    payload_factory, role_factory,
):
    """没有取数证据的 `[SOLUTION]` 段同样不能提交（宁可这一回合不交卷）"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _sandbox_result(
        phase_task,
        f"{TASK_SOLUTION_MARKER}task_1_beijing.md\n任务原文\n{TASK_SOLUTION_END}\n"
        f"{TASK_END_MARKER}\n",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands


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


# === issue #77：接口把错误包成 200 时不能当答案交上去（S2） ===


def test_sandbox_answer_rejects_api_error_body(payload_factory, role_factory):
    """接口用 200 包一层错误 JSON 时不能提交（复盘 PK590557 的 R16/R18）

    取数器只认"请求有响应"，错误 JSON 照样会进 `[SOLUTION]` 段；原样交上去
    就是又一次 0 分，还把 `TASK_SUBMIT_LIMIT` 的额度耗在一次注定不被放行的
    提交上。命中时这一回合不交卷，沙盒命令照常重跑。
    """
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    for body in (
        '{"status": "error", "message": "missing query param city"}',
        '{"error": "Unauthorized", "code": 401}',
        '{"code": 404, "detail": "no such endpoint"}',
        "404 Not Found",
        "Traceback (most recent call last):\nValueError: bad city",
    ):
        payload["lastCmdResult"] = _solution_result(
            phase_task, "task_1_beijing.md", body,
        )
        commands, _ = decide(payload)
        assert "10011" not in commands, body
    # 错误体不算答案，沙盒命令继续下发（下一回合重新取数）
    assert sandbox_command(payload) != ""


def test_sandbox_answer_keeps_real_data_that_mentions_errors(
    payload_factory, role_factory,
):
    """正常数据里出现 "error"/"Forbidden" 这类字眼时照常提交

    "故宫"的英文是 Forbidden City，一份完全正确的文化遗产答案里就会出现
    Forbidden；`{"error": null}` 也是常见的正常包装。这道闸门只拦"看着就是
    错误"的形态，不能把正确答案拦下来。
    """
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    for body in (
        "Forbidden City 故宫博物院",
        '{"error": null, "data": {"city": "北京", "count": 7}}',
        '{"code": "BJ-01", "name": "故宫"}',
        '{"status": "ok", "count": 7}',
    ):
        payload["lastCmdResult"] = _solution_result(
            phase_task, "task_1_beijing.md", body,
        )
        commands, _ = decide(payload)
        assert commands["10011"]["action"] == "submitAnswer", body
        assert commands["10011"]["taskAnswer"] == body


def test_task_error_body_matches_only_error_shapes():
    """`_task_error_body` 的判定边界：只认错误键/错误状态值/4xx-5xx/异常回溯"""
    bad = (
        '{"status":"error"}',
        '{"status": "failed"}',
        '{"status": 503}',
        '{"code": 500}',
        '{"code": "403"}',
        '{"error": "boom"}',
        "500 Internal Server Error",
        "[APIFAIL] http://localhost:8899 HTTPError 404",
        "urllib.error.URLError: <urlopen error>",
        # 沙盒命令自己的报错（PK590882/590921 的 R16）：没有错误键也没有状态码
        "/bin/bash: jq: command not found",
        "bash: line 3: curl: command not found",
        "jq: command not found",
        "cat: task_1_beijing.md: No such file or directory",
        '{"status":"error","message":"Endpoint not found: /api/docs"}',
        "curl: (23) Failed writing body",
    )
    good = (
        "Forbidden City 故宫博物院",
        "Not Found 是一首歌",
        '{"error": null}',
        '{"error": []}',
        '{"code": "BJ-01"}',
        '{"code": 2001}',
        '{"status": "ok", "count": 7}',
        "晴，26℃",
        # 这几个词是合法英文/正常数据，不能因为长得像就拦下来：
        # 没有工具名前缀的 command not found、任务描述里的 `No such file`、
        # 缺了 "or directory" 的整句都不是报错输出
        "command not found 是我的歌名",
        "No such file",
        "No such file in the archive",
        "Endpoint not found 章节在第 3 页",
        "Failed writing body 是 curl 手册里的一节",
    )
    for text in bad:
        assert brain._task_error_body(text), text
    for text in good:
        assert not brain._task_error_body(text), text


def test_task_error_body_matches_broken_script_output():
    """脚本跑不起来时的 stderr 同样不能当答案交上去（S1：PK591009 的 R14）

    LLM 给的命令调沙盒里的脚本时，脚本自己有毛病的话输出里一行取数结果都没有：
    CRLF 的 shebang（`bad interpreter`）、脚本不在（`sh: 1: ./check: not found`）、
    引号不配对（`unexpected EOF while looking for matching`）。这些被当成"取到的
    数"交上去的话，Judge 判 0 分，还白烧一次提交额度（见 `TASK_SUBMIT_LIMIT`）。
    """
    bad = (
        "/bin/sh^M: bad interpreter: No such file or directory",
        "sh: 1: ./check: not found",
        "dash: 3: ./check.sh: not found",
        "bash: -c: line 1: unexpected EOF while looking for matching",
        "bash: /tmp/check.sh: cannot execute binary file: Exec format error",
    )
    good = (
        "故宫博物院",
        '{"city": "北京", "weather": "晴"}',
        "not found 是英文里的否定说法",
        "第 3 页写着 not found 的来历",
    )
    for text in bad:
        assert brain._task_error_body(text), text
    for text in good:
        assert not brain._task_error_body(text), text


def test_cached_error_body_is_not_submitted(payload_factory, role_factory):
    """缓存里那条"答案"是错误体时同样不能交（提交闸门不能只拦一条路）

    缓存是在执行器输出上直接建的（`_remember_task_answers`），只按取数证据
    `[API]` 过滤，错误 JSON 一样会进缓存；`_cached_answer` 走的是与
    `_task_answer` 不同的那条路，闸门必须也装在这里。
    """
    phase_task = "请阅读task_1_alpha.md"
    # 执行器把 401 的错误 JSON 当成"取到的数"记进了缓存
    brain._remember_task_answers(
        "[exitCode:0]\n[TASK]上一个任务\n"
        "[API] http://localhost:8899/heritage?city=alpha => 42\n"
        f"{TASK_SOLUTION_MARKER}task_1_alpha.md\n"
        '{"status": "error", "code": 401}\n'
        f"{TASK_SOLUTION_END}\n"
    )
    assert "task_1_alpha.md" in brain._TASK_ANSWER_CACHE  # 缓存确实收下了它

    payload = payload_factory(
        round_no=16,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    assert brain._cached_answer(Turn.load(payload)) is None

    payload["lastCmdResult"] = _solution_result(
        phase_task, "task_1_alpha.md", '{"status": "error", "code": 401}',
    )
    commands, _ = decide(payload)
    assert "10011" not in commands


def test_task_brief_reports_error_body_reason(payload_factory, role_factory):
    """`task_brief` 把"交上去的是错误体"写成 state=error_body"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _solution_result(
        phase_task, "task_1_beijing.md", '{"status": "error", "code": 404}',
    )
    brief = brain.task_brief(Turn.load(payload))
    assert "state=error_body" in brief


# === issue #92：答案不能是沙盒里读到的文档原文（PK590847/590851） ===

# 沙盒里那份接口文档（复盘 PK590851 的 R13 交上去的就是它的原文）
_DOC_TEXT = (
    "# 国家文化遗产数字档案查询系统 — API 参考文档\n"
    "## 接口列表\n"
    "GET http://localhost:8899/heritage?city=<城市名>  查询该城市的文化遗产\n"
    '响应体为 JSON：{"city": "北京", "items": [...]}\n'
)


def _doc_hint(text: str = _DOC_TEXT) -> str:
    """执行器读到接口文档时打出的那行指纹（`[DOC]` + 文档开头）"""
    return f"{TASK_DOC_MARKER}{' '.join(text.split())[:brain.TASK_TEXT_HINT]}"


def _doc_echo_result(phase_task: str, task_file: str = "task_1_beijing.md") -> str:
    """构造一条"取回来的是文档页本身"的沙盒输出（执行器视角）"""
    return _sandbox_result(
        phase_task,
        f"{_doc_hint()}\n"
        f"{TASK_DATA_MARKER} http://localhost:8899/docs => {len(_DOC_TEXT)}\n"
        f"{TASK_SOLUTION_MARKER}{task_file}\n{_DOC_TEXT}\n{TASK_SOLUTION_END}\n"
        f"{TASK_END_MARKER}\n",
    )


def test_sandbox_answer_rejects_api_doc_echo(payload_factory, role_factory):
    """取回来的是接口文档页本身时不提交（复盘 PK590851 的 R13 交了文档原文）

    执行器把 `/docs` 这类地址的正文当成"取到的数"打进 `[SOLUTION]` 段：
    `_task_echo` 只认任务描述里的中文长句、`_task_error_body` 只认错误体，
    文档两者都不是，原样交上去就是 Judge 的 0 分。
    """
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _doc_echo_result(phase_task)
    commands, _ = decide(payload)

    assert "10011" not in commands
    # 文档不是答案：沙盒命令继续下发，下一回合照常取数
    assert sandbox_command(payload) != ""


def test_sandbox_answer_rejects_task_file_read_back(payload_factory, role_factory):
    """答案是沙盒里回读的任务文件正文时同样不提交

    `_task_dump` 每回合把沙盒里的任务文件读回来（供答案缓存与文件名识别用），
    这段正文出现在答案里说明交的是文件而不是取到的数。任务描述很短时
    （"请阅读task_1_beijing.md" 里没有 6 个字的中文长句）`_task_echo` 抓不住，
    只有这道闸门挡得住。
    """
    phase_task = "请阅读task_1_beijing.md"
    task_text = (
        "# 自进化任务 A-1：查询北京文化遗产\n"
        "## 任务背景\n"
        "请按接口文档取数后作答，答案要能通过 Judge 的校验。\n"
    )
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _sandbox_result(
        phase_task,
        f"{TASK_DATA_MARKER} http://localhost:8899/heritage?city=beijing => 33\n"
        f"{TASK_SOLUTION_MARKER}task_1_beijing.md\n{task_text}\n{TASK_SOLUTION_END}\n"
        f"{TASK_END_MARKER}\n"
        f"{TASK_FILE_MARKER}/tmp/selfEvolutionTask/task_1_beijing.md\n"
        f"{task_text}\n{TASK_FILE_END}\n",
    )
    commands, _ = decide(payload)

    assert "10011" not in commands


def test_task_text_answer_matches_only_document_prefixes():
    """`_task_text_answer` 的判定边界：只拦"复读文档开头"的答案

    正常取到的数据（哪怕提到文档里的地址、字段名）一律放行——任务千变万化，
    闸门只排掉"整段等于某份文档开头"这一种形态，误伤的窗口必须小到可以忽略。
    """
    result = f"{_doc_hint()}\n{TASK_DATA_MARKER} http://localhost:8899/docs => 4\n"

    # 文档原文（或它的开头）出现在答案里 -> 拦下
    assert brain._task_text_answer(_DOC_TEXT, result)
    assert brain._task_text_answer(f"{_DOC_TEXT}\n以上是接口说明", result)
    # 取数取到的数据 -> 放行
    assert not brain._task_text_answer('{"city": "北京", "items": [7]}', result)
    # 只重合一小段（地址、文档里的字段名）-> 放行
    assert not brain._task_text_answer(
        "GET http://localhost:8899/heritage?city=beijing 返回 3 条", result,
    )
    # 指纹本身太短时不作数（短文档的指纹撑不起"复读"的判定）
    assert not brain._task_text_answer("北京 7", f"{TASK_DOC_MARKER}# API\n")
    assert not brain._task_text_answer("", result)


def test_task_doc_body_matches_only_markdown_documents():
    """`_task_doc_body` 的判定边界：只拦"首行就是标题"的文档正文（S2）

    这条闸门补的是 `_task_text_answer` 的盲区：走 LLM 那条路时沙盒输出里
    没有 `[DOC]`/`[TASK_FILE]` 指纹，`cat 文档` 打回来的正文没人比得了。
    复盘 PK590836 的 R15 交上去的正是
    `# 国家文化遗产数字档案查询系统 — API 参考文档…`（Judge 判 0，还烧掉
    一次提交额度）。取到的数据一律放行。
    """
    assert brain._task_doc_body(_DOC_TEXT)
    assert brain._task_doc_body("# 国家文化遗产数字档案查询系统 — API 参考文档")
    # 取数取到的数据 / 短答案 -> 放行
    assert not brain._task_doc_body('{"city": "北京", "items": [7]}')
    assert not brain._task_doc_body("故宫 7")
    assert not brain._task_doc_body("")
    # 单行、标题里也没有"文档"字样 -> 放行（不误伤以井号开头的短答案）
    assert not brain._task_doc_body("#7 号坑位")
    # 多行但首行不是标题 -> 放行（数据里带井号注释是正常的）
    assert not brain._task_doc_body("城市 结果\n# 注释\n北京 7\n")


def test_cached_doc_text_is_not_submitted(payload_factory, role_factory):
    """缓存里那条"答案"是文档原文时不能进缓存（提交闸门不能只拦一条路）

    缓存是在执行器输出上直接建的（`_remember_task_answers`），只按取数证据
    `[API]` 过滤；`/docs` 这类地址的正文照样带着 `[API]` 证据，缓存下来等于
    把接口文档背了下来，下一个任务点一到手就会把它当答案秒交。
    """
    phase_task = "请阅读task_1_alpha.md"
    brain._remember_task_answers(_doc_echo_result(phase_task, "task_1_alpha.md"))

    assert "task_1_alpha.md" not in brain._TASK_ANSWER_CACHE

    payload = payload_factory(
        round_no=16,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    commands, _ = decide(payload)
    assert "10011" not in commands


def test_task_brief_reports_doc_text_reason(payload_factory, role_factory):
    """`task_brief` 把"交上去的是沙盒里那份文档"写成 state=doc_text"""
    phase_task = "请阅读task_1_beijing.md"
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["lastCmdResult"] = _doc_echo_result(phase_task)
    brief = brain.task_brief(Turn.load(payload))
    assert "state=doc_text" in brief


def test_llm_answer_rejects_doc_echo(payload_factory, role_factory):
    """LLM 把喂给它的接口文档原文当答案返回时不能交（照抄文档 ≠ 答案）

    `_task_prompt` 会把沙盒里捞回来的任务文件与接口文档一起喂给 LLM，
    LLM 完全可能把文档正文抄回来当作 `ANSWER:`——它和沙盒自己取回来的那份
    一样，都不是答案。
    """
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"
    evidence = f"{_doc_hint()}\n{TASK_DATA_MARKER} http://localhost:8899/docs => 4\n"
    decide(_llm_task_payload(
        payload_factory, role_factory, phase_task, 11, evidence=evidence,
    ))

    payload = _llm_task_payload(
        payload_factory, role_factory, phase_task, 12, evidence=evidence,
    )
    # LLM 把文档抄成一行贴回来（`ANSWER:` 只认一行，空白归一化后仍是文档原文）
    payload["llmResp"] = f"ANSWER: {' '.join(_DOC_TEXT.split())}"
    commands, _ = decide(payload)

    assert "10011" not in commands or commands["10011"]["action"] != "submitAnswer"


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


def test_night_controller_plugs_gap_next_to_its_weapon(payload_factory, role_factory):
    """夜晚射程内没有目标时，操控者挪到武器旁那格围墙缺口上堵住（T6 人肉城墙）

    复盘里"无围墙 + 全员空转、没有任何堵缺口行为"（PK590991/591016）：墙砌
    不起来时缺口就是机器人直通基地的门，站在缺口上照样操控得到武器，所以
    与其空着，不如把人挪上去。
    """
    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
    )
    holes = _wall_holes(Turn.load(payload))
    commands, _ = decide(payload)

    # 站的是缺口那一格本身（不是它的邻格），且仍在武器的操控范围内
    assert commands["10010"] == {
        "action": "move",
        "targetPos": [{"x": 8, "y": 23}],
    }
    assert Pos(8, 23) in holes
    assert distance(Pos(8, 23), Pos(9, 24)) <= 1


def test_night_plugs_priority_gap_without_weapon(payload_factory, role_factory):
    """没摊上武器的角色去堵优先级最高的缺口，堵上之后不再挪窝（T6）"""
    def _commands(x: int, y: int) -> dict:
        return decide(payload_factory(
            round_no=DAY_ROUNDS + 1,
            roles=[role_factory(10010, WORKER, x, y, backPackCapability=100)],
        ))[0]

    # 没有敌方单位时正面是 up（`_wall_side_order` 的默认顺序），
    # 上边外圈的第一个缺格就是 (13,26)
    assert _commands(14, 27)["10010"] == {
        "action": "move",
        "targetPos": [{"x": 13, "y": 26}],
    }
    # 已经堵在缺口上：这一回合不下指令（站着不动就是堵着，别把自己支使走）
    assert "10010" not in _commands(13, 26)


def test_night_attacks_instead_of_plugging_gap(
    payload_factory, role_factory, robot_factory,
):
    """射程内有机器人时照旧开火，操控者不会为了堵缺口让出武器（T6）"""
    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        robots=[robot_factory(30001, 12, 24, targetTeam="challenger")],
    )
    commands, _ = decide(payload)

    assert commands["10020"] == {
        "action": "attack",
        "targetPos": [{"x": 12, "y": 24}],
        "controllerId": "10010",
    }
    assert "10010" not in commands


def test_night_keeps_position_when_ring_is_complete(payload_factory, role_factory):
    """外墙一圈都在时没有缺口可堵，操控者照旧守在武器旁（T6 不改动原有行为）"""
    ring = _calc_wall_order(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,
        roles=[
            role_factory(10010, WORKER, 9, 23, backPackCapability=100),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
            *[
                role_factory(40000 + index, WALL, pos.x, pos.y)
                for index, pos in enumerate(ring)
            ],
        ],
    )
    commands, _ = decide(payload)

    assert "10010" not in commands


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
            role_factory(10020, ROCKET, sites[0].x, sites[0].y, attackRange=10),
            role_factory(10030, RAILGUN, sites[1].x, sites[1].y, attackRange=6),
        ],
    )

    # 默认计划: 金币够就直接开工第 3 座塔（加特林）
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": sites[2].x, "y": sites[2].y}],
        "name": GATLING,
    }

    # 计划只要求 2 座塔: 不再开工第 3 座
    payload["llmResp"] = "PLAN: tower=2"
    commands, _ = decide(payload)
    assert "10010" not in commands


def test_llm_plan_wall_quota_builds_walls_before_trading(
    payload_factory, role_factory,
):
    """计划里 wall=2 时石工先铺围墙，铜工照旧变现（V4 的站位分工）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            # 石工（R2）背着石材、站在围墙位旁 —— 砌墙是他的活
            role_factory(
                10010, WORKER, 12, 26, backPackCapability=100,
                backpack=[WALL_MATERIAL] * SELL_BATCH,
            ),
            # 铜矿工人（R3）在小贩旁边 —— 他只负责变现
            role_factory(
                10012, WORKER, 13, 27, backPackCapability=100,
                backpack=[COPPER_MINE] * SELL_BATCH,
            ),
        ],
        zones=[(VENDOR, 13, 28)],
    )

    # 默认计划: 铜工先把矿石变现（金币滚动起来）
    commands, _ = decide(payload)
    assert commands["10012"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": SELL_BATCH,
    }

    # 计划要求先铺 2 段围墙: 石工手里的石材先用于施工
    payload["llmResp"] = "PLAN: wall=2"
    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }
    # 分工不随计划变：铜工照旧去变现，不会跑去砌墙
    assert commands["10012"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": SELL_BATCH,
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
        "name": ROCKET,
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
            role_factory(10020, ROCKET, sites[0].x, sites[0].y, attackRange=10),
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
        "name": GATLING,
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


# === 开局建造节奏与塔位提示（issue #26） ===


def test_opening_round_keeps_second_tower_despite_plan(
    payload_factory, role_factory,
):
    """开局两回合内塔数有硬下限：计划把塔数压到 1 座也照样补第 2 座

    回归：复盘里"首日 3 回合只落地 1 座塔、R2 整回合零建造"，
    第 2 座塔拖到 R3 才开工，而同局敌方是单回合双建，火力成型快得多。
    """
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=2,
        gold=WEAPON_BUILD_COST + 5,  # 不够熔断线，但够再建一座塔
        roles=[
            role_factory(10010, WORKER, 9, 22, backPackCapability=100),
            role_factory(10020, ROCKET, sites[0].x, sites[0].y, attackRange=10),
        ],
    )
    payload["llmResp"] = "PLAN: tower=1"

    commands, _ = decide(payload)
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": sites[1].x, "y": sites[1].y}],
        "name": RAILGUN,
    }

    # 过了开局两回合后仍然听计划的：塔数压在 1 座就不再开工（留钱买升级券）
    payload["roundNo"] = 3
    commands, _ = decide(payload)
    assert "10010" not in commands


def test_strategy_prompt_includes_tower_sites(payload_factory, monkeypatch):
    """prompt 摊开塔位坐标与对应的武器类型

    复盘建议"prompt 注入最近可建位坐标，消除'先移动、下回合再建'的一回合延迟"，
    同时让"说建哪座塔"与"建的是哪种武器"对得上（塔型由 `TOWER_LOADOUT` 决定，
    不受 LLM 文字左右，只会照实告知）。
    """
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    prompt = _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {})

    # 默认基地 (10,24) 的三座塔位，武器按射程优先的顺序绑定
    assert "rocket(12,23)" in prompt
    assert "railgun(10,22)" in prompt
    assert "gatling(9,23)" in prompt


def test_plan_summary_reports_no_tower_site_without_station(payload_factory):
    """没有基地时塔位清单给出说明而不是崩溃"""
    turn = Turn.load(payload_factory(station=None))
    assert "暂无可用塔位" in _plan_summary(turn, LLM_PLAN_DEFAULT)


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
                # 石工（R2）站在围墙位旁、背着石材 —— V4：砌墙是石工的活
                role_factory(
                    10010, WORKER, 12, 26, backPackCapability=100,
                    backpack=[WALL_MATERIAL] * SELL_BATCH,
                ),
                # 铜矿工人（R3）在小贩旁边 —— V4：他只负责变现，不碰墙
                role_factory(
                    10012, WORKER, 13, 27, backPackCapability=100,
                    backpack=[COPPER_MINE] * SELL_BATCH,
                ),
            ],
            zones=[(VENDOR, 13, 28)],
        )
        payload["llmResp"] = "PLAN: wall=0"
        return payload

    # 防守方：石工手里的石材先砌成围墙
    commands, prompt = decide(_payload("defender"))
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }
    # LLM 看到的配额也是执行层真正要铺的段数（建议与指令同源）
    assert "优先铺围墙 1 段" in prompt

    # 铜矿工人不砌墙：手里的矿石照卖，经济线不被"还差一段墙"扣住
    assert commands["10012"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": SELL_BATCH,
    }

    # 进攻方同样分工：石工照旧砌墙，铜工照旧变现（下限只影响配额，不影响分工）
    commands, _ = decide(_payload("challenger"))
    assert commands["10012"] == {
        "action": "sell",
        "name": COPPER_MINE,
        "num": SELL_BATCH,
    }


# === 金币阶梯与资金闲置（issue #28） ===


def _idle_gold_payload(payload_factory, role_factory, gold: int, *,
                       towers: int = 3, tower_level: int = 1,
                       round_no: int = 1) -> dict:
    """构造"防线建完、金币闲置"的局面（塔数与塔的等级可调）

    默认三座武器都已建成、围墙整圈都在，工人站在武器商店旁，
    用来观察金币在没有建造目标时去了哪里。
    """
    base = Turn.load(payload_factory())
    roles = [
        role_factory(
            10020 + index, kind, pos.x, pos.y,
            attackRange=4, level=tower_level,
        )
        for index, (kind, pos) in enumerate(
            zip(TOWER_LOADOUT, _calc_tower_sites(base)[:towers])
        )
    ]
    roles += [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(_calc_wall_order(base))
    ]
    roles.append(
        role_factory(10010, WORKER, 20, 17, backPackCapability=100),
    )
    return payload_factory(
        round_no=round_no, gold=gold, roles=roles,
        zones=[(WEAPON_SHOP, 20, 16)],
    )


def test_idle_gold_buys_wall_upgrade_voucher(payload_factory, role_factory):
    """武器线花完的金币不再沉睡：买围墙升级券顶住正面

    回归：三座武器建完、武器券也买过之后金币再没有任何出口，三场复盘
    都出现"金币连续多回合冻结、无建造无购买"（586322/586323/586377）。
    """
    payload = _idle_gold_payload(
        payload_factory, role_factory, WALL_UPGRADE_GOLD, tower_level=3,
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "buy",
        "name": WALL_UPGRADE_VOUCHER,
        "num": 1,
    }


def test_worker_uses_wall_upgrade_voucher_on_wall(payload_factory, role_factory):
    """背包里有围墙升级券时，站在围墙旁使用（level1 -> level2）"""
    base = Turn.load(payload_factory())
    payload = payload_factory(
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, 20, 17, backPackCapability=100,
                backpack=[WALL_UPGRADE_VOUCHER],
            ),
            role_factory(10011, WALL, 20, 16),
            # 三座武器满编满级：金币没有别的去处，围墙券可以放心用
            *[
                role_factory(
                    10020 + index, kind, pos.x, pos.y,
                    attackRange=4, level=3,
                )
                for index, (kind, pos) in enumerate(
                    zip(TOWER_LOADOUT, _calc_tower_sites(base))
                )
            ],
        ],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "use",
        "name": WALL_UPGRADE_VOUCHER,
        "targetPos": [{"x": 20, "y": 16}],
    }


def test_wall_upgrade_waits_for_weapon_reserve(payload_factory, role_factory):
    """金币还没到"武器线储备 + 围墙券"时先攒着，围墙券不吃武器券的钱

    复盘建议的"金币优先转化战力"：一张武器券换 10 点攻击力与一段射程，
    比一面围墙多 500 血划算得多，所以围墙券只能用武器线花剩下的钱。
    """
    def _payload(gold: int) -> dict:
        payload = _idle_gold_payload(
            payload_factory, role_factory, gold, towers=2, round_no=3,
        )
        # 塔数压在 2 座（塔位还剩 1 个空着，武器线因此要留一座塔的钱）
        payload["llmResp"] = "PLAN: tower=2"
        return payload

    # 45 金 = 一座塔的储备(25) + 围墙券(20)：刚好够
    commands, _ = decide(_payload(WEAPON_BUILD_COST + WALL_UPGRADE_GOLD))
    assert commands["10010"] == {
        "action": "buy",
        "name": WALL_UPGRADE_VOUCHER,
        "num": 1,
    }

    # 差一金就买不了：这笔钱要先留给武器线
    commands, _ = decide(_payload(WEAPON_BUILD_COST + WALL_UPGRADE_GOLD - 1))
    assert "10010" not in commands


def test_worker_upgrades_level2_weapon_with_second_voucher(
    payload_factory, role_factory,
):
    """武器升到 level2 后金币仍有出口：用武器升级券2 继续升到 level3

    回归：以前只认 level1->level2，三座塔都到 level2 之后金币再没有出口。
    """
    base = Turn.load(payload_factory())
    site = _calc_tower_sites(base)[0]
    payload = payload_factory(
        round_no=3,
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, 20, 17, backPackCapability=100,
                backpack=[WEAPON_UPGRADE_VOUCHER2],
            ),
            role_factory(10020, ROCKET, site.x, site.y, attackRange=10, level=2),
            role_factory(10030, RAILGUN, 20, 16, attackRange=6, level=2),
        ],
    )
    payload["llmResp"] = "PLAN: tower=1"
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "use",
        "name": WEAPON_UPGRADE_VOUCHER2,
        "targetPos": [{"x": 20, "y": 16}],
    }


def test_gold_left_counts_purchases(payload_factory):
    """同一回合已经发出去的购买指令也要从余额里扣掉

    金币要等回合结算才扣，买券的路径变多之后（武器券/围墙券），
    两名工人各买一张券同样会超支——复盘里的"金币没扣、东西也没到手"。
    """
    turn = Turn.load(payload_factory(gold=UPGRADE_GOLD))
    build = {"action": "build", "name": ROCKET, "targetPos": []}
    buy = {"action": "buy", "name": WALL_UPGRADE_VOUCHER, "num": 1}

    assert _gold_left(turn, {}) == UPGRADE_GOLD
    assert _gold_left(turn, {1: build}) == UPGRADE_GOLD - WEAPON_BUILD_COST
    # 报文的价格表里没有围墙券，退回任务书4.6.3 的售价(20)
    assert _gold_left(turn, {1: buy}) == UPGRADE_GOLD - WALL_UPGRADE_GOLD

    # 报文给了售价时按报文的算
    payload = payload_factory(gold=UPGRADE_GOLD * 2)
    payload["weaponShopList"] = [{"name": WALL_UPGRADE_VOUCHER, "price": 120}]
    assert _gold_left(Turn.load(payload), {1: buy}) == UPGRADE_GOLD * 2 - 120


def test_strategy_prompt_includes_task_distance(
    payload_factory, role_factory, monkeypatch,
):
    """任务点带上到开拓者的距离，避免 LLM 凭感觉判断"距离远"而放弃任务

    复盘里 LLM 两次以"任务点距离远、风险未知"建议放弃任务，而两个任务点
    离我方基地只有 11~13 格（586377）。
    """
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    turn = Turn.load(payload_factory(
        round_no=1,
        tasks=[(23, 14)],
        roles=[role_factory(10010, PIONEER, 20, 16, backPackCapability=40)],
    ))
    prompt = _generate_strategy_prompt(turn, {})

    assert "(23,14)" in prompt
    assert "距我3格" in prompt


def test_strategy_prompt_states_tasks_and_gold_ownership(
    payload_factory, monkeypatch,
):
    """prompt 讲清权责：任务由客户端自动执行，富余金币的去处也摊开

    复盘里 LLM 建议"放弃任务/暂不造塔"、客户端却照旧去领任务建塔，
    建议与执行对不上（586322/586377）；复盘建议同时要求
    "提示词显式加'前期不存金币'规则"。
    """
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    prompt = _generate_strategy_prompt(Turn.load(payload_factory(round_no=1)), {})

    assert "不需要建议放弃任务" in prompt
    assert "不存金币" in prompt


# === 塔位占用校验与布防方位（issue #29） ===


def test_prompt_lists_only_pending_tower_sites(payload_factory, role_factory):
    """prompt 的塔位清单只列还没建成的塔位，并单列已有塔位

    回归：塔位清单以前把已建成的塔位一起列出来，LLM 照着陈旧坐标反复建议
    "再建一座火箭炮于(29,9)"，而那一格上一回合就已经建成了同款武器
    （586439 复盘："建议层未校验塔位占用，提示词缺乏当前已有塔清单"）。
    """
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=2,
        roles=[
            role_factory(
                10020, ROCKET, sites[0].x, sites[0].y, attackRange=10,
            ),
        ],
    )
    turn = Turn.load(payload)

    # 已有的塔位单独报，待建清单里不再出现它
    assert _tower_site_brief(turn, LLM_PLAN_DEFAULT) == (
        f"railgun({sites[1].x},{sites[1].y})、gatling({sites[2].x},{sites[2].y})"
    )
    summary = _plan_summary(turn, LLM_PLAN_DEFAULT)
    assert f"现有 1 座：rocket({sites[0].x},{sites[0].y})" in summary
    assert "待建塔位 railgun" in summary


def test_prompt_reports_no_pending_site_when_all_built(
    payload_factory, role_factory,
):
    """三座塔都建成后塔位清单给出说明，不再重复报出已建成的坐标"""
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    payload = payload_factory(
        round_no=2,
        roles=[
            role_factory(10020 + index, kind, site.x, site.y, attackRange=4)
            for index, (kind, site) in enumerate(zip(TOWER_LOADOUT, sites))
        ],
    )
    turn = Turn.load(payload)

    assert _tower_site_brief(turn, LLM_PLAN_DEFAULT) == "暂无可用塔位"
    assert "待建塔位 暂无可用塔位" in _plan_summary(turn, LLM_PLAN_DEFAULT)


def test_build_skips_site_occupied_by_robot(
    payload_factory, role_factory, robot_factory,
):
    """下发建造指令前校验占用：塔位被机器人踩住时改去下一座

    回归：`occupied_cells()` 只统计我方单位，机器人站在塔位上时建造指令照样
    下发、结算时判失败，这一回合的金币与施工都白费（复盘建议"下发前校验占用
    并自动改最近空位"）。
    """
    sites = _calc_tower_sites(Turn.load(payload_factory()))
    start = Pos(sites[0].x + 1, sites[0].y)  # 紧挨第 1 座塔位，本来会直接开工
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[role_factory(10010, WORKER, start.x, start.y, backPackCapability=100)],
        robots=[
            robot_factory(30001, sites[0].x, sites[0].y, targetTeam="challenger"),
        ],
    )
    commands, _ = decide(payload)

    command = commands["10010"]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    # 改去下一座塔位（railgun 位），而不是硬往被机器人踩住的那一格建
    assert distance(step, sites[1]) < distance(start, sites[1])


def test_llm_defend_yields_to_enemy_side(payload_factory, role_factory, monkeypatch):
    """敌方来路已知时按敌我坐标布防，LLM 猜的方位让位

    回归：复盘里计划一路写死 `defend=up`，敌方基地却在我方左（下）方，
    第一座塔因此压在没人来的那一侧（586440："defend 方位按敌我坐标推算，
    替换写死的 defend=up"）。看不到敌方单位时 `defend` 照旧生效。
    """
    monkeypatch.setattr(brain, "LLM_PROMPT_ENABLED", True)
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        roles=[role_factory(10010, WORKER, 8, 23, backPackCapability=100)],
        enemies=[role_factory(20010, WORKER, 3, 25)],  # 敌方单位在基地左侧
    )
    payload["llmResp"] = "PLAN: defend=up"
    turn = Turn.load(payload)

    # 敌方来路在左侧：第 1 座塔落在左侧，而不是计划里的 up
    assert _calc_tower_sites(turn, "up")[0] == Pos(9, 23)
    # 看不到敌方单位时 defend 照旧优先（与 issue #14 的行为一致）
    assert _calc_tower_sites(
        Turn.load(payload_factory(round_no=1)), "up",
    )[0] == Pos(10, 25)

    commands, prompt = decide(payload)
    # 工人站在左侧塔位旁，按敌方来路直接开工
    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 9, "y": 23}],
        "name": ROCKET,
    }
    # prompt 报出的布防方位与执行层同一套判定（建议与指令同源）
    assert "布防方位 敌方来路 up/left" in prompt


# === 塔位认领与通路人满（issue #34） ===


def test_second_worker_follows_claimed_tower_site(
    payload_factory, role_factory,
):
    """塔位都被赶路的同伴认领后，后一名工人跟着走，而不是掉头去采矿

    回归：R2 型空过回合（PK586411/536/619/647 连续四场复发）——工人朝塔位
    赶路时会顺手把塔位认领下来（防止几个人挤同一座塔），剩下的工人一个空位
    都挑不到，整回合被派去采矿，于是"PLAN 承诺 tower=2、roleCommandMap 却
    全为 move、无 build，金币 50 闲置到天亮"。
    """
    rocket = Pos(12, 23)  # 唯一还没建成的那座塔位
    payload = payload_factory(
        round_no=2,
        gold=WEAPON_BUILD_COST * 3,
        roles=[
            role_factory(10010, WORKER, 15, 30, backPackCapability=100),
            role_factory(10012, WORKER, 16, 30, backPackCapability=100),
            role_factory(10020, RAILGUN, 10, 22, attackRange=6),
            role_factory(10030, GATLING, 9, 23, attackRange=3),
        ],
        zones=[(STONE_MINE, 17, 30)],  # 紧邻第 2 名工人：旧实现会就地采石
    )
    commands, _ = decide(payload)

    # 第 1 名工人认领塔位并向它移动
    assert commands["10010"]["action"] == "move"
    # 第 2 名工人跟着奔向同一座塔，而不是把这一回合花在采石上
    assert commands["10012"]["action"] == "move"
    step = Pos(
        commands["10012"]["targetPos"][0]["x"],
        commands["10012"]["targetPos"][0]["y"],
    )
    assert distance(step, rocket) < distance(Pos(16, 30), rocket)


def test_step_toward_keeps_progress_when_first_step_claimed(
    payload_factory, role_factory,
):
    """队友认领了最顺路的那一步时仍然朝目标走，不再判定"目标走不通"

    回归：`_step_toward` 只按落脚点换路，"落在别人认领格子上的第一步"被整体
    跳过；所有落脚点的第一步都被认领时旧实现返回 None，调用方据此判定"目标
    走不通"，于是工人掉头去采矿、开拓者原地发呆（issue #34 的"计划承诺建塔、
    指令里一条 build 都没有"）。
    """
    payload = payload_factory(
        gold=0,
        roles=[role_factory(10010, WORKER, 15, 30, backPackCapability=100)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]
    target = Pos(12, 23)  # 第 1 座塔位

    # 本体这一回合能走的相邻格，除 (16,29) 外全被队友认领：落脚点都走得到，
    # 只是通往它们的每一步都被占了，这时仍然朝目标方向走
    claimed = set(get_neighbors(worker.pos)) - {Pos(16, 29)}
    assert _step_toward(turn, worker, target, claimed) == Pos(16, 29)

    # 离目标更近的相邻格全被认领时原地待命，不再折返（复盘里"工人
    # (30,7)→(31,6)→(30,7) 两回合原地打转"）
    assert _closest_step(
        turn, worker, target,
        {Pos(14, 29), Pos(15, 29), Pos(16, 29)}, False,
    ) is None


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


# === 战术参考 T2/T3/T4 ===


def test_tower_layout_each_reachable(payload_factory, role_factory):
    """T2：默认地图上，规划出的三座塔每座都有人能走到旁边操控"""
    turn = Turn.load(payload_factory(
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
    ))
    sites = _calc_tower_sites(turn)

    assert len(sites) == 3
    assert _tower_sites_reachable(turn, sites)


def test_tower_sites_reachable_rejects_sealed_site(payload_factory, role_factory):
    """T2：塔位八邻域全被围墙占住时判为不可达（塔建了也没人操控得了）"""
    sealed = Pos(20, 20)
    ring = [
        Pos(sealed.x + dx, sealed.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    ]
    roles = [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(ring)
    ]
    roles.append(role_factory(10010, WORKER, 5, 23, backPackCapability=100))
    turn = Turn.load(payload_factory(roles=roles))

    assert not _tower_sites_reachable(turn, (sealed,))


def test_tower_sites_reachable_rejects_walled_off_site(payload_factory, role_factory):
    """T2：塔位周边可站人、但被围墙围成孤岛时同样判为不可达"""
    sealed = Pos(20, 20)
    # 以 sealed 为中心、半径2的封闭方框（16 段围墙）
    box = [
        Pos(sealed.x + dx, sealed.y + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if max(abs(dx), abs(dy)) == 2
    ]
    roles = [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(box)
    ]
    roles.append(role_factory(10010, WORKER, 5, 23, backPackCapability=100))
    turn = Turn.load(payload_factory(roles=roles))

    # 塔旁有空格，但角色在方框外，走不进去
    assert not _tower_sites_reachable(turn, (sealed,))


def _payload_with_station_voucher(payload_factory, role_factory, station_health: int):
    """构造"塔已满级、基地残血、工人拿着基地升级券在旁边"的局面"""
    payload = payload_factory(
        round_no=1,
        gold=100,
        roles=[
            # 三座塔建在规划位上且已满级 -> 武器升级分支不会抢钱
            role_factory(10020, ROCKET, 12, 23, level=3, health=2000),
            role_factory(10030, RAILGUN, 10, 22, level=3, health=2000),
            role_factory(10040, GATLING, 9, 23, level=3, health=2000),
            role_factory(
                10010, WORKER, 9, 23 + 1, backPackCapability=100,
                backpack=[STATION_UPGRADE_VOUCHER],
            ),
        ],
    )
    payload["llmResp"] = "PLAN: wall=0"
    for role in payload["teamOur"]["roles"]:
        if role["roleType"] == "station":
            role["health"] = station_health
    return payload


def test_station_upgrade_when_low_health(payload_factory, role_factory):
    """T3：基地残血时用基地升级券（回满血 + 顺便升级）"""
    # 1500 的 40%，低于 60% 阈值
    payload = _payload_with_station_voucher(payload_factory, role_factory, 600)

    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "use",
        "name": STATION_UPGRADE_VOUCHER,
        "targetPos": [{"x": 10, "y": 24}],
    }


def test_station_upgrade_skipped_when_healthy(payload_factory, role_factory):
    """T3：基地血量健康时不用券（券要留给残血保命）"""
    payload = _payload_with_station_voucher(payload_factory, role_factory, 1500)

    commands, _ = decide(payload)

    assert STATION_UPGRADE_VOUCHER not in str(commands.get("10010", ""))


def _payload_with_damaged_walls(payload_factory, role_factory, damage_at,
                                worker_pos: Pos, **worker_kw):
    """构造"塔已满级、围墙已铺满、其中几段残血"的局面"""
    order = _calc_wall_order(Turn.load(payload_factory()))
    roles = [
        role_factory(
            40000 + index, WALL, pos.x, pos.y,
            health=400 if pos in damage_at else 1000,
        )
        for index, pos in enumerate(order)
    ]
    # 三座塔建在规划位上且已满级：否则"建塔/升级武器"会先花掉金币，
    # 轮不到围墙修复这条分支
    roles.extend([
        role_factory(10020, ROCKET, 12, 23, level=3, health=2000),
        role_factory(10030, RAILGUN, 10, 22, level=3, health=2000),
        role_factory(10040, GATLING, 9, 23, level=3, health=2000),
    ])
    roles.append(
        role_factory(10010, WORKER, worker_pos.x, worker_pos.y, **worker_kw)
    )
    payload = payload_factory(round_no=1, gold=0, roles=roles)
    payload["llmResp"] = "PLAN: wall=0"
    return payload


def test_repair_walls_with_fixer(payload_factory, role_factory):
    """T4：三段以上残血围墙时用围墙修复包回满"""
    damaged = {Pos(13, 26), Pos(12, 26), Pos(11, 26)}
    payload = _payload_with_damaged_walls(
        payload_factory, role_factory, damaged,
        Pos(13, 25), backPackCapability=100, backpack=[WALL_FIXER],
    )

    commands, _ = decide(payload)
    command = commands["10010"]

    assert command["action"] == "use"
    assert command["name"] == WALL_FIXER
    # 目标是残血墙里"周围残血墙最多"的那一段
    target = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    assert target in damaged


def test_repair_walls_skipped_when_few_damaged(payload_factory, role_factory):
    """T4：残血墙不足三段时不值得跑一趟"""
    damaged = {Pos(13, 26), Pos(12, 26)}
    payload = _payload_with_damaged_walls(
        payload_factory, role_factory, damaged,
        Pos(13, 25), backPackCapability=100, backpack=[WALL_FIXER],
    )

    commands, _ = decide(payload)

    assert WALL_FIXER not in str(commands.get("10010", ""))


def test_repair_walls_buys_fixer_when_missing(payload_factory, role_factory):
    """T4：背包里没有修复包时先去武器商店买"""
    damaged = {Pos(13, 26), Pos(12, 26), Pos(11, 26)}
    payload = _payload_with_damaged_walls(
        payload_factory, role_factory, damaged,
        Pos(5, 23), backPackCapability=100,
    )
    payload["teamOur"]["goldNum"] = 50
    payload["mapInfo"]["zones"].append(
        {"neutralType": WEAPON_SHOP, "pos": {"x": 6, "y": 23}},
    )

    commands, _ = decide(payload)

    assert commands["10010"] == {"action": "buy", "name": WALL_FIXER, "num": 1}


# === issue #39：沙盒读错文件 / 防守方经济冻结（PK589253/589255/589257） ===


def test_task_find_only_reads_document_task_files(
    payload_factory, role_factory,
):
    """沙盒里只回读文档类任务文件：命中 `task*` 的 docbook 样式表不算任务

    回归：三场复盘（PK589253/589255/589257）里沙盒每次返回的都是
    /usr/share/sgml/docbook/xsl-stylesheets-1.78.1/html/task.xsl（三万三千
    多字符的样式表），任务正文一次都没读回来，开拓者的 phase 因此卡了
    6~7 个回合、任务分全丢。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读task_1_beijing.md，获取任务信息",
    )
    command = sandbox_command(payload)

    # 文件名特征照旧（描述里点名的 task_1_beijing.md 仍然是第一候选）
    assert "task_1_beijing.md" in command
    assert '-name "task*"' in command
    # 但多了一道扩展名闸门：只认文档，*.xsl/*.xml 这类同名文件被挡在外面
    for ext in TASK_FILE_EXTS:
        assert f'-name "*{ext}"' in command
    assert '-name "*.xsl"' not in command
    assert '-name "*.xml"' not in command


def test_task_executor_prefers_local_api_over_doc_links():
    """执行器优先请求沙盒内的本地接口，文档里抓到的无关外链排在后面

    回归：`find_files(DOC_NAMES)` 从沙盘全盘捞回来的文档里什么外链都有，
    旧实现把抓到的外链排在本地接口前面，`MAX_CALLS` 被这些在无网沙盒里
    调不通的地址耗光，本地接口一次都没被请求到，答案区永远是空的。
    """
    assert "picked = local or [BASE] + " in brain.TASK_EXECUTOR


def test_defender_cashes_out_income_ore_while_wall_quota_open(
    payload_factory, role_factory,
):
    """围墙配额还没铺满时，防守方的铁/铜照样能变现（只有石材留给围墙）

    回归：三场复盘里防守方的金币从 R6/R8/R9 起一路冻结到 R17（连续 9~12 个
    回合为 0），工人背包里却一直躺着可卖的矿石——`wall_quota` 期间经济分支
    整段被跳过，采集分支又先就地补石材，矿石一直没有出口。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        team_type="defender",
        roles=[
            role_factory(
                10012, WORKER, 20, 17, backPackCapability=100,
                backpack=[IRON_MINE] * 2,
            ),
        ],
        # 地图上有石矿 -> 防守方的围墙配额生效（首段墙还没铺）；
        # 石矿远在另一头够不到，工人这一回合只能先把手头的铁矿变现
        zones=[(STONE_MINE, 35, 5), (VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "sell",
        "name": IRON_MINE,
        "num": 2,
    }


def test_defender_stops_hoarding_stone_for_first_wall(
    payload_factory, role_factory,
):
    """围墙配额还欠着时，工人凑够砌墙的那一块就回去施工，不再攒一整批

    回归：防守方为了攒够 STONE_BATCH 一批石材，在矿点和基地之间往返十几个
    回合，首段围墙拖到 R15 才出现（PK589253），另外两场更是全程 0 段——
    而 `wall_quota` 没解除之前，经济线整段被压住、金币一直冻结在 0。
    """
    payload = payload_factory(
        round_no=1,
        gold=0,
        team_type="defender",
        roles=[
            role_factory(
                10012, WORKER, 13, 27, backPackCapability=100,
                backpack=[WALL_MATERIAL],
            ),
        ],
        # 石矿就在手边：旧策略会继续采到 STONE_BATCH 块才动身去建墙
        zones=[(STONE_MINE, 12, 27)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "build",
        "targetPos": [{"x": 13, "y": 26}],
        "name": WALL,
    }


# === issue #42：任务止损 / 卖矿收入 / 开局围墙（PK589649/589653） ===


def _stuck_task_payload(
    payload_factory, role_factory, phase_task: str, round_no: int,
) -> dict:
    """构造"任务进行中、沙盒回读同一份任务文件"的局面（开拓者守在任务点旁）

    开拓者与任务点重合，它会一直在任务点周围一格内等答案（离开会强制结束
    任务）；基地旁留一座武器塔，便于观察它被放回去之后往哪走。
    """
    payload = payload_factory(
        round_no=round_no,
        gold=0,
        roles=[
            role_factory(10011, PIONEER, 14, 14, backPackCapability=40),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    # 沙盒每回合回读回来的都是这份任务文件（exitCode:0，答案区没有取数证据）
    payload["lastCmdResult"] = _sandbox_result(
        phase_task,
        "# 自进化任务 A-1：查询北京文化遗产\n## 任务背景\n"
        "请阅读task_1_beijing.md，获取任务信息\n",
    )
    return payload


def test_task_loop_breaker_releases_pioneer_after_repeated_sandbox_output(
    payload_factory, role_factory,
):
    """沙盒连续回读同一份文件时熔断：不提交任务原文、停发沙盒命令、开拓者回防

    回归：PK589649/589653 里沙盒从 R11 起连续 6~7 个回合返回逐字相同的输出
    （exitCode:0，但没有取数证据），开拓者被读文件死循环占死——任务分丢光，
    这名劳动力也一起白搭（任务书 5 章：单个任务时限 15 回合）。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    weapon = Pos(9, 24)
    waiting, released = [], None

    for offset in range(TASK_LOOP_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        commands, _ = decide(payload)
        if offset < TASK_LOOP_LIMIT - 1:
            waiting.append((commands, payload))
        else:
            released = (commands, payload)

    # 还没到止损线时继续守在任务点上等答案：不提交任务原文，也不放弃任务
    for commands, payload in waiting:
        command = commands.get("10011")
        # 等待期间不能交卷（沙盒没取到数）；但可以在任务圈内挪步（S1）
        assert command is None or command["action"] == "move"
        if command is not None and command["action"] == "move":
            step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格
        assert sandbox_command(payload) != ""

    # 第 TASK_LOOP_LIMIT 次读到同一份输出：放弃任务，开拓者回基地跟队
    commands, payload = released
    assert commands["10011"]["action"] == "move"
    step = Pos(
        commands["10011"]["targetPos"][0]["x"],
        commands["10011"]["targetPos"][0]["y"],
    )
    assert distance(step, weapon) < distance(Pos(14, 14), weapon)
    # 放弃之后不再下发读文件命令（避免把同一个死循环再跑一遍）
    assert sandbox_command(payload) == ""


def test_task_survives_repeated_failed_fetches_until_fail_limit(
    payload_factory, role_factory,
):
    """沙盒在取数、只是地址全猜错时，不按"读文件死循环"提前熔断（S1）

    回归：PK591771/PK591786 的 R11–R14，沙盒读题成功（key=yes、docs=2）但
    8 次请求全 404（api=0、fail=8）。地址是照文档猜的、猜法又是确定性的，
    同一份任务文件每回合算出来的候选清单逐字相同，输出因此逐字相同——看门狗
    r0→r3 于是撞上 `TASK_LOOP_LIMIT`（3），任务在接上后的第 4 个回合就被熔断
    （`rounds` 才 4、`fails` 才 3，取数连败线与超时线都还没到），LLM 兜底只
    来得及问两次，剩下的任务窗连同两个任务点一起作废。这种回合该走取数连败
    止损线（`TASK_API_FAIL_LIMIT`，比读文件循环线宽）：熔断发生在它到线的那
    一回合，而不是第 3 个回合。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    weapon = Pos(9, 24)

    for offset in range(TASK_API_FAIL_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        # 每回合试的是同一批地址（清单逐字相同），8 次请求全部 404
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            "".join(
                f"{TASK_API_FAIL_MARKER} http://localhost:8899/api/task/1"
                " HTTPError 404\n"
                for _ in range(8)
            ),
        )
        commands, _ = decide(payload)

        if offset < TASK_API_FAIL_LIMIT - 1:
            # 读文件循环线到不了：沙盒这一回合真在取数（`[APIFAIL]` 是取数证据）
            assert brain._TASK_WATCH is not None
            assert brain._TASK_WATCH.repeats < TASK_LOOP_LIMIT
            assert sandbox_command(payload) != ""  # 任务还在做，没有被提前止损
            command = commands.get("10011")
            assert command is None or command["action"] == "move"
            if command is not None and command["action"] == "move":
                step = Pos(
                    command["targetPos"][0]["x"],
                    command["targetPos"][0]["y"],
                )
                assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格
        else:
            # 到取数连败止损线：放弃任务，开拓者回基地跟队
            assert sandbox_command(payload) == ""
            assert commands["10011"]["action"] == "move"
            step = Pos(
                commands["10011"]["targetPos"][0]["x"],
                commands["10011"]["targetPos"][0]["y"],
            )
            assert distance(step, weapon) < distance(Pos(14, 14), weapon)


def test_task_survives_scan_output_that_never_calls_the_api(
    payload_factory, role_factory,
):
    """执行器跑过却一次请求都没发时，同样走取数连败线而不是读文件循环线（S1）

    回归：PK591684 的 R11–R13 与 PK591772 的 R12–R13。沙盒输出里
    `[SCAN] docs=2 urls=2 key=yes`、`[exitCode:0]` 全都正常，任务文件与接口
    文档都读到了，可 `api=0`、一行 `[APIFAIL]` 也没有——执行器确实开工了
    （`[SCAN]` 是它打的第一行），却连一次请求都没发出去（没找到任务文件，
    取数循环整段跳过）。旧判据只认"有 `[APIFAIL]`"，这种回合于是被归进
    "沙盒没在取数"：输出逐字相同，`repeats` 每回合累加，接上任务后的第 3 个
    回合就撞上 `TASK_LOOP_LIMIT` 被熔断（PK591684 的 R13：`watch=r3/t4/f3`，
    超时线与取数连败线都还远），重试与 LLM 兜底一次都没轮上。

    这里锁两件事：熔断发生在 `TASK_API_FAIL_LIMIT` 那一回合（不是第 3 个
    回合），且途中每一回合 `repeats` 都没到 `TASK_LOOP_LIMIT`。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    weapon = Pos(9, 24)

    for offset in range(TASK_API_FAIL_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        # 逐字相同的执行器输出：读到了文档，却一条取数记录都没有
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            f"{TASK_SCAN_MARKER} tasks=0 docs=2 urls=2 key=yes\n"
            f"{TASK_DOC_MARKER} 自进化任务 A-1 查询北京文化遗产\n"
            f"{TASK_END_MARKER}\npwd\n/\ntask_1_beijing.md\n",
        )
        commands, _ = decide(payload)

        if offset == TASK_API_FAIL_LIMIT - 1:
            # 到取数连败止损线：放弃任务，开拓者回基地跟队
            assert sandbox_command(payload) == ""
            assert commands["10011"]["action"] == "move"
            step = Pos(
                commands["10011"]["targetPos"][0]["x"],
                commands["10011"]["targetPos"][0]["y"],
            )
            assert distance(step, weapon) < distance(Pos(14, 14), weapon)
        else:
            # 还没到止损线：任务还在做，开拓者守在任务点旁，不把文档当答案交
            assert brain._TASK_WATCH is not None
            assert brain._TASK_WATCH.repeats < TASK_LOOP_LIMIT
            assert sandbox_command(payload) != ""
            command = commands.get("10011")
            assert command is None or command["action"] == "move"
            if command is not None and command["action"] == "move":
                step = Pos(
                    command["targetPos"][0]["x"], command["targetPos"][0]["y"],
                )
                assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格


def test_llm_command_round_is_not_a_read_file_loop(
    payload_factory, role_factory,
):
    """上一回合跑的是 LLM 给的命令时，不按"读文件死循环"熔断任务（S1）

    回归：PK591783 的 R13 与 PK591806 的 R11–R17。沙盒跑的是 LLM 给的一条
    命令（`cat task_1_alpha.md` 走错路径、回读同一份文档），输出里自然没有
    `[SCAN]`——那是"这一回合没派执行器去取数"，不是"读文件读不出新东西"。
    旧看门狗把它归进读文件死循环：输出逐字相同，`repeats` 每回合累加，
    接上任务后的第 3 个回合就熔断，`api=0` 恒 0 的任务连同这名劳动力一起
    丢掉，执行器的取数链路一次都没轮上。这类回合该走取数连败线
    （`TASK_API_FAIL_LIMIT`，比读文件循环线宽），LLM 的六次求助与执行器的
    重试才有机会跑完。
    """
    phase_task = "请阅读task_1_alpha.md，获取任务信息"
    token = _task_token(phase_task)
    brain._TASK_LLM_STATE.clear()
    brain._TASK_WATCH = None

    for offset in range(TASK_LOOP_LIMIT + 1):
        round_no = 11 + offset
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, round_no,
        )
        # LLM 给的命令走错了路径，每回合回读回来的都是同一行报错
        payload["lastCmdResult"] = _sandbox_result(
            phase_task, "cat: task_1_alpha.md: No such file or directory\n",
        )
        # 上一回合的沙盒跑的就是这条命令（`_llm_task_command` 记下的回合号）
        brain._TASK_LLM_STATE[token] = {
            "prompts": offset + 1, "pending_cmd": "", "cmd_round": round_no - 1,
            "answer": "",
        }

        commands, _ = decide(payload)

        assert brain._TASK_WATCH is not None
        assert brain._TASK_WATCH.repeats < TASK_LOOP_LIMIT
        assert sandbox_command(payload) != ""  # 任务还在做，没有被提前止损
        command = commands.get("10011")
        assert command is None or command["action"] == "move"
        if command is not None and command["action"] == "move":
            step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格


def test_scan_output_without_api_calls_counts_as_fetch_failure():
    """判据 3：`[SCAN]` 在、取数记录不在 = 取数失败；读到数就不算（S1）

    这个判据只用来决定看门狗按哪条止损线走，所以两边界都要锁住：执行器真
    取到数时（`[API]` 在）不能误判成失败，否则一个正常取数的任务会被算进
    取数连败、提前放弃。
    """
    scan_only = (
        f"{TASK_SCAN_MARKER} tasks=0 docs=2 urls=2 key=yes\n"
        f"{TASK_DOC_MARKER} 自进化任务 A-1\n{TASK_END_MARKER}\n"
    )
    assert brain._task_fetch_failed(scan_only) is True
    # 试过地址但全失败：老判据，照旧算失败
    assert brain._task_fetch_failed(
        f"{TASK_API_FAIL_MARKER} http://localhost:8899/x HTTPError 404\n",
    ) is True
    # 取到过数：哪怕后面还跟着失败，这一回合也算有进展
    assert brain._task_fetch_failed(
        f"{TASK_DATA_MARKER} http://localhost:8899/x => 40\n"
        f"{TASK_API_FAIL_MARKER} http://localhost:8899/y HTTPError 404\n",
    ) is False
    # 沙盒只回读了任务文件（没有执行器痕迹）：仍是读文件死循环那条线
    assert brain._task_fetch_failed("请阅读task_1_beijing.md，获取任务信息\n") is False


def _executor_rotate():
    """从生成的沙盒脚本里取出候选地址的轮转函数（`rotate`）"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]  # 去掉挑解释器那半句
    match = re.search(
        r"def rotate\(items, keep, offset\):.*?(?=\nfiles = task_files\(\))",
        script, re.S,
    )
    assert match
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102
    return namespace["rotate"]


def test_executor_rotates_fallback_candidates_across_rounds():
    """重试要换地址：候选清单按回合轮转，文档样例永远排在最前面（S1）

    复盘里沙盒连着几个回合输出逐字相同、api 恒 0——每回合算出来的候选清单
    一样，被 `TASK_API_MAX_CALLS` 截断后试的还是同一批地址，重试等于把同一批
    猜错的地址又试了一遍。轮转只动兜底的那一段，文档里给出的样例地址照旧最先试。
    """
    rotate = _executor_rotate()
    items = ["doc", "doc/city", "a", "b", "c"]

    # 前 keep 条（文档给出的样例地址）固定不动，其余按回合号轮转
    assert rotate(items, 2, 0) == items
    assert rotate(items, 2, 1) == ["doc", "doc/city", "b", "c", "a"]
    assert rotate(items, 2, 2) == ["doc", "doc/city", "c", "a", "b"]
    assert rotate(items, 2, 3) == items  # 转满一圈回到原样

    # 没有兜底候选时不动，轮转也不会把清单里的地址弄丢或弄重
    assert rotate(items[:2], 2, 3) == items[:2]
    assert sorted(rotate(items, 2, 1)) == sorted(items)


def test_sandbox_command_passes_round_number_to_the_executor(
    payload_factory, role_factory,
):
    """沙盒命令把当前回合号传进执行器当轮转量（S1）

    轮转量必须是"这一回合"的回合号：两回合拿到同一个量，候选清单就还是逐字
    相同，多出来的重试回合等于白等。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    commands = {}
    for round_no in (13, 14):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, round_no,
        )
        commands[round_no] = sandbox_command(payload)

    for round_no, command in commands.items():
        assert command != ""
        assert f"KEEP = {TASK_API_KEEP}" in command
        assert f"OFFSET = {round_no}" in command
        # 清单必须真的过一遍轮转函数，否则注入的回合号没有任何作用
        assert "for url in rotate(candidates(text, name, urls), KEEP, OFFSET):" in command

    assert commands[13] != commands[14]  # 两回合试的地址不再是同一批


def test_abandoned_task_still_submits_the_answer_in_hand(
    payload_factory, role_factory,
):
    """止损回合手里的答卷先交掉：答案与止损线撞在同一回合时不再白丢

    回归：LLM 直接给的答案（`_llm_direct_answer`）与答案缓存都不依赖沙盒，
    而"放弃"这一步排在"提交"前面——取数连败到线的那个回合答案才到手时，
    这份答案就跟着任务一起被丢掉，一次提交机会都没有（PK591011 的 R13、
    PK591537 的 R14 都是沙盒卡死后被直接放弃，任务分全丢）。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"

    for offset in range(TASK_API_FAIL_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            f"{TASK_API_FAIL_MARKER} http://localhost:8899/guess{offset}"
            " HTTPError 404\n",
        )
        if offset == TASK_API_FAIL_LIMIT - 1:
            payload["llmResp"] = "ANSWER: 北京故宫"  # 止损这一回合答案才到
        commands, _ = decide(payload)

        if offset < TASK_API_FAIL_LIMIT - 1:
            # 还没到止损线、手里也没有答卷：继续守在任务点旁等答案
            assert commands.get("10011", {}).get("action") != "submitAnswer"
            continue
        # 到线这一回合：先把这份答案交掉（放弃归放弃，答卷不能跟着一起丢）
        assert commands["10011"] == {
            "action": "submitAnswer", "taskAnswer": "北京故宫",
        }


def test_abandoned_task_hands_the_pioneer_to_the_other_task_point(
    payload_factory, role_factory,
):
    """止损之后转去另一个还开着的任务点，而不是直接回基地跟队

    回归：PK591595 里任务1 在 R15 被放弃后开拓者一路回基地，任务2 全程
    "可接/15回合"却再没人接（PK591537 的 R15-R18 同样）。止损放弃的只是
    这一个任务点，另一个点还开着就接着做——但刚放弃的那个点不能回头再接，
    看门狗按任务标识计数，同一个任务的沙盒还会照旧卡住。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    abandoned, other = Pos(14, 14), Pos(20, 14)

    for offset in range(TASK_LOOP_LIMIT):
        payload = payload_factory(
            round_no=11 + offset,
            gold=0,
            roles=[
                role_factory(10011, PIONEER, 14, 14, backPackCapability=40),
                role_factory(10020, GATLING, 9, 24, attackRange=4),
            ],
            tasks=[(14, 14), (20, 14, {"taskType": "自进化类2"})],
            phase_task=phase_task,
        )
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            "# 自进化任务 A-1：查询北京文化遗产\n## 任务背景\n"
            "请阅读task_1_beijing.md，获取任务信息\n",
        )
        commands, _ = decide(payload)

    # 止损这一回合：朝另一个任务点走（武器塔在 (9,24)，回基地是相反方向）
    assert commands["10011"]["action"] == "move"
    step = Pos(
        commands["10011"]["targetPos"][0]["x"],
        commands["10011"]["targetPos"][0]["y"],
    )
    assert distance(step, other) < distance(abandoned, other)


def test_task_timeout_releases_pioneer_when_sandbox_never_answers(
    payload_factory, role_factory,
):
    """沙盒一直解不出答案时按任务时限止损：占满 TASK_TIMEOUT 个回合就放弃

    每回合的输出都不一样（不是死循环，没触发读文件熔断），而且取数是成功的
    （`[API]` 证据在，取数连败那条线同样不触发），只是答案区始终拼不出能过
    提交闸门的内容，这时按回合数兜底——开拓者不再被一个拿不到答案的任务
    永久占死。
    """
    phase_task = "请阅读task_1_beijing.md"
    weapon = Pos(9, 24)

    for offset in range(TASK_TIMEOUT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        # 每回合的输出都不一样：取到数了，但没拼出答案段（`[SOLUTION]` 为空）
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            f"{TASK_DATA_MARKER} http://localhost:8899/heritage => 12 "
            f"（第 {offset} 次取数）\n",
        )
        commands, _ = decide(payload)

        if offset == TASK_TIMEOUT - 1:
            assert sandbox_command(payload) == ""
            assert commands["10011"]["action"] == "move"
            step = Pos(
                commands["10011"]["targetPos"][0]["x"],
                commands["10011"]["targetPos"][0]["y"],
            )
            assert distance(step, weapon) < distance(Pos(14, 14), weapon)
        else:
            # 时限内继续守在任务点旁等答案（可以在圈内挪步，但不能交卷）
            command = commands.get("10011")
            assert command is None or command["action"] == "move"
            if command is not None and command["action"] == "move":
                step = Pos(
                    command["targetPos"][0]["x"], command["targetPos"][0]["y"],
                )
                assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格
            assert sandbox_command(payload) != ""


def test_task_fail_limit_releases_pioneer_after_repeated_api_failures(
    payload_factory, role_factory,
):
    """沙盒连续几回合取数全失败时止损：到 TASK_API_FAIL_LIMIT 就放弃任务

    回归：PK590916/PK591014 里开拓者 R10 接了自进化任务后，沙盒每回合换一批
    `[APIFAIL] ... HTTPError 404` 的地址（一条命令 8 次机会全打光），输出既不
    重样（`repeats` 到不了 `TASK_LOOP_LIMIT`）也取不到数，只能一路拖到
    `TASK_TIMEOUT_ROUNDS`，整段任务窗都白等在任务点上。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    weapon = Pos(9, 24)

    for offset in range(TASK_API_FAIL_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        # 每回合猜的地址都不一样：不是读文件死循环，但同样没取到数
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            f"{TASK_API_FAIL_MARKER} http://localhost:8899/guess{offset}"
            " HTTPError 404 => {\"status\":\"error\",\"code\":404}\n",
        )
        commands, _ = decide(payload)

        if offset == TASK_API_FAIL_LIMIT - 1:
            # 到取数连败止损线：放弃任务，开拓者回基地跟队
            assert sandbox_command(payload) == ""
            assert commands["10011"]["action"] == "move"
            step = Pos(
                commands["10011"]["targetPos"][0]["x"],
                commands["10011"]["targetPos"][0]["y"],
            )
            assert distance(step, weapon) < distance(Pos(14, 14), weapon)
        else:
            # 还没到止损线：继续守在任务点旁等答案，但不交卷（沙盒没取到数）
            command = commands.get("10011")
            assert command is None or command["action"] == "move"
            if command is not None and command["action"] == "move":
                step = Pos(
                    command["targetPos"][0]["x"], command["targetPos"][0]["y"],
                )
                assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格
            assert sandbox_command(payload) != ""


def test_task_fail_streak_resets_when_data_comes_back(payload_factory, role_factory):
    """取数连败是"连续"计数：中间取到一次数就从头数，不会误伤到线的任务

    连撞几回合 404 之后有一回合取到了数（`[API]` 证据在），前面那几回合不该
    再算数；否则一个只是"前几回合运气不好"的任务会被提前放弃。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"

    for offset in range(TASK_API_FAIL_LIMIT - 1):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        payload["lastCmdResult"] = _sandbox_result(
            phase_task,
            f"{TASK_API_FAIL_MARKER} http://localhost:8899/guess{offset}"
            " HTTPError 404\n",
        )
        decide(payload)
        assert brain._TASK_WATCH is not None
        assert brain._TASK_WATCH.fails == offset + 1  # 还差一回合才到止损线

    # 再一回合取到了数：连败计数清零，任务不该被放弃
    ok = _stuck_task_payload(
        payload_factory, role_factory, phase_task, 11 + TASK_API_FAIL_LIMIT - 1,
    )
    ok["lastCmdResult"] = _solution_result(phase_task, "task_1_beijing.md", "北京")
    commands, _ = decide(ok)

    assert brain._TASK_WATCH is not None
    assert brain._TASK_WATCH.fails == 0
    assert sandbox_command(ok) != ""  # 任务还在做，没有被止损掉


def test_task_watchdog_starts_over_for_another_task(payload_factory, role_factory):
    """换了任务就从头计数：上一个任务的死循环不会把新任务一起拖下水"""
    for offset in range(TASK_LOOP_LIMIT - 1):
        decide(_stuck_task_payload(
            payload_factory, role_factory, "请阅读task_1_beijing.md", 11 + offset,
        ))

    # 同一局里的另一个任务（描述不同），沙盒输出的标识也随之改变
    payload = _stuck_task_payload(
        payload_factory, role_factory, "请阅读task_2_shanghai.md", 13,
    )
    commands, _ = decide(payload)

    # 仍在任务点旁等答案（圈内挪步），没有被上一个任务的计数带走
    command = commands.get("10011")
    assert command is None or command["action"] == "move"
    if command is not None and command["action"] == "move":
        step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
        assert distance(step, Pos(14, 14)) <= 1  # 不能离开任务点周围一格
    assert sandbox_command(payload) != ""


def test_income_ore_sells_without_waiting_for_batch(payload_factory, role_factory):
    """铁/铜这类纯收入矿石不攒批：手里有一块就卖给小贩换金币

    回归：两场复盘里防守方的背包一直堆着可卖的 iron/copper，金币却从 R6/R8 起
    冻结到 R17——卖矿卡在"攒够 SELL_BATCH 一批"上，手里那几块永远变不成钱。
    石材是例外（它同时是围墙材料，攒够一批再卖更省回合）。
    """
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=LOW_GOLD_THRESHOLD,  # 金币还没见底，攒批的规则仍然拦着它
        worker_pos=Pos(20, 17),
        backpack=[IRON_MINE] * MINERAL_SELL_THRESHOLD,
        zones=[(VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "sell",
        "name": IRON_MINE,
        "num": MINERAL_SELL_THRESHOLD,
    }


def test_stone_still_waits_for_a_batch(payload_factory, role_factory):
    """防线与塔都齐了时零散石材也卖（与上一个用例同一语义，保留两处入口）"""
    payload = _payload_with_full_defense(
        payload_factory, role_factory,
        gold=LOW_GOLD_THRESHOLD,
        worker_pos=Pos(20, 17),
        backpack=[WALL_MATERIAL] * (SELL_BATCH - 1),
        zones=[(VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10010"]["action"] == "sell"
    assert commands["10010"]["name"] == WALL_MATERIAL


def test_defender_starts_first_wall_before_towers(payload_factory, role_factory):
    """防守方开局：石工先跑防线，铜工只奔矿与变现（V4 的站位分工）

    回归：PK589649 的首段围墙落到 R10、PK589653 全程一段都没有——塔位优先的
    建造分支把两名工人都占在基地旁等金币。V4 把砌墙划给石工（R2）、把挖矿变现
    划给铜工（R3），两个人不再抢同一件事。
    """
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST * 2,  # 够建两座塔：旧策略会把两名工人都派去建塔
        team_type="defender",
        roles=[
            role_factory(10010, WORKER, 11, 22, backPackCapability=100),
            role_factory(10012, WORKER, 5, 23, backPackCapability=100),
        ],
        zones=[(STONE_MINE, 4, 24)],
    )
    turn = Turn.load(payload)
    stone, copper = turn.workers()

    # 石工今天有防线要修：破洞数 >0 且修墙路径要花回合（不是只盯着塔位）
    stone_plan = _day_plan(turn, stone)
    assert stone_plan.holes
    assert stone_plan.repair_rounds > 0

    # 铜矿工人的队列里没有石材（V4："挖石头可能反而有反作用"）
    assert STONE_MINE not in _day_plan(turn, copper).queue


def test_defender_cashes_income_ore_before_opening_wall_line(
    payload_factory, role_factory,
):
    """开局围墙与"卖矿解除金币冻结"撞在一起时，先变现再跑石材线

    手里那几块铁/铜是金币见底时唯一的收入，先绕去石矿只会把最后一点现金流
    拖到后面；围墙线下一回合再来。
    """
    payload = payload_factory(
        round_no=1,
        gold=LOW_GOLD_THRESHOLD - 1,
        team_type="defender",
        roles=[
            role_factory(10010, WORKER, 11, 22, backPackCapability=100),
            role_factory(
                10012, WORKER, 20, 17, backPackCapability=100,
                backpack=[IRON_MINE] * 2,
            ),
        ],
        zones=[(STONE_MINE, 4, 24), (VENDOR, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10012"] == {
        "action": "sell",
        "name": IRON_MINE,
        "num": 2,
    }


def test_defender_leaves_opening_wall_line_after_window(
    payload_factory, role_factory,
):
    """过了开局窗口就不再抢在塔前面：建造顺序回到"先塔后墙"

    开局窗口只覆盖前 `WALL_FIRST_ROUND` 个回合，之后仍按原来的优先级走，
    免得一整天压在石材线上，塔与升级反被拖住。
    """
    payload = payload_factory(
        round_no=WALL_FIRST_ROUND + 1,
        gold=WEAPON_BUILD_COST * 2,
        team_type="defender",
        roles=[
            role_factory(10010, WORKER, 11, 22, backPackCapability=100),
            role_factory(10012, WORKER, 5, 23, backPackCapability=100),
        ],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    # 经济工人这一回合不再被派去采石材，而是照常参与建塔
    assert commands["10012"]["action"] == "move"
    step = Pos(
        commands["10012"]["targetPos"][0]["x"],
        commands["10012"]["targetPos"][0]["y"],
    )
    sites = _calc_tower_sites(Turn.load(payload))
    assert any(distance(step, site) < distance(Pos(5, 23), site) for site in sites)


def test_early_wall_keeps_single_worker_on_towers(payload_factory, role_factory):
    """只剩一名工人时不抢石材线：一双手还是先建塔（围墙等金币见底再补）"""
    payload = payload_factory(
        round_no=1,
        gold=WEAPON_BUILD_COST,
        team_type="defender",
        roles=[role_factory(10010, WORKER, 11, 22, backPackCapability=100)],
        zones=[(STONE_MINE, 4, 24)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "build",
        "targetPos": [{"x": 12, "y": 23}],
        "name": ROCKET,
    }


# === 任务求助 LLM（任务期不占每日额度） ===


def _llm_task_payload(payload_factory, role_factory, phase_task, round_no,
                      evidence: str = "接口文档：GET http://localhost:8899/weather?city="):
    """任务进行中、沙盒已吐回文档、但还没取到数的局面"""
    payload = _stuck_task_payload(
        payload_factory, role_factory, phase_task, round_no,
    )
    payload["lastCmdResult"] = _sandbox_result(phase_task, evidence)
    return payload


def test_task_prompt_carries_task_and_sandbox_evidence(payload_factory, role_factory):
    """取不到数时把任务描述与沙盒证据交给 LLM，并要求 CMD/ANSWER 二选一"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md，查询北京天气"
    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 11)

    _, prompt = decide(payload)

    assert "CMD:" in prompt and "ANSWER:" in prompt
    assert phase_task in prompt
    assert "localhost:8899" in prompt  # 沙盒证据要带上，LLM 才看得到接口


def test_task_prompt_tells_llm_to_authenticate(payload_factory, role_factory):
    """喂给 LLM 的约束里写明鉴权（S3）：401/403 是头没带对，不是答案"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md，查询北京文化遗产"
    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 11)

    _, prompt = decide(payload)

    assert "Authorization" in prompt  # 文档里给的头长什么样
    assert "401" in prompt  # 别把鉴权失败的响应当答案交上去


def test_task_prompt_stops_after_limit(payload_factory, role_factory):
    """同一个任务问满次数后不再继续问（别把任务窗都耗在提问上）"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"

    for _ in range(brain.TASK_LLM_MAX_PROMPTS):
        payload = _llm_task_payload(payload_factory, role_factory, phase_task, 11)
        _, prompt = decide(payload)
        assert prompt  # 前面几次都在问

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    _, prompt = decide(payload)
    assert prompt == ""


def test_llm_command_is_run_in_sandbox(payload_factory, role_factory):
    """LLM 给的 CMD 下一回合作为沙盒命令下发（带任务标识，便于取答案）"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "CMD: curl -s http://localhost:8899/weather?city=beijing"
    decide(payload)

    command = sandbox_command(payload)
    assert "curl -s http://localhost:8899/weather?city=beijing" in command
    assert brain.TASK_MARKER in command  # 包装过，答案区能对上当前任务


def test_shell_command_check_rejects_broken_commands():
    """命令体检：引号成对且单行、不带 heredoc 才算合格（不合格的不拼进沙盒）"""
    assert brain._shell_command_ok('curl -s "http://localhost:8899/weather"')
    assert brain._shell_command_ok("grep -o 'x' task.md")
    assert not brain._shell_command_ok('curl -s "http://localhost:8899/weather')
    assert not brain._shell_command_ok("grep -o 'x task.md")
    assert not brain._shell_command_ok("echo a\necho b")
    assert not brain._shell_command_ok("")


def test_shell_command_check_rejects_heredoc():
    """heredoc 命令一律不合格（S2：单行命令里结束符独占一行这条要求满足不了）

    复盘 PK591011 的 R16：沙盒只回了一句
    `here-document at line 0 delimited by end-of-file (wanted 'EOF')`，
    末标记与末尾的 `:` 被当成 heredoc 正文吞掉；同一场的 R13 更彻底——
    解析阶段的报错让整条命令一个字都没跑，连任务标识都没打印
    （`sandbox=发送但 state=no_marker`）。这种命令不能下发，改走执行器兜底。
    `<<<`（here-string）当场就有内容，单行可跑，照旧放行。
    """
    for command in (
        "cat <<EOF",
        "python3 - <<EOF",
        "cat <<-EOF",
        "cat <<'EOF'",
        'cat <<"EOF"',
        "sed -n '1,5p' task.md <<EOF",
    ):
        assert not brain._shell_command_ok(command), command
    assert brain._shell_command_ok('curl -s http://localhost:8899/w <<< "x"')


def test_script_in_command_finds_the_script_not_its_arguments():
    """`_script_in_command` 认的是"要跑的脚本"，不是解释器/参数/地址（S1）"""
    assert brain._script_in_command("./check") == "./check"
    assert brain._script_in_command("sh ./check --round 3") == "./check"
    assert brain._script_in_command("/bin/sh ./check") == "./check"
    assert (
        brain._script_in_command("sudo bash /tmp/selfEvolutionTask/1-x/check.sh")
        == "/tmp/selfEvolutionTask/1-x/check.sh"
    )
    # 不是"跑脚本"的命令一律返回空串：地址、数据文件、解释器自己的路径
    for command in (
        "curl -s http://localhost:8899/weather?city=beijing",
        "grep -o 'x' ./task.md",
        "cat ./task_1_beijing.md",
        "/bin/cat /etc/hostname",
        "/usr/bin/python3 -c 'print(1)'",
        "bash -c 'cat ./task.md'",
        "",
    ):
        assert brain._script_in_command(command) == "", command


def test_crlf_safe_command_normalizes_the_script_it_runs():
    """跑脚本的命令带上行尾归一化（S1：./check 的 CRLF 坏解释器）

    CRLF 的脚本会让内核把 `\r` 一起读进 shebang，命令只换来一行
    `/bin/sh^M: bad interpreter`（复盘 PK591009 的 R14），一个任务回合白搭。
    归一化排在脚本调用之前，并且先判存在——LLM 猜的路径未必真有那个文件。
    """
    wrapped = brain._crlf_safe_command("./check")
    assert wrapped.endswith("./check")  # 归一化在前，原命令照原样跑
    assert '[ -f "$CRLF_P" ]' in wrapped  # 脚本不在时原命令照跑
    assert "sed -i" in wrapped
    assert wrapped.index("CRLF_P='./check'") < wrapped.index("sed -i")


def test_crlf_safe_command_leaves_other_commands_alone():
    """不是"跑脚本"的命令原样下发（取数命令不能被动过）"""
    for command in (
        "curl -s http://localhost:8899/weather?city=beijing",
        "grep -o 'x' ./task.md",
        "cat ./task_1_beijing.md",
        "/bin/cat /etc/hostname",
        "/usr/bin/python3 -c 'print(1)'",
        "bash -c 'cat ./task.md'",
        "",
    ):
        assert brain._crlf_safe_command(command) == command, command


def test_crlf_safe_command_completes_the_script_path_after_cd():
    """“先 cd 再跑脚本”的命令也要归一化行尾（S2，PK591806 的 R14）

    归一化的前置片段排在整条命令最前面，那时工作目录还没切过去：
    `[ -f ./check ]` 判的是沙盒的工作目录（`/`），脚本明明在
    `/tmp/selfEvolutionTask/1-x` 下，归一化静默跳过，`./check` 照旧以 CRLF
    落地、只换来一行 `/bin/sh^M: bad interpreter`（一个任务回合白搭）。
    """
    wrapped = brain._crlf_safe_command("cd /tmp/selfEvolutionTask/1-x && ./check")
    assert wrapped.startswith("CRLF_P='/tmp/selfEvolutionTask/1-x/check';")
    assert "sed -i" in wrapped
    assert wrapped.endswith("cd /tmp/selfEvolutionTask/1-x && ./check")


def test_script_dir_only_follows_a_cd_before_the_script():
    """只有脚本之前、且带命令分隔符的 `cd <目录>` 才用来补全路径

    补出来的路径不存在时 `_crlf_safe_command` 的 `[ -f ]` 会把归一化跳过，
    所以"补不出来"与"补错了"都不影响原命令照跑；但能补对的那几种要补对：
    脚本之后的 `cd` 与它无关，绝对路径的脚本不需要补。
    """
    assert brain._script_dir("cd /tmp/selfEvolutionTask/1-x && ./check", "./check") == (
        "/tmp/selfEvolutionTask/1-x/check"
    )
    assert brain._script_dir("cd /a && cd b && ./check", "./check") == "/a/b/check"
    assert brain._script_dir("cd /a; ./check --round 3", "./check") == "/a/check"
    assert brain._script_dir("./check && cd /tmp", "./check") == ""
    assert brain._script_dir("/tmp/selfEvolutionTask/1-x/check.sh", "/tmp/x/check.sh") == ""
    assert brain._script_dir("cd '/tmp/self evo' && ./check", "./check") == ""
    assert brain._script_dir("", "") == ""


def test_llm_check_command_reaches_sandbox_with_crlf_fix(
    payload_factory, role_factory,
):
    """LLM 给的 `./check` 下发时先做行尾归一化（S1）"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "CMD: ./check"
    decide(payload)

    command = sandbox_command(payload)
    assert "./check" in command
    assert "sed -i" in command  # 先归一化行尾再跑脚本
    assert command.index("CRLF_P='./check'") < command.index("sed -i")
    assert brain.TASK_MARKER in command  # 包装过，答案区能对上当前任务


def test_llm_command_with_unbalanced_quote_falls_back_to_executor(
    payload_factory, role_factory,
):
    """LLM 给的命令引号不配对时丢弃，改由执行器自己取数（PK590881 的 R14）

    少一个配对引号的整条命令会被 bash 判
    `unexpected EOF while looking for matching '"'`：沙盒输出里连任务标识都
    没有，`_task_answer` 只能判 `no_marker`，这一个回合的沙盒执行与提交机会
    一起白费。
    """
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = 'CMD: curl -s "http://localhost:8899/weather?city=beijing'
    decide(payload)

    command = sandbox_command(payload)
    assert "curl" not in command  # 坏命令没有下发
    assert "PYEOF" in command  # 改走执行器：读任务文件 + 按文档地址取数
    assert brain.TASK_MARKER in command


def test_llm_heredoc_command_falls_back_to_executor(
    payload_factory, role_factory,
):
    """LLM 给的 heredoc 命令丢弃，改由执行器自己取数（PK591011 的 R13/R16）

    单行命令里的 heredoc 收不了尾：结束符要独占一行，包装时另起一行接的
    `echo "[TASK_END]"` 与末尾的 `:` 会被当成正文吞掉，沙盒只回一句
    `here-document at line 0 delimited by end-of-file (wanted 'EOF')`。
    整条命令一个字都没跑，答案区里连任务标识都没有，这一个回合白费。
    """
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_alpha.md"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "CMD: python3 - <<EOF"
    decide(payload)

    command = sandbox_command(payload)
    assert "<<EOF" not in command  # 坏命令没有下发
    assert "PYEOF" in command  # 改走执行器：读任务文件 + 按文档地址取数
    assert brain.TASK_MARKER in command
    assert brain.TASK_END_MARKER in command  # 末标记照旧在，答案区能取出来


def test_task_answer_from_llm_command_output(payload_factory, role_factory):
    """LLM 命令的输出在下一回合直接交卷"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"

    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))
    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "CMD: curl -s http://localhost:8899/weather?city=beijing"
    decide(payload)
    sandbox_command(payload)  # 发出 LLM 给的那条命令

    payload = _llm_task_payload(
        payload_factory, role_factory, phase_task, 13, evidence="晴，26℃",
    )
    commands, _ = decide(payload)

    assert commands["10011"]["action"] == "submitAnswer"
    assert "晴" in commands["10011"]["taskAnswer"]


def test_llm_command_output_that_is_a_document_is_not_submitted(
    payload_factory, role_factory,
):
    """LLM 的 `cat 文档` 把接口文档打回来时不能交卷（PK590836 的 R15）

    这条路上沙盒输出里没有 `[DOC]`/`[TASK_FILE]` 指纹，`_task_text_answer`
    没有可比的东西，只能按"文档长什么样"拦一道（`_task_doc_body`）。交一次
    判 0 分、还烧掉一次提交额度，宁可这一回合不交、等下一份沙盒输出。
    """
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"

    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))
    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "CMD: cat /tmp/selfEvolutionTask/1-unknown-api/API.md"
    decide(payload)
    sandbox_command(payload)  # 发出 LLM 给的那条命令

    payload = _llm_task_payload(
        payload_factory, role_factory, phase_task, 13, evidence=_DOC_TEXT,
    )
    commands, _ = decide(payload)

    assert "10011" not in commands or commands["10011"]["action"] != "submitAnswer"


def test_task_answer_from_llm_direct_answer(payload_factory, role_factory):
    """LLM 直接给答案时不必绕沙盒，下一回合就交卷"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = "ANSWER: beijing: 晴 26℃"
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": "beijing: 晴 26℃",
    }


def test_task_answer_ignores_task_echo_from_llm(payload_factory, role_factory):
    """LLM 把任务原文当成答案返回时不能交（复读判定照样生效）"""
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md，获取任务信息并作答"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
    payload["llmResp"] = f"ANSWER: {phase_task}"
    commands, _ = decide(payload)

    assert "10011" not in commands or commands["10011"]["action"] != "submitAnswer"


def test_task_answer_rejects_error_body_from_llm(payload_factory, role_factory):
    """LLM 把命令的报错当成答案回过来时同样不能交（PK590882/590921 的 R14/R16）

    那两场里 submitAnswer 交的是沙盒的报错原文：R14 是 404 的错误 JSON
    （`{"status":"error","message":"Endpoint not found: /api/docs"}`），R16 是
    `jq: command not found` 这行工具报错。LLM 直接给答案这条路（`ANSWER:`）与
    命令输出那条路（`CMD:`）都要过同一道闸门——少装一道，报错原文就照旧会被
    当成答案交上去，既拿不到分又白烧一次提交额度。
    """
    brain._TASK_LLM_STATE.clear()
    phase_task = "请阅读task_1_beijing.md，查询北京文化遗产"
    decide(_llm_task_payload(payload_factory, role_factory, phase_task, 11))

    for reply in (
        'ANSWER: {"status":"error","message":"Endpoint not found: /api/docs"}',
        "ANSWER: /bin/bash: jq: command not found",
        "ANSWER: cat: task_1_beijing.md: No such file or directory",
    ):
        payload = _llm_task_payload(payload_factory, role_factory, phase_task, 12)
        payload["llmResp"] = reply
        commands, _ = decide(payload)
        # 就地待命时开拓者可以没有指令，但绝不能是 submitAnswer
        assert "10011" not in commands or commands["10011"]["action"] != (
            "submitAnswer"
        ), reply


# === issue #45：V4 日计划（布局 / 目标队列 / 夜战救急） ===


def test_wall_holes_counts_missing_ring(payload_factory, role_factory):
    """C：破洞数 = 规划里的墙 - 已经建好的墙（别处多砌的不抵消正面破洞）

    V4 的第一步是"计算破洞的数量 C"，这里按坐标逐格比对：圈外顺手多砌的
    墙不算数，正面缺的每一段都算破洞。
    """
    order = _calc_wall_order(Turn.load(payload_factory()))

    # 一段墙都没砌：破洞就是整圈
    assert _wall_holes(Turn.load(payload_factory())) == order

    payload = payload_factory(
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(40001, WALL, order[0].x, order[0].y),
            role_factory(40002, WALL, 35, 30),  # 圈外的另一段墙
        ],
    )
    holes = _wall_holes(Turn.load(payload))

    assert order[0] not in holes
    assert len(holes) == len(order) - 1


def test_stone_demand_follows_v4_formula(payload_factory, role_factory):
    """AR：ceil((C + Rmin - BR) / 10)，上限 2（V4 的三个例子逐条复现）

    第一天 C=18 -> AR=2、第二天 8 个破洞 + 2 块石材 -> AR=2、
    第二天 7 个破洞 + 2 块石材 -> AR=1。
    """
    ring = len(_calc_wall_order(Turn.load(payload_factory())))

    def demand(stones: int, walls: int) -> int:
        """砌好 walls 段围墙、背包里有 stones 块石材时的 AR"""
        order = _calc_wall_order(Turn.load(payload_factory()))
        roles = [
            role_factory(
                10010, WORKER, 5, 23, backPackCapability=100,
                backpack=[WALL_MATERIAL] * stones,
            ),
        ]
        roles += [
            role_factory(40000 + index, WALL, pos.x, pos.y)
            for index, pos in enumerate(order[:walls])
        ]
        turn = Turn.load(payload_factory(roles=roles))
        return _stone_demand(turn, _wall_holes(turn))

    # (19+5-0)/10 = 2.4 -> 向上取整 3 -> 上限 2
    assert demand(0, 0) == STONE_PLAN_MAX
    # (8+5-2)/10 = 1.1 -> 2
    assert demand(2, ring - 8) == 2
    # (7+5-2)/10 = 1.0 -> 1
    assert demand(2, ring - 7) == 1
    # 破洞补完、保底石材也留够了：今天不挖石矿
    assert demand(STONE_RESERVE_MIN, ring) == 0


def test_work_queue_replaces_head_with_stone(payload_factory):
    """目标队列：默认 5 个铜矿，AR 个石矿从队首替换进来（V4）"""
    turn = Turn.load(payload_factory())

    assert _work_queue(turn, 0) == (COPPER_MINE,) * QUEUE_TARGETS
    assert _work_queue(turn, 1) == (
        STONE_MINE,
    ) + (COPPER_MINE,) * (QUEUE_TARGETS - 1)
    assert _work_queue(turn, STONE_PLAN_MAX) == (
        (STONE_MINE,) * STONE_PLAN_MAX
        + (COPPER_MINE,) * (QUEUE_TARGETS - STONE_PLAN_MAX)
    )


def test_base_layout_follows_enemy_side(payload_factory, role_factory):
    """布局按来敌方位定向：敌人从左边来时正面/后排整体转到左右两列

    V4 的布局是"基地在右下角"的写法，正面由 `_wall_side_order` 现算，
    所以基地换到别处（或敌人从另一侧来）时同一套坐标自动反转。
    """
    default = _base_layout(Turn.load(payload_factory()))
    assert default is not None
    # 看不到敌人时正面是 up（前面没有敌人时的默认顺序），背面是 down
    assert (default.front, default.back) == ("up", "down")
    assert default.corners == (Pos(9, 25), Pos(12, 25), Pos(9, 22), Pos(11, 22))

    left = _base_layout(Turn.load(payload_factory(
        enemies=[role_factory(20010, WORKER, 3, 24)],  # 敌方单位在基地正左方
    )))
    assert left is not None
    assert (left.front, left.back) == ("left", "right")
    assert left.corners == (Pos(9, 25), Pos(9, 22), Pos(12, 25), Pos(12, 23))

    assert _base_layout(Turn.load(payload_factory(station=None))) is None


def test_layout_stand_covers_five_ring_walls(payload_factory):
    """墙角站位：3×3 修复范围正好盖住 5 段外墙（V4 的"一次修复 5 个墙"）"""
    turn = Turn.load(payload_factory())
    layout = _base_layout(turn)
    assert layout is not None
    ring = {pos for cells in _wall_ring(turn).values() for pos in cells}

    for stand in layout.corners[:3]:
        assert stand is not None
        covered = {
            pos for pos in ring if distance(stand, pos) <= brain.WALL_FIXER_RADIUS
        }
        assert len(covered) == 5


def test_repair_plan_ends_at_r2_stand(payload_factory, role_factory):
    """修墙路径（PT）以 R2 站位为终点（PLE），没有破洞时 PT=0（V4）"""
    turn = Turn.load(payload_factory(
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
    ))
    worker = turn.workers()[0]
    layout = _base_layout(turn)
    assert layout is not None
    r2 = layout.corners[1]

    holes = _wall_holes(turn)
    rounds, end = _repair_plan(turn, worker, holes)
    assert end == r2
    assert rounds >= len(holes) * WALL_BUILD_ROUNDS

    assert _repair_plan(turn, worker, ()) == (0, r2)


def test_day_plan_builds_queue_and_tall(payload_factory, role_factory):
    """当天的计划：破洞 -> AR -> 队列 -> 修墙路径 -> Tall（V4 的规划链）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
        zones=[(STONE_MINE, 4, 24), (COPPER_MINE, 30, 5)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]
    day = _day_plan(turn, worker)
    layout = _base_layout(turn)
    assert layout is not None

    assert day.holes == _wall_holes(turn)
    assert day.stone_mines == STONE_PLAN_MAX  # 整圈破洞 -> 上限 2
    # 队列由石矿打头（V4：石料是防线材料），并按当天的回合预算裁剪成
    # "一天真跑得完"的长度（Dnum 循环），至少留一件事做
    assert day.queue[:1] == (STONE_MINE,)
    assert 0 < day.queue.count(STONE_MINE) <= STONE_PLAN_MAX
    assert day.end_point == layout.corners[1]  # PLE = R2 站位
    assert day.repair_rounds > 0 and day.route_rounds > 0
    assert day.total_rounds == day.repair_rounds + day.route_rounds


def test_plan_extras_follow_v4_priority(payload_factory, role_factory):
    """额外目标优先级：小贩 > 生命药剂 > 武器商店 > 铁矿 > 石头（V4）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, 5, 23, backPackCapability=100, health=MEDICINE_HP - 20,
            ),
        ],
        zones=[(STONE_MINE, 4, 24)],
    )
    turn = Turn.load(payload)
    worker = turn.workers()[0]
    queue = _work_queue(turn, 1)

    # 队列里有铜矿 + 血量偏低：小贩与药剂都在，第 1 天不补修复包
    assert _plan_extras(turn, worker, queue, 0) == (
        VENDOR, MEDICINE, IRON_MINE, STONE_MINE,
    )
    # Tall 贴到白天的回合数上限：一个额外目标都不加
    assert _plan_extras(turn, worker, queue, DAY_PLAN_LIMIT) == ()

    # 血量健康时不买药；第 3 天起才轮到武器商店
    healthy = Turn.load(payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10010, WORKER, 5, 23, backPackCapability=100)],
        zones=[(STONE_MINE, 4, 24)],
    ))
    assert MEDICINE not in _plan_extras(healthy, healthy.workers()[0], queue, 0)
    assert WEAPON_SHOP not in _plan_extras(
        healthy, healthy.workers()[0], queue, 0,
    )


def test_mine_order_follows_day_queue(payload_factory, role_factory):
    """采集顺序按当天队列排：石工石矿打头，铜矿工人以铜矿为主矿（V4）"""
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(10010, WORKER, 5, 23, backPackCapability=100),
            role_factory(10012, WORKER, 20, 23, backPackCapability=100),
        ],
        zones=[(STONE_MINE, 4, 24), (COPPER_MINE, 20, 22), (IRON_MINE, 21, 22)],
    )
    turn = Turn.load(payload)
    stone_worker, copper_worker = turn.workers()
    day = _day_plan(turn, stone_worker)

    # 破洞还没补完：队列里石矿打头，石工先采石
    assert day.stone_mines >= 1
    assert day.queue[0] == STONE_MINE
    assert _mine_order(turn, stone_worker, plan=day)[0] == STONE_MINE
    # 铜矿工人（第 2 名）以铜矿为主矿，铁作次选
    assert _mine_order(turn, copper_worker, plan=day) == ECONOMY_MINE_ORDER


def test_mine_order_switches_to_copper_when_walls_done(
    payload_factory, role_factory,
):
    """破洞补完、保底石材也留够时队列只剩铜矿：石工也转去挖铜（V4）"""
    order = _calc_wall_order(Turn.load(payload_factory()))
    roles = [
        role_factory(40000 + index, WALL, pos.x, pos.y)
        for index, pos in enumerate(order)
    ]
    roles.append(
        role_factory(
            10010, WORKER, 5, 23, backPackCapability=100,
            backpack=[WALL_MATERIAL] * STONE_RESERVE_MIN,
        ),
    )
    turn = Turn.load(payload_factory(
        gold=0, roles=roles,
        zones=[(STONE_MINE, 4, 24), (COPPER_MINE, 20, 22)],
    ))
    worker = turn.workers()[0]
    day = _day_plan(turn, worker)

    assert day.stone_mines == 0
    assert _mine_order(turn, worker, plan=day)[0] == COPPER_MINE


def test_day_three_restocks_wall_fixer(payload_factory, role_factory):
    """第 3 天顺路补围墙修复包：第一次只买 2 个（V4）"""
    base = Turn.load(payload_factory())
    roles = [
        role_factory(10020 + index, kind, pos.x, pos.y, attackRange=4, level=3)
        for index, (kind, pos) in enumerate(
            zip(TOWER_LOADOUT, _calc_tower_sites(base))
        )
    ]
    roles.append(
        role_factory(10010, WORKER, 20, 17, backPackCapability=100),
    )
    payload = payload_factory(
        round_no=ROUNDS_PER_DAY * 2 + 1,  # 第 3 天第 1 个回合
        gold=100,
        roles=roles,
        zones=[(WEAPON_SHOP, 20, 16)],
    )
    payload["weaponShopList"] = [{"name": WALL_FIXER, "price": WALL_FIXER_GOLD}]
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "buy",
        "name": WALL_FIXER,
        "num": FIXER_STOCK_FIRST,
    }


def test_worker_breaks_through_weak_wall(payload_factory, role_factory):
    """无路可走时拆开身边的残墙开路（V4：血量 < 50% 的一级墙算"已经破了"）"""
    worker_pos = Pos(5, 23)
    weak = Pos(4, 23)  # 挡在工人和石矿之间的那一格
    stone = Pos(3, 23)
    enclosure = [pos for pos in get_neighbors(worker_pos) if pos != weak]
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[
            role_factory(
                10010, WORKER, worker_pos.x, worker_pos.y, backPackCapability=100,
            ),
            *[
                role_factory(40000 + index, WALL, pos.x, pos.y)
                for index, pos in enumerate(enclosure)
            ],
            # 一级墙满血 1000，400 血已经低于一半 -> 视为"已经破了"
            role_factory(49999, WALL, weak.x, weak.y, health=400),
        ],
        zones=[(STONE_MINE, stone.x, stone.y)],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "remove",
        "targetPos": [{"x": weak.x, "y": weak.y}],
    }


def test_night_drinks_medicine_when_hurt(payload_factory, role_factory):
    """夜晚血量 <= 50 时喝一剂生命药剂（V4）"""
    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,  # 第 1 天夜里
        roles=[
            role_factory(
                10010, WORKER, 9, 23, backPackCapability=100,
                health=MEDICINE_HP_NIGHT, backpack=[MEDICINE],
            ),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
    )
    commands, _ = decide(payload)

    assert commands["10010"] == {"action": "use", "name": MEDICINE}


def test_night_uses_station_voucher_when_base_critical(
    payload_factory, role_factory,
):
    """夜晚基地血量 < 150 时用基地升级券（V4：回满血 + 升级）"""
    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,
        roles=[
            role_factory(
                10010, WORKER, 9, 23, backPackCapability=100,
                backpack=[STATION_UPGRADE_VOUCHER],
            ),
            role_factory(10020, GATLING, 9, 24, attackRange=4),
        ],
    )
    for role in payload["teamOur"]["roles"]:
        if role["roleType"] == "station":
            role["health"] = 100
    commands, _ = decide(payload)

    assert commands["10010"] == {
        "action": "use",
        "name": STATION_UPGRADE_VOUCHER,
        "targetPos": [{"x": 10, "y": 24}],
    }


def test_night_fixer_repairs_wall_next_to_stand(payload_factory, role_factory):
    """夜晚站在墙角的人用修复包补身边那一段残墙（V4 的角落站位）"""
    layout = _base_layout(Turn.load(payload_factory()))
    assert layout is not None
    stand = layout.corners[0]  # R1：开拓者的站位
    assert stand is not None
    ring = _wall_ring(Turn.load(payload_factory()))
    reachable = sorted(
        (
            pos for cells in ring.values() for pos in cells
            if distance(stand, pos) <= brain.WALL_FIXER_RADIUS
        ),
        key=lambda pos: (pos.x, pos.y),
    )
    target = reachable[0]

    payload = payload_factory(
        round_no=DAY_ROUNDS + 1,
        roles=[
            role_factory(
                10011, PIONEER, stand.x, stand.y, backPackCapability=40,
                backpack=[WALL_FIXER],
            ),
            role_factory(10020, GATLING, stand.x, stand.y - 1, attackRange=4),
            role_factory(49999, WALL, target.x, target.y, health=80),
        ],
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "use",
        "name": WALL_FIXER,
        "targetPos": [{"x": target.x, "y": target.y}],
    }


def test_pioneer_shops_by_priority(payload_factory, role_factory):
    """开拓者收工后按优先级补货：先买 3 张武器升级券1（V4）"""
    base = Turn.load(payload_factory())
    roles = [
        role_factory(10011, PIONEER, 20, 17, backPackCapability=40),
        *[
            role_factory(10020 + index, kind, pos.x, pos.y, attackRange=4)
            for index, (kind, pos) in enumerate(
                zip(TOWER_LOADOUT, _calc_tower_sites(base))
            )
        ],
    ]
    payload = payload_factory(
        round_no=1,
        gold=PIONEER_WEAPON_VOUCHERS * UPGRADE_GOLD + 50,
        roles=roles,
        zones=[(WEAPON_SHOP, 20, 16)],
    )
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "buy",
        "name": WEAPON_UPGRADE_VOUCHER,
        "num": PIONEER_WEAPON_VOUCHERS,
    }


def test_pioneer_shopping_skips_expensive_items(
    payload_factory, role_factory,
):
    """开拓者买不起武器/基地券时先补围墙修复包（V4 的优先级顺延）"""
    base = Turn.load(payload_factory())
    roles = [
        role_factory(10011, PIONEER, 20, 17, backPackCapability=40),
        *[
            role_factory(
                10020 + index, kind, pos.x, pos.y, attackRange=4, level=3,
            )
            for index, (kind, pos) in enumerate(
                zip(TOWER_LOADOUT, _calc_tower_sites(base))
            )
        ],
    ]
    payload = payload_factory(
        round_no=1,
        gold=50,  # 不够一张基地升级券（100 金），够 3 张修复包（30 金）
        roles=roles,
        zones=[(WEAPON_SHOP, 20, 16)],
    )
    payload["weaponShopList"] = [{"name": WALL_FIXER, "price": WALL_FIXER_GOLD}]
    commands, _ = decide(payload)

    assert commands["10011"] == {
        "action": "buy",
        "name": WALL_FIXER,
        "num": FIXER_STOCK,
    }


# === V4 低风险三条：铜工专属语义 / 商店只能最后一站 / n−1 回退 ===


def _two_worker_payload(payload_factory, role_factory, *, round_no=1, gold=0,
                        stone_backpack=0, copper_backpack=0, zones=()):
    """构造"石工 + 铜工"的局面（V4 的两名工人分工）"""
    return payload_factory(
        round_no=round_no,
        gold=gold,
        team_type="challenger",
        roles=[
            role_factory(
                10010, WORKER, 12, 26, backPackCapability=100,
                backpack=[WALL_MATERIAL] * stone_backpack,
            ),
            role_factory(
                10012, WORKER, 20, 17, backPackCapability=100,
                backpack=[COPPER_MINE] * copper_backpack,
            ),
        ],
        zones=list(zones),
    )


def test_copper_worker_end_point_is_vendor(payload_factory, role_factory):
    """铜矿工人的 PLE 是小贩（V4）：他不施工，PT 只算回落脚点的路程"""
    payload = _two_worker_payload(
        payload_factory, role_factory, zones=[(VENDOR, 20, 16)],
    )
    turn = Turn.load(payload)
    copper = turn.workers()[1]

    rounds, end_point = _repair_plan(turn, copper, _wall_holes(turn))

    assert end_point == _nearest_zone(turn, VENDOR, copper.pos)
    assert rounds >= 0  # 从小贩回到 K0/站位的路程


def test_copper_worker_has_no_stone_in_queue(payload_factory, role_factory):
    """铜矿工人的队列里没有石材，也不去砌墙（V4 的铜工语义）"""
    payload = _two_worker_payload(payload_factory, role_factory)
    turn = Turn.load(payload)
    stone_worker, copper = turn.workers()

    assert STONE_MINE not in _day_plan(turn, copper).queue
    assert STONE_MINE in _day_plan(turn, stone_worker).queue
    # 铜工手里有石材也不会去砌墙（他等小贩那趟把矿变现）
    commands, _ = decide(_two_worker_payload(
        payload_factory, role_factory, stone_backpack=0, copper_backpack=SELL_BATCH,
        zones=[(VENDOR, 20, 16)],
    ))
    assert commands["10012"]["action"] == "sell"


def test_shop_visit_waits_until_last_stop(payload_factory, role_factory):
    """武器商店只能当最后一站：白天还早时不往商店跑（V4 的先后硬约束）"""
    # 早上的工人离商店十几格，剩下的白天回合远多于路程 -> 不是最后一站
    early = Turn.load(_two_worker_payload(
        payload_factory, role_factory, zones=[(WEAPON_SHOP, 25, 20)],
    ))
    assert _shop_is_last_stop(early, early.workers()[0]) is False

    # 天黑前只剩几个回合，路程刚好够 -> 可以去
    dusk = Turn.load(_two_worker_payload(
        payload_factory, role_factory, round_no=DAY_ROUNDS - 2,
        zones=[(WEAPON_SHOP, 12, 27)],
    ))
    assert _shop_is_last_stop(dusk, dusk.workers()[0]) is True


def test_day_plan_makes_room_for_selling_when_backpack_full(
    payload_factory, role_factory,
):
    """背包压着一批卖不掉的矿石时，计划优先腾出"去小贩"的一趟（V4 的 n−1）"""
    empty = Turn.load(_two_worker_payload(
        payload_factory, role_factory, zones=[(VENDOR, 20, 16)],
    ))
    loaded = Turn.load(_two_worker_payload(
        payload_factory, role_factory, copper_backpack=SELL_BATCH,
        zones=[(VENDOR, 20, 16)],
    ))
    stone_empty = empty.workers()[0]
    stone_loaded = loaded.workers()[0]
    copper_loaded = loaded.workers()[1]

    # 铜工背着可卖的矿：计划要在小贩那趟把它变现（PLE 就是小贩）
    copper_plan = _day_plan(loaded, copper_loaded)
    assert copper_plan.end_point == _nearest_zone(loaded, VENDOR, copper_loaded.pos)
    # 石工的队列不会因为别人背包里有矿而变长
    assert len(_day_plan(loaded, stone_loaded).queue) <= len(
        _day_plan(empty, stone_empty).queue
    )


# === issue #46：任务提交内容 / 沙盒探测收敛（PK590243 / PK590252） ===


def test_task_answer_never_submits_a_file_path(payload_factory, role_factory):
    """答案区里只有一条文件路径时绝不提交（PK590252 的 R17 交的就是路径）

    回归：开拓者找到了任务文件，却把
    "/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md"
    当成答案交了上去（`submitAnswer <路径>`），Judge 判 0 分、R18/R19 的 phase
    一直没清除，任务 2 也跟着连锁未接。同一个局面下换成真正的取数结果照样交卷
    ——路径闸门不会误伤正常答案。
    """
    phase_task = "请阅读task_1_beijing.md，获取任务信息"
    brain._TASK_LLM_STATE.clear()  # 上一个用例留下的 LLM 答案不参与本用例
    payload = _stuck_task_payload(payload_factory, role_factory, phase_task, 11)

    payload["lastCmdResult"] = _solution_result(
        phase_task,
        "task_1_beijing.md",
        "/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md",
    )
    commands, _ = decide(payload)

    command = commands.get("10011")
    assert command is None or command["action"] != "submitAnswer"
    # 路径不是答案：沙盒命令照旧下发，下一回合重新取数
    assert sandbox_command(payload) != ""

    # 取数取回来的才是答案：同一个局面下换成真正的取数结果照样交卷
    payload["lastCmdResult"] = _solution_result(
        phase_task, "task_1_beijing.md", '{"city": "北京", "count": 7}',
    )
    commands, _ = decide(payload)
    assert commands["10011"] == {
        "action": "submitAnswer",
        "taskAnswer": '{"city": "北京", "count": 7}',
    }


def test_llm_answer_is_not_submitted_when_it_is_a_path(
    payload_factory, role_factory,
):
    """LLM 回的 ANSWER 是一条文件路径时同样不提交（S1 的兜底闸门）

    复盘里 R17 那次提交与前几场的"提交任务原文"是同一类错误：交上去的东西
    不是沙盒里取到的答案。LLM 直接给答案这条路也走同一道闸门。
    """
    phase_task = "请阅读沙盒里的任务说明并作答"
    brain._TASK_LLM_STATE.clear()  # 上一个用例留下的 LLM 答案不参与本用例
    payload = payload_factory(
        round_no=11,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    payload["llmResp"] = (
        "ANSWER: /tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md"
    )
    commands, _ = decide(payload)

    command = commands.get("10011")
    assert command is None or command["action"] != "submitAnswer"


def test_task_stops_submitting_after_limit(payload_factory, role_factory):
    """同一份答案交满 TASK_SUBMIT_LIMIT 次仍没被放行时止损，开拓者回基地

    回归：PK590252 的 R17 交上文件路径之后，R18/R19 开拓者仍被这个任务占着
    （phase 一直没清除）。错答案再交第三遍同样放行不了，早点把开拓者还给
    战斗调度，比让它在任务点上耗到任务时限（`TASK_TIMEOUT_ROUNDS`）划算。
    """
    phase_task = "请阅读task_1_beijing.md"
    weapon = Pos(9, 24)
    brain._TASK_LLM_STATE.clear()  # 上一个用例留下的 LLM 答案不参与本用例
    # 沙盒里早就解出过这份任务：每一回合手里都有能交的答案
    brain._TASK_ANSWER_CACHE["task_1_beijing.md"] = '{"city": "北京", "count": 7}'

    for offset in range(TASK_SUBMIT_LIMIT):
        payload = _stuck_task_payload(
            payload_factory, role_factory, phase_task, 11 + offset,
        )
        # 每回合的沙盒输出都不一样：这里走的是提交闸门，不是读文件死循环
        # （同一份输出连着出现 `TASK_LOOP_LIMIT` 次会先触发读文件熔断）
        payload["lastCmdResult"] = _sandbox_result(
            phase_task, f"第 {offset} 次搜索，仍无任务文件\n",
        )
        commands, _ = decide(payload)
        assert commands["10011"] == {
            "action": "submitAnswer",
            "taskAnswer": '{"city": "北京", "count": 7}',
        }

    # 第 TASK_SUBMIT_LIMIT+1 个回合：不再交卷，开拓者回基地跟队
    payload = _stuck_task_payload(
        payload_factory, role_factory, phase_task, 11 + TASK_SUBMIT_LIMIT,
    )
    payload["lastCmdResult"] = _sandbox_result(
        phase_task, f"第 {TASK_SUBMIT_LIMIT} 次搜索，仍无任务文件\n",
    )
    commands, _ = decide(payload)

    assert commands["10011"]["action"] == "move"
    step = Pos(
        commands["10011"]["targetPos"][0]["x"],
        commands["10011"]["targetPos"][0]["y"],
    )
    assert distance(step, weapon) < distance(Pos(14, 14), weapon)
    assert sandbox_command(payload) == ""


def test_sandbox_probe_stops_after_limit(payload_factory, role_factory):
    """描述里没有文件名时最多探 TASK_PROBE_LIMIT 次，之后直接把任务根目录交给执行器

    回归：PK590252 的 R12-R17 六个回合里沙盒反复扫根目录与系统 docs
    （R13 整条命令 `[TIMEOUT]`、R14/R16 两次输出逐字相同），开拓者被占死到
    日志结束。探测只是给执行器指路，认不出文件名时该由执行器自己去捞。
    """
    phase_task = "请阅读沙盒里的任务说明并作答"
    brain._TASK_LLM_STATE.clear()  # 上一个用例留下的 LLM 答案不参与本用例
    token = _task_token(phase_task)
    # 探测回来的清单里一个任务文件都没有（只剩工作目录诊断）
    empty_probe = (
        f"[exitCode:0]\n{TASK_PROBE_MARKER}{token}\n"
        f"pwd\n/\n{TASK_END_MARKER}\n"
    )
    probes = 0
    final_command = ""
    for offset in range(TASK_PROBE_LIMIT + 1):
        payload = payload_factory(
            round_no=11 + offset,
            gold=0,
            roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
            tasks=[(14, 14)],
            phase_task=phase_task,
        )
        # 第一个回合还没有沙盒输出（命令刚下发），之后每回合都是空探测结果
        payload["lastCmdResult"] = empty_probe if offset else ""
        decide(payload)  # 先刷新看门狗，再要沙盒命令（服务器就是按这个顺序调的）
        final_command = sandbox_command(payload)
        if TASK_PROBE_MARKER in final_command:
            probes += 1

    assert probes == TASK_PROBE_LIMIT
    # 用满次数后换成执行器：带上本任务标识，并把任务根目录交给它
    assert TASK_PROBE_MARKER not in final_command
    assert TASK_MARKER in final_command
    assert TASK_ROOTS[0] in final_command


def test_sandbox_searches_task_root_and_skips_system_docs(
    payload_factory, role_factory,
):
    """找任务文件先扫任务根目录、系统文档树整段跳过（S2）

    回归：PK590252 的 R14/R16 两次读回来的都是
    /usr/share/doc/uom-se-1.0.4/README.md（与任务无关的库说明），照着它拼出来的
    地址一次都没取到数；R13 的全盘 find 则直接把整条命令拖到 `[TIMEOUT]`。
    """
    brain._TASK_LLM_STATE.clear()  # 上一个用例留下的 LLM 答案不参与本用例
    payload = payload_factory(
        round_no=1,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task="请阅读task_1_beijing.md",
    )
    command = sandbox_command(payload)

    # 回读任务文件：先扫任务根目录，那里没有才退到全盘
    assert f"find {TASK_ROOTS[0]} " in command
    assert 'if [ -z "$found" ]' in command
    assert "find / " in command
    # 系统文档树不扫：README 与 docbook 样式表都在这两棵树里
    assert '-path "/usr/share/doc"' in command
    assert '-path "/usr/share/sgml"' in command
    # 执行器同样把任务根目录当主搜索路径、接口文档也不去系统文档树里捞
    assert f"ROOTS = {TASK_ROOTS!r}" in command


# === 任务链路日志（#67：让复盘看到"卡在哪一步"） ===


def _task_payload(payload_factory, role_factory, phase_task, round_no, output=""):
    """构造"开拓者正在做某个任务"的局面（可带上一回合的沙盒输出）"""
    payload = payload_factory(
        round_no=round_no,
        gold=0,
        roles=[role_factory(10011, PIONEER, 14, 14, backPackCapability=40)],
        tasks=[(14, 14)],
        phase_task=phase_task,
    )
    if output:
        payload["lastCmdResult"] = _sandbox_result(phase_task, output)
    return payload


def test_task_brief_reports_reason_codes(payload_factory, role_factory):
    """task_brief 把"为什么没交卷"写成可解析字段（state=...）"""
    phase_task = "请阅读task_1_beijing.md"
    brain._TASK_LLM_STATE.clear()

    # 沙盒还没回本任务的输出
    turn = Turn.load(_task_payload(payload_factory, role_factory, phase_task, 11))
    brief = brain.task_brief(turn)
    assert "state=no_marker" in brief
    assert 'phase="请阅读task_1_beijing.md"' in brief

    # 沙盒回了输出但没取到数（执行器 APIFAIL 循环的典型形态，#67 的根因）
    turn = Turn.load(_task_payload(
        payload_factory, role_factory, phase_task, 12,
        output="[APIFAIL] http://localhost:8899） InvalidURL\n任务原文若干行\n",
    ))
    brief = brain.task_brief(turn)
    assert "state=no_api_data" in brief
    assert "fail=1" in brief  # 取数失败次数也进日志

    # 执行器取到数：state 变成 ok、api 计数 1
    turn = Turn.load(_task_payload(
        payload_factory, role_factory, phase_task, 13,
        output=f"{TASK_DATA_MARKER} http://localhost:8899/x => 12\n"
               f"{TASK_SOLUTION_MARKER}task_1_beijing.md\n晴，26℃\n{TASK_SOLUTION_END}\n",
    ))
    brief = brain.task_brief(turn)
    assert "state=ok" in brief
    assert "api=1" in brief

    # 答案是文档正文、输出里又没有 `[DOC]`/`[TASK_FILE]` 指纹可比（LLM 那条路
    # 的形态）时，改由"文档长什么样"的闸门拦下：state=doc_body（S2）
    turn = Turn.load(_task_payload(
        payload_factory, role_factory, phase_task, 14,
        output=f"{TASK_DATA_MARKER} http://localhost:8899/docs => 4\n"
               f"{TASK_SOLUTION_MARKER}task_1_beijing.md\n"
               f"{_DOC_TEXT}\n{TASK_SOLUTION_END}\n",
    ))
    brief = brain.task_brief(turn)
    assert "state=doc_body" in brief


def test_task_brief_without_task(payload_factory):
    """没有任务时不打任务字段（避免日志里出现无意义的行）"""
    turn = Turn.load(payload_factory())
    assert brain.task_brief(turn) == "phase=- state=no_task"


def test_executor_template_has_no_placeholders():
    """执行器命令里的占位符必须全部替换掉（漏一个脚本就整个跑不起来）"""
    command = _task_executor("task_1_beijing.md")
    assert re.findall(r"__[A-Z_]+__", command) == []


def _executor_refine_url():
    """从生成的沙盒脚本里取出 URL 净化部分（`HOST_SAFE` + `cut_host` + `refine_url`）"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]  # 去掉挑解释器那半句
    match = re.search(r"HOST_SAFE = \(.*?(?=\ndef endpoints\()", script, re.S)
    assert match
    namespace: dict = {}
    exec("import urllib.parse\n" + match.group(0), namespace)  # noqa: S102
    return namespace["refine_url"]


def test_executor_refine_url_survives_cjk_and_backticks():
    """沙盒执行器的 URL 净化：中文标点/反引号不再让 urllib 抛 InvalidURL（#67）

    复盘里 R12–R17 连续 6 回合 `APIFAIL ... ），API InvalidURL`，任务因此
    8 个回合读不到题面。这里直接从生成的脚本里取出 refine_url 验证行为。
    """
    refine = _executor_refine_url()

    assert refine("http://localhost:8899/weather?city=北京。") == (
        "http://localhost:8899/weather?city=%E5%8C%97%E4%BA%AC"
    )
    assert refine("`http://localhost:8899/x`") == "http://localhost:8899/x"
    assert refine("http://localhost:8899/a），") == "http://localhost:8899/a"
    assert refine("不是地址") == ""


def test_executor_refine_url_cuts_junk_in_host():
    """主机名里混进来的行文要被截断（PK590836/PK590849 的 R12–R17）

    复盘里的诊断行是 `[APIFAIL] http://localhost:8899`），API InvalidURL`：
    文档把本地接口写在句子里，端口后面紧跟反引号与全角标点，整段连一个 `/`
    都没有，urlsplit 于是把它全当成 netloc；`quote` 只覆盖 path/query，
    杂质原样进了地址——只剥两端的标点救不回来（末尾是 ASCII 的 `API`，没得剥）。
    `cut_host` 在第一个不属于主机名的字符处截断，地址回到本地接口的本相。
    """
    refine = _executor_refine_url()

    assert refine("http://localhost:8899`），API") == "http://localhost:8899"
    assert refine("http://localhost:8899（本地接口）") == "http://localhost:8899"
    # 查询词里的中文在主机名之外，照旧百分号编码，不受截断影响
    assert refine("http://localhost:8899?city=北京") == (
        "http://localhost:8899?city=%E5%8C%97%E4%BA%AC"
    )
    # 主机名整段都是杂质时拼不出地址：宁可不试，也不交一个注定 InvalidURL 的地址
    assert refine("http://`），API") == ""
    assert refine("http://localhost:8899 x") == "http://localhost:8899"


def _executor_queries():
    """从生成的沙盒脚本里取出查询词与候选地址的构造（`queries` + `candidates`）"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]  # 去掉挑解释器那半句
    skips = re.search(r"SKIP_WORDS = \([^)]*\)", script)
    body = re.search(r"def queries\(text, name\):.*?\n    return out\n", script, re.S)
    assert skips and body
    namespace: dict = {
        "BASE": brain.TASK_API_DEFAULT,
        "QUERY_MAX": brain.TASK_API_QUERY_MAX,
        "SUFFIXES": brain.TASK_API_PATH_SUFFIXES,
    }
    prelude = "import os\nimport re\n" + skips.group(0) + "\n"
    exec(prelude + body.group(0), namespace)  # noqa: S102
    return namespace["queries"], namespace["candidates"]


def test_executor_query_words_drop_task_file_stem():
    """查询词取文件名里有信息量的一段，不再用整段任务文件名（PK590884/PK590920）

    两份报告的 P0-1 里，沙盒整轮 404 的诊断行都是
    `[APIFAIL] http://localhost:8899/task_1_alpha`：任务文件的整段词干被当成
    "查询词"拼进了样例地址。文档问的是城市名，拿文件名去问只会 404，而
    `QUERY_MAX` 只有两个名额，它先占掉一个，文档里真正有用的词就上不了场。
    """
    queries, _ = _executor_queries()

    # task_1_beijing.md -> beijing：文件名里的英文词才是文档要的查询值
    assert queries("请查询该城市的文化遗产", "task_1_alpha.md") == ["alpha"]
    # 正文里再提到一次这个文件名时，同样不该把它当成查询词
    assert queries("请阅读task_1_alpha.md", "task_1_alpha.md") == ["alpha"]


def test_executor_candidates_never_append_task_file_stem():
    """候选地址里不再有"根地址 + 任务文件名"这条必然 404 的拼接"""
    _, candidates = _executor_queries()

    out = candidates("请查询该城市的文化遗产", "task_1_alpha.md", [])
    assert f"{brain.TASK_API_DEFAULT}/task_1_alpha" not in out
    assert f"{brain.TASK_API_DEFAULT}/alpha" in out


def test_executor_candidates_swap_numeric_path_sample():
    """样例地址末尾是数字时补一条"换成查询词"的候选（PK591787 的 R16）

    接口文档里的调用样例常常写成 `http://localhost:8899/api/task/1` 这样的
    模板，照抄下来只会拿到一行 `Endpoint not found:/api/task/1`（R16 的
    404 诊断行），而把查询词接在样例后面（`.../api/task/1/beijing`）同样不是
    文档里那个接口。这条候选把末尾那段数字换成自己的查询词，正解
    `.../api/task/beijing` 因此进得了取数清单。
    """
    _, candidates = _executor_queries()

    out = candidates(
        "请查询该城市的文化遗产", "task_1_beijing.md",
        ["http://localhost:8899/api/task/1"],
    )
    # 样例本身排在最前（先照文档原样试一次），两条改写都要在清单里
    assert out[0] == "http://localhost:8899/api/task/1"
    assert "http://localhost:8899/api/task/beijing" in out
    assert "http://localhost:8899/api/task/1/beijing" in out
    # 清单没有被取数名额截断：正解确实轮得到（一条命令最多请求 MAX_CALLS 次）
    assert len(out) <= brain.TASK_API_MAX_CALLS


def test_executor_candidates_leave_real_paths_alone():
    """样例地址不是模板（末尾没有数字）时不凭空造候选，清单维持原样"""
    _, candidates = _executor_queries()

    out = candidates(
        "请查询该城市的文化遗产", "task_1_beijing.md",
        ["http://localhost:8899/api/city"],
    )
    assert out == [
        "http://localhost:8899/api/city",
        "http://localhost:8899/api/city/beijing",
        f"{brain.TASK_API_DEFAULT}/",
        f"{brain.TASK_API_DEFAULT}/api",
        f"{brain.TASK_API_DEFAULT}/docs",
    ]
    # 根地址（host 后面没有路径）同样不换
    assert candidates("", "task_1_beijing.md", [])[:2] == [
        brain.TASK_API_DEFAULT,
        f"{brain.TASK_API_DEFAULT}/beijing",
    ]


def test_executor_candidates_fill_an_empty_query_sample():
    """样例地址的查询值是空的时候也要把查询词填进去（S1）

    文档写成 `GET http://localhost:8899/heritage?city=<城市名>` 时，`endpoints`
    的地址正则在 `<` 处截断，抓到的样例就是 `...?city=`；文档本来就写成空值
    （`...?city=`）时也是同一形状。旧写法要求 `=` 后面"至少有一个字符"
    （`[^&/]+`），空值样例一个带查询词的候选都生不出来——请求照原样发出去，
    问的是空查询词，接口只会回一行取数失败的诊断，任务分照旧是 0。
    """
    _, candidates = _executor_queries()

    out = candidates(
        "请查询该城市的文化遗产", "task_1_beijing.md",
        ["http://localhost:8899/heritage?city="],
    )
    # 样例本身照旧排在最前（先照文档原样试一次），带查询词的候选紧随其后
    assert out[0] == "http://localhost:8899/heritage?city="
    assert "http://localhost:8899/heritage?city=beijing" in out

    # 样例里已经写了查询值时行为不变：换成任务自己的那一个
    out = candidates(
        "请查询该城市的文化遗产", "task_1_beijing.md",
        ["http://localhost:8899/heritage?city=alpha"],
    )
    assert "http://localhost:8899/heritage?city=beijing" in out
    assert "http://localhost:8899/heritage?city=alpha" in out


def test_executor_api_fail_reports_status_and_body(capsys):
    """取数失败的诊断行带上 HTTP 状态码与响应体（#77 的 S1）

    复盘里只有 `[APIFAIL] http://localhost:8899 HTTPError`，分不清 401（缺鉴权）
    还是 404（地址不对），下一轮修复只能靠猜；401/404 的响应体里通常就写着
    缺什么。诊断还必须和状态码挤在同一行：`[APIFAIL]` 的行数就是"取数失败了
    几次"（`task_brief` 的 fail 计数），一次失败拆成两行会让计数翻倍。
    """
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]
    match = re.search(r"def fetch\(url, headers\):.*?(?=\ndef )", script, re.S)
    assert match

    class HTTPError(Exception):
        code = 401

        def read(self):
            return b'{"status":"error","message":"missing Authorization"}'

    def _urlopen(request, timeout=None):
        raise HTTPError()

    class _Request:
        def __init__(self, url, headers=None):
            self.url = url

    class _RequestModule:
        Request = _Request
        urlopen = staticmethod(_urlopen)

    class _Urllib:
        request = _RequestModule

    namespace = {
        "FAIL": brain.TASK_API_FAIL_MARKER,
        "TIMEOUT": brain.TASK_API_TIMEOUT,
        "BODY_LIMIT": brain.TASK_API_BODY_LIMIT,
        "urllib": _Urllib,
    }
    exec(match.group(0), namespace)  # noqa: S102
    assert namespace["fetch"]("http://localhost:8899/weather", {"Accept": "*/*"}) == ""

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(brain.TASK_API_FAIL_MARKER)
    assert "HTTPError 401" in lines[0]  # 状态码要看得出来是 401 还是 404
    assert "missing Authorization" in lines[0]  # 响应体里的线索一并带回


def test_executor_api_fail_without_status_keeps_one_line(capsys):
    """连不上（无状态码）时诊断行仍然只有一行，且不带多余空格"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]
    match = re.search(r"def fetch\(url, headers\):.*?(?=\ndef )", script, re.S)
    assert match

    class URLError(Exception):
        pass  # 没有 code 也没有 read（连不上时就是这样）

    def _urlopen(request, timeout=None):
        raise URLError()

    class _Request:
        def __init__(self, url, headers=None):
            self.url = url

    class _RequestModule:
        Request = _Request
        urlopen = staticmethod(_urlopen)

    class _Urllib:
        request = _RequestModule

    namespace = {
        "FAIL": brain.TASK_API_FAIL_MARKER,
        "TIMEOUT": brain.TASK_API_TIMEOUT,
        "BODY_LIMIT": brain.TASK_API_BODY_LIMIT,
        "urllib": _Urllib,
    }
    exec(match.group(0), namespace)  # noqa: S102

    assert namespace["fetch"]("http://localhost:8899/x", {"Accept": "*/*"}) == ""
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{brain.TASK_API_FAIL_MARKER} http://localhost:8899/x URLError"]


def _executor_auth():
    """从生成的沙盒脚本里取出鉴权部分（`api_key` + `request_headers`）"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]  # 去掉挑解释器那半句
    key_fn = re.search(r"def api_key\(text\):.*?(?=\ndef request_headers\()", script, re.S)
    head_fn = re.search(r"def request_headers\(key\):.*?(?=\ndef find_files\()", script, re.S)
    assert key_fn and head_fn
    namespace: dict = {
        "re": re,
        "KEY_PATTERNS": brain.TASK_API_KEY_PATTERNS,
        "KEY_MIN": brain.TASK_API_KEY_MIN_LEN,
        "KEY_PLACEHOLDERS": brain.TASK_API_KEY_PLACEHOLDERS,
        "AUTH_HEADER": brain.TASK_API_AUTH_HEADER,
        "KEY_HEADER": brain.TASK_API_KEY_HEADER,
        "BEARER": brain.TASK_API_BEARER,
    }
    exec(key_fn.group(0) + head_fn.group(0), namespace)  # noqa: S102
    return namespace["api_key"], namespace["request_headers"]


def test_executor_api_key_follows_document_headers():
    """接口文档里写明的 Key 要抠出来随请求发（S3，PK590918/PK590917 的 R16 401）

    复盘里我方请求只带 `Accept`，接口回
    `[APIFAIL] ... HTTPError 401 => missing 'Authorization' header`，同一回合
    对手已经带着 Bearer 取到数。文档给出 Key 的写法各家不同，几种常见写法都要认。
    """
    key, _ = _executor_auth()

    assert key("请求头：Authorization: Bearer sk-abc123456") == "sk-abc123456"
    assert key("X-API-Key: 8f3c1d9e2b") == "8f3c1d9e2b"
    assert key("api_key=token-abcdef") == "token-abcdef"
    assert key("| API Key | abc123456 | 用于鉴权 |") == "abc123456"


def test_executor_api_key_skips_document_prose():
    """文档里的占位写法与行文不能当成 Key（发一个假 Key 只会再撞一次 401）"""
    key, _ = _executor_auth()

    assert key("Authorization: Bearer YOUR_API_KEY") == ""
    assert key("| API Key | required |") == ""
    assert key("Authorization：<你的 API Key>") == ""
    # `Bearer` 后面直接换行、下一行是地址：那不是 Key
    assert key("Authorization: Bearer\nhttp://localhost:8899/x") == ""
    assert key("请阅读 task_1_beijing.md，获取任务信息") == ""


def test_executor_request_headers_add_bearer_and_key():
    """拿到 Key 时两个头一起带（多带一个不影响无鉴权接口，少带一个必然 401）"""
    _, headers = _executor_auth()

    assert headers("") == {"Accept": "*/*"}
    got = headers("sk-abc123456")
    assert got[brain.TASK_API_AUTH_HEADER] == f"{brain.TASK_API_BEARER}sk-abc123456"
    assert got[brain.TASK_API_KEY_HEADER] == "sk-abc123456"


def test_executor_fetch_sends_the_headers_it_gets():
    """`fetch` 必须把请求头带进 `urllib.request.Request`（S3 的最后一环）"""
    command = _task_executor("task_1_beijing.md")
    script = command.split("\n", 1)[1]
    match = re.search(r"def fetch\(url, headers\):.*?(?=\ndef )", script, re.S)
    assert match

    seen: dict = {}

    class _Response:
        def read(self):
            return b'{"city":"beijing"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Request:
        def __init__(self, url, headers=None):
            seen["headers"] = headers

    def _urlopen(request, timeout=None):
        return _Response()

    class _RequestModule:
        Request = _Request
        urlopen = staticmethod(_urlopen)

    class _Urllib:
        request = _RequestModule

    namespace = {
        "FAIL": brain.TASK_API_FAIL_MARKER,
        "TIMEOUT": brain.TASK_API_TIMEOUT,
        "BODY_LIMIT": brain.TASK_API_BODY_LIMIT,
        "urllib": _Urllib,
    }
    exec(match.group(0), namespace)  # noqa: S102

    headers = {"Accept": "*/*", "Authorization": "Bearer sk-abc123456"}
    body = namespace["fetch"]("http://localhost:8899/weather", headers)
    assert body == '{"city":"beijing"}'
    assert seen["headers"] == headers


def _executor_fetch_loop():
    """从生成的沙盒脚本里取出取数主循环（含“没有任务文件”时的那条兜底）"""
    command = _task_executor("task_1_alpha.md")
    script = command.split("\n", 1)[1]  # 去掉挑解释器那半句
    match = re.search(
        r"\ndeadline = time\.time\(\) \+ TIME_BUDGET\n.*?(?=\nPYEOF)", script, re.S,
    )
    assert match
    return match.group(0)


def test_executor_calls_the_api_without_a_task_file(capsys):
    """一份任务文件都没找到时，执行器照样把接口地址试一遍（S1，PK591783 的 R11–R12）

    复盘里沙盒读到接口文档（`[SCAN] docs=2 urls=2 key=yes`）、`[exitCode:0]`
    全都正常，可 `api=0` 且一行 `[APIFAIL]` 都没有——取数循环挂在"任务文件
    列表"上，任务描述里的文件名与沙盒里的对不上时（`files` 为空）整段跳过，
    这一回合就成了"执行器跑过了、却连一次请求都没发"。接口文档与本地接口
    本来就在手里，没有任务文件也得试一遍：`[API]` / `[APIFAIL]` 是"读题之后
    真的去调了 API"的唯一凭据，看门狗也靠它判断该走哪条止损线。
    """
    _, candidates = _executor_queries()
    rotate = _executor_rotate()
    asked: list[str] = []

    class _Time:
        """冻结时间：deadline 判定不参与这条测试"""

        @staticmethod
        def time():
            return 0.0

    def _fetch(url, headers):
        asked.append(url)
        return '{"city":"alpha"}'

    namespace = {
        "os": os,
        "re": re,
        "time": _Time,
        "files": [],  # 沙盒里一份任务文件都没找到
        "read": lambda path: "",
        "rotate": rotate,
        "candidates": candidates,
        "fetch": _fetch,
        "SOLVE_MAX": brain.TASK_SOLVE_MAX,
        "KEEP": brain.TASK_API_KEEP,
        "OFFSET": 5,
        "MAX_CALLS": brain.TASK_API_MAX_CALLS,
        "TIME_BUDGET": brain.TASK_API_TIME_BUDGET,
        "DATA": TASK_DATA_MARKER,
        "SOLUTION": TASK_SOLUTION_MARKER,
        "SOLUTION_END": TASK_SOLUTION_END,
        "TASK_PATH": "task_1_alpha.md",
        "doc_text": "接口文档：GET http://localhost:8899/api/city",
        "urls": ["http://localhost:8899/api/city"],
    }
    exec(_executor_fetch_loop(), namespace)  # noqa: S102

    # 请求真的发出去了（复盘里 api=0 且没有 [APIFAIL]，等于一个回合白跑）
    assert asked
    assert asked[0].startswith(brain.TASK_API_DEFAULT)
    # 取到的数据照样打进答案段，段名用任务描述里点名的那份文件：
    # `_solution_answer` 按文件名认领本任务的答案，占位名会让这一份取数作废
    out = capsys.readouterr().out
    assert TASK_DATA_MARKER in out
    assert f"{TASK_SOLUTION_MARKER}task_1_alpha.md" in out
    assert TASK_SOLUTION_END in out


def test_executor_without_task_file_keeps_a_usable_solution_name(capsys):
    """任务描述里没点名文件时，答案段挂一个占位名而不是任务根目录的目录名

    `_sandbox_command` 在探测次数用满后会把任务根目录（`/tmp/selfEvolutionTask`）
    交给执行器；那份目录名不是文档，`_solution_answer` 会按"描述里没给文件名"
    取第一段答案，所以段名是什么不影响取答案——但不能把目录名当成任务文件名
    带出去（它会进答案缓存，见 `_remember_task_answers`）。
    """
    _, candidates = _executor_queries()
    rotate = _executor_rotate()

    class _Time:
        @staticmethod
        def time():
            return 0.0

    def _fetch(url, headers):
        return '{"city":"alpha"}'

    namespace = {
        "os": os,
        "re": re,
        "time": _Time,
        "files": [],
        "read": lambda path: "",
        "rotate": rotate,
        "candidates": candidates,
        "fetch": _fetch,
        "SOLVE_MAX": brain.TASK_SOLVE_MAX,
        "KEEP": brain.TASK_API_KEEP,
        "OFFSET": 0,
        "MAX_CALLS": brain.TASK_API_MAX_CALLS,
        "TIME_BUDGET": brain.TASK_API_TIME_BUDGET,
        "DATA": TASK_DATA_MARKER,
        "SOLUTION": TASK_SOLUTION_MARKER,
        "SOLUTION_END": TASK_SOLUTION_END,
        "TASK_PATH": brain.TASK_ROOTS[0],
        "doc_text": "",
        "urls": [],
    }
    exec(_executor_fetch_loop(), namespace)  # noqa: S102

    out = capsys.readouterr().out
    assert f"{TASK_SOLUTION_MARKER}task\n" in out
    assert brain.TASK_ROOTS[0] not in out.split(TASK_SOLUTION_MARKER)[1].split("\n")[0]


def test_executor_without_task_file_still_asks_with_the_task_query_word():
    """没有任务文件时，查询词仍然取自任务描述里点名的那份文件（S1）

    兜底那一次取数用的是接口文档里的样例地址，而样例常常是个模板
    （`.../weather?city=`，见 `candidates`）：真正的查询词要靠 `queries` 填进去，
    最可靠的来源就是任务文件名（`task_1_beijing.md` -> `beijing`）。沙盒里
    没有这份文件时，这个名字是唯一还握在手里的查询词来源——漏传就只能拿
    文档正文里随手挑的英文词（`GET`、`weather` 这类）去填样例，地址拼不对，
    任务照旧 0 分（复盘里"读题成功、api 恒 0"就是这么来的）。
    """
    _, candidates = _executor_queries()
    rotate = _executor_rotate()
    asked: list[str] = []

    class _Time:
        """冻结时间：deadline 判定不参与这条测试"""

        @staticmethod
        def time():
            return 0.0

    def _fetch(url, headers):
        asked.append(url)
        return ""

    namespace = {
        "os": os,
        "re": re,
        "time": _Time,
        "files": [],  # 沙盒里一份任务文件都没找到
        "read": lambda path: "",
        "rotate": rotate,
        "candidates": candidates,
        "fetch": _fetch,
        "SOLVE_MAX": brain.TASK_SOLVE_MAX,
        "KEEP": brain.TASK_API_KEEP,
        "OFFSET": 0,
        "MAX_CALLS": brain.TASK_API_MAX_CALLS,
        "TIME_BUDGET": brain.TASK_API_TIME_BUDGET,
        "DATA": TASK_DATA_MARKER,
        "SOLUTION": TASK_SOLUTION_MARKER,
        "SOLUTION_END": TASK_SOLUTION_END,
        "TASK_PATH": "task_1_beijing.md",
        "doc_text": "接口文档：GET http://localhost:8899/weather?city=",
        "urls": ["http://localhost:8899/weather?city="],
    }
    exec(_executor_fetch_loop(), namespace)  # noqa: S102

    # 问的是任务自己的查询词，而不是把样例里那个空查询词原样发出去
    assert "http://localhost:8899/weather?city=beijing" in asked

    # 描述里没点名文件（`TASK_PATH` 是任务根目录）时照旧不拿目录名当查询词：
    # 候选清单与文档正文里挑出来的词一致，不会多出一条以目录名结尾的地址
    asked.clear()
    namespace["TASK_PATH"] = brain.TASK_ROOTS[0]
    exec(_executor_fetch_loop(), namespace)  # noqa: S102

    assert all(not url.endswith("selfEvolutionTask") for url in asked)
