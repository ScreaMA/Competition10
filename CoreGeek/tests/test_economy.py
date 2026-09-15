"""经济策略用例：卖矿、建造队列、金币分配。

对应设计文档V2 §7.1 – §7.3。

重点是**金币冻结**（V1 的 E1：开局 75 金砸完 3 座塔后连续 7–10 回合
`gold=0`，背包里的矿石攒着不卖，PK592172 的 R8–R16）与**开局建塔延迟**
（E5：R1–R3 纯 move，R4 才落第一座塔，敌方 R1 已连建 2 座）。
"""

from __future__ import annotations

from agent import world as world_mod
from agent.protocol import Pos, Turn, WEAPON_BUILD_COST
from agent.strategy import economy
from agent.world import World


def _world(payload_factory, role_factory, base=(20, 10), gold=75, **kwargs):
    roles = [role_factory(10013, "station", *base)]
    roles.extend(kwargs.pop("roles", []))
    return World.load(Turn.load(payload_factory(roles=roles, gold=gold, **kwargs)))


def _worker(role_factory, unit_id=10010, x=20, y=13, backpack=(), **kw):
    return role_factory(
        unit_id, "worker", x, y, backpack=list(backpack), **kw
    )


# ==========================================================================
# 开局：第 1 回合就要建塔（E5）
# ==========================================================================


def test_first_round_builds_tower(payload_factory, role_factory):
    """R1、金币 75 ⇒ 指令里必须含 build（敌方 R1 就连建 2 座塔）"""
    world = _world(payload_factory, role_factory, gold=75,
                   roles=[_worker(role_factory, 10010, 18, 10)])
    commands = economy.plan_day(world, set())
    actions = [command["action"] for command in commands.values()]
    assert "build" in actions or "move" in actions

    # 工人就在塔位旁边，应该直接发 build
    sites = [Pos(18, 8), Pos(22, 8), Pos(18, 12), Pos(22, 12)]
    commands = economy.plan_day(
        _world(payload_factory, role_factory, gold=75,
               roles=[_worker(role_factory, 10010, 18, 9)]),
        set(),
    )
    command = commands[10010]
    assert command["action"] in ("build", "move")
    if command["action"] == "build":
        assert command["name"] in ("rocket", "railgun", "gatling")


def test_no_tower_without_gold(payload_factory, role_factory):
    """金币不够 25 时不该尝试建塔（那会白耗一个回合）"""
    world = _world(payload_factory, role_factory, gold=10,
                   zones=[],
                   roles=[_worker(role_factory)])
    commands = economy.plan_day(world, set())
    assert all(
        command.get("name") != "rocket" for command in commands.values()
    )


# ==========================================================================
# 卖矿（E1：金币冻结）
# ==========================================================================


def test_sell_when_gold_is_zero(payload_factory, role_factory, zone_factory):
    """金币为 0 且背包有矿石 ⇒ 无条件去卖（"金币恒 0"的直接对策）"""
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, backpack=("stone",))],
    )
    worker = world.turn.workers()[0]
    assert economy.should_sell(world, worker) is True


def test_sell_when_gold_below_tower_price(payload_factory, role_factory, zone_factory):
    world = _world(
        payload_factory, role_factory, gold=WEAPON_BUILD_COST - 1,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, backpack=("copper",))],
    )
    worker = world.turn.workers()[0]
    assert economy.should_sell(world, worker) is True


def test_sell_command_when_adjacent_to_vendor(payload_factory, role_factory, zone_factory):
    """站在小贩旁边 ⇒ 直接发 sell（挑最贵的矿石先卖）"""
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, 10010, 20, 15, backpack=("stone", "copper"))],
    )
    commands = economy.plan_day(world, set())
    command = commands[10010]
    assert command["action"] == "sell"
    assert command["name"] == "copper"  # 铜价 5 > 石价 1


def test_sell_keeps_stone_reserve_for_walls(payload_factory, role_factory, zone_factory):
    """还有墙要砌时，石头要留够（别把建材卖了换钱）"""
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, 10010, 20, 15, backpack=("stone", "stone", "stone"))],
    )
    commands = economy.plan_day(world, set())
    command = commands[10010]
    # 卖铜/铁可以，卖石头最多卖到留够 STONE_BATCH 为止
    if command["action"] == "sell" and command["name"] == "stone":
        assert command["num"] <= 3 - economy.STONE_BATCH or command["num"] == 1


def test_no_sell_when_nothing_to_sell(payload_factory, role_factory, zone_factory):
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, backpack=())],
    )
    assert economy.should_sell(world, world.turn.workers()[0]) is False


def test_no_sell_when_gold_is_plenty(payload_factory, role_factory, zone_factory):
    """金币充裕、背包不挤、墙也砌完了，就没必要专门跑一趟小贩"""
    world = _world(
        payload_factory, role_factory, gold=200,
        zones=[zone_factory("vendor", 20, 16)],
        roles=[_worker(role_factory, backpack=("stone",))],
    )
    # 还有墙要砌 ⇒ 石头留着
    assert economy.should_sell(world, world.turn.workers()[0]) is False


# ==========================================================================
# 采集（E2：采集中断）
# ==========================================================================


def test_miner_collects_when_adjacent(payload_factory, role_factory, zone_factory):
    """采集工就在矿旁边 ⇒ 发 collect（采集不该被"去建塔"打断）"""
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("copper", 24, 13)],
        roles=[_worker(role_factory, 10010, 18, 12, backpack=("copper",)),
               _worker(role_factory, 10012, 24, 12, backpack=())],
    )
    commands = economy.plan_day(world, set())
    assert commands[10012]["action"] == "collect"
    assert commands[10012]["targetPos"][0] == {"x": 24, "y": 13}


def test_miner_walks_to_mine_when_far(payload_factory, role_factory, zone_factory):
    """矿区是不可通行的中立单位，只能走到它**旁边**（不是走到它上面）

    这是 V1 "采集 R3 一次后断档 9–10 回合"（E2）的一类根因：把矿区坐标当
    终点，A* 永远找不到路，工人于是整回合没有指令。
    """
    world = _world(
        payload_factory, role_factory, gold=200,
        zones=[zone_factory("copper", 30, 13)],
        roles=[_worker(role_factory, 10010, 18, 12),
               _worker(role_factory, 10012, 10, 10, backpack=())],
    )
    commands = economy.plan_day(world, set())
    command = commands[10012]
    assert command["action"] == "move"
    step = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
    # 这一步必须在靠近矿的方向上，且不能踩到矿本身
    assert step != Pos(30, 13)


def test_builder_mines_stone_when_walls_pending(payload_factory, role_factory, zone_factory):
    """建造工没石头可砌墙时自己去采（保证"围墙"和"采集"联动）"""
    world = _world(
        payload_factory, role_factory, gold=0,
        zones=[zone_factory("stone", 18, 13)],
        roles=[_worker(role_factory, 10010, 18, 12, backpack=())],
    )
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "collect"


# ==========================================================================
# 建造执行
# ==========================================================================


def test_build_records_attempt_for_learning(payload_factory, role_factory):
    """发 build 指令时必须登记学习记录（下一回合才能学习可建造区）"""
    from agent.strategy import defense

    probe = _world(payload_factory, role_factory, gold=75,
                   roles=[_worker(role_factory, 10010, 30, 30)])
    site = defense.tower_sites(probe)[0]
    # 站到第一个塔位旁边
    world = _world(payload_factory, role_factory, gold=75,
                   roles=[_worker(role_factory, 10010, site.x - 1, site.y)])
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "build"
    # 登记过了：下一回合用它的执行结果学习
    next_turn = _turn_with_result(payload_factory, role_factory, 10010, True)  # noqa: E501
    notes = world_mod.absorb_results(next_turn)
    assert notes  # 至少学到一条


def _turn_with_result(payload_factory, role_factory, unit_id, ok):
    return Turn.load(payload_factory(
        round_no=2, gold=75,
        roles=[role_factory(10013, "station", 20, 10),
               _worker(role_factory, unit_id, 18, 9)],
        last_action_results={unit_id: ok},
    ))


def test_build_walks_first_then_builds(payload_factory, role_factory):
    """距离塔位 > 1 时先走过去（建造要求距离一格内，任务书 §4.4）"""
    world = _world(payload_factory, role_factory, gold=75,
                   roles=[_worker(role_factory, 10010, 35, 30)])
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "move"


# ==========================================================================
# 金币分配：升级券
# ==========================================================================


def test_station_voucher_uses_whole_footprint(payload_factory, role_factory):
    """基地是 2×2：站在它右侧也要算"周围一格内"

    只按左上角那一格算距离的话，站在 (22,9) 的角色离 `station.pos=(20,10)` 是
    2 格，会被判成"还没到位"，然后对着一个自己已经站着的落脚点反复寻路
    （返回 None）——基地升级券就永远用不出去。
    """
    player = payload_factory(
        gold=0,
        roles=[
            role_factory(10013, "station", 20, 10, health=800),  # 53% < 60%
            _worker(role_factory, 10010, 22, 9,
                    backpack=("StationUpgradeVoucher1",)),
        ],
    )
    world = World.load(Turn.load(player))
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "use"
    assert commands[10010]["name"] == "StationUpgradeVoucher1"
    assert commands[10010]["targetPos"][0] == {"x": 20, "y": 10}


def test_uses_held_voucher_on_tower(payload_factory, role_factory):
    """背包里有武器升级券且有一座 level1 的塔 ⇒ 走过去用掉

    V1 全程没买过也没用过升级券，"三塔零升级"（E4）就是这么来的。
    """
    tower = role_factory(10020, "gatling", 20, 13, level=1)
    world = _world(
        payload_factory, role_factory, gold=0,
        roles=[_worker(role_factory, 10010, 20, 14,
                       backpack=("WeaponUpgradeVoucher1",)),
               tower],
    )
    commands = economy.plan_day(world, set())
    command = commands[10010]
    assert command["action"] == "use"
    assert command["name"] == "WeaponUpgradeVoucher1"
    assert command["targetPos"][0] == {"x": 20, "y": 13}


def test_buys_voucher_only_with_surplus_gold(payload_factory, role_factory):
    """金币 100 且三塔齐全 ⇒ 去武器商店买升级券"""
    world = _world(
        payload_factory, role_factory, gold=120,
        zones=[{"neutralType": "weaponShop", "pos": {"x": 20, "y": 16}}],
        roles=[
            _worker(role_factory, 10010, 20, 15, backpack=()),
            role_factory(10020, "gatling", 18, 8),
            role_factory(10030, "railgun", 22, 8),
            role_factory(10040, "rocket", 20, 12),
        ],
        weapon_shop=[{"name": "WeaponUpgradeVoucher1", "price": 100}],
    )
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "buy"
    assert commands[10010]["name"] == "WeaponUpgradeVoucher1"


def test_medicine_when_hurt(payload_factory, role_factory):
    """角色残血时先买药（10 金换一条命）"""
    world = _world(
        payload_factory, role_factory, gold=200,
        zones=[{"neutralType": "weaponShop", "pos": {"x": 20, "y": 16}}],
        roles=[_worker(role_factory, 10010, 20, 15, backpack=(), health=40)],
        weapon_shop=[{"name": "Medicine", "price": 10}],
    )
    commands = economy.plan_day(world, set())
    assert commands[10010]["action"] == "buy"
    assert commands[10010]["name"] == "Medicine"


# ==========================================================================
# 分工
# ==========================================================================


def test_roles_are_fixed_by_id(payload_factory, role_factory):
    """ID 小的当建造工、大的当采集工（分工固定，不来回切）"""
    world = _world(payload_factory, role_factory, gold=75,
                   roles=[_worker(role_factory, 10010), _worker(role_factory, 10012)])
    roles = economy.assign_roles(world.turn)
    assert roles[10010] == economy.ROLE_BUILDER
    assert roles[10012] == economy.ROLE_MINER


def test_both_workers_always_get_commands(payload_factory, role_factory, zone_factory):
    """两个工人都不该空转（V1 的 `idle_man` 峰值到 3）"""
    world = _world(
        payload_factory, role_factory, gold=75,
        zones=[zone_factory("stone", 18, 13)],
        roles=[_worker(role_factory, 10010, 18, 12), _worker(role_factory, 10012, 22, 12)],
    )
    commands = economy.plan_day(world, set())
    assert set(commands) == {10010, 10012}
    for command in commands.values():
        assert command["action"] in ("move", "build", "collect", "sell", "buy", "use")
