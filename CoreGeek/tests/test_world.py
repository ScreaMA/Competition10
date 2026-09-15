"""世界模型用例：可建造区在线学习、敌方来向推导。

对应设计文档V2 第 5 章。

接口报文里**没有**任何字段说明哪些格子能建造，所以 V2 只能在线学：
拿真实 `build` 指令的执行结果当标签，记成"相对基地 footprint 的偏移"。
这一组用例覆盖学习的三个关键点——写入、消歧、换边复用。
"""

from __future__ import annotations

import pytest

from agent import world as world_mod
from agent.protocol import Pos, Turn
from agent.world import World


def _turn(payload_factory, role_factory, base=(20, 10), **kwargs):
    roles = [role_factory(10013, "station", *base)]
    roles.extend(kwargs.pop("roles", []))
    return Turn.load(payload_factory(roles=roles, **kwargs))


# ==========================================================================
# 可建造区
# ==========================================================================


def test_build_success_is_learned(payload_factory, role_factory):
    """建成成功 ⇒ 该偏移被记为可建造"""
    turn = _turn(payload_factory, role_factory)
    # 基地 footprint 原点 = (20, 9)，所以 (19, 8) 的偏移是 (-1, -1)
    world_mod.record_build(turn, 10013, Pos(19, 8), "rocket")
    # 下一回合：该 build 成功了
    next_turn = _turn(payload_factory, role_factory, round_no=2,
                      last_action_results={10013: True})
    world_mod.absorb_results(next_turn)

    zone = world_mod.zone()
    assert zone.known("rocket", Pos(-1, -1)) is True
    assert zone.allows("rocket", Pos(-1, -1)) is True


def test_clean_failure_is_learned_as_forbidden(payload_factory, role_factory):
    """格子空闲 + 角色相邻 + 钱够，却仍然失败 ⇒ 这个偏移不可建

    三个前提缺一不可：失败也可能是"目标格被抢"或"金币不够"造成的，
    那跟格子能不能建没有关系。
    """
    worker = role_factory(10010, "worker", 18, 8)  # 与目标 (19,8) 相邻
    turn = _turn(payload_factory, role_factory, gold=100, roles=[worker])
    world_mod.record_build(turn, 10010, Pos(19, 8), "rocket")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100,
                      roles=[worker], last_action_results={10010: False})
    world_mod.absorb_results(next_turn)

    zone = world_mod.zone()
    assert zone.known("rocket", Pos(-1, -1)) is False
    assert zone.forbidden("rocket", Pos(-1, -1)) is True


def test_unclean_failure_is_discarded(payload_factory, role_factory):
    """失败原因可能不是"格子不允许建"——这时不写入否定标签

    误标一个可建造格为不可建，代价是那座塔永远不会再被考虑，
    比少学一条记录严重得多（设计文档V2 §5.1）。这里让目标格在计划时就被
    另一个单位占着（`cell_free=False`），失败显然不能归因于格子本身。
    """
    worker = role_factory(10010, "worker", 18, 8)
    other = role_factory(10012, "worker", 19, 8)  # 目标格被占
    turn = _turn(payload_factory, role_factory, gold=100, roles=[worker, other])
    world_mod.record_build(turn, 10010, Pos(19, 8), "rocket")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100,
                      roles=[worker, other], last_action_results={10010: False})
    world_mod.absorb_results(next_turn)

    assert world_mod.zone().known("rocket", Pos(-1, -1)) is None


def test_no_result_in_payload_is_not_learned(payload_factory, role_factory):
    """报文没提这个角色 ⇒ 不学习（避免把"没提"误读成"失败"）"""
    turn = _turn(payload_factory, role_factory, gold=100,
                 roles=[role_factory(10010, "worker", 19, 8)])
    world_mod.record_build(turn, 10010, Pos(19, 8), "rocket")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100)
    world_mod.absorb_results(next_turn)
    assert world_mod.zone().known("rocket", Pos(-1, -1)) is None


def test_wall_and_weapon_zones_are_separate(payload_factory, role_factory):
    """武器区与围墙区是两张独立的表（任务书 §4.1 的蓝区/黄区）"""
    turn = _turn(payload_factory, role_factory, gold=100,
                 roles=[role_factory(10010, "worker", 19, 8)])
    world_mod.record_build(turn, 10010, Pos(19, 8), "wall")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100,
                      roles=[role_factory(10010, "worker", 19, 8)],
                      last_action_results={10010: True})
    world_mod.absorb_results(next_turn)

    zone = world_mod.zone()
    assert zone.known("wall", Pos(-1, -1)) is True
    assert zone.known("rocket", Pos(-1, -1)) is None


def test_offset_survives_base_move(payload_factory, role_factory):
    """偏移是相对基地的 ⇒ 换边（上下半场互换位置）后学习成果依然有效"""
    turn = _turn(payload_factory, role_factory, base=(20, 10), gold=100,
                 roles=[role_factory(10010, "worker", 19, 8)])
    world_mod.record_build(turn, 10010, Pos(19, 8), "rocket")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100,
                      roles=[role_factory(10010, "worker", 19, 8)],
                      last_action_results={10010: True})
    world_mod.absorb_results(next_turn)

    # 换到地图另一角：基地 (5, 25)，原点 (5, 24)，偏移 (-1,-1) 对应 (4, 23)
    moved = _turn(payload_factory, role_factory, base=(5, 25))
    world = World.load(moved)
    assert world.absolute_of(Pos(-1, -1)) == Pos(4, 23)
    assert world.zone().allows("rocket", Pos(-1, -1)) is True


def test_record_build_only_once_per_unit(payload_factory, role_factory):
    """同一角色同回合只登记一次（第二个建造尝试会覆盖第一个的观测）"""
    turn = _turn(payload_factory, role_factory, gold=100,
                 roles=[role_factory(10010, "worker", 19, 8)])
    world_mod.record_build(turn, 10010, Pos(19, 8), "rocket")
    world_mod.record_build(turn, 10010, Pos(21, 8), "rocket")
    next_turn = _turn(payload_factory, role_factory, round_no=2, gold=100,
                      roles=[role_factory(10010, "worker", 19, 8)],
                      last_action_results={10010: True})
    world_mod.absorb_results(next_turn)
    # 只有第一次登记的那个生效
    assert world_mod.zone().known("rocket", Pos(-1, -1)) is True
    assert world_mod.zone().known("rocket", Pos(1, -1)) is None


# ==========================================================================
# 敌方来向
# ==========================================================================


def test_enemy_sides_from_visible_station(payload_factory, role_factory):
    """敌方基地全局可见（接口文档 §1.4），用它推来向"""
    turn = _turn(
        payload_factory, role_factory, base=(20, 10),
        enemies=[role_factory(20013, "station", 35, 25)],
    )
    sides = World.load(turn).enemy_sides()
    assert sides == frozenset({"right", "up"})


def test_enemy_sides_falls_back_to_nearest_unit(payload_factory, role_factory):
    turn = _turn(
        payload_factory, role_factory, base=(20, 10),
        enemies=[role_factory(20010, "worker", 5, 10)],
    )
    assert World.load(turn).enemy_sides() == frozenset({"left"})


def test_enemy_sides_empty_when_nothing_visible(payload_factory, role_factory):
    """看不到任何敌方单位时返回空集——不做方位假设（设计文档V2 §11.3）"""
    turn = _turn(payload_factory, role_factory)
    assert World.load(turn).enemy_sides() == frozenset()


# ==========================================================================
# 坐标换算与查询
# ==========================================================================


def test_origin_and_offsets(payload_factory, role_factory):
    turn = _turn(payload_factory, role_factory, base=(20, 10))
    world = World.load(turn)
    assert world.origin == Pos(20, 9)
    assert world.offset_of(Pos(20, 9)) == Pos(0, 0)
    assert world.absolute_of(Pos(2, 3)) == Pos(22, 12)


def test_my_task_points_only_own_team(payload_factory, role_factory, zone_factory):
    """只认己方阵营的任务点（任务书 §4.6.2：在敌方任务点执行任务无效）"""
    turn = _turn(
        payload_factory, role_factory, team="challenger",
        zones=[zone_factory("challengerTaskPoint1", 14, 14),
               zone_factory("defenderTaskPoint1", 23, 14)],
    )
    assert turn.my_task_points() == (Pos(14, 14),)


def test_buildable_now_excludes_occupied(payload_factory, role_factory):
    turn = _turn(payload_factory, role_factory,
                 roles=[role_factory(10010, "worker", 25, 25)])
    world = World.load(turn)
    assert world.buildable_now(Pos(30, 30)) is True
    assert world.buildable_now(Pos(25, 25)) is False   # 工人占着
    assert world.buildable_now(Pos(20, 10)) is False   # 基地占着


# ==========================================================================
# 布防方位：**每回合从报文里推，且分主次**（不能写死，也不能不分主次）
# ==========================================================================
#
# 两个出生基地在地图对角（左上 vs 右下），所以敌基地**几乎总是斜的**——横向
# 与纵向都命中。只把它们当成"都算来敌方向"，谁排前面就由环数、坐标这些与敌情
# 无关的因素决定了。实测 10 个敌基地方位里有 4 个把塔摆到了次要那一边。


@pytest.mark.parametrize(
    ("enemy", "expect"),
    [
        # 敌基地在左上方：横向偏移更大 ⇒ 左是主轴
        ((8, 22), ("left", "up", "down", "right")),
        # 在正上方偏左：纵向偏移更大 ⇒ 上是主轴
        ((20, 24), ("up", "left", "down", "right")),
        # 在右下方：右是主轴
        ((36, 14), ("right", "up", "down", "left")),
        # 正下方：只有下
        ((30, 2), ("down", "up", "left", "right")),
        # 正左方：只有左
        ((5, 10), ("left", "up", "down", "right")),
    ],
)
def test_defence_order_puts_the_dominant_axis_first(
    payload_factory, role_factory, enemy, expect
):
    """主轴（偏移更大的那一维）必须排在布防顺序最前面"""
    base = (30, 10)   # station 左上角；原点 (30,9)
    turn = _turn(
        payload_factory, role_factory, base=base,
        enemies=[role_factory(9001, "station", *enemy)],
    )
    assert World.load(turn).defence_order() == expect


def test_defence_order_mirrors_when_the_team_switches_side(
    payload_factory, role_factory
):
    """上下半场换边：同一份代码必须自动跟着镜像，不能出现写死的方位

    上半场我们在右下、敌人在左上；下半场反过来。塔位也跟着翻到另一侧——
    "朝哪边布防"完全由报文推出来。
    """
    lower_right = _turn(
        payload_factory, role_factory, base=(30, 10),
        enemies=[role_factory(9001, "station", 8, 22)],
    )
    upper_left = _turn(
        payload_factory, role_factory, base=(10, 22),
        enemies=[role_factory(9001, "station", 32, 10)],
    )
    first = World.load(lower_right).defence_order()[0]
    second = World.load(upper_left).defence_order()[0]
    assert first == "left" and second == "right", (first, second)
