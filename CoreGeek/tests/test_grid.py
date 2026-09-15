"""几何与寻路用例。

对应设计文档V2 §5.3。
"""

from __future__ import annotations

import pytest

from agent import grid
from agent.protocol import Pos, Turn, Unit


def _turn(payload_factory, role_factory, zones=None, robots=None):
    worker = role_factory(10010, "worker", 5, 5)
    return Turn.load(payload_factory(
        roles=[worker, role_factory(10013, "station", 20, 10)],
        zones=zones or [],
        robots=robots or [],
    )), Unit.load(worker)


def test_step_toward_straight_line(payload_factory, role_factory):
    turn, worker = _turn(payload_factory, role_factory)
    step = grid.next_step(turn, worker, Pos(9, 5))
    assert step == Pos(6, 5)


def test_step_toward_diagonal(payload_factory, role_factory):
    """八方向移动：对角线走一步两个坐标同时变（任务书 §4.5.4 第 2 条）"""
    turn, worker = _turn(payload_factory, role_factory)
    step = grid.next_step(turn, worker, Pos(9, 9))
    assert step == Pos(6, 6)


def test_step_toward_returns_none_when_arrived(payload_factory, role_factory):
    turn, worker = _turn(payload_factory, role_factory)
    assert grid.next_step(turn, worker, worker.pos) is None


def test_path_goes_around_obstacles(payload_factory, role_factory, zone_factory):
    """障碍物要绕开（任务书 §4.1：矿区/中立单位/建筑都阻挡移动）"""
    # 在 (6,5) 与 (6,6) 之间竖一堵矿区墙，逼它绕行
    zones = [zone_factory("stone", 6, y) for y in range(0, 32) if y != 0]
    turn, worker = _turn(payload_factory, role_factory, zones=zones)
    step = grid.next_step(turn, worker, Pos(9, 5))
    assert step is not None
    # 不能直接往右撞进矿区
    assert step != Pos(6, 5)


def test_unreachable_returns_none(payload_factory, role_factory, zone_factory):
    """完全被围死时返回 None（而不是给一个撞墙的步）"""
    ring = [
        zone_factory("stone", 5 + dx, 5 + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    ]
    turn, worker = _turn(payload_factory, role_factory, zones=ring)
    assert grid.next_step(turn, worker, Pos(20, 20)) is None


def test_reserved_cells_are_avoided(payload_factory, role_factory):
    """本回合被别人认领的格子不能再走（避免"目标点争夺"碰撞）"""
    turn, worker = _turn(payload_factory, role_factory)
    step = grid.next_step(turn, worker, Pos(9, 5), reserved={Pos(6, 5)})
    assert step != Pos(6, 5)


def test_reachable_and_reachable_any(payload_factory, role_factory):
    turn, worker = _turn(payload_factory, role_factory)
    assert grid.reachable(turn, Pos(5, 5), Pos(9, 9)) is True
    assert grid.reachable(turn, Pos(5, 5), Pos(5, 5)) is True
    assert grid.reachable_any(turn, Pos(5, 5), (Pos(9, 9), Pos(8, 8))) is True


def test_reachable_respects_extra_blocked(payload_factory, role_factory):
    """`extra_blocked` 就是"建成预演"的基础：假设某些格被建筑占掉"""
    turn, worker = _turn(payload_factory, role_factory)
    wall = {Pos(x, 5) for x in range(0, 41) if x != 5}
    assert grid.reachable(turn, Pos(5, 5), Pos(20, 5), wall) is False


def test_stand_cells_are_adjacent_and_land(payload_factory, role_factory, zone_factory):
    """落脚点必须在目标周围一格内、是空地、且当前空闲"""
    turn, worker = _turn(payload_factory, role_factory)
    cells = grid.stand_cells(turn, Pos(10, 10))
    assert len(cells) == 8
    for cell in cells:
        assert max(abs(cell.x - 10), abs(cell.y - 10)) == 1
        assert turn.is_land(cell)


def test_stand_cells_filters_busy(payload_factory, role_factory):
    turn, worker = _turn(payload_factory, role_factory)
    busy = {Pos(9, 9), Pos(11, 11)}
    cells = grid.stand_cells(turn, Pos(10, 10), busy=busy)
    assert Pos(9, 9) not in cells
    assert Pos(11, 11) not in cells


def test_cells_in_radius_clips_to_map():
    cells = grid.cells_in_radius(Pos(0, 0), 1, width=41, height=32)
    assert Pos(-1, -1) not in cells
    assert Pos(0, 1) in cells
    assert len(cells) == 3


def test_within_cone_matches_gatling_rule():
    """加特林多目标必须在同一 90° 锥内（任务书 §4.5.4）"""
    origin = Pos(10, 10)
    anchor = Pos(13, 10)          # 正东
    inside = Pos(13, 11)          # 东北 ~45°
    outside = Pos(6, 13)          # 西北，与正东夹角 ~143° > 90°
    assert grid.within_cone(origin, anchor, inside, 90) is True
    assert grid.within_cone(origin, anchor, outside, 90) is False
    assert grid.within_cone(origin, anchor, anchor, 90) is True


def test_step_toward_any_picks_closest_goal(payload_factory, role_factory):
    turn, worker = _turn(payload_factory, role_factory)
    step = grid.step_toward_any(turn, worker, (Pos(20, 5), Pos(7, 5)))
    assert step == Pos(6, 5)  # 朝更近的 (7,5) 走
