"""地图模块测试（设计文档 3.4 节）"""

from __future__ import annotations

from agent.grid import cells_in_range, get_neighbors, next_step
from agent.protocol import STONE_MINE, WORKER, Pos, Turn, distance


def test_get_neighbors():
    """八方向相邻格子"""
    neighbors = get_neighbors(Pos(5, 5))
    assert len(neighbors) == 8
    assert Pos(4, 4) in neighbors
    assert Pos(6, 6) in neighbors
    assert Pos(5, 5) not in neighbors


def test_cells_in_range():
    """范围内格子（切比雪夫距离）"""
    cells = cells_in_range(Pos(5, 5), 2)
    assert Pos(5, 5) in cells
    assert Pos(3, 3) in cells
    assert Pos(7, 7) in cells
    assert Pos(2, 2) not in cells  # 超出范围
    assert Pos(5, 8) not in cells
    assert len(cells) == 5 * 5  # (2r+1)^2


def _worker_turn(payload_factory, role_factory, **kwargs) -> Turn:
    """构造一个只含基地与一个工人的回合，供寻路测试使用"""
    payload = payload_factory(
        station=(20, 20),
        roles=[role_factory(10010, WORKER, 0, 0, backPackCapability=100)],
        **kwargs,
    )
    return Turn.load(payload)


def test_next_step_moves_closer(payload_factory, role_factory):
    """寻路返回的下一步必须靠近目标且与当前位置相邻"""
    turn = _worker_turn(payload_factory, role_factory)
    worker = turn.workers()[0]
    goal = Pos(5, 0)

    step = next_step(turn, worker, goal)
    assert step is not None
    assert distance(step, worker.pos) == 1  # 每回合只能移动一格
    assert distance(step, goal) < distance(worker.pos, goal)


def test_next_step_same_position(payload_factory, role_factory):
    """起点即终点时直接返回该点（不应崩溃）"""
    turn = _worker_turn(payload_factory, role_factory)
    worker = turn.workers()[0]
    assert next_step(turn, worker, worker.pos) == worker.pos


def test_next_step_avoids_blocked_cells(payload_factory, role_factory):
    """阻挡格（矿区/基地）不会作为落点"""
    turn = _worker_turn(
        payload_factory, role_factory, zones=[(STONE_MINE, 1, 0), (STONE_MINE, 1, 1)],
    )
    worker = turn.workers()[0]

    step = next_step(turn, worker, Pos(5, 0))
    assert step is not None
    assert step not in turn.blocked(worker)
    assert turn.land(step)


def test_next_step_unreachable(payload_factory, role_factory):
    """被围死时返回 None"""
    turn = _worker_turn(
        payload_factory,
        role_factory,
        zones=[
            (STONE_MINE, 1, 0),
            (STONE_MINE, 0, 1),
            (STONE_MINE, 1, 1),
        ],
    )
    worker = turn.workers()[0]
    assert next_step(turn, worker, Pos(5, 0)) is None


def test_next_step_reaches_goal_around_obstacle(payload_factory, role_factory):
    """绕开障碍后仍能到达目标（多步模拟）"""
    turn = _worker_turn(
        payload_factory, role_factory, zones=[(STONE_MINE, 1, 0)],
    )
    goal = Pos(3, 0)

    # 模拟移动，最多 20 步
    for _ in range(20):
        worker = turn.workers()[0]
        if worker.pos == goal:
            break
        step = next_step(turn, worker, goal)
        assert step is not None
        # 手动把工人挪到下一步（重建 Turn）
        turn = Turn.load(_payload_with_worker(payload_factory, step))

    assert turn.workers()[0].pos == goal


def _payload_with_worker(payload_factory, pos: Pos) -> dict:
    """构造工人位于指定坐标的 payload（配合多步寻路测试使用）"""
    return payload_factory(
        station=(20, 20),
        roles=[{
            "id": 10010,
            "pos": {"x": pos.x, "y": pos.y},
            "roleType": WORKER,
            "health": 220,
            "attackPower": 0,
            "attackRange": 0,
            "backPackCapability": 100,
            "backpack": [],
        }],
        zones=[(STONE_MINE, 1, 0)],
    )
