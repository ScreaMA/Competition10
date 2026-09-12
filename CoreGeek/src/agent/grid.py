"""地图模块：路径规划、地图查询。

对应设计文档 3.4 节。
"""

from heapq import heappop, heappush
from itertools import count

from .protocol import Pos, Turn, Unit, distance

# 八方向移动
_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def next_step(turn: Turn, moving: Unit, goal: Pos) -> Pos | None:
    """
    计算从当前位置到目标的下一步
    使用A*算法找到最短路径,返回第一步应该移动到的位置
    """
    blocked = turn.blocked(moving)
    order = count()  # 用于打破优先级相同时的平局

    # 优先队列: (f值, 实际代价, 序号, 位置)
    frontier: list[tuple[int, int, int, Pos]] = [
        (distance(moving.pos, goal), 0, next(order), moving.pos)
    ]

    came_from: dict[Pos, Pos] = {}
    best_cost = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier:
        _, cost, _, current = heappop(frontier)

        if current in seen:
            continue

        # 找到目标
        if current == goal:
            return _first_step(came_from, moving.pos, goal)

        seen.add(current)

        # 探索相邻格子
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)

            # 跳过不可通行的格子
            if step in blocked or not turn.land(step):
                continue

            new_cost = cost + 1

            # 跳过已有更优路径的格子
            if new_cost >= best_cost.get(step, new_cost + 1):
                continue

            best_cost[step] = new_cost
            came_from[step] = current

            # 加入优先队列
            heappush(
                frontier,
                (
                    new_cost + distance(step, goal),  # f = g + h
                    new_cost,
                    next(order),
                    step,
                ),
            )

    # 无法到达
    return None


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    """回溯路径,找到起点的下一步"""
    # 起点即终点时无需移动
    if goal == start:
        return goal

    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current


def get_neighbors(pos: Pos) -> tuple[Pos, ...]:
    """获取八方向相邻格子"""
    return tuple(Pos(pos.x + dx, pos.y + dy) for dx, dy in _STEPS)


def cells_in_range(center: Pos, radius: int) -> set[Pos]:
    """获取指定范围内的所有格子（切比雪夫距离）"""
    cells = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) <= radius:
                cells.add(Pos(center.x + dx, center.y + dy))
    return cells
