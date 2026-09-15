"""几何与寻路：A*、可达性、区域枚举。

对应设计文档V2 §5.3。

这里的所有函数都是**纯函数**：输入是 `Turn`（或 `World`）与坐标，输出是坐标，
不读也不写任何跨回合状态。
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count

from .protocol import (
    LAND,
    NEIGHBOUR_OFFSETS,
    Pos,
    Turn,
    Unit,
    distance,
    neighbours,
)

# 把邻居展开成偏移量，避免在内层循环里反复构造 Pos
_STEPS = NEIGHBOUR_OFFSETS


def manhattan_priority(a: Pos, b: Pos) -> int:
    """A* 的启发式：用切比雪夫距离（与实际移动代价一致，可采纳）"""
    return distance(a, b)


def next_step(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    reserved: frozenset[Pos] | set[Pos] = frozenset(),
    extra_blocked: frozenset[Pos] | set[Pos] = frozenset(),
) -> Pos | None:
    """朝 goal 走一步（A*，八方向）

    参数:
        turn: 当前回合
        moving: 要移动的角色
        goal: 目标格
        reserved: 本回合已被其他角色认领的格子（避免自家人抢同一格导致碰撞）
        extra_blocked: 额外视为障碍的格子（用于"预演建成后的连通性"）

    返回:
        下一步坐标；起点即终点或不可达时返回 None
    """
    if moving.pos == goal:
        return None

    blocked = set(turn.blocked_for(moving)) | set(extra_blocked)
    blocked.discard(moving.pos)

    start = moving.pos
    order = count()
    frontier: list[tuple[int, int, int, int, int, Pos]] = [
        (manhattan_priority(start, goal), manhattan_priority(start, goal),
         _manhattan(start, goal), 0, next(order), start)
    ]
    came_from: dict[Pos, Pos] = {}
    best: dict[Pos, int] = {start: 0}
    seen: set[Pos] = set()

    while frontier:
        _, _, _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current == goal:
            return _first_step(came_from, start, goal)
        seen.add(current)

        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in reserved or step in blocked or not turn.is_land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (
                    new_cost + manhattan_priority(step, goal),  # f：总代价估计
                    manhattan_priority(step, goal),             # h：越接近目标越优先
                    _manhattan(step, goal),                     # 越贴近直线越优先
                    new_cost,                                   # g
                    next(order),
                    step,
                ),
            )
    return None


def _manhattan(a: Pos, b: Pos) -> int:
    """曼哈顿距离，仅用于 A* 的**平局打破**

    切比雪夫距离下"斜着绕一步"和"直着走一步"代价相同，不加这一项时 A*
    会随机挑一条——表现就是角色走位歪歪扭扭（对战复盘里
    "三角色在基地周边小幅挪动"的现象之一）。h 相同（同样接近目标）时，
    优先选曼哈顿距离更小的那一步，等价于"尽量走直线"。
    """
    return abs(a.x - b.x) + abs(a.y - b.y)


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from.get(current, start) != start:
        current = came_from[current]
    return current


def step_toward_any(
    turn: Turn,
    moving: Unit,
    goals: tuple[Pos, ...],
    reserved: frozenset[Pos] | set[Pos] = frozenset(),
) -> Pos | None:
    """朝一组候选目标中最近的一个走一步

    按切比雪夫距离由近到远试，**第一个能走通的就是最优解**，立刻返回——
    不继续试更远的目标。多跑几次 A* 在每回合都要调用几十次的情况下是
    实打实的开销（尤其是 `tower_sites` 里那种成组的落脚点）。
    """
    if not goals:
        return None
    ranked = sorted(goals, key=lambda g: (distance(moving.pos, g), g.x, g.y))
    for goal in ranked:
        step = next_step(turn, moving, goal, reserved)
        if step is not None:
            return step
    return None


def steps_to_any(
    turn: Turn,
    origin: Pos,
    goals: tuple[Pos, ...] | frozenset[Pos] | set[Pos],
    limit: int,
) -> int | None:
    """`origin` 走到任意一个 `goals` 的最少步数（BFS，八方向等代价）

    超过 `limit` 步就返回 None——调用方（天黑前回防）只关心"来不来得及"，
    不关心确切步数，**深度上限就是性能上限**：不限深的话，白天每回合给每个
    角色做一次全图 BFS，测试套件直接从 8s 涨到 28s。

    用 BFS 步数而不是切比雪夫距离，是因为围墙围起来之后"直线 3 格"可能要绕到
    缺口再进去、实际十几步；按直线距离判断出发时机，角色会在路上过完前半个
    夜晚——那正是"炮塔没人操控"的形态。
    """
    targets = set(goals)
    if not targets:
        return None
    if origin in targets:
        return 0
    if limit <= 0:
        return None

    blocked = {pos for pos, name in turn.zones.items() if name != LAND}
    blocked.discard(origin)
    seen = {origin}
    frontier = deque([(origin, 0)])
    while frontier:
        current, depth = frontier.popleft()
        if depth >= limit:
            continue
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in seen or step in blocked or not turn.is_land(step):
                continue
            if step in targets:
                return depth + 1
            seen.add(step)
            frontier.append((step, depth + 1))
    return None


def reachable(
    turn: Turn,
    origin: Pos,
    goal: Pos,
    extra_blocked: frozenset[Pos] | set[Pos] = frozenset(),
) -> bool:
    """origin 能否走到 goal（BFS 可达性，忽略其他角色的占位）

    用于"预演建成后的连通性"（设计文档V2 §5.4）：假设若干格被新建筑占据，
    判断某个落脚点是否还能从基地走到。
    """
    if origin == goal:
        return True
    blocked = {pos for pos, name in turn.zones.items() if name != LAND}
    blocked |= set(extra_blocked)
    blocked.discard(origin)
    if goal in blocked:
        return False

    seen = {origin}
    queue = deque([origin])
    while queue:
        current = queue.popleft()
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in seen or step in blocked:
                continue
            if not turn.is_land(step):
                continue
            if step == goal:
                return True
            seen.add(step)
            queue.append(step)
    return False


def reachable_any(
    turn: Turn,
    origin: Pos,
    goals: tuple[Pos, ...],
    extra_blocked: frozenset[Pos] | set[Pos] = frozenset(),
) -> bool:
    return any(reachable(turn, origin, goal, extra_blocked) for goal in goals)


def stand_cells(
    turn: Turn,
    target: Pos,
    *,
    busy: frozenset[Pos] | set[Pos] = frozenset(),
    occupants: frozenset[Pos] | set[Pos] = frozenset(),
) -> tuple[Pos, ...]:
    """目标格周围一格内、可站立的落脚点

    "周围一格"= 切比雪夫距离 1（任务书 §4.5.4）。采集/建造/贩卖/购买/领任务
    都要求角色在这个范围内。

    参数:
        busy: 本回合已被别人认领的格子
        occupants: 被单位占据的格子。**必须传**（调用方记得把自己的位置从里面
            减掉）：把别人占着的格子当落脚点会让 A* 搜遍全图也走不到，
            只是白花时间——实测这一项占掉了整回合决策时间的七成。
    """
    cells = []
    for cell in neighbours(target):
        if not turn.is_land(cell):
            continue
        if cell in busy or cell in occupants:
            continue
        cells.append(cell)
    return tuple(sorted(cells, key=lambda p: (p.x, p.y)))


def stand_cells_for(
    turn: Turn,
    target: Pos,
    mover: Unit,
    *,
    busy: frozenset[Pos] | set[Pos] = frozenset(),
) -> tuple[Pos, ...]:
    """`stand_cells` 的常用包装：自动把 mover 自己从占据集合里减掉"""
    return stand_cells(
        turn,
        target,
        busy=busy,
        occupants=turn.occupied() - {mover.pos},
    )


def cells_in_radius(center: Pos, radius: int, width: int, height: int) -> tuple[Pos, ...]:
    """以 center 为中心、切比雪夫半径 radius 的方格环（含边界裁剪，不含中心）"""
    result = []
    for x in range(center.x - radius, center.x + radius + 1):
        for y in range(center.y - radius, center.y + radius + 1):
            pos = Pos(x, y)
            if pos == center:
                continue
            if 0 <= x < width and 0 <= y < height:
                result.append(pos)
    return tuple(result)


def ring_at(cells: tuple[Pos, ...], center: Pos, radius: int) -> tuple[Pos, ...]:
    """从 cells 中挑出距离 center 恰好为 radius 的那一圈"""
    return tuple(sorted(c for c in cells if distance(c, center) == radius))


def angle_rank(origin: Pos, target: Pos) -> float:
    """target 相对 origin 的极角（弧度，供锥形判定使用）"""
    import math

    return math.atan2(target.y - origin.y, target.x - origin.x)


def within_cone(origin: Pos, anchor: Pos, other: Pos, degrees: float) -> bool:
    """other 是否落在以 anchor 为轴、张角为 degrees 的锥形内

    加特林的多目标必须落在同一 90° 锥内（任务书 §4.5.4）：任意两个目标相对
    加特林的方向夹角 ≤ 90°，否则整次攻击非法。用"锚点 + 张角"实现，
    比两两比较更简单也更严格。
    """
    import math

    if anchor == origin:
        return True
    a = angle_rank(origin, anchor)
    b = angle_rank(origin, other)
    diff = abs(a - b) % (2 * math.pi)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return math.degrees(diff) <= degrees
