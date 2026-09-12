"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。
"""

from typing import Any

from .grid import next_step, get_neighbors, cells_in_range
from .protocol import (
    Turn,
    Unit,
    Pos,
    distance,
    # 单位类型
    WORKER,
    PIONEER,
    STATION,
    GATLING,
    RAILGUN,
    ROCKET,
    WALL,
    TOWER_TYPES,
    # 矿石
    STONE_MINE,
    IRON_MINE,
    COPPER_MINE,
    WALL_MATERIAL,
    # 建造成本
    WEAPON_BUILD_COST,
    # 指令构建
    move_command,
    collect_command,
    build_command,
    attack_command,
    sell_command,
    station_footprint,
)

# 策略常量
TOWER_LOADOUT = (GATLING, RAILGUN, ROCKET)  # 武器建造顺序
STONE_BATCH = 6  # 工人采集石头的批次大小
WALL_BUILD_PRIORITY = 1000  # 围墙建造优先级


def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    主决策函数
    输入: 判题系统发来的游戏状态
    输出: {角色ID字符串: 指令字典}
    """
    turn = Turn.load(payload)
    commands: dict[int, dict[str, Any]] = {}

    if turn.is_day:
        _decide_day(turn, commands)
    else:
        _decide_night(turn, commands)

    # 转换key为字符串
    return {str(key): value for key, value in commands.items()}


def _decide_day(turn: Turn, commands: dict[int, dict[str, Any]]) -> None:
    """白天策略: 建造、采集、任务"""
    # 计算需要建造的位置
    tower_sites = _calc_tower_sites(turn)
    wall_order = _calc_wall_order(turn)

    # 统计已建造的武器和围墙
    standing_towers = {unit.pos for unit in turn.weapons()}
    standing_walls = {unit.pos for unit in turn.walls()}
    occupied = turn.occupied_cells()

    # 计算缺少的建筑
    towers_missing = [pos for pos in tower_sites if pos not in standing_towers]
    walls_missing = [pos for pos in wall_order if pos not in standing_walls]

    # 过滤掉已被占据的位置
    free_towers = [pos for pos in towers_missing if pos not in occupied]
    free_walls = [pos for pos in walls_missing if pos not in occupied]

    # 已分配的位置（防止多个角色走向同一位置）
    claimed: set[Pos] = set()

    # 为每个工人分配任务
    for worker in turn.workers():
        _worker_day_logic(
            turn, worker, tower_sites, free_towers, free_walls, claimed, commands,
        )

    # 开拓者行为（任务、宝藏）
    for pioneer in turn.pioneers():
        _pioneer_day_logic(
            turn, pioneer, tower_sites, wall_order, claimed, commands,
        )


def _worker_day_logic(
    turn: Turn,
    worker: Unit,
    tower_sites: tuple[Pos, ...],
    towers_missing: list[Pos],
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """工人白天逻辑"""
    # 优先建造武器
    if towers_missing and turn.gold >= WEAPON_BUILD_COST:
        for index, site in enumerate(tower_sites):
            if site in towers_missing and site not in claimed:
                weapon_type = TOWER_LOADOUT[index % len(TOWER_LOADOUT)]
                _build_or_walk(turn, worker, site, weapon_type, claimed, commands)
                return

    # 没有围墙要建,直接返回
    if not walls_missing:
        return

    # 检查背包里的石头数量
    stones = worker.backpack.count(WALL_MATERIAL)

    # 如果旁边有矿且石头不足,采集
    mine = _adjacent_mine(turn, worker, STONE_MINE)
    if mine is not None and stones < STONE_BATCH:
        commands[worker.unit_id] = collect_command(mine)
        claimed.add(mine)
        return

    # 如果有石头,去建造围墙
    if stones > 0:
        for site in walls_missing:
            if site not in claimed:
                _build_or_walk(turn, worker, site, WALL, claimed, commands)
                return
        return

    # 没石头,去采矿
    _go_mine(turn, worker, STONE_MINE, claimed, commands)


def _pioneer_day_logic(
    turn: Turn,
    pioneer: Unit,
    tower_sites: tuple[Pos, ...],
    wall_order: tuple[Pos, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """开拓者白天逻辑"""
    # TODO: 实现任务领取和完成逻辑
    # 当前策略: 跟随武器塔,不阻挡工人
    weapons = turn.weapons()
    if not weapons:
        return

    # 找到最近的武器塔
    nearest_weapon = min(weapons, key=lambda w: distance(pioneer.pos, w.pos))

    # 如果已经在武器旁边且不在围墙建造点上,不动
    if distance(pioneer.pos, nearest_weapon.pos) <= 1:
        if pioneer.pos not in wall_order:
            return

    # 否则向武器塔靠近（但只在基地周围移动）
    step = _step_toward(turn, pioneer, nearest_weapon.pos, claimed, inside_only=True)
    if step is not None:
        commands[pioneer.unit_id] = move_command(step)


def _decide_night(turn: Turn, commands: dict[int, dict[str, Any]]) -> None:
    """夜晚策略: 操控武器攻击"""
    claimed: set[Pos] = set()

    # 为每个武器配对一个操控角色
    for controller, weapon in _pair_controllers_and_weapons(turn):
        # 检查操控角色是否在武器旁边
        if distance(controller.pos, weapon.pos) <= 1:
            # 武器在冷却中,跳过
            if weapon.cooldown > 0:
                continue

            # 寻找攻击目标
            target = _find_attack_target(turn, weapon)
            if target is not None:
                commands[weapon.unit_id] = attack_command(controller.unit_id, [target])
            continue

        # 操控角色不在武器旁边,向武器移动
        step = _step_toward(turn, controller, weapon.pos, claimed)
        if step is not None:
            commands[controller.unit_id] = move_command(step)


def _pair_controllers_and_weapons(turn: Turn) -> list[tuple[Unit, Unit]]:
    """为武器配对操控角色"""
    controllers = turn.controllable()
    weapons = turn.weapons()
    return list(zip(controllers, weapons))


def _find_attack_target(turn: Turn, weapon: Unit) -> Pos | None:
    """为武器寻找攻击目标（优先机器人）"""
    weapon_range = weapon.range_of_attack()

    # 优先攻击机器人
    robots = turn.alive_robots_targeting_me()
    targets_in_range = [
        robot for robot in robots
        if distance(weapon.pos, robot.pos) <= weapon_range
    ]

    if targets_in_range:
        # 攻击最近的机器人
        nearest = min(targets_in_range, key=lambda r: (
            distance(weapon.pos, r.pos),
            r.robot_id,
        ))
        return nearest.pos

    # 没有机器人,尝试攻击敌方单位
    enemies = turn.enemies
    enemy_targets = [
        enemy for enemy in enemies
        if enemy.is_alive and distance(weapon.pos, enemy.pos) <= weapon_range
    ]

    if enemy_targets:
        nearest = min(enemy_targets, key=lambda e: (
            distance(weapon.pos, e.pos),
            e.unit_id,
        ))
        return nearest.pos

    return None


# === 辅助函数 ===


def _adjacent_mine(turn: Turn, unit: Unit, mine_type: str) -> Pos | None:
    """查找相邻的指定类型矿点"""
    mines = turn.get_mines(mine_type)
    adjacent = [
        mine for mine in mines
        if unit.pos != mine and distance(unit.pos, mine) <= 1
    ]
    if not adjacent:
        return None
    # 返回最近的矿点
    return min(adjacent, key=lambda m: (distance(unit.pos, m), m.x, m.y))


def _go_mine(
    turn: Turn,
    unit: Unit,
    mine_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """让单位去采矿"""
    if unit.backpack_full:
        return False

    mines = [
        m for m in turn.get_mines(mine_type)
        if m not in claimed
    ]
    if not mines:
        return False

    # 按距离排序
    mines.sort(key=lambda m: (distance(unit.pos, m), m.x, m.y))

    for mine in mines:
        # 如果已经相邻,采集
        if unit.pos != mine and distance(unit.pos, mine) <= 1:
            commands[unit.unit_id] = collect_command(mine)
            claimed.add(mine)
            return True

        # 否则向矿点移动
        step = _step_toward(turn, unit, mine, claimed)
        if step is not None:
            commands[unit.unit_id] = move_command(step)
            return True

    return False


def _build_or_walk(
    turn: Turn,
    unit: Unit,
    target: Pos,
    building_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """如果在建造位置旁边则建造,否则向目标移动"""
    # 已经在目标旁边(但不是目标本身),执行建造
    if unit.pos != target and distance(unit.pos, target) <= 1:
        commands[unit.unit_id] = build_command(target, building_type)
        claimed.add(target)
        return

    # 否则向目标移动
    step = _step_toward(turn, unit, target, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)


def _step_toward(
    turn: Turn,
    unit: Unit,
    target: Pos,
    claimed: set[Pos],
    *,
    inside_only: bool = False,
) -> Pos | None:
    """计算向目标移动的下一步"""
    # 计算可停留的格子
    stand_cells = _valid_stand_cells(turn, unit, target, claimed, inside_only)

    for stand in stand_cells:
        # 已经在目标位置
        if stand == unit.pos:
            return None

        # 计算路径
        step = next_step(turn, unit, stand)
        if step is None or step in claimed:
            continue

        claimed.add(step)
        return step

    return None


def _valid_stand_cells(
    turn: Turn,
    unit: Unit,
    target: Pos,
    claimed: set[Pos],
    inside_only: bool = False,
) -> list[Pos]:
    """计算目标周围可停留的格子"""
    station = turn.station()
    footprint = station_footprint(station.pos) if station else ()
    blocked = turn.blocked(unit)

    # 目标的八方向相邻格子
    neighbors = get_neighbors(target)

    cells = [
        pos for pos in neighbors
        if turn.land(pos)
        and pos not in blocked
        and (pos == unit.pos or pos not in claimed)
        and (
            not inside_only
            or _footprint_distance(pos, footprint) <= 1
        )
    ]

    # 按离基地的距离排序（优先靠近基地）
    cells.sort(key=lambda pos: (_footprint_distance(pos, footprint), pos.x, pos.y))
    return cells


def _footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    """计算点到footprint的最小距离"""
    if not footprint:
        return 0
    return min(distance(pos, cell) for cell in footprint)


def _calc_tower_sites(turn: Turn) -> tuple[Pos, ...]:
    """计算武器塔建造位置（基地周围一圈）"""
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)

    # 基地footprint周围距离为1的格子
    candidates = []
    for cell in footprint:
        for neighbor in get_neighbors(cell):
            if neighbor not in footprint and turn.land(neighbor):
                candidates.append(neighbor)

    # 去重并排序
    candidates = list(set(candidates))
    candidates.sort(key=lambda pos: (_footprint_distance(pos, footprint), pos.x, pos.y))

    # 取前3个位置
    return tuple(candidates[:3])


def _calc_wall_order(turn: Turn) -> tuple[Pos, ...]:
    """计算围墙建造顺序（基地周围第二圈）"""
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 矩形外围（留一个入口）
    order = [
        # 上边（从右到左）
        *(Pos(x, ymax + 2) for x in range(xmax + 2, xmin - 3, -1)),
        # 左边（从上到下）
        *(Pos(xmin - 2, y) for y in range(ymax + 1, ymin - 2, -1)),
        # 下边（从左到右）
        *(Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)),
        # 右边（从下到上）
        *(Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)),
    ]

    # 留一个入口（右下角）
    entrance = Pos(xmax + 2, ymin - 1)

    return tuple(
        pos for pos in order
        if pos != entrance and turn.land(pos)
    )
