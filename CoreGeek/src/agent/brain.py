"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。

策略概览：
    白天：工人优先建造武器工事（加特林/电磁狙击炮/火箭发射台），
          再采集石头建造围墙；围墙建完后把多余石头卖给小贩换金币。
          开拓者优先完成自进化类任务（任务点领取），无任务时跟随武器塔。
    夜晚：每个角色操控一座武器攻击机器人，优先攻击最近的目标。

本模块为无状态决策：每回合从 `Turn` 重新解析地图与单位状态，
不依赖任何跨回合缓存，可自动适应矿区刷新、单位移动与视野变化。
"""

import os
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
    # 中立单位
    VENDOR,
    # 任务点
    CHALLENGER_TASK_1,
    CHALLENGER_TASK_2,
    DEFENDER_TASK_1,
    DEFENDER_TASK_2,
    # 时间
    ROUNDS_PER_DAY,
    # 建造成本
    WEAPON_BUILD_COST,
    # 指令构建
    move_command,
    collect_command,
    build_command,
    attack_command,
    sell_command,
    accept_task_command,
    station_footprint,
)

# 策略常量
TOWER_LOADOUT = (GATLING, RAILGUN, ROCKET)  # 武器建造顺序
STONE_BATCH = 6  # 工人采集石头的批次大小
WALL_BUILD_PRIORITY = 1000  # 围墙建造优先级
SELL_BATCH = 10  # 卖给小贩的石头批次大小

# 是否在每天第一个回合提交LLM策略咨询prompt（可用环境变量 LLM_PROMPT=0 关闭）
# 每个游戏日的LLM调用有限额，每天只请求一次以节省额度
LLM_PROMPT_ENABLED = os.getenv("LLM_PROMPT", "1") != "0"

# 各阵营的任务点类型
_TASK_POINTS_BY_TEAM = {
    "challenger": (CHALLENGER_TASK_1, CHALLENGER_TASK_2),
    "defender": (DEFENDER_TASK_1, DEFENDER_TASK_2),
}


def decide(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    """
    主决策函数
    输入: 判题系统发来的游戏状态
    输出: (角色ID字符串: 指令字典, 提交给LLM的prompt)
    """
    turn = Turn.load(payload)
    commands: dict[int, dict[str, Any]] = {}

    if turn.is_day:
        _decide_day(turn, commands)
    else:
        _decide_night(turn, commands)

    prompt = _generate_strategy_prompt(turn, payload)

    # 转换key为字符串
    return {str(key): value for key, value in commands.items()}, prompt


# === 白天决策 ===


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
    """工人白天逻辑

    优先级: 建造武器工事 > 采集石头 > 建造围墙 > 卖石头换金币

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        tower_sites: 武器工事的规划位置（按建造顺序）
        towers_missing: 尚未建造的武器工事位置
        walls_missing: 尚未建造的围墙位置
        claimed: 已被其他角色占用的目标集合，用于避免多个角色争抢同一格
        commands: 指令输出字典（角色ID -> 指令）
    """
    # 优先建造武器
    if towers_missing and turn.gold >= WEAPON_BUILD_COST:
        for index, site in enumerate(tower_sites):
            if site in towers_missing and site not in claimed:
                weapon_type = TOWER_LOADOUT[index % len(TOWER_LOADOUT)]
                _build_or_walk(turn, worker, site, weapon_type, claimed, commands)
                return

    # 围墙已建完: 把多余石头卖给小贩换金币
    if not walls_missing:
        _trade_logic(turn, worker, claimed, commands)
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
    """开拓者白天逻辑

    优先级: 维持进行中的任务 > 前往任务点领取任务 > 跟随武器塔

    任务规则（任务书5章）:
        - 开拓者需在己方任务点周围一格内领取任务
        - 领取后离开任务点周围一格会导致任务强制结束
        - 任务结束后需要等待冷却，冷却期内 isValid 为 false

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        tower_sites: 武器工事规划位置（备用，供后续扩展）
        wall_order: 围墙建造顺序，用于避免开拓者占住建造点
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    # 1. 任务进行中: 留在任务点周围（离开会强制结束任务）
    if turn.phase_task:
        task_pos = _nearest_task_position(turn, pioneer.pos)
        if task_pos is not None:
            if distance(pioneer.pos, task_pos) > 1:
                step = _step_toward(turn, pioneer, task_pos, claimed)
                if step is not None:
                    commands[pioneer.unit_id] = move_command(step)
            return

    # 2. 有可接取的任务: 前往任务点并领取
    valid_tasks = [task for task in turn.player_tasks if task.is_valid]
    if valid_tasks:
        nearest_task = min(valid_tasks, key=lambda task: (
            distance(pioneer.pos, task.task_position),
            task.task_position.x,
            task.task_position.y,
        ))
        if distance(pioneer.pos, nearest_task.task_position) <= 1:
            commands[pioneer.unit_id] = accept_task_command()
            return
        step = _step_toward(turn, pioneer, nearest_task.task_position, claimed)
        if step is not None:
            commands[pioneer.unit_id] = move_command(step)
            return

    # 3. 没有可接取的任务: 跟随武器塔,为夜晚操控武器做准备
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


def _trade_logic(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """资源交易逻辑：围墙建完后把多余石头卖给小贩换金币

    小贩收购价随世界新闻波动（任务书4.6.1节），卖出所得可用于购买升级券。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    if worker.backpack.count(WALL_MATERIAL) < SELL_BATCH:
        return

    vendor = _nearest_zone(turn, VENDOR, worker.pos)
    if vendor is None:
        return

    # 已在小贩旁边: 直接贩卖
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(WALL_MATERIAL, SELL_BATCH)
        return

    # 否则走向小贩
    step = _step_toward(turn, worker, vendor, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)


# === 夜晚决策 ===


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
    """为武器配对操控角色

    配对策略:
        - 角色列表按角色ID升序排列（工人1 -> 工人2 -> 开拓者）
        - 武器列表按坐标排序（x 优先，其次 y）
        - 使用 zip() 顺序配对，角色数多于武器数时多余的被忽略

    返回:
        [(操控角色, 武器工事), ...] 配对列表
    """
    controllers = turn.controllable()
    weapons = turn.weapons()
    return list(zip(controllers, weapons))


def _find_attack_target(turn: Turn, weapon: Unit) -> Pos | None:
    """为武器寻找攻击目标（优先机器人）

    参数:
        turn: 当前回合信息
        weapon: 待操控的武器工事

    返回:
        攻击目标坐标；无可攻击目标时返回 None

    优先级:
        1. 攻击我方的存活机器人（选择最近的一台）
        2. 视野内的敌方单位（选择最近的一个）
    """
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


def _nearest_zone(turn: Turn, zone_type: str, origin: Pos) -> Pos | None:
    """查找离指定位置最近的中立元素（小贩、武器商店等）"""
    positions = [
        pos for pos, kind in turn.zones.items() if kind == zone_type
    ]
    if not positions:
        return None
    return min(positions, key=lambda pos: (distance(origin, pos), pos.x, pos.y))


def _nearest_task_position(turn: Turn, origin: Pos) -> Pos | None:
    """查找离指定位置最近的己方任务点坐标

    优先使用 playerTasks 中的任务点信息，缺失时回退到地图 zones 中
    本阵营的任务点类型。
    """
    positions = [task.task_position for task in turn.player_tasks]
    if not positions:
        kinds = _TASK_POINTS_BY_TEAM.get(turn.team_type, ())
        positions = [
            pos for pos, kind in turn.zones.items() if kind in kinds
        ]
    if not positions:
        return None
    return min(positions, key=lambda pos: (distance(origin, pos), pos.x, pos.y))


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
    """计算目标周围可停留的格子

    参数:
        turn: 当前回合信息
        unit: 移动的单位
        target: 目标位置（建造点、矿点、武器或小贩坐标）
        claimed: 已被其他单位声明的格子（防止多个角色在同一回合争抢同一格）
        inside_only: 为 True 时只保留基地周围1格范围内的格子，
                     用于避免开拓者跑去远处挡住工人的建造路线

    返回:
        按离基地距离升序排列的可停留格子列表（优先靠近基地）

    逻辑:
        1. 取目标位置的八方向相邻格子
        2. 过滤掉非陆地、被阻挡（建筑/单位/机器人/中立元素）以及已被占用的格子
        3. inside_only 为 True 时进一步限制在基地周围
        4. 按离基地的切比雪夫距离排序
    """
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
    """计算武器塔建造位置（基地周围一圈）

    取基地 2x2 占地周围距离为1且可通行的格子，按离基地的距离排序后取前3个，
    分别对应加特林、电磁狙击炮、火箭发射台。
    """
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
    """计算围墙建造顺序（基地周围第二圈）

    按“上边 -> 左边 -> 下边 -> 右边”的顺序环绕基地铺一圈围墙，
    并在右下角留一个入口供角色进出。超出地图或落在非陆地上的点会被过滤掉。
    """
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
        if pos != entrance
        and turn.land(pos)
        and 0 <= pos.x < turn.width
        and 0 <= pos.y < turn.height
    )


# === LLM 策略咨询 ===


def _generate_strategy_prompt(turn: Turn, payload: dict[str, Any]) -> str:
    """生成提交给LLM的策略咨询prompt（每个游戏日只请求一次）

    接口文档规定每个游戏日有LLM调用次数限制（errorCode=5），
    因此只在每天的第一个回合请求一次，并带上上一回合的LLM回复作为上下文。

    返回:
        需要提交给LLM的prompt；本回合不需要咨询时返回空字符串
    """
    if not LLM_PROMPT_ENABLED:
        return ""
    # 每天的第一个回合（第1、131、261...回合）
    if turn.round_no % ROUNDS_PER_DAY != 1:
        return ""

    robots = turn.alive_robots_targeting_me()
    weapons = turn.weapons()
    previous = str(payload.get("llmResp") or "").strip()

    lines = [
        "你是《未来战争》塔防对战的策略顾问。以下是当前局面，请给出本回合的作战建议。",
        f"回合 {turn.round_no}（第{(turn.round_no - 1) // ROUNDS_PER_DAY + 1}天白天）",
        f"阵营: {turn.team_type}，金币: {turn.gold}，积分: {turn.total_score}",
        f"我方武器: {len(weapons)}/3 座，围墙: {len(turn.walls())} 段",
        f"可控制角色: {len(turn.controllable())} 个",
        f"来袭机器人: {len(robots)} 个"
        f"（小型/中型/大型/BOSS尽量优先处理大型与BOSS）",
        f"可领取任务点: {sum(1 for t in turn.player_tasks if t.is_valid)} 个",
        "请用不超过5行中文说明：优先建造或升级什么、角色如何站位、是否值得去做任务。",
    ]
    if previous:
        lines.insert(1, f"上一回合LLM建议: {previous[:500]}")

    return "\n".join(lines)
