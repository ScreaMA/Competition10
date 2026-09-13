"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。

策略概览：
    白天：工人优先建造武器工事（加特林/电磁狙击炮/火箭发射台），
          再采集石头建造围墙；围墙建完后用富余资源换取金币和武器升级。
          开拓者优先完成自进化类任务（任务点领取 + 沙盒作答），
          无任务时跟随武器塔；天黑前工人回防到武器旁。
    夜晚：每个角色操控一座武器攻击机器人，优先攻击威胁最高的目标。

本模块为无状态决策：每回合从 `Turn` 重新解析地图与单位状态，
不依赖任何跨回合缓存，可自动适应矿区刷新、单位移动与视野变化。
任务答案同理，直接从上一回合的沙盒输出（`lastCmdResult`）中解析。
"""

import os
import re
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
    WEAPON_SHOP,
    # 任务点
    CHALLENGER_TASK_1,
    CHALLENGER_TASK_2,
    DEFENDER_TASK_1,
    DEFENDER_TASK_2,
    # 时间
    DAY_ROUNDS,
    ROUNDS_PER_DAY,
    # 建造成本
    WEAPON_BUILD_COST,
    # 指令构建
    move_command,
    collect_command,
    build_command,
    attack_command,
    sell_command,
    buy_command,
    use_command,
    accept_task_command,
    submit_answer_command,
    station_footprint,
)

# 策略常量
TOWER_LOADOUT = (GATLING, RAILGUN, ROCKET)  # 武器建造顺序
STONE_BATCH = 3  # 工人采集石头的批次大小（越小围墙越早开工）
WALL_BUILD_PRIORITY = 1000  # 围墙建造优先级
SELL_BATCH = 10  # 卖给小贩的石头批次大小
DUSK_ROUNDS = 5  # 天黑前提前回防的回合数
WEAPON_UPGRADE_VOUCHER = "WeaponUpgradeVoucher1"  # 武器升级券（level1->level2）
UPGRADE_GOLD = 100  # 购买一张武器升级券所需金币

# 基地四个方位（用于让武器塔分散布防，顺序仅用于同分时的稳定排序）
TOWER_SIDES = ("up", "left", "down", "right")

# 机器人威胁等级：数值越大越优先处理（与LLM prompt中的提示保持一致）
ROBOT_THREAT = {
    "bossRobot": 3,
    "largeRobot": 2,
    "middleRobot": 1,
    "smallRobot": 0,
}

# 自进化任务：从任务描述中识别需要在沙盒里读取的文件名
# 只认ASCII字符，避免把“请阅读”这类描述文字一起吃进文件名
TASK_FILE_PATTERN = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:md|txt|json|csv|log)")
# 沙盒输出中的任务标识前缀，用于确认输出属于当前任务
TASK_MARKER = "[TASK]"

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


def sandbox_command(payload: dict[str, Any]) -> str:
    """生成提交给沙盒执行的命令（自进化任务期间使用）

    判题系统仅在接受任务到任务结束期间允许执行沙盒命令，命令的输出会在
    下一回合通过请求的 `lastCmdResult` 字段返回，再由 `_task_answer`
    解析成 `submitAnswer` 的答案。

    返回:
        需要提交给沙盒执行的shell命令；非任务期间或已有答案时返回空字符串
    """
    return _sandbox_command(Turn.load(payload))


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

    # 天黑前留出回防时间，避免夜晚武器无人操控而空转
    dusk = _rounds_to_night(turn) <= DUSK_ROUNDS

    # 为每个工人分配任务
    for worker in turn.workers():
        if dusk:
            _fall_back_to_weapons(turn, worker, claimed, commands)
            continue
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

    优先级: 建造武器工事 > 采集石头 > 建造围墙
            > 围墙建完后: 武器升级 > 卖石头换金币

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

    # 围墙已建完: 把富余资源换成战力（武器升级 > 卖石头换金币）
    if not walls_missing:
        if _upgrade_weapon_with_gold(turn, worker, claimed, commands):
            return
        _trade_logic(turn, worker, claimed, commands)
        if worker.unit_id in commands:
            return
        # 手里还没有可卖的矿石: 继续采集,攒够一批再换金币
        _go_mine(turn, worker, STONE_MINE, claimed, commands)
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

    优先级: 维持进行中的任务（含提交答案） > 前往任务点领取任务 > 跟随武器塔

    任务规则（任务书5章）:
        - 开拓者需在己方任务点周围一格内领取任务
        - 领取后离开任务点周围一格会导致任务强制结束
        - 任务结束后需要等待冷却，冷却期内 isValid 为 false
        - 自进化类任务需在沙盒中取数后作答，答案经 `submitAnswer` 提交

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        tower_sites: 武器工事规划位置（备用，供后续扩展）
        wall_order: 围墙建造顺序，用于避免开拓者占住建造点
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    # 1. 任务进行中: 留在任务点周围（离开会强制结束任务），拿到沙盒输出后作答
    if turn.phase_task:
        task_pos = _nearest_task_position(turn, pioneer.pos)
        if task_pos is not None:
            if distance(pioneer.pos, task_pos) > 1:
                step = _step_toward(turn, pioneer, task_pos, claimed)
                if step is not None:
                    commands[pioneer.unit_id] = move_command(step)
                return
            # 沙盒命令的输出上一回合才返回，这里按任务标识取出本任务的答案
            answer = _task_answer(turn)
            if answer is not None:
                commands[pioneer.unit_id] = submit_answer_command(answer)
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


def _upgrade_weapon_with_gold(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """把富余金币换成武器升级券并用于武器（level1 -> level2）

    任务书4.6.3节：升级券在武器商店购买，需在目标武器周围一格内使用，
    升级后武器恢复到满血，攻击力与射程同时提升。

    因为升级券先买后用、跨回合存在背包里，这里按背包内容分两步走：
    背包里已有券就直接去武器旁使用，否则到武器商店购买。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（购买/使用/移动），调用方应直接返回
    """
    # 1. 身上有券: 去武器旁使用
    if WEAPON_UPGRADE_VOUCHER in worker.backpack:
        upgradable = [
            weapon for weapon in turn.weapons()
            if weapon.level < 2 and weapon.pos not in claimed
        ]
        if not upgradable:
            return False
        weapon = min(
            upgradable,
            key=lambda w: (distance(worker.pos, w.pos), w.pos.x, w.pos.y),
        )
        if distance(worker.pos, weapon.pos) <= 1:
            commands[worker.unit_id] = use_command(
                WEAPON_UPGRADE_VOUCHER, weapon.pos,
            )
            claimed.add(weapon.pos)
            return True
        step = _step_toward(turn, worker, weapon.pos, claimed)
        if step is not None:
            commands[worker.unit_id] = move_command(step)
            return True
        return False

    # 2. 金币足够: 去武器商店购买
    if turn.gold < UPGRADE_GOLD or worker.backpack_full:
        return False

    shop = _nearest_zone(turn, WEAPON_SHOP, worker.pos)
    if shop is None:
        return False

    if distance(worker.pos, shop) <= 1:
        commands[worker.unit_id] = buy_command(WEAPON_UPGRADE_VOUCHER)
        return True

    step = _step_toward(turn, worker, shop, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True
    return False


def _rounds_to_night(turn: Turn) -> int:
    """距离天黑还剩多少回合（含当前回合）"""
    day_round = (turn.round_no - 1) % ROUNDS_PER_DAY
    return DAY_ROUNDS - day_round


def _fall_back_to_weapons(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """天黑前把角色召回最近的武器旁待命

    武器工事必须由角色操控才会开火（任务书4.4节），提前回防可以避免
    夜晚首个回合武器无人操控而白白空转。

    参数:
        turn: 当前回合信息
        unit: 待召回的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    weapons = turn.weapons()
    if not weapons:
        return

    nearest = min(
        weapons,
        key=lambda w: (distance(unit.pos, w.pos), w.pos.x, w.pos.y),
    )
    if distance(unit.pos, nearest.pos) <= 1:
        return

    step = _step_toward(turn, unit, nearest.pos, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)


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
        - 每轮取"距离最近的 操控者-武器 组合"，配对后双方一起移出候选，
          再继续配对剩下的，直到角色或武器用完
        - 距离相同时按角色ID、武器ID排序，保证结果稳定

    按距离就近配对可以让角色少跑路，夜晚首个回合更容易全部就位开火。

    返回:
        [(操控角色, 武器工事), ...] 配对列表
    """
    candidates = sorted(
        (
            (distance(controller.pos, weapon.pos),
             controller.unit_id, weapon.unit_id, controller, weapon)
            for controller in turn.controllable()
            for weapon in turn.weapons()
        ),
        key=lambda item: item[:3],
    )

    pairs: list[tuple[Unit, Unit]] = []
    paired_controllers: set[int] = set()
    paired_weapons: set[int] = set()
    for _, controller_id, weapon_id, controller, weapon in candidates:
        if controller_id in paired_controllers or weapon_id in paired_weapons:
            continue
        paired_controllers.add(controller_id)
        paired_weapons.add(weapon_id)
        pairs.append((controller, weapon))

    return pairs


def _find_attack_target(turn: Turn, weapon: Unit) -> Pos | None:
    """为武器寻找攻击目标（优先机器人）

    参数:
        turn: 当前回合信息
        weapon: 待操控的武器工事

    返回:
        攻击目标坐标；无可攻击目标时返回 None

    优先级:
        1. 攻击我方的存活机器人（先按威胁分级，同级取最近的一台）
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
        # 大型/BOSS机器人威胁更高，优先处理；同级再取最近的
        nearest = min(targets_in_range, key=lambda r: (
            -ROBOT_THREAT.get(r.kind, 0),
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


# === 自进化任务（沙盒） ===


def _sandbox_command(turn: Turn) -> str:
    """任务期间需要提交给沙盒执行的命令

    自进化类任务的原文描述通常形如“请阅读task_1_beijing.md”，需要在沙盒
    中读取对应文件后才能作答。命令带上任务标识，便于下一回合确认输出
    属于当前任务；已经拿到本任务的输出后就不再重复执行。
    """
    if not turn.phase_task or _task_answer(turn) is not None:
        return ""

    target = _task_file(turn.phase_task)
    if target is None:
        return ""

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    return f'echo "{marker}"; cat -- "{target}" 2>&1'


def _task_file(phase_task: str) -> str | None:
    """从任务描述里找出需要在沙盒中读取的文件名"""
    match = TASK_FILE_PATTERN.search(phase_task)
    return match.group(0) if match else None


def _task_token(phase_task: str) -> str:
    """任务短标识：长任务描述只会用到开头几个可打印字符"""
    return re.sub(r"\W+", "", phase_task)[:16]


def _task_answer(turn: Turn) -> str | None:
    """从上一回合的沙盒输出中解析当前任务的答案

    输出格式约定为 "[exitCode:N]\\n<输出>"（见接口文档），因此只有执行成功
    且带有本任务标识的输出才会被当作答案，避免答非所问或复用上一个任务的结果。
    """
    if not turn.phase_task:
        return None

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    result = turn.last_cmd_result
    if marker not in result or "[exitCode:0]" not in result:
        return None

    answer = result.split(marker, 1)[1].strip()
    return answer or None


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

    取基地 2x2 占地周围距离为1且可通行的格子，按上/左/下/右四个方位各取
    一个代表点，再优先选择朝向地图内侧的三个方位（朝向内侧意味着有更大
    的来敌空间），使三座武器覆盖不同方向而不挤在基地同一侧。
    分别对应加特林、电磁狙击炮、火箭发射台。
    """
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 基地footprint周围距离为1的格子，按方位分组
    groups: dict[str, list[Pos]] = {side: [] for side in TOWER_SIDES}
    for cell in footprint:
        for neighbor in get_neighbors(cell):
            if neighbor in footprint or not turn.land(neighbor):
                continue
            groups[_side_of(neighbor, xmin, xmax, ymax)].append(neighbor)

    # 每个方位取最居中的一个格子作为代表，再按“朝向地图内侧”的程度排序
    sites = [
        (side, _side_representative(side, cells))
        for side, cells in groups.items()
        if cells
    ]
    sites.sort(key=lambda item: (
        -_side_room(item[0], turn, xmin, xmax, ymin, ymax),
        TOWER_SIDES.index(item[0]),
    ))

    # 取前3个位置
    return tuple(pos for _, pos in sites[:3])


def _side_of(pos: Pos, xmin: int, xmax: int, ymax: int) -> str:
    """判断外围格子位于基地的哪一侧（上边=ymax+1，下边=ymin-1）"""
    if pos.x < xmin:
        return "left"
    if pos.x > xmax:
        return "right"
    if pos.y > ymax:
        return "up"
    return "down"


def _side_representative(side: str, cells: list[Pos]) -> Pos:
    """取某个方位上最居中的候选格子，避免武器全部偏向一侧的角落"""
    if side in ("left", "right"):
        middle = (min(pos.y for pos in cells) + max(pos.y for pos in cells)) / 2
        return min(cells, key=lambda pos: (abs(pos.y - middle), pos.x, pos.y))

    middle = (min(pos.x for pos in cells) + max(pos.x for pos in cells)) / 2
    return min(cells, key=lambda pos: (abs(pos.x - middle), pos.x, pos.y))


def _side_room(
    side: str,
    turn: Turn,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
) -> int:
    """方位朝向地图内侧的空间：空间越大越可能迎来敌人"""
    if side == "left":
        return xmin
    if side == "right":
        return turn.width - 1 - xmax
    if side == "up":
        return turn.height - 1 - ymax
    return ymin


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
