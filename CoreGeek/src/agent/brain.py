"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。

策略概览：
    白天：工人优先建造武器工事（加特林/电磁狙击炮/火箭发射台），
          再采集石头建造围墙（先封敌方来路那一侧）；
          围墙建完后用富余资源换取金币和武器升级。
          开拓者优先完成自进化类任务（任务点领取 + 沙盒作答，
          描述里没给文件名时先探测沙盒任务目录），
          无任务时跟随武器塔；天黑前工人回防到武器旁，
          但火力/围墙不达标时先抢建，角色不会整回合空转。
    夜晚：每个角色操控一座武器攻击机器人，优先攻击威胁最高的目标。

本模块为无状态决策：每回合从 `Turn` 重新解析地图与单位状态，
不依赖任何跨回合缓存，可自动适应矿区刷新、单位移动与视野变化。
任务答案同理，直接从上一回合的沙盒输出（`lastCmdResult`）中解析；
LLM 建议也从请求里的 `llmResp` 现解析成有界计划（`_llm_plan`），
建议与指令出自同一套决策函数。
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from .grid import next_step, get_neighbors, cells_in_range
from .protocol import (
    Turn,
    Unit,
    PlayerTask,
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
SELL_BATCH = 10  # 卖给小贩的矿石批次大小
DUSK_ROUNDS = 5  # 天黑前提前回防的回合数
MIN_TOWERS_BEFORE_NIGHT = 2  # 入夜前的最低火力：不足时优先抢建而不是回防待命
# 入夜前的最低围墙段数：复盘里首夜防线只有一座光塔、零段围墙，这里要求
# 临天黑时再抢铺一段（手里有石材才抢建，没石材仍然按原策略回防）
MIN_WALLS_BEFORE_NIGHT = 2
# 防守方每天至少要保证铺好的围墙段数：复盘里防守方整天零围墙、正面毫无阻挡，
# 机器人直接贴脸打基地，而同一局的进攻方反倒把来路封得严严实实。
# LLM 计划把墙压到 0 时防守方仍按下限留出石材（见 `_wall_target`）。
DEFENDER_WALL_QUOTA = 1
WEAPON_UPGRADE_VOUCHER = "WeaponUpgradeVoucher1"  # 武器升级券（level1->level2）
UPGRADE_GOLD = 100  # 购买一张武器升级券所需金币
# 金币闲置熔断线：手里攥着够再建两座塔的金币时，不允许再把塔数配额压到满编
# 以下（复盘里"金币连续多回合冻结在 50，无塔无墙无升级"就是这么来的）。
# 一座塔 25 金换 10 点火力和一段射程，比攒到 100 金升一级划算得多，
# 所以金币越积越多时优先把它变成塔，而不是留在手里。
GOLD_FLUSH_TOWERS = WEAPON_BUILD_COST * 2

# 可卖给小贩的矿石（按优先级排序，石矿既是围墙材料也是主要收入来源）
SELLABLE_MINES = (STONE_MINE, IRON_MINE, COPPER_MINE)
# 负责"矿石换金币"的工人的采集顺序：铁/铜是纯收入来源，石材只作兜底
ECONOMY_MINE_ORDER = (IRON_MINE, COPPER_MINE, STONE_MINE)

# 任务点排序权重：报文缺少 timeoutRounds 时用最大值，不抢占"临期优先"
TASK_TIMEOUT_UNKNOWN = 10 ** 9
# 任务冷却剩余回合不超过该值时，开拓者提前到任务点旁待命
TASK_WAIT_ROUNDS = 3

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
# 沙盒输出中的答案结束标记：它之后的诊断信息（目录列表等）永远不会被当成答案
TASK_END_MARKER = "[TASK_END]"
# 沙盒探测标记：任务描述里没给文件名时先探一次任务目录，这个标记下的输出
# 只是目录/文件名清单，`_task_answer` 永远不会把它当成答案提交（见 `_sandbox_probe`）
TASK_PROBE_MARKER = "[TASK_PROBE]"
# 沙盒里自进化任务的目录约定：描述没给文件名时到这些目录里找任务文件
TASK_PROBE_DIRS = ("/tmp/selfEvolutionTask", "/tmp/selfEvolution")

# 是否提交LLM策略咨询prompt（可用环境变量 LLM_PROMPT=0 关闭）
# 接口文档：每队每个游戏日的 LLM 调用额度为 3 次（自进化任务期间不计入），
# 额度在当日首回合重置，所以每个游戏日最多咨询 LLM_PROMPT_PER_DAY 次
LLM_PROMPT_ENABLED = os.getenv("LLM_PROMPT", "1") != "0"
LLM_PROMPT_PER_DAY = 3  # 每个游戏日用满的咨询次数（接口文档的每日额度）

# LLM 建议里可以被决策层执行的部分：只开放有限几个"旋钮"，让建议和指令
# 出自同一套决策函数，而不是各说各话（复盘里的"LLM建议与指令脱节"）。
LLM_PLAN_LINE = re.compile(r"PLAN\s*[:：]\s*(?P<body>[^\r\n]*)", re.IGNORECASE)
LLM_PLAN_ITEM = re.compile(r"([A-Za-z_]+)\s*=\s*([A-Za-z0-9]+)")
# 提示词里的占位写法用尖括号：LLM 照抄模板时不会被解析成"tower=0"这类误读
LLM_PLAN_TEMPLATE = (
    "PLAN: tower=<1-3> wall=<0-3> upgrade=<on|off>"
    " defend=<left|right|up|down>"
)
LLM_PLAN_TRUE = ("on", "true", "yes", "1")
LLM_PLAN_FALSE = ("off", "false", "no", "0")
# 计划里塔数的下限：误读成 0 会让白天完全不设防，宁可保守也不接受
LLM_MIN_TOWERS = 1
LLM_MAX_TOWERS = len(TOWER_LOADOUT)
LLM_MAX_WALLS = 3


@dataclass(frozen=True, slots=True)
class LlmPlan:
    """LLM 建议中被采纳的部分（全部有界，且都有保守默认值）

    字段:
        tower: 白天要保证建成的武器塔数量（LLM_MIN_TOWERS..LLM_MAX_TOWERS）
        wall: 优先铺好的围墙段数（0..LLM_MAX_WALLS），0 表示没有配额
        upgrade: 是否允许白天花金币买武器升级券
        defend: 优先布防的方位（up/left/down/right），None 表示按默认顺序

    默认值等价于"完全按客户端原策略执行"，所以 LLM 不回复 PLAN 行、
    回复里字段缺失或值越界时，策略与改造前完全一致。
    """

    tower: int = LLM_MAX_TOWERS
    wall: int = 0
    upgrade: bool = True
    defend: str | None = None


LLM_PLAN_DEFAULT = LlmPlan()

# 沙盒输出中的错误特征：命中说明任务文件没读到，不能当作答案提交
TASK_ERROR_MARKERS = ("No such file", "Permission denied", "Is a directory")

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
    # 上一回合的LLM建议解析成有界计划，和指令生成器共用（解析不出来时是默认计划）
    plan = _llm_plan(payload)
    commands: dict[int, dict[str, Any]] = {}

    if turn.is_day:
        _decide_day(turn, commands, plan)
    else:
        _decide_night(turn, commands)

    prompt = _generate_strategy_prompt(turn, payload, plan)

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


def _decide_day(
    turn: Turn,
    commands: dict[int, dict[str, Any]],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> None:
    """白天策略: 建造、采集、任务

    所有分支跑完后还有一个兜底（`_idle_gather`）：白天还没有任何指令的角色
    就近采一铲矿，保证不会整回合零动作。

    参数:
        turn: 当前回合信息
        commands: 指令输出字典（角色ID -> 指令）
        plan: 本回合的LLM计划（默认计划等价于原有策略）
    """
    # 计算需要建造的位置（塔位排序会参考敌我相对位置与LLM指定的布防方位）
    tower_sites = _calc_tower_sites(turn, plan.defend)
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

    # 天黑前留出回防时间，避免夜晚武器无人操控而空转；
    # 但防线没达标时（火力不足或围墙不足两段）先抢建：多一座塔、多一段墙
    # 比多一个站在武器旁待命的角色更能提升夜晚防御
    dusk = _rounds_to_night(turn) <= DUSK_ROUNDS
    must_build = dusk and _needs_last_build(turn, free_towers, free_walls)

    # 为每个工人分配任务
    for worker in turn.workers():
        if dusk and not must_build:
            _fall_back_to_weapons(turn, worker, claimed, commands)
            continue
        _worker_day_logic(
            turn, worker, tower_sites, free_towers, free_walls, claimed,
            commands, plan,
        )

    # 开拓者行为（任务、宝藏）
    for pioneer in turn.pioneers():
        _pioneer_day_logic(
            turn, pioneer, tower_sites, wall_order, claimed, commands,
        )

    # 兜底：走到这里还没有任何指令的角色就近采一铲矿，保证白天不会整回合
    # 零动作（复盘里的"三个单位原地小步挪动、金币冻结"）。黄昏回防与任务
    # 待命是有意为之的"原地不动"，不在此列（见 `_idle_gather`）。
    if not dusk:
        for unit in turn.controllable():
            _idle_gather(turn, unit, claimed, commands)


def _idle_gather(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """兜底：没有任何目标的角色就近采一铲矿

    决策的各条分支都会尽量给角色安排动作，但任务点够不到、武器还没建成、
    地图上一座矿都采不了时，角色会整回合没有任何指令（复盘里的"角色原地
    挪位、金币连续多回合冻结"）。这里做最后一道兜底——按 石→铁→铜 就近
    采集，采不到就不下指令，交给下一回合重新判断。

    不打扰的情况:
        - 本回合已经有指令的角色（决策层已经给了更优先的动作）
        - 任务进行中的开拓者：任务要求它留在任务点周围一格内，
          任何移动都可能让任务强制结束
        - 黄昏（调用方不调用）：回防到武器旁待命比多采一铲矿更重要，
          武器要有角色操控才会开火

    参数:
        turn: 当前回合信息
        unit: 待兜底的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    if unit.unit_id in commands:
        return
    if unit.kind == PIONEER and turn.phase_task:
        return
    for mine_type in SELLABLE_MINES:
        if _go_mine(turn, unit, mine_type, claimed, commands):
            return


def _worker_day_logic(
    turn: Turn,
    worker: Unit,
    tower_sites: tuple[Pos, ...],
    towers_missing: list[Pos],
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> None:
    """工人白天逻辑

    优先级: 建造武器工事 > 换金币/升级武器(经济分工) > 采集石头 > 建造围墙
            > 围墙建完后: 武器升级 > 卖矿换金币 > 采集任意矿石

    任何分支最后都会落到"采集/交易"上，保证工人每回合都有产出，
    不会出现整回合没有任何指令的空转。

    建造位一旦认领（`claimed`）就归该工人：多个工人会分头去建不同的塔/
    围墙段，而不是几个人同时奔着同一个位置去，白走一趟还互相挡路。

    `plan` 是LLM建议落下来的有界计划（见 `_llm_plan`）：今天要保证几座塔、
    先铺几段围墙、能不能买升级券。默认计划与改造前的行为完全一致；计划里的
    塔数还要再经 `_tower_target` 做一次金币闲置熔断，金币富余时不会被压低，
    围墙段数同理要走一遍 `_wall_target`（防守方有下限）。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        tower_sites: 武器工事的规划位置（按建造顺序）
        towers_missing: 尚未建造的武器工事位置
        walls_missing: 尚未建造的围墙位置
        claimed: 已被其他角色占用的目标集合，用于避免多个角色争抢同一格
        commands: 指令输出字典（角色ID -> 指令）
        plan: 本回合的LLM计划
    """
    economy = _is_economy_worker(turn, worker)
    # 围墙配额还没铺满时先施工、不急着变现，对应复盘建议的
    # "先补第 2 座炮塔，再沿进攻路径铺 2 段围墙"（防守方还有下限，见 `_wall_target`）
    wall_quota = len(turn.walls()) < _wall_target(turn, plan)

    # 优先建造武器（塔数由 `_tower_target` 决定：计划配额 + 金币闲置熔断）
    if (
        towers_missing
        and len(turn.weapons()) < _tower_target(turn, plan)
        and turn.gold >= WEAPON_BUILD_COST
    ):
        # 就近认领: 每个工人挑离自己最近的那座塔，两个工人自然分头开工，
        # 而不是都盯着建造顺序表里的第一座（都挤过去的结果是另一座塔整局没人管）
        picks = sorted(
            (
                (distance(worker.pos, site), index, site)
                for index, site in enumerate(tower_sites)
                if site in towers_missing and site not in claimed
            ),
            key=lambda pick: pick[:2],
        )
        picks = _retry_sites(turn, worker, picks)
        if picks:
            for _, index, site in picks:
                weapon_type = TOWER_LOADOUT[index % len(TOWER_LOADOUT)]
                if _build_or_walk(turn, worker, site, weapon_type, claimed, commands):
                    # 认领建造位: 其他工人改去下一座塔,不会几个人挤在同一个位置上
                    claimed.add(site)
                    return
            # 所有塔位这一回合都走不通（无路可走或被抢占）时不再空手过回合，
            # 而是继续往下走：有石头就建围墙，否则去采集。复盘里工人连续多回合
            # 只下 move、金币零增长，就是这里直接返回造成的。

    # 把富余资源换成战力（武器升级 > 卖矿换金币）：
    # 围墙建完时人人有责；围墙没建完时由分工里的"经济工人"负责，
    # 否则要等近二十段围墙全部铺完才会花钱，金币会闲置一整天
    # （围墙配额还没铺满时先铺墙，金币留到围墙立起来再花）
    if (not walls_missing or economy) and not wall_quota:
        if _upgrade_weapon_with_gold(
            turn, worker, claimed, commands, allow=plan.upgrade,
        ):
            return
        if _trade_logic(turn, worker, claimed, commands):
            return
        if not walls_missing:
            # 手里还没有可卖的矿石: 继续采集,攒够一批再换金币
            _gather_logic(turn, worker, claimed, commands)
            return

    # 检查背包里的石头数量
    stones = worker.backpack.count(WALL_MATERIAL)

    # 如果旁边有矿且石头不足,采集
    # （负责矿石变现的工人跳过这一步：否则它会一直就地采石，
    #   永远轮不到铁/铜，矿种分工就落空了；但围墙配额没铺满时全员先采石）
    if _mine_order(turn, worker, prefer_stone=wall_quota)[0] == STONE_MINE:
        mine = _adjacent_mine(turn, worker, STONE_MINE)
        if mine is not None and stones < STONE_BATCH:
            commands[worker.unit_id] = collect_command(mine)
            claimed.add(mine)
            return

    # 如果有石头,去建造围墙（位置都被其他角色占住时继续往下走,别空转）
    if stones > 0:
        sites = _retry_sites(
            turn, worker, [site for site in walls_missing if site not in claimed],
        )
        if sites:
            _build_or_walk(turn, worker, sites[0], WALL, claimed, commands)
            return

    # 没石头(或暂时没位置建): 就近采矿; 采不到就把背包里的矿石卖掉腾地方
    if _gather_logic(turn, worker, claimed, commands, prefer_stone=wall_quota):
        return
    _trade_logic(turn, worker, claimed, commands)


def _retry_sites(
    turn: Turn,
    unit: Unit,
    candidates: list[Any],
) -> list[Any]:
    """上一回合的动作失败时跳过第一个候选,换一处重试

    任务书4.5.4节的碰撞规则下,移动/建造会因为"目标点被夺取"而失败;
    下一回合原地重复同一条指令往往还是失败,换一个建造位重试才能把建造
    推进下去。

    参数:
        turn: 当前回合信息
        unit: 当前决策的单位
        candidates: 按优先级排序的候选建造位

    返回:
        本回合实际可用的候选列表；上一回合没失败时原样返回
    """
    if len(candidates) > 1 and turn.action_failed(unit.unit_id):
        return candidates[1:]
    return candidates


def _is_economy_worker(turn: Turn, worker: Unit) -> bool:
    """该工人是否负责"矿石换金币"这条经济线

    分工按角色ID顺序静态划定，不依赖任何跨回合缓存：两名工人时第1名管石材
    与围墙、第2名管矿石变现，于是两名工人不会一起挤在同一个矿点上，经济也
    总有人推进；只剩一名工人（夜里阵亡后常见）时它兼顾两件事，否则金币会
    一直闲置到围墙圈建完。

    参数:
        turn: 当前回合信息
        worker: 待判断的工人

    返回:
        True 表示该工人负责卖矿换金币与武器升级
    """
    workers = turn.workers()
    if len(workers) <= 1:
        return True
    return worker.unit_id != workers[0].unit_id


def _mine_order(
    turn: Turn,
    worker: Unit,
    *,
    prefer_stone: bool = False,
) -> tuple[str, ...]:
    """该工人本回合的采集矿种顺序

    "矿种互补"只在还有另一名工人兜底采石材时成立：同一时刻最多一名工人去
    采铁/铜换金币，其余人继续采石材保证围墙不停工；只剩一名工人时它必须
    石材优先（围墙是防守的根本），所以退回默认顺序。

    参数:
        turn: 当前回合信息
        worker: 待判断的工人
        prefer_stone: 为 True 时全员石材优先（围墙配额还没铺满时用）

    返回:
        按优先级排序的矿种元组
    """
    if (
        prefer_stone
        or len(turn.workers()) < 2
        or not _is_economy_worker(turn, worker)
    ):
        return SELLABLE_MINES
    return ECONOMY_MINE_ORDER


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
        - 任务点2占据两格，站在任意一格旁边都算"在任务点周围一格内"
          （判定见 `_task_distance`）
        - 任务结束后需要等待冷却，冷却期内 isValid 为 false
        - 自进化类任务需在沙盒中取数后作答，答案经 `submitAnswer` 提交

    任务点的选择是"临期优先、其次就近"：单个任务只有 15 回合时限，
    先去快过期的那个才能把两个任务的分数都拿到手。两个任务点会按这个
    顺序依次尝试，最优先的那个走不通时退而先去下一个，不会整局放弃任务。

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
            if _task_distance(turn, pioneer.pos, task_pos) > 1:
                target = _nearest_task_cell(turn, pioneer.pos, task_pos)
                step = _step_toward(turn, pioneer, target, claimed)
                if step is not None:
                    commands[pioneer.unit_id] = move_command(step)
                return
            # 沙盒命令的输出上一回合才返回，这里按任务标识取出本任务的答案
            answer = _task_answer(turn)
            if answer is not None:
                commands[pioneer.unit_id] = submit_answer_command(answer)
            return

    # 2. 有可接取的任务: 按优先级依次尝试领取
    # 之前的实现只试最优先的那个任务点：那一格被挡住/绕不过去时开拓者就整回合
    # 放弃任务、跑去跟随武器塔，两个任务点（合计160分+160金币）都会白白过期。
    valid_tasks = [task for task in turn.player_tasks if task.is_valid]
    if valid_tasks:
        ordered = sorted(valid_tasks, key=lambda task: (
            task.timeout_rounds if task.timeout_rounds > 0 else TASK_TIMEOUT_UNKNOWN,
            distance(pioneer.pos, task.task_position),
            task.task_position.x,
            task.task_position.y,
        ))
        for task in ordered:
            if _head_to_task(turn, pioneer, task.task_position, claimed, commands):
                return

    # 3. 任务点都在冷却中: 冷却快结束时提前到任务点旁待命，任务一开放就能接
    elif _rounds_until_task(turn) <= TASK_WAIT_ROUNDS:
        task_pos = _nearest_task_position(turn, pioneer.pos)
        if task_pos is not None and _head_to_task(
            turn, pioneer, task_pos, claimed, commands,
        ):
            return

    # 4. 没有可接取的任务: 跟随武器塔,为夜晚操控武器做准备
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


def _head_to_task(
    turn: Turn,
    pioneer: Unit,
    task_pos: Pos,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """走向任务点，到了就领取任务

    到达判定用任务点的全部格子（见 `_task_distance`）：任务点2占据两格，
    开拓者站在另一格旁边同样可以领取，不必再绕到报文给出的那一格。

    返回:
        True 表示本回合已下达指令（领取或移动）
    """
    if _task_distance(turn, pioneer.pos, task_pos) <= 1:
        commands[pioneer.unit_id] = accept_task_command()
        return True

    step = _step_toward(
        turn, pioneer, _nearest_task_cell(turn, pioneer.pos, task_pos), claimed,
    )
    if step is None:
        return False

    commands[pioneer.unit_id] = move_command(step)
    return True


def _task_point_cells(turn: Turn) -> tuple[Pos, ...]:
    """己方任务点在地图上占据的全部格子

    任务点2占据两格（任务书4.6.2节），报文与地图里都可能只给出其中一格，
    因此把 playerTasks 的坐标与地图上本阵营任务点的坐标合并去重。
    """
    kinds = _TASK_POINTS_BY_TEAM.get(turn.team_type, ())
    cells = [task.task_position for task in turn.player_tasks]
    cells += [pos for pos, kind in turn.zones.items() if kind in kinds]
    return tuple(dict.fromkeys(cells))


def _task_cells_of(turn: Turn, task_pos: Pos) -> tuple[Pos, ...]:
    """某个任务点占据的格子：报文给出的那一格 + 与它相邻的同阵营任务点格子"""
    return (task_pos,) + tuple(
        cell for cell in _task_point_cells(turn)
        if cell != task_pos and distance(cell, task_pos) <= 1
    )


def _task_distance(turn: Turn, origin: Pos, task_pos: Pos) -> int:
    """origin 到该任务点的距离（到它的任意一格）"""
    return min(distance(origin, cell) for cell in _task_cells_of(turn, task_pos))


def _nearest_task_cell(turn: Turn, origin: Pos, task_pos: Pos) -> Pos:
    """任务点里离 origin 最近的一格（两格任务点走最近的那格）"""
    return min(
        _task_cells_of(turn, task_pos),
        key=lambda cell: (distance(origin, cell), cell.x, cell.y),
    )


def _rounds_until_task(turn: Turn) -> int:
    """己方任务点里最早可以再接任务的剩余冷却回合数

    任务点数据缺失（报文没有 playerTasks）时返回 0：此时按地图上的任务点
    直接前往，到点后下一回合再领取，避免整局都不去任务点。
    """
    if not turn.player_tasks:
        return 0
    return min(task.cold_down_rounds for task in turn.player_tasks)


def _trade_logic(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """资源交易逻辑：把背包里多余的矿石卖给小贩换金币

    小贩收购价随世界新闻波动（任务书4.6.1节），卖出所得可用于购买升级券。
    背包还有空间时攒够一批再卖；背包已经满了就先卖掉手头最多的那种矿腾地方，
    既换到金币又避免工人因为塞满背包而无法采集。小贩离得太远、跑一趟回不来
    时不出门，先就近采集，等靠近了再卖（见 `_can_return_before_dusk`）。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（贩卖或走向小贩）
    """
    quantities = [
        (mine_type, worker.backpack.count(mine_type))
        for mine_type in SELLABLE_MINES
    ]
    # 数量最多的那种优先卖；数量相同时按 SELLABLE_MINES 的顺序取石材
    mine_type, amount = max(
        quantities,
        key=lambda item: (item[1], -SELLABLE_MINES.index(item[0])),
    )
    # 背包满了就卖一批腾地方,否则等攒够一批再卖
    if amount < (1 if worker.backpack_full else SELL_BATCH):
        return False

    vendor = _nearest_zone(turn, VENDOR, worker.pos)
    if vendor is None:
        return False

    # 已在小贩旁边: 直接贩卖
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(mine_type, amount)
        return True

    # 否则走向小贩（路太远、天黑前回不来时不出这趟门）
    if not _can_return_before_dusk(turn, worker, vendor):
        return False
    step = _step_toward(turn, worker, vendor, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True

    return False


def _upgrade_weapon_with_gold(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow: bool = True,
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
        allow: 是否允许花金币买券（LLM计划里"今天不买"时为 False，
               已经买好的券仍然照用不误）

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
        if not _can_return_before_dusk(turn, worker, weapon.pos):
            return False
        step = _step_toward(turn, worker, weapon.pos, claimed)
        if step is not None:
            commands[worker.unit_id] = move_command(step)
            return True
        return False

    # 2. 金币足够且计划允许: 去武器商店购买
    if not allow or turn.gold < UPGRADE_GOLD or worker.backpack_full:
        return False

    shop = _nearest_zone(turn, WEAPON_SHOP, worker.pos)
    if shop is None:
        return False

    if distance(worker.pos, shop) <= 1:
        commands[worker.unit_id] = buy_command(WEAPON_UPGRADE_VOUCHER)
        return True

    if not _can_return_before_dusk(turn, worker, shop):
        return False
    step = _step_toward(turn, worker, shop, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True
    return False


def _rounds_to_night(turn: Turn) -> int:
    """距离天黑还剩多少回合（含当前回合）"""
    day_round = (turn.round_no - 1) % ROUNDS_PER_DAY
    return DAY_ROUNDS - day_round


def _can_return_before_dusk(turn: Turn, unit: Unit, target: Pos) -> bool:
    """去 target 办完事，还来得及在天黑前回到基地吗

    采集和建造都在基地旁边，来回一两回合就够；卖矿、买升级券却要跑到地图
    另一头，跑远了回不来就会让夜晚的武器没人操控（任务书4.4节：武器要有
    角色操控才会开火）。这里按"往返路费 + 提前回防的回合数"做个粗算，
    路太远就不出这趟门。

    参数:
        turn: 当前回合信息
        unit: 准备出门的单位
        target: 目的地坐标

    返回:
        True 表示这一趟来回之后还剩回防时间
    """
    return _rounds_to_night(turn) > 2 * distance(unit.pos, target) + DUSK_ROUNDS


def _needs_last_build(
    turn: Turn,
    towers_missing: list[Pos],
    walls_missing: list[Pos],
) -> bool:
    """天黑前是否还要抢建（入夜前的火力/防线预算检查）

    只在天黑前的最后几个回合使用。火力不够又买得起塔时先补塔，手里有石头
    却还没铺够 `MIN_WALLS_BEFORE_NIGHT` 段围墙时先补墙——首夜只有一座光塔、
    零段围墙时防线没有纵深（复盘里的"首夜崩盘"）。这两件事都在基地旁边
    完成，做完再回防也来得及，比整队提前回防更划算。

    参数:
        turn: 当前回合信息
        towers_missing: 尚未建造（且未被占据）的武器位置
        walls_missing: 尚未建造（且未被占据）的围墙位置

    返回:
        True 表示本回合应当继续建造而不是回防
    """
    if (
        towers_missing
        and len(turn.weapons()) < MIN_TOWERS_BEFORE_NIGHT
        and turn.gold >= WEAPON_BUILD_COST
    ):
        return True

    if walls_missing and len(turn.walls()) < MIN_WALLS_BEFORE_NIGHT:
        return any(WALL_MATERIAL in worker.backpack for worker in turn.workers())

    return False


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

    一条命令里尽量多拿信息（复盘里敌方逐次试错 401→400，白丢好几个回合）：
    描述里的路径读不到时，按文件名在沙盒里再找一次；末尾附上目录列表作为
    诊断线索。答案由 `TASK_END_MARKER` 界定，诊断信息不会被当成答案提交。

    描述里连文件名都没有时（"请按沙盒里的任务说明作答"这类），先按
    `_sandbox_probe` 探一次沙盒任务目录，下一回合从探测结果里认出文件名
    再走上面的读文件流程——复盘里"任务卡在任务点反复答非所问、整个任务
    周期空转"就是从"不知道该读哪个文件"开始的。
    """
    if not turn.phase_task or _task_answer(turn) is not None:
        return ""

    # 描述里没给文件名时，用上一回合的探测结果找；还没探过就先探一次
    target = _task_file(turn.phase_task) or _task_file(turn.last_cmd_result)
    if target is None:
        return _sandbox_probe(turn)

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    # 描述里给的可能是带目录的路径，兜底搜索只按文件名找
    base = target.replace("\\", "/").rsplit("/", 1)[-1]
    read = (
        f'cat -- "{target}" 2>/dev/null'
        f' || find . -maxdepth 3 -type f -name "{base}" -exec cat -- {{}} + 2>&1'
    )
    scan = "ls -a -- . 2>&1 | head -40"
    return f'echo "{marker}"; {read}; echo "{TASK_END_MARKER}"; {scan}'


def _sandbox_probe(turn: Turn) -> str:
    """任务描述里找不到文件名时，探测沙盒里的任务文件

    列出沙盒里存放任务说明的目录（`TASK_PROBE_DIRS`），并把目录下
    文件名像任务文件的那些连同完整路径一起打印出来，下一回合
    `_sandbox_command` 就能从输出里认出该读哪个文件。

    探测输出带 `TASK_PROBE_MARKER`：`_task_answer` 只认 `TASK_MARKER`，
    所以目录列表永远不会被当成答案提交；`find` 排在 `ls` 前面是为了先拿到
    完整路径（`cat` 才找得到文件），`ls` 只作诊断线索。

    参数:
        turn: 当前回合信息

    返回:
        需要提交给沙盒执行的探测命令
    """
    token = _task_token(turn.phase_task)
    dirs = " ".join(f'"{path}"' for path in TASK_PROBE_DIRS)
    # 先找任务文件本身（task*），再退到 spec/说明文档：只按文件名找，
    # 免得把沙盒里的其他文档当成任务文件读回来
    found = "; ".join(
        f'find "{path}" -maxdepth 2 -type f -name "task*" 2>/dev/null;'
        f' find "{path}" -maxdepth 2 -type f'
        ' \\( -name "spec*" -o -name "*.md" \\) 2>/dev/null'
        for path in TASK_PROBE_DIRS
    )
    return (
        f'echo "{TASK_PROBE_MARKER}{token}"; '
        f"( {found} ) | head -40; "
        f"ls -a -- {dirs} 2>&1 | head -40; "
        f'echo "{TASK_END_MARKER}"'
    )


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
    答案取任务标识到 `TASK_END_MARKER` 之间的内容，命令末尾的诊断信息（目录
    列表等）因此不会被误当成答案提交。
    命中错误特征的输出（文件不存在等）同样不能提交：错误答案既拿不到分，
    又白白消耗任务冷却，所以宁可这一回合不提交，等下一条沙盒输出。
    """
    if not turn.phase_task:
        return None

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    result = turn.last_cmd_result
    if marker not in result or "[exitCode:0]" not in result:
        return None

    answer = result.split(marker, 1)[1].split(TASK_END_MARKER, 1)[0].strip()
    if not answer or any(bad in answer for bad in TASK_ERROR_MARKERS):
        return None
    return answer


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


def _gather_logic(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    prefer_stone: bool = False,
) -> bool:
    """保证工人每回合都有产出：按矿种分工就近采集

    石材是围墙材料，优先采；两名工人时按"矿种互补"分工（见 `_mine_order`），
    一名采石、一名采铁/铜换金币，不会一起挤在同一个矿点上；分工里负责的矿种
    附近没有（或已被其他角色占住）时退回其他矿种，保证不会整回合没有指令。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        prefer_stone: 为 True 时全员石材优先（围墙配额还没铺满时用）

    返回:
        True 表示本回合已下达采集或移动指令
    """
    for mine_type in _mine_order(turn, worker, prefer_stone=prefer_stone):
        if _go_mine(turn, worker, mine_type, claimed, commands):
            return True
    return False


def _build_or_walk(
    turn: Turn,
    unit: Unit,
    target: Pos,
    building_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """如果在建造位置旁边则建造,否则向目标移动

    返回:
        True 表示本回合已下达指令（建造或移动）；False 表示无路可走，
        调用方应把建造位留给其他角色而不是空占着
    """
    # 已经在目标旁边(但不是目标本身),执行建造
    if unit.pos != target and distance(unit.pos, target) <= 1:
        commands[unit.unit_id] = build_command(target, building_type)
        claimed.add(target)
        return True

    # 否则向目标移动
    step = _step_toward(turn, unit, target, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)
        return True
    return False


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
        3. 白天再过滤掉武器塔/围墙的建造点（角色占住建造位会让建筑永远建不起来）；
           目标本身就是建造点时（走去施工）不过滤，否则角色会因为"相邻格全是
           建造点"而无处落脚，反而建不起来
        4. inside_only 为 True 时进一步限制在基地周围
        5. 按离基地的切比雪夫距离排序
    """
    station = turn.station()
    footprint = station_footprint(station.pos) if station else ()
    blocked = turn.blocked(unit)

    # 目标的八方向相邻格子
    neighbors = get_neighbors(target)

    # 白天避开建造点: 角色站上去会把这一格占住,武器/围墙就再也建不起来了
    reserved: frozenset[Pos] = frozenset()
    if turn.is_day:
        build_sites = _reserved_build_sites(turn)
        if target not in build_sites:
            reserved = build_sites

    cells = [
        pos for pos in neighbors
        if turn.land(pos)
        and pos not in blocked
        and pos not in reserved
        and (pos == unit.pos or pos not in claimed)
        and (
            not inside_only
            or _footprint_distance(pos, footprint) <= 1
        )
    ]

    # 按离基地的距离排序（优先靠近基地）
    cells.sort(key=lambda pos: (_footprint_distance(pos, footprint), pos.x, pos.y))
    return cells


def _reserved_build_sites(turn: Turn) -> frozenset[Pos]:
    """本回合规划中的建造点（武器塔 + 围墙）

    这些格子要留给施工：一旦被角色占住，建造点就会被判为"已占用"而从
    待建列表里消失，对应的塔或围墙整局都建不起来。
    """
    return frozenset(_calc_tower_sites(turn)) | frozenset(_calc_wall_order(turn))


def _footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    """计算点到footprint的最小距离"""
    if not footprint:
        return 0
    return min(distance(pos, cell) for cell in footprint)


def _calc_tower_sites(
    turn: Turn,
    preferred: str | None = None,
) -> tuple[Pos, ...]:
    """计算武器塔建造位置（基地周围一圈）

    取基地 2x2 占地周围距离为1且可通行的格子，按上/左/下/右四个方位各取
    一个代表点，再按"先朝敌方来路、后朝地图内侧"的顺序取前三个，使三座
    武器覆盖不同方向而不挤在基地同一侧，并优先罩住敌人来的那一侧
    （复盘指出塔位只按地图空间选，没参考敌方来路）。
    分别对应加特林、电磁狙击炮、火箭发射台。

    参数:
        turn: 当前回合信息
        preferred: LLM 计划指定的布防方位（up/left/down/right），None 表示自动

    返回:
        按建造顺序排列的武器塔位置（最多3个）
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

    # 每个方位取最居中的一个格子作为代表，再按"敌人在哪边"排序
    enemy_sides = _enemy_sides(turn)
    sites = [
        (side, _side_representative(side, cells))
        for side, cells in groups.items()
        if cells
    ]
    sites.sort(key=lambda item: (
        _side_priority(item[0], enemy_sides, preferred),
        -_side_room(item[0], turn, xmin, xmax, ymin, ymax),
        TOWER_SIDES.index(item[0]),
    ))

    # 取前3个位置
    return tuple(pos for _, pos in sites[:3])


def _enemy_sides(turn: Turn) -> frozenset[str]:
    """敌方主力大致来自基地的哪几个方位

    优先用可见的敌方基地定位来路（敌方单位都是从它出发的），看不到敌方基地
    时退回最近的敌方单位。两个都看不到时返回空集合，塔位排序退回原来的
    "朝地图内侧空间"，策略与改造前完全一致。

    参数:
        turn: 当前回合信息

    返回:
        方位集合（up/left/down/right 的子集）
    """
    station = turn.station()
    if station is None:
        return frozenset()

    visible = [enemy for enemy in turn.enemies if enemy.is_alive]
    candidates = [unit for unit in visible if unit.kind == STATION] or visible
    if not candidates:
        return frozenset()

    enemy = min(
        candidates,
        key=lambda unit: (distance(unit.pos, station.pos), unit.pos.x, unit.pos.y),
    )
    dx = enemy.pos.x - station.pos.x
    dy = enemy.pos.y - station.pos.y

    sides = set()
    if dx < 0:
        sides.add("left")
    elif dx > 0:
        sides.add("right")
    if dy < 0:
        sides.add("down")
    elif dy > 0:
        sides.add("up")
    return frozenset(sides)


def _side_priority(
    side: str,
    enemy_sides: frozenset[str],
    preferred: str | None,
) -> int:
    """塔位排序的第一权重：LLM 指定 > 朝敌方来路 > 其他"""
    if preferred is not None and side == preferred:
        return 0
    if side in enemy_sides:
        return 1
    return 2


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


def _wall_side_order(turn: Turn) -> tuple[str, ...]:
    """围墙的铺设优先顺序：先封敌方来路，其余按"上 -> 左 -> 下 -> 右"

    复用塔位那套方位判定（`_enemy_sides`），让围墙与炮塔压在同一侧：来路先被
    墙压窄、再被塔罩住，机器人只能顶着火力拆墙（复盘里"防守方 0 段围墙、正面
    无任何阻挡"，以及"用墙把来路压缩进塔的射程"）。

    看不到敌方单位时各方位的优先级相同，顺序退化成"上 -> 左 -> 下 -> 右"，
    与改造前完全一致。

    参数:
        turn: 当前回合信息

    返回:
        四个方位的铺设顺序（up/left/down/right 的一个排列）
    """
    enemy_sides = _enemy_sides(turn)
    return tuple(sorted(
        TOWER_SIDES,
        key=lambda side: (
            _side_priority(side, enemy_sides, None),
            TOWER_SIDES.index(side),
        ),
    ))


def _calc_wall_order(turn: Turn) -> tuple[Pos, ...]:
    """计算围墙建造顺序（基地周围第二圈）

    按 `_wall_side_order` 给出的方位顺序（先敌方来路、其余上左下右）环绕基地
    铺一圈围墙，并在右下角留一个入口供角色进出。超出地图或落在非陆地上的点
    会被过滤掉。
    """
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 矩形四条边各自的格子（同一格不会落在两条边上）
    by_side: dict[str, list[Pos]] = {
        # 上边（从右到左）
        "up": [Pos(x, ymax + 2) for x in range(xmax + 2, xmin - 3, -1)],
        # 左边（从上到下）
        "left": [Pos(xmin - 2, y) for y in range(ymax + 1, ymin - 2, -1)],
        # 下边（从左到右）
        "down": [Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)],
        # 右边（从下到上）
        "right": [Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)],
    }
    order = [
        pos
        for side in _wall_side_order(turn)
        for pos in by_side[side]
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


def _llm_plan(payload: dict[str, Any]) -> LlmPlan:
    """把上一回合的 LLM 建议（llmResp）解析成有界的作战计划

    复盘里反复出现的"LLM建议与指令脱节"：建议内容从来没进过指令生成器。
    这里只认 prompt 约定的 `PLAN:` 行，逐键做白名单 + 范围钳制——LLM 只能在
    既有策略的旋钮里做选择，不能凭空发明动作：

        tower   今天要保证建成的塔数（钳到 LLM_MIN_TOWERS..LLM_MAX_TOWERS）
        wall    今天优先铺好的围墙段数（钳到 0..LLM_MAX_WALLS）
        upgrade 今天是否允许买武器升级券
        defend  优先布防的方位（up/left/down/right）

    解析不出来时（没写 PLAN 行、字段缺失、值越界）逐项退回默认值，默认值
    等价于改造前的策略，所以 LLM 不配合也不会让决策变差。建议里先复述了
    prompt 的占位模板、再给出真正的计划时，按"先解析出来的值优先"合并，
    模板本身（`tower=<1-3>` 这类尖括号写法）解析不出任何字段。

    计划每回合都从 `llmResp` 现解析（不落任何跨回合缓存），因此它的生效
    窗口与"报文里带着 LLM 回复"的回合一致，回复消失后自动回到默认计划；
    塔位/塔数/围墙配额这些旋钮的作用期正好是开局那几天，与回复到达的时机
    吻合。

    参数:
        payload: 判题系统原始请求（取 llmResp 字段）

    返回:
        解析后的计划；无法解析时返回 LLM_PLAN_DEFAULT
    """
    text = str(payload.get("llmResp") or "")
    # 建议里可能先复述了 prompt 的占位模板、再给出真正的计划，
    # 因此逐行扫所有 PLAN 行，按"先解析出来的值优先"合并，别被模板带偏
    items: dict[str, str] = {}
    for match in LLM_PLAN_LINE.finditer(text):
        for key, value in LLM_PLAN_ITEM.findall(match.group("body")):
            items.setdefault(key.lower(), value.lower())
    if not items:
        return LLM_PLAN_DEFAULT

    def _bounded(key: str, default: int, low: int, high: int) -> int:
        raw = items.get(key)
        if raw is None or not raw.isdigit():
            return default
        return min(max(int(raw), low), high)

    upgrade = LLM_PLAN_DEFAULT.upgrade
    if items.get("upgrade") in LLM_PLAN_TRUE:
        upgrade = True
    elif items.get("upgrade") in LLM_PLAN_FALSE:
        upgrade = False

    defend = items.get("defend")
    if defend not in TOWER_SIDES:
        defend = None

    return LlmPlan(
        tower=_bounded(
            "tower", LLM_PLAN_DEFAULT.tower, LLM_MIN_TOWERS, LLM_MAX_TOWERS,
        ),
        wall=_bounded("wall", LLM_PLAN_DEFAULT.wall, 0, LLM_MAX_WALLS),
        upgrade=upgrade,
        defend=defend,
    )


def _tower_target(turn: Turn, plan: LlmPlan) -> int:
    """本回合要保证建成的武器塔数量（含金币闲置熔断）

    正常情况下就是 LLM 计划里的 `tower`（默认满编 3 座）；但金币已经攒到
    `GOLD_FLUSH_TOWERS`（够再建两座塔）时一律提到满编：复盘里"金币 75 只花
    25、余下 50 连躺三个回合"的根因就是计划把塔数配额压低后金币再没有出口。
    金币留在手里不产生任何防御力，宁可多建一座塔。

    熔断只在金币富余时生效（阈值高于单座造价），所以 LLM 仍然可以为
    "留钱买升级券"而少建一座塔；`upgrade` 开关与围墙配额都不受影响。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        白天要保证建成的塔数（LLM_MIN_TOWERS..LLM_MAX_TOWERS）
    """
    if turn.gold >= GOLD_FLUSH_TOWERS:
        return LLM_MAX_TOWERS
    return plan.tower


def _plan_summary(turn: Turn, plan: LlmPlan) -> str:
    """把本回合的既定计划写成一句人话

    计划出自 `_calc_tower_sites`/任务排序等同一套决策函数，LLM 因此可以对
    具体数字提意见，而不是和指令生成器各说各话。塔数与围墙段数报的是
    `_tower_target`/`_wall_target`（含金币闲置熔断与防守方下限），
    所以 LLM 看到的就是执行层真正要建的座数/段数。
    """
    return (
        f"武器目标 {_tower_target(turn, plan)} 座（现有 {len(turn.weapons())} 座）；"
        f"优先铺围墙 {_wall_target(turn, plan)} 段（现有 {len(turn.walls())} 段）；"
        f"升级券 {'可买' if plan.upgrade else '今天不买'}；"
        f"布防方位 {plan.defend or _enemy_brief(turn)}"
    )


def _wall_target(turn: Turn, plan: LlmPlan) -> int:
    """本回合要保证铺好的围墙段数（防守方有下限）

    正常情况下就是 LLM 计划里的 `wall`；但防守方在此基础上至少保证
    `DEFENDER_WALL_QUOTA` 段：防守方考的是扛住进攻，正面没有围墙时机器人会
    直接贴脸打基地（复盘里"防守方围墙 0 段、正面无任何阻挡、射程只覆盖基地
    贴脸区"）。进攻方不受影响，仍然完全按 LLM 计划走（0 段就是不铺）。

    配额管的是"今天要铺几段"这个目标：配额没铺满时工人优先采石、先施工，
    矿石不急着变现、金币也留到围墙立起来再花（见 `_worker_day_logic`）。
    地图上既没有石矿、背包里也没有存货时限额自动失效——没有石材就铺不出墙，
    再扣着经济线只会让矿石卖不掉、金币闲置。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        要保证铺好的围墙段数（0..LLM_MAX_WALLS）
    """
    if turn.team_type == "defender":
        # 没有石材来源（地图上没石矿、背包里也没存货）时围墙根本无从铺起，
        # 这时不能因为"还差一段墙"把经济线扣住——矿石卖不掉就是金币闲置
        has_stone = bool(turn.stone_mines()) or any(
            WALL_MATERIAL in unit.backpack for unit in turn.workers()
        )
        if has_stone:
            return max(plan.wall, DEFENDER_WALL_QUOTA)
    return plan.wall


def _enemy_brief(turn: Turn) -> str:
    """敌我相对方位，看不到敌方单位时说明塔位是按地图空间选的"""
    sides = _enemy_sides(turn)
    if not sides:
        return "按地图内侧空间选择（当前看不到敌方单位）"
    return "敌方来路 " + "/".join(side for side in TOWER_SIDES if side in sides)


def _task_brief(turn: Turn) -> str:
    """可接任务点的坐标/奖励/剩余回合，供LLM判断值不值得去做任务"""
    valid = [task for task in turn.player_tasks if task.is_valid]
    if not valid:
        return ""
    items = [
        f"({task.task_position.x},{task.task_position.y}){task.score_reward}分"
        + (f"/剩{task.timeout_rounds}回合" if task.timeout_rounds > 0 else "")
        for task in valid[:2]
    ]
    return "：" + "；".join(items)


def _generate_strategy_prompt(
    turn: Turn,
    payload: dict[str, Any],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> str:
    """生成提交给LLM的策略咨询prompt（用满每个游戏日的调用额度）

    接口文档规定每个游戏日有LLM调用次数限制（errorCode=5：每队每个游戏日
    3 次，在当日首回合重置），因此只在每天的前 `LLM_PROMPT_PER_DAY` 个回合
    各请求一次，并带上上一回合的LLM回复作为上下文。复盘里"只在第 1 回合
    调用过 LLM，R2~R4 的 prompt 为空、指令退化为无目标移动"，只问一次
    等于把额度白白浪费掉，这里把当日额度用满。

    prompt 里一并给出"本回合既定计划"并要求最后回一行 `PLAN:`：建议因此
    能落到具体数字上，回复里的 PLAN 行由 `_llm_plan` 解析回指令生成器
    （复盘里的"策略与执行脱节"）。

    参数:
        turn: 当前回合信息
        payload: 判题系统原始请求（取 llmResp 作为上下文）
        plan: 上一回合LLM建议解析出的计划，用于说明当前策略

    返回:
        需要提交给LLM的prompt；本回合不需要咨询时返回空字符串
    """
    if not LLM_PROMPT_ENABLED:
        return ""
    # 当日前几个回合（第1、2、3回合与131、132、133回合...）
    if (turn.round_no - 1) % ROUNDS_PER_DAY >= LLM_PROMPT_PER_DAY:
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
        f"可领取任务点: {sum(1 for t in turn.player_tasks if t.is_valid)} 个"
        f"{_task_brief(turn)}",
        f"本回合既定计划: {_plan_summary(turn, plan)}",
        "请用不超过5行中文说明：优先建造或升级什么、角色如何站位、是否值得去做任务。",
        "最后一行必须输出作战计划，客户端会照它调整指令（值越界会被忽略）："
        f"{LLM_PLAN_TEMPLATE}",
    ]
    if previous:
        lines.insert(1, f"上一回合LLM建议: {previous[:500]}")

    return "\n".join(lines)
