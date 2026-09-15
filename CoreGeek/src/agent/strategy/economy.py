"""经济策略：采集、贩卖、建造执行、金币分配。

对应设计文档V2 §7.1 – §7.3。

V1 最刺眼的问题是**金币冻结**：开局 75 金三个回合全部砸进 3 座塔，之后连续
7–10 个回合 `gold=0`，背包里的石材/铜材攒着不卖（PK592172 的 R8–R16、
PK592173 的 R9–R16）。根因是采集没有专职化、卖矿没有触发条件。

V2 的做法：

- **专职分工**：`worker1` 当建造工（顺便给自己采石），`worker2` 当采集工
  （采铜/铁卖给小贩换金币）。分工固定，不会在"建造"和"采集"之间来回切。
- **卖矿兜底**：`gold == 0` 且背包里有矿石 ⇒ 无条件卖一次，哪怕只有一块石头。
  这条规则直接消灭"金币恒 0"。
- **金币预留**：花金币前先看下一优先级的门槛，不把最后一枚金币花光。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import grid
from ..protocol import (
    COPPER_MINE,
    IRON_MINE,
    MEDICINE,
    STATION_UPGRADE_VOUCHER,
    STONE_MINE,
    VENDOR,
    WALL,
    WALL_FIXER,
    WALL_MATERIAL,
    WEAPON_BUILD_COST,
    WEAPON_SHOP,
    WEAPON_UPGRADE_VOUCHER,
    Pos,
    Turn,
    Unit,
    buy_command,
    collect_command,
    distance,
    move_command,
    station_footprint,
    sell_command,
    use_command,
)
from ..world import World
from . import defense

# 采集工的矿石偏好（按小贩收购价从高到低，任务书 §4.6.1 + docs/request.txt）
MINER_ORDER: tuple[str, ...] = (COPPER_MINE, IRON_MINE, STONE_MINE)

# **地图上没有小贩时**的采集顺序
#
# 铜/铁的唯一用途就是卖给小贩换金币；没有小贩，它们就是纯负重。而墙只有石头
# 能砌，所以此时全员采石。真实对局里出现过"金矿从 R9 起恒为 0、工人背包里
# 攒着铜一直没卖、围墙全程 0 段"——光看日志判不出是哪一种，`neutral=` 里
# 有没有 `vendor` 一看便知（见 `战术参考/日志分析模板V2.md` §1.1）。
MINER_ORDER_NO_VENDOR: tuple[str, ...] = (STONE_MINE,)

# 建造工采石的目标批次：够砌几段墙就够。
# 一次多采几块能显著减少往返——真实对局里最近的石矿离基地十几格，
# 来回一趟 20+ 回合，采 3 块就回来的话大部分时间都花在路上。
STONE_BATCH = 6

# 背包占用超过这个比例就去卖一次（防止采满背包再卖，浪费回合）
BACKPACK_SELL_RATIO = 0.6

# 金币低于这个数就优先卖矿（一座塔 25 金）
GOLD_LOW = WEAPON_BUILD_COST

# 建造工随身保留的石材下限（超出需求的部分可以卖掉换金币）
STONE_RESERVE = 1

# 生命药剂的价格（任务书 §4.6.3）
MEDICINE_PRICE = 10

# 残血墙少于这个段数就不值得跑一趟修复包（10 金 + 一个回合）
REPAIR_MIN_DAMAGED_WALLS = 3


# 固定分工（见模块文档：分工不切换是"采集中断 9 回合"的对策）
ROLE_BUILDER = "builder"
ROLE_MINER = "miner"


def assign_roles(turn: Turn) -> dict[int, str]:
    """给每个工人分配固定分工

    按 ID 排序后第一个当建造工、第二个当采集工（ID 固定：worker1=10010/20010、
    worker2=10012/20012，接口文档 §1.3.1）。只剩一个工人时，它一个人干两件事，
    由 `plan_worker` 内部的优先级决定这一回合干什么。
    """
    workers = turn.workers()
    roles: dict[int, str] = {}
    for index, worker in enumerate(workers):
        roles[worker.unit_id] = ROLE_BUILDER if index == 0 else ROLE_MINER
    return roles


# ==========================================================================
# 主入口
# ==========================================================================


def plan_day(world: World, claimed: set[Pos]) -> dict[int, dict]:
    """白天全部工人的指令

    `claimed` 由调用方（`brain`）持有并在任务链路之间共享，避免己方角色抢
    同一格导致移动碰撞（任务书 §4.5.4 的"目标点争夺"）。
    """
    commands: dict[int, dict] = {}
    roles = assign_roles(world.turn)
    for worker in world.turn.workers():
        role = roles.get(worker.unit_id, ROLE_MINER)
        # 只剩一个工人时优先保证建造（没有塔的夜晚基地会直接掉血）
        if len(roles) == 1 and not _tower_ready(world):
            role = ROLE_BUILDER
        command = _worker_command(world, worker, role, claimed)
        if command is not None:
            commands[worker.unit_id] = command
    return commands


def _worker_command(
    world: World,
    worker: Unit,
    role: str,
    claimed: set[Pos],
) -> dict | None:
    # 天快黑了：**先回夜间站位**。
    # 这条排在最前面：白天角色常在十几格外的矿区，等天黑再往回走，走回来的
    # 这几个回合炮塔是空的，小怪直接推进（真实对局"角色没在那边操作炮塔
    # 清理小怪"最直接的一条根因）。
    command = defense.dusk_recall(world, worker, claimed)
    if command is not None:
        return command

    # 金币见底：先把背包里的矿石换成钱，再谈别的。
    # 这一条对所有角色生效（不只是采集工）——只剩一个工人时它既是建造工又
    # 是采集工，V1 的"金币恒 0" 正是发生在建造工背着石头不卖的时候。
    if sell_urgent(world, worker):
        command = _sell(world, worker, claimed)
        if command is not None:
            return command

    # 残血先买药（10 金换一条命，比多建一段墙划算）
    command = _buy_medicine(world, worker, claimed)
    if command is not None:
        return command
    if role == ROLE_BUILDER:
        command = _builder(world, worker, claimed)
        if command is not None:
            return command
    return _miner(world, worker, claimed)


# ==========================================================================
# 建造工
# ==========================================================================


def _tower_ready(world: World) -> bool:
    return len(world.turn.towers()) >= defense.MAX_TOWERS


def _builder(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """建造工的优先级队列

    0. 基地残血 + 手里有基地升级券 ⇒ 走到基地旁使用（回满血还顺便升级）
    1. 手里的武器升级券 ⇒ 走到武器旁使用（`战术参考`：优先花钱给炮台升级）
    2. 塔未满 3 座且金币够 ⇒ 建塔（开局第 1 回合就要建，复盘里敌方 R1 已建 2 座）
    3. 手里有围墙修复包且残血墙多 ⇒ 走到墙最密处使用
    4. 计划中的围墙段未砌完 ⇒ 有石头就砌，没石头去采
    5. 金币富余 ⇒ 去武器商店买升级券
    6. 兜底 ⇒ 采石（背包里的石头是围墙的原料）
    """
    turn = world.turn

    command = _use_held_voucher(world, worker, claimed)
    if command is not None:
        return command

    if not _tower_ready(world) and turn.gold >= WEAPON_BUILD_COST:
        command = _build_tower(world, worker, claimed)
        if command is not None:
            return command

    command = _use_wall_fixer(world, worker, claimed)
    if command is not None:
        return command

    command = _build_wall(world, worker, claimed)
    if command is not None:
        return command

    if turn.gold >= 100:
        command = _buy_upgrade_voucher(world, worker, claimed)
        if command is not None:
            return command

    return _go_mine(world, worker, STONE_MINE, claimed)


def _build_tower(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """建塔：走向塔位，到了就建"""
    sites = defense.tower_sites(world)
    if not sites:
        return None
    kind = defense.next_tower_type(world.turn)
    if kind is None:
        return None
    for site in sorted(sites, key=lambda p: (distance(worker.pos, p), p.x, p.y)):
        if site in claimed:
            continue
        command = _go_and_build(world, worker, site, kind, claimed)
        if command is not None:
            return command
    return None


def _build_wall(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """砌墙：有石头才砌，没石头去采（这一步保证"围墙"和"采集"联动）

    **已建够 `WALL_TARGET_SEGMENTS` 段就收手**。`wall_sites` 返回的是"还没建的
    候选段"，不封顶的话建造工会围着基地一圈一圈砌下去——本地模拟里出现过
    "9 段墙 + 1071 金币，建设升级券一次都没买"（`day_summary upgrade=0`），
    钱全躺在账上，正是复盘里"金币闲置"那一类。
    """
    if len(world.turn.walls()) >= defense.WALL_TARGET_SEGMENTS:
        return None
    sites = defense.wall_sites(world)
    if not sites:
        return None
    stones = worker.count(WALL_MATERIAL)
    if stones <= 0:
        return _go_mine(world, worker, STONE_MINE, claimed)

    # **人已经在矿区就把这一批采满再走。** 之前这里还挂着"没有更急的事才顺手
    # 采"（建塔 / 金币 ≥100 要买券就不采），结果是"采 1 块 → 走 10 格回基地 →
    # 砌 1 段 → 再走 10 格回矿"——本地模拟实测 13 回合才出 1 段墙，首夜防线
    # 根本来不及成型（真实对局同理，开局 75 金全砸进 3 座塔之后围墙全程 0 段）。
    # 账很好算：**在矿边多采 1 块只要 1 回合，回一趟矿要 ~20 回合。**
    if stones < STONE_BATCH:
        mine = _adjacent_mine(world, worker, STONE_MINE)
        if mine is not None:
            claimed.add(mine)
            return collect_command(mine)

    for site in sorted(sites, key=lambda p: (distance(worker.pos, p), p.x, p.y)):
        if site in claimed:
            continue
        command = _go_and_build(world, worker, site, WALL, claimed)
        if command is not None:
            return command
    return None


def _go_and_build(
    world: World,
    worker: Unit,
    site: Pos,
    name: str,
    claimed: set[Pos],
) -> dict | None:
    """走到建造位置并建造

    建造要求"目标格与自身距离一格内"（任务书 §4.4），所以先走到目标格旁边，
    到了再发 build。走到位的那一回合只发 move——建造下回合再做。
    """
    from ..protocol import build_command

    if distance(worker.pos, site) <= 1 and worker.pos != site:
        claimed.add(site)
        # 登记这次建造，下一回合用它的执行结果学习可建造区（world.absorb_results）
        from ..world import record_build

        record_build(world.turn, worker.unit_id, site, name)
        return build_command(site, name)

    step = grid.next_step(world.turn, worker, site, claimed)
    if step is None:
        return None
    claimed.add(step)
    return move_command(step)


# ==========================================================================
# 采集工
# ==========================================================================


def miner_order(world: World, worker: Unit) -> tuple[str, ...]:
    """采集工该采什么

    有小贩 ⇒ 按收购价采（铜 5 > 铁 3 > 石 1）；没有小贩 ⇒ 只采石（墙的唯一原料）。
    """
    if _nearest_zone(world.turn, VENDOR, worker.pos) is None:
        return MINER_ORDER_NO_VENDOR
    return MINER_ORDER


def _miner(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """采集工：卖矿优先于采集，但两端之间不做无谓往返

    只有"值得卖"的时候才跑一趟小贩：金币见底、背包快满、或者囤的石头
    超过围墙需求。其余时间一直待在矿区采。
    """
    if should_sell(world, worker):
        command = _sell(world, worker, claimed)
        if command is not None:
            return command
    for kind in miner_order(world, worker):
        command = _go_mine(world, worker, kind, claimed)
        if command is not None:
            return command
    return None


def should_sell(world: World, worker: Unit) -> bool:
    """该不该去小贩那儿卖矿

    四条触发条件，任意一条成立即可：

    1. **金币见底**（`gold < 25`）：这是"金币冻结"的直接对策——只要背包里有
       卖得掉的东西，就一定要把它换成金币。
    2. **金币为 0 的兜底**：哪怕只有一块石头也卖（`gold == 0`）。
    3. **背包快满**：采满了再卖会浪费回合。
    4. **石材囤积**：超过围墙需求的部分留着没有意义（V1 里 stone 从 1 块
       堆到 3 块、金币从 R6 起恒 0 到 R17，工人背着石头空转）。
    """
    turn = world.turn
    sellable = [item for item in worker.backpack if item in ("stone", "iron", "copper")]
    if not sellable:
        return False
    if turn.gold == 0:
        return True
    if turn.gold < GOLD_LOW:
        return True
    if worker.capacity and len(worker.backpack) >= worker.capacity * BACKPACK_SELL_RATIO:
        return True
    pending_walls = len(defense.wall_sites(world))
    if pending_walls == 0 and worker.count(WALL_MATERIAL) > STONE_RESERVE:
        return True
    return False


def _sell(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """去小贩那儿卖矿：优先卖贵的，最后才卖石头"""
    vendor = _nearest_zone(world.turn, VENDOR, worker.pos)
    if vendor is None:
        return None

    holding = [
        item
        for item in ("copper", "iron", "stone")
        if worker.count(item) > 0
    ]
    if not holding:
        return None

    if distance(worker.pos, vendor) <= 1 and worker.pos != vendor:
        name = holding[0]
        count = worker.count(name)
        # 石材留给围墙：至少留 STONE_RESERVE 块（还有墙要砌时留 STONE_BATCH）
        if name == WALL_MATERIAL:
            keep = STONE_BATCH if defense.wall_sites(world) else STONE_RESERVE
            count -= keep
        if count <= 0:
            return None
        return sell_command(name, count)

    return _approach(world, worker, vendor, claimed)


def _go_mine(
    world: World,
    worker: Unit,
    kind: str,
    claimed: set[Pos],
) -> dict | None:
    """去采某种矿：到了就采，没到就走"""
    mines = world.turn.mines(kind)
    if not mines:
        return None
    if worker.backpack_full:
        return None

    ranked = sorted(
        (m for m in mines if m not in claimed),
        key=lambda m: (distance(worker.pos, m), m.x, m.y),
    )
    for mine in ranked:
        if distance(worker.pos, mine) <= 1 and worker.pos != mine:
            claimed.add(mine)
            return collect_command(mine)
        command = _approach(world, worker, mine, claimed)
        if command is not None:
            return command
    return None


def _approach(world: World, unit: Unit, target: Pos, claimed: set[Pos]) -> dict | None:
    """朝 target **旁边的一格**走一步

    矿区、小贩、武器商店都是中立单位，`is_land` 为假、属于障碍物
    （任务书 §4.1），所以不能把它们的坐标当成终点——那样 A* 永远找不到路，
    返回 None，采集中断/卖矿中断正是这么来的。正确做法是走到它们周围一格内。
    """
    stands = grid.stand_cells_for(world.turn, target, unit)
    if not stands:
        return None
    if unit.pos in stands:
        return None  # 已经到位，由调用方决定发什么指令
    step = grid.step_toward_any(world.turn, unit, stands, claimed)
    if step is None:
        return None
    claimed.add(step)
    return move_command(step)


def _has_sellable(worker: Unit) -> bool:
    return any(item in ("stone", "iron", "copper") for item in worker.backpack)


def sell_urgent(world: World, worker: Unit) -> bool:
    """是否应该**立刻**去卖矿（金币见底）

    与 `should_sell` 的区别：这里的触发条件只有"没钱"一条，优先级高于一切
    建造/采集动作——没金币就什么都建不了、买不了。
    """
    if not _has_sellable(worker):
        return False
    return world.turn.gold < GOLD_LOW


def _adjacent_mine(world: World, worker: Unit, kind: str) -> Pos | None:
    mines = sorted(
        (m for m in world.turn.mines(kind) if distance(worker.pos, m) <= 1 and worker.pos != m),
        key=lambda m: (distance(worker.pos, m), m.x, m.y),
    )
    return mines[0] if mines else None


def _nearest_zone(turn: Turn, kind: str, origin: Pos) -> Pos | None:
    zones = turn.zones_of(kind)
    if not zones:
        return None
    return min(zones, key=lambda p: (distance(origin, p), p.x, p.y))


# ==========================================================================
# 道具使用
# ==========================================================================


def _use_held_voucher(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """用掉背包里的升级券

    券是"买回来放在背包里"的（任务书 §4.6.3），所以报文的 `backpack` 字段就是
    最好的状态来源——不需要任何跨回合记忆。V1 全程没买过也没用过升级券，
    "三塔零升级"（E4）就是这么来的。
    """
    turn = world.turn

    # 基地残血：先救基地（回满血 + 顺便升级，聊天记录 18:20）
    station = turn.station()
    if station is not None:
        for level in (2, 3):
            voucher = STATION_UPGRADE_VOUCHER[level]
            if worker.count(voucher) <= 0:
                continue
            if station.level >= level or station.health_ratio() > 0.6:
                continue
            command = _use_at(world, worker, voucher, station.pos, claimed)
            if command is not None:
                return command

    # 武器升级：优先升等级最低的那座（每一级的提升都一样，先补短板）
    towers = sorted(
        (t for t in turn.towers() if t.level < 3),
        key=lambda t: (t.level, distance(worker.pos, t.pos)),
    )
    for tower in towers:
        voucher = WEAPON_UPGRADE_VOUCHER.get(tower.level + 1)
        if not voucher or worker.count(voucher) <= 0:
            continue
        command = _use_at(world, worker, voucher, tower.pos, claimed)
        if command is not None:
            return command
    return None


def _reach_cells(world: World, target: Pos, mover: Unit) -> tuple[Pos, ...]:
    """目标建筑周围一格内、当前可站立的格子

    基地占 2×2（接口文档 §1.3.1 注：`pos` 是左上角），"周围一格内"必须按
    整个 footprint 算——只按左上角那一格算的话，站在基地右侧的角色会被判成
    "还没到位"，然后对着一个自己已经站着的落脚点反复 `next_step`（返回
    None），升级券就永远用不出去。
    """
    station = world.turn.station()
    spots = (
        station_footprint(target)
        if station is not None and target == station.pos
        else (target,)
    )
    occupants = world.turn.occupied() - {mover.pos}
    cells: list[Pos] = []
    for spot in spots:
        for cell in grid.stand_cells(world.turn, spot, occupants=occupants):
            if cell not in cells:
                cells.append(cell)
    return tuple(cells)


def _use_at(
    world: World,
    worker: Unit,
    name: str,
    target: Pos,
    claimed: set[Pos],
) -> dict | None:
    """走到目标建筑旁边并使用道具

    升级券要求"在目标建筑周围一格内使用并指定目标位置"（任务书 §4.6.3），
    所以先就位再用。
    """
    reach = _reach_cells(world, target, worker)
    if worker.pos in reach:
        return use_command(name, target)
    step = grid.step_toward_any(world.turn, worker, reach, claimed)
    if step is None:
        return None
    claimed.add(step)
    return move_command(step)


def _use_wall_fixer(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """围墙修复包：残血墙 ≥ 3 段时，挑"3×3 内残血墙最密集"的那一段修

    任务书 §4.6.3 写的是"目标坐标所在围墙回满血"，聊天记录（18:22）说的是
    "可以给周围 3×3 的围墙回满血，一次可以给 5 个墙奶满"。取交集：目标选
    残血墙最密集处，单体口径命中该段、群体口径一次覆盖最多段。
    """
    if worker.count(WALL_FIXER) <= 0:
        return None

    damaged = [w for w in world.turn.walls() if w.health_ratio() < 0.9]
    if len(damaged) < REPAIR_MIN_DAMAGED_WALLS:
        return None

    best, best_score = None, -1
    for candidate in damaged:
        score = sum(
            1
            for other in damaged
            if distance(candidate.pos, other.pos) <= 1
        )
        if score > best_score:
            best_score, best = score, candidate
    if best is None:
        return None
    return _use_at(world, worker, WALL_FIXER, best.pos, claimed)


def _buy_upgrade_voucher(world: World, worker: Unit, claimed: set[Pos]) -> dict | None:
    """去武器商店买升级券

    买哪张取决于当前塔的最低等级：还有 level1 的塔就买 1 级券，
    全到 level2 了才买 2 级券。
    """
    turn = world.turn
    towers = turn.towers()
    if not towers:
        return None
    lowest = min(t.level for t in towers)
    if lowest >= 3:
        return None
    voucher = WEAPON_UPGRADE_VOUCHER.get(lowest + 1)
    if not voucher:
        return None
    price = turn.shop_price(voucher)
    if price is None or turn.gold < price:
        return None
    # 金币预留：买完还得留得下一座塔的钱（除非塔已经满了）
    if not _tower_ready(world) and turn.gold - price < WEAPON_BUILD_COST:
        return None

    shop = _nearest_zone(turn, WEAPON_SHOP, worker.pos)
    if shop is None:
        return None
    if distance(worker.pos, shop) <= 1 and worker.pos != shop:
        return buy_command(voucher, 1)
    return _approach(world, worker, shop, claimed)


def _buy_medicine(world: World, unit: Unit, claimed: set[Pos]) -> dict | None:
    """角色残血时买一瓶生命药剂（10 金换一条命，很划算）"""
    turn = world.turn
    if turn.gold < MEDICINE_PRICE + WEAPON_BUILD_COST:
        return None
    if unit.health_ratio() > 0.5 or unit.count(MEDICINE) > 0:
        return None
    shop = _nearest_zone(turn, WEAPON_SHOP, unit.pos)
    if shop is None:
        return None
    if distance(unit.pos, shop) <= 1 and unit.pos != shop:
        return buy_command(MEDICINE, 1)
    return None


def station_cells(turn: Turn) -> tuple[Pos, ...]:
    """基地 footprint 周围一圈的可站格（夜间回防的落点）"""
    station = turn.station()
    if station is None:
        return ()
    cells: list[Pos] = []
    for cell in station_footprint(station.pos):
        for neighbour in grid.stand_cells(turn, cell):
            if neighbour not in cells:
                cells.append(neighbour)
    return tuple(cells)
