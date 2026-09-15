"""防御策略：塔位与墙线规划、夜晚火力控制。

对应设计文档V2 §7.4 与第 8 章。

V1 的塔位是一组写死的常量（实测落在 `(30,8)/(29,9)/(30,11)`，三塔贴基地同侧
零升级），换边之后完全失效。V2 改成**由基地位置现算 + 在线学习可建造区**：

1. 候选格从"相对基地 footprint 的偏移"生成，换边后自动跟着平移。
2. 只使用已确证可建造的偏移（`world.zone()` 里学到 `True` 的那些）；不足时
   按由近到远试探未知偏移。
3. 选定组合必须通过**建成预演**：假设这些塔全部建成，每座塔旁边都还有一格能
   从基地走到——直接对应聊天记录里"很容易出现炮塔把路堵住然后有一个炮塔
   碰不到"的坑（`战术参考/修改建议.md` 的 T2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from .. import grid
from ..protocol import (
    CHARACTER_TYPES,
    GATLING,
    GATLING_CONE_DEGREES,
    LAND,
    RAILGUN,
    ROCKET,
    ROCKET_MISSILE_DAMAGE,
    ROCKET_SPLASH_DAMAGE,
    WALL,
    Pos,
    Robot,
    Turn,
    Unit,
    attack_command,
    distance,
    move_command,
    neighbours,
    station_footprint,
)
from ..world import World

# 武器建造顺序：射程优先（火箭 10 > 电磁狙击炮 6 > 加特林 3）。
# 对战复盘里敌方开局就建射程 10 的火箭发射台，我方却先建射程 3 的加特林，
# 机器人一路走到基地跟前才开始挨打。`战术参考/聊天记录_解码.txt`（18:18）的
# 结论也是"2 导弹 1 电磁或者 3 导弹，导弹性价比高一点"。
TOWER_LOADOUT: tuple[str, ...] = (ROCKET, RAILGUN, GATLING)

# 全局同时最多 3 座武器（任务书 §4.5.1）
MAX_TOWERS = 3

# 围墙目标段数：够盖住来向 + 两侧即可（石头是稀缺资源，铺满一圈不现实）
WALL_TARGET_SEGMENTS = 8

SIDE_ORDER = ("up", "down", "left", "right")

# 天黑前预留几个回合回防（路程正好等于剩余回合时才出发就太紧了）
DUSK_MARGIN = 1

# 判断"够不够得着这座塔"时，最多向外搜几步。
#
# 这不是可有可无的优化：夜间配位每回合都要问 人×塔 次"够不够得着"，不限深
# 就是全图 BFS，实测把单回合决策从 0.9ms 抬到 10.4ms（local_check 的 1300
# 回合基线）；限深 8 之后是 2.5ms，而要求是 <1s/回合，余量足够。
#
# 限深还有个好性质：**它只改变"有区分度"的判断**。从任务点往回走的角色可能
# 离三座塔都超过 8 步，那时三座塔一律判"够不着"，排序自动退回按距离——正是
# 该有的行为。
REACH_PROBE_STEPS = 8

# 只在"距天黑还剩这么多回合"以内才去算回防距离。
# 再早就不用回：哪怕要走 20 步也来得及。这条早退是性能上限——
# 否则白天每回合都要给每个角色跑一次寻路。
DUSK_RECALL_WINDOW = 25


# ==========================================================================
# 方位
# ==========================================================================


def side_of(offset: Pos) -> str:
    """相对偏移属于基地的哪一侧

    基地 footprint 占偏移 `x ∈ [0, 1] × y ∈ [0, 1]`（2×2，原点是左下角），
    所以"基地外"的距离要先把这 2×2 减掉：`x = -1` 是左侧而不是右侧，
    `x = 2` 才是右侧。按绝对值大的那一维定方位，斜角归给更远的那个方向。
    """
    dx = _outside(offset.x)
    dy = _outside(offset.y)
    if dx == 0 and dy == 0:
        return "right"  # 基地本体内部，实际不会出现
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "up" if dy > 0 else "down"


def _outside(value: int) -> int:
    """该坐标离基地 footprint（占偏移 0 与 1 两格）有多远（带方向）

    基地 footprint 相对原点占 `[0, 1]` 两格，所以负方向的距离就是原值
    （`-1` 表示基地左边一格），正方向要减掉那两格（`2` 表示基地右边一格）。
    """
    if value < 0:
        return value
    if value > 1:
        return value - 1
    return 0


def side_rank(side: str, enemy_sides: frozenset[str]) -> int:
    if side in enemy_sides:
        return 0
    return 1 + (SIDE_ORDER.index(side) if side in SIDE_ORDER else 9)


def ordered_sides(world: World) -> tuple[str, ...]:
    enemy = world.defence_sides()
    return tuple(sorted(SIDE_ORDER, key=lambda s: (side_rank(s, enemy), SIDE_ORDER.index(s))))


# ==========================================================================
# 塔位
# ==========================================================================


def next_tower_type(turn: Turn) -> str | None:
    """这一座该建哪种武器

    按 `TOWER_LOADOUT` 的顺序**补缺**：已经有一座火箭、两座加特林而火箭被打掉了，
    下一个补的仍然是火箭。射程 10 的火力对夜防的边际价值最高，不能因为
    "塔数已经够了"就随便补一个型号。
    """
    if len(turn.towers()) >= MAX_TOWERS:
        return None
    have = {kind: 0 for kind in TOWER_LOADOUT}
    for tower in turn.towers():
        if tower.kind in have:
            have[tower.kind] += 1
    for kind in TOWER_LOADOUT:
        if have[kind] == 0:
            return kind
    return None


def candidate_offsets(world: World, name: str, radius: int = 3) -> tuple[Pos, ...]:
    """某类建筑的候选偏移（相对基地原点），按战术优先级排序

    排序依据：
      1. 来敌方向优先（`side_rank`）
      2. 已确证可建造的排在未知的前面（学到的知识立刻生效）
      3. 同档次内按距离基地由近到远
    """
    zone = world.zone()
    enemy = world.defence_sides()
    offsets: list[Pos] = []
    for ring in range(1, radius + 1):
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                if max(abs(dx), abs(dy)) != ring:
                    continue
                offset = Pos(dx, dy)
                if zone.forbidden(name, offset):
                    continue
                offsets.append(offset)
    return tuple(sorted(offsets, key=lambda o: _offset_key(o, zone, name, enemy)))


def _offset_key(offset: Pos, zone, name: str, enemy: frozenset[str]) -> tuple:
    known = zone.known(name, offset)
    return (
        side_rank(side_of(offset), enemy),
        0 if known is True else 1,
        max(abs(offset.x), abs(offset.y)),
        abs(offset.x) + abs(offset.y),
        offset.x,
        offset.y,
    )


def tower_sites(world: World, limit: int = MAX_TOWERS) -> tuple[Pos, ...]:
    """塔位候选（绝对坐标）

    先按战术优先级取前 8 个可用偏移，再从中挑出一个**通过建成预演**的组合。
    预演不通过就换下一个组合（候选很少，直接组合枚举）。
    """
    turn = world.turn
    station = turn.station()
    if station is None:
        return ()

    need = MAX_TOWERS - len(turn.towers())
    if need <= 0:
        return ()

    usable: list[Pos] = []
    for offset in candidate_offsets(world, ROCKET):
        absolute = world.absolute_of(offset)
        if absolute in station_footprint(station.pos):
            continue
        if not world.buildable_now(absolute):
            continue
        usable.append(absolute)
        if len(usable) >= 8:
            break
    if not usable:
        return ()

    standing = tuple(t.pos for t in turn.towers())
    for size in range(min(need, limit), 0, -1):
        for group in combinations(usable, size):
            if layout_ok(turn, standing + group):
                return group
    # 一个组合都不通过：退回最靠前的几个。宁可堵一点，也不能不建——
    # 没有塔的夜晚基地会直接掉血（复盘里"基地单回合掉血 1350"）。
    return tuple(usable[: min(need, limit)])


def layout_ok(turn: Turn, sites: tuple[Pos, ...]) -> bool:
    """建成预演：这些格子被占掉之后，每座建筑是否仍有人能操控到

    这是 `战术参考/修改建议.md` 的 T2（唯一"会导致整局失效"级的问题）：
    三座塔一旦把 6×6 区域的通道切断，就会出现"塔在，但没人能操控"。
    """
    station = turn.station()
    if station is None:
        return True
    blocked = set(sites)
    for site in sites:
        stands = tuple(
            cell
            for cell in neighbours(site)
            if turn.is_land(cell) and cell not in blocked
        )
        if not stands:
            return False
        if not grid.reachable_any(turn, station.pos, stands, blocked):
            return False
    return True


# ==========================================================================
# 墙线
# ==========================================================================


def wall_sites(world: World, target: int = WALL_TARGET_SEGMENTS) -> tuple[Pos, ...]:
    """围墙候选（绝对坐标），按"先封来敌方向"排序

    取基地周围**第二圈**（切比雪夫半径 2）：第一圈紧贴基地，建满了会把基地
    围死、角色出不去。
    """
    turn = world.turn
    station = turn.station()
    if station is None:
        return ()
    zone = world.zone()
    enemy = world.defence_sides()
    footprint = set(station_footprint(station.pos))
    existing = {w.pos for w in turn.walls()}

    offsets: list[Pos] = []
    for dx in range(-2, 4):
        for dy in range(-2, 4):
            if max(abs(dx), abs(dy)) != 2:
                continue
            offset = Pos(dx, dy)
            if zone.forbidden(WALL, offset):
                continue
            offsets.append(offset)

    result: list[Pos] = []
    for offset in sorted(offsets, key=lambda o: _offset_key(o, zone, WALL, enemy)):
        absolute = world.absolute_of(offset)
        if absolute in footprint or absolute in existing:
            continue
        if not world.buildable_now(absolute):
            continue
        result.append(absolute)
        if len(result) >= target:
            break
    return tuple(result)


# ==========================================================================
# 天黑前回防
# ==========================================================================


def rounds_until_night(turn: Turn) -> int:
    """距离天黑还有几个回合（白天最后 1 回合返回 1，夜晚返回 0）

    任务书 §4.2：白天 70 回合、夜晚 60 回合，每个游戏日 130 回合。
    """
    if not turn.is_day:
        return 0
    from ..protocol import DAY_ROUNDS, ROUNDS_PER_DAY

    round_in_day = (turn.round_no - 1) % ROUNDS_PER_DAY + 1
    return max(0, DAY_ROUNDS + 1 - round_in_day)


def station_cells(world: World) -> tuple[Pos, ...]:
    """夜间站位：每座武器周围的落脚点；没有武器时退回基地周围一圈"""
    turn = world.turn
    cells: list[Pos] = []
    for tower in turn.towers():
        for cell in grid.stand_cells(turn, tower.pos, occupants=turn.occupied()):
            if cell not in cells:
                cells.append(cell)
    if cells:
        return tuple(cells)
    station = turn.station()
    if station is None:
        return ()
    return tuple(
        grid.cells_in_radius(station.pos, 2, turn.width, turn.height)
    )


def station_distance(world: World, unit: Unit, limit: int) -> int:
    """到最近夜间站位的步数（没有站位时返回 0 = 不用回防）

    超过 `limit` 步一律返回 `limit + 1`（"来不来得及"的语义足够），
    算不出路径时退回切比雪夫距离——宁可按老办法估一个（因此提前一点出发），
    也不要返回 0 导致根本不回防。
    """
    cells = station_cells(world)
    if not cells:
        return 0
    steps = grid.steps_to_any(world.turn, unit.pos, cells, limit)
    if steps is not None:
        return steps
    if grid.reachable_any(world.turn, unit.pos, cells):
        return limit + 1          # 能到，但比 limit 远
    return min(distance(unit.pos, cell) for cell in cells)


def dusk_recall(world: World, unit: Unit, claimed: set[Pos]) -> dict | None:
    """天快黑了：放下手里的活，先回到夜间站位

    这是"炮塔没人操控、小怪直接推进"最直接的一条根因——白天角色在十几格外的
    矿区/商店，而**就位逻辑是天黑之后才启动的**，等它走回来已经是好几个回合
    之后，塔在这段时间里一直是空的。

    判据用"路够不够走"而不是固定回合数：`到站位的距离 >= 剩余白天回合` 才出发，
    所以近的角色继续干活到最后一刻，远的会提前走。
    """
    if not world.turn.is_day:
        return None
    left = rounds_until_night(world.turn)
    if left <= 0 or left > DUSK_RECALL_WINDOW:
        return None
    threshold = left + DUSK_MARGIN
    if station_distance(world, unit, threshold) < threshold:
        return None
    return guard_weapon(world, unit, claimed)


# ==========================================================================
# 夜晚火力控制
# ==========================================================================


@dataclass(slots=True)
class NightAction:
    """夜间一条待下发的角色指令"""

    unit: Unit
    command: dict | None  # None 表示这一回合该角色不动
    role: str  # fire / approach / plug / idle


def night_actions(
    world: World,
    exclude: frozenset[int] = frozenset(),
) -> list[NightAction]:
    """夜晚的全部角色指令

    顺序：
      1. **开火**：角色已经在武器旁（距离 ≤ 1）且武器没在冷却 → 算出落点并攻击。
      2. **就位**：角色不在武器旁 → 朝这座武器旁边的落脚格移动。
      3. **堵口**：没摊上武器的角色去站防线缺口（人肉城墙）。
      4. **待命**：都没有 → 站到最近的武器旁边等下回合。
    """
    turn = world.turn
    robots = list(turn.alive_robots())
    characters = [c for c in turn.characters() if c.unit_id not in exclude]
    if not characters:
        return []

    weapons = list(turn.towers())
    claimed: set[Pos] = set()
    actions: list[NightAction] = []
    paired_characters: set[int] = set()

    if robots:
        # `assigned` 只装**已经有人站到位、真的在开火**的塔。装"所有已配对的塔"
        # 是不行的：三塔三人时最后一个人看到的就是全集，换塔那条路直接死掉。
        covered: set[int] = set()
        for weapon, controller in _pair(world, characters, weapons):
            paired_characters.add(controller.unit_id)
            actions.append(_act(
                world, weapon, controller, robots, claimed,
                assigned=frozenset(covered),
            ))
            if distance(controller.pos, weapon.pos) <= 1:
                covered.add(weapon.unit_id)

    for character in characters:
        if character.unit_id in paired_characters:
            continue
        command = plug_gap(world, character, claimed) or guard_weapon(world, character, claimed)
        if command is not None:
            actions.append(NightAction(character, command, "plug"))
    return actions


def _static_blockers(world: World) -> set[Pos]:
    """反向可达性用的障碍：**建筑 + 中立 + 机器人**，不含角色

    `grid.reachable()` 默认只看地形（它本来是为"预演建成后的连通性"写的），
    但夜里真正把路堵死的是**机器人**：任务书 §4.1 规定机器人与角色都阻挡移动。
    实测报文里开拓者就是这么被闷在基地东南角的——旁边的机器人一直不动，
    地形上算"连通"，实际一步也走不了。

    **刻意不把角色算进去**：反向搜索要回答的是"这个人够不够得着这座塔"，
    把他自己当成障碍会让每个人把自己判成到不了。代价是忽略了"队友挡路"
    这一个小情形——真发生时 `_act` 的换塔兜底会接住。
    """
    turn = world.turn
    blocked = {pos for pos, name in turn.zones.items() if name != LAND}
    blocked |= {
        pos
        for unit in turn.ours
        if unit.is_alive and unit.kind not in CHARACTER_TYPES
        for pos in turn.footprint(unit)
    }
    blocked |= {robot.pos for robot in turn.robots if robot.is_alive}
    return blocked


def _pair(
    world: World,
    characters: list[Unit],
    weapons: list[Unit],
) -> list[tuple[Unit, Unit]]:
    """角色 <-> 武器配对：**先看够不够得着，再看离得近不近**

    只按距离贪心会配出"隔着基地"的组合，而且代价是持续的。实测报文里开拓者
    站在基地东侧 `(30,11)`、离加特林只有 1 格，却因为"加特林被 id 更小的工人
    先挑走"而被配到基地西侧那两座塔之一——绕过去要 6 步，机器人一压过来路就
    断了，于是它整晚站在原地：**72 个夜战回合只开了 3 次火，三座塔长期只有
    两座在开火**，首夜基地被打掉 1415 血。

    两处都不能省：

    1. **只按距离贪心不够，要整体指派。** 贪心挑"单个最优"会漏掉"加特林离
       两个人都只有 1 格、而开拓者除了加特林哪座都去不了"这种局面——它先把
       加特林给 id 更小的工人，开拓者就只剩那座走不到的塔。枚举全指派
       （武器 ≤3，最多十几组）才看得出该把加特林让给谁。
    2. **可达性要限深。** 每回合问 人×塔 次"够不够得着"，不限深就是全图 BFS，
       实测把单回合决策从 0.9ms 抬到 10.4ms。

    外加一条稳态短路：人人都贴着某座塔时（夜间绝大多数回合），配对没有歧义，
    一次搜索都不用跑——那正是 0.9ms 基线的情形。
    """
    if not weapons:
        return []

    plain = _assign(characters, weapons, lambda ci, wi: True)
    if all(distance(w.pos, c.pos) <= 1 for w, c in plain):
        return plain

    static = _static_blockers(world)
    reach_sets = {
        ci: grid.reachable_set(
            world.turn, [characters[ci].pos], static, limit=REACH_PROBE_STEPS
        )
        for ci in range(len(characters))
    }
    stands_cache: dict[int, tuple[Pos, ...]] = {}

    def reachable(ci: int, wi: int) -> bool:
        if wi not in stands_cache:
            stands_cache[wi] = grid.stand_cells(world.turn, weapons[wi].pos)
        return any(cell in reach_sets[ci] for cell in stands_cache[wi])

    return _assign(characters, weapons, reachable)


def _assign(characters, weapons, reachable) -> list[tuple[Unit, Unit]]:
    """枚举"哪几个人上塔 × 怎么配"，取最好的那组

    排序键：① 够不着的人最少 ② 总距离最短 ③ 已经在位的人最多（少折腾）

    要**同时**枚举"哪几个人"和"怎么配"：塔比人少时（1 塔 2 人）
    `permutations(range(1), 2)` 是空集，只枚举配对会把所有人都漏掉。
    """
    size = min(len(characters), len(weapons))
    best_key: tuple | None = None
    best_slots: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    for chosen in combinations(range(len(characters)), size):
        for perm in permutations(range(len(weapons)), size):
            unreachable = 0
            total = 0
            in_place = 0
            for slot, ci in enumerate(chosen):
                wi = perm[slot]
                if not reachable(ci, wi):
                    unreachable += 1
                gap = distance(weapons[wi].pos, characters[ci].pos)
                total += gap
                if gap <= 1:
                    in_place += 1
            key = (unreachable, total, -in_place, chosen, perm)
            if best_key is None or key < best_key:
                best_key, best_slots = key, (chosen, perm)

    chosen, perm = best_slots
    return [(weapons[perm[slot]], characters[ci]) for slot, ci in enumerate(chosen)]


def _act(
    world: World,
    weapon: Unit,
    controller: Unit,
    robots: list[Robot],
    claimed: set[Pos],
    assigned: frozenset[int] = frozenset(),
) -> NightAction:
    """一座武器这一回合的动作"""
    if distance(controller.pos, weapon.pos) <= 1:
        targets = fire_targets(world.turn, weapon, robots)
        if targets:
            command = attack_command(controller.unit_id, weapon, targets)
            if command is not None:
                return NightAction(controller, command, "fire")

    # **够不着就换一座有目标的塔。** 两种情况都走这里：
    #   1. 人已经在某座塔旁边，但这座射程内没目标（配对按时序硬配的）
    #   2. 人还没到位，而配对那座**根本走不过去**——隔着基地，或者被机器人堵死
    # 实测报文里缺的正是第 2 条：开拓者被配到基地另一侧那座塔，绕行要 6 步，
    # 机器人一压过来路就断了，于是整晚站在原地（72 个夜战回合只开了 3 次火）。
    switched = _switch(world, weapon, controller, robots, claimed, assigned)
    if switched is not None:
        return switched

    if distance(controller.pos, weapon.pos) <= 1:
        # 站在塔边、这座没目标、也换不了：站住别乱跑
        return NightAction(controller, None, "idle")

    stand = _stand_near(world, weapon, controller, claimed)
    if stand is None:
        return NightAction(controller, None, "idle")
    step = grid.next_step(world.turn, controller, stand, claimed)
    if step is None:
        return NightAction(controller, None, "idle")
    claimed.add(step)
    return NightAction(controller, move_command(step), "approach")


def _switch(
    world: World,
    weapon: Unit,
    controller: Unit,
    robots: list[Robot],
    claimed: set[Pos],
    assigned: frozenset[int],
) -> NightAction | None:
    """换到另一座"射程内有目标且没人管"的塔；没有就返回 None

    塔挨在一起时人可能**同时**邻着两座——那种情况直接开火，不用先走过去。
    """
    better = _tower_with_targets(world, controller, robots, assigned)
    if better is None or better.unit_id == weapon.unit_id:
        return None

    if distance(controller.pos, better.pos) <= 1:
        targets = fire_targets(world.turn, better, robots)
        if targets:
            command = attack_command(controller.unit_id, better, targets)
            if command is not None:
                return NightAction(controller, command, "fire")

    stand = _stand_near(world, better, controller, claimed)
    if stand is None or stand == controller.pos:
        return None
    step = grid.next_step(world.turn, controller, stand, claimed)
    if step is None:
        return None
    claimed.add(step)
    return NightAction(controller, move_command(step), "switch")


def _tower_with_targets(
    world: World,
    controller: Unit,
    robots: list[Robot],
    assigned: frozenset[int],
) -> Unit | None:
    """射程内有目标、且还没有人操控的塔里最近的那座"""
    candidates = [
        tower
        for tower in world.turn.towers()
        if tower.unit_id not in assigned
        and fire_targets(world.turn, tower, robots)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda t: (distance(controller.pos, t.pos), t.pos.x, t.pos.y),
    )


def _stand_near(
    world: World,
    weapon: Unit,
    controller: Unit,
    claimed: set[Pos],
) -> Pos | None:
    cells = grid.stand_cells_for(world.turn, weapon.pos, controller, busy=claimed)
    if not cells:
        return None
    return min(cells, key=lambda c: (distance(controller.pos, c), c.x, c.y))


def fire_targets(
    turn: Turn,
    weapon: Unit,
    robots: list[Robot] | None = None,
) -> tuple[Pos, ...]:
    """给一座武器算本回合的落点（按武器类型分派）

    三种武器的机制完全不同（任务书 §4.5.4 第 4 条），不能共用一套目标选择。
    """
    if weapon.cooldown > 0:
        return ()
    pool = list(robots) if robots is not None else list(turn.alive_robots())
    reach = weapon.range_of_attack()
    in_range = [r for r in pool if distance(weapon.pos, r.pos) <= reach]
    if not in_range:
        return ()
    if weapon.kind == RAILGUN:
        return (_railgun_target(weapon, in_range),)
    if weapon.kind == ROCKET:
        return _rocket_targets(weapon, in_range)
    if weapon.kind == GATLING:
        return _gatling_targets(weapon, in_range)
    return ()


def _gatling_targets(weapon: Unit, robots: list[Robot]) -> tuple[Pos, ...]:
    """加特林：威胁最高的前 level 个目标，全部落在同一 90° 锥内

    任务书 §4.5.4：任意两个目标相对加特林的方向夹角 ≤ 90°，否则**整次攻击非法**。
    凑不够目标就用锚点重复填充——重复落点只是"打同一个地方"，属于指令执行失败
    而不是异常，不会消耗队伍的异常预算（任务书 §8）。
    """
    ranked = sorted(robots, key=lambda r: (-r.threat, distance(weapon.pos, r.pos), r.robot_id))
    need = weapon.target_count()
    anchor = ranked[0].pos
    chosen = [anchor]
    for robot in ranked[1:]:
        if len(chosen) >= need:
            break
        if grid.within_cone(weapon.pos, anchor, robot.pos, GATLING_CONE_DEGREES):
            chosen.append(robot.pos)
    while len(chosen) < need:
        chosen.append(anchor)
    return tuple(chosen)


def _railgun_target(weapon: Unit, robots: list[Robot]) -> Pos:
    """电磁狙击炮：选弹道穿透收益最高的落点

    对每个机器人位置当落点，把"从炮到落点这条线"上的机器人按距离排序，模拟
    能量沿途扣减（任务书 §4.5.4：每只存活机器人受 `min(剩余能量, 当前血量)`
    伤害，能量按造成的伤害量扣减；能量耗尽或到达终点即止）。
    """
    energy = weapon.attack_points()
    best, best_damage = robots[0].pos, -1
    for candidate in robots:
        line = sorted(
            (r for r in robots if _on_line(weapon.pos, candidate.pos, r.pos)),
            key=lambda r: distance(weapon.pos, r.pos),
        )
        remaining = energy
        total = 0
        for robot in line:
            hit = min(remaining, robot.health)
            total += hit
            remaining -= hit
            if remaining <= 0:
                break
        if total > best_damage:
            best_damage, best = total, candidate.pos
    return best


def _on_line(start: Pos, end: Pos, point: Pos) -> bool:
    """点是否落在 start->end 的线段上（容差 0.6 格）

    攻击路径是两个坐标中心之间相连的直线（任务书 §4.5.4 第 3 条）。
    """
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return point == start
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    if t < -0.001 or t > 1.001:
        return False
    return (start.x + t * dx - point.x) ** 2 + (start.y + t * dy - point.y) ** 2 <= 0.36


def _rocket_targets(weapon: Unit, robots: list[Robot]) -> tuple[Pos, ...]:
    """火箭发射台：每枚导弹选一个 3×3 溅射收益最高的落点

    任务书 §4.5.4：每枚导弹中心伤害固定 20，落点周围 8 格溅射是中心伤害的
    一半，多枚导弹落点重叠时伤害叠加。导弹枚数 = 当前等级，所以等级 3 的火箭
    要给出 3 个落点（接口文档 §2.2）。
    """
    need = weapon.target_count()
    remaining = list(robots)
    chosen: list[Pos] = []
    for _ in range(need):
        if not remaining:
            break
        best, best_score = remaining[0].pos, -1.0
        for robot in remaining:
            score = 0.0
            for other in remaining:
                span = distance(robot.pos, other.pos)
                if span == 0:
                    score += min(ROCKET_MISSILE_DAMAGE, other.health) * other.threat
                elif span == 1:
                    score += min(ROCKET_SPLASH_DAMAGE, other.health) * other.threat
            if score > best_score:
                best_score, best = score, robot.pos
        chosen.append(best)
        # 已经覆盖到的目标从候选里去掉，让下一枚导弹去打别处
        remaining = [r for r in remaining if distance(r.pos, best) > 1]
    while len(chosen) < need:
        chosen.append(chosen[-1] if chosen else weapon.pos)
    return tuple(chosen)


# ==========================================================================
# 夜间移动
# ==========================================================================


def plug_gap(world: World, unit: Unit, claimed: set[Pos]) -> dict | None:
    """堵缺口：站到"计划要建但还没建的围墙"那一格上，用身体挡一回合

    见 `战术参考/人肉城墙堵缺口.png` 与聊天记录 18:27：
    "他直接拿小人当人肉城墙把缺口堵上了"。

    围墙缺口格是**空地**（还没建），所以可以直接把它当终点；已经在缺口上时
    不发指令（站着不动就是堵着）。
    """
    gaps = tuple(g for g in wall_sites(world) if g not in claimed)
    if unit.pos in gaps:
        return None
    step = grid.step_toward_any(world.turn, unit, gaps, claimed)
    if step is None:
        return None
    claimed.add(step)
    return move_command(step)


def guard_weapon(world: World, unit: Unit, claimed: set[Pos]) -> dict | None:
    """站到最近的武器旁边待命（下回合可以直接操控它）

    落脚点按"离自己最近"排序后一次性交给 `step_toward_any`——它会**先试最近
    的那个，走不通才换下一个**。逐格调 A* 的写法在落脚点被队友占着时会为
    每一个失败候选搜遍全图，实测能占掉整回合决策时间的七成。
    """
    towers = sorted(
        world.turn.towers(), key=lambda w: (distance(unit.pos, w.pos), w.pos.x, w.pos.y)
    )
    for tower in towers:
        stands = grid.stand_cells_for(world.turn, tower.pos, unit, busy=claimed)
        if unit.pos in stands:
            return None  # 已经站好了
        if not stands:
            continue
        step = grid.step_toward_any(world.turn, unit, stands, claimed)
        if step is not None:
            claimed.add(step)
            return move_command(step)
    return None
