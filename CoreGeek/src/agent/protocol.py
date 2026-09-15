"""协议层：规则常量、报文解析、指令构造。

对应设计文档V2 第 4 章。

这一层只做两件事：

1. 把判题系统发来的 JSON 报文变成强类型的 `Turn` 快照（接口文档 §1）。
2. 把策略层的意图变成合法的 `RoleCommand`（接口文档 §2）。

**不包含任何策略判断**——"该不该建塔"属于 strategy 层，"建塔指令长什么样"
属于这里。分成两层是因为异常预算（任务书 §8）只关心指令是否合法：
只要所有指令都从这里构造，合法性问题就只需要在这一个文件里保证。

任务的原文里有明确的口径（接口文档 §8 注）：

    异常   = 指令字段缺失或指令无法被识别（move/attack 缺 targetPos、
             use 眩晕法宝/范围炸弹未指定 targetPos、动作码非法）
    执行失败 = 指令合法但规则上没生效（移动碰撞、攻击落点无目标），
             只标记该条无效，不计异常

所以构造器宁可不发指令（返回 None），也不发一条字段不全的指令。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

# === 时间 ===

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS  # 130


# === 队伍与玩家 ===

CHALLENGER = "challenger"
DEFENDER = "defender"


# === 单位类型（接口文档 §1.3.1 roleType）===

STATION = "station"
GATLING = "gatling"
RAILGUN = "railgun"
ROCKET = "rocket"
WALL = "wall"
PIONEER = "pioneer"
WORKER = "worker"

TOWER_TYPES = (GATLING, RAILGUN, ROCKET)
CHARACTER_TYPES = (PIONEER, WORKER)


# === 机器人类型（接口文档 §1.5.1）===

SMALL_ROBOT = "smallRobot"
MIDDLE_ROBOT = "middleRobot"
LARGE_ROBOT = "largeRobot"
BOSS_ROBOT = "bossRobot"


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """机器人属性（任务书 §4.7.2）"""

    health: int
    attack: int
    attack_range: int
    score: int
    threat: int  # 威胁权重：用于目标选择，数值越大越优先


ROBOT_SPECS: dict[str, RobotSpec] = {
    # threat 的取法：攻击力为主，血量做次要权重。BOSS 打基地 40/回合，
    # 大型 20/回合，都比"打角色 5 点"的小型重要得多。
    SMALL_ROBOT: RobotSpec(40, 5, 3, 1, 5),
    MIDDLE_ROBOT: RobotSpec(60, 10, 3, 2, 10),
    LARGE_ROBOT: RobotSpec(500, 20, 3, 4, 30),
    BOSS_ROBOT: RobotSpec(800, 40, 3, 10, 60),
}


# === 建筑满血表（任务书 §4.5.1）===

STATION_MAX_HEALTH = {1: 1500, 2: 3000, 3: 4500}
WALL_MAX_HEALTH = {1: 1000, 2: 1500, 3: 2000}
WEAPON_MAX_HEALTH = {1: 1000, 2: 1500, 3: 2000}

# 角色满血值（任务书 §4.5.2）。角色没有 level 字段，血是固定的；
# 报文里也只有 `health` 没有上限，所以要靠这张表才能算出"剩多少血"。
CHARACTER_MAX_HEALTH = {PIONEER: 200, WORKER: 220}

MAX_LEVEL = 3


# === 武器属性（任务书 §4.5.1 + §4.5.4）===

# (level1, level2, level3)
WEAPON_ATTACK = {
    GATLING: (10, 20, 30),   # 10×等级
    RAILGUN: (10, 20, 30),   # 能量沿弹道穿透，按伤害扣减
    ROCKET: (20, 40, 60),    # 20×等级，中心固定 20？—— 见下方说明
}
WEAPON_RANGE = {
    GATLING: (3, 5, 7),
    RAILGUN: (6, 8, 10),
    ROCKET: (10, 15, 10**9),  # level3 全图
}

# 火箭发射台每枚导弹的伤害（任务书 §4.5.4 第 4 条）：
# "每枚导弹中心伤害固定 20；落点周围 8 格溅射是中心伤害的一半；
#   多枚导弹落点重叠时伤害叠加。"
# 表里 level2/3 的 40/60 是"等级 2 时 20*2"的乘数写法，与"每枚导弹 20、
# 导弹枚数 = 等级"是同一个口径：等级 n 的火箭 = n 枚导弹 × 中心 20。
# 本客户端按后者实现（`rocket_targets` 输出 n 个落点）。
ROCKET_MISSILE_DAMAGE = 20
ROCKET_SPLASH_DAMAGE = 10  # 中心伤害的一半（周围 8 格）

# 每回合攻击目标个数 = 当前武器等级（接口文档 §2.2 targetPos 说明）。
# 电磁狙击炮是唯一的例外：无论等级只传 1 个落点。
ATTACK_TARGET_COUNT_IS_LEVEL = (GATLING, ROCKET)
SINGLE_TARGET_WEAPONS = (RAILGUN,)

# 加特林的多目标必须落在同一 90° 锥内（任务书 §4.5.4）
GATLING_CONE_DEGREES = 90


# === 建造与资源（任务书 §4.5.1 / §4.6.3）===

WEAPON_BUILD_COST = 25  # 金币
WALL_MATERIAL = "stone"  # 围墙每段消耗 1 块石头

STONE_MINE = "stone"
IRON_MINE = "iron"
COPPER_MINE = "copper"


# === 中立元素类型（接口文档 §1.2.1）===

VENDOR = "vendor"
WEAPON_SHOP = "weaponShop"
TASK_POINT_TYPES = {
    CHALLENGER: ("challengerTaskPoint1", "challengerTaskPoint2"),
    DEFENDER: ("defenderTaskPoint1", "defenderTaskPoint2"),
}
LAND = "land"


# === 商店道具（接口文档 §1.1 weaponShopList，docs/request.txt 实证在位）===

WEAPON_UPGRADE_1 = "WeaponUpgradeVoucher1"
WEAPON_UPGRADE_2 = "WeaponUpgradeVoucher2"
WALL_UPGRADE_1 = "WallUpgradeVoucher1"
WALL_UPGRADE_2 = "WallUpgradeVoucher2"
STATION_UPGRADE_1 = "StationUpgradeVoucher1"
STATION_UPGRADE_2 = "StationUpgradeVoucher2"
WALL_FIXER = "WallFixer"
MEDICINE = "Medicine"
DIZZY_WEAPON = "DizzyWeapon"
BOMB = "Bomb"

# 升级券：目标等级 -> 道具名（任务书 §4.6.3）
WEAPON_UPGRADE_VOUCHER = {2: WEAPON_UPGRADE_1, 3: WEAPON_UPGRADE_2}
STATION_UPGRADE_VOUCHER = {2: STATION_UPGRADE_1, 3: STATION_UPGRADE_2}

# 需要指定 targetPos 的 "use" 动作（接口文档 §2.2 的指令错误口径里点名了两件）
USE_NEEDS_POS = frozenset(
    {
        WALL_FIXER,
        DIZZY_WEAPON,
        BOMB,
        WEAPON_UPGRADE_1,
        WEAPON_UPGRADE_2,
        WALL_UPGRADE_1,
        WALL_UPGRADE_2,
        STATION_UPGRADE_1,
        STATION_UPGRADE_2,
    }
)


# === 自进化任务（任务书 §5）===

# 报文没给 `timeoutRounds` 时的兜底（实测值 15，见对战日志）
TASK_DEFAULT_TIMEOUT = 15


# === 错误码（接口文档 §1.7）===
#
# 只保留决策层实际会读的两个：答案错误（要重取数据而不是重交）与任务超时
# （任务已经结束）。其余错误码（未知/网络/指令错误/LLM 额度）在客户端侧没有
# 可执行的动作分支，需要时按接口文档 §1.7 补即可。

ERR_TASK_TIMEOUT = 1
ERR_ANSWER_WRONG = 2


# ==========================================================================
# 坐标与几何
# ==========================================================================


@dataclass(frozen=True, slots=True, order=True)
class Pos:
    """地图坐标（任务书 §4.1：原点在左下角，x 向右、y 向上）"""

    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        return cls(int(raw["x"]), int(raw["y"]))

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    def shifted(self, dx: int, dy: int) -> "Pos":
        return Pos(self.x + dx, self.y + dy)


def distance(a: Pos, b: Pos) -> int:
    """切比雪夫距离（任务书 §4.5.4 第 1 条）"""
    return max(abs(a.x - b.x), abs(a.y - b.y))


NEIGHBOUR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def neighbours(pos: Pos) -> tuple[Pos, ...]:
    """八方向邻居（任务书 §4.5.4 第 2 条）"""
    return tuple(pos.shifted(dx, dy) for dx, dy in NEIGHBOUR_OFFSETS)


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    """基地占据的 2×2 格（接口文档 §1.3.1 注：pos 是左上角）"""
    return (pos, Pos(pos.x + 1, pos.y), Pos(pos.x, pos.y - 1), Pos(pos.x + 1, pos.y - 1))


def footprint_origin(footprint: Iterable[Pos]) -> Pos:
    """footprint 的 (xmin, ymin)，用作相对偏移的原点"""
    cells = tuple(footprint)
    return Pos(min(p.x for p in cells), min(p.y for p in cells))


# ==========================================================================
# 报文实体
# ==========================================================================


@dataclass(frozen=True, slots=True)
class Unit:
    """我方 / 敌方单位（接口文档 §1.3.1）"""

    unit_id: int
    pos: Pos
    kind: str
    health: int
    attack_power: int
    attack_range: int
    level: int
    cooldown: int
    capacity: int
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        return cls(
            unit_id=int(raw.get("id") or 0),
            pos=Pos.load(raw["pos"]),
            kind=str(raw.get("roleType") or ""),
            health=int(raw.get("health") or 0),
            attack_power=int(raw.get("attackPower") or 0),
            attack_range=int(raw.get("attackRange") or 0),
            level=int(raw.get("level") or 1),
            cooldown=int(raw.get("cooldown") or 0),
            capacity=int(raw.get("backPackCapability") or 0),
            backpack=tuple(str(item) for item in raw.get("backpack") or ()),
        )

    # --- 分类 ---

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def is_tower(self) -> bool:
        return self.kind in TOWER_TYPES

    @property
    def is_character(self) -> bool:
        return self.kind in CHARACTER_TYPES

    # --- 属性查询 ---

    def count(self, name: str) -> int:
        return self.backpack.count(name)

    @property
    def backpack_full(self) -> bool:
        return self.capacity > 0 and len(self.backpack) >= self.capacity

    @property
    def free_slots(self) -> int:
        return max(0, self.capacity - len(self.backpack)) if self.capacity else 0

    def max_health(self) -> int | None:
        """满血值；查不到返回 None

        角色用任务书 §4.5.2 的固定值（开拓者 200 / 工人 220），建筑用 §4.5.1
        的等级表。报文里只有 `health` 没有上限，所以缺了这张表就算不出
        "残血"——V1 从来没有判断过角色血量，`Medicine` 与基地升级券因此
        一次都没被用过。
        """
        if self.kind == STATION:
            return STATION_MAX_HEALTH.get(self.level)
        if self.kind == WALL:
            return WALL_MAX_HEALTH.get(self.level)
        if self.kind in TOWER_TYPES:
            return WEAPON_MAX_HEALTH.get(self.level)
        return CHARACTER_MAX_HEALTH.get(self.kind)

    def health_ratio(self) -> float:
        full = self.max_health()
        if not full:
            return 1.0
        return self.health / full

    def attack_points(self) -> int:
        """武器攻击力（按任务书等级表；报文里已有 attackPower 时以报文为准）"""
        if self.attack_power > 0:
            return self.attack_power
        table = WEAPON_ATTACK.get(self.kind)
        if not table:
            return 0
        return table[min(max(self.level, 1), MAX_LEVEL) - 1]

    def range_of_attack(self) -> int:
        """武器攻击距离（同理，报文优先）"""
        if self.attack_range > 0:
            return self.attack_range
        table = WEAPON_RANGE.get(self.kind)
        if not table:
            return 0
        return table[min(max(self.level, 1), MAX_LEVEL) - 1]

    def target_count(self) -> int:
        """本回合 attack 指令必须给出的 targetPos 个数（接口文档 §2.2）"""
        if self.kind in SINGLE_TARGET_WEAPONS:
            return 1
        if self.kind in ATTACK_TARGET_COUNT_IS_LEVEL:
            return min(max(self.level, 1), MAX_LEVEL)
        return 1


@dataclass(frozen=True, slots=True)
class Robot:
    """机器人（接口文档 §1.5.1）"""

    robot_id: int
    pos: Pos
    kind: str
    health: int
    abnormal: str
    target_team: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(
            robot_id=int(raw.get("id") or 0),
            pos=Pos.load(raw["pos"]),
            kind=str(raw.get("roleType") or ""),
            health=int(raw.get("health") or 0),
            abnormal=str(raw.get("abnormalState") or ""),
            target_team=str(raw.get("targetTeam") or ""),
        )

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def is_dizzy(self) -> bool:
        return self.abnormal == "dizzy"

    @property
    def spec(self) -> RobotSpec:
        return ROBOT_SPECS.get(self.kind, RobotSpec(40, 5, 3, 1, 5))

    @property
    def threat(self) -> float:
        """威胁权重：攻击力为主，残血目标略降权（不值得优先补刀）"""
        base = self.spec.threat
        return base * (0.6 + 0.4 * min(1.0, self.health / max(1, self.spec.health)))


@dataclass(frozen=True, slots=True)
class PlayerTask:
    """任务点信息（接口文档 §1.3.2）"""

    task_type: str
    pos: Pos
    cooldown: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "PlayerTask":
        return cls(
            task_type=str(raw.get("taskType") or ""),
            pos=Pos.load(raw["taskPosition"]),
            cooldown=int(raw.get("coldDownRounds") or 0),
            score_reward=int(raw.get("scoreReward") or 0),
            gold_reward=int(raw.get("goldReward") or 0),
            is_valid=bool(raw.get("isValid")),
            timeout_rounds=int(raw.get("timeoutRounds") or TASK_DEFAULT_TIMEOUT),
        )

    @property
    def ready(self) -> bool:
        return self.is_valid and self.cooldown <= 0


@dataclass(frozen=True, slots=True)
class ShopItem:
    """商店条目（接口文档 §1.1 vendorShopList / weaponShopList）"""

    name: str
    price: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "ShopItem":
        return cls(str(raw.get("name") or ""), int(raw.get("price") or 0))


@dataclass(frozen=True, slots=True)
class Error:
    """本轮错误（接口文档 §1.7）"""

    code: int
    description: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Error":
        return cls(int(raw.get("errorCode") or 0), str(raw.get("description") or ""))


# ==========================================================================
# 回合快照
# ==========================================================================


@dataclass(frozen=True, slots=True)
class Turn:
    """一回合的完整快照（接口文档 §1.1）

    无状态：每回合重新解析，不携带任何上一回合的推断结果。跨回合的**学习成果**
    存放在 `strategy.task.memory.Memory` 里，与这里严格分开。
    """

    round_no: int
    is_day: bool
    team_type: str
    team_id: str
    gold: int
    total_score: int
    width: int
    height: int
    zones: dict[Pos, str]
    ours: tuple[Unit, ...]
    enemies: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    player_tasks: tuple[PlayerTask, ...]
    phase_task: str
    last_cmd_result: str
    last_action_results: dict[int, bool]
    last_summon_result: int
    llm_resp: str
    errors: tuple[Error, ...]
    vendor_shop: tuple[ShopItem, ...]
    weapon_shop: tuple[ShopItem, ...]

    # --- 加载 ---

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        round_no = int(payload.get("roundNo") or 1)
        map_info = payload.get("mapInfo") or {}
        team = payload.get("teamOur") or {}
        enemy = payload.get("teamEnemy") or {}
        robot_box = payload.get("robot") or {}
        raw_cmd_result = str(payload.get("lastCmdResult") or "")

        return cls(
            round_no=round_no,
            is_day=(round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
            team_type=str(team.get("type") or ""),
            team_id=str(team.get("teamId") or ""),
            gold=int(team.get("goldNum") or 0),
            total_score=int(team.get("totalScore") or 0),
            width=int(map_info.get("width") or 41),
            height=int(map_info.get("height") or 32),
            zones={
                Pos.load(zone["pos"]): str(zone.get("neutralType") or "")
                for zone in map_info.get("zones") or ()
            },
            ours=tuple(Unit.load(role) for role in team.get("roles") or ()),
            enemies=tuple(Unit.load(role) for role in enemy.get("roles") or ()),
            robots=tuple(Robot.load(r) for r in robot_box.get("roles") or ()),
            player_tasks=tuple(
                PlayerTask.load(t) for t in team.get("playerTasks") or ()
            ),
            phase_task=str(payload.get("phaseTask") or ""),
            last_cmd_result=raw_cmd_result,
            last_action_results={
                int(key): bool(value)
                for key, value in (
                    payload.get("lastRoundRoleActionResults") or {}
                ).items()
                if str(key).lstrip("-").isdigit()
            },
            last_summon_result=int(payload.get("lastSummonTreasureResult") or 0),
            llm_resp=str(payload.get("llmResp") or ""),
            errors=tuple(Error.load(e) for e in payload.get("errors") or ()),
            vendor_shop=tuple(
                ShopItem.load(i) for i in payload.get("vendorShopList") or ()
            ),
            weapon_shop=tuple(
                ShopItem.load(i) for i in payload.get("weaponShopList") or ()
            ),
        )

    # --- 单位查询 ---

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        return tuple(u for u in self.ours if u.kind in kinds and u.is_alive)

    def station(self) -> Unit | None:
        for unit in self.ours:
            if unit.kind == STATION and unit.is_alive:
                return unit
        return None

    def workers(self) -> tuple[Unit, ...]:
        return tuple(
            sorted(self.alive((WORKER,)), key=lambda u: u.unit_id)
        )

    def pioneers(self) -> tuple[Unit, ...]:
        return tuple(self.alive((PIONEER,)))

    def characters(self) -> tuple[Unit, ...]:
        """全部可操控角色，按 ID 排序（ID 固定：worker1 < pioneer < worker2）"""
        return tuple(sorted(self.alive(CHARACTER_TYPES), key=lambda u: u.unit_id))

    def towers(self) -> tuple[Unit, ...]:
        return tuple(
            sorted(self.alive(TOWER_TYPES), key=lambda u: (u.pos.x, u.pos.y))
        )

    def walls(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive((WALL,)), key=lambda u: (u.pos.x, u.pos.y)))

    def alive_robots(self) -> tuple[Robot, ...]:
        return tuple(r for r in self.robots if r.is_alive)

    # --- 地图查询 ---

    def zones_of(self, kind: str) -> tuple[Pos, ...]:
        return tuple(pos for pos, name in self.zones.items() if name == kind)

    def mines(self, kind: str) -> tuple[Pos, ...]:
        return self.zones_of(kind)

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def is_land(self, pos: Pos) -> bool:
        """是否是空地（中立元素、建筑都不算）"""
        if not self.in_bounds(pos):
            return False
        return self.zones.get(pos, LAND) == LAND

    def neutral_at(self, pos: Pos) -> str:
        return self.zones.get(pos, LAND)

    def my_task_points(self) -> tuple[Pos, ...]:
        """己方阵营的 4 个任务点里属于我方的 2 个（任务书 §4.6.2）"""
        kinds = TASK_POINT_TYPES.get(self.team_type, ())
        points: list[Pos] = []
        for kind in kinds:
            points.extend(self.zones_of(kind))
        return tuple(sorted(points))

    def zone_at(self, pos: Pos) -> str | None:
        """该坐标上的中立元素类型（没有则 None）"""
        name = self.zones.get(pos, LAND)
        return None if name == LAND else name

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def occupied(self) -> frozenset[Pos]:
        """己方所有单位占据的格子"""
        cells: set[Pos] = set()
        for unit in self.ours:
            if unit.is_alive:
                cells.update(self.footprint(unit))
        return frozenset(cells)

    def blocked_for(self, moving: Unit) -> frozenset[Pos]:
        """对指定单位不可通行的格子

        任务书 §4.1：己方/敌方建筑、己方/敌方角色、机器人、中立单位、任务点、
        矿区**全部**阻挡移动。所以除了空地以外的一切都算障碍。
        """
        cells = {pos for pos, name in self.zones.items() if name != LAND}
        cells.update(self.occupied())
        cells.discard(moving.pos)
        for enemy in self.enemies:
            if enemy.is_alive:
                cells.update(self.footprint(enemy))
        for robot in self.robots:
            if robot.is_alive:
                cells.add(robot.pos)
        return frozenset(cells)

    # --- 上一回合结果 ---

    def action_ok(self, unit_id: int) -> bool:
        """上一回合该角色的动作是否合法（报文没给就当成合法）

        报文里没有的角色视为合法，防止决策层把"报文没提"误读成"失败了"，
        进而在同一个位置反复重试。
        """
        return self.last_action_results.get(unit_id, True)

    @property
    def answer_wrong(self) -> bool:
        """本轮是否出现了"答案错误"（接口文档 §1.7 errorCode=2）"""
        return any(e.code == ERR_ANSWER_WRONG for e in self.errors)

    @property
    def task_timed_out(self) -> bool:
        return any(e.code == ERR_TASK_TIMEOUT for e in self.errors)

    def shop_price(self, name: str) -> int | None:
        for item in self.weapon_shop:
            if item.name == name:
                return item.price
        return None


# ==========================================================================
# 指令构造
# ==========================================================================
#
# 所有构造器都返回 `dict | None`：
#   - 参数合法 -> 完整指令
#   - 参数非法 -> None（调用方该角色本回合不出指令）
#
# 这是"异常预算"（任务书 §8）的第一道防线：宁可少做一个动作，也不能因为
# 字段缺失或值域非法被记一次异常（累计 5 次就整场停止调度）。


def _pos_list(targets: Iterable[Pos]) -> list[dict[str, int]]:
    return [p.dump() for p in targets]


def move_command(target: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": _pos_list((target,))}


def collect_command(target: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": _pos_list((target,))}


def remove_command(target: Pos) -> dict[str, Any]:
    return {"action": "remove", "targetPos": _pos_list((target,))}


def build_command(target: Pos, name: str) -> dict[str, Any]:
    """建造：`name` 取 roleType（wall / gatling / railgun / rocket）"""
    return {"action": "build", "targetPos": _pos_list((target,)), "name": name}


def attack_command(
    controller_id: int,
    weapon: Unit,
    targets: Iterable[Pos],
) -> dict[str, Any] | None:
    """操控武器攻击

    `targetPos` 的**个数必须等于武器等级**（接口文档 §2.2）：加特林/火箭
    等级 3 就要给 3 个落点，给少了整次攻击非法。落点数不够时用最后一个
    落点补齐——重复落点只是"打同一个地方"，属于指令执行失败而不是异常。
    """
    points = list(targets)
    if not points:
        return None
    need = weapon.target_count()
    while len(points) < need:
        points.append(points[-1])
    points = points[:need]
    return {
        "action": "attack",
        "controllerId": str(controller_id),
        "targetPos": _pos_list(points),
    }


def sell_command(name: str, num: int = 1) -> dict[str, Any] | None:
    if num < 1:
        return None
    return {"action": "sell", "name": name, "num": num}


def buy_command(name: str, num: int = 1) -> dict[str, Any] | None:
    if num < 1:
        return None
    return {"action": "buy", "name": name, "num": num}


def use_command(
    name: str,
    target: Pos | None = None,
    num: int = 1,
) -> dict[str, Any] | None:
    """使用道具

    眩晕法宝 / 范围炸弹 / 各类升级券 / 围墙修复包**必须**带 `targetPos`，
    漏了就是指令错误（接口文档 §2.2 的指令错误口径明确点名了这两件）。
    这里按 `USE_NEEDS_POS` 强制校验，不依赖调用方记得。
    """
    if name in USE_NEEDS_POS and target is None:
        return None
    command: dict[str, Any] = {"action": "use", "name": name}
    if num != 1:
        command["num"] = num
    if target is not None:
        command["targetPos"] = _pos_list((target,))
    return command


def drop_command(name: str) -> dict[str, Any]:
    return {"action": "drop", "name": name}


def accept_task_command() -> dict[str, Any]:
    return {"action": "acceptTask"}


def submit_answer_command(answer: str) -> dict[str, Any] | None:
    """提交答案；空答案不许提交（判题器会当成格式/内容错误）"""
    text = (answer or "").strip()
    if not text:
        return None
    return {"action": "submitAnswer", "taskAnswer": text}


def summon_treasure_command(items: Iterable[str], target: Pos) -> dict[str, Any] | None:
    """献祭任务用品召唤宝藏（任务书 §5.2）"""
    payload = [str(item) for item in items]
    if not payload:
        return None
    return {
        "action": "summonTreasure",
        "targetPos": _pos_list((target,)),
        "item": payload,
    }


# ==========================================================================
# 响应
# ==========================================================================


def build_response(
    role_commands: dict[int, dict[str, Any]],
    prompt: str = "",
    sandbox_command: str = "",
) -> dict[str, Any]:
    """组装响应报文（接口文档 §2.1）

    `roleCommandMap` 的 key 必须是字符串：接口文档写的是 `Map<int, RoleCommand>`，
    JSON 序列化后 key 一律是字符串，判题器按字符串匹配。
    """
    return {
        "roleCommandMap": {
            str(unit_id): command for unit_id, command in role_commands.items()
        },
        "prompt": prompt or "",
        "executeCmd": sandbox_command or "",
    }


EMPTY_RESPONSE: dict[str, Any] = {
    "roleCommandMap": {},
    "prompt": "",
    "executeCmd": "",
}


def dumps(response: dict[str, Any]) -> str:
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))
