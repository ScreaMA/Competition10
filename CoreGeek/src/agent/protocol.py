"""协议模块：数据结构定义，游戏状态解析与指令构建。

对应设计文档 3.3 节。
"""

from dataclasses import dataclass
from typing import Any

# 时间常量
DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS

# 建造成本
WEAPON_BUILD_COST = 25  # 武器建造金币
WALL_MATERIAL = "stone"  # 围墙材料

# 单位类型
LAND = "land"
STATION = "station"
WALL = "wall"
WORKER = "worker"
PIONEER = "pioneer"
GATLING = "gatling"
RAILGUN = "railgun"
ROCKET = "rocket"

# 武器类型
TOWER_TYPES = (GATLING, RAILGUN, ROCKET)
CONTROLLABLE_TYPES = (WORKER, PIONEER)

# 武器射程表（按等级）
TOWER_RANGE_BY_LEVEL = {
    GATLING: (3, 5, 7),
    RAILGUN: (6, 8, 10),
    ROCKET: (10, 15, 10**9),  # level3全图
}

# 武器攻击力表
TOWER_DAMAGE_BY_LEVEL = {
    GATLING: (10, 20, 30),  # level1=10*1, level2=10*2, level3=10*3
    RAILGUN: (10, 20, 30),
    ROCKET: (20, 40, 60),
}

# 建筑满血表（任务书4.5.1节）。报文只给当前 health 与 level，
# 判断"残血"必须靠等级反查满血值；决策（残血用升级券/修复包）与
# 日志（hp=当前/满血）共用这一张表。
STATION_FULL_HEALTH = {1: 1500, 2: 3000, 3: 4500}
WALL_FULL_HEALTH = {1: 1000, 2: 1500, 3: 2000}


def station_full_health(level: int) -> int:
    """基地该等级的满血值（level 越界时退回 level1）"""
    return STATION_FULL_HEALTH.get(level, STATION_FULL_HEALTH[1])


def wall_full_health(level: int) -> int:
    """围墙该等级的满血值（level 越界时退回 level1）"""
    return WALL_FULL_HEALTH.get(level, WALL_FULL_HEALTH[1])

# 矿石类型
STONE_MINE = "stone"
IRON_MINE = "iron"
COPPER_MINE = "copper"

# 中立单位
VENDOR = "vendor"
WEAPON_SHOP = "weaponShop"

# 任务点
CHALLENGER_TASK_1 = "challengerTaskPoint1"
CHALLENGER_TASK_2 = "challengerTaskPoint2"
DEFENDER_TASK_1 = "defenderTaskPoint1"
DEFENDER_TASK_2 = "defenderTaskPoint2"


@dataclass(frozen=True, slots=True)
class Pos:
    """坐标点"""

    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        """从JSON加载"""
        return cls(int(raw["x"]), int(raw["y"]))

    def dump(self) -> dict[str, int]:
        """转为JSON"""
        return {"x": self.x, "y": self.y}

    def __add__(self, other: tuple[int, int]) -> "Pos":
        """支持坐标加法"""
        return Pos(self.x + other[0], self.y + other[1])


def distance(first: Pos, second: Pos) -> int:
    """切比雪夫距离（八方向距离）"""
    return max(abs(first.x - second.x), abs(first.y - second.y))


def manhattan_distance(first: Pos, second: Pos) -> int:
    """曼哈顿距离"""
    return abs(first.x - second.x) + abs(first.y - second.y)


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    """基地占据的2x2区域（pos是左上角）"""
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y - 1),
        Pos(pos.x + 1, pos.y - 1),
    )


@dataclass(frozen=True, slots=True)
class Unit:
    """单位（角色/建筑）"""

    unit_id: int
    pos: Pos
    kind: str  # roleType
    health: int
    level: int
    cooldown: int
    attack_range: int
    attack_power: int
    capacity: int | None  # backPackCapability
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        """从JSON加载"""
        raw_capacity = raw.get("backPackCapability")
        return cls(
            int(raw.get("id") or 0),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw["health"]),
            int(raw.get("level") or 1),
            int(raw.get("cooldown") or 0),
            int(raw.get("attackRange") or 0),
            int(raw.get("attackPower") or 0),
            int(raw_capacity) if raw_capacity is not None else None,
            tuple(str(item) for item in raw.get("backpack") or ()),
        )

    @property
    def backpack_full(self) -> bool:
        """背包是否已满"""
        if self.capacity is None:
            return False
        return len(self.backpack) >= self.capacity

    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.health > 0

    def range_of_attack(self) -> int:
        """获取攻击范围"""
        if self.attack_range > 0:
            return self.attack_range
        # 根据等级查表
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        if table is None:
            return 0
        level = min(max(self.level, 1), len(table))
        return table[level - 1]


@dataclass(frozen=True, slots=True)
class Robot:
    """敌方机器人"""

    robot_id: int
    pos: Pos
    kind: str  # roleType
    health: int
    abnormal_state: str  # dizzy / ""
    target_team: str  # challenger / defender

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        """从JSON加载"""
        return cls(
            int(raw["id"]),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw["health"]),
            str(raw.get("abnormalState") or ""),
            str(raw.get("targetTeam") or ""),
        )

    @property
    def is_dizzy(self) -> bool:
        """是否被眩晕"""
        return self.abnormal_state == "dizzy"

    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.health > 0


@dataclass(frozen=True, slots=True)
class PlayerTask:
    """任务点信息"""

    task_type: str
    task_position: Pos
    cold_down_rounds: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "PlayerTask":
        # 实测请求中部分字段可能缺失（如timeoutRounds），缺失时取默认值，保证不抛异常
        return cls(
            str(raw.get("taskType") or ""),
            Pos.load(raw["taskPosition"]),
            int(raw.get("coldDownRounds") or 0),
            int(raw.get("scoreReward") or 0),
            int(raw.get("goldReward") or 0),
            bool(raw.get("isValid")),
            int(raw.get("timeoutRounds") or 0),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    """回合信息（游戏状态快照）"""

    round_no: int
    is_day: bool
    team_type: str  # challenger / defender
    gold: int
    total_score: int
    width: int
    height: int
    zones: dict[Pos, str]  # 地图中立元素
    ours: tuple[Unit, ...]  # 己方所有单位
    enemies: tuple[Unit, ...]  # 敌方可见单位
    robots: tuple[Robot, ...]  # 机器人
    player_tasks: tuple[PlayerTask, ...]  # 任务点
    phase_task: str  # 当前任务描述
    last_cmd_result: str  # 上回合沙盒命令（executeCmd）的执行结果
    last_action_results: dict[int, bool]  # 上回合各角色动作是否执行成功
    vendor_shop: list[dict[str, Any]]  # 小贩价格表
    weapon_shop: list[dict[str, Any]]  # 武器商店价格表

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        """从请求payload加载"""
        round_no = int(payload["roundNo"])
        map_info = payload["mapInfo"]
        team_our = payload["teamOur"]
        team_enemy = payload.get("teamEnemy") or {}
        robot_data = payload.get("robot") or {}

        return cls(
            round_no=round_no,
            is_day=(round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
            team_type=str(team_our.get("type") or ""),
            gold=int(team_our.get("goldNum") or 0),
            total_score=int(team_our.get("totalScore") or 0),
            width=int(map_info["width"]),
            height=int(map_info["height"]),
            zones={
                Pos.load(zone["pos"]): str(zone["neutralType"])
                for zone in map_info.get("zones") or ()
            },
            ours=tuple(Unit.load(role) for role in team_our.get("roles") or ()),
            enemies=tuple(Unit.load(role) for role in team_enemy.get("roles") or ()),
            robots=tuple(Robot.load(r) for r in robot_data.get("roles") or ()),
            player_tasks=tuple(
                PlayerTask.load(t) for t in team_our.get("playerTasks") or ()
            ),
            phase_task=str(payload.get("phaseTask") or ""),
            last_cmd_result=str(payload.get("lastCmdResult") or ""),
            last_action_results={
                int(role_id): bool(success)
                for role_id, success in (
                    payload.get("lastRoundRoleActionResults") or {}
                ).items()
                if str(role_id).isdigit()
            },
            vendor_shop=payload.get("vendorShopList") or [],
            weapon_shop=payload.get("weaponShopList") or [],
        )

    # === 查询方法 ===

    def station(self) -> Unit | None:
        """获取己方基地"""
        for unit in self.ours:
            if unit.kind == STATION:
                return unit
        return None

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        """获取指定类型的存活单位"""
        return tuple(
            unit for unit in self.ours
            if unit.kind in kinds and unit.is_alive
        )

    def controllable(self) -> tuple[Unit, ...]:
        """获取可控制的角色（工人+开拓者）"""
        return tuple(sorted(
            self.alive(CONTROLLABLE_TYPES),
            key=lambda unit: unit.unit_id,
        ))

    def workers(self) -> tuple[Unit, ...]:
        """获取所有工人"""
        return tuple(sorted(
            self.alive((WORKER,)),
            key=lambda unit: unit.unit_id,
        ))

    def pioneers(self) -> tuple[Unit, ...]:
        """获取所有开拓者"""
        return tuple(sorted(
            self.alive((PIONEER,)),
            key=lambda unit: unit.unit_id,
        ))

    def weapons(self) -> tuple[Unit, ...]:
        """获取所有武器工事（按位置排序）"""
        return tuple(sorted(
            self.alive(TOWER_TYPES),
            key=lambda unit: (unit.pos.x, unit.pos.y),
        ))

    def walls(self) -> tuple[Unit, ...]:
        """获取所有围墙"""
        return self.alive((WALL,))

    def get_mines(self, mine_type: str) -> tuple[Pos, ...]:
        """获取指定类型的矿点"""
        return tuple(
            pos for pos, kind in self.zones.items() if kind == mine_type
        )

    def stone_mines(self) -> tuple[Pos, ...]:
        """获取石矿位置"""
        return self.get_mines(STONE_MINE)

    def iron_mines(self) -> tuple[Pos, ...]:
        """获取铁矿位置"""
        return self.get_mines(IRON_MINE)

    def copper_mines(self) -> tuple[Pos, ...]:
        """获取铜矿位置"""
        return self.get_mines(COPPER_MINE)

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        """获取单位占据的格子"""
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def land(self, pos: Pos) -> bool:
        """判断坐标是否为可通行陆地"""
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.height):
            return False
        return self.zones.get(pos, LAND) == LAND

    def occupied_cells(self) -> frozenset[Pos]:
        """获取己方占据的所有格子"""
        cells: set[Pos] = set()
        for unit in self.ours:
            cells.update(self.footprint(unit))
        return frozenset(cells)

    def blocked(self, moving: Unit) -> frozenset[Pos]:
        """获取对指定单位而言不可通行的格子"""
        cells = {pos for pos, kind in self.zones.items() if kind != LAND}
        cells.update(self.occupied_cells())
        cells.discard(moving.pos)  # 自己所在位置不算阻挡
        # 敌方单位
        for enemy in self.enemies:
            if enemy.is_alive:
                cells.update(self.footprint(enemy))
        # 机器人
        for robot in self.robots:
            if robot.is_alive:
                cells.add(robot.pos)
        return frozenset(cells)

    def alive_robots_targeting_me(self) -> tuple[Robot, ...]:
        """获取攻击我方的存活机器人"""
        return tuple(
            r for r in self.robots
            if r.is_alive and r.target_team == self.team_type
        )

    def action_failed(self, unit_id: int) -> bool:
        """该角色上一回合的动作是否执行失败

        任务书4.5.4节的碰撞规则会让移动/建造失败（目标格被抢占、
        与其他角色争夺同一格等）。报文里没有给出该角色的结果时视为成功，
        免得决策层误以为失败而反复换位置。
        """
        return self.last_action_results.get(unit_id, True) is False


# === 指令构建函数 ===


def move_command(pos: Pos) -> dict[str, Any]:
    """移动指令"""
    return {
        "action": "move",
        "targetPos": [pos.dump()],
    }


def collect_command(pos: Pos) -> dict[str, Any]:
    """采集指令"""
    return {
        "action": "collect",
        "targetPos": [pos.dump()],
    }


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    """建造指令"""
    return {
        "action": "build",
        "targetPos": [pos.dump()],
        "name": name,
    }


def remove_command(pos: Pos) -> dict[str, Any]:
    """拆除围墙指令"""
    return {
        "action": "remove",
        "targetPos": [pos.dump()],
    }


def attack_command(controller_id: int, targets: list[Pos]) -> dict[str, Any]:
    """攻击指令"""
    return {
        "action": "attack",
        "targetPos": [p.dump() for p in targets],
        "controllerId": str(controller_id),
    }


def sell_command(name: str, num: int = 1) -> dict[str, Any]:
    """贩卖指令"""
    return {
        "action": "sell",
        "name": name,
        "num": num,
    }


def buy_command(name: str, num: int = 1) -> dict[str, Any]:
    """购买指令"""
    return {
        "action": "buy",
        "name": name,
        "num": num,
    }


def use_command(name: str, target_pos: Pos | None = None) -> dict[str, Any]:
    """使用物品指令"""
    cmd: dict[str, Any] = {"action": "use", "name": name}
    if target_pos is not None:
        cmd["targetPos"] = [target_pos.dump()]
    return cmd


def drop_command(name: str) -> dict[str, Any]:
    """丢弃物品指令"""
    return {
        "action": "drop",
        "name": name,
    }


def accept_task_command() -> dict[str, Any]:
    """领取任务指令"""
    return {"action": "acceptTask"}


def submit_answer_command(answer: str) -> dict[str, Any]:
    """提交答案指令"""
    return {
        "action": "submitAnswer",
        "taskAnswer": answer,
    }


def summon_treasure_command(pos: Pos, items: list[str]) -> dict[str, Any]:
    """召唤宝藏指令"""
    return {
        "action": "summonTreasure",
        "targetPos": [pos.dump()],
        "item": items,
    }
