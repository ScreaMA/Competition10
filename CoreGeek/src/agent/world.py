"""世界模型：由 `Turn` 派生的决策视图。

对应设计文档V2 第 5 章。

职责：

1. 把"基地在哪、要来敌人从哪边来"这类**每回合可重算**的信息算出来（`World`）。
2. 维护**唯一**的跨回合学习成果——可建造区（`BuildableZone`）。接口报文里
   没有任何字段说明哪些格子能建造（任务书只给了"蓝区只能建武器、黄区只能建
   围墙"这条规则），所以只能在线学：拿真实 `build` 指令的执行结果当标签，
   把结果记成**相对基地 footprint 的偏移**，换边（上下半场互换位置）也能复用。

学习结果只增不减，且只在确信"失败是格子本身导致的"时才写入否定标签——
见 `BuildAttempt.confident_failure`。
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import (
    CHALLENGER,
    LAND,
    Pos,
    TOWER_TYPES,
    Turn,
    WALL,
    WEAPON_BUILD_COST,
    WALL_MATERIAL,
    distance,
    footprint_origin,
    neighbours,
    station_footprint,
)

#: 四个方位的固定次序。**只是同分时的兜底排序，不是「来敌方向」**——
#: 真正的来向由 `defence_order()` 每回合从报文里推（敌基地全图可见）。
SIDE_ORDER = ("up", "down", "left", "right")

#: 四个方位的单位向量（算"哪一面离来敌方向更近"用）
_DIRECTION_VECTORS = {"up": (0, 1), "down": (0, -1),
                      "left": (-1, 0), "right": (1, 0)}


# ==========================================================================
# 可建造区（在线学习）
# ==========================================================================


@dataclass(frozen=True, slots=True)
class BuildAttempt:
    """一次已下发、等待结果的建造动作

    保留足够的上下文，才能在下一回合区分"格子不允许建"和"这次没建成"
    （目标格被抢、金币不够、距离不对都会让 build 失败，但这跟格子能不能建无关）。
    """

    target: Pos
    name: str
    affordable: bool  # 金币/材料在计划时是够的
    adjacent: bool  # 建造者当时确实在目标格周围一格内
    cell_free: bool  # 目标格在计划时是空地且没人占

    @property
    def is_wall(self) -> bool:
        return self.name == WALL

    @property
    def confident_failure(self) -> bool:
        """这次失败是否**确信**由"格子不可建造"造成

        四个条件都满足时，才把该偏移标成不可建造。任何一个不满足就丢弃这次
        观测——误标一个可建造格为不可建，代价是那座塔/那段墙永远不会再被考虑，
        比少学一条记录严重得多。
        """
        return self.affordable and self.adjacent and self.cell_free


class BuildableZone:
    """以基地 footprint 的 (xmin, ymin) 为原点的可建造区模式

    两张表分别记录武器区与围墙区的**已知结论**：

        True  -> 在这个偏移上成功建成过
        False -> 在这个偏移上以"干净的尝试"失败过（见 `confident_failure`）

    查不到的偏移 = 未知，交给 `probe_order` 按由近到远的顺序试探。
    """

    def __init__(self) -> None:
        self.weapon: dict[Pos, bool] = {}
        self.wall: dict[Pos, bool] = {}

    # --- 学习 ---

    def observe(self, attempt: BuildAttempt, success: bool) -> None:
        """写入一次观测；`attempt.target` 必须已经是**相对基地原点**的偏移"""
        if success:
            self._table(attempt.name)[attempt.target] = True
        elif attempt.confident_failure:
            self._table(attempt.name)[attempt.target] = False

    def _table(self, name: str) -> dict[Pos, bool]:
        return self.wall if name == WALL else self.weapon

    # --- 查询 ---

    def known(self, name: str, offset: Pos) -> bool | None:
        return self._table(name).get(offset)

    def allows(self, name: str, offset: Pos) -> bool:
        """是否**确定**可以建（未知一律返回 False，由探测流程接手）"""
        return self._table(name).get(offset) is True

    def forbidden(self, name: str, offset: Pos) -> bool:
        return self._table(name).get(offset) is False

    def summary(self) -> str:
        def fmt(table: dict[Pos, bool]) -> str:
            yes = sum(1 for v in table.values() if v)
            no = len(table) - yes
            return f"{yes}/{no}"

        return f"weapon={fmt(self.weapon)} wall={fmt(self.wall)}"


# 模块级单例：学习成果活在整个客户端进程里（覆盖一场比赛的两个半场）。
# 用模块级而不是挂在 World 上，是因为 World 每回合重建。
_ZONE = BuildableZone()

# 上一回合下发的建造动作：unit_id -> BuildAttempt
_PENDING: dict[int, BuildAttempt] = {}

# 本回合已经登记过建造的 unit_id（防止同回合重复登记）
_ISSUED: set[int] = set()

def zone() -> BuildableZone:
    return _ZONE


def reset_learning() -> None:
    """清空学习成果（仅供测试使用）

    就地清空而不是重新构造 `_ZONE`：持有 `zone()` 返回值的调用方（如 `defense`
    里循环里的局部变量）不会因为一次 reset 而拿着旧对象继续写。
    """
    _ZONE.weapon.clear()
    _ZONE.wall.clear()
    _PENDING.clear()
    _ISSUED.clear()


def _relative(origin: Pos, pos: Pos) -> Pos:
    return Pos(pos.x - origin.x, pos.y - origin.y)


def origin_of(turn: Turn) -> Pos:
    """基地 footprint 的原点；没有基地时退回地图中心（决策会因此退化，但不会崩）"""
    station = turn.station()
    if station is None:
        return Pos(turn.width // 2, turn.height // 2)
    return footprint_origin(station_footprint(station.pos))


def absorb_results(turn: Turn) -> list[str]:
    """把上一回合建造动作的执行结果并进可建造区

    必须在决策**之前**调用一次。返回给日志用的人类可读摘要。
    """
    notes: list[str] = []
    origin = origin_of(turn)
    for unit_id, attempt in _PENDING.items():
        if unit_id not in turn.last_action_results:
            continue  # 报文没提这个角色，这一回合不学习（避免误判）
        success = turn.last_action_results[unit_id]
        offset = _relative(origin, attempt.target)
        before = _ZONE.known(attempt.name, offset)
        _ZONE.observe(
            BuildAttempt(
                target=offset,
                name=attempt.name,
                affordable=attempt.affordable,
                adjacent=attempt.adjacent,
                cell_free=attempt.cell_free,
            ),
            success,
        )
        after = _ZONE.known(attempt.name, offset)
        if before is None and after is not None:
            label = "wall" if attempt.name == WALL else "weapon"
            notes.append(f"{label}@{offset.x},{offset.y}={'ok' if after else 'no'}")
    _PENDING.clear()
    _ISSUED.clear()
    return notes


def record_build(turn: Turn, unit_id: int, target: Pos, name: str) -> None:
    """登记本回合下发的建造动作，下一回合用它的执行结果学习"""
    if unit_id in _ISSUED:
        return
    unit = next((u for u in turn.ours if u.unit_id == unit_id), None)
    if unit is None:
        return
    if name == WALL:
        affordable = unit.count(WALL_MATERIAL) >= 1
    else:
        affordable = turn.gold >= WEAPON_BUILD_COST
    _PENDING[unit_id] = BuildAttempt(
        target=target,
        name=name,
        affordable=affordable,
        adjacent=distance(unit.pos, target) <= 1 and unit.pos != target,
        cell_free=turn.is_land(target) and target not in turn.occupied(),
    )
    _ISSUED.add(unit_id)


# ==========================================================================
# 世界
# ==========================================================================


@dataclass(frozen=True, slots=True)
class World:
    """一个回合的决策视图

    只做**无争议的派生**：基地原点、敌方来向、各类格子的绝对坐标换算。
    所有策略判断留在 `strategy/` 里。
    """

    turn: Turn
    origin: Pos  # 基地 footprint 原点

    @classmethod
    def load(cls, turn: Turn) -> "World":
        return cls(turn=turn, origin=origin_of(turn))

    # --- 坐标换算 ---

    def offset_of(self, pos: Pos) -> Pos:
        return Pos(pos.x - self.origin.x, pos.y - self.origin.y)

    def absolute_of(self, offset: Pos) -> Pos:
        return Pos(self.origin.x + offset.x, self.origin.y + offset.y)

    @property
    def station_pos(self) -> Pos:
        station = self.turn.station()
        return station.pos if station else self.origin

    # --- 敌方来向 ---

    def enemy_sides(self) -> frozenset[str]:
        """敌方主力大致从基地的哪几个方位来

        优先用**可见的敌方基地**（接口文档 §1.4：敌方基地与围墙全图可见），
        看不到就退回最近的可见敌方单位。两者都没有时返回空集——此时不做任何
        方位假设，策略退回"朝地图中心"的保守选择。

        设计文档V2 §11.3：不对不可见信息做推断。
        """
        station = self.turn.station()
        if station is None:
            return frozenset()

        visible = [u for u in self.turn.enemies if u.is_alive]
        bases = [u for u in visible if u.kind == "station"]
        candidates = bases or visible
        if not candidates:
            return frozenset()

        target = min(
            candidates,
            key=lambda u: (distance(u.pos, station.pos), u.pos.x, u.pos.y),
        )
        return self.sides_between(station.pos, target.pos)

    @staticmethod
    def sides_between(origin: Pos, target: Pos) -> frozenset[str]:
        dx = target.x - origin.x
        dy = target.y - origin.y
        sides: set[str] = set()
        if dx < 0:
            sides.add("left")
        elif dx > 0:
            sides.add("right")
        if dy < 0:
            sides.add("down")
        elif dy > 0:
            sides.add("up")
        return frozenset(sides) if sides else frozenset({"left", "right"})

    def enemy_vector(self) -> tuple[int, int] | None:
        """敌方来向的**原始向量**（我方基地 → 最近的可见敌方单位）

        优先用可见的敌方基地（接口文档 §1.4：敌基地全图可见），看不到就退回
        最近的可见敌方单位；都没有时返回 None——不做方位假设。
        """
        station = self.turn.station()
        if station is None:
            return None
        visible = [u for u in self.turn.enemies if u.is_alive]
        candidates = [u for u in visible if u.kind == "station"] or visible
        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda u: (distance(u.pos, station.pos), u.pos.x, u.pos.y),
        )
        return (target.pos.x - station.pos.x, target.pos.y - station.pos.y)

    def defence_ranks(self) -> dict[str, int]:
        """每个方位的**布防名次**（越小越该优先布防；**同分的方位并列**）

        名次由方位向量与敌我连线的点积决定：

            正对来敌的那一面  >  两翼（垂直方向）  >  **背面**

        两个出生基地在地图对角（左上 vs 右下），敌基地**几乎总是斜的**，横竖
        两维都会"命中"。按命中集合排序（旧实现）有两个后果：

          - **斜向时**谁排前面由切比雪夫环数、坐标这些与敌情无关的因素决定——
            实测敌基地 `(20,24)`、我方 `(30,10)`（dx=-10 dy=+14，"上"是主轴）
            时，塔位压到了左边；
          - **正上/正下时**，"下"排在左右两翼**之前**——八段墙里三段砌在了
            背对敌人的那一面。

        **并列是必须的，不能靠固定次序把两翼分出先后。** "正对来敌"和"两翼"
        之间是质的差别，"左翼"和"右翼"之间没有——分先后会让围墙全砌到同一侧：
        实测敌人在正上方时，八段墙里五段砌在左翼、右翼一段没有。

        看不到任何敌方单位时退回"朝地图中心"的保守选择（设计文档V2 §11.3：
        不对不可见信息做推断）。**方向每回合从报文里推、不写死**：换边之后
        同一份代码自动镜像。
        """
        vector = self.enemy_vector() or _DIRECTION_VECTORS[self.map_center_side()]
        vx, vy = vector
        dots = {
            side: sx * vx + sy * vy
            for side, (sx, sy) in _DIRECTION_VECTORS.items()
        }
        # 按点积从大到小排名次；点积相同 ⇒ 名次相同（两翼并列）
        levels = sorted(set(dots.values()), reverse=True)
        return {side: levels.index(dot) for side, dot in dots.items()}

    def defence_order(self) -> tuple[str, ...]:
        """布防顺序（名次 + 固定次序兜底，给需要序列的调用方）"""
        ranks = self.defence_ranks()
        return tuple(sorted(SIDE_ORDER, key=lambda s: (ranks[s], SIDE_ORDER.index(s))))

    def map_center_side(self) -> str:
        """看不到任何敌方单位时的保守来向：朝地图中心的那一边"""
        return self.closer_side_to(
            self.turn.station().pos if self.turn.station() else self.origin,
            Pos(self.turn.width // 2, self.turn.height // 2),
        )

    def closer_side_to(self, origin: Pos, target: Pos) -> str:
        dx = target.x - origin.x
        dy = target.y - origin.y
        if abs(dx) >= abs(dy):
            return "right" if dx >= 0 else "left"
        return "up" if dy >= 0 else "down"

    # --- 常用集合 ---

    def my_task_cells(self) -> tuple[Pos, ...]:
        return self.turn.my_task_points()

    @property
    def is_early_game(self) -> bool:
        return self.turn.round_no <= 5

    @property
    def team_is_challenger(self) -> bool:
        return self.turn.team_type == CHALLENGER

    def neighbours_land(self, pos: Pos) -> tuple[Pos, ...]:
        return tuple(p for p in neighbours(pos) if self.turn.is_land(p))

    def buildable_now(self, pos: Pos) -> bool:
        """该格此刻是不是空地、且没有被任何单位占据（可以被建造）"""
        return self.turn.is_land(pos) and pos not in self.turn.occupied()

    def zone(self) -> BuildableZone:
        """在线学习出来的可建造区（`strategy/defense` 用它筛候选格）"""
        return _ZONE

    def weapon_count(self) -> int:
        return len(self.turn.towers())

    def wall_count(self) -> int:
        return len(self.turn.walls())
