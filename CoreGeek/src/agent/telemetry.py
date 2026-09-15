"""回合遥测：把逐回合的"增量"算出来，供复盘流水线消费。

对应设计文档V2 §9.3 – §9.4。

对战复盘（GitHub Issues 里的自动分析报告）唯一的输入就是 `debug.log`。
V1 的日志只有"当前状态快照"，复盘要回答"这一夜打死了几个机器人""基地掉血
多少""哪几个角色空转"时只能靠人工比对相邻两行——报告里大量
"日志未覆盖"的结论就是这么来的。

这一层把"相邻两回合的差"算出来：

    kills / lost / station_damage / idle_units / towers_lost / walls_lost

它是**唯一**允许保留跨回合战场状态的地方，而且这些状态只服务于日志，
不参与任何决策（决策仍然是无状态的，见 `brain` 模块文档）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import ROBOT_SPECS, ROUNDS_PER_DAY, Turn


@dataclass(slots=True)
class RoundDelta:
    """本回合相对上一回合的增量"""

    kills: dict[str, int] = field(default_factory=dict)
    kill_score: int = 0
    station_damage: int = 0
    towers_lost: int = 0
    walls_lost: int = 0

    @property
    def total_kills(self) -> int:
        return sum(self.kills.values())

    def text(self) -> str:
        parts = [f"{kind[:1]}={count}" for kind, count in sorted(self.kills.items())]
        return f"kills={self.total_kills}({' '.join(parts) or '-'}) score={self.kill_score}"


class Telemetry:
    """逐回合增量的计算器（模块级单例 `TELEMETRY`）

    只保留**上一回合**的快照：机器人在场表、基地血量、建筑 id 集合。
    回合号不连续（换局、重启）时自动重置，避免把上一局的账算到这一局头上。
    """

    def __init__(self) -> None:
        self.round_no = -1
        self.was_day = True
        self.robots: dict[int, str] = {}
        self.station_health = -1
        self.towers: set[int] = set()
        self.walls: set[int] = set()
        # 当日累计（每个游戏日第一条 `day_summary` 重置）
        self.day_kills = 0
        self.day_kill_score = 0
        self.day_damage = 0
        self.day_gold_in = 0
        self.day_gold_out = 0
        self.day_sandbox = 0
        self.day_submits = 0
        self.day_idle = 0
        self.last_gold = -1

    # --- 重置 ---

    def reset(self) -> None:
        self.__init__()

    # --- 主入口 ---

    def observe(self, turn: Turn) -> RoundDelta:
        """在决策**之前**调用：算出本回合相对上一回合的增量"""
        if turn.round_no != self.round_no + 1:
            # 换局或断档：只记快照，不产出增量
            self._snapshot(turn)
            self._reset_day()
            self.last_gold = turn.gold
            return RoundDelta()

        delta = RoundDelta()
        current = {r.robot_id: r.kind for r in turn.alive_robots()}

        # 机器人消失 = 被击杀。但**天亮清场**除外：夜里活到早上是被系统清掉的，
        # 不是我们打死的（任务书 §4.7.3）。
        dawn = turn.is_day and not self.was_day
        if not dawn:
            for robot_id, kind in self.robots.items():
                if robot_id not in current:
                    delta.kills[kind] = delta.kills.get(kind, 0) + 1
                    spec = ROBOT_SPECS.get(kind)
                    delta.kill_score += spec.score if spec else 0

        station = turn.station()
        if station is not None and self.station_health >= 0:
            delta.station_damage = max(0, self.station_health - station.health)

        tower_ids = {u.unit_id for u in turn.towers()}
        wall_ids = {u.unit_id for u in turn.walls()}
        delta.towers_lost = len(self.towers - tower_ids)
        delta.walls_lost = len(self.walls - wall_ids)

        if turn.gold > self.last_gold >= 0:
            self.day_gold_in += turn.gold - self.last_gold
        elif 0 <= turn.gold < self.last_gold:
            self.day_gold_out += self.last_gold - turn.gold

        self.day_kills += delta.total_kills
        self.day_kill_score += delta.kill_score
        self.day_damage += delta.station_damage
        self.last_gold = turn.gold
        self._snapshot(turn)
        return delta

    # --- 内部 ---

    def _snapshot(self, turn: Turn) -> None:
        self.round_no = turn.round_no
        self.was_day = turn.is_day
        self.robots = {r.robot_id: r.kind for r in turn.alive_robots()}
        station = turn.station()
        self.station_health = station.health if station else -1
        self.towers = {u.unit_id for u in turn.towers()}
        self.walls = {u.unit_id for u in turn.walls()}
        self.last_gold = turn.gold

    def _reset_day(self) -> None:
        self.day_kills = 0
        self.day_kill_score = 0
        self.day_damage = 0
        self.day_gold_in = 0
        self.day_gold_out = 0
        self.day_sandbox = 0
        self.day_submits = 0
        self.day_idle = 0

    # --- 事件计数（由 brain 调用）---

    def count_submit(self) -> None:
        self.day_submits += 1

    def count_sandbox(self) -> None:
        self.day_sandbox += 1

    def count_idle(self, count: int) -> None:
        self.day_idle += count

    # --- 汇总文本 ---

    def is_new_day(self, turn: Turn) -> bool:
        """是否到了新游戏日的第一回合（每个白天的第一个回合）"""
        return turn.round_no > 1 and (turn.round_no - 1) % ROUNDS_PER_DAY == 0

    def day_summary(self, turn: Turn) -> str:
        """刚结束的那一天的账

        这条日志是在**新一天的第一个回合**打的（`is_new_day`），所以统计的是
        第 `index` 天（新一天是第 index+1 天），区间是
        `[(index-1)*130+1, index*130]`。
        """
        index = (turn.round_no - 1) // ROUNDS_PER_DAY
        start = (index - 1) * ROUNDS_PER_DAY + 1
        end = index * ROUNDS_PER_DAY
        return (
            f"day={index} rounds={start}-{end}"
            f" kills={self.day_kills} kill_score={self.day_kill_score}"
            f" station_damage={self.day_damage}"
            f" gold_in={self.day_gold_in} gold_out={self.day_gold_out}"
            f" submits={self.day_submits} sandbox={self.day_sandbox}"
            f" idle_rounds={self.day_idle}"
        )

    def roll_day(self) -> None:
        self._reset_day()


# 模块级单例：遥测状态活在整个客户端进程里
TELEMETRY = Telemetry()
