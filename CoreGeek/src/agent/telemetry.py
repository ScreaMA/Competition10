"""回合遥测：把逐回合的"增量"算出来，供复盘流水线消费。

对应设计文档V2 §9.3 – §9.4。

对战复盘（GitHub Issues 里的自动分析报告）唯一的输入就是 `debug.log`。
V1 的日志只有"当前状态快照"，复盘要回答"这一夜打死了几个机器人""基地掉血
多少""哪几个角色空转"时只能靠人工比对相邻两行——报告里大量
"日志未覆盖"的结论就是这么来的。

这一层把"相邻两回合的差"算出来：

    kills / station_damage / towers_lost / walls_lost / failed / 空转

它是**唯一**允许保留跨回合战场状态的地方，而且这些状态只服务于日志，
不参与任何决策（决策仍然是无状态的，见 `brain` 模块文档）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import ROBOT_SPECS, ROUNDS_PER_DAY, WEAPON_BUILD_COST, Turn

ROBOT_KINDS = ("smallRobot", "middleRobot", "largeRobot", "bossRobot")
ROBOT_LETTER = {"smallRobot": "s", "middleRobot": "m", "largeRobot": "l", "bossRobot": "b"}

# 连续多少个回合"没钱也没花出去"就打一次冻结告警。
# 复盘里"金币恒 0 达 9+ 回合"是最常见的一类问题，靠人工扫 gold 序列才能发现；
# 这里把它变成日志里显式的一行 `freeze_alert`。
FREEZE_ALERT_STALL = 20


@dataclass(slots=True)
class RoundDelta:
    """本回合相对上一回合的增量"""

    kills: dict[str, int] = field(default_factory=dict)
    kill_score: int = 0
    station_damage: int = 0
    towers_lost: int = 0
    walls_lost: int = 0
    # 上一回合下发、但被判题器判成"执行失败"的指令（`角色ID:动作`）
    failed: list[str] = field(default_factory=list)
    # 金币相对上一回合的变化
    gold_delta: int = 0

    @property
    def total_kills(self) -> int:
        return sum(self.kills.values())

    def kills_text(self) -> str:
        """`kills=23(s10 m8 l4 b1) score=31`"""
        detail = " ".join(
            f"{ROBOT_LETTER[kind]}={self.kills.get(kind, 0)}" for kind in ROBOT_KINDS
        )
        return (
            f"kills={self.total_kills}({detail}) kill_score={self.kill_score}"
        )


class Telemetry:
    """逐回合增量的计算器（模块级单例 `TELEMETRY`）

    只保留**上一回合**的快照：机器人在场表、基地血量、建筑 id 集合，
    外加一份"上一回合给每个角色下了什么动作"（用来把判题器的失败回执翻译成
    `角色ID:动作` 而不只是 ID）。

    回合号不连续（换局、重启）时自动重置，避免把上一局的账算到这一局头上。
    """

    def __init__(self) -> None:
        self.round_no = -1
        self.was_day = True
        self.robots: dict[int, str] = {}
        self.station_health = -1
        self.towers: set[int] = set()
        self.walls: set[int] = set()
        self.actions: dict[int, str] = {}
        self.last_gold = -1
        self.reset_day()

    # --- 重置 ---

    def reset(self) -> None:
        self.__init__()

    def reset_day(self) -> None:
        self.day_kills = 0
        self.day_kill_score = 0
        self.day_damage = 0
        self.day_gold_in = 0
        self.day_gold_out = 0
        self.day_sandbox = 0
        self.day_submits = 0
        self.day_accepts = 0
        self.day_idle = 0
        self.day_commands = 0
        self.day_builds = 0
        self.day_upgrades = 0
        self.day_sells = 0
        self.gold_peak = 0
        self.stall_rounds = 0
        self.freeze_reported = 0

    # --- 主入口 ---

    def observe(self, turn: Turn) -> RoundDelta:
        """在决策**之前**调用：算出本回合相对上一回合的增量"""
        if turn.round_no != self.round_no + 1:
            # 换局或断档：只记快照，不产出增量
            self._snapshot(turn)
            self.reset_day()
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

        if self.last_gold >= 0:
            delta.gold_delta = turn.gold - self.last_gold
            if delta.gold_delta > 0:
                self.day_gold_in += delta.gold_delta
            elif delta.gold_delta < 0:
                self.day_gold_out -= delta.gold_delta

        # 上一回合下发、这一回合被判失败的指令：把动作名翻出来
        for unit_id, ok in turn.last_action_results.items():
            if not ok:
                action = self.actions.get(unit_id, "?")
                delta.failed.append(f"{unit_id}:{action}")

        self.day_kills += delta.total_kills
        self.day_kill_score += delta.kill_score
        self.day_damage += delta.station_damage
        self.gold_peak = max(self.gold_peak, turn.gold)
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

    # --- 事件计数（决策完成后由 brain 调用）---

    def record_round(
        self,
        turn: Turn,
        commands: dict[int, dict],
        *,
        spend: int,
        idle_units: list[int],
        sandbox: bool,
        submissions: int,
        accepts: int,
    ) -> None:
        """把本回合的账记进当日总账，并维护"金币停滞"计数"""
        self.actions = {
            unit_id: str(command.get("action") or "?")
            for unit_id, command in commands.items()
        }
        for command in commands.values():
            action = command.get("action")
            if action == "build":
                self.day_builds += 1
            elif action == "sell":
                self.day_sells += 1
            elif action == "use":
                self.day_upgrades += 1
        self.day_commands += len(commands)
        self.day_idle += len(idle_units)
        self.day_sandbox += 1 if sandbox else 0
        self.day_submits += submissions
        self.day_accepts += accepts

        # 金币停滞：连续"没钱也没花出去"的回合数。
        # 这是复盘里"金币冻结 9+ 回合"那条结论的自动版——不用人工扫 gold 序列。
        if turn.gold < WEAPON_BUILD_COST and spend <= 0:
            self.stall_rounds += 1
        else:
            self.stall_rounds = 0
            self.freeze_reported = 0

    def freeze_alert(self, turn: Turn, bag: str) -> str:
        """返回冻结告警行（不到阈值时返回空串）

        每累积 `FREEZE_ALERT_STALL` 个停滞回合报一次，不在同一个停滞期里刷屏。
        """
        if self.stall_rounds < FREEZE_ALERT_STALL:
            return ""
        if self.stall_rounds // FREEZE_ALERT_STALL <= self.freeze_reported:
            return ""
        self.freeze_reported = self.stall_rounds // FREEZE_ALERT_STALL
        return (
            f"freeze_alert round={turn.round_no} gold={turn.gold} "
            f"stalled_rounds={self.stall_rounds} bag={bag or '-'}"
        )

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
            f" gold_peak={self.gold_peak}"
            f" gold_in={self.day_gold_in} gold_out={self.day_gold_out}"
            f" build={self.day_builds} upgrade={self.day_upgrades}"
            f" sell={self.day_sells}"
            f" task_accept={self.day_accepts} task_submit={self.day_submits}"
            f" sandbox={self.day_sandbox}"
            f" commands={self.day_commands} idle_units={self.day_idle}"
        )

    def roll_day(self) -> None:
        """结完一天的账后清空当日累计（保留跨日的停滞计数）"""
        stall, reported = self.stall_rounds, self.freeze_reported
        self.reset_day()
        self.stall_rounds, self.freeze_reported = stall, reported


# 模块级单例：遥测状态活在整个客户端进程里
TELEMETRY = Telemetry()
