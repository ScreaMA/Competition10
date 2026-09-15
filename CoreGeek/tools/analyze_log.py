#!/usr/bin/env python3
"""分析 debug.log，统计回合处理情况并渲染复盘模板。

用法:
    python CoreGeek/tools/analyze_log.py                     # 回合统计摘要
    python CoreGeek/tools/analyze_log.py --template          # 渲染日志分析模板 V2
    python CoreGeek/tools/analyze_log.py --issue             # 渲染 Issue 总结模板 V2
    python CoreGeek/tools/analyze_log.py --task              # 只看自进化任务链路
    python CoreGeek/tools/analyze_log.py --full              # 任务全量日志（原文，还原成多行）
    python CoreGeek/tools/analyze_log.py --rounds 1-130      # 只看某个回合区间
    python CoreGeek/tools/analyze_log.py --out report.md     # 写到文件

对应设计文档V2 第 9 章。

`debug.log` 每回合固定三行 + 事件行，全部**单行、`key=value` 可解析**：

    request_decoded round=… day=… tod=… gold=… hp=… towers=… walls=… robots=…
                    enemy_visible=… mines=… tasks=[…] phase=… task=… plan=…
                    chars=… bag=… zone=…
    strategy_done   round=… commands=… elapsed=… gold_spent=… actions=…
                    fail=[…] sandbox=… note=… learn=…
    round_end       round=… kills=…(…) kill_score=… station_damage=…
                    towers_lost=… walls_lost=… weapons=… manned=…
                    idle_weapon=… idle_target=… commands=… idle_units=…
    task_event      round=… event=… key=… family=… step=… rounds=… left=… detail=…
    day_summary     day=… rounds=… kills=… gold_peak=… build=… task_submit=…
    freeze_alert    round=… gold=… stalled_rounds=… bag=…

`--template` / `--issue` 会把能自动算的格子直接填上，算不出来的留 `{待人工}`。
`--issue` 还会**留出一段"日志原文"的空位**——那是分析师（或 LLM）判断根因时
唯一可信的依据，位号比结论重要。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "CoreGeek" / "debug.log"
TEMPLATE_HINT = ROOT / "战术参考" / "日志分析模板V2.md"
ISSUE_HINT = ROOT / "战术参考" / "Issue总结模板V2.md"

PENDING = "{待人工}"

KV_RE = re.compile(r"([A-Za-z_][\w.]*)=(\S+)")
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b")
HP_RE = re.compile(r"^(\d+)/(\d+)$")
TOWERS_RE = re.compile(r"^(\d+)\[(.*)\]$")
TOWER_ITEM_RE = re.compile(r"^([a-zA-Z]+)(\d+)@(-?\d+),(-?\d+)$")
WALLS_RE = re.compile(r"^(\d+)\[(.*)\]$")
WALL_ITEM_RE = re.compile(r"^l(\d+):(\d+)$")
ACTIONS_RE = re.compile(r"(?=\d+:)")
# `neutral=stone:6[29,14 12,3],vendor:1[21,15]` —— 值里有空格，要用 `_field_value`
NEUTRAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*):(\d+)\[([^\]]*)\]")
POS_RE = re.compile(r"(-?\d+),(-?\d+)")
BASE_RE = re.compile(r"\((-?\d+),(-?\d+)\)")

# 中立元素的中文名（只有这几类值得单独点名，其余按原名显示）
NEUTRAL_NAMES = {
    "stone": "石矿", "iron": "铁矿", "copper": "铜矿",
    "vendor": "小贩", "weaponShop": "武器商店",
}

TOWER_NAMES = {"gatling": "加特林", "railgun": "电磁", "rocket": "火箭"}
# ==========================================================================
# 解析
# ==========================================================================


@dataclass
class Round:
    """一个回合的原始记录"""

    round_no: int
    state: dict[str, str] = field(default_factory=dict)      # request_decoded
    decision: dict[str, str] = field(default_factory=dict)   # strategy_done
    outcome: dict[str, str] = field(default_factory=dict)    # round_end
    events: list[dict[str, str]] = field(default_factory=list)  # task_event


@dataclass
class Stats:
    """跨回合汇总"""

    rounds: dict[int, Round] = field(default_factory=dict)
    day_lines: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 逐回合序列
    gold_series: list[tuple[int, int]] = field(default_factory=list)
    hp_series: list[tuple[int, int, int]] = field(default_factory=list)  # (回合, 当前, 满血)
    score_series: list[tuple[int, int]] = field(default_factory=list)

    # 战斗
    kills: int = 0
    kill_score: int = 0
    damage: int = 0
    towers_lost: int = 0
    walls_lost: int = 0
    idle_units: int = 0
    night_rounds: int = 0

    # 经济
    gold_in: int = 0
    gold_out: int = 0
    builds: int = 0
    sells: int = 0
    upgrades: int = 0
    spends: list[tuple[int, int]] = field(default_factory=list)  # (round, gold_spent)

    # 任务
    task_events: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    accepted: int = 0
    submitted: int = 0
    sandbox_rounds: int = 0

    # 时间线（首次出现的事件）
    events: list[tuple[int, str]] = field(default_factory=list)

    # 任务全量转储：[(回合, kind, 原文)]，来自 DEBUG 级的 `task_dump` 行
    dumps: list[tuple[int, str, str]] = field(default_factory=list)

    # --- 汇总属性 ---

    @property
    def ordered(self) -> list[int]:
        return sorted(self.rounds)

    @property
    def first_round(self) -> int:
        return self.ordered[0] if self.rounds else 0

    @property
    def last_round(self) -> int:
        return self.ordered[-1] if self.rounds else 0

    @property
    def last(self) -> Round | None:
        return self.rounds[self.last_round] if self.rounds else None

    @property
    def team_type(self) -> str:
        return self.last.state.get("team", "") if self.last else ""

    @property
    def team_id(self) -> str:
        return self.last.state.get("team_id", "") if self.last else ""

    @property
    def gold_peak(self) -> int:
        return max((gold for _, gold in self.gold_series), default=0)

    @property
    def gold_final(self) -> int:
        return self.gold_series[-1][1] if self.gold_series else 0

    @property
    def score_final(self) -> int:
        return self.score_series[-1][1] if self.score_series else 0

    @property
    def hp(self) -> tuple[str, str]:
        if not self.hp_series:
            return PENDING, PENDING
        _, now, full = self.hp_series[-1]
        return str(now), str(full)

    @property
    def idle_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        return self.idle_units / (len(self.rounds) * 3)  # 三个可操控角色

    def towers(self) -> list[dict[str, str]]:
        """当前塔明细：[{kind, level, pos}]"""
        if self.last is None:
            return []
        match = TOWERS_RE.match(self.last.state.get("towers", ""))
        if not match:
            return []
        out = []
        for item in match.group(2).split():
            parsed = TOWER_ITEM_RE.match(item)
            if parsed:
                out.append({
                    "kind": parsed.group(1),
                    "level": parsed.group(2),
                    "pos": f"{parsed.group(3)},{parsed.group(4)}",
                })
        return out

    def tower_timeline(self) -> list[tuple[int, str]]:
        """塔的建成顺序（第一次出现的回合）"""
        seen: dict[str, int] = {}
        for round_no in self.ordered:
            state = self.rounds[round_no].state
            match = TOWERS_RE.match(state.get("towers", ""))
            if not match:
                continue
            for item in match.group(2).split():
                parsed = TOWER_ITEM_RE.match(item)
                if not parsed:
                    continue
                label = f"{parsed.group(1)}@{parsed.group(3)},{parsed.group(4)}"
                seen.setdefault(label, round_no)
        return sorted(seen.items(), key=lambda kv: kv[1])

    def wall_timeline(self) -> list[tuple[int, int]]:
        """(回合, 围墙段数)，只保留段数变化的那些回合"""
        out: list[tuple[int, int]] = []
        previous = -1
        for round_no in self.ordered:
            match = WALLS_RE.match(self.rounds[round_no].state.get("walls", ""))
            if not match:
                continue
            count = int(match.group(1))
            if count != previous:
                out.append((round_no, count))
                previous = count
        return out

    def walls_now(self) -> str:
        if self.last is None:
            return "0"
        match = WALLS_RE.match(self.last.state.get("walls", ""))
        if not match:
            return "0"
        detail = []
        for item in match.group(2).split():
            parsed = WALL_ITEM_RE.match(item)
            if parsed:
                detail.append(f"L{parsed.group(1)}×{parsed.group(2)}")
        return f"{match.group(1)}段（{' / '.join(detail) or '-'}）"

    def enemies_now(self) -> str:
        if self.last is None:
            return PENDING
        state = self.last.state
        return (
            f"可见{state.get('enemy_visible', '?')}个单位 / "
            f"塔{state.get('enemy_towers', '?')}座 / 墙{state.get('enemy_walls', '?')}段"
        )

    def base_pos(self) -> tuple[int, int] | None:
        """基地 footprint 原点坐标（`base=(x,y)`）"""
        if self.last is None:
            return None
        match = BASE_RE.match(self.last.state.get("base", ""))
        return (int(match.group(1)), int(match.group(2))) if match else None

    def neutral_at(self, round_no: int) -> dict[str, list[tuple[int, int]]]:
        """某个回合的中立元素坐标：类型 -> [(x, y), …]

        `neutral=` 的值里有空格（`stone:6[29,14 12,3]`），必须走
        `_field_value` 取完整值——`_fields` 用的 `\\S+` 只会截到第一个坐标。
        """
        record = self.rounds.get(round_no)
        if record is None:
            return {}
        out: dict[str, list[tuple[int, int]]] = {}
        for name, _count, body in NEUTRAL_RE.findall(record.state.get("neutral", "")):
            out[name] = [(int(x), int(y)) for x, y in POS_RE.findall(body)]
        return out

    def economy_diagnosis(self) -> list[str]:
        """从 `neutral=` 直接得出的两条经济结论（模板 §二 的判据表）

        这一格以前只能人工翻日志原文，而**没有小贩**与**有小贩但调度没去卖**
        是完全相反的两个结论：前者要改采集目标（`MINER_ORDER_NO_VENDOR`），
        后者才是调度问题，两者的修法不通用。矿到基地的距离同理——往返一趟的
        成本决定"值不值得去采"，没有坐标这个账根本算不了。
        """
        neutrals = self.neutral_at(self.first_round)
        if not neutrals:
            return [f"中立元素：{PENDING}（日志里没有可解析的 `neutral=`）"]

        origin = self.base_pos() or (0, 0)
        notes: list[str] = []

        vendor = neutrals.get("vendor") or []
        if vendor:
            spots = " / ".join(f"({x},{y})" for x, y in vendor)
            notes.append(
                f"小贩：有，{spots}（最近离基地 {_chebyshev(vendor[0], origin)} 格）"
            )
        else:
            notes.append(
                "小贩：**没有** ⇒ 矿石卖不出去、金币再也回不来；"
                "此时采集工应当改采石（`MINER_ORDER_NO_VENDOR`），"
                "而不是当成「调度没去卖」去改调度"
            )

        for kind in ("stone", "iron", "copper"):
            spots = neutrals.get(kind) or []
            if not spots:
                notes.append(f"{NEUTRAL_NAMES[kind]}：地图上没有")
                continue
            distance, nearest = min(
                (_chebyshev(spot, origin), spot) for spot in spots
            )
            notes.append(
                f"{NEUTRAL_NAMES[kind]}：{len(spots)} 处，"
                f"最近 ({nearest[0]},{nearest[1]}) 离基地 {distance} 格"
            )
        return notes

    def task_summary(self) -> str:
        return f"交卷{self.submitted}次 / 领任务{self.accepted}次"

    def task_timeline(self) -> list[str]:
        out = []
        for round_no, event in self.task_events:
            name = event.get("event", "?")
            label = {
                "start": "任务开始",
                "accept": "领取任务",
                "step": "推进步骤",
                "submit": "提交答案",
                "done": "任务结束",
                "abandon": "放弃任务",
                "skill_saved": "技能入库",
                "gate_reject": "答案被闸门拦下",
                "llm_exec": "执行 LLM 兜底命令",
                "llm_command": "拿到 LLM 兜底命令",
                "sandbox_slow": "沙盒超时",
                "learn": "学到新事实",
                "family": "重新判族",
                # 别写死成"回退到取数步骤"：detail 才是回退目标，工程修复族
                # 回的是 check（`back_to_check#N`），api-query 族才回 query。
                "rewind": "回退重试",
                "suppress": "任务点抑制（阶梯走完后不再接）",
            }.get(name, name)
            detail = event.get("detail", "")
            suffix = f"（{detail}）" if detail else ""
            step = event.get("step", "")
            context = ""
            if "rounds" in event and "left" in event:
                context = f" 已用{event['rounds']}回合/剩余{event['left']}"
            elif step:
                context = f" step={step}"
            out.append(f"R{round_no} {label}{context}{suffix}")
        return out


def _fields(line: str) -> dict[str, str]:
    """把一行里的 `key=value` 都抽出来（值带引号时去掉引号）"""
    out: dict[str, str] = {}
    for key, value in KV_RE.findall(line):
        out[key] = value.strip('"')
    return out


def _field_value(line: str, key: str) -> str:
    """取出 `key=` 的**完整**值

    `actions=10010:build→(28,7) wall 10011:acceptTask` 这类值内部含空格，
    通用的"非空白字符"正则只会截到第一个空格，把后面几条动作全丢掉。
    """
    match = re.search(
        r"(?:^|\s)%s=(.*?)(?=\s+[A-Za-z_][\w.]*=|$)" % re.escape(key), line
    )
    return match.group(1).strip() if match else ""


def _unescape(text: str) -> str:
    """还原 `task_dump` 的转义（`brain._escape` 的逆运算）

    必须**从左到右扫描**，不能连着做几次 `replace`：原文里本来就可能有
    反斜杠（我们下发给沙盒的命令里全是），逐次替换会把它二次解读。
    """
    out: list[str] = []
    index = 0
    source = text or ""
    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source):
            nxt = source[index + 1]
            if nxt == "n":
                out.append("\n")
                index += 2
                continue
            if nxt == "r":
                out.append("\r")
                index += 2
                continue
            if nxt == "\\":
                out.append("\\")
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _int(text: str, default: int = 0) -> int:
    match = re.match(r"-?\d+", text or "")
    return int(match.group(0)) if match else default


def _float(text: str, default: float = 0.0) -> float:
    """从 `1.29ms` 这类带单位的字段里取数值"""
    match = re.match(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group(0)) if match else default


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """切比雪夫距离——任务书 §4.5.4 规定的移动/攻击距离口径"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _add_event(stats: Stats, round_no: int, text: str, limit: int = 40) -> None:
    if any(existing == text for _, existing in stats.events):
        return
    if len(stats.events) < limit:
        stats.events.append((round_no, text))


def _scan_actions(stats: Stats, round_no: int, actions: str) -> None:
    """从 `actions=` 里挑出值得进时间线的动作

    动作之间用空格分隔，但动作内部也可能带空格（`build→(11,22) rocket`），
    所以按 `角色ID:` 的位置切分，而不是按空格切。
    """
    for chunk in ACTIONS_RE.split(actions):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        _, command = chunk.split(":", 1)
        target = command.split("→")[1].split()[0] if "→" in command else ""
        if command.startswith("build"):
            name = command.split()[-1] if " " in command else ""
            label = "围墙" if name == "wall" else "武器塔"
            _add_event(stats, round_no, f"{label} {name}{'→' + target if target else ''}".strip())
        elif command.startswith("sell"):
            _add_event(stats, round_no, "首次卖矿换金币")
        elif command.startswith("buy"):
            _add_event(stats, round_no, f"买道具 {command.split()[-1]}")
        elif command.startswith("use"):
            _add_event(stats, round_no, f"用道具 {command.split()[-1]}")
        elif command.startswith("attack"):
            _add_event(stats, round_no, "首次武器攻击")
        elif command.startswith("collect"):
            _add_event(stats, round_no, "首次采集")
        elif command.startswith("acceptTask"):
            _add_event(stats, round_no, "领任务")
        elif command.startswith("submitAnswer"):
            _add_event(stats, round_no, "交卷")


def analyze(log_file: Path, window: tuple[int, int] | None = None) -> Stats | None:
    """解析日志并返回统计；日志不存在时返回 None"""
    if not log_file.is_file():
        print(f"日志文件不存在: {log_file}")
        return None

    stats = Stats()
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ERROR_RE.search(line):
                stats.errors.append(line.strip()[:200])

            if "request_decoded" in line:
                fields = _fields(line)
                round_no = _int(fields.get("round", "0"))
                record = stats.rounds.setdefault(round_no, Round(round_no))
                record.state = fields
                record.state["towers"] = _field_value(line, "towers")
                record.state["tasks"] = _field_value(line, "tasks")
                record.state["phase"] = _field_value(line, "phase")
                record.state["robots"] = _field_value(line, "robots")
                record.state["walls"] = _field_value(line, "walls")
                record.state["chars"] = _field_value(line, "chars")
                record.state["bag"] = _field_value(line, "bag")
                # `neutral=` 的值里有空格（`stone:6[29,14 12,3]`），`\S+` 只会
                # 截到第一个坐标，必须取完整值
                record.state["neutral"] = _field_value(line, "neutral")
                continue

            if "strategy_done" in line:
                fields = _fields(line)
                round_no = _int(fields.get("round", "0"))
                record = stats.rounds.setdefault(round_no, Round(round_no))
                fields["actions"] = _field_value(line, "actions")
                record.decision = fields
                continue

            if "round_end" in line:
                fields = _fields(line)
                round_no = _int(fields.get("round", "0"))
                record = stats.rounds.setdefault(round_no, Round(round_no))
                record.outcome = fields
                continue

            if "task_event" in line:
                fields = _fields(line)
                round_no = _int(fields.get("round", "0"))
                rounds = stats.rounds.setdefault(round_no, Round(round_no))
                rounds.events.append(fields)
                continue

            if "day_summary" in line:
                stats.day_lines.append(line.strip())
                continue

            if "freeze_alert" in line:
                stats.alerts.append(line.strip())
                continue

            if "task_dump" in line:
                fields = _fields(line)
                stats.dumps.append((
                    _int(fields.get("round", "0")),
                    fields.get("kind", "?"),
                    _unescape(_field_value(line, "text")),
                ))
                continue

    if window is not None:
        low, high = window
        stats.rounds = {
            k: v for k, v in stats.rounds.items() if low <= k <= high
        }

    _aggregate(stats)
    return stats


def _aggregate(stats: Stats) -> None:
    """把逐回合记录压成汇总"""
    for round_no in stats.ordered:
        record = stats.rounds[round_no]
        state, outcome = record.state, record.outcome
        if "gold" in state:
            stats.gold_series.append((round_no, _int(state["gold"])))
        if "score" in state:
            stats.score_series.append((round_no, _int(state["score"])))
        if state.get("hp"):
            match = HP_RE.match(state["hp"])
            if match:
                stats.hp_series.append(
                    (round_no, _int(match.group(1)), _int(match.group(2)))
                )

        if state.get("tod") == "night":
            stats.night_rounds += 1

        if outcome:
            stats.kills += _int(outcome.get("kills", "0"))
            stats.kill_score += _int(outcome.get("kill_score", "0"))
            stats.damage += _int(outcome.get("station_damage", "0"))
            stats.towers_lost += _int(outcome.get("towers_lost", "0"))
            stats.walls_lost += _int(outcome.get("walls_lost", "0"))
            stats.idle_units += _int(outcome.get("idle_units", "0"))

        decision = record.decision
        if decision:
            spend = _int(decision.get("gold_spent", "0"))
            if spend:
                stats.spends.append((round_no, spend))
            if decision.get("sandbox") == "下发":
                stats.sandbox_rounds += 1
            actions = decision.get("actions", "")
            # 动作名统一从 `角色ID:动作` 里的冒号后面取。
            # 不要写成 `:use `（带空格）——带目标位置的动作渲染成 `use→(x,y)`，
            # 空格版会漏掉它们，`day_summary upgrade=` 就会永远是 0。
            counts: dict[str, int] = {}
            for name in re.findall(r":([a-zA-Z]+)", actions):
                counts[name] = counts.get(name, 0) + 1
            stats.builds += counts.get("build", 0)
            stats.sells += counts.get("sell", 0)
            stats.upgrades += counts.get("use", 0)
            stats.accepted += counts.get("acceptTask", 0)
            stats.submitted += counts.get("submitAnswer", 0)
            _scan_actions(stats, round_no, actions)

        for event in record.events:
            stats.task_events.append((round_no, event))

    # 金币收支：用相邻回合的 gold 差值（比逐回合累加 gold_delta 更抗缺行）
    for index in range(1, len(stats.gold_series)):
        delta = stats.gold_series[index][1] - stats.gold_series[index - 1][1]
        if delta > 0:
            stats.gold_in += delta
        elif delta < 0:
            stats.gold_out -= delta


# ==========================================================================
# 输出
# ==========================================================================


def print_summary(stats: Stats, log_file: Path) -> None:
    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"文件大小: {log_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    if not stats.rounds:
        print("未解析到回合记录（日志可能为空或格式不符）")
        return

    elapsed = [
        _float(record.decision["elapsed"])
        for record in stats.rounds.values()
        if "elapsed" in record.decision
    ]
    commands = [
        _int(record.decision.get("commands", "0"))
        for record in stats.rounds.values()
        if record.decision
    ]

    print(f"处理回合数: {len(stats.rounds)}")
    print(f"回合范围  : {stats.first_round} - {stats.last_round}")
    print(f"阵营/队伍 : {stats.team_type or PENDING} / {stats.team_id or PENDING}")
    if elapsed:
        values = sorted(elapsed)
        print(f"决策耗时  : 平均 {sum(values) / len(values):.2f}ms"
              f"，中位 {values[len(values) // 2]:.2f}ms，最慢 {values[-1]:.2f}ms")
    if commands:
        print(f"指令数    : 平均 {sum(commands) / len(commands):.1f} 条/回合")
    print(f"金币      : 峰值 {stats.gold_peak}，终值 {stats.gold_final}"
          f"（收 {stats.gold_in} / 支 {stats.gold_out}）")
    print(f"积分      : 终值 {stats.score_final}")
    print(f"基地血量  : {stats.hp[0]} / {stats.hp[1]}，累计掉血 {stats.damage}")
    print(f"击杀/得分 : {stats.kills} / {stats.kill_score}")
    print(f"建筑损失  : 塔 {stats.towers_lost} 座，墙 {stats.walls_lost} 段")
    print(f"角色空转  : {stats.idle_units} 人·回合（占比 {stats.idle_ratio:.1%}）")
    print(f"任务      : {stats.task_summary()}，下发沙盒 {stats.sandbox_rounds} 回合")
    print(f"建造/卖矿 : {stats.builds} 次建造，{stats.sells} 次卖矿")
    for note in stats.economy_diagnosis():
        print(f"  地图     : {note}")
    if stats.alerts:
        print(f"冻结告警  : {len(stats.alerts)} 次（首次 "
              f"R{_int(_fields(stats.alerts[0]).get('round', '0'))}）")
    print(f"错误条数  : {len(stats.errors)}")
    for line in stats.errors[:10]:
        print(f"  {line}")


def print_full(stats: Stats) -> None:
    """打印自进化任务的**全量日志**（`--full`）

    这些是 `task_dump` 行（INFO 级，stdout 与 `debug.log` 各一份），还原成
    多行后打印：任务描述原文、我们下发的沙盒命令、沙盒原样回的什么、提交的
    答案、LLM 的 prompt 与回复。结构化那几行是给复盘按字段读的，排查问题
    要看的是这一组。
    """
    if not stats.dumps:
        print("（日志里没有 task_dump 行）")
        print("  只有涉及自进化任务的回合才会打；确认这份日志覆盖到了任务期间。")
        return

    current = None
    for round_no, kind, text in stats.dumps:
        if round_no != current:
            current = round_no
            print()
            print("=" * 70)
            print(f"回合 R{round_no}")
            print("=" * 70)
        print(f"--- {kind} ---")
        print(text if text.strip() else "（空）")


def print_task_trace(stats: Stats) -> None:
    """只打印自进化任务链路"""
    print("=" * 60)
    print("自进化任务链路")
    print("=" * 60)
    print(f"{stats.task_summary()}，下发沙盒命令 {stats.sandbox_rounds} 回合")
    if not stats.task_events:
        print("（日志里没有 task_event 行）")
        return
    print()
    for line in stats.task_timeline():
        print(f"  {line}")


def _bullet(items: list[str], empty: str = PENDING) -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def render_template(stats: Stats, log_path: Path) -> str:
    """渲染 `战术参考/日志分析模板V2.md` 的填空稿"""
    if not stats.rounds:
        return "（日志为空，无法生成分析稿）"

    hp_now, hp_full = stats.hp
    towers = stats.towers()
    tower_text = " / ".join(
        f"{TOWER_NAMES.get(t['kind'], t['kind'])}{t['pos']}L{t['level']}"
        for t in towers
    ) or "-"
    tower_order = " → ".join(
        f"{TOWER_NAMES.get(label.split('@')[0], label.split('@')[0])}"
        f"({label.split('@')[1]}, R{round_no})"
        for label, round_no in stats.tower_timeline()
    ) or PENDING
    wall_order = " → ".join(
        f"R{round_no}:{count}段" for round_no, count in stats.wall_timeline()
    ) or PENDING

    lines = [
        "# 对战分析（由 debug.log 自动生成）",
        "",
        "> 本稿由 `CoreGeek/tools/analyze_log.py --template` 生成，"
        f"对应模板 `{TEMPLATE_HINT.name}`。",
        "> 标 `{待人工}` 的格子必须照着 `日志原文` 一节读原文补，不要靠猜。",
        "",
        "## 元信息",
        "",
        "| 项目 | 内容 |",
        "|-----|------|",
        "| 对战ID | {待人工} |",
        f"| 我方队名 / teamId | {stats.team_id or PENDING} |",
        f"| 我方阵营 | {stats.team_type or PENDING}（challenger 挑战方 / defender 防守方） |",
        f"| 得分 | {stats.score_final} |",
        f"| 对战结果 | {PENDING}（胜/负/平，需对着最终比分填） |",
        f"| 日志范围 | 第{stats.first_round}回合 - 第{stats.last_round}回合"
        f"（{len(stats.rounds)}个回合，其中夜战{stats.night_rounds}回合） |",
        f"| 源文件 | `{log_path}` |",
        "",
        "---",
        "",
        "## 一、战况概览",
        "",
        "### 1.1 关键数据对比",
        "",
        "| 指标 | 我方 | 敌方 | 差距分析 |",
        "|-----|-----|-----|---------|",
        f"| **最终积分** | {stats.score_final} | {PENDING}（报文不提供） | {PENDING} |",
        f"| **基地血量** | {hp_now} / {hp_full} | {PENDING} | 累计掉血 {stats.damage} |",
        f"| **武器塔** | {len(towers)}座（{tower_text}） | {stats.enemies_now()} | "
        f"损失 {stats.towers_lost} 座 |",
        f"| **围墙** | {stats.walls_now()} | 见上 | 损失 {stats.walls_lost} 段 |",
        f"| **任务** | {stats.task_summary()} | {PENDING} | 见 §4 |",
        f"| **金币** | 峰值 {stats.gold_peak}，终值 {stats.gold_final}"
        f"（收 {stats.gold_in} / 支 {stats.gold_out}） | 报文不提供 | {PENDING} |",
        f"| **机器人击杀** | {stats.kills}（战斗分 {stats.kill_score}） | {PENDING} | {PENDING} |",
        f"| **角色空转** | {stats.idle_units} 人·回合（{stats.idle_ratio:.1%}） | {PENDING} | {PENDING} |",
        "",
        "> 敌方只有基地与围墙全图可见（接口文档 §1.4），塔/经济/击杀都拿不到，"
        "`enemy_visible` 用来区分\"真没有\"与\"没看见\"。",
        "",
        "### 1.2 时间线关键节点",
        "",
        "| 回合 | 事件 | 影响 |",
        "|-----|------|------|",
    ]
    for round_no, event in stats.events:
        lines.append(f"| R{round_no} | {event} | {PENDING} |")
    if not stats.events:
        lines.append(f"| {PENDING} | 日志里没有可识别的建造/任务动作 | {PENDING} |")

    lines += [
        "",
        "### 1.3 版本与迭代节奏",
        "",
        f"- 塔的建成顺序：{tower_order}",
        f"- 围墙变化：{wall_order}",
        f"- 建造 {stats.builds} 次 / 卖矿 {stats.sells} 次 / 用道具 {stats.upgrades} 次",
        f"- 金币停滞告警：{len(stats.alerts)} 次"
        + (f"（首次 R{_int(_fields(stats.alerts[0]).get('round', '0'))}）" if stats.alerts else ""),
        "",
        "---",
        "",
        "## 二、经济与资源",
        "",
        "### 2.1 金币流水",
        "",
        "```",
        _gold_sparkline(stats),
        "```",
        "",
        "**问题识别**：",
        "- [ ] 金币冻结（连续 ≥20 个回合 `gold < 25` 且无支出 → 日志里有 `freeze_alert`）",
        "- [ ] 经济断层（金币清零后长时间无收入）",
        "- [ ] 升级延迟（有金币但没买升级券，`use=` 计数为 0）",
        "",
        "### 2.2 采集与贩卖",
        "",
        f"- 卖矿次数：{stats.sells}（其中若干次可能是因为背包满）",
        f"- 金币总收入 {stats.gold_in}，总支出 {stats.gold_out}",
        f"- 建造支出合计：{sum(spend for _, spend in stats.spends)}"
        f"（{len(stats.spends)} 个回合有支出）",
        "",
        "**地图侧（`neutral=` 自动解析，判据见 §二）**：",
    ]
    lines += [f"- {note}" for note in stats.economy_diagnosis()]
    lines += [
        "",
        "> 往返一趟矿的成本就写在上面的距离里（切比雪夫距离，任务书 §4.5.4）；"
        "「工人一直没去采」之前要先看这一格——**没有小贩**与**有小贩没去卖**"
        "是两条不同的修法。",
        "",
        "---",
        "",
        "## 三、建造与布局",
        "",
        "```",
        f"塔建成顺序: {tower_order}",
        f"塔位:       {tower_text}",
        f"围墙变化:   {wall_order}",
        "```",
        "",
        "**问题识别**：",
        "- [ ] 塔位堵路（`fail=[ID:build]` 频繁出现 → 建造被拒）",
        "- [ ] 塔位贴基/同侧（对着上面的坐标判断是否分散、是否朝来路）",
        "- [ ] 射程覆盖不足（夜里 `idle_target` 长期 > 0 → 有人操控但够不着）",
        "- [ ] 建造延迟（首塔回合 vs 对手）",
        "",
        "---",
        "",
        "## 四、自进化任务链路（重点）",
        "",
        "### 4.1 任务事件时间线",
        "",
        _bullet(stats.task_timeline()),
        "",
        "### 4.2 汇总",
        "",
        f"- 领任务 {stats.accepted} 次 / 交卷 {stats.submitted} 次 / "
        f"下发沙盒命令 {stats.sandbox_rounds} 回合",
        "- 每个任务的步进次数（正常应该在 2~4 回合内闭环：recon → query → submit）",
        "",
        "**问题识别**：",
        "- [ ] 任务未接（有 `isValid` 任务点却长时间 `task=idle`）",
        "- [ ] 接取后从未提交（有 `event=start` 但没有 `event=submit`）",
        "- [ ] 沙盒空转（`event=step` 反复推进同一 step）",
        "- [ ] 答案被拦（`event=gate_reject`）",
        "- [ ] 任务被放弃（`event=abandon`，看 detail 里的原因）",
        "- [ ] 技能没入库（同族第二个任务没有 `event=skill_saved` 之后的加速）",
        "",
        "---",
        "",
        "## 五、战斗与防守",
        "",
        "| 游戏日 | 夜战回合 | 击杀 | 基地受伤 | 无人操控的塔 | 够不着目标的塔 |",
        "|-------|---------|-----|---------|------------|--------------|",
    ]
    lines += _night_table(stats)

    lines += [
        "",
        "**问题识别**：",
        "- [ ] 武器空转（`idle_weapon > 0` → 有人没到位，回防/配位问题）",
        "- [ ] 射程不足（`idle_target > 0` → 到位了但够不着，塔位问题）",
        "- [ ] 围墙被突破（`walls_lost` 增长）",
        "- [ ] 目标选择（对 `actions=` 里的 `attack→(x,y) <型号>`，看有没有先打 BOSS/大型）",
        "",
        "---",
        "",
        "## 六、日志原文",
        "",
        "> **这一节是分析的依据，不能省。** 上面所有 `{待人工}` 都要回到这里读原文补。",
        "> 用 `--rounds A-B` 只截取相关回合，或者手工从 `debug.log` 里拷。",
        "",
        "```log",
        "{待人工}：粘贴相关的 request_decoded / strategy_done / round_end / task_event 原文",
        "```",
        "",
        "---",
        "",
        "## 七、结论与优化建议",
        "",
        "### 7.1 直接失分项",
        "",
        "| 失分点 | 证据（回合 + 日志字段） | 根因 | 对应模块 |",
        "|-------|----------------------|------|---------|",
        f"| {PENDING} | {PENDING} | {PENDING} | {PENDING} |",
        "",
        "> 对应模块请按 `README.md` 的「改代码时该看哪个文件」表填，"
        "**不要写 `brain.py`**——V2 已经拆成 10 个模块。",
        "",
        "### 7.2 优化建议（P0 必改 / P1 高价值 / P2 可选）",
        "",
        f"**P0｜{PENDING}**",
        f"- 现象：{PENDING}",
        f"- 根因：{PENDING}",
        f"- 修复：{PENDING}",
        f"- 预期：{PENDING}",
        "",
        "### 7.3 行动计划",
        "",
        "- [ ] 立即修复（本轮）：{待人工}",
        "- [ ] 后续优化：{待人工}",
        "",
        "> 走自动化修复请按 `战术参考/Issue总结模板V2.md` 生成 Issue。",
        "",
    ]

    filled = sum(
        1 for value in (
            str(stats.score_final), hp_now, str(len(towers)), stats.walls_now(),
            str(stats.kills), str(stats.gold_peak), stats.task_summary(),
        )
        if value and not value.startswith(PENDING) and value not in ("0", "")
    )
    lines += [
        f"> 自动填充：{filled}/7 项（比分、基地血量、塔、围墙、击杀、金币、任务）。",
        "",
    ]
    return "\n".join(lines)


def _gold_sparkline(stats: Stats, buckets: int = 40) -> str:
    """金币曲线（文本化），一眼看出有没有长时间贴合 0"""
    series = stats.gold_series
    if not series:
        return "（没有金币序列）"
    step = max(1, len(series) // buckets)
    sampled = series[::step]
    peak = max(gold for _, gold in sampled) or 1
    out = []
    for round_no, gold in sampled:
        bar = "█" * max(0, round(gold / peak * 20))
        out.append(f"R{round_no:>4} {gold:>4} {bar}")
    return "\n".join(out)


def _night_table(stats: Stats) -> list[str]:
    """按游戏日聚合夜战数据"""
    days: dict[int, dict[str, int]] = {}
    for round_no in stats.ordered:
        record = stats.rounds[round_no]
        if record.state.get("tod") != "night":
            continue
        day = _int(record.state.get("day", "0"))
        bucket = days.setdefault(
            day, {"rounds": 0, "kills": 0, "damage": 0, "idle_weapon": 0, "idle_target": 0}
        )
        bucket["rounds"] += 1
        if record.outcome:
            bucket["kills"] += _int(record.outcome.get("kills", "0"))
            bucket["damage"] += _int(record.outcome.get("station_damage", "0"))
            bucket["idle_weapon"] += _int(record.outcome.get("idle_weapon", "0"))
            bucket["idle_target"] += _int(record.outcome.get("idle_target", "0"))
    if not days:
        return [f"| {PENDING} | - | - | - | - | - |"]
    return [
        f"| 第{day}天 | {b['rounds']} | {b['kills']} | {b['damage']} | "
        f"{b['idle_weapon']} | {b['idle_target']} |"
        for day, b in sorted(days.items())
    ]


def render_issue(stats: Stats, log_path: Path) -> str:
    """渲染 `战术参考/Issue总结模板V2.md` 的正文骨架

    刻意**留出"日志原文"的空位**：自动化那边真正缺的不是结论，是可信的原始
    证据——V1 的自动修复就是因为只有"反推的常量名"而没有原文，才反复改错地方。
    """
    hp_now, hp_full = stats.hp
    towers = stats.towers()
    tower_text = " / ".join(
        f"{TOWER_NAMES.get(t['kind'], t['kind'])}{t['pos']}L{t['level']}" for t in towers
    ) or "-"

    sample = _sample_lines(stats, limit=30)
    return "\n".join([
        f"<!-- 本骨架由 analyze_log.py --issue 生成，对应模板 {ISSUE_HINT.name} -->",
        "## 对战分析报告 ({待人工} PK 号)",
        "",
        "**结果**: teamAName={待人工}; teamBName={待人工}; "
        "scoreA={待人工}; scoreB={待人工}",
        "",
        f"**我方**: teamId={stats.team_id or PENDING}，"
        f"{stats.team_type or PENDING}",
        f"**日志范围**: 第{stats.first_round}回合 - 第{stats.last_round}回合"
        f"（{len(stats.rounds)}回合）",
        "",
        "---",
        "",
        "### 1. 战况概览",
        "",
        "| 指标 | 我方 | 敌方 |",
        "|-----|-----|-----|",
        f"| 最终积分 | {stats.score_final} | {PENDING} |",
        f"| 基地血量 | {hp_now} / {hp_full} | {PENDING} |",
        f"| 武器塔 | {len(towers)}座（{tower_text}） | {stats.enemies_now()} |",
        f"| 围墙 | {stats.walls_now()} | {PENDING} |",
        f"| 任务 | {stats.task_summary()} | {PENDING} |",
        f"| 金币 | 峰值 {stats.gold_peak}，终值 {stats.gold_final} | 报文不提供 |",
        f"| 机器人击杀 | {stats.kills}（战斗分 {stats.kill_score}） | {PENDING} |",
        f"| 角色空转 | {stats.idle_units} 人·回合 | {PENDING} |",
        "",
        "### 2. 任务链路",
        "",
        _bullet(stats.task_timeline()),
        "",
        "### 3. 日志原文（分析依据）",
        "",
        "> **不要删这一节。** 下面是从 `debug.log` 里摘的关键回合原文；",
        "> 每条结论都必须能在原文里指到具体字段。",
        "> 需要更多回合就加参数：`python tools/analyze_log.py "
        "--log <日志> --rounds A-B --issue`。",
        "",
        "```log",
        f"# 源文件: {log_path}",
        *sample,
        "# {待人工}：追加「现象发生的那几个回合」的完整原文",
        "# 字段：request_decoded / strategy_done / round_end / task_event",
        "```",
        "",
        "### 4. 发现的问题",
        "",
        "**P1｜{待人工：问题标题}**",
        "- **现象**：{待人工：回合号 + 日志字段，例如 `R11-R16 round_end idle_units=3`}",
        "- **根因**：{待人工：模块 + 函数/常量，按模块地图填}",
        "- **影响**：{待人工：失多少分 / 是否导致失败}",
        "",
        "### 5. 优化建议",
        "",
        "**S1｜{待人工：建议标题}**",
        "- 涉及：`CoreGeek/src/agent/{待人工}.py`",
        "- 做法：{待人工：具体改动，写出常量名/函数名}",
        "- 预期：{待人工：可验证的效果}",
        "",
        "### 6. 日志中的原始证据",
        "",
        "| 结论 | 日志字段 | 回合 | 原文片段 |",
        "|-----|---------|-----|---------|",
        "| {待人工} | {待人工} | {待人工} | `{待人工：从上面拷一行}` |",
        "",
        "---",
        "",
        "```claude-prompt",
        "根据对战分析报告的以下建议修改代码：",
        "",
        "S1: {待人工：建议标题}",
        "- 文件：CoreGeek/src/agent/{待人工}.py",
        "- 改动：{待人工：具体做法}",
        "- 预期：{待人工：效果}",
        "",
        "约束：",
        "1. 保持无状态决策设计（每回合从 Turn 重新解析，不引入跨回合战场缓存；",
        "   任务链路的学习成果只放 strategy/task/memory.py）",
        "2. 修改后必须通过 `cd CoreGeek && python -m pytest tests/ -q`",
        "3. 新增行为要补单测，按功能放进对应的测试文件：",
        "   protocol / grid / world / telemetry(日志) / economy / defense /",
        "   sandbox / answer / skills / solver / brain(端到端)",
        "4. 不要改 protocol.py 里已有的报文字段名与动作码",
        "5. 不要改 main3.py / run.sh（判题系统入口）",
        "6. 改日志字段时同步更新 tools/analyze_log.py 与 战术参考/日志分析模板V2.md",
        "```",
        "",
        "*本 issue 由对战分析生成，结论必须能在 §3 的日志原文里指到出处。*",
        "",
    ])


def _sample_lines(stats: Stats, limit: int = 30) -> list[str]:
    """挑出最有信息量的若干回合原文，作为 Issue 里的"日志原文"起点

    优先挑：任务事件回合、任务回合前后、夜战里掉血/被突破的回合、最后几个回合。
    不是全量——全量靠 `--rounds` 再取。
    """
    picked: list[int] = []
    interesting = set()
    for round_no, _ in stats.task_events:
        interesting.add(round_no)
        interesting.add(round_no + 1)
    for round_no in stats.ordered:
        outcome = stats.rounds[round_no].outcome
        if outcome and (
            _int(outcome.get("station_damage", "0")) > 0
            or _int(outcome.get("towers_lost", "0")) > 0
            or _int(outcome.get("walls_lost", "0")) > 0
        ):
            interesting.add(round_no)
    picked.extend(sorted(interesting)[: max(0, limit - 4)])
    picked.extend(stats.ordered[-3:])

    out: list[str] = []
    seen: set[int] = set()
    for round_no in sorted(set(picked)):
        record = stats.rounds.get(round_no)
        if record is None:
            continue
        seen.add(round_no)
        for label, fields in (
            ("request_decoded", record.state),
            ("strategy_done", record.decision),
            ("round_end", record.outcome),
        ):
            if fields:
                out.append(
                    f"{label} " + " ".join(
                        f"{k}={v}" for k, v in fields.items() if k != "round"
                    )
                )
        for event in record.events:
            out.append("task_event " + " ".join(f"{k}={v}" for k, v in event.items()))
        if len(out) > limit * 4:
            out.append(f"# …（截断，共选中 {len(picked)} 个回合）")
            break
    return out or ["# {待人工}：本次日志里没有明显的异常回合，请手工挑几段"]


# ==========================================================================
# 入口
# ==========================================================================


def _window(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    if "-" in text:
        low, _, high = text.partition("-")
        return (_int(low, 1), _int(high, 10**9))
    value = _int(text, 0)
    return (value, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析客户端 debug.log")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="日志文件路径")
    parser.add_argument("--template", action="store_true",
                        help="渲染 战术参考/日志分析模板V2.md 的填空稿")
    parser.add_argument("--issue", action="store_true",
                        help="渲染 战术参考/Issue总结模板V2.md 的正文骨架")
    parser.add_argument("--task", action="store_true", help="只打印任务链路")
    parser.add_argument("--full", action="store_true",
                        help="打印自进化任务的全量日志（任务描述/沙盒命令/沙盒输出/提交的答案）")
    parser.add_argument("--rounds", default="", help="只看某个回合区间，如 1-130 或 85")
    parser.add_argument("--out", default="", help="输出文件（默认打印到终端）")
    args = parser.parse_args()

    log_path = Path(args.log)
    stats = analyze(log_path, _window(args.rounds))
    if stats is None:
        raise SystemExit(1)

    if args.template:
        body = render_template(stats, log_path)
    elif args.issue:
        body = render_issue(stats, log_path)
    elif args.task:
        print_task_trace(stats)
        raise SystemExit(0)
    elif args.full:
        print_full(stats)
        raise SystemExit(0)
    else:
        print_summary(stats, log_path)
        raise SystemExit(0)

    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(body)


if __name__ == "__main__":
    main()
