#!/usr/bin/env python3
"""分析 debug.log，统计回合处理情况并渲染复盘填空稿。

用法:
    python CoreGeek/tools/analyze_log.py                 # 默认分析 CoreGeek/debug.log
    python CoreGeek/tools/analyze_log.py --log x.log
    python CoreGeek/tools/analyze_log.py --template      # 按对战分析模板生成填空稿
    python CoreGeek/tools/analyze_log.py --task          # 只看自进化任务链路

对应设计文档V2 第 9 章。

V2 的日志格式是**结构化单行**（见 `brain._log_turn`）：

    request_decoded round=… team=… team_id=… gold=… hp=… towers=… walls=…
                    robots=…(s m l b) tasks=[…] phase=… task=… zone=… bag=…
    strategy_done   round=… commands=… actions=… sandbox=… note=… learn=… fail=[…]
    round_end       round=… kills=…(…) score=… station_damage=… idle=…
    day_summary     day=… rounds=… kills=… gold_in=… submits=… sandbox=…

这四行合起来覆盖了 `战术参考/对战分析模板.md` 里的全部关键指标，
所以 `--template` 能自动填的部分比 V1 多得多（尤其是 V1 里大量
"日志未覆盖"的击杀/掉血/空转数据）。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "CoreGeek" / "debug.log"

PENDING = "{待人工}"

KV_RE = re.compile(r"([A-Za-z_][\w.]*)=(\S+)")
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b")
HP_RE = re.compile(r"^(\d+)/(\d+|-)$")

TOWER_NAMES = {"gatling": "加特林", "railgun": "电磁", "rocket": "火箭"}


@dataclass
class Stats:
    """逐回合累积的统计"""

    rounds: set[int] = field(default_factory=set)
    elapsed: list[float] = field(default_factory=list)
    last: dict[str, str] = field(default_factory=dict)
    gold_series: list[tuple[int, int]] = field(default_factory=list)
    team_type: str = ""
    team_id: str = ""

    kills: int = 0
    kill_score: int = 0
    damage: int = 0
    towers_lost: int = 0
    walls_lost: int = 0
    idle_rounds: int = 0
    commands: list[int] = field(default_factory=list)

    submits: int = 0
    accepts: int = 0
    sandbox_rounds: int = 0
    step_advances: list[str] = field(default_factory=list)

    day_lines: list[str] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # --- 汇总 ---

    @property
    def first_round(self) -> int:
        return min(self.rounds) if self.rounds else 0

    @property
    def last_round(self) -> int:
        return max(self.rounds) if self.rounds else 0

    @property
    def gold_peak(self) -> int:
        return max((gold for _, gold in self.gold_series), default=0)

    @property
    def gold_final(self) -> int:
        value = self.last.get("gold", "0")
        return int(value) if value.lstrip("-").isdigit() else 0

    @property
    def idle_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        return self.idle_rounds / (len(self.rounds) * 3)  # 三个可操控角色

    def hp_pair(self) -> tuple[str, str]:
        match = HP_RE.match(self.last.get("hp", ""))
        return (match.group(1), match.group(2)) if match else (PENDING, PENDING)

    def towers(self) -> tuple[str, str]:
        """(塔数, "火箭(29,8)L1 / 电磁(29,9)L1")

        日志里用 `;` 分隔多座塔——坐标本身含逗号，用逗号分的话复盘的
        "3 座塔"会被数成 6 座。
        """
        raw = self.last.get("towers", "-")
        if not raw or raw == "-":
            return "0", PENDING
        detail = []
        for item in raw.split(";"):
            name, _, spec = item.partition("@")
            kind = "".join(c for c in name if not c.isdigit())
            level = "".join(c for c in name if c.isdigit()) or "1"
            detail.append(f"{TOWER_NAMES.get(kind, kind)}({spec})L{level}")
        return str(len(detail)), " / ".join(detail)

    def walls(self) -> str:
        return self.last.get("walls", "0")

    def task_summary(self) -> str:
        return f"交卷{self.submits}次/领任务{self.accepts}次"


def _fields(line: str) -> dict[str, str]:
    """把一行里的 `key=value` 都抽出来（值带引号时去掉引号）"""
    out: dict[str, str] = {}
    for key, value in KV_RE.findall(line):
        out[key] = value.strip('"')
    return out


def _field_value(line: str, key: str) -> str:
    """取出 `key=` 的**完整**值

    `actions=10010:build→(28,7) wall 10011:acceptTask` 这类值内部含空格，
    通用的"非空白字符"正则只会截到第一个空格，把后面几条动作全丢掉——
    V1 的复盘报告里"指令数对不上"就有一部分是这么来的。
    """
    match = re.search(
        r"(?:^|\s)%s=(.*?)(?=\s+[A-Za-z_][\w.]*=|$)" % re.escape(key), line
    )
    return match.group(1).strip() if match else ""


def _int(text: str, default: int = 0) -> int:
    """从 `0(-)` 这种带注记的值里取出前面的整数"""
    match = re.match(r"-?\d+", text or "")
    return int(match.group(0)) if match else default


def _add_event(stats: Stats, round_no: int, text: str, limit: int = 40) -> None:
    """记录时间线上的关键节点（同类事件只留第一次）"""
    if any(existing == text for _, existing in stats.events):
        return
    if len(stats.events) < limit:
        stats.events.append((round_no, text))


def _scan_actions(stats: Stats, round_no: int, actions: str) -> None:
    """从 `actions=` 里挑出值得进时间线的动作

    动作之间用空格分隔，但动作内部也可能带空格（`build→(11,22) rocket`），
    所以按 `角色ID:` 的位置切分，而不是按空格切。
    """
    for chunk in re.split(r"(?=\d+:)", actions):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        _, command = chunk.split(":", 1)
        target = command.split("→")[1].split()[0] if "→" in command else ""
        if command.startswith("build"):
            name = command.split()[-1] if " " in command else ""
            label = "围墙" if name == "wall" else "武器塔"
            spot = f"→{target}" if target else ""
            _add_event(stats, round_no, f"{label} {name}{spot}".strip())
        elif command.startswith("sell"):
            _add_event(stats, round_no, "首次卖矿换金币")
        elif command.startswith("buy"):
            _add_event(stats, round_no, f"买道具 {command.split()[-1]}")
        elif command.startswith("use"):
            _add_event(stats, round_no, f"用道具 {command.split()[-1]}")
        elif command.startswith("acceptTask"):
            stats.accepts += 1
            _add_event(stats, round_no, "领任务")
        elif command.startswith("submitAnswer"):
            stats.submits += 1
            _add_event(stats, round_no, "交卷")
        elif command.startswith("attack"):
            _add_event(stats, round_no, "首次武器攻击")
        elif command.startswith("collect"):
            _add_event(stats, round_no, "首次采集")


def _scan_task(stats: Stats, round_no: int, fields: dict[str, str]) -> None:
    """任务链路的关键事件（步骤推进 / 放弃 / 求助 LLM）

    领任务与交卷由 `_scan_actions` 统计（那里能看到动作原文），这里不重复记。
    """
    note = fields.get("note", "")
    if not note or note == "-":
        return
    if note.startswith("step="):
        step = note.split()[0][len("step="):]
        stats.step_advances.append(f"R{round_no}:{step}")
        _add_event(stats, round_no, f"任务步骤 {step}")
    elif note.startswith("abandon"):
        _add_event(stats, round_no, f"放弃任务（{note}）")
    elif note.startswith("travel"):
        _add_event(stats, round_no, "前往任务点")
    elif note.startswith("night_defend"):
        _add_event(stats, round_no, "夜晚先守夜")


def analyze(log_file: Path) -> Stats | None:
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
                stats.rounds.add(round_no)
                stats.team_type = stats.team_type or fields.get("team", "")
                stats.team_id = stats.team_id or fields.get("team_id", "")
                stats.last = fields
                stats.gold_series.append((round_no, _int(fields.get("gold", "0"))))
                continue

            if "strategy_done" in line:
                fields = _fields(line)
                round_no = _int(fields.get("round", "0"))
                if "elapsed" in fields:
                    try:
                        stats.elapsed.append(float(fields["elapsed"]))
                    except ValueError:
                        pass
                stats.commands.append(_int(fields.get("commands", "0")))
                if fields.get("sandbox") == "下发":
                    stats.sandbox_rounds += 1
                _scan_actions(stats, round_no, _field_value(line, "actions"))
                _scan_task(stats, round_no, fields)
                continue

            if "round_end" in line:
                fields = _fields(line)
                stats.kills += _int(fields.get("kills", "0"))
                stats.kill_score += _int(fields.get("score", "0"))
                stats.damage += _int(fields.get("station_damage", "0"))
                stats.towers_lost += _int(fields.get("towers_lost", "0"))
                stats.walls_lost += _int(fields.get("walls_lost", "0"))
                stats.idle_rounds += _int(fields.get("idle", "0"))
                continue

            if "day_summary" in line:
                stats.day_lines.append(line.strip())
                continue

    return stats


def print_summary(stats: Stats, log_file: Path) -> None:
    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"文件大小: {log_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    if not stats.rounds:
        print("未解析到回合记录（日志可能为空或格式不符）")
        return

    print(f"处理回合数: {len(stats.rounds)}")
    print(f"回合范围  : {stats.first_round} - {stats.last_round}")
    print(f"阵营/队伍 : {stats.team_type or PENDING} / {stats.team_id or PENDING}")
    if stats.elapsed:
        values = sorted(stats.elapsed)
        print(f"决策耗时  : 平均 {sum(values) / len(values):.2f}ms"
              f"，中位 {values[len(values) // 2]:.2f}ms"
              f"，最慢 {values[-1]:.2f}ms")
    if stats.commands:
        print(f"指令数    : 平均 {sum(stats.commands) / len(stats.commands):.1f} 条/回合")
    print(f"金币      : 峰值 {stats.gold_peak}，终值 {stats.gold_final}")
    print(f"击杀/得分 : {stats.kills} / {stats.kill_score}")
    print(f"基地掉血  : {stats.damage}")
    print(f"建筑损失  : 塔 {stats.towers_lost} 座，墙 {stats.walls_lost} 段")
    print(f"角色空转  : {stats.idle_rounds} 人·回合（占比 {stats.idle_ratio:.1%}）")
    print(f"任务      : {stats.task_summary()}，下发沙盒 {stats.sandbox_rounds} 回合")
    if stats.step_advances:
        print(f"任务步骤  : {' → '.join(stats.step_advances[-12:])}")
    print(f"错误条数  : {len(stats.errors)}")
    for line in stats.errors[:10]:
        print(f"  {line}")


def print_task_trace(stats: Stats) -> None:
    """只打印自进化任务链路（排查任务相关问题时用）"""
    print("=" * 60)
    print("自进化任务链路")
    print("=" * 60)
    print(f"领取任务 {stats.accepts} 次，提交答案 {stats.submits} 次，"
          f"下发沙盒命令 {stats.sandbox_rounds} 回合")
    if not stats.events and not stats.step_advances:
        print("（日志里没有任务链路事件）")
        return
    print("\n步骤推进:")
    for item in stats.step_advances:
        print(f"  {item}")
    print("\n时间线:")
    for round_no, text in stats.events:
        print(f"  R{round_no:>4}  {text}")


def render_template(stats: Stats) -> str:
    """把统计渲染成对战分析模板的填空稿"""
    if not stats.rounds:
        return "（日志为空，无法生成分析稿）"

    hp_now, hp_max = stats.hp_pair()
    towers_n, towers_detail = stats.towers()

    lines = [
        "# 对战分析（由 debug.log 自动生成）",
        "",
        "## 元信息",
        "",
        "| 项目 | 内容 |",
        "|-----|------|",
        f"| 对战ID | {PENDING} |",
        f"| 我方队伍 | teamId={stats.team_id or PENDING} |",
        f"| 我方阵营 | {stats.team_type or PENDING} |",
        f"| 日志范围 | 第{stats.first_round}回合 - 第{stats.last_round}回合"
        f"（{len(stats.rounds)}个回合） |",
        "",
        "## 一、战况概览",
        "",
        "### 1.1 关键数据对比",
        "",
        "| 指标 | 我方 | 敌方 | 差距分析 |",
        "|-----|------|------|---------|",
        f"| **最终积分** | {stats.last.get('score', PENDING)} | {PENDING} | {PENDING} |",
        f"| **基地血量** | {hp_now} / {hp_max} | {PENDING} | 累计掉血 {stats.damage} |",
        f"| **武器塔数量** | {towers_n}座（{towers_detail}） | {PENDING} | "
        f"损失 {stats.towers_lost} 座 |",
        f"| **围墙段数** | {stats.walls()}段 | {PENDING} | 损失 {stats.walls_lost} 段 |",
        f"| **任务完成** | {stats.task_summary()} | {PENDING} | {PENDING} |",
        f"| **金币累积** | 峰值{stats.gold_peak}，终值{stats.gold_final} | {PENDING} | {PENDING} |",
        f"| **机器人击杀** | {stats.kills}（战斗分 {stats.kill_score}） | {PENDING} | {PENDING} |",
        f"| **角色空转** | {stats.idle_rounds} 人·回合 | {PENDING} | "
        f"占比 {stats.idle_ratio:.1%} |",
        "",
        "### 1.2 时间线关键节点",
        "",
        "| 回合 | 事件 |",
        "|-----|------|",
    ]
    for round_no, event in stats.events:
        lines.append(f"| R{round_no} | {event} |")
    if not stats.events:
        lines.append(f"| {PENDING} | 日志里没有可识别的建造/任务动作 |")

    lines += ["", "### 1.3 自进化任务链路", ""]
    if stats.step_advances:
        lines.append(f"- 步骤推进：{' → '.join(stats.step_advances)}")
    else:
        lines.append(f"- 步骤推进：{PENDING}（本次日志没有任务链路事件）")
    lines.append(f"- 交卷 {stats.submits} 次，领任务 {stats.accepts} 次，"
                 f"沙盒命令下发 {stats.sandbox_rounds} 回合")

    lines += ["", "### 1.4 每日总账（日志原文）", "", "```"]
    lines += stats.day_lines or [
        f"{PENDING}：本次日志未覆盖到一天结束（每130回合一行 day_summary）"
    ]
    lines += ["```", ""]

    filled = sum(
        1 for value in (
            stats.last.get("score", ""), hp_now, towers_n, stats.walls(),
            str(stats.kills), str(stats.gold_peak), str(stats.submits),
        )
        if value not in ("", PENDING, "0")
    )
    lines += [
        f"> 自动填充：{filled}/7 项（积分、基地血量、塔型与等级、围墙段数、"
        "击杀、金币、任务交卷）。",
        "> 敌方经济与塔型报文里不提供——敌方只有基地和围墙全图可见，"
        "其余要进视野，属正常缺失。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析客户端 debug.log")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="日志文件路径")
    parser.add_argument("--template", action="store_true",
                        help="按对战分析模板生成填空稿")
    parser.add_argument("--task", action="store_true", help="只打印任务链路")
    parser.add_argument("--out", default="", help="填空稿输出路径（默认打印到终端）")
    args = parser.parse_args()

    stats = analyze(Path(args.log))
    if stats is None:
        raise SystemExit(1)

    if args.template:
        report = render_template(stats)
        if args.out:
            Path(args.out).write_text(report, encoding="utf-8")
            print(f"分析稿已写入 {args.out}")
        else:
            print(report)
        raise SystemExit(0)

    if args.task:
        print_task_trace(stats)
        raise SystemExit(0)

    print_summary(stats, Path(args.log))


if __name__ == "__main__":
    main()
