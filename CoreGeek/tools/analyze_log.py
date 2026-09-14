#!/usr/bin/env python3
"""分析 debug.log，统计回合处理情况与每回合决策耗时。

用法:
    python CoreGeek/tools/analyze_log.py                 # 默认分析 CoreGeek/debug.log
    python CoreGeek/tools/analyze_log.py --log x.log
    python CoreGeek/tools/analyze_log.py --template      # 按对战分析模板生成填空稿

`--template` 会把日志渲染成 `战术参考/对战分析模板.md` 的填空版：
能自动算出来的直接填上（积分、基地血量、塔型与等级、围墙等级分布、
金币峰值/终值、击杀数、任务交卷数），算不出来的留 `{待人工}`。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "CoreGeek" / "debug.log"
TEMPLATE_PATH = ROOT / "战术参考" / "对战分析模板.md"

ROUND_RE = re.compile(r"request_decoded id=(\d+) round=(\d+)")
KV_RE = re.compile(r"(\w+)=(\S+)")
DONE_RE = re.compile(
    r"strategy_done id=(\d+) round=(\d+) commands=(\d+) elapsed=([\d.]+)ms"
    r"(?: sandbox=(\S+))?(?: gold_spent=(\d+))?(?: actions=(.*))?"
)
END_RE = re.compile(r"round_end id=(\d+) round=(\d+)")
DAY_RE = re.compile(r"day_summary id=(\d+) day=(\d+)")
SENT_RE = re.compile(r"response_sent id=(\d+) status=(\d+) bytes=(\d+)")
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b")
SLOW_RE = re.compile(r"decision slow at round (\d+)")
TOWER_RE = re.compile(r"^(\d+)\[(.*)\]$")
WALL_RE = re.compile(r"^(\d+)\[(.*)\]$")
HP_RE = re.compile(r"^(\d+)/(\d+)$")

PENDING = "{待人工}"


class RoundStats:
    """逐回合累积的统计（只做汇总，不做决策）"""

    def __init__(self) -> None:
        self.rounds: set[int] = set()
        self.gold_series: list[tuple[int, int]] = []
        self.elapsed: list[float] = []
        self.slow_rounds: list[int] = []
        self.last: dict[str, str] = {}
        self.team_type = ""
        self.team_id = ""
        self.kills = 0
        self.loses = 0
        self.task_subs = 0
        self.task_accepts = 0
        self.day_lines: list[str] = []
        self.day_peaks: list[int] = []
        self.events: list[tuple[int, str]] = []
        self.statuses: dict[int, int] = {}
        self.errors: list[str] = []
        self.truncated = 0

    # === 汇总 ===

    @property
    def first_round(self) -> int:
        return min(self.rounds) if self.rounds else 0

    @property
    def last_round(self) -> int:
        return max(self.rounds) if self.rounds else 0

    @property
    def gold_peak(self) -> int:
        """金币峰值：逐回合 gold 序列与每日总账里的 gold_peak 取最大"""
        peaks = [gold for _, gold in self.gold_series] + self.day_peaks
        return max(peaks, default=0)

    @property
    def gold_final(self) -> int:
        return self.last.get("gold", "0") and int(self.last["gold"])

    def hp_pair(self) -> tuple[str, str]:
        raw = self.last.get("hp", "")
        match = HP_RE.match(raw)
        if not match:
            return PENDING, PENDING
        return match.group(1), match.group(2)

    def towers(self) -> tuple[str, str]:
        match = TOWER_RE.match(self.last.get("towers", ""))
        if not match:
            return PENDING, PENDING
        return match.group(1), match.group(2) or PENDING

    def walls(self) -> tuple[str, str]:
        match = WALL_RE.match(self.last.get("walls", ""))
        if not match:
            return PENDING, PENDING
        return match.group(1), match.group(2) or PENDING

    def kills_text(self) -> str:
        return str(self.kills) if self.rounds else PENDING


def _add_event(stats: RoundStats, round_no: int, text: str, limit: int = 40) -> None:
    """记录时间线上的关键节点（同一类事件只留第一次）"""
    if any(existing == text for _, existing in stats.events):
        return
    if len(stats.events) < limit:
        stats.events.append((round_no, text))


def _scan_actions(stats: RoundStats, round_no: int, actions: str) -> None:
    """从 actions= 里挑出值得进时间线的动作

    动作之间用空格分隔，但动作内部也可能带空格（如 `build→(11,22) rocket`），
    所以按"角色ID:"的位置切分，而不是按空格切。
    """
    for chunk in re.split(r"(?=\d+:)", actions):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        _, command = chunk.split(":", 1)
        target = command.split("→")[1].split()[0] if "→" in command else ""
        if command.startswith("build"):
            parts = command.split()
            name = parts[-1] if len(parts) > 1 else ""
            label = "围墙" if name == "wall" else "武器塔"
            where = f"→{target}" if target else ""
            _add_event(stats, round_no, f"{label} {name}{where}".strip())
        elif command.startswith("sell"):
            _add_event(stats, round_no, "首次卖矿换金币")
        elif command.startswith("buy"):
            _add_event(stats, round_no, f"买道具 {command.split()[-1]}")
        elif command.startswith("use"):
            _add_event(stats, round_no, f"用道具 {command.split()[-1]}")
        elif command.startswith("acceptTask"):
            stats.task_accepts += 1
            _add_event(stats, round_no, "领任务")
        elif command.startswith("submitAnswer"):
            stats.task_subs += 1
            _add_event(stats, round_no, "交卷")


def analyze(log_file: Path) -> RoundStats | None:
    """解析日志并返回统计；日志为空时返回 None"""
    if not log_file.is_file():
        print(f"日志文件不存在: {log_file}")
        return None

    stats = RoundStats()
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "truncated" in line and "request_raw" in line:
                stats.truncated += 1
            if ERROR_RE.search(line):
                stats.errors.append(line.strip()[:200])

            if "request_decoded" in line:
                fields = dict(KV_RE.findall(line))
                round_no = int(fields.get("round", 0))
                stats.rounds.add(round_no)
                stats.team_type = stats.team_type or fields.get("team_type", "")
                stats.team_id = stats.team_id or fields.get("team", "")
                stats.last = fields
                stats.gold_series.append((round_no, int(fields.get("gold", 0))))
                continue

            match = DONE_RE.search(line)
            if match:
                stats.elapsed.append(float(match.group(4)))
                if match.group(7):
                    _scan_actions(stats, int(match.group(2)), match.group(7))
                continue

            match = END_RE.search(line)
            if match:
                fields = dict(KV_RE.findall(line))
                kills = fields.get("kills", "0/0").split("/")[0]
                stats.kills += int(kills) if kills.isdigit() else 0
                loses = fields.get("loses", "0")
                stats.loses += int(loses) if loses.isdigit() else 0
                continue

            match = DAY_RE.search(line)
            if match:
                stats.day_lines.append(line.strip())
                peak = dict(KV_RE.findall(line)).get("gold_peak", "")
                if peak.isdigit():
                    stats.day_peaks.append(int(peak))
                continue

            match = SENT_RE.search(line)
            if match:
                status = int(match.group(2))
                stats.statuses[status] = stats.statuses.get(status, 0) + 1
                continue

            match = SLOW_RE.search(line)
            if match:
                stats.slow_rounds.append(int(match.group(1)))

    return stats


def print_summary(stats: RoundStats, log_file: Path) -> None:
    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"文件大小: {log_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    if not stats.rounds:
        print("未解析到回合记录（日志可能为空或格式不符）")
        return

    print(f"处理回合数: {len(stats.rounds)}")
    print(f"回合范围  : {stats.first_round} - {stats.last_round}")
    if stats.elapsed:
        values = sorted(stats.elapsed)
        print(f"决策耗时  : 平均 {sum(values) / len(values):.2f}ms"
              f"，中位 {values[len(values) // 2]:.2f}ms"
              f"，最慢 {values[-1]:.2f}ms")
        if stats.slow_rounds:
            print(f"超时预警回合(>{len(stats.slow_rounds)}个): {stats.slow_rounds[:10]}")
    print(f"金币      : 峰值 {stats.gold_peak}，终值 {stats.gold_final}")
    print(f"击杀/损失 : {stats.kills} / {stats.loses}")
    print(f"任务      : 领任务 {stats.task_accepts} 次，交卷 {stats.task_subs} 次")
    if stats.statuses:
        print(f"HTTP状态  : {stats.statuses}")
    if stats.truncated:
        print(f"截断的请求日志条目: {stats.truncated}")
    print(f"错误条数  : {len(stats.errors)}")
    for line in stats.errors[:10]:
        print(f"  {line}")


def render_template(stats: RoundStats) -> str:
    """把统计渲染成对战分析模板的填空稿

    返回:
        Markdown 文本；日志为空时给出提示
    """
    if not stats.rounds:
        return "（日志为空，无法生成分析稿）"

    hp_now, hp_max = stats.hp_pair()
    towers_n, towers_detail = stats.towers()
    walls_n, walls_detail = stats.walls()
    enemy_towers = stats.last.get("enemy_towers", PENDING)
    enemy_walls = stats.last.get("enemy_walls", PENDING)
    enemy_score = stats.last.get("enemy_score", PENDING)
    enemy_visible = stats.last.get("enemy_visible", PENDING)

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
        f"| **最终积分** | {stats.last.get('score', PENDING)} | {enemy_score} | {PENDING} |",
        f"| **基地血量** | {hp_now} / {hp_max} | {PENDING} | {PENDING} |",
        f"| **武器塔数量** | {towers_n}座（{towers_detail}） | {enemy_towers}座"
        f"（可见单位{enemy_visible}） | {PENDING} |",
        f"| **围墙段数** | {walls_n}段（{walls_detail}） | {enemy_walls}段"
        f"（敌方围墙全图可见） | {PENDING} |",
        f"| **任务完成** | 交卷{stats.task_subs}次/领{stats.task_accepts}次 | {PENDING} | {PENDING} |",
        f"| **金币累积** | 峰值{stats.gold_peak}，终值{stats.gold_final} | {PENDING} | {PENDING} |",
        f"| **机器人击杀** | {stats.kills_text()} | {PENDING} | {PENDING} |",
        f"| **单位损失** | {stats.loses} | {PENDING} | {PENDING} |",
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

    lines += ["", "### 1.3 每日总账（日志原文）", "", "```"]
    lines += stats.day_lines or [f"{PENDING}：本次日志未覆盖到一天结束（每130回合一行）"]
    lines += ["```", ""]

    filled = sum(
        1 for value in (
            stats.last.get("score", ""), hp_now, towers_n, walls_n,
            stats.kills_text(), stats.gold_peak, stats.task_subs,
        )
        if value not in ("", PENDING, "0")
    )
    lines += [
        f"> 自动填充：{filled}/7 项（最终积分、基地血量、塔型与等级、围墙等级分布、"
        "击杀、金币峰值/终值、任务交卷）。",
        "> 敌方经济（金币/任务/击杀）与敌方塔型等级报文里不提供——敌方只有基地和"
        "围墙全图可见，其余要进视野，属正常缺失。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析客户端 debug.log")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="日志文件路径")
    parser.add_argument("--template", action="store_true",
                        help="按对战分析模板生成填空稿")
    parser.add_argument("--out", default="", help="填空稿输出路径（默认打印到终端）")
    args = parser.parse_args()

    log_file = Path(args.log)
    stats = analyze(log_file)
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

    print_summary(stats, log_file)


if __name__ == "__main__":
    main()
