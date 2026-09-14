#!/usr/bin/env python3
"""分析 debug.log，统计回合处理情况与每回合决策耗时。

用法:
    python CoreGeek/tools/analyze_log.py                 # 默认分析 CoreGeek/debug.log
    python CoreGeek/tools/analyze_log.py --log x.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "CoreGeek" / "debug.log"

ROUND_RE = re.compile(r"request_decoded id=(\d+) round=(\d+)")
DONE_RE = re.compile(
    r"strategy_done id=(\d+) round=(\d+) commands=(\d+) elapsed=([\d.]+)ms"
    r" sandbox=(\S+)"
)
SANDBOX_RE = re.compile(r"sandbox_result id=(\d+) round=(\d+)")
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b")
SLOW_RE = re.compile(r"decision slow at round (\d+)")


def analyze(log_file: Path) -> int:
    """返回进程退出码（0=正常）"""
    if not log_file.is_file():
        print(f"日志文件不存在: {log_file}")
        return 1

    rounds: dict[int, int] = {}      # request_id -> round
    processed: dict[int, int] = {}   # round -> commands
    elapsed: dict[int, float] = {}   # round -> 耗时ms
    sandbox: dict[int, str] = {}     # round -> 沙盒命令状态（下发/空闲）
    errors: list[str] = []
    slow_rounds: list[int] = []
    sandbox_results = 0

    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ERROR_RE.search(line):
                errors.append(line.strip()[:200])

            match = ROUND_RE.search(line)
            if match:
                rounds[int(match.group(1))] = int(match.group(2))
                continue

            match = DONE_RE.search(line)
            if match:
                round_no = int(match.group(2))
                processed[round_no] = int(match.group(3))
                elapsed[round_no] = float(match.group(4))
                sandbox[round_no] = match.group(5)
                continue

            if SANDBOX_RE.search(line):
                sandbox_results += 1
                continue

            match = SLOW_RE.search(line)
            if match:
                slow_rounds.append(int(match.group(1)))

    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"文件大小: {log_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    if not processed:
        print("未解析到回合记录（日志可能为空或格式不符）")
        return 1

    numbers = sorted(processed)
    print(f"处理回合数: {len(processed)}")
    print(f"回合范围  : {numbers[0]} - {numbers[-1]}")
    print(f"指令总数  : {sum(processed.values())}"
          f"（平均 {sum(processed.values()) / len(processed):.1f}/回合）")
    print(f"零指令回合: {sum(1 for count in processed.values() if count == 0)}")

    if elapsed:
        values = sorted(elapsed.values())
        print(f"决策耗时  : 平均 {sum(values) / len(values):.2f}ms"
              f"，中位 {values[len(values) // 2]:.2f}ms"
              f"，最慢 {values[-1]:.2f}ms")
        if slow_rounds:
            print(f"超时预警回合(>{len(slow_rounds)}个): {slow_rounds[:10]}")

    if sandbox:
        counts = {}
        for state in sandbox.values():
            counts[state] = counts.get(state, 0) + 1
        print(f"沙盒命令  : {counts}（下发=任务期间提交了 executeCmd）")
    if sandbox_results:
        print(f"沙盒输出  : {sandbox_results} 条（任务期间的 lastCmdResult 预览）")

    print(f"错误条数  : {len(errors)}")
    for line in errors[:10]:
        print(f"  {line}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="分析客户端 debug.log")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="日志文件路径")
    args = parser.parse_args()
    raise SystemExit(analyze(Path(args.log)))


if __name__ == "__main__":
    main()
