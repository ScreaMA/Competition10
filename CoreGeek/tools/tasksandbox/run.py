#!/usr/bin/env python3
"""本地任务沙盒的命令行入口。

用法（在 CoreGeek 目录下）:

    python tools/tasksandbox/run.py                     # 全部题目跑一遍
    python tools/tasksandbox/run.py --family api        # 只跑 API 查询族
    python tools/tasksandbox/run.py --family fix        # 只跑工程修复族
    python tools/tasksandbox/run.py --city 西安          # 只跑某座城
    python tools/tasksandbox/run.py --variant beta      # 只跑某个修复变体
    python tools/tasksandbox/run.py --mode stale        # 服务端按"文档口径"校验
    python tools/tasksandbox/run.py --keep .tasksandbox # 留下目录，不删

它回答的问题是：**照现在这份客户端代码，题做得完吗？**

    A 族跑通 = `[ANSWER]` 与 `apiserver.expected_answer()` 逐字段相同
    B 族跑通 = `./check` 输出 `TOKEN:`，且客户端的 `verify` 步抓到了它

这也是**每次改完客户端都要跑一遍**的那条命令（`tests/test_tasksandbox.py`
是同一个判据的 pytest 版本）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))          # agent.*
sys.path.insert(0, str(ROOT))                  # tools.tasksandbox.*

from tools.tasksandbox import env, fix_tasks, runner                    # noqa: E402
from tools.tasksandbox.apiserver import (                               # noqa: E402
    CITIES, HeritageAPI, expected_answer,
)


def _banner(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def run_api(sandbox: env.Sandbox, city: str) -> bool:
    """家族 A：recon → query → 答案对账"""
    _banner("家族 A · api-query · %s" % city)

    phase = sandbox.api_phase(city)
    recon = runner.run_step("recon", sandbox, phase_task=phase)
    query = runner.run_step("query", sandbox, phase_task=phase)

    for line in query.output.splitlines():
        if line.startswith(("[QUERY]", "[PARSE]", "[API]", "[SCAN]", "[DATA]",
                            "[ANSWER]", "[PROFILE]")):
            print("  " + line[:150])
    failures = [line[:150] for line in query.output.splitlines()
                if line.startswith("[APIFAIL]")]
    if failures:
        print("  [APIFAIL] ×%d，例如：%s" % (len(failures), failures[0]))
    if query.errors():
        print("  脚本错误: %s" % query.errors())
        return False

    found = query.answer()
    if found is None:
        print("  ✗ 没有产出 [ANSWER]")
        return False

    ok = True
    for key, value in expected_answer(city).items():
        got = found.get(key)
        if got != value:
            ok = False
        print("  %s %-22s 期望 %-26s 实际 %s"
              % ("✓" if got == value else "✗", key, value, got))
    print("  => %s" % ("通过" if ok else "**没通过**"))
    return ok


def run_fix(sandbox: env.Sandbox, variant: fix_tasks.FixVariant) -> bool:
    """家族 B：recon → normalize → check → repair → verify → TOKEN"""
    _banner("家族 B · engineering-fix · %s" % variant.code)

    facts: dict[str, str] = {}
    recon = runner.run_step("recon", sandbox, phase_task=fix_tasks.phase_of(variant),
                            cwd=sandbox.ws_root)
    _learn(recon, facts)

    # 工作区三个路径用**相对形式**，并把 cwd 钉在 ws 上。
    # 真沙盒里它们是从侦察结果学来的绝对路径（`/tmp/selfEvolutionTask/…`），
    # 走的代码路径完全一样；本地只能是相对形式——Git Bash 的 `sh` 执行不了
    # Windows 绝对路径（`C:\Users\…` 的反斜杠会被当成转义符吃掉一层）。
    # 这是**脚手架**的妥协，不是客户端的。
    params = {"ws": ".", "spec": "spec.md", "check": "./check"}
    phase = fix_tasks.phase_of(variant)

    norm = runner.run_step("normalize", sandbox, phase_task=phase,
                           facts=facts, params=params, cwd=sandbox.ws)
    print("  [normalize] %s" % norm.markers.get("FIX", "-"))

    check = runner.run_step("check", sandbox, phase_task=phase,
                            facts=facts, params=params, cwd=sandbox.ws)
    print("  [check    ] ok=%s code=%s"
          % (check.kv("CHECK.ok", "?"), check.kv("CHECK.code", "?")))

    repair = runner.run_step("repair", sandbox, phase_task=phase,
                             facts=facts, params=params,
                             check_output=check.markers.get("CHECKBODY", ""),
                             cwd=sandbox.ws)
    print("  [repair   ] %s" % repair.markers.get("FIX", "-"))

    verify = runner.run_step("verify", sandbox, phase_task=phase,
                             facts=facts, params=params, cwd=sandbox.ws)
    token = verify.token()
    code, out = sandbox.run_check()
    ok = token == variant.token and sandbox.check_passes()
    print("  token=%s（期望 %s）  ./check 本身%s通过"
          % (token, variant.token, "" if sandbox.check_passes() else "**没**"))
    print("  => %s" % ("通过" if ok else "**没通过**"))
    return ok


def _learn(result: runner.StepResult, facts: dict[str, str]) -> None:
    """把侦察回来的路径事实并进 facts（等价于 `solver.learn_from_output`）"""
    from agent.strategy.task.memory import (
        F_CHECK_CMD, F_ROOT, F_SPEC_PATH, F_WS_ROOT,
    )

    pairs = {
        "RECON.root": F_ROOT,
        "RECON.ws": F_WS_ROOT,
        "FIND.ws": F_WS_ROOT,
        "FIND.spec": F_SPEC_PATH,
        "FIND.check": F_CHECK_CMD,
    }
    for source, key in pairs.items():
        value = result.kv(source)
        if value:
            facts[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="本地任务沙盒")
    parser.add_argument("--family", choices=("api", "fix", "both"), default="both")
    parser.add_argument("--city", default="", help="只跑这一座城（默认全部）")
    parser.add_argument("--variant", default="", help="只跑这个修复变体（默认全部）")
    parser.add_argument("--mode", choices=("real", "stale"), default="real",
                        help="real=真接口口径（§6.2）；stale=文档口径")
    parser.add_argument("--keep", default="", help="把沙盒目录留在这个路径")
    args = parser.parse_args()

    cities = (args.city,) if args.city else CITIES
    variants = (
        [v for v in (fix_tasks.KNOWN_VARIANT,) + fix_tasks.NEW_VARIANTS
         if v.code == args.variant] if args.variant
        else [fix_tasks.KNOWN_VARIANT] + list(fix_tasks.NEW_VARIANTS)
    )

    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="tasksandbox-"))
    results: list[tuple[str, bool]] = []
    try:
        with HeritageAPI(mode=args.mode) as api:
            print("服务端口径: %s（%s）" % (args.mode, api.base))
            print("沙盒目录: %s" % root)
            for city in cities:
                if args.family not in ("api", "both"):
                    break
                sandbox = env.Sandbox(root, api.base).build(cities=(city,))
                results.append(("api/%s" % city, run_api(sandbox, city)))
            for variant in variants:
                if args.family not in ("fix", "both"):
                    break
                sandbox = env.Sandbox(root, api.base).build(
                    cities=("北京",), variant=variant
                )
                results.append(("fix/%s" % variant.code, run_fix(sandbox, variant)))
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    print()
    print("=" * 68)
    for name, ok in results:
        print("  %s %s" % ("✓" if ok else "✗", name))
    failed = [name for name, ok in results if not ok]
    print("结果: %d/%d 通过%s"
          % (len(results) - len(failed), len(results),
             "" if not failed else "，失败：" + "、".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
