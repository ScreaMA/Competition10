"""本地联调脚本：用真实请求样例验证客户端的响应格式与稳定性。

用法（在 CoreGeek 目录下）:
    python tools/local_check.py [port]

会做三件事：
1. 直接调用 brain.decide()，用 docs/request.txt 的真实报文校验输出格式；
2. 启动HTTP服务器，通过真实POST请求校验端到端链路；
3. 构造边界场景（缺字段、无基地、空角色、大量机器人等）验证不崩溃。
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.brain import decide  # noqa: E402
from agent.server import Handler  # noqa: E402

SAMPLE = ROOT.parent / "docs" / "request.txt"


def load_sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def check_format(commands: dict) -> None:
    """校验响应中每条指令的字段格式"""
    assert isinstance(commands, dict), "roleCommandMap 必须是字典"
    for role_id, command in commands.items():
        assert isinstance(role_id, str) and role_id.isdigit(), f"非法角色ID: {role_id}"
        assert isinstance(command, dict), f"指令必须是字典: {command}"
        assert "action" in command, f"指令缺少action: {command}"
        if "targetPos" in command:
            assert isinstance(command["targetPos"], list), "targetPos 必须是数组"
            for pos in command["targetPos"]:
                assert set(pos) == {"x", "y"}, f"坐标格式错误: {pos}"
        if command["action"] == "attack":
            assert "controllerId" in command, "attack必须带controllerId"
            assert isinstance(command["controllerId"], str), "controllerId必须是字符串"
        if command["action"] == "build":
            assert "name" in command, "build必须带name"


def stats(commands: dict) -> str:
    counts: dict[str, int] = {}
    for command in commands.values():
        counts[command["action"]] = counts.get(command["action"], 0) + 1
    return ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())) or "无指令"


def case_direct() -> None:
    """1. 直接调用决策函数"""
    payload = load_sample()
    commands = decide(payload)
    check_format(commands)
    print(f"[1] 直接调用 decide(): round={payload['roundNo']} -> {stats(commands)}")
    print(f"    {json.dumps(commands, ensure_ascii=False)}")


def case_http(port: int) -> None:
    """2. 端到端HTTP请求"""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    try:
        for round_no in (1, 71, 85, 1300):
            payload = load_sample()
            payload["roundNo"] = round_no
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/",
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode("utf-8")
                assert response.status == 200
            parsed = json.loads(raw)
            assert set(parsed) == {"roleCommandMap", "prompt", "executeCmd"}
            check_format(parsed["roleCommandMap"])
            phase = "白天" if (round_no - 1) % 130 < 70 else "夜晚"
            print(f"[2] HTTP round={round_no}({phase}) -> {stats(parsed['roleCommandMap'])}")
    finally:
        server.shutdown()
        server.server_close()


def case_robustness() -> None:
    """3. 边界场景：任何情况下都必须返回合法字典"""
    sample = load_sample()

    scenarios: dict[str, dict] = {}

    payload = copy.deepcopy(sample)
    payload["roundNo"] = 1
    scenarios["白天第1回合"] = payload

    payload = copy.deepcopy(sample)
    del payload["teamEnemy"]
    del payload["robot"]
    scenarios["缺少teamEnemy与robot"] = payload

    payload = copy.deepcopy(sample)
    payload["teamOur"]["roles"] = []
    scenarios["空角色列表"] = payload

    payload = copy.deepcopy(sample)
    payload["teamOur"]["roles"] = [
        role for role in payload["teamOur"]["roles"] if role["roleType"] != "station"
    ]
    scenarios["无基地"] = payload

    payload = copy.deepcopy(sample)
    payload["mapInfo"]["zones"] = []
    scenarios["无矿区与任务点"] = payload

    payload = copy.deepcopy(sample)
    payload["roundNo"] = 1
    payload["teamOur"]["goldNum"] = 0
    scenarios["白天且金币为0"] = payload

    payload = copy.deepcopy(sample)
    payload["robot"]["roles"] = []
    scenarios["夜晚无机器人"] = payload

    payload = copy.deepcopy(sample)
    payload["robot"]["roles"] = [
        {
            "id": 30000 + index,
            "pos": {"x": 20, "y": 24},
            "roleType": "bossRobot",
            "health": 800,
            "abnormalState": "",
            "targetTeam": "challenger",
        }
        for index in range(60)
    ]
    scenarios["夜晚大量机器人"] = payload

    payload = copy.deepcopy(sample)
    payload["teamOur"]["playerTasks"] = []
    scenarios["无任务点数据"] = payload

    for name, payload in scenarios.items():
        commands = decide(payload)
        check_format(commands)
        print(f"[3] {name} -> {stats(commands)}")


def case_soak(rounds: int = 1300) -> None:
    """4. 连续1300回合的决策耗时检查（复用同一份地图，逐回合推进）"""
    payload = load_sample()
    slowest = 0.0
    total = 0.0
    for round_no in range(1, rounds + 1):
        payload["roundNo"] = round_no
        started = time.perf_counter()
        commands = decide(payload)
        elapsed = time.perf_counter() - started
        total += elapsed
        slowest = max(slowest, elapsed)
        check_format(commands)
    print(
        f"[4] 连续{rounds}回合决策: 总耗时{total:.2f}s, "
        f"单回合最慢{slowest * 1000:.1f}ms, 平均{total / rounds * 1000:.2f}ms"
    )


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8124
    case_direct()
    case_http(port)
    case_robustness()
    case_soak()
    print("全部检查通过")


if __name__ == "__main__":
    main()
