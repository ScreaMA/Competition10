"""服务器模块：HTTP服务器，接收判题系统的POST请求。

对应设计文档 3.2 节。

注意（实测踩坑）：判题系统是向 `POST /` 发送请求的，部分实现会只在
`POST /action` 上路由而收不到消息。本模块不检查 `self.path`，
任何路径的POST请求都会被正常处理。
"""

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide, sandbox_command
from .protocol import COPPER_MINE, IRON_MINE, STONE_MINE, Turn

LOGGER = logging.getLogger(__name__)

# 沙盒输出预览的最大长度：任务相关的线索留这么多就够，不打全量报文
MAX_SANDBOX_PREVIEW = 300

# 交卷内容在动作日志里的预览长度
ANSWER_PREVIEW = 80

# 机器人的种类顺序，概览里的 s/m/l/b 与接口文档的四种机器人一一对应
ROBOT_KINDS = ("smallRobot", "middleRobot", "largeRobot", "bossRobot")

# 决策耗时告警阈值（设计文档6.6节：预留0.2秒缓冲）
DECISION_BUDGET_WARN_MS = 800

# 请求ID计数器
_request_id = 0


def _truncate(text: str, limit: int = MAX_SANDBOX_PREVIEW) -> str:
    """超长文本截断，保留长度信息"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _one_line(text: str, limit: int = MAX_SANDBOX_PREVIEW) -> str:
    """把多行文本压成一行并截断（日志按行解析，多行会把一条记录拆散）"""
    return _truncate(" | ".join(line for line in text.splitlines() if line.strip()), limit)


def _base_brief(turn: Turn) -> str:
    """己方基地坐标"""
    station = turn.station()
    if station is None:
        return "-"
    return f"({station.pos.x},{station.pos.y})"


def _robot_brief(turn: Turn) -> str:
    """机器人数量与构成（s/m/l/b 依次为小/中/大/BOSS）"""
    counts = " ".join(
        f"{kind[0]}{sum(1 for robot in turn.robots if robot.kind == kind)}"
        for kind in ROBOT_KINDS
    )
    return f"{len(turn.robots)}({counts})"


def _task_brief(turn: Turn) -> str:
    """任务点与当前任务描述（自进化类任务的进度全看这两个字段）"""
    parts = []
    for task in turn.player_tasks:
        state = "可接" if task.is_valid else f"冷却{task.cold_down_rounds}"
        parts.append(
            f"{task.task_type or '任务'}"
            f"@({task.task_position.x},{task.task_position.y})"
            f"{state}/{task.timeout_rounds}回合"
        )
    phase = f'phase="{turn.phase_task}"' if turn.phase_task else "phase=-"
    return f"tasks=[{' '.join(parts) or '-'}] {phase}"


def _failed_brief(turn: Turn) -> str:
    """上一回合执行失败的角色（报文的 lastRoundRoleActionResults）"""
    failed = [
        str(role_id)
        for role_id, success in sorted(turn.last_action_results.items())
        if not success
    ]
    return f"fail=[{' '.join(failed)}]"


def _role_order(role_id: Any) -> tuple[int, int, str]:
    """角色ID排序键：数字ID按数值排，非数字ID排在后面（日志不该因脏ID报错）"""
    text = str(role_id)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def _command_brief(command: dict[str, Any]) -> str:
    """单条指令的摘要，如 collect→(25,3)、build→(11,22) railgun、submitAnswer"""
    brief = str(command.get("action") or "?")
    targets = command.get("targetPos") or []
    if targets:
        brief += "→" + ",".join(f"({pos['x']},{pos['y']})" for pos in targets)
    if command.get("name"):
        brief += f" {command['name']}"
    if "num" in command:
        brief += f" x{command['num']}"
    if command.get("taskAnswer"):
        # 交卷内容也留一小段（任务执行成败就在这里），全量内容不打印
        brief += f' "{_truncate(str(command["taskAnswer"]), ANSWER_PREVIEW)}"'
    return brief


def _actions_brief(commands: dict[str, dict[str, Any]]) -> str:
    """本回合下发的关键动作（角色ID:动作+目标）"""
    if not commands:
        return "-"
    return " ".join(
        f"{role_id}:{_command_brief(command)}"
        for role_id, command in sorted(commands.items(), key=lambda item: _role_order(item[0]))
    )


class Handler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def do_POST(self) -> None:
        """处理POST请求（不区分路径，`/` 与 `/action` 均处理）"""
        global _request_id
        _request_id += 1
        req_id = _request_id
        round_no: Any = "?"
        started = time.perf_counter()

        try:
            # 1. 读取请求体
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)

            # 2. 解析JSON
            payload = json.loads(raw.decode("utf-8"))
            round_no = payload.get("roundNo", 0)
            team_info = payload.get("teamOur", {})
            turn = Turn.load(payload)

            # 3. 记录本回合的资源与任务概览（请求全量报文只在 debug.log 里留档）
            LOGGER.info(
                "request_decoded id=%d round=%d team=%s team_type=%s roles=%d "
                "gold=%d score=%d base=%s towers=%d walls=%d robots=%s mines=%d "
                "%s %s",
                req_id, round_no, team_info.get("teamId", "?"),
                turn.team_type, len(turn.ours),
                turn.gold, turn.total_score, _base_brief(turn),
                len(turn.weapons()), len(turn.walls()), _robot_brief(turn),
                sum(
                    1 for kind in turn.zones.values()
                    if kind in (STONE_MINE, IRON_MINE, COPPER_MINE)
                ),
                _task_brief(turn), _failed_brief(turn),
            )

            # 4. 任务期间的沙盒输出是任务成败的唯一线索，单独留一行预览
            if turn.last_cmd_result:
                LOGGER.info("sandbox_result id=%d round=%d text=%s",
                            req_id, round_no, _one_line(turn.last_cmd_result))

            # 5. 记录完整的格式化请求（DEBUG级别，只写 debug.log）
            LOGGER.debug("=" * 80)
            LOGGER.debug("REQUEST round %d:", round_no)
            LOGGER.debug(json.dumps(payload, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

            # 6. 调用决策引擎（返回指令与可选的LLM prompt）
            response, llm_prompt = decide(payload)

            # 7. 构建完整响应（executeCmd 仅在自进化任务期间有内容）
            sandbox_cmd = sandbox_command(payload)
            full_response = {
                "roleCommandMap": response,
                "prompt": llm_prompt,
                "executeCmd": sandbox_cmd,  # 沙盒命令
            }

            # 8. 记录策略完成（含决策耗时与关键动作）
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.info("strategy_done id=%d round=%d commands=%d elapsed=%.2fms "
                        "sandbox=%s actions=%s",
                        req_id, round_no, len(response), elapsed_ms,
                        "下发" if sandbox_cmd else "空闲", _actions_brief(response))
            if elapsed_ms > DECISION_BUDGET_WARN_MS:
                LOGGER.warning("decision slow at round %s: %.2fms",
                               round_no, elapsed_ms)

            # 9. 编码响应
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

            # 10. 记录完整的格式化响应（DEBUG级别，只写 debug.log）
            LOGGER.debug("RESPONSE round %d:", round_no)
            LOGGER.debug(json.dumps(full_response, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

        except Exception:
            # 11. 异常处理：返回空指令
            LOGGER.exception("decision failed at round %s", round_no)
            full_response = {
                "roleCommandMap": {},
                "prompt": "",
                "executeCmd": "",
            }
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

        # 12. 发送HTTP响应
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """屏蔽默认的HTTP日志"""
        return


def serve(port: int) -> None:
    """启动HTTP服务器"""
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()
