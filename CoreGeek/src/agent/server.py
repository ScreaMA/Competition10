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

from .brain import decide

LOGGER = logging.getLogger(__name__)

# 单条 request_raw 日志的最大长度，超出则截断，避免日志文件过大
MAX_LOG_BODY_LENGTH = 5000

# 决策耗时告警阈值（设计文档6.6节：预留0.2秒缓冲）
DECISION_BUDGET_WARN_MS = 800

# 请求ID计数器
_request_id = 0


def _truncate(text: str, limit: int = MAX_LOG_BODY_LENGTH) -> str:
    """超长文本截断，保留长度信息"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


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

            # 2. 记录原始请求（INFO级别，单行JSON）
            LOGGER.info("request_raw id=%d path=%s bytes=%d body=%s",
                        req_id, self.path, length, _truncate(raw.decode("utf-8")))

            # 3. 解析JSON
            payload = json.loads(raw.decode("utf-8"))
            round_no = payload.get("roundNo", 0)

            # 4. 记录解析后的请求信息
            team_info = payload.get("teamOur", {})
            LOGGER.info("request_decoded id=%d round=%d team=%s team_type=%s roles=%d",
                        req_id, round_no,
                        team_info.get("teamId", "?"),
                        team_info.get("type", "?"),
                        len(team_info.get("roles", [])))

            # 5. 记录完整的格式化请求（DEBUG级别）
            LOGGER.debug("=" * 80)
            LOGGER.debug("REQUEST round %d:", round_no)
            LOGGER.debug(json.dumps(payload, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

            # 6. 调用决策引擎（返回指令与可选的LLM prompt）
            response, llm_prompt = decide(payload)

            # 7. 构建完整响应
            full_response = {
                "roleCommandMap": response,
                "prompt": llm_prompt,
                "executeCmd": "",  # 沙盒命令预留
            }

            # 8. 记录策略完成（含决策耗时）
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.info("strategy_done id=%d round=%d commands=%d elapsed=%.2fms",
                        req_id, round_no, len(response), elapsed_ms)
            if elapsed_ms > DECISION_BUDGET_WARN_MS:
                LOGGER.warning("decision slow at round %s: %.2fms",
                               round_no, elapsed_ms)

            # 9. 编码响应
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

            # 10. 记录原始响应（INFO级别，单行JSON）
            LOGGER.info("response_raw id=%d body=%s", req_id, body.decode("utf-8"))

            # 11. 记录完整的格式化响应（DEBUG级别）
            LOGGER.debug("RESPONSE round %d:", round_no)
            LOGGER.debug(json.dumps(full_response, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

        except Exception:
            # 12. 异常处理：返回空指令
            LOGGER.exception("decision failed at round %s", round_no)
            full_response = {
                "roleCommandMap": {},
                "prompt": "",
                "executeCmd": "",
            }
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

        # 13. 发送HTTP响应
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

        # 14. 记录响应发送完成
        LOGGER.info("response_sent id=%d status=200 bytes=%d", req_id, len(body))

    def log_message(self, format: str, *args: Any) -> None:
        """屏蔽默认的HTTP日志"""
        return


def serve(port: int) -> None:
    """启动HTTP服务器"""
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()
