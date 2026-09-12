"""服务器模块：HTTP服务器，接收判题系统的POST请求。

对应设计文档 3.2 节。
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide

LOGGER = logging.getLogger(__name__)

# 请求ID计数器
_request_id = 0


class Handler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def do_POST(self) -> None:
        """处理POST请求"""
        global _request_id
        _request_id += 1
        req_id = _request_id
        round_no: Any = "?"

        try:
            # 1. 读取请求体
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)

            # 2. 记录原始请求（INFO级别，单行JSON）
            LOGGER.info("request_raw id=%d bytes=%d body=%s",
                        req_id, length, raw.decode("utf-8"))

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

            # 6. 调用决策引擎
            response = decide(payload)

            # 7. 构建完整响应
            full_response = {
                "roleCommandMap": response,
                "prompt": "",  # LLM调用预留
                "executeCmd": "",  # 沙盒命令预留
            }

            # 8. 记录策略完成
            LOGGER.info("strategy_done id=%d round=%d commands=%d",
                        req_id, round_no, len(response))

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
