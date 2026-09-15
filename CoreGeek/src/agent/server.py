"""HTTP 服务：接收判题系统的报文，返回本回合的调度指令。

对应设计文档V2 §3.1 与第 9 章。

接口文档 §1：「每局比赛将双方选手代码同时启动，选手代码通过 httpserver 方式
启动运行并监听端口，判题系统通过 http 请求方式向选手程序发送当前地图状态
数据并获取选手调度命令响应数据。」

两个硬要求：

- **响应必须快**。请求超时 5 秒判超时（任务书 §8），累计 5 次异常整场停调度。
  `decide` 是纯计算，实测在毫秒级；这里额外做了一层超时保护——决策哪怕
  因为边界数据卡住，也要在 [`DECIDE_TIMEOUT`] 秒内返回空指令，而不是让判题器
  等超时。
- **任何路径都要响应**。判题系统发往 `POST /`，但测试脚本会打 `/action`，
  这里对所有路径一视同仁。
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide
from . import protocol

LOGGER = logging.getLogger("agent.server")

# 决策的软超时（秒）。判题器给的窗口是 5 秒，留足余量。
DECIDE_TIMEOUT = 2.0

# 请求体的上限（正常报文 ~10KB，超出的直接拒掉，避免被畸形输入拖住）
MAX_BODY = 4 * 1024 * 1024

# 超限时最多把请求体读掉多少（读完再回，避免客户端还在写就被重置连接）。
# 再大就直接关连接——那种请求不可能是判题器发来的。
DRAIN_LIMIT = 16 * 1024 * 1024

_state = {"count": 0, "last_round": -1, "lock": threading.Lock()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CoreGeek/2.0"

    # --- 路由 ---

    def do_POST(self) -> None:
        payload = self._read_payload()
        if payload is None:
            self._reply(protocol.EMPTY_RESPONSE)
            return
        response = self._decide_guarded(payload)
        self._reply(response)

    def do_GET(self) -> None:
        """健康检查：判题系统与联调脚本都可能先探一下服务是否起来了"""
        body = b'{"status":"ok","client":"coregeek-v2"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- 内部 ---

    def _read_payload(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            return None
        if length <= 0:
            return None
        if length > MAX_BODY:
            # 先把请求体读掉再回包：直接回包会让还在写数据的客户端吃到
            # 连接重置（Windows 上是 WinError 10053），而判题器把这种
            # 情况算作"响应格式错误"，是要吃异常预算的。
            self._drain(length)
            return None

        raw = self.rfile.read(length)
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return json.loads(raw.decode(encoding))
            except (UnicodeDecodeError, ValueError):
                continue
        LOGGER.error("bad payload (%d bytes)", length)
        return None

    def _drain(self, length: int) -> None:
        """把超限的请求体读掉（有上限），必要时关连接"""
        remaining = length
        if length > DRAIN_LIMIT:
            remaining = DRAIN_LIMIT
            self.close_connection = True
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _decide_guarded(self, payload: dict[str, Any]) -> dict[str, Any]:
        """带超时保护的决策

        超时不是"异常响应"（任务书 §8 的三类异常里没有这一条），但让它发生
        也毫无意义——返回空指令至少保证协议完整、不吃判题器的超时。
        """
        round_no = int(payload.get("roundNo") or 0)
        box: dict[str, Any] = {}

        def run() -> None:
            box["response"] = decide(payload)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(DECIDE_TIMEOUT)
        if worker.is_alive():
            LOGGER.error("decide timeout round=%s", round_no)
            return dict(protocol.EMPTY_RESPONSE)

        with _state["lock"]:
            _state["count"] += 1
            _state["last_round"] = round_no
        return box.get("response", dict(protocol.EMPTY_RESPONSE))

    def _reply(self, response: dict[str, Any]) -> None:
        body = protocol.dumps(response).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # 判题器提前断开：这一回合的响应没送出去，但不值得把服务拖垮
            LOGGER.warning("client disconnected before response")

    def log_message(self, format: str, *args: Any) -> None:
        """屏蔽 BaseHTTPRequestHandler 的默认访问日志（它只往 stderr 打）"""
        return


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    LOGGER.info("listening on 0.0.0.0:%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        server.server_close()


def stats() -> dict[str, Any]:
    with _state["lock"]:
        return dict(_state)
