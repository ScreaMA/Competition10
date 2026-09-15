"""HTTP 服务用例。

对应设计文档V2 §3.1。

接口文档 §1 只规定了一件事：判题系统用 http POST 把地图状态发过来，拿回
调度命令。所以这里的用例也就围绕三件事：

1. **响应快**（请求超时 5 秒判超时，任务书 §8）
2. **任何路径都响应**（判题系统打 `/`，联调脚本还会打 `/action`）
3. **畸形输入不把服务搞挂**（服务挂了这一整场比赛就没了）
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agent import protocol, server


@pytest.fixture
def live_server():
    """起一个真实端口的服务，用完关掉"""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def _post(url: str, body: bytes, content_type="application/json"):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_post_root_returns_valid_response(live_server, payload_factory, base_roles):
    payload = payload_factory(round_no=3, roles=base_roles)
    status, response = _post(live_server + "/", protocol.dumps(payload).encode("utf-8"))
    assert status == 200
    assert set(response) == {"roleCommandMap", "prompt", "executeCmd"}


def test_any_path_is_accepted(live_server, payload_factory, base_roles):
    """判题系统打 `/`，联调脚本还会打 `/action`，两者都要响应"""
    payload = payload_factory(round_no=3, roles=base_roles)
    body = protocol.dumps(payload).encode("utf-8")
    for path in ("/action", "/api/decide", "/"):
        status, response = _post(live_server + path, body)
        assert status == 200
        assert "roleCommandMap" in response


def test_get_is_health_check(live_server):
    with urllib.request.urlopen(live_server + "/", timeout=5) as response:
        assert response.status == 200
        assert json.loads(response.read())["status"] == "ok"


def test_malformed_json_returns_empty_response(live_server):
    """畸形报文也要返回合法响应（返回空指令不是异常）"""
    status, response = _post(live_server + "/", b"{not json")
    assert status == 200
    assert response == protocol.EMPTY_RESPONSE


def test_empty_body_returns_empty_response(live_server):
    status, response = _post(live_server + "/", b"")
    assert status == 200
    assert response == protocol.EMPTY_RESPONSE


def test_oversized_body_is_rejected(live_server):
    status, response = _post(live_server + "/", b"x" * (server.MAX_BODY + 10))
    assert status == 200
    assert response == protocol.EMPTY_RESPONSE


def test_gbk_payload_is_decoded(live_server, payload_factory, base_roles):
    """判题器如果把中文按 GBK 编码发过来，也不能整回合丢掉"""
    payload = payload_factory(round_no=3, roles=base_roles)
    body = protocol.dumps(payload).encode("gbk")
    status, response = _post(live_server + "/", body)
    assert status == 200
    assert "roleCommandMap" in response


def test_response_content_type_is_json(live_server, payload_factory, base_roles):
    payload = payload_factory(round_no=3, roles=base_roles)
    request = urllib.request.Request(
        live_server + "/", data=protocol.dumps(payload).encode("utf-8"), method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert "application/json" in response.headers["Content-Type"]


def test_decide_timeout_returns_empty(monkeypatch, payload_factory, base_roles):
    """决策卡住时必须在超时保护内返回空指令，不让判题器等超时"""
    import time

    def slow(payload):
        time.sleep(server.DECIDE_TIMEOUT + 1)
        return dict(protocol.EMPTY_RESPONSE)

    monkeypatch.setattr(server, "decide", slow)
    handler = server.Handler.__new__(server.Handler)
    response = handler._decide_guarded(payload_factory(round_no=1, roles=base_roles))
    assert response == protocol.EMPTY_RESPONSE


def test_stats_counts_requests(live_server, payload_factory, base_roles):
    before = server.stats()["count"]
    _post(live_server + "/", protocol.dumps(
        payload_factory(round_no=7, roles=base_roles)
    ).encode("utf-8"))
    assert server.stats()["count"] >= before + 1
    assert server.stats()["last_round"] == 7
