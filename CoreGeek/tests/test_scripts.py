"""沙盒命令生成用例。

对应设计文档V2 §6.3.5。

这一组用例的重点是**静态语法校验**：生成的脚本会被塞进 `sh -c` 的 heredoc
里丢到沙盒执行，沙盒里一旦出现语法错误，表现只是"这一回合没有任何输出"——
在对战日志上与"卡死"长得一模一样，事后根本查不出来。所以每一条命令的
Python 部分都必须能 `compile()` 通过。

另外还要检查脚本遵守的三条纪律：
    1. 有 `[DONE]` 收尾
    2. 参数通过 JSON blob 注入（`__PARAMS__` 只剩一个，且是合法 JSON）
    3. 命令行里不出现未替换的占位符
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from agent.strategy.task import scripts
from agent.strategy.task.memory import StepSpec

HEREDOC = re.compile(r"<<'PYEOF' 2>&1\n(.*)\nPYEOF\Z", re.S)
PARAMS = re.compile(r"P = json\.loads\((.*)\)\n")


def _body(command: str) -> str:
    match = HEREDOC.search(command)
    assert match, "命令必须用 <<'PYEOF' … PYEOF 包裹"
    return match.group(1)


def _params(command: str) -> dict:
    match = PARAMS.search(_body(command))
    assert match, "脚本必须先解析参数 blob"
    return json.loads(eval(match.group(1)))  # noqa: S307 —— 生成值是 repr(str)


ALL_STEPS = ("recon", "query", "normalize", "check", "repair", "verify", "generic")


@pytest.mark.parametrize("name", ALL_STEPS)
def test_generated_script_compiles(name):
    """每一条生成的脚本都必须是合法的 Python"""
    command = scripts.build(
        StepSpec(name),
        phase_task="请阅读task_1_beijing.md，获取任务信息",
        facts={},
    )
    compile(_body(command), f"<{name}>", "exec")


@pytest.mark.parametrize("name", ALL_STEPS)
def test_generated_script_has_done_marker(name):
    """脚本必须以 `[DONE]` 收尾——没有它就分不清"卡死"与"没跑完" """
    command = scripts.build(StepSpec(name), phase_task="", facts={})
    body = _body(command)
    assert 'finish("' in body


@pytest.mark.parametrize("name", ALL_STEPS)
def test_no_unreplaced_placeholder(name):
    command = scripts.build(StepSpec(name), phase_task="", facts={})
    assert "__PARAMS__" not in command
    assert "__" not in command.split("P = json.loads")[0][-200:]


def test_wrapper_probes_both_python_names():
    """沙盒里解释器可能叫 python3 也可能叫 python"""
    command = scripts.build(StepSpec("recon"), phase_task="", facts={})
    assert "for P in python3 python" in command
    assert "$P -u -" in command


def test_params_carry_task_hint():
    command = scripts.build(
        StepSpec("recon"),
        phase_task="请阅读task_1_beijing.md，获取任务信息",
        facts={},
    )
    assert _params(command)["task_hint"] == "task_1_beijing.md"


def test_query_params_carry_known_facts():
    """命中技能后，已知的接口事实要注入到命令里（不必再读文档猜）"""
    command = scripts.build(
        StepSpec("query"),
        phase_task="",
        facts={
            "api.base_url": "http://localhost:8899",
            "api.auth_value": "Authorization: Bearer k",
            "api.param": "location",
        },
    )
    params = _params(command)
    assert params["base"] == "http://localhost:8899"
    assert params["auth"] == "Authorization: Bearer k"
    assert params["param"] == "location"


def test_engineering_params_carry_workspace():
    command = scripts.build(
        StepSpec("repair"),
        phase_task="",
        facts={"sandbox.ws": "/t/ws_1", "sandbox.check": "/t/ws_1/check"},
        check_output="bad interpreter: /bin/sh^M",
    )
    params = _params(command)
    assert params["ws"] == "/t/ws_1"
    assert params["check"] == "/t/ws_1/check"
    assert params["check_output"].startswith("bad interpreter")


def test_unknown_step_still_produces_valid_script():
    """未知步骤也要发一条能跑完的最小命令（保证每回合都有产出）"""
    command = scripts.build(StepSpec("nope"), phase_task="", facts={})
    body = _body(command)
    compile(body, "<unknown>", "exec")
    assert 'finish("unknown")' in body


def test_query_script_tries_multiple_auth_variants():
    """query 的候选矩阵里同时包含 Authorization / X-API-Key / 无鉴权

    故障 T3：R14 裸调接口吃 401（`Missing 'Authorization' header`）。
    候选矩阵保证了"同一条命令里把几种鉴权都试一遍"，不会一个 401 就整回合白费。
    """
    body = _body(scripts.build(StepSpec("query"), phase_task="", facts={}))
    assert "Authorization: " in body
    assert "X-API-Key: " in body
    assert 'auths.append("")' in body


def test_query_script_aggregates_from_task_template():
    """聚合规则来自任务原文的模板说明，而不是写死的字段名"""
    body = _body(scripts.build(StepSpec("query"), phase_task="", facts={}))
    assert "parse_answer_spec" in body
    assert "总" in body and "最早" in body and "不重复" in body


def test_query_script_has_builtin_era_order():
    """年代排序用内置表（对战日志明确记着"不要按字符串字典序"）"""
    body = _body(scripts.build(StepSpec("query"), phase_task="", facts={}))
    assert "旧石器时代" in body
    assert "近现代" in body


def test_risky_command_filter():
    assert scripts.sanitize_llm_command("ls -la") == "ls -la"
    assert scripts.sanitize_llm_command("mkfs.ext4 /dev/sda1") is None
    assert scripts.sanitize_llm_command("wget http://x/y") is None
    assert scripts.sanitize_llm_command("") is None


def test_task_hint_from_phase_task():
    assert scripts.task_hint("请阅读task_1_beijing.md，获取任务信息") == "task_1_beijing.md"
    assert scripts.task_hint("请阅读 spec.md") == "spec.md"
    assert scripts.task_hint("没有文件名") == ""


def test_search_roots_cover_observed_paths():
    """实测任务目录在 /tmp/selfEvolutionTask 下（对战日志里的沙盒路径）"""
    assert "/tmp/selfEvolutionTask" in scripts.SEARCH_ROOTS
    assert scripts.DEFAULT_BASE == "http://localhost:8899"


# ==========================================================================
# 真的执行一遍生成的脚本
# ==========================================================================
#
# 上面的 `compile()` 类用例只查语法。`emit()` 签名这种"调用时才暴露"的错误
# 在静态层面永远是绿的，而沙盒里一跑就 `TypeError`——对战日志上只表现为
# "这一步没有任何输出"，跟"卡死"长得一模一样。所以下面几条用例把生成的
# Python 原样执行一次，断言它真的产出了该产出的标记。


def _run_body(command: str, cwd) -> str:
    """把命令里的 Python 部分跑起来，返回 stdout+stderr

    `PYTHONIOENCODING=utf-8`：脚本的标记正文里有中文，不钉死编码的话
    Windows 上会按本地代码页（GBK）写 stdout，这里解出来就是乱码——
    那是**测试脚手架**的问题，不是脚本的问题（判题器跑在 Linux 上）。
    """
    script = cwd / "_generated_script.py"
    # newline=""：Windows 上默认会把 \n 翻成 \r\n，那样脚本自己就成了一个
    # "带 CRLF 的文件"，会被 normalize 步顺手改掉，把 crlf 计数搅浑
    script.write_text(_body(command), encoding="utf-8", newline="")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


@pytest.mark.parametrize("name", ("check", "verify"))
def test_engineering_steps_actually_emit_check_marker(name, tmp_path):
    """check / verify 必须真的打出 `[CHECK]` 标记

    故障：脚本里写的是 `emit("CHECK", ok=…, code=…)`，而 `emit` 只收位置参数
    ⇒ `TypeError: emit() got an unexpected keyword argument 'ok'`，脚本第一步
    就崩。`[CHECK]` 于是从未出现过，`_step_satisfied("check")` 永远为假，
    阶梯在 check→repair→verify 之间空转到任务超时——实测报文里第二个任务
    整整 13 个回合全耗在这个循环上。
    """
    ws = tmp_path / "ws_1"
    ws.mkdir()
    spec = ws / "spec.md"
    spec.write_text("## 目录要求\n- logs/alpha/ 必须存在\n", encoding="utf-8")
    check = ws / "check"
    check.write_text("exit 1\n", encoding="utf-8")

    step = StepSpec(name).with_params(ws=str(ws), spec=str(spec), check=str(check))
    out = _run_body(scripts.build(step, phase_task="", facts={}), ws)
    assert "TypeError" not in out, out
    assert "Traceback" not in out, out
    assert "[CHECK]" in out, out
    assert "[DONE] step=%s" % name in out, out


def test_check_step_without_script_still_emits_marker(tmp_path):
    """连 check 脚本都找不到时也要打 `[CHECK]`（同一个 emit 关键字参数错误）"""
    ws = tmp_path / "ws_1"
    ws.mkdir()
    step = StepSpec("check").with_params(
        ws=str(ws), spec=str(ws / "spec.md"), check=str(ws / "missing-check")
    )
    out = _run_body(scripts.build(step, phase_task="", facts={}), ws)
    assert "TypeError" not in out, out
    assert "[CHECK]" in out and "reason=no_check_script" in out, out


# --- query：对着一个真的 HTTP 服务跑一遍 ----------------------------------

STUB_KEY = "sk-heritage-2026"

# 沙盒里那个接口的全部记录（用来算答案）
STUB_RECORDS = [
    {"name": "周口店遗址", "era": "旧石器时代", "type": "古遗址",
     "protected_level": "世界遗产"},
    {"name": "故宫", "era": "明", "type": "古建筑",
     "protected_level": "世界遗产"},
    {"name": "天坛", "era": "明", "type": "古建筑",
     "protected_level": "全国重点文物保护单位"},
]


class StubApi:
    """模拟沙盒里那个接口；`accepts` 决定服务端实际认哪个头

    `accepts` 与文档里写的是**两回事**——这正是实测报文里的坑：文档写
    `X-API-Key: <key>`，服务端只认 `Authorization: Bearer <key>`，于是
    "抄对了密钥"照样 401（任务原文自己就提示"文档中的部分字段内容已经发生
    变化，描述不再准确"）。
    """

    def __init__(self, base: str):
        self.base = base
        self.accepts = "X-API-Key"
        self.hits: list[str] = []

    def authorized(self, headers) -> bool:
        if self.accepts == "X-API-Key":
            return headers.get("X-API-Key") == STUB_KEY
        if self.accepts == "Bearer":
            return headers.get("Authorization") == "Bearer " + STUB_KEY
        return False

    def missing_message(self) -> str:
        return ("Missing 'X-API-Key' header"
                if self.accepts == "X-API-Key"
                else "Missing 'Authorization' header")


@pytest.fixture
def stub_api():
    """模拟沙盒里那个"端点存在、但不带鉴权就 401"的接口

    复刻实测报文里的三个特征：
      1. 端点要鉴权头，不带就 401（默认认 `X-API-Key`，可用 `.accepts` 改成
         `Bearer` 来复刻"文档过时"）；
      2. 任务文档里给的"接口地址"是**带查询串的示例**；
      3. 查询参数是中文，需要转义。
    """
    api = StubApi("http://127.0.0.1:0")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 —— BaseHTTPRequestHandler 的接口
            parsed = urlsplit(self.path)
            if not api.authorized(self.headers):
                self._reply(401, {"error": api.missing_message()})
                return
            if parsed.path != "/api/v1/heritage/search":
                self._reply(404, {"error": "no such endpoint"})
                return
            if parse_qs(parsed.query).get("city") != ["北京"]:
                self._reply(200, {"total": 0, "records": []})
                return
            api.hits.append(self.path)
            self._reply(200, {"total": len(STUB_RECORDS), "records": STUB_RECORDS})

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # 别把测试输出刷满
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    api.base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        yield api
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def _query_task_doc(base: str) -> str:
    """任务文档：刻意带上"带查询串的示例地址"与真实鉴权头（复刻实测报文）"""
    return "\n".join([
        "# 自进化任务 A-1：查询北京文化遗产",
        "",
        "系统提供了一个 API 服务，API 文档在 `API_DOCS.md` 中。",
        "注意：由于该系统经过了长期迭代，文档中的部分字段内容已经发生变化。",
        "",
        "- 鉴权头：`X-API-Key: %s`" % STUB_KEY,
        "- 调用示例：`%s/api/v1/heritage/search?city=北京&limit=100`" % base,
        "",
        "## 任务要求",
        "",
        "从 API 查询北京市的**全部**文化遗产记录，然后提交以下统计信息：",
        "",
        "```json",
        "{",
        '  "city": "北京",',
        '  "total_count": <总记录条数>,',
        '  "world_heritage_count": <保护级别为"世界遗产"的数量>,',
        '  "types": ["<所有不重复的遗产类型，顺序不限>"],',
        '  "oldest_era": "<年代最早的遗产名称>"',
        "}",
        "```",
        "",
        "## 提交形式",
        "",
        '```json',
        '{"city":"北京","total_count":0,"world_heritage_count":0,'
        '"types":["a","b"],"oldest_era":"c"}',
        "```",
        "",
        "- 提交答案为数字/字符串敏感型，不能将数字0写成\"0\"，否则算错",
        "",
    ])


def test_query_script_end_to_end_against_stub_api(stub_api, tmp_path):
    """query 脚本必须真的取到数、算出正确的答案

    这一条覆盖实测报文的三个连环坑（R13 的 `[APIFAIL]` 全在这一条里）：

      1. **示例地址带查询串** → `split_base` 不砍掉它，拼出来就是
         `…/search?city=北京&limit=100/api/v1/heritage/search` 这种畸形 URL；
      2. **中文没转义** → `UnicodeEncodeError`（日志上伪装成"服务端出错"）；
      3. **鉴权候选全是自指垃圾** → 旧实现把文档里的**头名**当成值，
         生成 `X-API-Key: X-API-Key`，真正的密钥永远排不进 `auths[:4]`，
         4 次请求全 401，最后交上去一份 `total_count: 0` 的废卷（得 16/80 分）。

    修好之前这条用例必然失败：记录取不到，答案里每个聚合字段都是空/零。
    """
    (tmp_path / "task_1_alpha.md").write_text(
        _query_task_doc(stub_api.base), encoding="utf-8"
    )
    step = StepSpec("query").with_params(ws=str(tmp_path))
    out = _run_body(
        scripts.build(step, phase_task="请阅读task_1_alpha.md，获取任务信息", facts={}),
        tmp_path,
    )

    assert "Traceback" not in out, out
    assert "UnicodeEncodeError" not in out, out
    assert "[API]" in out and "status=200" in out, out
    assert "[APIFAIL]" not in out.split("[SCAN]")[0], out

    match = re.search(r"^\[ANSWER\] (\{.*\})$", out, re.M)
    assert match, out
    answer = json.loads(match.group(1))
    assert answer["city"] == "北京"
    assert answer["total_count"] == 3
    assert answer["world_heritage_count"] == 2
    assert answer["types"] == ["古建筑", "古遗址"]
    assert answer["oldest_era"] == "周口店遗址"


def test_query_survives_stale_header_name(stub_api, tmp_path):
    """文档写的头名过时了，也要能连上（每个值再派生 Bearer 形态）

    实测报文里就是这个形态：从 `API_DOCS.md` 抄到的 `X-API-Key: <key>` 四次
    全 401，而同一次请求在历史记录里留下的真实报错是
    `Missing 'Authorization' header` —— **值是对的，头名过时了**。
    任务原文自己就提示"文档中的部分字段内容已经发生变化，描述不再准确"。

    修好之前这条必然失败：候选里只有 `X-API-Key`，服务端只认 Bearer，
    取数 0 条，交上去一份 `total_count: 0` 的废卷。
    """
    stub_api.accepts = "Bearer"          # 服务端只认 Authorization
    (tmp_path / "task_1_alpha.md").write_text(
        _query_task_doc(stub_api.base), encoding="utf-8"   # 文档只写了 X-API-Key
    )
    step = StepSpec("query").with_params(ws=str(tmp_path))
    out = _run_body(
        scripts.build(step, phase_task="请阅读task_1_alpha.md，获取任务信息", facts={}),
        tmp_path,
    )

    assert "[API]" in out and "status=200" in out, out
    answer = json.loads(re.search(r"^\[ANSWER\] (\{.*\})$", out, re.M).group(1))
    assert answer["total_count"] == 3
    assert answer["world_heritage_count"] == 2


def test_apifail_carries_response_body(stub_api, tmp_path):
    """401 必须把**响应体**打出来——只报"鉴权没过"说明不了服务端要什么头"""
    stub_api.accepts = "never"           # 怎么试都不认
    (tmp_path / "task_1_alpha.md").write_text(
        _query_task_doc(stub_api.base), encoding="utf-8"
    )
    step = StepSpec("query").with_params(ws=str(tmp_path))
    out = _run_body(
        scripts.build(step, phase_task="请阅读task_1_alpha.md，获取任务信息", facts={}),
        tmp_path,
    )

    assert "reason=missing_auth" in out, out
    assert "body=" in out, out
    assert "Missing" in out and "header" in out, out


def test_normalize_actually_fixes_crlf(tmp_path):
    """normalize 步必须真的把 CRLF 修掉

    故障：`read()` 是文本模式，universal newlines 在**读的时候**就把 `\\r\\n`
    变成了 `\\n`，于是 `if "\\r" not in body` 恒真 —— `[FIX] crlf=0` 修了个寂寞。
    而工程修复族的 `check` 脚本 shebang 上带着 `\\r`，内核直接拒执行
    （`/bin/sh^M: bad interpreter`，code=126），verify 永远拿不到 TOKEN，
    整条链路必 0 分（实测报文两轮都是 `[FIX] crlf=0` + `code=126`）。
    """
    ws = tmp_path / "ws_1"
    (ws / "bin").mkdir(parents=True)
    spec = ws / "spec.md"
    spec.write_text("## 目录要求\n- logs/alpha/ 必须存在\n", encoding="utf-8")
    check = ws / "check"
    check.write_bytes(b"#!/bin/sh\r\nexit 0\r\n")
    start = ws / "bin" / "start.sh"
    start.write_bytes(b"#!/bin/sh\r\necho hi\r\n")

    step = StepSpec("normalize").with_params(ws=str(ws), spec=str(spec), check=str(check))
    out = _run_body(scripts.build(step, phase_task="", facts={}), ws)

    assert "crlf=2" in out, out
    assert b"\r" not in check.read_bytes()
    assert b"\r" not in start.read_bytes()

