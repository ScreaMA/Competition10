"""任务沙盒用例：**每次改完客户端都要跑的回归**。

对应设计文档V2 §6.2。

这里的题分两类，来源不同：

    api/<城市>     取自实测报文的那一族（同一套接口、换参数），另加 6 座城
    fix/<代号>     取自实测报文的那一族（照 spec 修工作区），另加 5 个变体

**判据不是"跑完不报错"，而是判分口径**：
    A 族  —— `[ANSWER]` 与 `apiserver.expected_answer()` 逐字段相同
    B 族  —— `./check` 真的通过，且客户端 `verify` 步抓到了 `TOKEN`

还有两条**守题**用例（`*_problems_are_real`）：题目本身必须是坏的——
B 族的预置工作区不修就过不了 `check`，A 族的答案必须与服务端记录对得上。
不然题目提前就是绿的，"通过了"什么也说明不了。
"""

from __future__ import annotations

import re

import pytest

from agent.strategy.task.memory import MEMORY
from tools.tasksandbox import env as sandbox_env
from tools.tasksandbox import fix_tasks, runner
from tools.tasksandbox.apiserver import CITIES, HeritageAPI, expected_answer

VARIANTS = (fix_tasks.KNOWN_VARIANT,) + fix_tasks.NEW_VARIANTS


@pytest.fixture(scope="module")
def heritage_api():
    """一个模块级的文化遗产服务（优先绑真沙盒的 8899 端口）"""
    with HeritageAPI() as api:
        yield api


@pytest.fixture
def sandbox(tmp_path, heritage_api):
    """一份干净的任务环境；每个用例一份，互不干扰"""
    def _make(cities=("北京",), variant=fix_tasks.KNOWN_VARIANT):
        return sandbox_env.Sandbox(tmp_path, heritage_api.base).build(
            cities=tuple(cities), variant=variant
        )
    return _make


def _query_params(sandbox, api):
    """query 步的显式参数

    服务端没绑在 8899 上时（端口被占）必须把 base 显式传下去——否则客户端会
    先往种子里的 `localhost:8899` 打一排请求、把 7 秒预算烧光，测出来的失败
    是环境造成的。绑在 8899 上时返回空，**让种子事实走正常路径**。
    """
    return {} if api.canonical else {"base": sandbox.base_url}


def _answer_for(sandbox, api, city):
    phase = sandbox.api_phase(city)
    runner.run_step("recon", sandbox, phase_task=phase)
    return runner.run_step(
        "query", sandbox, phase_task=phase, params=_query_params(sandbox, api)
    )


def _fix_chain(sandbox, variant):
    """家族 B 的完整链路，返回 (normalize, check, repair, verify) 四步结果

    工作区三个路径用**相对形式**并把 cwd 钉在 ws 上：真沙盒里它们是绝对路径
    （`/tmp/selfEvolutionTask/…`），代码路径一致；本地只能用相对形式——
    Git Bash 的 `sh` 执行不了 Windows 绝对路径（反斜杠被当转义符吃掉）。
    """
    facts: dict[str, str] = {}
    phase = fix_tasks.phase_of(variant)
    recon = runner.run_step("recon", sandbox, phase_task=phase, cwd=sandbox.ws_root)
    for source, key in (("RECON.root", "sandbox.root"), ("RECON.ws", "sandbox.ws")):
        value = recon.kv(source)
        if value:
            facts[key] = value

    params = {"ws": ".", "spec": "spec.md", "check": "./check"}
    norm = runner.run_step("normalize", sandbox, phase_task=phase,
                           facts=facts, params=params, cwd=sandbox.ws)
    check = runner.run_step("check", sandbox, phase_task=phase,
                            facts=facts, params=params, cwd=sandbox.ws)
    repair = runner.run_step("repair", sandbox, phase_task=phase,
                             facts=facts, params=params,
                             check_output=check.markers.get("CHECKBODY", ""),
                             cwd=sandbox.ws)
    verify = runner.run_step("verify", sandbox, phase_task=phase,
                             facts=facts, params=params, cwd=sandbox.ws)
    return norm, check, repair, verify


# ==========================================================================
# A 族：API 查询
# ==========================================================================


@pytest.mark.parametrize("city", CITIES)
def test_api_task_completes(sandbox, heritage_api, city):
    """同一套接口、换参数 —— 答案要逐字段对上（这就是 80 分那道题）"""
    box = sandbox(cities=(city,))
    result = _answer_for(box, heritage_api, city)

    assert not result.errors(), result.errors()
    assert "[API]" in result.output, result.output
    assert "status=200" in result.output, result.output

    answer = result.answer()
    assert answer is not None, result.output
    assert answer == expected_answer(city), result.output


@pytest.mark.parametrize("city", CITIES)
def test_api_problems_are_real(city):
    """守题：服务端记录与"标准答案"必须自洽（题目本身不能是坏的）"""
    answer = expected_answer(city)
    assert answer["total_count"] > 0
    assert answer["oldest_era"]
    assert answer["types"] == sorted(answer["types"])
    assert 0 <= answer["world_heritage_count"] <= answer["total_count"]


def test_client_carries_no_baked_in_api_answers(sandbox, heritage_api):
    """**客户端不许预置这道题的答案。** 事实区开局必须是空的

    这一条是本套用例的"宪法"。实测报文里这道题的三个坑（鉴权头、
    参数名、记录字段名）都有已知正确答案，把它们内置进去确实能过题——
    但那是"我们事先知道答案"，换一套接口就废了。要的是**过程**：
    试 → 读服务端的报错 → 补候选 → 再试。
    """
    from agent.strategy.task.memory import MEMORY

    assert MEMORY.facts == {}, MEMORY.facts

    box = sandbox(cities=("北京",))
    result = _answer_for(box, heritage_api, "北京")
    assert result.answer() == expected_answer("北京"), result.output


def test_client_learns_the_header_and_param_from_error_bodies(sandbox, heritage_api):
    """两个坑都得由**服务端自己的报错**教会客户端

        {"code":401,… "Missing or invalid 'Authorization' header" …}
        {"code":400,… "请求参数错误：缺少 'location'"}

    第一轮按文档打（`X-API-Key` + `?city=`）会吃到 401/400；把响应体里的
    名字抽出来补进候选，下一轮就带上了。这条用例把"文档过时"那一族的
    通用解法钉死：**换掉服务端、换掉字段名，同一段逻辑照样收敛**。
    """
    box = sandbox(cities=("北京",))
    result = _answer_for(box, heritage_api, "北京")

    assert not result.errors(), result.errors()
    # 学到的东西要留痕，复盘才看得出"它自己发现了什么"
    assert result.has("LEARN"), result.output
    learned = result.markers["LEARN"]
    assert "param=" in learned and "header=" in learned, learned
    # 学到之后要真的用上：最终那次请求必须带对头和参数
    assert "status=200" in result.output, result.output
    assert result.answer() == expected_answer("北京"), result.output


def test_discovery_survives_a_renamed_param_and_header(tmp_path):
    """**换个服务端说法，同一段逻辑还得认出来**（这才是"过程"的成色）

    把服务端换成用 `X-Auth-Token` 认鉴权、用 `province` 收参数——两个名字
    文档里都没有。客户端只能靠报错里的名字收敛。
    """
    from tools.tasksandbox import apiserver as api_mod

    class Renamed(api_mod.HeritageHandler):
        """头名和参数名都换掉的服务端"""

        def do_GET(self):  # noqa: N802
            parsed = api_mod.urlsplit(self.path)
            if self.headers.get("X-Auth-Token") != api_mod.BEARER_VALUE:
                self._reply(401, {"code": 401,
                                  "message": "Missing or invalid 'X-Auth-Token' header"})
                return
            query = api_mod.parse_qs(parsed.query)
            if "province" not in query:
                self._reply(400, {"code": 400,
                                  "message": "请求参数错误：缺少 'province'"})
                return
            records = api_mod.RECORDS.get(query["province"][0], [])
            self._reply(200, {"code": 200, "data": {"records": records,
                                                    "pagination": {"total_count": len(records)}}})

    from http.server import ThreadingHTTPServer
    import threading

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Renamed)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = "http://127.0.0.1:%d" % httpd.server_address[1]
        box = sandbox_env.Sandbox(tmp_path, base).build(cities=("北京",))
        result = runner.run_step(
            "query", box, phase_task=box.api_phase("北京"),
            params={"base": base},
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)

    assert not result.errors(), result.errors()
    assert "status=403" not in result.output
    assert result.answer() == expected_answer("北京"), result.output


def test_stale_document_mode_would_have_worked(tmp_path, heritage_api):
    """服务端按**文档口径**校验时反而能连上 —— 反证"真接口和文档不一样"

    这一条把题眼的因果讲清楚：如果服务端真的像 `API_DOCS.md` 写的那样要
    `X-API-Key` + `?city=`，客户端继续试文档候选就能打中。实测的服务端不是。
    """
    with HeritageAPI(mode="stale") as api:
        box = sandbox_env.Sandbox(tmp_path, api.base).build(cities=("北京",))
        result = _answer_for(box, api, "北京")
        assert not result.errors(), result.errors()
        assert "[API]" in result.output, result.output
        assert result.answer() == expected_answer("北京"), result.output


# ==========================================================================
# B 族：工程修复
# ==========================================================================


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.code for v in VARIANTS])
def test_fix_task_completes(sandbox, variant):
    """照 spec 修工作区 → 跑 `./check` → 抓 TOKEN（这就是那 80 分）"""
    box = sandbox(variant=variant)
    norm, check, repair, verify = _fix_chain(box, variant)

    for result in (norm, check, repair, verify):
        assert not result.errors(), (result.step, result.errors(), result.output)

    # normalize 必须真的把 CRLF 修掉（`read()` 的换行归一 bug 就在这里现形）
    assert "crlf=0" not in norm.markers.get("FIX", ""), norm.markers
    # repair 必须真的改了东西（曾长期是 `actions=0 applied=0`）
    assert "actions=0" not in repair.markers.get("FIX", ""), repair.markers

    assert box.check_passes(), box.run_check()[1]
    assert verify.token() == variant.token, verify.output


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.code for v in VARIANTS])
def test_fix_problems_are_real(tmp_path, variant):
    """守题：**不修就过不了**，手工修好一定过得了

    两半都要：前半句保证题目是坏的（否则"通过了"没有意义），
    后半句保证题目是**可解的**（否则"做不完"是我们的实现问题、不是题的）。
    """
    box = sandbox_env.Sandbox(tmp_path, "http://127.0.0.1:1").build(
        cities=("北京",), variant=variant
    )
    code, out = box.run_check()
    assert not (code == 0 and "TOKEN" in out), "预置工作区不该已经通过"

    box.plant_fix(variant)
    assert box.check_passes(), box.run_check()[1]


def test_normalize_repairs_crlf_and_exec_bit(sandbox):
    """CRLF 与执行位：实测报文里 `@ <path>: /bin/sh^M: bad interpreter` 的根因"""
    variant = fix_tasks.KNOWN_VARIANT
    box = sandbox(variant=variant)
    assert b"\r" in box.check.read_bytes()          # 预置就是 CRLF

    norm, *_ = _fix_chain(box, variant)
    assert "crlf=" in norm.markers.get("FIX", "")
    assert b"\r" not in box.check.read_bytes()


# ==========================================================================
# 沙盒自身的自洽性
# ==========================================================================


def test_phase_text_is_the_zero_keyword_shape():
    """题面必须是实测那种"27 字节、零关键词"的形态

    否则族识别就不是在生产路径上被考验的——`classify` 必须靠侦察输出定族，
    这正是之前 0 分的根因（见 commit 844a4cf）。
    """
    for city in CITIES:
        index = CITIES.index(city) + 1
        from tools.tasksandbox.apiserver import CITY_SLUG
        text = "请阅读task_%d_%s.md，获取任务信息" % (index, CITY_SLUG[city])
        assert len(text) < 40, text
        for token in ("api", "接口", "spec", "修复", "配置", "check"):
            assert token not in text.lower(), (text, token)


def test_era_tables_stay_in_sync():
    """沙盒的年代表必须与注入脚本里的 `BUILTIN_ERA` 一致

    那份表写在 `scripts.py` 的 raw string 模板里导不出来，所以只能从源码里抠。
    漂了就会判分歧义：`oldest_era` 算错时到底怪客户端还是怪题？
    """
    from agent.strategy.task import scripts
    from tools.tasksandbox import eras

    source = open(scripts.__file__, encoding="utf-8").read()
    block = re.search(r"BUILTIN_ERA = \{(.*?)\n\}", source, re.S)
    assert block, "没在 scripts.py 里找到 BUILTIN_ERA"
    pairs = re.findall(r'"([^"]+)":\s*(\d+)', block.group(1))
    server_side = {name: int(rank) for name, rank in pairs}
    assert server_side == eras.ERA_ORDER, (
        set(server_side.items()) ^ set(eras.ERA_ORDER.items())
    )
