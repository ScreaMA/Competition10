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
import re

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
