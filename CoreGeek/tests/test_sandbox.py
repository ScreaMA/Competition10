"""沙盒输出解析与指纹的用例。

对应设计文档V2 §6.6。

这一层的存在理由是 V1 的一个具体教训：`lastCmdResult` 里"读文件死循环"、
"命令被掐断"、"执行器没找到文件"、"取回空结果集"四种完全不同的状态在
文本上长得一模一样，止损判据因此不得不写五套、互相打架。V2 让脚本显式
打出结构化标记，观测层只读标记、不猜。
"""

from __future__ import annotations

from agent.strategy.task import sandbox
from agent.strategy.task.sandbox import (
    advanced,
    fingerprint,
    parse_output,
    repeated,
)

from conftest import AUTH_FAIL_OUTPUT, SUCCESS_OUTPUT, V1_DEAD_LOOP_OUTPUT


def test_parse_exit_code_and_body():
    out = parse_output("[exitCode:0]\n[SCAN] files=3\n")
    assert out.status == "ok"
    assert out.get("SCAN.files") == "3"
    assert out.has("SCAN")


def test_parse_non_zero_exit_code():
    out = parse_output("[exitCode:2]\n[CHECK] ok=no code=2\n")
    assert out.status == "ok"
    assert out.get("CHECK.ok") == "no"


def test_parse_timeout():
    out = parse_output("[TIMEOUT]\n[SCAN] api_calls=1\n")
    assert out.status == "timeout"
    assert out.has_done is False


def test_parse_judger_error():
    out = parse_output("[JUDGER_ERROR]\n沙盒不可用\n")
    assert out.status == "judge_error"


def test_parse_truncated_flag():
    out = parse_output("[exitCode:0]\nhello\n[TRUNCATED]")
    assert out.truncated is True
    assert "hello" in out.raw


def test_parse_empty():
    out = parse_output("")
    assert out.status == "empty"
    assert out.answer is None
    assert out.has_done is False


def test_parse_answer_payload():
    out = parse_output(SUCCESS_OUTPUT)
    assert out.answer is not None
    assert out.answer.startswith('{"city"')
    assert out.api_ok is True
    assert out.api_calls == 1
    assert out.get("DATA.n") == "137"
    assert out.has_done


def test_parse_token():
    out = parse_output("[exitCode:0]\n[TOKEN] ab12cd34ef\n[DONE] step=verify\n")
    assert out.token == "ab12cd34ef"


def test_parse_mapping_pairs():
    out = parse_output(SUCCESS_OUTPUT)
    assert out.get("MAPPING.oldest_era") == "name"
    assert out.get("MAPPING.world_heritage_count") == "protected_level"


def test_parse_api_fail():
    out = parse_output(AUTH_FAIL_OUTPUT)
    assert out.answer is None
    assert out.get("APIFAIL.status") == "401"
    assert out.get("APIFAIL.reason") == "missing_auth"


def test_kv_sets_first_occurrence_only():
    out = parse_output("[exitCode:0]\n[API] status=401\n[API] status=200\n")
    assert out.get("API.status") == "401"  # 第一次出现的保留
    assert out.payloads["API"] == "status=401"


def test_evidence_keys_are_stable():
    out = parse_output(AUTH_FAIL_OUTPUT)
    keys = out.evidence_keys()
    assert any(key.startswith("SCAN.api_calls=") for key in keys)
    assert any(key.startswith("APIFAIL=") for key in keys)


# ==========================================================================
# 指纹
# ==========================================================================


def test_fingerprint_ignores_timestamps_and_elapsed():
    """时间戳与耗时每次执行都不同，但不代表内容变了"""
    a = fingerprint("2026-09-14 07:04:48 [DONE] step=query elapsed=1.20s")
    b = fingerprint("2026-09-15 11:22:31 [DONE] step=query elapsed=9.99s")
    assert a == b


def test_fingerprint_ignores_whitespace():
    assert fingerprint("[SCAN] files=9\n\ndirs=2") == fingerprint("[SCAN]   files=9 dirs=2")


def test_fingerprint_differs_on_content():
    assert fingerprint("[SCAN] docs=2") != fingerprint("[SCAN] docs=3")


def test_repeated_detects_dead_loop():
    """V1 最致命的形态：读题成功却反复重读同一份文档（T4）"""
    first = parse_output(V1_DEAD_LOOP_OUTPUT)
    second = parse_output(V1_DEAD_LOOP_OUTPUT)
    assert repeated(first, second) is True


def test_advanced_detects_new_evidence():
    recon = parse_output(
        "[exitCode:0]\n[RECON] root=/t1 task=/t1/task_1_beijing.md\n"
        "[SCAN] files=9 dirs=2 timed_out=no\n[DONE] step=recon\n"
    )
    query = parse_output(SUCCESS_OUTPUT)
    assert advanced(recon, query) is True
    assert repeated(recon, query) is False


def test_advanced_false_for_same_output():
    out = parse_output(V1_DEAD_LOOP_OUTPUT)
    assert advanced(out, out) is False


def test_advanced_true_without_previous():
    assert advanced(None, parse_output("[DONE] step=recon\n")) is True


def test_brief_is_single_line():
    brief = parse_output(SUCCESS_OUTPUT).brief()
    assert "\n" not in brief
    assert "answer=yes" in brief
    assert "done=True" in brief


def test_long_payload_is_clipped():
    body = "[exitCode:0]\n[ANSWER] " + "x" * 10000 + "\n"
    out = parse_output(body)
    assert len(out.answer) <= sandbox.MAX_ANSWER_LEN
