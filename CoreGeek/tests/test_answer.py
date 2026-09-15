"""答案闸门用例。

对应设计文档V2 §6.7。

`answer.gate` 是**唯一**允许产出 `submitAnswer` 载荷的地方。这一组用例覆盖
闸门的 6 条规则，重点是两条历史上踩过的坑：

- 把任务文档原文当答案交上去（PK590557 的 R14/R16/R18）
- `{"error": null}` 被误判成错误体（main 分支上修过一次的假阳性）
"""

from __future__ import annotations

import json

import pytest

from agent.strategy.task import answer as answer_mod
from agent.strategy.task.answer import (
    gate,
    is_doc_echo,
    is_error_body,
    normalize_candidate,
    parse_spec,
    type_violations,
)

from conftest import BEIJING_TASK_TEXT

GOOD = json.dumps({
    "city": "北京",
    "total_count": 137,
    "world_heritage_count": 12,
    "types": ["古遗址", "古建筑"],
    "oldest_era": "周口店遗址",
}, ensure_ascii=False)


# ==========================================================================
# 规格解析
# ==========================================================================


def test_parse_spec_extracts_keys_and_annotations():
    spec = parse_spec(BEIJING_TASK_TEXT)
    assert set(spec.keys) == {
        "city", "total_count", "world_heritage_count", "types", "oldest_era"
    }
    assert "总记录条数" in spec.annotations["total_count"]
    assert "遗产名称" in spec.annotations["oldest_era"]


def test_parse_spec_extracts_example_types():
    """示例答案决定字段类型（任务书原文：不能将数字 0 写成 "0"）"""
    spec = parse_spec(BEIJING_TASK_TEXT)
    assert spec.example["total_count"] == 0
    assert isinstance(spec.example["total_count"], int)
    assert isinstance(spec.example["types"], list)


def test_parse_spec_on_empty_text():
    spec = parse_spec("")
    assert spec.known is False


# ==========================================================================
# 规则 2：文档回声
# ==========================================================================


def test_doc_echo_detected():
    assert is_doc_echo(BEIJING_TASK_TEXT, BEIJING_TASK_TEXT) is True


def test_doc_echo_partial_still_detected():
    chunk = BEIJING_TASK_TEXT[200:900]
    assert is_doc_echo(chunk, BEIJING_TASK_TEXT) is True


def test_real_answer_is_not_doc_echo():
    assert is_doc_echo(GOOD, BEIJING_TASK_TEXT) is False


def test_gate_rejects_doc_echo():
    answer, reason = gate(BEIJING_TASK_TEXT, BEIJING_TASK_TEXT)
    assert answer is None
    assert reason == "doc_echo"


# ==========================================================================
# 规则 3：错误体
# ==========================================================================


def test_error_body_json_with_message():
    assert is_error_body('{"error": "Unauthorized"}') is True


def test_error_body_status_code():
    assert is_error_body('{"code": 401, "message": "Missing header"}') is True


def test_error_body_traceback():
    assert is_error_body("Traceback (most recent call last):\n  File ...") is True


def test_error_body_shell_failure():
    assert is_error_body("sh: 1: ./check: Permission denied") is True


def test_error_null_is_not_error_body():
    """回归：`{"error": null}` 是合法载荷（V1 在这里有过一次假阳性）"""
    payload = json.dumps({
        "city": "北京", "total_count": 1, "world_heritage_count": 0,
        "types": ["a"], "oldest_era": "b", "error": None,
    }, ensure_ascii=False)
    assert is_error_body(payload) is False


def test_empty_json_is_error_body():
    assert is_error_body("{}") is True


def test_error_empty_list_value_is_not_error_body():
    payload = json.dumps({"city": "北京", "total_count": 1, "error": []},
                         ensure_ascii=False)
    assert is_error_body(payload) is False


def test_good_answer_is_not_error_body():
    assert is_error_body(GOOD) is False


# ==========================================================================
# 规则 4/5：字段名与类型
# ==========================================================================


def test_normalize_drops_unknown_and_empty_keys():
    spec = parse_spec(BEIJING_TASK_TEXT)
    candidate = json.dumps({
        "city": "北京", "total_count": 137,
        "unrelated": "x", "empty": "", "nothing": None,
    }, ensure_ascii=False)
    normalized = normalize_candidate(candidate, spec)
    payload = json.loads(normalized)
    assert set(payload) == {"city", "total_count"}


def test_normalize_returns_none_when_all_empty():
    spec = parse_spec(BEIJING_TASK_TEXT)
    assert normalize_candidate('{"city": ""}', spec) is None


def test_type_violations_flags_string_for_number():
    spec = parse_spec(BEIJING_TASK_TEXT)
    payload = {"total_count": "137", "world_heritage_count": 12}
    assert type_violations(payload, spec) == ["total_count"]


def test_gate_rejects_type_mismatch():
    bad = json.dumps({
        "city": "北京", "total_count": "137", "world_heritage_count": "12",
        "types": ["a"], "oldest_era": "b",
    }, ensure_ascii=False)
    answer, reason = gate(bad, BEIJING_TASK_TEXT)
    assert answer is None
    assert reason.startswith("type_mismatch")


def test_gate_accepts_partial_answer():
    """部分完成也有分（任务书 §6：通过率 = 正确字段数 / 全量字段数）"""
    partial = json.dumps({"city": "北京", "total_count": 137}, ensure_ascii=False)
    answer, reason = gate(partial, BEIJING_TASK_TEXT)
    assert answer is not None
    assert reason == "ok"


def test_gate_rejects_answer_without_any_known_field():
    other = json.dumps({"foo": 1, "bar": 2})
    answer, reason = gate(other, BEIJING_TASK_TEXT)
    assert answer is None
    assert reason == "no_known_field"


# ==========================================================================
# 规则 1/6：长度与去重
# ==========================================================================


def test_gate_rejects_short_candidate():
    answer, reason = gate("ab", BEIJING_TASK_TEXT)
    assert answer is None
    assert reason == "too_short"


def test_gate_rejects_empty_candidate():
    assert gate(None, BEIJING_TASK_TEXT) == (None, "no_candidate")


def test_gate_deduplicates_submitted():
    answer, reason = gate(GOOD, BEIJING_TASK_TEXT, submitted=[GOOD])
    assert answer is None
    assert reason == "already_submitted"


def test_gate_refuses_rejected_answer():
    """被判错的答案不许重交（V1 的 PK592108 R15-R17 就是复读到额度烧完）"""
    answer, reason = gate(GOOD, BEIJING_TASK_TEXT, rejected=[GOOD])
    assert answer is None
    assert reason == "already_rejected"


# ==========================================================================
# token（工程修复族）
# ==========================================================================


def test_gate_accepts_plain_token():
    answer, reason = gate("ab12cd34ef56", "请修复工程并运行 ./check 获取 TOKEN")
    assert answer == "ab12cd34ef56"
    assert reason == "token"


def test_gate_rejects_long_non_json():
    text = "这是一段很长的中文说明" * 20
    answer, reason = gate(text, BEIJING_TASK_TEXT)
    assert answer is None
