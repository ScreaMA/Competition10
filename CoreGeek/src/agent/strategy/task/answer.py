"""答案提取与校验闸门。

对应设计文档V2 §6.7。

**这是整个客户端里唯一允许产出 `submitAnswer` 载荷的地方。** 所有提交路径
都必须经过 `gate()`。

为什么需要闸门：对战复盘里我方交上去的"答案"有两次是任务文档原文，
有三次是接口的 401/404 错误 JSON（PK590557 的 R14/R16/R18）。判题器把它们
当成错误答案，`errorCode=2`，任务分归零。V1 的提取逻辑是"沙盒输出了什么就
交什么"，缺的正是这里的一道闸门。

同时闸门也**不能太严**：任务书 §6 规定部分完成按 `通过率 = 正确字段数 /
全量字段数` 给分，"交一份对了一半的答案"严格优于"不交"。所以闸门拒绝的是
**明显不是答案的东西**（文档回声、错误体、空壳），而不是"不够完美的答案"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# 答案原文的下限/上限
MIN_ANSWER_LEN = 4
MAX_ANSWER_LEN = 4000

# 文档回声判定：候选答案的 12 字滑窗有多大比例出现在任务原文里
SHINGLE = 12
ECHO_RATIO = 0.6

# 低于这个长度的候选**不做回声判定**。
# 任务原文里带着答案模板与示例（`{"city":"北京","total_count":0,…}`），一份
# 短小但正确的答案（比如只填了 city 与 total_count）与示例前缀天然重合，
# 按纯比例判定会把合法答案误杀。真实的"文档回声"动辄上千字，用长度先分流。
ECHO_MIN_LEN = 200

# 错误体的结构化判据
_ERROR_STATUS = re.compile(r"\b(400|401|403|404|405|422|429|500|502|503|504)\b")
_ERROR_WORDS = (
    "Traceback (most recent call last)",
    "No such file or directory",
    "Permission denied",
    "Is a directory",
    "bad interpreter",
    "Connection refused",
    "TIMEOUT",
)


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """从任务原文里解析出来的"答案应该长什么样"

    字段:
        keys: 任务原文的 JSON 模板里点名的字段名（按出现顺序）
        annotations: 字段名 -> 模板里的中文说明（`"total_count": <总记录条数>`）
        example: 任务原文里给的**示例答案**（`{"city":"北京","total_count":0,…}`）
                 用来做类型保真校验。任务原文明确要求"不能将数字 0 写成 \\"0\\""
                 （实测报文里就有这句），示例是判断"该字段是数字还是字符串"的
                 唯一可靠依据。
    """

    keys: tuple[str, ...] = ()
    annotations: dict[str, str] = field(default_factory=dict)
    example: dict[str, Any] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return bool(self.keys)


def parse_spec(task_text: str) -> AnswerSpec:
    """从任务原文里抽出答案模板

    任务原文里有两段 JSON：

        ## 任务要求
        { "city": "北京", "total_count": <总记录条数>, … }      <- 模板（占位符）

        ## 提交形式
        {"city":"北京","total_count":0,"types":["a","b"],…}      <- 示例（合法 JSON）

    两段都要用：模板给字段名与中文说明，示例给字段类型。
    """
    text = task_text or ""
    keys: tuple[str, ...] = ()
    annotations: dict[str, str] = {}

    for block in re.findall(r"\{[^{}]*\}", text, re.S):
        pairs = re.findall(
            r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(<[^>]*>|\[[^\]]*\]|"[^"]*"|[-0-9.]+)',
            block,
        )
        if len(pairs) > len(keys):
            keys = tuple(key for key, _ in pairs)
            annotations = {key: value for key, value in pairs}

    example: dict[str, Any] = {}
    for block in re.findall(r"\{.*?\}", text, re.S):
        try:
            parsed = json.loads(block)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed and set(keys) & set(parsed):
            if len(parsed) > len(example):
                example = parsed

    return AnswerSpec(keys=keys, annotations=annotations, example=example)


# ==========================================================================
# 闸门
# ==========================================================================


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _shingles(text: str) -> set[str]:
    norm = _normalize(text)
    if len(norm) <= SHINGLE:
        return {norm} if norm else set()
    return {norm[i : i + SHINGLE] for i in range(len(norm) - SHINGLE + 1)}


def is_doc_echo(candidate: str, task_text: str) -> bool:
    """候选答案是不是任务原文/文档正文的回声

    直接把任务书原文交上去是 V1 最典型的一次事故（T7）。判据用 12 字滑窗的
    重合比例：真正的答案是几个短字段，与一段几千字的文档几乎不会有长公共
    片段；而文档回声会接近 100%。

    两道保险避免误杀：

    1. **短候选直接放过**（`ECHO_MIN_LEN`）。任务原文自带答案模板与示例
       （`{"city":"北京","total_count":0,…}`），一份只填了两个字段的正确答卷
       与示例前缀天然重合七成以上，按纯比例判定会把它误杀。
    2. **长候选才比对全文**。一份真正的"文档回声"动辄上千字，重合比例接近
       100%，长度门槛不会漏掉它。

    注意**不能**在比对前把任务原文里的 `{...}` 块（模板与示例）剥掉——它们在
    文档里占三成以上，剥掉之后连"整篇原文"这个最典型的回声都掉到阈值以下。
    """
    normalized = _normalize(candidate)
    if len(normalized) < ECHO_MIN_LEN:
        return False
    pieces = _shingles(candidate)
    if not pieces:
        return True
    source = _shingles(task_text)
    if not source:
        return False
    overlap = len(pieces & source) / len(pieces)
    return overlap >= ECHO_RATIO


def is_error_body(candidate: str) -> bool:
    """候选答案是不是一段错误信息

    两种形态：

    1. **结构化 JSON**：`{"error": "..."}`、`{"code": 401, …}`。这里必须看
       **值**而不是键——V1 曾经因为只看键名，把一份合法的
       `{"error": null, "city": "北京", …}` 判成了错误体（main 分支上修过一次
       的假阳性）。值为 `null` / `[]` / `{}` / `""` 时一律不算错误。
    2. **裸文本**：Python 回溯、`No such file`、HTTP 状态短语等。
    """
    text = (candidate or "").strip()
    if not text:
        return True

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        error_value = parsed.get("error")
        if error_value not in (None, "", [], {}):
            return True
        status = parsed.get("code", parsed.get("status", parsed.get("statusCode")))
        if isinstance(status, int) and status >= 400:
            return True
        if isinstance(status, str) and status.lower() in ("error", "fail", "failed"):
            return True
        if not parsed:
            return True
        # 一眼就是错误结构、且没有任务字段的载荷
        if set(parsed) <= {"error", "message", "code", "status", "detail", "reason"}:
            return True
        return False

    for word in _ERROR_WORDS:
        if word in text:
            return True
    if _ERROR_STATUS.search(text) and len(text) < 200:
        return True
    return False


def _as_dict(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_candidate(candidate: str, spec: AnswerSpec) -> str | None:
    """把候选答案裁剪成"恰好是任务要的那些字段"

    - 键在模板里点名过的保留，多出来的丢掉（多交字段不会加分，反而可能被
      判成格式错误）
    - 值为空字符串 / 空列表 / None 的字段丢掉（留着也是错的，丢掉更干净）
    - 一个有效字段都没有就返回 None
    """
    text = (candidate or "").strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    if spec.known:
        kept = {k: v for k, v in parsed.items() if k in spec.keys}
    else:
        kept = dict(parsed)

    kept = {
        key: value
        for key, value in kept.items()
        if value not in (None, "", [], {})
    }
    if not kept:
        return None
    return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))


def type_violations(payload: dict[str, Any], spec: AnswerSpec) -> list[str]:
    """类型保真检查：数字字段被写成字符串时返回违规字段名

    任务原文明确要求"提交答案为数字/字符串敏感型，不能将数字 0 写成 \\"0\\""。
    判据用示例答案的类型：示例里 `"total_count": 0` 是数字，答案里
    `"total_count": "0"` 就是违规。
    """
    bad: list[str] = []
    for key, sample in spec.example.items():
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(sample, bool):
            continue
        if isinstance(sample, int) and isinstance(value, str):
            bad.append(key)
        elif isinstance(sample, float) and isinstance(value, str):
            bad.append(key)
        elif isinstance(sample, list) and not isinstance(value, list):
            bad.append(key)
    return bad


def _canonical(text: str) -> str:
    """把一份答案归一化成可比较的形式（用于去重）

    `{"a": 1}` 与 `{"a":1}` 是同一份答案，不能因为空格差异被当成两次提交。
    """
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)
    except (ValueError, TypeError):
        return (text or "").strip()


def grade(candidate: str, spec: AnswerSpec) -> tuple[str | None, str]:
    """闸门主入口：返回 (可提交的答案, 理由)

    理由串用于日志——复盘时要能看出"为什么这一回合没交"。
    """
    text = (candidate or "").strip()
    if len(text) < MIN_ANSWER_LEN:
        return None, "too_short"
    if len(text) > MAX_ANSWER_LEN:
        return None, "too_long"
    if is_error_body(text):
        return None, "error_body"

    normalized = normalize_candidate(text, spec)
    if normalized is None:
        parsed = _as_dict(text)
        if parsed is not None and spec.known and not (set(parsed) & set(spec.keys)):
            return None, "no_known_field"
        # 不是 JSON：只接受"看起来就是个凭证"的短串（工程修复族的 token）
        if re.fullmatch(r"[A-Za-z0-9._\-]{6,128}", text):
            return text, "token"
        return None, "not_answer"

    try:
        payload = json.loads(normalized)
    except (ValueError, TypeError):
        return None, "not_answer"

    if spec.known and not (set(payload) & set(spec.keys)):
        return None, "no_known_field"

    bad = type_violations(payload, spec)
    if bad:
        # 类型不对就整体拒绝：交一个"数字写成字符串"的答案，通过率并不会
        # 比不交更好（判题器按字段比对），但它会占掉一次提交额度。
        # 把类型修正后再交更划算——这里先拒绝，让下一回合重取。
        return None, "type_mismatch:" + ",".join(bad)

    return normalized, "ok"


def gate(
    candidate: str | None,
    task_text: str,
    *,
    submitted: list[str] | None = None,
    rejected: list[str] | None = None,
) -> tuple[str | None, str]:
    """完整闸门：解析规格 -> 去重 -> 判分

    参数:
        candidate: 沙盒给出的候选答案（`[ANSWER]` / `[TOKEN]` / `[SOLUTION]`）
        task_text: 当前任务原文
        submitted: 本任务已经提交过的答案（完全相同的答案不重复提交）
        rejected: 被判定错误的答案（这些**必须**重取数据，不许重复提交）
    """
    if not candidate:
        return None, "no_candidate"

    spec = parse_spec(task_text)

    if is_doc_echo(candidate, task_text):
        return None, "doc_echo"

    answer, reason = grade(candidate, spec)
    if answer is None:
        return None, reason

    canonical = _canonical(answer)
    if rejected and any(_canonical(item) == canonical for item in rejected):
        return None, "already_rejected"
    if submitted and any(_canonical(item) == canonical for item in submitted):
        return None, "already_submitted"
    return answer, reason
