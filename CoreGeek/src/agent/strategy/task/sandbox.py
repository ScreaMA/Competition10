"""沙盒输出的解析与指纹。

对应设计文档V2 §6.6。

沙盒脚本与客户端之间的**唯一接口**是一组结构化标记：

    [TAG] key=value key=value      带键值的标记
    [TAG] <自由文本>                带正文的标记

设计动机（来自对战复盘）：V1 用同一个正则去 `lastCmdResult` 里捞答案，
"读文件死循环"、"命令被掐断"、"执行器没找到文件"、"取回空结果集"四种
完全不同的状态在输出上长得一模一样，于是止损判据不得不写五套、互相打架。
把状态显式打成标记之后，观测层只需要读标记，不需要猜。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# 标记行：`[TAG] 正文`
MARKER_RE = re.compile(r"^\s*\[([A-Z][A-Z0-9_]*)\]\s?(.*)$")

# 标记正文里的 key=value（key 允许点号分层，例如 API.status）。
# 值可以带双引号——鉴权头这类值本身含空格（`Authorization: Bearer xxx`），
# 不引号包裹的话解析出的就是半个值，学到的鉴权信息是错的。
KV_RE = re.compile(r'([a-zA-Z_][\w.]*)=(?:"([^"]*)"|(\S+))')

# 指纹归一化：时间戳、耗时、超长数字、空白。
# `elapsed=1.20s` 每次执行都不同但内容完全一样——不归一化它，指纹判据会在
# 每一回合都误判成"有变化"，T4 的"反复重读同一份文档"就抓不出来了。
_NORM_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T][\d:.,]+"
    r"|elapsed=\d+(?:\.\d+)?s?"
    r"|\b\d{6,}\b"
    r"|\s+"
)

# JSON 对象的粗略匹配（用于从 [ANSWER] / [SOLUTION] 里取载荷）
_JSON_RE = re.compile(r"\{.*\}", re.S)

MAX_ANSWER_LEN = 4000


@dataclass(frozen=True, slots=True)
class SandboxOutput:
    """一次 `executeCmd` 的解析结果"""

    status: str  # ok / timeout / judge_error / empty
    raw: str  # 原始 lastCmdResult
    tags: tuple[str, ...]  # 出现过的标记名（按出现顺序）
    payloads: dict[str, str]  # 标记名 -> 第一次出现的正文
    kv: dict[str, str]  # "TAG.key" -> value
    records: tuple[dict, ...]  # [DATA] 段里带的记录（可选）
    values: dict[str, str]  # 单值标记：ANSWER / SOLUTION / TOKEN
    truncated: bool
    has_done: bool
    fingerprint: str

    # --- 查询辅助 ---

    def has(self, tag: str) -> bool:
        return tag in self.payloads

    def get(self, key: str, default: str = "") -> str:
        return self.kv.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.kv[key])
        except (KeyError, ValueError):
            return default

    @property
    def answer(self) -> str | None:
        return self.values.get("ANSWER")

    @property
    def solution(self) -> str | None:
        return self.values.get("SOLUTION")

    @property
    def token(self) -> str | None:
        return self.values.get("TOKEN")

    @property
    def complete(self) -> bool:
        """脚本是否跑到了收尾（`[DONE]`）—— 没跑到就不该按"死循环"判罚"""
        return self.has_done

    @property
    def api_calls(self) -> int:
        """本回合真正发出去过的接口请求数

        这是"到底有没有去取数"的直接凭据。V1 的 `api=0` 死循环就是缺这个信号：
        读题成功的输出与取数成功的输出在日志上都是 `exitCode:0`。
        """
        return self.get_int("SCAN.api_calls")

    @property
    def api_ok(self) -> bool:
        return self.get_int("API.status") == 200

    def evidence_keys(self) -> frozenset[str]:
        """本回合"拿到过新信息"的证据键（用于与上一回合比较是否前进）"""
        keys = set()
        # `SCAN.api_calls` 是"这一回合到底有没有去取数"的直接凭据，必须进
        # 证据集：V1 的 `api=0` 死循环在日志上唯一的特征就是它恒为 0。
        for key in (
            "RECON.root", "RECON.task", "SCAN.files", "SCAN.api_calls",
            "SCAN.hits", "QUERY.root", "API.status", "DATA.n",
        ):
            if key in self.kv:
                keys.add(f"{key}={self.kv[key]}")
        for tag in ("TOKEN", "ANSWER", "SOLUTION", "CHECK", "FIX", "APIFAIL"):
            if tag in self.payloads:
                keys.add(f"{tag}={self.payloads[tag][:120]}")
        return frozenset(keys)

    def brief(self) -> str:
        """一行摘要（进 stdout 日志用；全文进 debug.log）"""
        bits = [f"status={self.status}"]
        if self.tags:
            bits.append("tags=" + ",".join(self.tags[:8]))
        if self.answer:
            bits.append("answer=yes")
        if self.token:
            bits.append("token=yes")
        bits.append(f"done={self.has_done}")
        bits.append(f"fp={self.fingerprint}")
        return " ".join(bits)


def fingerprint(raw: str) -> str:
    """输出的归一化指纹

    去掉时间戳与长数字、压缩空白后取 sha1 前 12 位。用途只有一个：
    **判断"这两回合沙盒说的是一模一样的话"**——那是卡死的直接证据。

    归一化必须去掉的东西：
      - 时间戳（`2026-09-14 07:04:48`）：每次执行都不同，但内容没变
      - 6 位以上的数字（任务 ID、耗时毫秒数）：同上
      - 空白差异（\\r\\n 与 \\n、行尾空格）
    """
    normalized = _NORM_RE.sub(" ", raw or "").strip()
    return hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()[:12]


def parse_output(raw: str) -> SandboxOutput:
    """解析 `lastCmdResult`

    `raw` 是接口文档 §1.1 里那一整串，形如 `"[exitCode:0]\\n<输出>"`。
    调用方一般直接传 `turn.sandbox.body`。
    """
    text = raw or ""
    status = "ok"
    truncated = False

    if not text.strip():
        return _empty("empty", "")

    stripped = text.rstrip()
    if stripped.endswith("[TRUNCATED]"):
        truncated = True
        stripped = stripped[: -len("[TRUNCATED]")]

    head, sep, rest = stripped.partition("\n")
    head = head.strip()
    if head == "[TIMEOUT]":
        status, text = "timeout", rest
    elif head == "[JUDGER_ERROR]":
        status, text = "judge_error", rest
    elif head.startswith("[exitCode:"):
        text = rest
    # 判题器没按约定加前缀时 text 就是原文（不能因为格式变化丢信息）

    tags: list[str] = []
    payloads: dict[str, str] = {}
    kv: dict[str, str] = {}
    values: dict[str, str] = {}
    records: list[dict] = []
    seen: set[str] = set()

    for line in text.splitlines():
        match = MARKER_RE.match(line)
        if not match:
            continue
        tag, body = match.group(1), match.group(2).strip()
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
        if tag not in payloads:
            payloads[tag] = body

        for key, quoted, plain in KV_RE.findall(body):
            kv.setdefault(f"{tag}.{key}", quoted if quoted else plain)

        if tag in ("ANSWER", "SOLUTION", "TOKEN"):
            values.setdefault(tag, _extract_value(tag, body))
        elif tag == "DATA" and "{" in body:
            payload = _extract_json(body)
            if isinstance(payload, dict):
                bucket = payload.get("records")
                if isinstance(bucket, list):
                    records.extend(x for x in bucket if isinstance(x, dict))

    return SandboxOutput(
        status=status,
        raw=text,
        tags=tuple(tags),
        payloads=payloads,
        kv=kv,
        records=tuple(records),
        values=values,
        truncated=truncated,
        has_done="DONE" in seen,
        fingerprint=fingerprint(text),
    )


def _empty(status: str, raw: str) -> SandboxOutput:
    return SandboxOutput(
        status=status,
        raw=raw,
        tags=(),
        payloads={},
        kv={},
        records=(),
        values={},
        truncated=False,
        has_done=False,
        fingerprint=fingerprint(raw),
    )


def _extract_value(tag: str, body: str) -> str:
    """从标记正文里取出要提交的值

    `[ANSWER] {"city":"北京",…}` 取 JSON 对象；
    `[TOKEN] a3f1c9…` 取第一个词。截断到 `MAX_ANSWER_LEN`，防止把
    一整份文档当成答案交上去（那正是 V1 的 T7 故障）。
    """
    text = body.strip()
    if tag == "TOKEN":
        return text.split()[0][:256] if text else ""
    match = _JSON_RE.search(text)
    if match and match.group(0).startswith("{"):
        return match.group(0)[:MAX_ANSWER_LEN]
    return text[:MAX_ANSWER_LEN]


def _extract_json(body: str) -> dict | None:
    match = _JSON_RE.search(body)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def advanced(previous: SandboxOutput | None, current: SandboxOutput) -> bool:
    """这一回合是否比上一回合"多知道了点什么"

    只看证据键的差集（新增的键，或者同名键的新值）。指纹相同必然不前进，
    但指纹不同也不一定前进（比如输出里只是多了一行时间戳），所以用证据键
    判断，指纹只作为兜底。
    """
    if previous is None:
        return True
    if current.fingerprint == previous.fingerprint:
        return False
    return bool(current.evidence_keys() - previous.evidence_keys())


def repeated(previous: SandboxOutput | None, current: SandboxOutput) -> bool:
    """与上一回合的输出是否逐字相同（归一化后）"""
    return previous is not None and current.fingerprint == previous.fingerprint
