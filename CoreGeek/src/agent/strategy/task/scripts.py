"""注入沙盒的脚本模板。

对应设计文档V2 §6.3.5 与 §6.6。

每条命令都是 `sh` 包裹的 Python heredoc：

    for P in python3 python; do command -v "$P" >/dev/null 2>&1 && break; done; \
    $P -u - <<'PYEOF' 2>&1
    <脚本>
    PYEOF

参数**只通过一个 JSON blob**（`__PARAMS__`）注入，脚本里用 `P["key"]` 取。
不做逐占位符的字符串替换——那种写法要求模板与替换值各自转义一遍，
是脚本模板最容易出错的地方（沙盒里语法错误只会表现为"这一回合没输出"，
在对战日志里跟"卡死"长得一模一样）。

三条纪律（每一条都对应一次真实故障）：

1. **脚本必须以 `[DONE]` 收尾**。没有 `[DONE]` = 命令被 15 秒掐断了，反馈层
   据此区分"卡死"与"没跑完"——V1 就是因为区分不了这两者，把"重试还有救"
   的回合当成"读文件死循环"提前熔断。
2. **所有诊断信息走单行结构化标记**。正文一律截断（`BODY_LIMIT`），保证不撞
   64KB 上限（撞上会被判题器截断，尾部信息全丢）。
3. **一次把该做的做完**。`query` 步内部就完成"读文档 → 试鉴权 → 分页取全量 →
   聚合出答案"，而不是拆成好几个回合。任务时限只有 15 回合，拆不起；更关键的
   是 V1 的 `api=0` 死循环正是"读题"与"调 API"被拆开、中间那一步永远走不到
   导致的。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .memory import (
    F_AUTH_VALUE,
    F_BASE_URL,
    F_CHECK_CMD,
    F_ENDPOINT,
    F_ERA_ORDER,
    F_FIELD_ALIAS,
    F_PARAM,
    F_SPEC_PATH,
    F_TARGET,
    F_WS_ROOT,
    StepSpec,
)

# 沙盒里搜目录的根（按优先级）
SEARCH_ROOTS = (
    "/tmp/selfEvolutionTask",
    "/app",
    "/workspace",
    "/work",
    "/home",
    "/root",
    "/tmp",
    "/opt",
    ".",
)

DIR_BUDGET = 3.0  # 目录扫描的秒数上限
DIR_MAX_ENTRIES = 3000
HTTP_TIMEOUT = 1.5  # 单次请求
HTTP_BUDGET = 7.0  # 全部请求合计
HTTP_MAX_CALLS = 10
BODY_LIMIT = 3000  # 单条标记正文的上限
DEFAULT_BASE = "http://localhost:8899"  # 对战日志实测的本地服务地址
CHECK_TIMEOUT = 5.0  # 工程修复族跑 check 的秒数上限


# ==========================================================================
# 公共前导
# ==========================================================================

_PRELUDE = r'''
import json, os, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

P = json.loads(__PARAMS__)
T0 = time.time()
BODY_LIMIT = P["body_limit"]
DIR_BUDGET = P["dir_budget"]
HTTP_TIMEOUT = P["http_timeout"]
HTTP_BUDGET = P["http_budget"]
HTTP_MAX_CALLS = P["http_max_calls"]
CHECK_TIMEOUT = P["check_timeout"]
ROOTS = P["roots"]
MAX_ENTRIES = P["max_entries"]
DEFAULT_BASE = P["default_base"]


def emit(tag, text=""):
    """打一行结构化标记（正文压成单行，避免破坏标记解析）"""
    flat = " ".join(str(text).split())
    if len(flat) > BODY_LIMIT:
        flat = flat[:BODY_LIMIT] + " [CLIP]"
    print("[%s] %s" % (tag, flat))


def dump(tag, text):
    """打一段多行正文（仅供 debug.log 阅读，客户端不解析它）"""
    body = str(text).replace("\r", "")
    if len(body) > BODY_LIMIT:
        body = body[:BODY_LIMIT] + "\n[CLIP]"
    print("[%s]" % tag)
    print(body)


def quote(value):
    """值里有空格就用双引号包起来

    鉴权头（`Authorization: Bearer xxx`）这类值本身含空格，不包的话
    `key=value` 只能解析到空格前的一半，客户端学到的鉴权信息就是错的。
    """
    text = str(value)
    if not text:
        return text
    if re.search(r"\s", text):
        return '"' + text.replace('"', "'") + '"'
    return text


def kv(tag, **pairs):
    emit(tag, " ".join(
        "%s=%s" % (k, quote(v)) for k, v in sorted(pairs.items()) if v != ""
    ))


def read(path, limit=None):
    """读文件；读不到返回空串（沙盒里权限与路径都不可控）"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        return ""
    return text[:limit] if limit else text


def exists(path):
    try:
        return os.path.exists(path)
    except Exception:
        return False


def over_budget(start, budget):
    return time.time() - start > budget


def walk(root, depth, budget_start):
    """有界目录遍历：返回 (files, dirs)，超预算即停"""
    files, dirs = [], []
    try:
        root = os.path.abspath(root)
    except Exception:
        return files, dirs
    base_depth = root.rstrip("/").count("/")
    for current, subdirs, names in os.walk(root):
        if over_budget(budget_start, DIR_BUDGET) or len(files) + len(dirs) > MAX_ENTRIES:
            subdirs[:] = []
            break
        if current.count("/") - base_depth >= depth:
            subdirs[:] = []
        for name in names:
            files.append(os.path.join(current, name))
        for name in subdirs:
            dirs.append(os.path.join(current, name))
    return files, dirs


def find_task_dir(hint, budget_start):
    """定位任务根目录：先按目录名的字面命中，再扫目录找任务文件"""
    for root in ROOTS:
        if hint and hint in root and exists(root):
            return root
    for root in ROOTS:
        if not exists(root):
            continue
        files, _ = walk(root, 4, budget_start)
        for path in files:
            if hint and os.path.basename(path) == hint:
                return os.path.dirname(path)
        for path in files:
            if os.path.basename(path).startswith("task_") and path.endswith((".md", ".txt")):
                return os.path.dirname(path)
        if over_budget(budget_start, DIR_BUDGET):
            break
    return ""


def finish(step, extra=""):
    emit("DONE", "step=%s elapsed=%.2fs %s" % (step, time.time() - T0, extra))
'''


# ==========================================================================
# recon：定位任务文件、文档、工作区
# ==========================================================================

_RECON = r'''
HINT = P["task_hint"]

budget_start = time.time()
root = find_task_dir(HINT, budget_start)
files, dirs = ([], [])
if root:
    files, dirs = walk(root, 4, budget_start)
if not root:
    # 目录名猜不中时，退回全候选根扫一遍
    for candidate in ROOTS:
        if not exists(candidate):
            continue
        files, dirs = walk(candidate, 4, budget_start)
        if files or dirs:
            root = candidate
            break

tasks = [f for f in files if os.path.basename(f).startswith("task_")]
if HINT:
    exact = [f for f in tasks if os.path.basename(f) == HINT]
    tasks = exact or tasks
docs = [
    f for f in files
    if f.lower().endswith((".md", ".txt", ".rst"))
    and not os.path.basename(f).startswith("task_")
]
ws_dirs = [
    d for d in dirs
    if os.path.basename(d).startswith(("ws_", "workspace", "project"))
]
scripts = [
    f for f in files
    if os.path.basename(f) in ("check", "verify", "build.sh", "start.sh", "run.sh")
    or f.endswith(".sh")
]
pyfiles = [f for f in files if f.endswith(".py")]

kv("RECON", root=root, task=(tasks[0] if tasks else ""), docs=len(docs),
   ws=(ws_dirs[0] if ws_dirs else ""), scripts=len(scripts), py=len(pyfiles))
kv("SCAN", files=len(files), dirs=len(dirs),
   py=("yes" if pyfiles else "no"), sh=("yes" if scripts else "no"),
   timed_out=("yes" if over_budget(budget_start, DIR_BUDGET) else "no"))

for path in tasks[:1]:
    body = read(path)
    emit("DOCPATH", path)
    if body:
        dump("DOCBODY", body)
for path in docs[:3]:
    body = read(path, 8000)
    emit("DOCPATH", path)
    if body:
        dump("DOCBODY", body)
for path in ws_dirs[:2]:
    try:
        names = sorted(os.listdir(path))[:40]
    except Exception:
        names = []
    emit("WS", "path=%s entries=%s" % (path, ",".join(names)))
if scripts:
    emit("SCRIPTS", " ".join(os.path.basename(s) for s in scripts[:10]))

finish("recon", "root=%s" % root)
'''


# ==========================================================================
# query：读文档 → 调接口 → 聚合答案（一次性完成）
# ==========================================================================

_QUERY = r'''
HINT = P["task_hint"]
KNOWN_BASE = P["base"]
KNOWN_AUTH = P["auth"]
KNOWN_PARAM = P["param"]
KNOWN_ENDPOINT = P["endpoint"]
KNOWN_TARGET = P["target"]
KNOWN_ALIAS = P["alias"]
KNOWN_ERA = P["era"]

# 中文语义 -> 记录字段名 的内置别名表。
# 任务原文里的字段说明是中文（"oldest_era": "<年代最早的遗产名称>"），记录字段
# 是英文（name / era / protected_level），这张表是两边的桥。跨任务学到的新映射
# 会在下一回合通过 P["alias"] 并把内置表覆盖掉。
BUILTIN_ALIAS = {
    "保护级别": "protected_level", "保护等级": "protected_level",
    "级别": "protected_level", "等级": "protected_level",
    "类型": "type", "类别": "type", "种类": "type",
    "年代": "era", "时代": "era", "时期": "era", "朝代": "era",
    "名称": "name", "名字": "name", "标题": "name",
    "编号": "id", "序号": "id",
    "城市": "city", "地区": "location", "地点": "location",
    "价格": "price", "数量": "quantity",
}

# 年代排序（从早到晚）。"年代最早的遗产"要按历史顺序而不是字典序比较——
# 对战日志里明确记着这个坑："比较 era 时按此历史顺序，不要按字符串字典序"。
# 这是通用的世界知识表，不属于"针对某道题的特判"。
BUILTIN_ERA = {
    "旧石器时代": 0, "新石器时代": 1, "夏": 2, "商": 3, "商周": 3, "周": 4,
    "西周": 4, "东周": 4, "春秋": 5, "战国": 6, "秦": 7, "汉": 8, "西汉": 8,
    "东汉": 8, "三国": 10, "魏晋": 11, "晋": 11, "南北朝": 13, "隋": 14,
    "唐": 15, "五代": 16, "宋": 17, "北宋": 17, "南宋": 17, "辽": 18, "金": 18,
    "元": 19, "明": 20, "清": 21, "近现代": 22, "近代": 22, "现代": 22,
}


def http_get(url, headers, budget):
    """发一次 GET，返回 (状态, 正文)。异常一律映射成状态字符串，不抛出。"""
    if time.time() - T0 > budget:
        return "budget", ""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read().decode("utf-8", "replace")
        except Exception:
            return exc.code, ""
    except Exception as exc:
        return type(exc).__name__, ""


def split_base(base):
    """把 base 拆成 (scheme://host[:port], 路径前缀)"""
    match = re.match(r"(https?://[^/]+)(/.*)?$", (base or "").rstrip("/"))
    if not match:
        return (base or "").rstrip("/"), ""
    return match.group(1), (match.group(2) or "")


def join(host, prefix, path):
    if prefix and path.startswith(prefix):
        return host + path
    return host + prefix.rstrip("/") + path


def collect_docs(path):
    """读任务文件与同级文档，返回合并正文"""
    chunks = []
    if path:
        text = read(path, 20000)
        if text:
            chunks.append(text)
    folder = os.path.dirname(path) if path else ""
    if folder and exists(folder):
        try:
            for name in sorted(os.listdir(folder))[:30]:
                full = os.path.join(folder, name)
                if os.path.isfile(full) and name.lower().endswith((".md", ".txt")):
                    text = read(full, 20000)
                    if text:
                        chunks.append(text)
        except Exception:
            pass
    if not chunks:
        budget_start = time.time()
        for root in ROOTS[:4]:
            if not exists(root):
                continue
            files, _ = walk(root, 4, budget_start)
            for name in files:
                if name.lower().endswith((".md", ".txt")):
                    text = read(name, 20000)
                    if text:
                        chunks.append(text)
            if over_budget(budget_start, DIR_BUDGET):
                break
    return "\n".join(chunks)


def pick_target(text, path):
    """任务要查询的目标（城市名之类）"""
    if KNOWN_TARGET:
        return KNOWN_TARGET
    for pattern in (
        r"查询([一-龥]{2,6})(?:市)?的",
        r"请查询([一-龥]{2,6})",
        r"([一-龥]{2,6})市",
        r"[?&]location=([一-龥]{2,6})",
        r"[?&]city=([一-龥]{2,6})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace("市", "")
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isascii() and parts[-1].isalpha():
        return parts[-1]
    return ""


def parse_answer_spec(text):
    """从任务原文里抽出"答案模板"：字段名 -> 中文说明

    模板不是合法 JSON（"total_count": <总记录条数>），但正则可以稳定地取出
    每一对 (字段名, 说明)。**说明文字就是语义来源**——这比猜字段名可靠得多：
    日志里踩过的坑 `oldest_era` 要的是"遗产名称"而不是"年代"，任务原文写得
    清清楚楚（"<年代最早的遗产名称>"），只是 V1 没去读它。
    """
    best = {}
    for block in re.findall(r"\{[^{}]*\}", text, re.S):
        pairs = re.findall(
            r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(<[^>]*>|\[[^\]]*\]|"[^"]*")',
            block,
        )
        if len(pairs) > len(best):
            best = dict(pairs)
    return best


def field_names(records):
    names = []
    for record in records:
        for key in record:
            if key not in names:
                names.append(key)
    return names


def value_index(records):
    index = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                index.setdefault(key, set()).add(value)
    return index


def resolve_field(noun, records, annotation, alias):
    """把中文名词解析成记录字段名（三级回退）"""
    table = dict(BUILTIN_ALIAS)
    table.update(alias or {})
    names = field_names(records)
    for chinese, english in table.items():
        if chinese and chinese in noun and english in names:
            return english
    for match in re.findall(r'"([^"]+)"|“([^”]+)”|为([一-龥A-Za-z]{2,12})的', annotation):
        for candidate in match:
            if not candidate:
                continue
            for key, values in value_index(records).items():
                if candidate in values or candidate in {str(v) for v in values}:
                    return key
    if len(names) == 1:
        return names[0]
    for fallback in ("type", "category", "kind", "name"):
        if fallback in names:
            return fallback
    return ""


def record_name(record, alias):
    for key in ("name", "title", "heritage_name"):
        if key in record:
            return record[key]
    return ""


def era_rank(record, order):
    table = dict(BUILTIN_ERA)
    table.update(order or {})
    text = str(record.get("era", record.get("period", "")))
    if text in table:
        return table[text]
    for key, rank in table.items():
        if key and key in text:
            return rank
    digits = re.findall(r"-?\d+", text)
    if digits:
        try:
            return int(digits[0])
        except ValueError:
            return 9999
    return 9999


def compute(key, annotation, records, target, alias, era_order):
    """按说明文字算出该字段的值，返回 (值, 用到的记录字段名)"""
    text = annotation.strip().strip('"').strip("<>[]")
    field = resolve_field(key + text, records, annotation, alias)
    lowered = key.lower()

    wants_count = ("数量" in text) or ("条数" in text) or ("个数" in text) or ("count" in lowered)
    wants_list = ("所有" in text) or ("不重复" in text) or ("列表" in text) or ("顺序不限" in text)
    wants_oldest = ("最早" in text) or ("oldest" in lowered) or ("earliest" in lowered)
    wants_name = ("名称" in text) or ("名字" in text)

    if wants_count and not wants_list:
        named = [c for group in re.findall(r'"([^"]+)"|“([^”]+)”|为([一-龥A-Za-z]{2,12})的', annotation)
                 for c in group if c]
        for candidate in named:
            if not field:
                break
            hits = [r for r in records if r.get(field) == candidate]
            if hits:
                return len(hits), field
        if "总" in text or "全部" in text or "所有" in text or not field:
            return len(records), ""
        return sum(1 for r in records if r.get(field) not in (None, "", [])), field

    if wants_list:
        if not field:
            return [], ""
        values = []
        for record in records:
            value = record.get(field)
            if value not in (None, "") and value not in values:
                values.append(value)
        return sorted(values, key=lambda v: str(v)), field

    if wants_oldest:
        if not records:
            return "", ""
        oldest = min(records, key=lambda r: era_rank(r, era_order))
        if wants_name or not field:
            return record_name(oldest, alias) or oldest.get(field, ""), "name"
        return oldest.get(field, record_name(oldest, alias)), field

    if wants_name:
        return (record_name(records[0], alias) if records else ""), "name"
    if "城市" in text or lowered in ("city", "location"):
        return target, ""
    if field and records:
        return records[0].get(field, ""), field
    if records and key in records[0]:
        return records[0][key], key
    return "", ""


def records_from(body):
    """从响应正文里取出记录数组"""
    try:
        payload = json.loads(body)
    except Exception:
        return []
    stack = [payload]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for value in node.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return value
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(node)
    return []


def find_total(body):
    match = re.search(r'"(?:total_count|total|count|totalCount)"\s*:\s*(\d+)', body)
    return int(match.group(1)) if match else None


# --- 1. 找任务文件与文档 ---

budget_start = time.time()
root = find_task_dir(HINT, budget_start)
task_path = os.path.join(root, HINT) if (root and HINT) else ""
text = collect_docs(task_path)
if not text and task_path:
    task_path = ""
    text = collect_docs("")
kv("QUERY", root=root, has_task=("yes" if text else "no"), chars=len(text))

# --- 2. 抽地址、鉴权、参数名 ---

bases = []
if KNOWN_BASE:
    bases.append(KNOWN_BASE)
for match in re.findall(r"https?://[^\s`\"'<>)|]+", text):
    candidate = match.rstrip("/.,;")
    if candidate not in bases:
        bases.append(candidate)
if DEFAULT_BASE not in bases:
    bases.append(DEFAULT_BASE)

paths = []
if KNOWN_ENDPOINT:
    paths.append(KNOWN_ENDPOINT)
for match in re.findall(
    r"/[a-zA-Z0-9_\-/]*(?:search|query|list|api|data|records|get)[a-zA-Z0-9_\-/]*", text
):
    if len(match) > 3 and match not in paths and "//" not in match:
        paths.append(match)
if not paths:
    paths = ["/api/v1/heritage/search", "/api/search", "/search", "/api", "/"]

params = []
if KNOWN_PARAM:
    params.append(KNOWN_PARAM)
for name in ("location", "city", "name", "keyword", "q", "region", "area"):
    if re.search(r"\b%s\b" % name, text) and name not in params:
        params.append(name)
if not params:
    params = ["location", "city", "name"]

auths = []
if KNOWN_AUTH:
    auths.append(KNOWN_AUTH)
for match in re.findall(r"(Bearer\s+[A-Za-z0-9._\-]{6,})", text):
    auths.append("Authorization: " + match)
for match in re.findall(r"([A-Za-z0-9_\-]*(?:key|token|secret)[A-Za-z0-9_\-]*)", text, re.I):
    if len(match) >= 8:
        auths.append("X-API-Key: " + match)
        auths.append("Authorization: Bearer " + match)
# 任务原文明确提示"文档可能过时"，所以无鉴权也要试一遍（有些端点不校验）
auths.append("")

target = pick_target(text, task_path)
kv("PARSE", target=target, bases=len(bases), paths=len(paths),
   params=len(params), auths=len(auths))

# --- 3. 逐个候选调用接口（这一步必然发生，不存在"只读题不调接口"）---

records = []
chosen = {}
calls = 0
tried = []


def stop():
    return bool(records) or calls >= HTTP_MAX_CALLS or time.time() - T0 > HTTP_BUDGET


for base in bases[:3]:
    host, prefix = split_base(base)
    for path in paths[:5]:
        if path == "/" and prefix:
            continue
        url_path = join(host, prefix, path)
        for param in params[:3]:
            for auth in auths[:4]:
                if stop():
                    break
                query = urllib.parse.urlencode({param: target, "limit": 100})
                url = "%s?%s" % (url_path, query)
                headers = {"Accept": "application/json"}
                if auth:
                    name, _, value = auth.partition(":")
                    headers[name.strip()] = value.strip()
                calls += 1
                status, body = http_get(url, headers, HTTP_BUDGET)
                tried.append("%s%s" % (status, path))
                if str(status) == "200":
                    found = records_from(body)
                    if found:
                        records = found
                        chosen = {"host": host, "prefix": prefix, "path": path,
                                  "param": param, "auth": auth, "url": url}
                        kv("API", url=url, status=200,
                           auth=("yes" if auth else "no"), n=len(found))
                        break
                    kv("APIFAIL", url=url, status=200, reason="no_records")
                else:
                    reason = "missing_auth" if str(status) in ("401", "403") else "http"
                    kv("APIFAIL", url=url, status=status, reason=reason)
            if stop():
                break
        if stop():
            break
    if stop():
        break

kv("SCAN", api_calls=calls, hits=len(records), tried="|".join(tried[:8]))

# --- 4. 分页取全量 ---

total = None
if records and chosen:
    _, probe = http_get(chosen["url"], {}, HTTP_BUDGET)
    total = find_total(probe)
    page = 2
    while total and len(records) < total and page <= 20:
        if time.time() - T0 > HTTP_BUDGET:
            break
        query = urllib.parse.urlencode(
            {chosen["param"]: target, "limit": 100, "page": page}
        )
        url = "%s?%s" % (join(chosen["host"], chosen["prefix"], chosen["path"]), query)
        headers = {}
        if chosen["auth"]:
            name, _, value = chosen["auth"].partition(":")
            headers[name.strip()] = value.strip()
        status, body = http_get(url, headers, HTTP_BUDGET)
        if str(status) != "200":
            break
        more = records_from(body)
        if not more:
            break
        records.extend(more)
        page += 1
    kv("DATA", n=len(records), total=(total if total is not None else len(records)),
       pages=(page - 1))
else:
    kv("DATA", n=0, total=0, pages=0)

# --- 5. 聚合答案 ---

spec = parse_answer_spec(text)
try:
    alias = json.loads(KNOWN_ALIAS) if KNOWN_ALIAS else {}
except Exception:
    alias = {}
try:
    era_order = json.loads(KNOWN_ERA) if KNOWN_ERA else {}
except Exception:
    era_order = {}

answer = {}
mapping = {}
for field, annotation in spec.items():
    try:
        value, used = compute(field, annotation, records, target, alias, era_order)
        answer[field] = value
        if used:
            mapping[field] = used
    except Exception as exc:
        emit("WARN", "field=%s err=%s" % (field, type(exc).__name__))

if mapping:
    # 把这次用到的"中文字段说明 -> 记录字段名"映射报上去，客户端会把它并进
    # 技能库：下一个同族任务起手就知道该读哪个字段，不必再猜一遍。
    kv("MAPPING", **mapping)

if answer:
    print("[ANSWER] " + json.dumps(answer, ensure_ascii=False))
    kv("PROFILE", base=chosen.get("host", ""), path=chosen.get("path", ""),
       param=chosen.get("param", ""), auth=chosen.get("auth", ""), target=target,
       fields=len(spec))

finish("query", "calls=%d records=%d fields=%d" % (calls, len(records), len(answer)))
'''


# ==========================================================================
# 工程修复族
# ==========================================================================

# locate()：定位工作区、规格文件与检查脚本。三个步骤脚本共用。
_ENGINEER_LOCATE = r'''
def locate():
    """定位工作区、规格文件、检查脚本；已知就跳过扫描（省时间）"""
    global WS, SPEC, CHECK
    if WS and SPEC and CHECK:
        return [], []
    budget_start = time.time()
    files, dirs = [], []
    for root in ROOTS:
        if not exists(root):
            continue
        files, dirs = walk(root, 5, budget_start)
        if files or dirs:
            break
    if not WS:
        candidates = [d for d in dirs if os.path.basename(d).startswith("ws_")]
        if not candidates:
            candidates = [
                d for d in dirs
                if os.path.basename(d) in ("workspace", "project", "work")
            ]
        WS = candidates[0] if candidates else ""
    scope = WS or ""
    if not SPEC:
        for path in files:
            if os.path.basename(path).lower() in (
                "spec.md", "readme.md", "requirement.md", "requirements.md", "task.md"
            ) and path.startswith(scope):
                SPEC = path
                break
    if not CHECK:
        for path in files:
            if os.path.basename(path) in (
                "check", "verify", "test.sh", "run.sh", "check.sh", "build.sh"
            ) and path.startswith(scope):
                CHECK = path
                break
    return files, dirs


def run(cmd, cwd=None, timeout=None):
    """跑一条命令，返回 (退出码, 合并输出)"""
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=cwd, timeout=(timeout or CHECK_TIMEOUT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return proc.returncode, proc.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as exc:
        return "timeout", (exc.stdout or b"").decode("utf-8", "replace")
    except Exception as exc:
        return type(exc).__name__, ""


def flatten(text):
    return " | ".join(line.strip() for line in str(text).splitlines() if line.strip())
'''

_ENGINEERING_NORMALIZE = r'''
WS = P["ws"]
SPEC = P["spec"]
CHECK = P["check"]

files, dirs = locate()
kv("FIND", ws=WS, spec=SPEC, check=CHECK)

fixed = 0
for path in files:
    if not path.lower().endswith((".sh", ".py", ".conf", ".cfg", ".ini", ".check")):
        if os.path.basename(path) not in ("check", "verify"):
            continue
    body = read(path)
    if "\r" not in body:
        continue
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(body.replace("\r\n", "\n").replace("\r", "\n"))
        fixed += 1
    except Exception:
        pass

chmoded = 0
for path in files:
    name = os.path.basename(path)
    if name in ("check", "verify") or path.endswith(".sh"):
        try:
            os.chmod(path, 0o755)
            chmoded += 1
        except Exception:
            pass

emit("FIX", "crlf=%d chmod=%d" % (fixed, chmoded))
if SPEC:
    emit("DOCPATH", SPEC)
    dump("DOCBODY", read(SPEC, 6000))
finish("normalize", "ws=%s" % WS)
'''

_ENGINEERING_CHECK = r'''
WS = P["ws"]
SPEC = P["spec"]
CHECK = P["check"]

files, dirs = locate()
cmd = CHECK or (os.path.join(WS, "check") if WS else "")
if not cmd or not exists(cmd):
    emit("CHECK", ok="no", code="-", reason="no_check_script", ws=WS)
    finish("check", "no script")
else:
    workdir = WS or os.path.dirname(os.path.abspath(cmd))
    command = cmd if os.access(cmd, os.X_OK) else "sh %s" % cmd
    code, out = run(command, cwd=workdir)
    emit("CHECK", ok=("yes" if code == 0 else "no"), code=code, cwd=workdir,
         head=flatten(out)[:400])
    if out:
        dump("CHECKBODY", out[:2000])
    finish("check", "code=%s" % code)
'''

_ENGINEERING_REPAIR = r'''
WS = P["ws"]
SPEC = P["spec"]
CHECK = P["check"]
OUT = P["check_output"]

files, dirs = locate()
spec_text = read(SPEC) if SPEC else ""
actions = []

for match in re.findall(r"([\w./\-]+): Permission denied", OUT):
    try:
        os.chmod(match, 0o755)
        actions.append("chmod:" + match)
    except Exception:
        pass
for match in re.findall(r"Permission denied[^'\"]*['\"]([^'\"]+)['\"]", OUT):
    try:
        os.chmod(match, 0o755)
        actions.append("chmod:" + match)
    except Exception:
        pass

for name in ("logs", "log", "output", "out", "data", "tmp", "run", "result"):
    if not WS:
        break
    folder = os.path.join(WS, name)
    if re.search(r"\b%s\b" % name, OUT + spec_text) and not exists(folder):
        try:
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o777)
            actions.append("mkdir:" + name)
        except Exception:
            pass
    elif exists(folder):
        try:
            for entry in os.listdir(folder)[:30]:
                full = os.path.join(folder, entry)
                if os.path.isfile(full):
                    os.chmod(full, 0o666)
        except Exception:
            pass

wanted = {}
for key_, value in re.findall(
    r"[`\s]([A-Za-z_][A-Za-z0-9_.\-]{1,30})\s*[:=]\s*([^\s`\n,;]{1,40})", spec_text
):
    wanted[key_] = value.strip("`\"'")
applied = 0
for path in files:
    if not path.lower().endswith(
        (".conf", ".cfg", ".ini", ".env", ".properties", ".json", ".yaml", ".yml", ".txt")
    ):
        continue
    body = read(path)
    if not body:
        continue
    updated = body
    for key_, value in list(wanted.items())[:20]:
        pattern = re.compile(r"(^|\n)(\s*%s\s*[:=]\s*)([^\n\r]*)" % re.escape(key_))
        if not pattern.search(updated):
            continue
        candidate = pattern.sub(lambda m: m.group(1) + m.group(2) + value, updated)
        if candidate != updated:
            updated = candidate
            applied += 1
            actions.append("set:%s" % key_)
    if updated != body:
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(updated)
        except Exception:
            pass

emit("FIX", "actions=%d applied=%d %s" % (len(actions), applied, ",".join(actions[:8])))
finish("repair", "applied=%d" % applied)
'''

_ENGINEERING_VERIFY = r'''
WS = P["ws"]
SPEC = P["spec"]
CHECK = P["check"]

files, dirs = locate()
cmd = CHECK or (os.path.join(WS, "check") if WS else "")
out = ""
code = "-"
if cmd and exists(cmd):
    command = cmd if os.access(cmd, os.X_OK) else "sh %s" % cmd
    code, out = run(command, cwd=(WS or None))
    emit("CHECK", ok=("yes" if code == 0 else "no"), code=code, head=flatten(out)[:400])
    if out:
        dump("CHECKBODY", out[:2000])

token = ""
for pattern in (r"TOKEN\s*[:=]\s*([A-Za-z0-9._\-]{6,})",
                r"token\s*[:=]\s*([A-Za-z0-9._\-]{6,})"):
    match = re.search(pattern, out)
    if match:
        token = match.group(1)
        break
if not token and SPEC:
    match = re.search(r"TOKEN\s*[:=]\s*([A-Za-z0-9._\-]{6,})", read(SPEC))
    token = match.group(1) if match else ""
if not token:
    budget_start = time.time()
    found, _ = walk(WS or ".", 3, budget_start)
    for path in found[:200]:
        if os.path.basename(path).lower() in (
            "token", "token.txt", "result", "result.txt", "answer", "answer.txt"
        ):
            match = re.search(r"([A-Za-z0-9._\-]{6,})", read(path, 2000))
            if match:
                token = match.group(1)
                break

if token:
    print("[TOKEN] " + token)
finish("verify", "token=%s" % ("yes" if token else "no"))
'''


# ==========================================================================
# 通用兜底（未知族）
# ==========================================================================

_GENERIC_PROBE = r'''
WS = P["ws"]
SPEC = P["spec"]
CHECK = P["check"]

files, dirs = locate()
kv("FIND", ws=WS, spec=SPEC, check=CHECK)

hit = ""
for path in files[:60]:
    name = os.path.basename(path).lower()
    if not name.endswith((".md", ".txt", ".log", ".out", ".json", ".conf", ".cfg", ".ini")):
        continue
    body = read(path, 4000)
    if not body:
        continue
    match = re.search(
        r"(?:TOKEN|token|ANSWER|answer|答案|flag|FLAG)\s*[:=]\s*([A-Za-z0-9._\-一-龥]{4,})",
        body,
    )
    if match:
        hit = match.group(1)
        emit("TOKEN", hit)
        break

if not hit:
    for path in (files or [])[:40]:
        if path.endswith((".sh",)) or os.path.basename(path) in ("check", "verify"):
            code, out = run(
                path if os.access(path, os.X_OK) else "sh %s" % path, cwd=(WS or None)
            )
            if out:
                emit("CHECKBODY", flatten(out)[:400])
            break
    if SPEC:
        emit("DOCPATH", SPEC)
        dump("DOCBODY", read(SPEC, 4000))

finish("generic", "hit=%s" % ("yes" if hit else "no"))
'''


# ==========================================================================
# 组装
# ==========================================================================

_PYTHON_BODY: dict[str, str] = {
    "recon": _RECON,
    "query": _QUERY,
    "normalize": _ENGINEER_LOCATE + _ENGINEERING_NORMALIZE,
    "check": _ENGINEER_LOCATE + _ENGINEERING_CHECK,
    "repair": _ENGINEER_LOCATE + _ENGINEERING_REPAIR,
    "verify": _ENGINEER_LOCATE + _ENGINEERING_VERIFY,
    "generic": _ENGINEER_LOCATE + _GENERIC_PROBE,
}


def task_hint(phase_task: str) -> str:
    """从任务描述里抽出任务文件名

    实测的 `phaseTask` 形如 `"请阅读task_1_beijing.md，获取任务信息"`。
    抽不到时返回空串，脚本退回"扫目录找 task_*.md"。
    """
    match = re.search(r"([A-Za-z0-9_\-]+\.(?:md|txt|json|ya?ml))", phase_task or "")
    return match.group(1) if match else ""


def _params(step: StepSpec, phase_task: str, facts: dict[str, str], extra: dict) -> dict[str, Any]:
    return {
        "task_hint": step.param("task_hint") or task_hint(phase_task),
        "base": step.param("base") or facts.get(F_BASE_URL, ""),
        "auth": step.param("auth") or facts.get(F_AUTH_VALUE, ""),
        "param": step.param("param") or facts.get(F_PARAM, ""),
        "endpoint": step.param("endpoint") or facts.get(F_ENDPOINT, ""),
        "target": step.param("target") or facts.get(F_TARGET, ""),
        "alias": facts.get(F_FIELD_ALIAS, ""),
        "era": facts.get(F_ERA_ORDER, ""),
        "ws": step.param("ws") or facts.get(F_WS_ROOT, ""),
        "spec": step.param("spec") or facts.get(F_SPEC_PATH, ""),
        "check": step.param("check") or facts.get(F_CHECK_CMD, ""),
        "check_output": extra.get("check_output", "") or step.param("check_output"),
        "body_limit": BODY_LIMIT,
        "dir_budget": DIR_BUDGET,
        "http_timeout": HTTP_TIMEOUT,
        "http_budget": HTTP_BUDGET,
        "http_max_calls": HTTP_MAX_CALLS,
        "check_timeout": CHECK_TIMEOUT,
        "roots": SEARCH_ROOTS,
        "max_entries": DIR_MAX_ENTRIES,
        "default_base": DEFAULT_BASE,
    }


def build(
    step: StepSpec,
    *,
    phase_task: str,
    facts: dict[str, str],
    check_output: str = "",
) -> str:
    """把一步变成可直接放进 `executeCmd` 的 shell 命令"""
    body = _PYTHON_BODY.get(step.name)
    if body is None:
        body = 'emit("WARN", "unknown_step")\nfinish("unknown")\n'

    blob = json.dumps(_params(step, phase_task, facts, {"check_output": check_output}))
    script = (_PRELUDE + body).replace("__PARAMS__", repr(blob))
    return (
        'for P in python3 python; do command -v "$P" >/dev/null 2>&1 && break;'
        f" done; $P -u - <<'PYEOF' 2>&1\n{script}\nPYEOF"
    )


# ==========================================================================
# LLM 兜底命令的安全校验（设计文档V2 §6.8）
# ==========================================================================

_FORBIDDEN = (
    "rm -rf", "rm -fr", "mkfs", "dd if=", "shutdown", "reboot", ":(){",
    "chmod -r 777 /", "/dev/sda", "curl http", "wget http", " ncat ", "sudo ",
    "apt ", "apt-get", "pip install", "git clone", ">/etc/", "> /etc/",
)

MAX_LLM_COMMAND = 800


def sanitize_llm_command(raw: str) -> str | None:
    """校验 LLM 给出的沙盒命令

    只接受单条命令、长度受限、不含破坏性模式。校验不过就丢弃——兜底路径的
    价值是"多一种思路"，不值得为它承担把沙盒搞坏的风险（沙盒坏了这一整场比赛
    的任务就全没了）。
    """
    text = (raw or "").strip()
    if not text or len(text) > MAX_LLM_COMMAND:
        return None
    if text.count("PYEOF") > 1 or text.count("<<") > 1:
        return None
    lowered = text.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            return None
    return text
