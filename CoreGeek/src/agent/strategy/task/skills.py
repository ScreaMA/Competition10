"""任务族识别与 SOP 阶梯。

对应设计文档V2 §6.3.2 与 §6.3.4。

任务书 §5.3 对自进化类的定义是：

    玩家需要根据任务1探索的内容，形成固定 SOP 或者 SKILL，实现 Agent 自进化，
    进而快速做出后续任务。

这里把"固定 SOP"落成**可执行的步骤序列**（`StepSpec`），而不是一句自然语言
描述——下一场对局里它会被真的执行。族识别 + 阶梯是"第一次怎么做"，
`Memory` 里的 `Skill` 是"第二次开始怎么做"。
"""

from __future__ import annotations

from .answer import parse_spec
from .memory import Skill, StepSpec, TaskRun
from .sandbox import SandboxOutput

FAMILY_API = "api-query"
FAMILY_ENGINEERING = "engineering-fix"
FAMILY_UNKNOWN = "unknown"

# 各族的阶梯（按顺序执行；命中的 SKILL 会覆盖它）
LADDERS: dict[str, tuple[str, ...]] = {
    # api-query：recon 摸清沙盒 → query 一次性完成"读文档→调接口→聚合答案"。
    # 只有两步，因为 query 必须自己把该做的做完——V1 的 api=0 死循环就是
    # "读题"和"调 API"之间那一步永远走不到。
    FAMILY_API: ("recon", "query"),
    # engineering-fix：跑一次 check 才知道要修什么，所以是
    # normalize（修 CRLF）→ check（看失败）→ repair（按 spec 修）→ verify（重跑拿 TOKEN）
    FAMILY_ENGINEERING: ("recon", "normalize", "check", "repair", "verify"),
    # 未知族：摸清沙盒 + 通读文档找 credential
    FAMILY_UNKNOWN: ("recon", "generic"),
}

# 识别用的关键词（中文任务原文与英文沙盒路径都覆盖）
_API_TOKENS = (
    "http://", "https://", "api_docs", "api key", "x-api-key", "authorization",
    "api", "接口", "查询", "请求", "文档", "端点", "url",
)
_ENGINEER_TOKENS = (
    "spec.md", "ws_1", "./check", "bad interpreter", "chmod", "权限", "配置",
    "修复", "工程", "部署", "工作区", "脚本", "目录要求", "compile", "构建",
)

# **路径本身就是最强的信号**：任务目录名直接把族写在里面
# —— `/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/` 与 `.../2-engineering-fix/`。
# 这些命中一次就够定族，所以给一个压过所有关键词的权重。
_API_PATH_TOKENS = ("unknown-api", "selfevolutiontask/1-fixed-step/1-")
_ENGINEER_PATH_TOKENS = ("engineering-fix", "selfevolutiontask/1-fixed-step/2-")

PATH_WEIGHT = 8


def classify(task_text: str, recon: SandboxOutput | None) -> tuple[str, str]:
    """识别任务族，返回 (家族, 证据串)

    判据是**任务原文 + 沙盒侦察的原文**，而不是任务原文一个。

    实测教训：`phaseTask` 在很多任务里只是一句"请阅读 task_1_beijing.md，
    获取任务信息"（27 字节），**里面一个关键词都没有**——只按它分类必然得到
    `unknown`，于是走 `recon → generic` 阶梯，**`query` 那一步从头到尾没走过**，
    任务直接 0 分。真正的族信号在侦察回来的沙盒输出里：文件路径、`API_DOCS.md`、
    `X-API-Key`、`spec.md`、`./check`……

    评分而不是短路：`engineering-fix` 的原文里也可能出现"读取文档"之类的词。
    """
    text = task_text or ""
    # 侦察输出是第二份证据；截断防止把整篇文档喂进关键词匹配
    scout = (recon.raw[:8000] if recon is not None else "")
    evidence_text = text + "\n" + scout
    lowered = evidence_text.lower()

    api_score = sum(1 for token in _API_TOKENS if token in lowered)
    eng_score = sum(1 for token in _ENGINEER_TOKENS if token in lowered)

    path_api = sum(1 for token in _API_PATH_TOKENS if token in lowered)
    path_eng = sum(1 for token in _ENGINEER_PATH_TOKENS if token in lowered)
    api_score += path_api * PATH_WEIGHT
    eng_score += path_eng * PATH_WEIGHT

    if recon is not None:
        if recon.has("PROFILE") or recon.has("API"):
            api_score += 3
        if recon.get("RECON.ws") or recon.get("FIND.ws"):
            eng_score += 3
        if recon.has("CHECK") or recon.has("CHECKBODY") or recon.has("FIX"):
            eng_score += 3

    evidence = (
        f"api={api_score} eng={eng_score}"
        f"(path {path_api}/{path_eng})"
    )
    if eng_score > api_score:
        return FAMILY_ENGINEERING, evidence
    if api_score > 0:
        return FAMILY_API, evidence
    return FAMILY_UNKNOWN, evidence


def signature(family: str, task_text: str) -> str:
    """技能的匹配键

    刻意**不含任务的目标参数**（城市名之类）——"查询北京文化遗产"与
    "查询上海文化遗产"必须命中同一条技能，那正是自进化的意义所在。
    用"族 + 答案字段集合"作签名：字段一样说明要做的统计口径一样，
    参数不同而已；字段不一样（比如天气任务的字段完全不同）就不该串用。
    """
    spec = parse_spec(task_text)
    fields = ",".join(sorted(spec.keys)) if spec.known else "-"
    return f"{family}|{fields}"


def ladder(family: str) -> tuple[str, ...]:
    return LADDERS.get(family, LADDERS[FAMILY_UNKNOWN])


def first_steps(run: TaskRun, skill: Skill | None) -> tuple[StepSpec, ...]:
    """这次任务要走的步骤序列

    命中 SKILL 且未被证伪时用缓存的步骤（这是"快速做出后续任务"的路径）；
    否则用族的默认阶梯。
    """
    if skill is not None and skill.steps:
        return tuple(skill.steps)
    return tuple(StepSpec(name) for name in ladder(run.family))


def build_skill(run: TaskRun, facts: dict[str, str]) -> Skill | None:
    """由一次**成功的任务**提炼技能

    只把实际走通的步骤（`run.used_steps`）收进去，并且要求至少走完两步——
    一步就完成的"技能"没有复用价值，反而会污染匹配。

    族专属的事实（接口地址、鉴权头、参数名、字段别名）一并带上：下一个同族
    任务起手就知道该连哪个地址、带哪个头，不必重新探索。
    """
    if len(run.used_steps) < 1:
        return None
    skill_facts = {key: value for key, value in facts.items() if key.startswith("api.")}
    if run.family == FAMILY_ENGINEERING:
        skill_facts.update(
            {key: value for key, value in facts.items() if key.startswith("sandbox.")}
        )
    return Skill(
        family=run.family,
        signature=run.key,
        steps=list(run.used_steps),
        facts=skill_facts,
        wins=1,
    )


def next_step_name(family: str, step_index: int) -> str | None:
    order = ladder(family)
    if 0 <= step_index < len(order):
        return order[step_index]
    return None


def step_count(family: str, skill: Skill | None) -> int:
    if skill is not None and skill.steps:
        return len(skill.steps)
    return len(ladder(family))
