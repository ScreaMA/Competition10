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
_API_TOKENS = ("http://", "https://", "api", "接口", "查询", "请求", "文档", "端点")
_ENGINEER_TOKENS = (
    "spec", "修复", "配置", "工程", "工作区", "check", "构建", "编译",
    "权限", "脚本", "部署", "服务", "启动",
)


def classify(task_text: str, recon: SandboxOutput | None) -> tuple[str, str]:
    """识别任务族，返回 (家族, 证据串)

    只看**任务原文**与**沙盒侦察输出**这两样，不猜。识别结果会连同证据一起
    写进 `Memory.facts`，同族的下一个任务可以直接沿用。

    评分而不是短路：`1-fixed-step/2-engineering-fix` 的原文里也可能出现
    "读取文档"之类的词，单纯按第一个命中的关键词分类会分错。
    """
    text = task_text or ""
    lowered = text.lower()
    api_score = sum(1 for token in _API_TOKENS if token in lowered or token in text)
    eng_score = sum(1 for token in _ENGINEER_TOKENS if token in lowered or token in text)

    if recon is not None:
        if recon.has("PROFILE") or recon.has("API"):
            api_score += 3
        if recon.get("RECON.ws"):
            eng_score += 3
        if recon.has("CHECK") or recon.has("FIX"):
            eng_score += 3
        if recon.get("SCAN.py") == "yes" and recon.get("SCAN.sh") == "yes":
            eng_score += 1

    if eng_score > api_score:
        return FAMILY_ENGINEERING, f"eng={eng_score} api={api_score}"
    if api_score > 0:
        return FAMILY_API, f"api={api_score} eng={eng_score}"
    return FAMILY_UNKNOWN, "no_signal"


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
