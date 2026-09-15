"""跨回合持久层：事实、技能、当前任务状态。

对应设计文档V2 §6.3。

这是 V2 里**唯一**允许跨回合累积的地方，也是任务书 §5.3 那句
"形成固定 SOP 或者 SKILL，实现 Agent 自进化"的落点。

三个区：

    FACTS   键值事实   —— "这个接口用 Authorization: Bearer xxx"
    SKILLS  任务族技能 —— "api-query 族的走法是 recon → query，query 里带鉴权"
    RUN     当前任务   —— 走到阶梯第几步、这份输出重复了几次、交过哪些答案

战场状态（单位位置、血量、金币）一律不进这里——那些每回合从报文重算。
判断标准：**这条信息在下一场对局里还有用吗？** 有用才进 Memory。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sandbox import SandboxOutput

# 事实的键（带命名空间，避免不同族互相覆盖）
F_FAMILY = "task.family"            # 上次识别出的任务族
F_BASE_URL = "api.base_url"         # API 根地址
F_ENDPOINT = "api.endpoint"         # 端点路径
F_AUTH_VALUE = "api.auth_value"     # 鉴权头（形如 "Authorization: Bearer xxx"）
F_PARAM = "api.param"               # 查询参数名（location / city / …）
F_FIELD_ALIAS = "api.field_alias"   # 中文字段说明 -> 记录字段名（JSON 串）
F_ERA_ORDER = "api.era_order"       # 年代排序（JSON 串）
F_TARGET = "task.target"            # 任务的目标参数（如"北京"）
F_WS_ROOT = "sandbox.ws"            # 工程修复族的工作区路径
F_SPEC_PATH = "sandbox.spec"        # 工程修复族的规格文件路径
F_CHECK_CMD = "sandbox.check"       # 工程修复族的检查命令


class RunState(str, Enum):
    """任务运行状态机的状态"""

    IDLE = "idle"            # 没有任务在身
    TRAVEL = "travel"        # 正在赶往任务点
    EXPLORE = "explore"      # 任务进行中，正在推进阶梯
    SUBMIT = "submit"        # 本轮要提交答案
    DONE = "done"            # 任务结束（成功或超时），等待下一轮调度
    ABANDON = "abandon"      # 已放弃（等开拓者离开任务点）


@dataclass(frozen=True, slots=True)
class StepSpec:
    """阶梯上的一步

    `name` 决定用哪个脚本模板（`scripts.build`），`params` 是该步骤的已知参数
    （从 FACTS 与任务文本里填）。参数是**快照**而不是引用——技能库里的步骤
    要能原样复用，不能依赖当时的 Memory 内容。
    """

    name: str
    params: tuple[tuple[str, str], ...] = ()

    def param(self, key: str, default: str = "") -> str:
        for name, value in self.params:
            if name == key:
                return value
        return default

    def with_params(self, **extra: str) -> "StepSpec":
        merged = dict(self.params)
        merged.update({k: v for k, v in extra.items() if v})
        return StepSpec(self.name, tuple(sorted(merged.items())))

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "StepSpec":
        params = raw.get("params") or {}
        return cls(str(raw.get("name") or ""), tuple(sorted(params.items())))


@dataclass(slots=True)
class Skill:
    """任务族的已验证走法

    `steps` 是**实际走通过**的步骤序列（含参数）。`wins` 是成功次数，
    `losses` 是连续失败次数——连续失败 2 次就把这条技能降级停用
    （设计文档V2 原则 P5：学习要能被证伪）。
    """

    family: str
    signature: str
    steps: list[StepSpec] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    retired: bool = False

    @property
    def usable(self) -> bool:
        return not self.retired and self.losses < 2

    def record_win(self, facts: dict[str, str]) -> None:
        self.wins += 1
        self.losses = 0
        # 只并入非空事实，避免把一次的偶然空值写成通则
        for key, value in facts.items():
            if value:
                self.facts[key] = value

    def record_loss(self) -> None:
        self.losses += 1

    def summary(self) -> str:
        steps = "→".join(s.name for s in self.steps)
        return f"{self.family}[{steps}] wins={self.wins} losses={self.losses}"


@dataclass(slots=True)
class TaskRun:
    """当前任务的运行状态"""

    key: str                      # 技能签名（`skills.signature`），同类任务共用
    family: str
    accepted_round: int
    timeout: int
    deadline: int                 # accepted_round + timeout - 安全余量
    step_index: int = 0
    attempts: int = 0             # 当前这一步已经尝试过几次
    last_output: SandboxOutput | None = None
    evidence: frozenset[str] = frozenset()
    repeats: int = 0              # 当前这一步的输出连续重复了几次
    submitted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    best_answer: str | None = None
    state: RunState = RunState.EXPLORE
    skill_used: bool = False
    used_steps: list[StepSpec] = field(default_factory=list)
    # 工程修复族：已经跑过几轮 check→repair
    repair_rounds: int = 0
    # 上一次 `[CHECKBODY]`（repair 步要用它做修复决策）
    check_output: str = ""
    # 连续"没有进展"的回合数（用于 LLM 兜底的触发）
    stalled: int = 0
    # 是否已经发过 LLM 咨询 / 是否拿到了可用的兜底命令
    llm_requested: bool = False
    llm_command: str = ""
    # 通用探测只做一次
    generic_tried: bool = False
    # 最近一次下发 `executeCmd` 的回合号（用来认领 `lastCmdResult`）
    pending_round: int = 0

    @property
    def answer(self) -> str | None:
        """目前手上最好的一份答案"""
        if self.last_output is None:
            return None
        return self.last_output.answer or self.last_output.token or self.last_output.solution

    def rounds_left(self, round_no: int) -> int:
        return self.deadline - round_no

    def remember_step(self, step: StepSpec) -> None:
        """记下这一步（用于任务成功后生成技能）

        连续同名去重：同一步骤因为超时/重试跑了两次，在技能里也只算一步——
        技能库要的是"走通的路径"，不是执行流水账。
        """
        if self.used_steps and self.used_steps[-1].name == step.name:
            return
        self.used_steps.append(step)

    def accept_step_output(self, output: SandboxOutput) -> None:
        """吸收本回合的沙盒输出，更新重复计数与证据集

        `repeats` 是**这一步**的输出连续重复了几次。它是 V2 唯一的前进判据：
        同一份输出出现第二次就说明这一步卡住了，立刻跳下一步——这是对
        V1 "读题成功却反复重读同一份文档 6 个回合"（T4）的直接对策。
        """
        repeated = (
            self.last_output is not None
            and output.fingerprint == self.last_output.fingerprint
        )
        self.repeats = self.repeats + 1 if repeated else 0
        self.evidence = self.evidence | output.evidence_keys()
        self.last_output = output
        self.attempts += 1


@dataclass(slots=True)
class Memory:
    """跨回合持久层（模块级单例见文件末尾的 MEMORY）"""

    facts: dict[str, str] = field(default_factory=dict)
    skills: dict[str, Skill] = field(default_factory=dict)
    run: TaskRun | None = None
    log: list[str] = field(default_factory=list)

    # --- 事实 ---

    def fact(self, key: str, default: str = "") -> str:
        return self.facts.get(key, default)

    def set_fact(self, key: str, value: str) -> None:
        if value:
            self.facts[key] = value

    # --- 技能（按 signature 建索引，见 `skills.signature`）---

    def skill_for(self, signature: str) -> Skill | None:
        skill = self.skills.get(signature)
        return skill if skill and skill.usable else None

    def store_skill(self, skill: Skill) -> None:
        existing = self.skills.get(skill.signature)
        if existing is None:
            self.skills[skill.signature] = skill
            self.note(f"skill+ {skill.summary()}")
            return
        existing.wins += skill.wins
        existing.losses = 0
        existing.retired = False
        if skill.steps and (not existing.steps or len(skill.steps) < len(existing.steps)):
            existing.steps = list(skill.steps)
        existing.facts.update({k: v for k, v in skill.facts.items() if v})
        self.note(f"skill~ {existing.summary()}")

    def punish_skill(self, signature: str) -> None:
        skill = self.skills.get(signature)
        if skill is None:
            return
        skill.record_loss()
        if not skill.usable:
            self.note(f"skill! {signature} 连续失败，降级回阶梯")

    # --- 运行 ---

    def start_run(self, run: TaskRun) -> None:
        self.run = run
        self.note(f"run> {run.key} family={run.family} deadline={run.deadline}")

    def end_run(self, reason: str) -> None:
        if self.run is not None:
            self.note(f"run< {self.run.key} {reason}")
        self.run = None

    @property
    def in_task(self) -> bool:
        return self.run is not None

    # --- 日志（供复盘）---

    def note(self, line: str) -> None:
        self.log.append(line)
        if len(self.log) > 200:
            del self.log[:100]

    def drain_log(self) -> list[str]:
        out = list(self.log)
        self.log.clear()
        return out


# 模块级单例：学习成果活在客户端进程里，覆盖一场比赛（上半场 + 下半场）。
MEMORY = Memory()


def reset() -> None:
    """清空学习成果（仅供测试）

    **就地清空**而不是重新构造：`TaskSolver.memory` 的默认值、测试模块里的
    `from ... import MEMORY` 都持有这个对象的引用，重新绑定模块名只会让它们
    继续指向旧对象——那会造成"上一个用例的任务状态泄漏到下一个用例"这种极难
    定位的测试污染。
    """
    MEMORY.facts.clear()
    MEMORY.skills.clear()
    MEMORY.run = None
    MEMORY.log.clear()
