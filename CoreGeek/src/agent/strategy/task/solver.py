"""任务状态机：调度、阶梯推进、反馈、终态。

对应设计文档V2 §6.3.4 – §6.3.7。

这是自进化任务链路的门面。对外只有一个函数：

    plan(world, turn) -> TaskPlan

它回答四个问题：

1. 现在该不该做任务？派谁去？（§6.5 调度）
2. 这一回合该下发什么 `executeCmd`？（§6.3.4 阶梯）
3. 手上有没有可以交的答案？该不该交？（§6.3.6 反馈 / §6.7 闸门）
4. 该不该放弃？（§6.3.7 终态）

设计上最重要的一条：**阶梯是"必然推进"的，不是"条件推进"的**。
每一回合结束前，要么拿到答案进 `SUBMIT`，要么让 `step_index` 前进或
`repeats` 增加。不存在 V1 那种"这一回合什么判断都没命中，于是把上回合的
命令原样再发一遍"的状态——那正是 6 个回合反复重读同一份文档的成因。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ... import grid
from ...protocol import (
    Pos,
    Turn,
    Unit,
    accept_task_command,
    distance,
    move_command,
    submit_answer_command,
)
from ...world import World
from . import answer as answer_mod
from . import scripts, skills
from .memory import (
    F_AUTH_VALUE,
    F_BASE_URL,
    F_CHECK_CMD,
    F_ENDPOINT,
    F_FAMILY,
    F_FIELD_ALIAS,
    F_PARAM,
    F_SPEC_PATH,
    F_TARGET,
    F_WS_ROOT,
    MEMORY,
    Memory,
    RunState,
    StepSpec,
    TaskRun,
)
from . import sandbox as sandbox_mod
from .sandbox import SandboxOutput


class Action:
    """本回合任务链路的动作"""

    IDLE = "idle"          # 没任务可做（任务点都不可接 / 开拓者阵亡）
    TRAVEL = "travel"      # 开拓者赶往任务点
    ACCEPT = "accept"      # 到达任务点，领取任务
    EXECUTE = "execute"    # 任务进行中，下发 executeCmd
    SUBMIT = "submit"      # 本回合提交答案
    ABANDON = "abandon"    # 放弃任务，开拓者离开任务点回防


@dataclass(slots=True)
class TaskPlan:
    """任务链路对外的唯一产物"""

    action: str = Action.IDLE
    pioneer_command: dict[str, Any] | None = None
    sandbox_command: str = ""
    prompt: str = ""
    note: str = ""
    submit_payload: str | None = None
    # `hold=True` 表示"开拓者应该原地不动"（任务中驻守，或在任务点旁蹲守）。
    # 编排器据此不把它交给空转兜底——否则兜底逻辑会把它拉去武器旁边站岗，
    # 而离开任务点一格内就等于放弃任务（任务书 §5）。
    hold: bool = False

    @property
    def active(self) -> bool:
        return self.action not in (Action.IDLE,)


# --- 可调常量 -------------------------------------------------------------

# 每个步骤允许尝试的回合数上限。超了就前进到下一步——**这是 V2 唯一的
# 止损机制**：不数失败形态、不看具体错误，只数"这一步花了几回合"。
STEP_MAX_ATTEMPTS: dict[str, int] = {
    "recon": 2,
    "query": 3,
    "normalize": 1,
    "check": 2,
    "repair": 2,
    "verify": 2,
    "generic": 2,
    "llm": 1,
}
DEFAULT_MAX_ATTEMPTS = 2

# 同一份沙盒输出重复出现几次就判定这一步卡死（1 = 出现第二次就跳步）
REPEAT_LIMIT = 1

# 距任务超时还剩几个回合就必须收手（留出提交与离开的时间）
DEADLINE_MARGIN = 2

# 工程修复族最多来回几轮 check→repair
ENGINEERING_MAX_ROUNDS = 3

# 连续几个回合没有任何进展才去问 LLM（任务期间 LLM 免费，但一次问答延迟一回合）
LLM_STALL_ROUNDS = 2

# 任务点冷却快结束时提前去旁边等的余量：剩余冷却 ≤ 路程 + 这个余量就出发
WAIT_MARGIN_ROUNDS = 3



# ==========================================================================
# 输出 -> 事实（学习）
# ==========================================================================

_FENCE = re.compile(r"```(?:sh|bash|shell)?\s*(.+?)```", re.S)
_CMD_HINT = re.compile(r"(?:CMD|COMMAND|命令)\s*[:：]\s*(.+)")


def learn_from_output(memory: Memory, output: SandboxOutput) -> list[str]:
    """把沙盒输出里的"可复用事实"并进记忆

    这是自进化的**入口**：`[PROFILE]` 里带着这次成功用到的地址、鉴权头、
    参数名，`[MAPPING]` 里带着"中文字段说明 -> 记录字段名"的对应关系。
    下一个同族任务起手就能直接连上，不必重新探索。
    """
    notes: list[str] = []
    pairs = {
        "PROFILE.base": F_BASE_URL,
        "PROFILE.path": F_ENDPOINT,
        "PROFILE.param": F_PARAM,
        "PROFILE.auth": F_AUTH_VALUE,
        "PROFILE.target": F_TARGET,
        "RECON.ws": F_WS_ROOT,
        "FIND.ws": F_WS_ROOT,
        "FIND.spec": F_SPEC_PATH,
        "FIND.check": F_CHECK_CMD,
    }
    for source, key in pairs.items():
        value = output.get(source)
        if value and memory.fact(key) != value:
            memory.set_fact(key, value)
            notes.append(f"{key}={value[:48]}")

    mapping = {
        key[len("MAPPING.") :]: value
        for key, value in output.kv.items()
        if key.startswith("MAPPING.")
    }
    if mapping:
        try:
            merged = json.loads(memory.fact(F_FIELD_ALIAS) or "{}")
        except ValueError:
            merged = {}
        if not isinstance(merged, dict):
            merged = {}
        before = dict(merged)
        merged.update(mapping)
        if merged != before:
            memory.set_fact(F_FIELD_ALIAS, json.dumps(merged, ensure_ascii=False))
            notes.append(f"mapping={len(merged)}")
    return notes


def extract_llm_command(llm_resp: str) -> str | None:
    """从 `llmResp` 里抽出可执行的沙盒命令

    LLM 的输出格式不可控，所以按"围栏代码块 -> CMD: 行 -> 整段"三级降级提取，
    提取结果还要过 `scripts.sanitize_llm_command`。
    """
    text = (llm_resp or "").strip()
    if not text:
        return None
    match = _FENCE.search(text)
    if match:
        return scripts.sanitize_llm_command(match.group(1).strip())
    match = _CMD_HINT.search(text)
    if match:
        return scripts.sanitize_llm_command(match.group(1).strip())
    if len(text) <= scripts.MAX_LLM_COMMAND and "\n" not in text:
        return scripts.sanitize_llm_command(text)
    return None


# ==========================================================================
# 状态机
# ==========================================================================


@dataclass(slots=True)
class TaskSolver:
    """任务链路的状态机（每回合构造一次，状态存在 `Memory` 里）"""

    memory: Memory = field(default_factory=lambda: MEMORY)
    diagnostics: list[str] = field(default_factory=list)

    # --- 对外入口 ---

    def plan(self, world: World) -> TaskPlan:
        turn = world.turn
        run = self.memory.run

        # 任务结束（提交成功 / 超时 / 开拓者死亡 / 离开任务点）
        if run is not None and not turn.phase_task:
            return self._finish_run(world, run)

        # 任务开始（phaseTask 从空变为非空）
        if run is None:
            if not turn.phase_task:
                return self._schedule(world)
            run = self._start_run(turn)

        self._observe(turn, run)

        # 硬性放弃条件（优先级最高：先保命再交卷）
        abandon_reason = self._abandon_reason(world, run)
        if abandon_reason:
            return self._abandon(world, run, abandon_reason)

        return self._drive(world, run)

    # --- 观测 ---

    def _start_run(self, turn: Turn) -> TaskRun:
        """任务开始的唯一入口：识别族、查技能、定截止回合"""
        # 分类要看沙盒输出（侦察结果能直接指出族），但 `turn.sandbox` 是协议层
        # 的原始结构，这里先解析成带标记的 `SandboxOutput`
        family, evidence = skills.classify(turn.phase_task, self._last_output(turn))
        key = skills.signature(family, turn.phase_task)
        timeout = self._timeout_of(turn)
        run = TaskRun(
            key=key,
            family=family,
            accepted_round=turn.round_no,
            timeout=timeout,
            deadline=turn.round_no + max(1, timeout - DEADLINE_MARGIN),
        )
        self.memory.set_fact(F_FAMILY, family)
        self.memory.start_run(run)
        self.diagnostics.append(f"family={family}({evidence}) skill={key}")
        return run

    @staticmethod
    def _last_output(turn: Turn) -> SandboxOutput | None:
        """上一回合的沙盒输出（解析成标记级结构）；没有则返回 None"""
        if not turn.last_cmd_result.strip():
            return None
        return sandbox_mod.parse_output(turn.last_cmd_result)

    @staticmethod
    def _timeout_of(turn: Turn) -> int:
        """当前任务的超时回合数（取最近的一个任务点声明值）"""
        if turn.player_tasks:
            return min(t.timeout_rounds for t in turn.player_tasks)
        return 15

    def _observe(self, turn: Turn, run: TaskRun) -> None:
        """把上一回合的沙盒输出与 LLM 回复吸收进运行状态

        只认领"上一回合确实是我们下发的那条命令"的结果（`pending_round`），
        避免把上一个任务残留的 `lastCmdResult` 当成这一步的产出——那会让
        指纹比对得出"重复"的错误结论，进而把一个还有救的步骤跳过去。
        """
        if run.pending_round == turn.round_no - 1:
            output = sandbox_mod.parse_output(turn.last_cmd_result)
            run.accept_step_output(output)
            learned = learn_from_output(self.memory, output)
            if learned:
                self.diagnostics.append("learn:" + ",".join(learned[:3]))
            if output.payloads.get("CHECKBODY"):
                run.check_output = output.payloads["CHECKBODY"]

        if turn.llm_resp:
            command = extract_llm_command(turn.llm_resp)
            if command:
                run.llm_command = command
                self.diagnostics.append("llm:command")

    # --- 调度：该不该做任务 ---

    def _schedule(self, world: World) -> TaskPlan:
        turn = world.turn
        pioneers = turn.pioneers()
        if not pioneers:
            return TaskPlan(Action.IDLE, note="no_pioneer")

        # 任务书 §5.3 + 设计文档 §6.5：**只要存在 isValid 的任务点，就要有人在做**。
        # V1 因为"接完一个再说"导致任务 2 整场未接（T5），这里把这条写死成原则。
        candidates = [t for t in turn.player_tasks if t.ready]
        if not candidates:
            # 任务点在冷却时，**快到点**才提前去旁边等——省掉"回基地再折返"
            # 的来回（V1 的 `_task_wait_logic` 就是这么做的），但也不能一整段
            # 冷却都干等：那会让开拓者连续二三十个回合空转。
            pioneer_now = pioneers[0]
            cooling = sorted(
                (t for t in turn.player_tasks if t.cooldown > 0),
                key=lambda t: (t.cooldown, distance(pioneer_now.pos, t.pos)),
            )
            if not cooling:
                return TaskPlan(Action.IDLE, note="no_task_point")
            soonest = cooling[0]
            travel = distance(pioneer_now.pos, soonest.pos)
            if soonest.cooldown <= travel + WAIT_MARGIN_ROUNDS:
                return self._wait_at(world, pioneer_now, soonest.pos)
            return TaskPlan(Action.IDLE, note="cooling")

        pioneer = pioneers[0]
        # 夜晚且尚未接上任务时先守夜：开拓者是三个操控者之一，为了赶路少操控
        # 一座武器，换来的只是"早到几个回合"。白天有 70 个回合，路一定赶得上。
        if not turn.is_day and turn.alive_robots():
            return TaskPlan(Action.IDLE, note="night_defend")

        target = min(
            candidates,
            key=lambda t: (distance(pioneer.pos, t.pos), -t.score_reward, t.pos.x, t.pos.y),
        )
        cells = self._task_stand_cells(turn, target.pos, pioneer)
        if not cells:
            return TaskPlan(Action.IDLE, note="task_unreachable")

        if any(distance(pioneer.pos, cell) <= 1 for cell in cells) or (
            distance(pioneer.pos, target.pos) <= 1
        ):
            return TaskPlan(
                Action.ACCEPT,
                pioneer_command=accept_task_command(),
                note=f"accept@{target.pos.x},{target.pos.y}",
            )

        step = grid.step_toward_any(turn, pioneer, tuple(cells))
        if step is None:
            return TaskPlan(Action.IDLE, note="task_no_path")
        return TaskPlan(
            Action.TRAVEL,
            pioneer_command=move_command(step),
            note=f"travel@{target.pos.x},{target.pos.y}",
        )

    def _wait_at(self, world: World, pioneer: Unit, point: Pos) -> TaskPlan:
        """任务点冷却中，提前站到它旁边等开放"""
        turn = world.turn
        cells = self._task_stand_cells(turn, point, pioneer)
        if not cells:
            return TaskPlan(Action.IDLE, note="wait_unreachable")
        if any(distance(pioneer.pos, cell) <= 1 for cell in cells):
            return TaskPlan(Action.IDLE, note="waiting", hold=True)
        step = grid.step_toward_any(turn, pioneer, tuple(cells))
        if step is None:
            return TaskPlan(Action.IDLE, note="wait_no_path")
        return TaskPlan(
            Action.TRAVEL,
            pioneer_command=move_command(step),
            note=f"wait@{point.x},{point.y}",
        )

    # --- 驱动：任务进行中 ---

    def _drive(self, world: World, run: TaskRun) -> TaskPlan:
        turn = world.turn

        # 1) 手上有答案 -> 立刻提交（这是唯一能让状态机进入终态的"好"路径）
        rewound = False
        candidate = self._candidate(run)
        if candidate:
            answer, reason = answer_mod.gate(
                candidate,
                turn.phase_task,
                submitted=run.submitted,
                rejected=run.rejected,
            )
            if answer:
                return self._submit(world, run, answer)
            if reason not in ("already_submitted", "already_rejected"):
                self.diagnostics.append(f"gate:{reason}")
            if reason == "already_rejected":
                # 交过且被判错：必须重取数据，不能重交同一份（V1 PK592108 R15-17）。
                # 回退之后**跳过本回合的 `_advance`**——否则上一份被判错的答案
                # 与这一份逐字相同，`repeats` 会立刻把刚退回去的那一步再次推走。
                run.state = RunState.EXPLORE
                run.step_index = self._rewind_to_query(run, turn)
                run.attempts = 0
                run.repeats = 0
                rewound = True

        # 2) 上次超时 -> 下一步命令压缩预算重试一次
        slow = run.last_output is not None and run.last_output.status == "timeout"

        # 3) 阶梯推进
        if not rewound:
            self._advance(run, turn, slow=slow)

        # 4) 走完了 -> 问 LLM 兜底（任务期间免费且不限次）
        step = self._current_step(run)
        if step is None:
            return self._escalate(world, run)

        return self._execute(world, run, step)

    # --- 阶梯 ---

    def _steps(self, run: TaskRun) -> tuple[StepSpec, ...]:
        skill = self.memory.skill_for(run.key)
        run.skill_used = skill is not None
        return skills.first_steps(run, skill)

    def _current_step(self, run: TaskRun) -> StepSpec | None:
        steps = self._steps(run)
        if run.step_index >= len(steps):
            return None
        step = steps[run.step_index]
        # 新任务的第一回合还不知道子目录长什么样，`ws`/`spec` 这些参数是空的；
        # 每次取用时用当前事实补一次（技能库里的旧参数只作为兜底）
        return step.with_params(
            ws=self.memory.fact(F_WS_ROOT),
            spec=self.memory.fact(F_SPEC_PATH),
            check=self.memory.fact(F_CHECK_CMD),
        )

    def _advance(self, run: TaskRun, turn: Turn, *, slow: bool) -> None:
        """推进阶梯

        这是"反馈"环节的全部逻辑，也是 V1 最大的漏洞所在。判据只有三条，
        不再区分"读文件死循环/取数连败/命令被掐断/空结果集"那四五种形态：

            repeats >= REPEAT_LIMIT  同一份输出又来了一遍 -> 这一步卡死，跳步
            attempts >= 上限         这一步花的回合数超了     -> 跳步
            slow                     上次被掐断             -> 允许再试一次
        """
        step = self._current_step(run)
        if step is None:
            return

        if slow:
            self.diagnostics.append("slow")
            if run.attempts >= 2:
                self._step_forward(run, turn, "slow")
            return

        if run.repeats >= REPEAT_LIMIT:
            self._step_forward(run, turn, "repeat")
            return

        cap = STEP_MAX_ATTEMPTS.get(step.name, DEFAULT_MAX_ATTEMPTS)
        if run.attempts >= cap:
            self._step_forward(run, turn, "capped")
            return

        if self._step_satisfied(step, run):
            self._step_forward(run, turn, "ok")

    @staticmethod
    def _step_satisfied(step: StepSpec, run: TaskRun) -> bool:
        """这一步是否已经拿到了它该拿到的东西"""
        output = run.last_output
        if output is None:
            return False
        if step.name == "recon":
            # recon 的职责只是"看看沙盒里有什么"。执行器跑完并留下了结构化标记
            # 就算交差——`query` 自己会重新定位任务文件与文档（脚本里有完整的
            # 兜底搜索），所以这里**不需要**等到 `[RECON]` 里带上 root。
            # 这一条很关键：为了等一个完美的 recon 结果而多耗两个回合，
            # 在只有 15 回合的任务预算里是纯粹的浪费。
            return output.has_done and bool(output.tags)
        if step.name == "query":
            # query 只有在产出答案时才算完成；取到数但没聚合出答案还要再试
            return output.answer is not None
        if step.name == "normalize":
            return output.has("FIX")
        if step.name == "check":
            return output.has("CHECK")
        if step.name == "repair":
            return output.has("FIX")
        if step.name == "verify":
            return output.token is not None
        if step.name == "generic":
            return output.token is not None or output.has("DOCPATH")
        if step.name == "llm":
            return True
        return True

    def _step_forward(self, run: TaskRun, turn: Turn, reason: str) -> None:
        step = self._current_step(run)
        run.step_index += 1
        run.attempts = 0
        run.repeats = 0
        name = step.name if step else "-"
        self.diagnostics.append(f"step+ {name}({reason})")

        # 工程修复族：verify 走完还没拿到 TOKEN 就回到 check 再来一轮
        if (
            step is not None
            and step.name == "verify"
            and run.family == skills.FAMILY_ENGINEERING
            and run.repair_rounds < ENGINEERING_MAX_ROUNDS
            and (run.last_output is None or run.last_output.token is None)
        ):
            run.repair_rounds += 1
            order = list(skills.ladder(run.family))
            if "check" in order:
                run.step_index = order.index("check")
                self.diagnostics.append(f"rewind check#{run.repair_rounds}")

    def _rewind_to_query(self, run: TaskRun, turn: Turn) -> int:
        """答案被打回时回到取数那一步（而不是重交同一个答案）"""
        order = list(skills.ladder(run.family))
        for name in ("query", "verify", "check", "generic"):
            if name in order:
                return order.index(name)
        return 0

    # --- 执行 ---

    def _execute(self, world: World, run: TaskRun, step: StepSpec) -> TaskPlan:
        run.remember_step(step)
        run.pending_round = world.turn.round_no
        run.state = RunState.EXPLORE
        command = scripts.build(
            step,
            phase_task=world.turn.phase_task,
            facts=self.memory.facts,
            check_output=run.check_output,
        )
        note = f"step={step.name} idx={run.step_index} r={run.attempts}"
        return TaskPlan(
            Action.EXECUTE,
            pioneer_command=self._hold_position(world, run),
            sandbox_command=command,
            prompt=self._maybe_prompt(run, world),
            note=note,
            hold=True,
        )

    def _hold_position(self, world: World, run: TaskRun) -> dict[str, Any] | None:
        """任务进行中，开拓者**不许离开任务点周围一格**

        任务书 §5 明确：离开己方任务点周围一格内即视为任务结束。所以这里
        只在真的走远了的时候把它拉回来；正常情况下返回 None（不出指令），
        把角色指令槽让给别的用途也不影响——`executeCmd` 与角色动作是
        响应报文里两个互不相干的字段（接口文档 §2.1）。
        """
        turn = world.turn
        pioneer = self._pioneer(turn)
        if pioneer is None:
            return None
        points = [t.pos for t in turn.player_tasks]
        if not points:
            return None
        nearest = min(points, key=lambda p: distance(pioneer.pos, p))
        if distance(pioneer.pos, nearest) <= 1:
            return None
        cells = self._task_stand_cells(turn, nearest, pioneer)
        step = grid.step_toward_any(turn, pioneer, tuple(cells))
        return move_command(step) if step else None

    # --- 提交 ---

    def _submit(self, world: World, run: TaskRun, answer: str) -> TaskPlan:
        pioneer = self._pioneer(world.turn)
        command = submit_answer_command(answer)
        if pioneer is None or command is None:
            return TaskPlan(Action.IDLE, note="submit_no_pioneer")
        run.submitted.append(answer)
        run.best_answer = answer
        run.state = RunState.SUBMIT
        # 交完不重置阶梯：判题器可能打回（errorCode=2），那时按 `_rewind_to_query`
        # 回到取数步；在此之前保持现状，避免把已经拿到的答案丢掉。
        run.repeats = 0
        self.diagnostics.append(f"submit#{len(run.submitted)}")
        return TaskPlan(
            Action.SUBMIT,
            pioneer_command=command,
            note=f"submit len={len(answer)}",
            submit_payload=answer,
        )

    # --- 放弃 ---

    def _abandon_reason(self, world: World, run: TaskRun) -> str:
        turn = world.turn
        if turn.round_no >= run.deadline:
            return "deadline"
        pioneer = self._pioneer(turn)
        if pioneer is None:
            return "pioneer_dead"
        if not turn.player_tasks:
            return "no_task_point"
        # 基地告急 + 已经交过答案（部分分已经到手）-> 回防
        station = turn.station()
        if station is not None and run.submitted:
            if station.health_ratio() < BASE_DANGER_RATIO:
                if self._needs_hands(world):
                    return "base_danger"
        return ""

    def _needs_hands(self, world: World) -> bool:
        """夜晚有武器没人操控，说明确实缺人手"""
        turn = world.turn
        if turn.is_day:
            return False
        return len(turn.characters()) <= len(turn.towers())

    def _abandon(self, world: World, run: TaskRun, reason: str) -> TaskPlan:
        """放弃任务：**先提交手上的答案，再离开**

        任务书 §6 规定部分完成按通过率给分，"交一份对了一半的答案"严格优于
        "不交"。V1 有整整一场对局（PK592172）接任务后 9 个回合一次都没提交，
        最后按 0 分计——放弃路径上也必须提交，这是硬规则。
        """
        self.diagnostics.append(f"abandon:{reason}")
        candidate = self._candidate(run)
        answer = None
        if candidate:
            answer, why = answer_mod.gate(
                candidate,
                world.turn.phase_task,
                submitted=run.submitted,
                rejected=run.rejected,
            )
            if answer is None and why != "already_submitted":
                self.diagnostics.append(f"abandon_gate:{why}")
        if answer:
            return self._submit(world, run, answer)

        self._close_run(run, reason)
        pioneer = self._pioneer(world.turn)
        command = None
        if pioneer is not None:
            station = world.turn.station()
            goal = station.pos if station else Pos(world.turn.width // 2, world.turn.height // 2)
            step = grid.step_toward_any(world.turn, pioneer, tuple(
                grid.cells_in_radius(goal, 2, world.turn.width, world.turn.height)
            ))
            command = move_command(step) if step else None
        return TaskPlan(Action.ABANDON, pioneer_command=command, note=f"abandon:{reason}")

    # --- 收尾 ---

    def _finish_run(self, world: World, run: TaskRun) -> TaskPlan:
        """任务结束（`phaseTask` 空了）：结算并学习"""
        turn = world.turn
        rejected = turn.answer_wrong
        if run.submitted and not rejected:
            skill = skills.build_skill(run, self.memory.facts)
            if skill is not None:
                self.memory.store_skill(skill)
                self.diagnostics.append(f"skill:saved({len(skill.steps)}steps)")
        elif run.skill_used:
            self.memory.punish_skill(run.key)
        self._close_run(run, "closed" + ("_rejected" if rejected else "_ok"))
        return TaskPlan(
            Action.IDLE,
            pioneer_command=self._return_to_base(world),
            note="task_closed",
        )

    def _close_run(self, run: TaskRun, reason: str) -> None:
        self.memory.end_run(reason)
        run.state = RunState.DONE

    def _return_to_base(self, world: World) -> dict[str, Any] | None:
        """任务结束后开拓者回防（回到基地附近）"""
        turn = world.turn
        pioneer = self._pioneer(turn)
        station = turn.station()
        if pioneer is None or station is None:
            return None
        if distance(pioneer.pos, station.pos) <= 3:
            return None
        cells = tuple(
            grid.cells_in_radius(station.pos, 2, turn.width, turn.height)
        )
        step = grid.step_toward_any(turn, pioneer, cells)
        return move_command(step) if step else None

    # --- LLM 兜底 ---

    def _escalate(self, world: World, run: TaskRun) -> TaskPlan:
        """阶梯走完还没有答案：先用 LLM 给的那条命令，再退到通用探测

        任务期间 LLM 调用**不计入每日限额**（接口文档 §1.7 的注），所以这条
        兜底路径是免费的。它只在确定性路径全部失败后才启用，且拿到的命令要过
        `sanitize_llm_command`（见设计文档V2 §6.8）。
        """
        turn = world.turn
        if run.llm_command:
            command = run.llm_command
            run.llm_command = ""
            run.pending_round = turn.round_no
            self.diagnostics.append("llm:exec")
            return TaskPlan(
                Action.EXECUTE,
                pioneer_command=self._hold_position(world, run),
                sandbox_command=command,
                note="step=llm",
                hold=True,
            )
        if not run.llm_requested:
            run.llm_requested = True
            run.stalled += 1
            if run.stalled >= LLM_STALL_ROUNDS:
                return TaskPlan(
                    Action.EXECUTE,
                    pioneer_command=self._hold_position(world, run),
                    prompt=self._prompt_body(world, run),
                    note="llm:ask",
                    hold=True,
                )

        # 通用探测只做一次，之后放弃
        if not run.generic_tried:
            run.generic_tried = True
            step = StepSpec("generic").with_params(
                ws=self.memory.fact(F_WS_ROOT),
                spec=self.memory.fact(F_SPEC_PATH),
                check=self.memory.fact(F_CHECK_CMD),
            )
            run.pending_round = turn.round_no
            run.remember_step(step)
            command = scripts.build(
                step, phase_task=turn.phase_task, facts=self.memory.facts
            )
            return TaskPlan(
                Action.EXECUTE,
                pioneer_command=self._hold_position(world, run),
                sandbox_command=command,
                note="step=generic",
                hold=True,
            )

        return self._abandon(world, run, "ladder_exhausted")

    def _maybe_prompt(self, run: TaskRun, world: World) -> str:
        """任务期间每 K 回合发一次 LLM 咨询（免费额度，提前换取兜底素材）"""
        if run.llm_requested:
            return ""
        if run.attempts == 0 and run.step_index == 0:
            run.llm_requested = True
            return self._prompt_body(world, run)
        return ""

    def _prompt_body(self, world: World, run: TaskRun) -> str:
        """给 LLM 的 prompt：只发任务原文 + 沙盒标记摘要，不发全文"""
        turn = world.turn
        markers = ""
        if run.last_output is not None:
            markers = " ".join(run.last_output.tags[:12])
        return (
            "你在一个没有外网的 Linux 沙盒里执行自进化任务。\n"
            "=== 任务原文 ===\n"
            f"{turn.phase_task[:1500]}\n"
            "=== 上一回合沙盒标记 ===\n"
            f"{markers}\n"
            "=== 已知接口事实 ===\n"
            f"{json.dumps(dict(list(self.memory.facts.items())[:12]), ensure_ascii=False)}\n"
            "请只输出一条可直接在 sh 中执行的命令（用 ```sh 代码块包裹），"
            "不要解释。命令必须把该做的事一次做完。"
        )

    # --- 工具 ---

    @staticmethod
    def _pioneer(turn: Turn) -> Unit | None:
        pioneers = turn.pioneers()
        return pioneers[0] if pioneers else None

    @staticmethod
    def _candidate(run: TaskRun) -> str | None:
        if run.last_output is None:
            return None
        return (
            run.last_output.answer
            or run.last_output.token
            or run.last_output.solution
        )

    @staticmethod
    def _task_stand_cells(
        turn: Turn, point: Pos, mover: Unit | None = None
    ) -> tuple[Pos, ...]:
        """任务点周围一格内可站立、且当前空闲的格子

        任务点 2 占两格（任务书 §4.6.2），开拓者站到任一格旁边都算在场。
        被队友占着的格子不算——走不到，而且开拓者站在**别人**占的格上本来就
        不成立。
        """
        # 任务点 2 占两格：找与它相邻的同类型任务点格
        points = [point]
        for zone_pos, zone in turn.zones.items():
            if zone in ("challengerTaskPoint2", "defenderTaskPoint2") and (
                distance(zone_pos, point) <= 1
            ):
                points.append(zone_pos)
        occupants = turn.occupied()
        if mover is not None:
            occupants = occupants - {mover.pos}
        cells: list[Pos] = []
        for spot in points:
            for cell in grid.stand_cells(turn, spot, occupants=occupants):
                if cell not in cells:
                    cells.append(cell)
        return tuple(cells)


def plan(world: World) -> TaskPlan:
    """模块级入口（每次构造 solver 的代价可以忽略）"""
    return TaskSolver().plan(world)
