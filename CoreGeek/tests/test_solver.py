"""自进化任务状态机的回归用例。

对应设计文档V2 §10.2。

这里的每一个用例都对应 GitHub Issues #1–#199 里**反复出现**的一类故障。
用例名里的编号是设计文档 §1.1 的故障编号。

跑这套用例的意义：V1 的每一轮自动修复都在"改一处、坏两处"之间打转，
因为它们只有对战复盘这种事后证据。这些用例把复盘结论固化成**可执行的断言**，
让同一个故障不能改回来。
"""

from __future__ import annotations

import json

import pytest

from conftest import (
    AUTH_FAIL_OUTPUT,
    BEIJING_TASK_TEXT,
    SUCCESS_OUTPUT,
    V1_DEAD_LOOP_OUTPUT,
)
from agent.brain import decide
from agent.protocol import Turn
from agent.strategy.task import answer as answer_mod
from agent.strategy.task import sandbox as sandbox_mod
from agent.strategy.task import skills
from agent.strategy.task.memory import MEMORY, Memory, RunState
from agent.strategy.task.solver import Action, TaskSolver
from agent.world import World


def _world(payload_factory, **kwargs):
    turn = Turn.load(payload_factory(**kwargs))
    return World.load(turn)


def _task_point(task_factory, x=24, y=12):
    return [task_factory("自进化类1", x, y)]


# ==========================================================================
# T1：读题成功但从不调 API（`api=0` 死循环）
# ==========================================================================


def test_recon_advances_to_query_not_repeat(payload_factory, base_roles, task_factory):
    """recon 拿到 `[RECON]` 后，下一回合必须切到 query（不再重复读题）

    故障：PK591684 R11–R13、PK591772 R12–R13、PK592172 R12–R16，
    沙盒读题成功（`docs=2`、`key=yes`、`exitCode:0`）却 `api=0`，
    整整 6 个回合反复重读同一份文档。
    """
    solver = TaskSolver()
    # 第 1 回合：接到任务，下发 recon
    plan = solver.plan(_world(
        payload_factory,
        round_no=10,
        roles=base_roles,
        player_tasks=_task_point(task_factory),
        phase_task=BEIJING_TASK_TEXT,
    ))
    assert plan.action == Action.EXECUTE
    assert plan.note.startswith("step=recon")

    # 第 2 回合：recon 回来了（有 [RECON]）——必须前进到 query
    recon_output = (
        "[exitCode:0]\n"
        "[RECON] root=/tmp/selfEvolutionTask/t1 task=/t1/task_1_beijing.md "
        "docs=2 ws=0 scripts=0 py=1\n"
        "[SCAN] files=9 dirs=2 py=yes sh=no timed_out=no\n"
        "[DONE] step=recon elapsed=1.0s\n"
    )
    plan = solver.plan(_world(
        payload_factory,
        round_no=11,
        roles=base_roles,
        player_tasks=_task_point(task_factory),
        phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=recon_output,
    ))
    assert plan.action == Action.EXECUTE
    assert plan.note.startswith("step=query"), plan.note
    assert "urllib" in plan.sandbox_command  # query 脚本里真的会发 HTTP 请求


def test_repeated_output_advances_step(payload_factory, base_roles, task_factory):
    """同一份沙盒输出连续出现 2 次 ⇒ 必须换一步（不再原样重发）

    故障 T4：沙盒反复重读同一份文档，输出逐字节相同，`watch r1/t2→t6`。
    V2 的判据只有一条：输出指纹相同 ⇒ 这一步卡死 ⇒ 跳步。
    """
    solver = TaskSolver()
    common = dict(
        roles=base_roles,
        player_tasks=_task_point(task_factory),
        phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=V1_DEAD_LOOP_OUTPUT,
    )
    first = solver.plan(_world(payload_factory, round_no=20, **common))
    assert first.note.startswith("step=recon")

    # 第二回合下发的内容必须与第一回合**不同**（因为 recon 已经"完成"了）
    second = solver.plan(_world(payload_factory, round_no=21, **common))
    assert second.sandbox_command != first.sandbox_command
    assert second.note.startswith("step=query"), second.note


def test_query_attempt_cap_then_escalate(payload_factory, base_roles, task_factory):
    """query 连续拿不到答案时，有限次后必须换策略，不能无限 query"""
    solver = TaskSolver()
    notes = []
    for round_no in range(30, 38):
        plan = solver.plan(_world(
            payload_factory,
            round_no=round_no,
            roles=base_roles,
            player_tasks=_task_point(task_factory),
            phase_task=BEIJING_TASK_TEXT,
            last_cmd_result=AUTH_FAIL_OUTPUT,
        ))
        notes.append(plan.note)
    # 一定出现过 step+ 的推进（不再一直停在 query）
    assert any("generic" in note or "llm" in note or "abandon" in note for note in notes), notes


def test_reclassify_switches_to_engineering_after_recon(
    payload_factory, base_roles, task_factory
):
    """对称回归：工程修复族的短 phaseTask 也要在侦察后切到 normalize

    真实对局 R23–R26：`phaseTask` 是"请阅读task_1_alpha.md，获取任务信息"，
    侦察回 `2-engineering-fix` + `ws_1` + `spec.md` + `bad interpreter: /bin/sh^M`，
    却仍在 `unknown` 阶梯上打转、5 个回合后放弃。
    """
    bare_phase = "请阅读task_1_alpha.md，获取任务信息"
    recon_output = (
        "[exitCode:0]\n"
        "[RECON] docs=1 py=0 root=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix scripts=2 task=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/task_1_alpha.md ws=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1\n"
        "[SCAN] dirs=4 files=6 py=no sh=yes timed_out=no\n"
        "[DOCBODY]\n"
        "# 应用 alpha 部署规范\n"
        "[WS] path=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1\n"
        "[SCRIPTS] check start.sh\n"
        "[DONE] step=recon elapsed=0.00s\n"
    )
    solver = TaskSolver()
    solver.plan(_world(
        payload_factory, round_no=20, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=bare_phase,
    ))
    assert MEMORY.run.family == skills.FAMILY_UNKNOWN

    plan = solver.plan(_world(
        payload_factory, round_no=21, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=bare_phase,
        last_cmd_result=recon_output,
    ))
    assert MEMORY.run.family == skills.FAMILY_ENGINEERING, MEMORY.run.family
    assert plan.note.startswith("step=normalize"), plan.note
    # normalize 步要真的去修 CRLF（实测报的就是 bad interpreter: /bin/sh^M）
    assert "crlf" in plan.sandbox_command or "\r" in plan.sandbox_command

def test_replay_real_task_sequence_reaches_submit(
    payload_factory, role_factory, task_factory
):
    """按真实对局的顺序回放：R12 侦察 → R13 判族并 query → R14 提交

    用的是 2026-09-15 02:04 那份日志里的真实文本（`phaseTask` 27 字节、
    侦察输出带 `API_DOCS.md` 与 `X-API-Key`）。修之前这条链路是
    recon → generic → 放弃（`event=abandon detail=ladder_exhausted`），整场 0 分。
    """
    roles = [role_factory(10013, "station", 30, 9),
             role_factory(10011, "pioneer", 22, 13)]
    tasks = [task_factory("自进化类1", 23, 14)]
    bare = "请阅读task_1_beijing.md，获取任务信息"
    common = dict(roles=roles, player_tasks=tasks, phase_task=bare)

    first = decide(payload_factory(round_no=12, **common))
    assert first["executeCmd"]

    recon = (
        "[exitCode:0]\n"
        "[RECON] docs=1 py=0 root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api task=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md\n"
        "[DOCBODY]\n"
        "**基础URL**: `http://localhost:8899`\n"
        "X-API-Key: heritage-api-key-2024\n"
        "[DONE] step=recon\n"
    )
    second = decide(payload_factory(round_no=13, last_cmd_result=recon, **common))
    assert "urllib" in second["executeCmd"]  # 真的去调接口，不是再读一遍文档

    answer = (
        "[exitCode:0]\n"
        "[DATA] n=15 total=15 pages=1\n"
        '[ANSWER] {"city":"北京","total_count":15,"world_heritage_count":7,'
        '"types":["古建筑","古遗址"],"oldest_era":"周口店遗址"}\n'
        "[DONE] step=query\n"
    )
    third = decide(payload_factory(round_no=14, last_cmd_result=answer, **common))
    command = third["roleCommandMap"]["10011"]
    assert command["action"] == "submitAnswer", command
    assert json.loads(command["taskAnswer"])["total_count"] == 15


def test_reclassify_switches_to_query_after_recon(
    payload_factory, base_roles, task_factory
):
    """实测回归：`phaseTask` 没有关键词时，侦察回来必须重新判族并切到 query

    真实对局里 `phaseTask` 只有"请阅读task_1_beijing.md，获取任务信息"这一句
    （27 字节、零关键词），分类成 `unknown`，阶梯是 recon→generic，
    **`query` 那一步从头到尾没走过**，任务 5 个回合就被放弃、整场 0 分。
    """
    bare_phase = "请阅读task_1_beijing.md，获取任务信息"
    recon_output = (
        "[exitCode:0]\n"
        "[RECON] docs=1 py=0 root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api task=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md\n"
        "[SCAN] dirs=0 files=2 py=no sh=no timed_out=no\n"
        "[DOCBODY]\n"
        "**基础URL**: `http://localhost:8899`\n"
        "X-API-Key: heritage-api-key-2024\n"
        "[DONE] step=recon elapsed=0.00s\n"
    )
    solver = TaskSolver()
    # 第 1 回合：只有短 phaseTask ⇒ 判不出族，走 recon
    first = solver.plan(_world(
        payload_factory, round_no=10, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=bare_phase,
    ))
    assert first.note.startswith("step=recon")
    assert MEMORY.run.family == skills.FAMILY_UNKNOWN

    # 第 2 回合：侦察回来了 ⇒ 必须重判为 api-query，并直接进入 query
    second = solver.plan(_world(
        payload_factory, round_no=11, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=bare_phase,
        last_cmd_result=recon_output,
    ))
    assert MEMORY.run.family == skills.FAMILY_API, MEMORY.run.family
    assert second.note.startswith("step=query"), second.note
    assert "urllib" in second.sandbox_command
    # 换族要留痕，复盘才看得出发生过什么
    assert any("event=family" in e for e in second.events), second.events

# ==========================================================================
# T2：接取任务后从未提交（0 分）
# ==========================================================================


def test_success_output_triggers_submit(payload_factory, base_roles, task_factory):
    solver = TaskSolver()
    # 先起任务
    solver.plan(_world(
        payload_factory, round_no=40, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
    ))
    plan = solver.plan(_world(
        payload_factory, round_no=41, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    ))
    assert plan.action == Action.SUBMIT, plan.note
    assert plan.pioneer_command is not None
    assert plan.pioneer_command["action"] == "submitAnswer"
    payload = json.loads(plan.pioneer_command["taskAnswer"])
    assert payload["total_count"] == 137
    assert payload["world_heritage_count"] == 12


def test_abandon_still_submits(payload_factory, base_roles, task_factory):
    """放弃任务时，手上只要有答案就必须先提交（部分分 > 0 分）

    故障 T2：PK592172 接任务后 9 个回合一次都没提交，任务分归零。
    任务书 §6 规定部分完成按通过率给分，"交一份对了一半的答案"严格优于"不交"。
    """
    solver = TaskSolver()
    # 接任务（设成很早的回合，让 deadline 落在后面）
    solver.plan(_world(
        payload_factory, round_no=50, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
    ))
    # 拿到答案
    solver.plan(_world(
        payload_factory, round_no=51, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    ))
    # 但答案被拒绝了 —— 换一份"部分正确"的答案，越过 deadline 时必须提交
    run = MEMORY.run
    assert run is not None
    run.rejected.clear()
    run.submitted.clear()
    run.deadline = 52  # 下一回合就到截止
    plan = solver.plan(_world(
        payload_factory, round_no=52, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    ))
    # deadline 命中放弃分支，但放弃前必须先提交手上的答案
    assert plan.pioneer_command is not None
    assert plan.pioneer_command["action"] == "submitAnswer"


def test_rejected_answer_is_not_resubmitted(payload_factory, base_roles, task_factory):
    """被判错的答案不许重复提交，必须回到取数步骤

    故障：PK592108 的 R15–R17 —— 执行器解出一次答案之后再没跑过，
    任务线只剩"复读同一个答案直到提交额度烧完"这一种结局。
    """
    solver = TaskSolver()
    solver.plan(_world(
        payload_factory, round_no=60, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
    ))
    ok = _world(
        payload_factory, round_no=61, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    )
    submitted = solver.plan(ok).submit_payload
    assert submitted

    # 判题器打回：errorCode=2，任务仍在进行
    run = MEMORY.run
    run.rejected.append(submitted)
    run.submitted.remove(submitted)
    plan = solver.plan(_world(
        payload_factory, round_no=62, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
        errors=[{"errorCode": 2, "description": "答案错误"}],
    ))
    assert plan.action != Action.SUBMIT or plan.submit_payload != submitted
    assert run.step_index <= 1  # 回到取数步骤


# ==========================================================================
# T3：缺鉴权头吃 401
# ==========================================================================


def test_auth_failure_leads_to_retry_with_auth(payload_factory, base_roles, task_factory):
    """401 之后的下一条命令仍要带鉴权尝试（query 的候选矩阵里有鉴权变体）"""
    solver = TaskSolver()
    solver.plan(_world(
        payload_factory, round_no=70, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
    ))
    plan = solver.plan(_world(
        payload_factory, round_no=71, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=AUTH_FAIL_OUTPUT,
    ))
    assert plan.sandbox_command
    # query 脚本内建了鉴权候选（Authorization / X-API-Key / 无头），一定会在同一条
    # 命令里把这些都试一遍，而不是只发一次裸请求
    assert "Authorization" in plan.sandbox_command
    assert "X-API-Key" in plan.sandbox_command


# ==========================================================================
# T5：任务 2 全程可接未接
# ==========================================================================


def test_task_two_is_accepted_after_first(
    payload_factory, base_roles, task_factory
):
    """任务 1 结束后，任务 2 可接 ⇒ 开拓者必须去接

    故障 T5：PK592172 / PK592173 里任务 2 全程 `isValid` 却从未接取。
    """
    solver = TaskSolver()
    tasks = [task_factory("自进化类2", 24, 12)]
    # 任务 1 刚结束（phaseTask 空），开拓者就在任务点旁边
    plan = solver.plan(_world(
        payload_factory,
        round_no=80,
        roles=[
            {"id": 10013, "pos": {"x": 20, "y": 10}, "roleType": "station",
             "health": 1500, "attackPower": 0, "attackRange": 0, "level": 1,
             "backPackCapability": 0, "backpack": []},
            {"id": 10011, "pos": {"x": 24, "y": 13}, "roleType": "pioneer",
             "health": 200, "attackPower": 0, "attackRange": 0,
             "backPackCapability": 40, "backpack": []},
        ],
        player_tasks=tasks,
    ))
    assert plan.action == Action.ACCEPT


def test_accept_only_when_really_adjacent(payload_factory, role_factory, task_factory):
    """离任务点 **2 格**时不能发 acceptTask（判题器会判非法，且开拓者会原地卡死）

    回归：真实对局里 `20011:acceptTask` 连续 7 个回合被拒（`fail=[20011:acceptTask]`），
    开拓者停在 (24,12)、任务点在 (23,14)，距离 2。根因是判据写成
    "离**落脚点**一格内"——落脚点是任务点的邻居，于是距离 2 也被当成到位；
    而 ACCEPT 分支不发移动指令，开拓者就一直站着重复接任务。
    """
    solver = TaskSolver()
    plan = solver.plan(_world(
        payload_factory,
        round_no=5,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10011, "pioneer", 24, 12)],
        player_tasks=[task_factory("自进化类1", 23, 14)],
    ))
    assert plan.action == Action.TRAVEL, plan.note
    assert plan.pioneer_command["action"] == "move"


def test_accept_when_adjacent(payload_factory, role_factory, task_factory):
    """站在任务点旁边才是接受任务的时候"""
    solver = TaskSolver()
    plan = solver.plan(_world(
        payload_factory,
        round_no=5,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10011, "pioneer", 23, 13)],
        player_tasks=[task_factory("自进化类1", 23, 14)],
    ))
    assert plan.action == Action.ACCEPT, plan.note


def test_wait_holds_only_when_adjacent(payload_factory, role_factory, task_factory):
    """冷却期"原地等"同样只能在真的站到位之后"""
    solver = TaskSolver()
    plan = solver.plan(_world(
        payload_factory,
        round_no=5,
        roles=[role_factory(10013, "station", 20, 10),
               role_factory(10011, "pioneer", 24, 12)],
        player_tasks=[task_factory("自进化类1", 23, 14, cooldown=5)],
    ))
    assert plan.action == Action.TRAVEL, plan.note
    assert plan.pioneer_command["action"] == "move"


def test_pioneer_travels_to_task_point(payload_factory, role_factory, task_factory):
    """开拓者离任务点远时应该往那儿走"""
    solver = TaskSolver()
    plan = solver.plan(_world(
        payload_factory,
        round_no=81,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10011, "pioneer", 5, 5),
        ],
        player_tasks=[task_factory("自进化类1", 24, 12)],
    ))
    assert plan.action == Action.TRAVEL
    assert plan.pioneer_command["action"] == "move"


# ==========================================================================
# 自进化：第二条同族任务必须更快
# ==========================================================================


def test_skill_is_saved_after_success(payload_factory, base_roles, task_factory):
    """任务成功结束后必须沉淀出 SKILL（任务书 §5.3 的"形成固定 SOP 或者 SKILL"）"""
    solver = TaskSolver()
    solver.plan(_world(
        payload_factory, round_no=90, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
    ))
    solver.plan(_world(
        payload_factory, round_no=91, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task=BEIJING_TASK_TEXT,
        last_cmd_result=SUCCESS_OUTPUT,
    ))
    run = MEMORY.run
    assert run is not None and run.submitted

    # 任务结束（phaseTask 变空、没有 errorCode=2）
    plan = solver.plan(_world(
        payload_factory, round_no=92, roles=base_roles,
        player_tasks=_task_point(task_factory), phase_task="",
    ))
    assert plan.action == Action.IDLE
    signature = skills.signature(skills.FAMILY_API, BEIJING_TASK_TEXT)
    skill = MEMORY.skills.get(signature)
    assert skill is not None, MEMORY.skills
    assert skill.wins == 1
    # 技能里带着这次学到的事实，下一个同族任务可以直接用
    assert any(key.startswith("api.") for key in skill.facts)


def test_skill_reuse_skips_recon(payload_factory, base_roles, task_factory):
    """学到 SKILL 之后，第二个同族任务**首回合就下发 query**（不再 recon）

    这是任务书 §5.3 "实现 Agent 自进化，进而快速做出后续任务"的可执行形式：
    第一个任务探索出"要调哪个地址、带什么鉴权"，第二个任务直接照做。
    """
    signature = skills.signature(skills.FAMILY_API, BEIJING_TASK_TEXT)
    MEMORY.skills[signature] = skills.Skill(
        family=skills.FAMILY_API,
        signature=signature,
        steps=[
            skills.StepSpec("query", (("base", "http://localhost:8899"),
                                      ("auth", "Authorization: Bearer heritage-api-key-2024"),
                                      ("param", "location"))),
        ],
        facts={"api.base_url": "http://localhost:8899"},
        wins=1,
    )

    solver = TaskSolver()
    plan = solver.plan(_world(
        payload_factory,
        round_no=100,
        roles=base_roles,
        player_tasks=_task_point(task_factory),
        phase_task=BEIJING_TASK_TEXT,
    ))
    assert plan.note.startswith("step=query"), plan.note
    # 已知事实被注入了命令里（不必再去读文档猜）
    assert "localhost:8899" in plan.sandbox_command
    assert "heritage-api-key-2024" in plan.sandbox_command


def test_skill_is_retired_after_two_losses():
    """技能连续失败 2 次必须被停用（设计文档V2 原则 P5：学习要能被证伪）"""
    memory = Memory()
    skill = skills.Skill(family=skills.FAMILY_API, signature="api-query|x", steps=[])
    memory.store_skill(skill)
    assert memory.skill_for("api-query|x") is not None
    memory.punish_skill("api-query|x")
    assert memory.skill_for("api-query|x") is not None  # 第 1 次还留着
    memory.punish_skill("api-query|x")
    assert memory.skill_for("api-query|x") is None  # 第 2 次停用


# ==========================================================================
# T7：把文档原文/错误体当答案交上去
# ==========================================================================


def test_doc_echo_is_rejected():
    """任务原文不能当答案提交（PK590557 的 R14/R16/R18）"""
    answer, reason = answer_mod.gate(BEIJING_TASK_TEXT, BEIJING_TASK_TEXT)
    assert answer is None
    assert reason == "doc_echo"


def test_error_body_is_rejected():
    answer, reason = answer_mod.gate(
        '{"error": "Unauthorized", "code": 401}', BEIJING_TASK_TEXT
    )
    assert answer is None
    assert reason == "error_body"


def test_error_null_is_not_rejected():
    """`{"error": null}` 是合法答案（main 分支上修过一次的假阳性回归）"""
    payload = json.dumps({
        "city": "北京", "total_count": 137, "world_heritage_count": 12,
        "types": ["a"], "oldest_era": "周口店遗址", "error": None,
    }, ensure_ascii=False)
    answer, reason = answer_mod.gate(payload, BEIJING_TASK_TEXT)
    assert answer is not None, reason


def test_type_mismatch_rejected():
    """任务原文要求"不能将数字 0 写成 \\"0\\""，类型不对要拒绝"""
    payload = json.dumps({
        "city": "北京", "total_count": "137", "world_heritage_count": "12",
        "types": ["a"], "oldest_era": "周口店遗址",
    }, ensure_ascii=False)
    answer, reason = answer_mod.gate(payload, BEIJING_TASK_TEXT)
    assert answer is None
    assert reason.startswith("type_mismatch")


# ==========================================================================
# 调度与边界
# ==========================================================================


def test_night_keeps_pioneer_on_defense(payload_factory, base_roles, task_factory, robot_factory):
    """夜晚且还没接上任务时，开拓者先守夜（不为了赶路放弃操控武器）"""
    solver = TaskSolver()
    # 第 84 回合是夜晚（(84-1) % 130 = 83 >= 70）
    plan = solver.plan(_world(
        payload_factory,
        round_no=84,
        roles=[r for r in base_roles],
        player_tasks=[task_factory("自进化类1", 24, 12)],
        robots=[robot_factory(30001, 30, 5)],
    ))
    assert plan.action == Action.IDLE
    assert plan.note == "night_defend"


def test_pioneer_holds_position_during_task(payload_factory, role_factory, task_factory):
    """任务进行中开拓者不许离开任务点周围一格（离开即任务结束）"""
    solver = TaskSolver()
    # 站在任务点旁边两格的位置
    plan = solver.plan(_world(
        payload_factory,
        round_no=110,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10011, "pioneer", 24, 14),
        ],
        player_tasks=[task_factory("自进化类1", 24, 12)],
        phase_task=BEIJING_TASK_TEXT,
    ))
    assert plan.action == Action.EXECUTE
    # 距离 2 > 1，应该被拉回任务点
    assert plan.pioneer_command is not None
    assert plan.pioneer_command["action"] == "move"


def test_no_task_point_is_idle(payload_factory, base_roles):
    solver = TaskSolver()
    plan = solver.plan(_world(payload_factory, round_no=1, roles=base_roles))
    assert plan.action == Action.IDLE
    assert plan.note == "no_task_point"


# ==========================================================================
# 观测层：指纹与前进判据
# ==========================================================================


def test_fingerprint_ignores_timestamps():
    a = sandbox_mod.fingerprint("[DONE] step=query elapsed=1.20s 2026-09-14 07:04:48")
    b = sandbox_mod.fingerprint("[DONE] step=query elapsed=9.99s 2026-09-15 11:22:31")
    assert a == b


def test_repeated_detects_identical_output():
    out = sandbox_mod.parse_output(V1_DEAD_LOOP_OUTPUT)
    assert sandbox_mod.repeated(out, out)


def test_parse_markers_and_answer():
    out = sandbox_mod.parse_output(SUCCESS_OUTPUT)
    assert out.answer is not None
    assert out.get("RECON.root") == "" and out.get("QUERY.root")
    assert out.api_calls == 1
    assert out.has_done


def test_timeout_marks_incomplete():
    """`[TIMEOUT]` 且没有 `[DONE]` ⇒ 不能按"死循环"判罚"""
    out = sandbox_mod.parse_output("[TIMEOUT]\n[SCAN] api_calls=0\n")
    assert out.status == "timeout"
    assert not out.complete
