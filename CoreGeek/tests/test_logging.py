"""日志格式用例。

对应设计文档V2 第 9 章。

日志是复盘流水线唯一的输入，而复盘模板（`战术参考/日志分析模板V2.md`）
是按字段名逐格设计的。所以这里做**往返测试**：

    造报文 → 跑 `decide()` → 收集日志行 → 交给 `analyze_log` 解析
    → 断言解析出来的值与输入一致

字段名一改，这个用例就会红——比等到复盘时发现模板填不出格子早得多。

`--template` / `--issue` 的渲染也在这里顺带跑一遍：只要它们不抛异常、
并且关键格子被填上，就说明分析侧与日志侧还对得上。
"""

from __future__ import annotations

import logging

import pytest

from agent.brain import decide
from agent.telemetry import TELEMETRY


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """遥测是模块级单例，跨用例残留会让"第 1 回合"被当成续局"""
    TELEMETRY.reset()
    yield
    TELEMETRY.reset()


def _lines(caplog) -> list[str]:
    """把 caplog 里的记录还原成日志行（去掉时间戳前缀）"""
    return [record.getMessage() for record in caplog.records]


def _dump(tmp_path, lines: list[str]) -> "object":
    """把捕获到的行写成 debug.log 的样子（带时间戳前缀）"""
    path = tmp_path / "debug.log"
    path.write_text(
        "\n".join(f"2026-09-15 00:00:00,000 | INFO | agent.brain | {l}" for l in lines),
        encoding="utf-8",
    )
    return path


def _night_payload(payload_factory, role_factory, robot_factory, round_no=75):
    """一个有塔、有人操控、有机器人的夜战回合"""
    return payload_factory(
        round_no=round_no,
        roles=[
            role_factory(10013, "station", 20, 10),
            role_factory(10010, "worker", 19, 13, backpack=["stone"]),
            role_factory(10012, "worker", 21, 13),
            role_factory(10020, "gatling", 20, 13, attackRange=5),
            role_factory(40000, "wall", 18, 8),
        ],
        robots=[robot_factory(30001, 24, 13, "smallRobot")],
    )


# ==========================================================================
# 行格式
# ==========================================================================


def test_round_emits_three_fixed_lines(
    payload_factory, role_factory, robot_factory, caplog
):
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory))
    lines = _lines(caplog)
    heads = [line.split(" ", 1)[0] for line in lines]
    assert heads[:3] == ["request_decoded", "strategy_done", "round_end"]


def test_every_line_is_single_line(
    payload_factory, role_factory, robot_factory, caplog
):
    """多行会把记录拆散——分析侧是按行解析的"""
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory))
    for line in _lines(caplog):
        assert "\n" not in line
        assert "\r" not in line


def test_request_decoded_has_documented_fields(
    payload_factory, role_factory, robot_factory, caplog
):
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory, 75))
    line = next(l for l in _lines(caplog) if l.startswith("request_decoded"))
    for field in (
        "round=75", "day=1", "tod=night", "round_in_day=75",
        "team=challenger", "gold=", "gold_delta=", "score=",
        "base=(20,9)", "hp=1500/1500", "towers=", "walls=", "robots=",
        "enemy_visible=", "mines=", "tasks=[", "phase=", "task=", "plan=",
        "chars=", "bag=", "zone=",
    ):
        assert field in line, f"{field} 不在 request_decoded 里：{line}"
    # 塔明细带类型/等级/坐标，机器人带构成
    assert "gatling1@20,13" in line
    assert "[s1 m0 l0 b0]" in line
    assert "near=" in line
    assert "walls=1[l1:1]" in line


def test_strategy_done_has_documented_fields(
    payload_factory, role_factory, robot_factory, caplog
):
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory, 75))
    line = next(l for l in _lines(caplog) if l.startswith("strategy_done"))
    for field in ("round=75", "commands=", "elapsed=", "gold_spent=",
                  "actions=", "fail=[", "sandbox=", "note=", "learn="):
        assert field in line, f"{field} 不在 strategy_done 里：{line}"
    assert "ms" in line  # 耗时带单位


def test_attack_action_carries_robot_kind(
    payload_factory, role_factory, robot_factory, caplog
):
    """`attack` 要带落点上的机器人型号——判断目标选择全靠它"""
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory, 75))
    line = next(l for l in _lines(caplog) if l.startswith("strategy_done"))
    assert "attack" in line
    assert "smallRobot" in line


def test_round_end_has_documented_fields(
    payload_factory, role_factory, robot_factory, caplog
):
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(_night_payload(payload_factory, role_factory, robot_factory, 75))
    line = next(l for l in _lines(caplog) if l.startswith("round_end"))
    for field in ("round=75", "kills=", "kill_score=", "station_damage=",
                  "towers_lost=", "walls_lost=", "weapons=", "manned=",
                  "idle_weapon=", "idle_target=", "commands=", "idle_units="):
        assert field in line, f"{field} 不在 round_end 里：{line}"
    # 夜战里武器空转必须给出数值（白天才是 `-`）
    assert "manned=" in line and "manned=-" not in line


def test_daytime_weapon_stats_are_dash(payload_factory, base_roles, caplog):
    """白天角色该去干活，没人站岗不是问题，所以打 `-`"""
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(payload_factory(round_no=1, roles=base_roles))
    line = next(l for l in _lines(caplog) if l.startswith("round_end"))
    assert "manned=- idle_weapon=- idle_target=-" in line


def test_failed_action_reports_action_name(
    payload_factory, role_factory, caplog
):
    """失败回执要翻成 `角色ID:动作`，只给 ID 复盘看不出是什么动作"""
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        # 第 1 回合先下发一条指令，让遥测记住"这个角色在做什么"
        decide(payload_factory(
            round_no=1,
            roles=[role_factory(10013, "station", 20, 10),
                   role_factory(10010, "worker", 30, 30)],
        ))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(payload_factory(
            round_no=2,
            roles=[role_factory(10013, "station", 20, 10),
                   role_factory(10010, "worker", 30, 30)],
            last_action_results={"10010": False},
        ))
    line = next(l for l in _lines(caplog) if l.startswith("strategy_done"))
    assert "fail=[10010:" in line


def test_task_events_only_on_transitions(
    payload_factory, role_factory, task_factory, caplog
):
    """`task_event` 只在转折点打，不每回合刷"""
    roles = [role_factory(10013, "station", 20, 10),
             role_factory(10011, "pioneer", 24, 13)]
    tasks = [task_factory("自进化类1", 24, 12)]
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(payload_factory(round_no=1, roles=roles, player_tasks=tasks))
    accept_events = [l for l in _lines(caplog) if l.startswith("task_event")]
    assert any("event=accept" in line for line in accept_events)
    assert all("round=1" in line for line in accept_events)


def test_no_task_event_on_quiet_round(payload_factory, base_roles, caplog):
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(payload_factory(round_no=1, roles=base_roles))
    assert not [l for l in _lines(caplog) if l.startswith("task_event")]


# ==========================================================================
# 往返：日志 → 分析器
# ==========================================================================


def test_analyzer_parses_what_the_logger_emits(
    payload_factory, role_factory, task_factory, tmp_path, caplog
):
    """**往返测试**：日志字段一改，模板的自动填充就会失效，这里必须红"""
    import analyze_log

    roles = [
        role_factory(10013, "station", 20, 10),
        role_factory(10010, "worker", 5, 23),
        role_factory(10011, "pioneer", 24, 11),   # 站在任务点旁边
        role_factory(10012, "worker", 10, 16),
    ]
    tasks = [task_factory("自进化类1", 24, 12)]

    with caplog.at_level(logging.INFO, logger="agent.brain"):
        for round_no in (1, 2, 3, 71):
            decide(payload_factory(
                round_no=round_no, gold=100, roles=roles, player_tasks=tasks,
            ))
    path = _dump(tmp_path, _lines(caplog))

    stats = analyze_log.analyze(path)
    assert stats is not None
    assert stats.rounds, "分析器没解析到任何回合"
    assert stats.first_round == 1 and stats.last_round == 71
    assert stats.team_type == "challenger"
    assert stats.gold_peak == 100
    assert stats.night_rounds == 1  # 只有 R71 是夜
    assert stats.accepted >= 1, "领任务没被统计到"
    # 塔/墙的明细字段能被拆开
    assert stats.walls_now()
    assert isinstance(stats.towers(), list)


def test_analyzer_renders_both_templates(
    payload_factory, role_factory, task_factory, tmp_path, caplog
):
    """两个渲染入口都要能跑通，且关键格子自动填上"""
    import analyze_log

    roles = [
        role_factory(10013, "station", 20, 10),
        role_factory(10010, "worker", 5, 23),
        role_factory(10011, "pioneer", 24, 11),
        role_factory(10012, "worker", 10, 16),
    ]
    with caplog.at_level(logging.INFO, logger="agent.brain"):
        for round_no in (1, 2):
            decide(payload_factory(
                round_no=round_no, gold=100, roles=roles,
                player_tasks=[task_factory("自进化类1", 24, 12)],
            ))
    path = _dump(tmp_path, _lines(caplog))
    stats = analyze_log.analyze(path)

    report = analyze_log.render_template(stats, path)
    assert "关键数据对比" in report
    assert "日志原文" in report          # 分析稿也要留原文位
    assert report.count("{待人工}") < 40  # 自动填得动的格子不该太多

    issue = analyze_log.render_issue(stats, path)
    assert "```claude-prompt" in issue   # 自动化系统靠这个围栏提取
    assert "```log" in issue             # 留出的日志原文位
    assert "日志原文" in issue
    # 建议必须落到新模块上，不能指向已经废弃的 `src/agent/brain.py`
    fence = issue.split("```claude-prompt")[1]
    assert "agent/brain.py" not in fence
    assert "strategy/" in fence or "{待人工" in fence


def test_summary_and_task_trace_do_not_crash(
    payload_factory, base_roles, tmp_path, caplog, capsys
):
    import analyze_log

    with caplog.at_level(logging.INFO, logger="agent.brain"):
        decide(payload_factory(round_no=1, roles=base_roles))
    path = _dump(tmp_path, _lines(caplog))
    stats = analyze_log.analyze(path)

    analyze_log.print_summary(stats, path)
    analyze_log.print_task_trace(stats)
    assert "回合" in capsys.readouterr().out


# ==========================================================================
# 自进化任务的全量日志（DEBUG 级 task_dump）
# ==========================================================================


def test_task_dump_emitted_for_task_rounds(
    payload_factory, role_factory, task_factory, caplog
):
    """任务回合要打全量日志：任务描述 / 下发的沙盒命令 / 沙盒回包 / 提交的答案

    INFO 那三行是给复盘按字段读的，一律截断压行；排查任务问题要看的是**原文**，
    所以这些走 DEBUG（只进 `debug.log`，不进判题器收到的 stdout）。
    """
    roles = [role_factory(10013, "station", 20, 10),
             role_factory(10011, "pioneer", 24, 11)]
    tasks = [task_factory("自进化类1", 24, 12)]

    with caplog.at_level(logging.DEBUG, logger="agent.brain"):
        # 第 1 回合领任务、第 2 回合任务开始并下发 recon
        decide(payload_factory(round_no=1, roles=roles, player_tasks=tasks))
        decide(payload_factory(
            round_no=2, roles=roles, player_tasks=tasks,
            phase_task="请阅读task_1_beijing.md，获取任务信息" * 3,
        ))
    dumps = [r.getMessage() for r in caplog.records if "task_dump" in r.getMessage()]
    kinds = {line.split("kind=")[1].split()[0] for line in dumps}
    assert "phase_task" in kinds, dumps
    assert "execute_cmd" in kinds, dumps


def test_task_dump_not_emitted_without_task(payload_factory, base_roles, caplog):
    """没任务就一行都不多打——否则 debug.log 会被撑爆"""
    with caplog.at_level(logging.DEBUG, logger="agent.brain"):
        decide(payload_factory(round_no=1, roles=base_roles))
    assert not [r for r in caplog.records if "task_dump" in r.getMessage()]


def test_task_dump_is_single_line(payload_factory, role_factory, task_factory, caplog):
    """多行正文必须转义（分析侧按行读）"""
    with caplog.at_level(logging.DEBUG, logger="agent.brain"):
        decide(payload_factory(
            round_no=2,
            roles=[role_factory(10013, "station", 20, 10),
                   role_factory(10011, "pioneer", 24, 11)],
            player_tasks=[task_factory("自进化类1", 24, 12)],
            phase_task="第一行\n第二行\r\n第三行",
        ))
    for record in caplog.records:
        message = record.getMessage()
        if "task_dump" in message:
            assert "\n" not in message
            assert "\r" not in message


def test_task_dump_survives_roundtrip_through_analyzer(
    payload_factory, role_factory, task_factory, tmp_path, caplog
):
    """转义 → 还原必须逐字节一致（原文里有反斜杠也不能被吃掉）

    沙盒命令里全是转义过的换行，还原错一位这条日志就没法用来查问题了。
    """
    import analyze_log

    raw = 'python3 - <<' + "'PYEOF'\n" + 'print("a\\nb")\nPYEOF'
    with caplog.at_level(logging.DEBUG, logger="agent.brain"):
        decide(payload_factory(
            round_no=2,
            roles=[role_factory(10013, "station", 20, 10),
                   role_factory(10011, "pioneer", 24, 11)],
            player_tasks=[task_factory("自进化类1", 24, 12)],
            phase_task=raw,
        ))
    path = _dump(tmp_path, [r.getMessage() for r in caplog.records])
    stats = analyze_log.analyze(path)
    restored = [text for _, kind, text in stats.dumps if kind == "phase_task"]
    assert restored, stats.dumps
    assert restored[0] == raw


def test_full_mode_reports_when_nothing_to_show(capsys, tmp_path):
    import analyze_log

    path = tmp_path / "empty.log"
    path.write_text(
        "2026-09-15 00:00:00,000 | INFO | agent.brain | request_decoded round=1\n",
        encoding="utf-8",
    )
    stats = analyze_log.analyze(path)
    analyze_log.print_full(stats)
    assert "task_dump" in capsys.readouterr().out
