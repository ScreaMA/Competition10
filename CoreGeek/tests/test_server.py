"""日志模块测试（战术参考 `日志优化方案.md` 的 G1~G14）

覆盖 `agent/server.py` 的 brief 函数与跨回合统计，以及
`tools/analyze_log.py` 的解析与 `--template` 渲染。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.protocol import Pos, Turn
from agent.server import (
    _Telemetry,
    _bag_brief,
    _enemy_brief,
    _failed_brief,
    _gold_spent,
    _hp_brief,
    _idle_brief,
    _tower_brief,
    _wall_brief,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import analyze_log  # noqa: E402


# === brief 函数 ===


def test_hp_brief(payload_factory):
    """G2：基地血量按等级反查满血值"""
    turn = Turn.load(payload_factory())
    assert _hp_brief(turn) == "hp=1500/1500"

    turn = Turn.load(payload_factory(station=None))
    assert _hp_brief(turn) == "hp=-"


def test_tower_and_wall_brief(payload_factory, role_factory):
    """G3/G4：武器塔型+等级、围墙等级分布"""
    payload = payload_factory(
        roles=[
            role_factory(10020, "rocket", 9, 24, level=2),
            role_factory(10030, "railgun", 10, 25, level=1),
            role_factory(40000, "wall", 13, 26, level=1),
            role_factory(40001, "wall", 12, 26, level=2),
        ],
    )
    turn = Turn.load(payload)

    assert _tower_brief(turn) == "2[rocket2,railgun1]"
    assert _wall_brief(turn) == "2[l1:1,l2:1]"
    # 明细里不能有空格，否则按行解析会被截断
    assert " " not in _tower_brief(turn)
    assert " " not in _wall_brief(turn)


def test_tower_and_wall_brief_empty(payload_factory):
    turn = Turn.load(payload_factory())
    assert _tower_brief(turn) == "0[]"
    assert _wall_brief(turn) == "0[]"


def test_enemy_brief(payload_factory, role_factory):
    """G1：敌方可见单位（塔/墙全图可见，其余要进视野）"""
    payload = payload_factory(
        enemies=[
            role_factory(20013, "station", 30, 10),
            role_factory(20020, "rocket", 28, 12),
            role_factory(41000, "wall", 28, 7),
        ],
    )
    brief = _enemy_brief(Turn.load(payload), payload)

    assert "enemy_towers=1" in brief
    assert "enemy_walls=1" in brief
    assert "enemy_visible=3" in brief
    # 报文里没有敌方积分 -> 打 "-"，不能瞎猜
    assert "enemy_score=-" in brief


def test_bag_brief(payload_factory, role_factory):
    """G11：可控制角色的背包合并统计"""
    payload = payload_factory(
        roles=[
            role_factory(10010, "worker", 5, 23, backPackCapability=100,
                         backpack=["stone", "stone", "iron"]),
            role_factory(10012, "worker", 10, 16, backPackCapability=100,
                         backpack=["stone"]),
        ],
    )
    assert _bag_brief(Turn.load(payload)) == "bag=iron:1,stone:3"


def test_failed_brief_with_action_type(payload_factory, role_factory):
    """G8 降级方案：失败角色带上他上一回合下的动作类型"""
    payload = payload_factory(
        roles=[role_factory(10010, "worker", 5, 23, backPackCapability=100)],
    )
    payload["lastRoundRoleActionResults"] = {"10010": False, "10012": True}
    turn = Turn.load(payload)

    brief = _failed_brief(turn, {"10010": {"action": "move"}})
    assert brief == "fail=[10010:move]"


def test_idle_brief(payload_factory, role_factory, robot_factory):
    """G12：无人操控与有人操控但无目标分开统计"""
    payload = payload_factory(
        roles=[
            role_factory(10010, "worker", 9, 23, backPackCapability=100),
            role_factory(10020, "gatling", 9, 24, attackRange=4),
            role_factory(10030, "railgun", 10, 25, attackRange=7),
        ],
        # (12,24) 离加特林 3 格（射程内）、离电磁炮 2 格（射程内）
        robots=[robot_factory(30001, 12, 24, targetTeam="challenger")],
    )
    turn = Turn.load(payload)
    brief = _idle_brief(turn)

    # 加特林有人操控且有目标；电磁炮没人操控；第三座塔不存在
    assert "idle_man=1" in brief
    assert "idle_target=0" in brief


def test_gold_spent_counts_build_and_buy(payload_factory):
    """G9：理论花费（建塔 25 金、买券按商店价）"""
    turn = Turn.load(payload_factory())

    commands = {
        "10010": {"action": "build", "name": "rocket", "targetPos": []},
        "10012": {"action": "build", "name": "wall", "targetPos": []},
        "10011": {"action": "buy", "name": "WeaponUpgradeVoucher1", "num": 1},
    }
    assert _gold_spent(commands, turn) == 125


# === 跨回合统计 ===


def test_telemetry_counts_kills_and_losses(payload_factory, robot_factory, role_factory):
    """G6：夜里机器人减少算击杀，跨昼夜清场不算"""
    telemetry = _Telemetry()

    night1 = Turn.load(payload_factory(
        round_no=71,
        roles=[role_factory(10010, "worker", 5, 23, backPackCapability=100)],
        robots=[robot_factory(30001, 4, 4), robot_factory(30002, 5, 4)],
    ))
    telemetry.observe(night1)

    night2 = Turn.load(payload_factory(
        round_no=72,
        roles=[role_factory(10010, "worker", 5, 23, backPackCapability=100)],
        robots=[robot_factory(30001, 4, 4)],
    ))
    telemetry.observe(night2)
    assert telemetry.kills == 1

    # 天亮清场：夜晚最后一回合(130) -> 次日首回合(131)，机器人被系统清掉，不算击杀
    telemetry.observe(Turn.load(payload_factory(
        round_no=130,
        robots=[robot_factory(30003, 6, 4)],
    )))
    telemetry.observe(Turn.load(payload_factory(round_no=131, robots=[])))
    assert telemetry.kills == 0


def test_telemetry_counts_unit_losses(payload_factory, role_factory):
    """单位消失计为损失"""
    telemetry = _Telemetry()

    telemetry.observe(Turn.load(payload_factory(
        round_no=71,
        roles=[
            role_factory(10010, "worker", 5, 23, backPackCapability=100),
            role_factory(10012, "worker", 10, 16, backPackCapability=100),
        ],
    )))
    telemetry.observe(Turn.load(payload_factory(
        round_no=72,
        roles=[role_factory(10010, "worker", 5, 23, backPackCapability=100)],
    )))
    assert telemetry.losses == 1


def test_telemetry_day_summary(payload_factory, role_factory):
    """每日总账：金币峰值、建造/交易次数、任务接取与交卷"""
    telemetry = _Telemetry()
    telemetry.observe(Turn.load(payload_factory(round_no=1, gold=75)))
    telemetry.note_commands({
        "10010": {"action": "build", "name": "rocket", "targetPos": []},
        "10011": {"action": "acceptTask"},
    })
    telemetry.observe(Turn.load(payload_factory(round_no=2, gold=120)))
    telemetry.note_commands({
        "10010": {"action": "sell", "name": "stone", "num": 10},
        "10011": {"action": "submitAnswer", "taskAnswer": "42"},
    })

    brief = telemetry.day_summary_brief(2, 1, Turn.load(payload_factory(round_no=2, gold=120)))
    assert "gold_peak=120" in brief
    assert "build=1" in brief
    assert "sell=1" in brief
    assert "task_accept=1" in brief
    assert "task_done=1" in brief


# === analyze_log.py ===


SAMPLE_LINES = """\
2026-09-14 10:00:00,000 | INFO | agent.server | request_decoded id=1 round=1 team=1 team_type=challenger roles=9 gold=75 score=0 base=(10,24) towers=0[] walls=0[] robots=0(s0 m0 l0 b0) mines=6 tasks=[] phase=- fail=[] hp=1500/1500 enemy_score=- enemy_towers=0 enemy_walls=1 enemy_visible=2 bag=-
2026-09-14 10:00:00,010 | INFO | agent.server | strategy_done id=1 round=1 commands=1 elapsed=3.00ms sandbox=空闲 gold_spent=25 actions=10010:build→(11,22) rocket
2026-09-14 10:00:00,020 | INFO | agent.server | request_decoded id=2 round=71 team=1 team_type=challenger roles=9 gold=50 score=80 base=(10,24) towers=1[rocket1] walls=5[l1:5] robots=3(s2 m1 l0 b0) mines=6 tasks=[] phase=- fail=[10010:move] hp=1400/1500 enemy_score=- enemy_towers=0 enemy_walls=1 enemy_visible=2 bag=stone:3
2026-09-14 10:00:00,030 | INFO | agent.server | round_end id=2 round=71 gold=50 hp=1400/1500 towers=1[rocket1] walls=5[l1:5] kills=2/3 loses=0 idle_man=0 idle_target=1
2026-09-14 10:00:00,040 | INFO | agent.server | strategy_done id=2 round=71 commands=2 elapsed=4.00ms sandbox=空闲 gold_spent=0 actions=10010:attack→(12,24) 10011:move→(11,22)
2026-09-14 10:00:00,050 | INFO | agent.server | day_summary id=3 day=1 gold_peak=120 gold_final=50 build=3 upgrade=1 sell=4 task_accept=2 task_done=1 score_gain=80 hp_lost=100 kills=7 loses=0
2026-09-14 10:00:00,060 | INFO | agent.server | response_sent id=2 status=200 bytes=300
"""


@pytest.fixture
def sample_log(tmp_path):
    path = tmp_path / "debug.log"
    path.write_text(SAMPLE_LINES, encoding="utf-8")
    return path


def test_analyze_log_summary(sample_log):
    """解析：回合范围、金币峰值、击杀、任务计数"""
    stats = analyze_log.analyze(sample_log)

    assert stats is not None
    assert stats.first_round == 1
    assert stats.last_round == 71
    assert stats.gold_peak == 120 or stats.gold_peak == 50  # 逐回合 gold 序列
    assert stats.kills == 2
    assert stats.task_accepts == 0  # 样本里没有领任务动作
    assert stats.statuses == {200: 1}


def test_template_fills_key_metrics(sample_log):
    """--template：§1.1 的关键指标能自动填出来"""
    stats = analyze_log.analyze(sample_log)
    report = analyze_log.render_template(stats)

    # 塔型与等级、围墙等级分布、金币、击杀都要填上
    assert "1座（rocket1）" in report
    assert "5段（l1:5）" in report
    assert "峰值120" in report
    assert "| **机器人击杀** | 2 |" in report
    # 敌方经济报文里没有，必须留待人工而不是编造
    assert "{待人工}" in report
    assert "自动填充：" in report


def test_template_timeline_picks_build(sample_log):
    """时间线要能认出建塔动作"""
    report = analyze_log.render_template(analyze_log.analyze(sample_log))
    assert "武器塔 rocket→(11,22)" in report


def test_analyze_empty_log(tmp_path):
    """空日志不崩溃"""
    path = tmp_path / "empty.log"
    path.write_text("", encoding="utf-8")
    stats = analyze_log.analyze(path)

    assert stats is not None
    assert stats.rounds == set()
    assert "无法生成" in analyze_log.render_template(stats)
