"""编排器：把一回合的报文变成一回合的响应。

对应设计文档V2 §3.2。

`decide(payload) -> response` 是整个客户端唯一的集成面。它做四件事：

1. **解析**：报文 → `Turn` → `World`（无状态，每回合重算）。
2. **学习**：把上一回合建造动作的执行结果并进可建造区（`world.absorb_results`）。
3. **决策**：白天/夜晚分别取任务、经济、防御三条线各自的指令。
4. **消解冲突**：一个角色一回合只能有一条指令（接口文档 §2.3 注），
   按 任务 > 建造/采集 > 防御移动 的优先级合并。

异常预算是这里最需要小心的东西（任务书 §8：累计 5 次异常即整场停调度）。
所有指令都由 `protocol.py` 的构造器生成，构造器返回 `None` 表示"这条指令
构不出来"，此时该角色本回合不出指令——宁可少做一个动作，也不能发一条字段
不全的指令（那会直接消耗异常预算）。
"""

from __future__ import annotations

import logging
from typing import Any

from . import protocol
from .protocol import Pos, Turn, Unit, build_response
from .strategy import defense, economy
from .strategy.task import solver
from .strategy.task.solver import Action, TaskPlan
from .telemetry import TELEMETRY, RoundDelta
from .world import World, absorb_results

LOGGER = logging.getLogger("agent.brain")

ROBOT_KINDS = ("smallRobot", "middleRobot", "largeRobot", "bossRobot")


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    """一回合的入口。任何异常都退化成"空指令"，不让进程崩掉。

    空指令不算异常（任务书 §8 只把"指令字段缺失/无法识别"记为异常），
    所以这条兜底路径是安全的。
    """
    try:
        return _decide(payload)
    except Exception:
        LOGGER.exception("decision failed")
        return dict(protocol.EMPTY_RESPONSE)


def _decide(payload: dict[str, Any]) -> dict[str, Any]:
    turn = Turn.load(payload)
    world = World.load(turn)

    # 遥测：算出"相对上一回合的增量"（击杀、掉血、损失）。放在最前面，
    # 因为它读的是**上一回合**的快照。
    delta = TELEMETRY.observe(turn)

    # 学习：上一回合的建造结果 → 可建造区（必须在决策之前完成）
    learned = absorb_results(turn) if turn.round_no > 1 else []

    commands: dict[int, dict[str, Any]] = {}
    claimed: set[Pos] = set()

    # 1) 任务链路：开拓者的指令优先级最高
    task_plan = solver.plan(world)
    used: set[int] = set()
    pioneer = _pioneer(turn)
    if task_plan.pioneer_command is not None and pioneer is not None:
        commands[pioneer.unit_id] = task_plan.pioneer_command
        used.add(pioneer.unit_id)
        claimed.add(_landing(task_plan.pioneer_command, pioneer.pos))
    elif task_plan.hold and pioneer is not None:
        # 任务中驻守 / 在任务点旁蹲守：不许被空转兜底拉走
        # （离开任务点周围一格内即视为任务结束，任务书 §5）
        used.add(pioneer.unit_id)

    # 2) 白天走经济，夜晚走防御
    if turn.is_day:
        for unit_id, command in economy.plan_day(world, claimed).items():
            if unit_id not in used:
                commands[unit_id] = command
                used.add(unit_id)
    else:
        # 任务进行中的开拓者留在任务点：任务书 §5 规定"离开己方任务点周围一格内"
        # 即视为任务结束，回防的代价是把整个任务丢掉。
        exclude = frozenset({pioneer.unit_id}) if (task_plan.active and pioneer) else frozenset()
        for action in defense.night_actions(world, exclude):
            unit_id = action.unit.unit_id
            if action.command is None or unit_id in used:
                continue
            commands[unit_id] = action.command
            used.add(unit_id)

    # 3) 兜底：别让任何角色整回合空转
    _fill_idle(world, commands, used, claimed)

    _account(turn, commands, task_plan)
    _log_turn(turn, world, commands, task_plan, learned, delta)
    return build_response(
        commands,
        prompt=task_plan.prompt,
        sandbox_command=task_plan.sandbox_command,
    )


def _account(turn: Turn, commands: dict[int, dict[str, Any]], task_plan: TaskPlan) -> None:
    """把本回合的事件记进当日总账（供 `day_summary` 使用）"""
    if task_plan.sandbox_command:
        TELEMETRY.count_sandbox()
    for command in commands.values():
        if command.get("action") == "submitAnswer":
            TELEMETRY.count_submit()
    busy = set(commands)
    if task_plan.sandbox_command:
        pioneer = _pioneer(turn)
        if pioneer is not None:
            busy.add(pioneer.unit_id)
    TELEMETRY.count_idle(
        sum(1 for unit in turn.characters() if unit.unit_id not in busy)
    )


def _pioneer(turn: Turn) -> Unit | None:
    pioneers = turn.pioneers()
    return pioneers[0] if pioneers else None


def _landing(command: dict[str, Any], fallback: Pos) -> Pos:
    """指令的落点（用于占位，避免别人走到同一格）"""
    positions = command.get("targetPos") or []
    return Pos.load(positions[0]) if positions else fallback


def _fill_idle(
    world: World,
    commands: dict[int, dict[str, Any]],
    used: set[int],
    claimed: set[Pos],
) -> None:
    """给没摊上指令的角色找一个不太差的动作

    V1 的对战日志里 `idle_man` 峰值到 3（三个角色一整回合不动），那是最纯粹的
    浪费。兜底动作是"向最近的武器靠拢"：夜里是就位操控，白天是提前站好位、
    天黑不用再跑一趟。
    """
    for unit in world.turn.characters():
        if unit.unit_id in used:
            continue
        command = defense.guard_weapon(world, unit, claimed)
        if command is not None:
            commands[unit.unit_id] = command
            used.add(unit.unit_id)


# ==========================================================================
# 日志（设计文档V2 第 9 章）
# ==========================================================================


def _log_turn(
    turn: Turn,
    world: World,
    commands: dict[int, dict[str, Any]],
    task_plan: TaskPlan,
    learned: list[str],
    delta: RoundDelta,
) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    if TELEMETRY.is_new_day(turn):
        LOGGER.info("day_summary %s", TELEMETRY.day_summary(turn))
        TELEMETRY.roll_day()

    station = turn.station()
    hp = f"{station.health}/{station.max_health()}" if station else "-"
    # 用 `;` 分隔而不是 `,`：坐标里本来就有逗号，用 `,` 分的话复盘侧
    # 分不清"下一座塔"和"y 坐标"，会把 3 座塔数成 6 座。
    towers = ";".join(
        f"{t.kind}{t.level}@{t.pos.x},{t.pos.y}" for t in turn.towers()
    ) or "-"
    robots = turn.alive_robots()
    counts = {kind: 0 for kind in ROBOT_KINDS}
    for robot in robots:
        counts[robot.kind] = counts.get(robot.kind, 0) + 1
    tasks = " ".join(
        f"{t.task_type}@{t.pos.x},{t.pos.y}"
        f"{'可接' if t.ready else f'/冷却{t.cooldown}'}"
        f"/{t.timeout_rounds}回/{t.score_reward}分"
        for t in turn.player_tasks
    ) or "-"
    bag = " ".join(
        f"{u.unit_id}{{{','.join(u.backpack)}}}"
        for u in turn.characters()
        if u.backpack
    ) or "-"

    LOGGER.info(
        "request_decoded round=%d team=%s team_id=%s gold=%d score=%d base=(%d,%d) hp=%s "
        "towers=%s walls=%d robots=%d(s%d m%d l%d b%d) tasks=[%s] "
        "phase=%r task=%s zone=%s bag=%s",
        turn.round_no,
        turn.team_type,
        turn.team_id,
        turn.gold,
        turn.total_score,
        world.origin.x,
        world.origin.y,
        hp,
        towers,
        len(turn.walls()),
        len(robots),
        counts["smallRobot"],
        counts["middleRobot"],
        counts["largeRobot"],
        counts["bossRobot"],
        tasks,
        turn.phase_task[:60],
        task_plan.action,
        world.zone().summary(),
        bag,
    )

    actions = " ".join(
        _describe(unit_id, command) for unit_id, command in sorted(commands.items())
    )
    failed = [str(unit_id) for unit_id, ok in turn.last_action_results.items() if not ok]
    LOGGER.info(
        "strategy_done round=%d commands=%d actions=%s sandbox=%s note=%s "
        "learn=%s fail=[%s]",
        turn.round_no,
        len(commands),
        actions or "-",
        ("下发" if task_plan.sandbox_command else "空闲"),
        task_plan.note or "-",
        ",".join(learned[:4]) or "-",
        ",".join(failed),
    )

    # 任务中的开拓者不算空转：它的产出走 `executeCmd`（接口文档 §2.1 的两个
    # 字段互不占用），角色指令槽空着是设计使然。
    busy = set(commands)
    if task_plan.sandbox_command and _pioneer(turn) is not None:
        busy.add(_pioneer(turn).unit_id)
    idle = [unit.unit_id for unit in turn.characters() if unit.unit_id not in busy]
    LOGGER.info(
        "round_end round=%d %s station_damage=%d towers_lost=%d walls_lost=%d "
        "commands=%d idle=%d%s",
        turn.round_no,
        delta.text(),
        delta.station_damage,
        delta.towers_lost,
        delta.walls_lost,
        len(commands),
        len(idle),
        f" idle_ids={','.join(str(i) for i in idle)}" if idle else "",
    )


def _describe(unit_id: int, command: dict[str, Any]) -> str:
    action = command.get("action", "?")
    targets = command.get("targetPos") or []
    spot = ""
    if targets:
        spot = f"→({targets[0]['x']},{targets[0]['y']})"
        if len(targets) > 1:
            spot += f"+{len(targets) - 1}"
    name = command.get("name")
    label = f"{action}{spot}"
    if name:
        label += f" {name}"
    if action == "submitAnswer":
        label = f"submitAnswer({len(command.get('taskAnswer') or '')}B)"
    return f"{unit_id}:{label}"


def describe(task_plan: TaskPlan) -> str:
    """任务计划的单行摘要（给 server 的诊断日志用）"""
    bits = [f"task={task_plan.action}"]
    if task_plan.note:
        bits.append(task_plan.note)
    if task_plan.sandbox_command:
        bits.append("sandbox=1")
    if task_plan.prompt:
        bits.append("prompt=1")
    return " ".join(bits)
