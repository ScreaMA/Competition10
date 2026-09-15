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
import os
import time
from typing import Any

from . import protocol
from .protocol import Pos, Turn, Unit, build_response, distance
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
    started = time.perf_counter()
    turn = Turn.load(payload)
    world = World.load(turn)

    # 遥测：算出"相对上一回合的增量"（击杀、掉血、损失、失败回执）。放在最前面，
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

    # 4) 记账 + 日志（记账必须在日志之前：`record_round` 会把本回合的动作存下来，
    #    下一回合用它把判题器的失败回执翻译成"角色ID:动作"）
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _dump_task_context(turn, task_plan, commands)
    _account(turn, world, commands, task_plan, learned, delta, elapsed_ms)
    return build_response(
        commands,
        prompt=task_plan.prompt,
        sandbox_command=task_plan.sandbox_command,
    )


# 单条 task_dump 的长度上限（可用环境变量 TASK_DUMP_LIMIT 覆盖）。
# 沙盒单次输出本身就被判题器截到 64KB，这里留出余量；调小它的场景通常是
# 判题器侧的 stdout 采集有大小限制。
TASK_DUMP_LIMIT = int(os.getenv("TASK_DUMP_LIMIT", "200000"))


def _dump_task_context(
    turn: Turn,
    task_plan: TaskPlan,
    commands: dict[int, dict[str, Any]],
) -> None:
    """**自进化任务相关的全量日志**（INFO 级，**同时进 stdout 与 `debug.log`**）

    上面那三行是给复盘流水线按字段读的，所以一律截断、压行；任务出问题时
    真正要看的却是**原文**：任务描述写了什么、我们下发了哪条沙盒命令、沙盒
    原样回了什么、提交的答案长什么样。这些在结构化行里看不到，于是 V1 的
    复盘只能靠"反推"——报告里那句"日志未覆盖"多半就是指这个。

    三条设计取舍：

    - **走 INFO，即 stdout 与 `debug.log` 各一份。** 判题系统采集的是进程的
      stdout，`debug.log` 是给本地工具用的；只写文件的话，对局结束时留在
      容器里的日志根本拿不出来。
    - **换行转义成字面量 `
`**。stdout 那边可能被别的工具按行读，压成一行
      最保险；本地看的话 `analyze_log.py --full` 会还原成多行。
    - **只在涉及任务时打**。非任务回合一行都不多写，不会稀释掉结构化行。

    只打**与当前这一回合相关**的东西，不做全量战场转储——后者在
    `request_decoded` 里已经有了。
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return

    # 只有**这一回合确实碰了任务**才打。否则每个普通回合都跟着一条
    # `run_state`，debug.log 会被无意义的内容撑大。
    touches_task = bool(
        turn.phase_task
        or turn.last_cmd_result
        or task_plan.sandbox_command
        or task_plan.prompt
        or turn.llm_resp
        or task_plan.events
        or any(
            c.get("action") in ("acceptTask", "submitAnswer")
            for c in commands.values()
        )
    )
    if not touches_task:
        return

    dumps: list[tuple[str, str]] = []
    if turn.phase_task:
        dumps.append(("phase_task", turn.phase_task))
    if turn.last_cmd_result:
        dumps.append(("last_cmd_result", turn.last_cmd_result))
    if task_plan.sandbox_command:
        dumps.append(("execute_cmd", task_plan.sandbox_command))
    if task_plan.prompt:
        dumps.append(("llm_prompt", task_plan.prompt))
    if turn.llm_resp:
        dumps.append(("llm_resp", turn.llm_resp))
    for unit_id, command in sorted(commands.items()):
        if command.get("action") == "submitAnswer":
            dumps.append((f"submit_answer@{unit_id}", str(command.get("taskAnswer") or "")))

    # 任务子系统的内部状态：这是"为什么这一步又重试了 / 技能为什么没复用"的唯一
    # 直接证据——`task_event` 只给状态变化，中间态在这里。
    state = solver.describe_state()
    if state:
        dumps.append(("run_state", state))

    # 打在任务链路事件之前不合适（事件是 INFO 行），所以这里只放 DEBUG 转储

    for kind, text in dumps:
        body = _escape(text)
        if len(body) > TASK_DUMP_LIMIT:
            body = body[:TASK_DUMP_LIMIT] + "\\n[CLIPPED]"
        # INFO：stdout（判题器采集）与 debug.log（本地复盘）各一份
        LOGGER.info(
            "task_dump round=%d kind=%s bytes=%d text=%s",
            turn.round_no, kind, len(text), body,
        )


def _escape(text: str) -> str:
    """把多行正文压成单行

    **先转义反斜杠**：原文里可能本来就写着转义过的换行（我们下发给沙盒的那条
    命令里全是这种字面量），不先处理它就会被后面的换行转义二次解读，
    `analyze_log --full` 还原出来就是错的。
    """
    return (
        text.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _account(
    turn: Turn,
    world: World,
    commands: dict[int, dict[str, Any]],
    task_plan: TaskPlan,
    learned: list[str],
    delta: RoundDelta,
    elapsed_ms: float,
) -> None:
    """先记账（无条件），再打日志（受日志级别控制）

    两件事必须分开：`TELEMETRY` 的状态是**下一回合**日志与告警的输入，
    不能因为 INFO 级别被关掉就断掉。
    """
    reports = _round_report(turn, world, commands, task_plan, delta)
    TELEMETRY.record_round(
        turn,
        commands,
        spend=reports["spend"],
        idle_units=reports["idle_units"],
        sandbox=bool(task_plan.sandbox_command),
        submissions=reports["submissions"],
        accepts=reports["accepts"],
    )
    _log_turn(turn, world, commands, task_plan, learned, delta, reports, elapsed_ms)


def _round_report(
    turn: Turn,
    world: World,
    commands: dict[int, dict[str, Any]],
    task_plan: TaskPlan,
    delta: RoundDelta,
) -> dict[str, Any]:
    """本回合的派生指标（日志与记账共用一份，避免两处算法漂移）"""
    manned, idle_weapon, idle_target = _weapon_stats(turn, commands)
    busy = set(commands)
    pioneer = _pioneer(turn)
    if task_plan.sandbox_command and pioneer is not None:
        busy.add(pioneer.unit_id)
    return {
        "spend": _spend_brief(turn, commands),
        "manned": manned,
        "idle_weapon": idle_weapon,
        "idle_target": idle_target,
        "idle_units": [u.unit_id for u in turn.characters() if u.unit_id not in busy],
        "submissions": sum(
            1 for c in commands.values() if c.get("action") == "submitAnswer"
        ),
        "accepts": sum(
            1 for c in commands.values() if c.get("action") == "acceptTask"
        ),
        "bag": _bag_brief(turn),
    }


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
#
# 每回合固定四行，全部**单行、机器可解析**（`tools/analyze_log.py` 按 `key=value`
# 提取，多行会把记录拆散）：
#
#     request_decoded  决策前的状态快照
#     strategy_done    本回合下发了什么、花了多少、失败回执
#     round_end        相对上一回合的增量（击杀/掉血/损失/空转）
#     task_event       任务链路的状态变化（0..n 行，只在转折点打）
#     day_summary      每 130 回合一次的总账
#     freeze_alert     金币停滞告警（条件触发）
#
# 这套字段是照着 `战术参考/日志分析模板V2.md` 的每一格设计的：模板里
# `{待人工}` 的格子越少，复盘的成本就越低。新增字段时请同步更新
# `tools/analyze_log.py` 与那份模板。


def _hp_brief(turn: Turn) -> str:
    """基地血量——胜负的第一判据（模板 §1.1）"""
    station = turn.station()
    if station is None:
        return "-"
    full = station.max_health()
    return f"{station.health}/{full if full else '?'}"


def _tower_brief(turn: Turn) -> str:
    """武器明细，如 `3[rocket1@9,25 railgun1@10,25 gatling1@9,24]`

    用 `;`/空格分隔而不是 `,`：坐标里本来就有逗号，用 `,` 分的话复盘侧
    分不清"下一座塔"和"y 坐标"，会把 3 座塔数成 6 座。
    """
    towers = turn.towers()
    detail = " ".join(f"{t.kind}{t.level}@{t.pos.x},{t.pos.y}" for t in towers)
    return f"{len(towers)}[{detail}]"


def _wall_brief(turn: Turn) -> str:
    """围墙段数与等级分布，如 `5[l1:3 l2:2]`（模板 §1.1）"""
    walls = turn.walls()
    if not walls:
        return "0[]"
    counts: dict[int, int] = {}
    for wall in walls:
        counts[wall.level] = counts.get(wall.level, 0) + 1
    detail = " ".join(f"l{level}:{n}" for level, n in sorted(counts.items()))
    return f"{len(walls)}[{detail}]"


def _robot_brief(turn: Turn) -> str:
    """机器人波次构成 + 离基地最近的那一台

    `near=` 是回答"塔够不够得着 / 机器人压到哪了"的唯一线索——报告里
    大量"日志未覆盖"的射程问题都是缺这个。
    """
    robots = turn.alive_robots()
    counts = {kind: 0 for kind in ROBOT_KINDS}
    for robot in robots:
        counts[robot.kind] = counts.get(robot.kind, 0) + 1
    detail = " ".join(
        f"{letter}{counts[kind]}" for letter, kind in zip("smlb", ROBOT_KINDS)
    )
    near = ""
    if robots:
        station = turn.station()
        origin = station.pos if station else Pos(turn.width // 2, turn.height // 2)
        closest = min(robots, key=lambda r: (distance(r.pos, origin), r.robot_id))
        near = f" near={distance(closest.pos, origin)}@{closest.pos.x},{closest.pos.y}"
    return f"{len(robots)}[{detail}]{near}"


def _enemy_brief(turn: Turn) -> str:
    """敌方可见信息

    接口文档 §1.4：`teamEnemy` 只含 `roles`，且只有基地与围墙全图可见，
    其余单位要进视野。所以 `enemy_towers=0` 既可能是"真没有"也可能是
    "没看见"——`enemy_visible` 就是用来区分这两者的，分析时不要误判。
    """
    visible = [u for u in turn.enemies if u.is_alive]
    towers = sum(1 for u in visible if u.is_tower)
    walls = sum(1 for u in visible if u.kind == protocol.WALL)
    return f"enemy_visible={len(visible)} enemy_towers={towers} enemy_walls={walls}"



def _neutral_brief(turn: Turn) -> str:
    """地图上的中立元素清单（矿 / 小贩 / 武器商店 / 任务点）

    只报矿是不够的：**没有小贩 = 永远卖不出矿 = 金币永远回不来**，而这一条在
    日志里曾经完全看不出来（真实对局里金矿从 R9 起恒为 0、工人背包里攒着铜却
    一直没卖，光看 `mines=` 根本判不出是"没小贩"还是"调度没去卖"）。
    """
    counts: dict[str, int] = {}
    for name in turn.zones.values():
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "-"
    # 矿放前面（最常用），其余按名字排
    order = {"stone": 0, "iron": 1, "copper": 2, "vendor": 3, "weaponShop": 4}
    return ",".join(
        f"{name}:{count}"
        for name, count in sorted(
            counts.items(), key=lambda kv: (order.get(kv[0], 9), kv[0])
        )
    )


def _task_brief(turn: Turn) -> str:
    if not turn.player_tasks:
        return "-"
    return " ".join(
        f"{t.task_type}@{t.pos.x},{t.pos.y}"
        f"{'可接' if t.ready else f'/冷却{t.cooldown}'}"
        f"/{t.timeout_rounds}回/{t.score_reward}分"
        for t in turn.player_tasks
    )


def _char_brief(turn: Turn) -> str:
    """角色位置快照——回防与配对诊断全靠它（模板 §5.2）"""
    return " ".join(
        f"{u.unit_id}@{u.pos.x},{u.pos.y}" for u in turn.characters()
    ) or "-"


def _bag_brief(turn: Turn) -> str:
    """背包占用，重复物品压缩成 `copper×21`

    不压缩的话采满一轮的工作者会打出一行几百字符的 `copper,copper,...`，
    复盘时既看不出重点，也把日志行撑得很长。
    """
    parts = []
    for unit in turn.characters():
        if not unit.backpack:
            continue
        counts: dict[str, int] = {}
        for item in unit.backpack:
            counts[item] = counts.get(item, 0) + 1
        detail = ",".join(
            f"{name}×{n}" if n > 1 else name for name, n in sorted(counts.items())
        )
        parts.append(f"{unit.unit_id}{{{detail}}}")
    return " ".join(parts) or "-"


def _spend_brief(turn: Turn, commands: dict[int, dict[str, Any]]) -> int:
    """本回合的**理论**金币支出

    按任务书的价格表算（武器 25、围墙 1 石头不算钱、购买按在售价），
    实际扣款以下一回合的 `gold_delta` 为准——两者不一致就说明有指令被拒。
    """
    total = 0
    for command in commands.values():
        action = command.get("action")
        if action == "build":
            if command.get("name") != protocol.WALL:
                total += protocol.WEAPON_BUILD_COST
        elif action == "buy":
            price = turn.shop_price(str(command.get("name") or ""))
            total += (price or 0) * int(command.get("num") or 1)
    return total


def _weapon_stats(
    turn: Turn, commands: dict[int, dict[str, Any]]
) -> tuple[str, str, str]:
    """(被操控的塔数, 无人操控的塔数, 有人操控但射程内无目标)

    两件事性质完全不同，不能混成一个"空转"：
        `idle_weapon` = 没人站到塔旁边 → 回防/配位失败
        `idle_target` = 有人站好了、射程内没机器人 → 射程覆盖不足
    """
    # 白天角色本来就该去采集/建造，没人站岗不是问题——只有夜里才算空转。
    if turn.is_day:
        return "-", "-", "-"
    manned = 0
    idle_weapon = 0
    idle_target = 0
    controllers = {u.unit_id: u for u in turn.characters()}
    for tower in turn.towers():
        near = [
            unit
            for unit in controllers.values()
            if distance(unit.pos, tower.pos) <= 1
        ]
        if not near:
            idle_weapon += 1
            continue
        manned += 1
        # 有 controllerId 指向这座塔的 attack 指令 = 真的开火了
        firing = any(
            command.get("action") == "attack"
            and str(command.get("controllerId") or "") in {str(u.unit_id) for u in near}
            for command in commands.values()
        )
        if not firing:
            idle_target += 1
    return manned, idle_weapon, idle_target


def _log_turn(
    turn: Turn,
    world: World,
    commands: dict[int, dict[str, Any]],
    task_plan: TaskPlan,
    learned: list[str],
    delta: RoundDelta,
    reports: dict[str, Any],
    elapsed_ms: float,
) -> None:
    """打出本回合的全部日志行（设计文档V2 第 9 章）"""
    if not LOGGER.isEnabledFor(logging.INFO):
        return

    day = (turn.round_no - 1) // protocol.ROUNDS_PER_DAY + 1
    tod = "day" if turn.is_day else "night"
    round_in_day = (turn.round_no - 1) % protocol.ROUNDS_PER_DAY + 1
    bag = reports["bag"]

    # --- 1) 状态快照 ---
    LOGGER.info(
        "request_decoded round=%d day=%d tod=%s round_in_day=%d "
        "team=%s team_id=%s gold=%d gold_delta=%+d score=%d "
        "base=(%d,%d) hp=%s towers=%s walls=%s "
        "robots=%s %s neutral=%s "
        "tasks=[%s] phase=%r task=%s plan=%s "
        "chars=%s bag=%s zone=%s",
        turn.round_no,
        day,
        tod,
        round_in_day,
        turn.team_type,
        turn.team_id,
        turn.gold,
        delta.gold_delta,
        turn.total_score,
        world.origin.x,
        world.origin.y,
        _hp_brief(turn),
        _tower_brief(turn),
        _wall_brief(turn),
        _robot_brief(turn),
        _enemy_brief(turn),
        _neutral_brief(turn),
        _task_brief(turn),
        turn.phase_task[:60],
        task_plan.action,
        task_plan.note or "-",
        _char_brief(turn),
        bag,
        world.zone().summary(),
    )

    # --- 2) 决策结果 ---
    spend = _spend_brief(turn, commands)
    actions = " ".join(
        _describe(unit_id, command, turn)
        for unit_id, command in sorted(commands.items())
    )
    manned, idle_weapon, idle_target = _weapon_stats(turn, commands)
    busy = set(commands)
    pioneer = _pioneer(turn)
    if task_plan.sandbox_command and pioneer is not None:
        busy.add(pioneer.unit_id)
    idle_units = [u.unit_id for u in turn.characters() if u.unit_id not in busy]

    LOGGER.info(
        "strategy_done round=%d commands=%d elapsed=%.2fms gold_spent=%d "
        "actions=%s fail=[%s] sandbox=%s note=%s learn=%s",
        turn.round_no,
        len(commands),
        elapsed_ms,
        spend,
        actions or "-",
        " ".join(delta.failed),
        ("下发" if task_plan.sandbox_command else "空闲"),
        task_plan.note or "-",
        ",".join(learned[:4]) or "-",
    )

    # --- 3) 回合结算 ---
    LOGGER.info(
        "round_end round=%d %s station_damage=%d towers_lost=%d walls_lost=%d "
        "weapons=%d manned=%s idle_weapon=%s idle_target=%s "
        "commands=%d idle_units=%d%s",
        turn.round_no,
        delta.kills_text(),
        delta.station_damage,
        delta.towers_lost,
        delta.walls_lost,
        len(turn.towers()),
        reports["manned"],
        reports["idle_weapon"],
        reports["idle_target"],
        len(commands),
        len(reports["idle_units"]),
        (
            f" idle_ids={','.join(str(i) for i in reports['idle_units'])}"
            if reports["idle_units"]
            else ""
        ),
    )

    # --- 4) 任务链路事件（只在状态变化时才有）---
    for event in task_plan.events:
        LOGGER.info("task_event round=%d %s", turn.round_no, event)

    # --- 5) 冻结告警（条件触发）---
    alert = TELEMETRY.freeze_alert(turn, bag)
    if alert:
        LOGGER.info("%s", alert)

    # --- 6) 每日总账（放在最后：它结的是"刚过去的那一天"）---
    if TELEMETRY.is_new_day(turn):
        LOGGER.info("day_summary %s", TELEMETRY.day_summary(turn))
        TELEMETRY.roll_day()


def _describe(unit_id: int, command: dict[str, Any], turn: Turn) -> str:
    """把一条指令渲染成一行可读文本

    `attack` 会带上落点处**机器人的型号**（如 `attack→(24,13) smallRobot`）——
    复盘判断"目标选择是否合理"（有没有先打 BOSS）全靠这个，光有坐标看不出
    打的是什么。
    """
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
    elif action == "attack":
        kind = _robot_at(turn, targets[0]) if targets else ""
        if kind:
            label += f" {kind}"
    return f"{unit_id}:{label}"


def _robot_at(turn: Turn, spot: dict[str, int]) -> str:
    """落点上的机器人型号（没有则空串）"""
    for robot in turn.alive_robots():
        if robot.pos.x == spot["x"] and robot.pos.y == spot["y"]:
            return robot.kind
    return ""



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
