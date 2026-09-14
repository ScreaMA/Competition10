"""服务器模块：HTTP服务器，接收判题系统的POST请求。

对应设计文档 3.2 节。

注意（实测踩坑）：判题系统是向 `POST /` 发送请求的，部分实现会只在
`POST /action` 上路由而收不到消息。本模块不检查 `self.path`，
任何路径的POST请求都会被正常处理。

日志（战术参考 `日志优化方案.md`）：
    INFO 行保持单行、机器可解析（`tools/analyze_log.py` 按行正则提取），
    新字段一律追加在行尾，不破坏已有解析。三条常规行分别是
    `request_decoded`（本回合开局状态）、`strategy_done`（本回合指令）、
    `round_end`（上一回合战果结算），外加每日一行 `day_summary` 总账。
"""

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide, sandbox_command
from .protocol import (
    COPPER_MINE,
    IRON_MINE,
    ROUNDS_PER_DAY,
    STONE_MINE,
    TOWER_TYPES,
    WALL,
    WEAPON_BUILD_COST,
    Turn,
    distance,
    station_full_health,
    wall_full_health,
)

LOGGER = logging.getLogger(__name__)

# 沙盒输出预览的最大长度：任务相关的线索留这么多就够，不打全量报文
MAX_SANDBOX_PREVIEW = 300

# 交卷内容在动作日志里的预览长度
ANSWER_PREVIEW = 80

# 机器人的种类顺序，概览里的 s/m/l/b 与接口文档的四种机器人一一对应
ROBOT_KINDS = ("smallRobot", "middleRobot", "largeRobot", "bossRobot")

# 决策耗时告警阈值（设计文档6.6节：预留0.2秒缓冲）
DECISION_BUDGET_WARN_MS = 800

# 商品售价兜底（任务书4.6.3）。`weaponShopList` 里有价就用报文里的，
# 这里只用于把 `gold_spent` 这个"理论花费"算出来（实际扣款看下一回合 gold 差值）。
ITEM_PRICE_FALLBACK = {
    "WeaponUpgradeVoucher1": 100,
    "WeaponUpgradeVoucher2": 150,
    "WallUpgradeVoucher1": 20,
    "WallUpgradeVoucher2": 30,
    "StationUpgradeVoucher1": 100,
    "StationUpgradeVoucher2": 150,
    "WallFixer": 10,
    "Medicine": 10,
    "DizzyWeapon": 100,
    "Bomb": 100,
    "SmallRobotSummonOrder": 20,
    "MiddleRobotSummonOrder": 30,
    "LargeRobotSummonOrder": 100,
    "BossRobotSummonOrder": 200,
}

# 请求ID计数器
_request_id = 0


def _truncate(text: str, limit: int = MAX_SANDBOX_PREVIEW) -> str:
    """超长文本截断，保留长度信息"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _one_line(text: str, limit: int = MAX_SANDBOX_PREVIEW) -> str:
    """把多行文本压成一行并截断（日志按行解析，多行会把一条记录拆散）"""
    return _truncate(" | ".join(line for line in text.splitlines() if line.strip()), limit)


# === 状态概览（request_decoded 用） ===


def _base_brief(turn: Turn) -> str:
    """己方基地坐标"""
    station = turn.station()
    if station is None:
        return "-"
    return f"({station.pos.x},{station.pos.y})"


def _hp_brief(turn: Turn) -> str:
    """基地血量（当前/满血）——胜负的第一判据

    报文只给 health，满血值按等级查任务书4.5.1（与 `brain._station_full_health`
    共用同一张表）。
    """
    station = turn.station()
    if station is None:
        return "hp=-"
    return f"hp={station.health}/{station_full_health(station.level)}"


def _robot_brief(turn: Turn) -> str:
    """机器人数量与构成（s/m/l/b 依次为小/中/大/BOSS）"""
    counts = " ".join(
        f"{kind[0]}{sum(1 for robot in turn.robots if robot.kind == kind)}"
        for kind in ROBOT_KINDS
    )
    return f"{len(turn.robots)}({counts})"


def _tower_brief(turn: Turn) -> str:
    """武器明细：座数[型号+等级]，如 3[rocket2,railgun1,gatling1]

    明细内部用逗号分隔（不含空格），保证 `analyze_log.py` 的 `\\S+` 能整段取到。
    """
    weapons = turn.weapons()
    detail = ",".join(f"{weapon.kind}{weapon.level}" for weapon in weapons)
    return f"{len(weapons)}[{detail}]" if detail else "0[]"


def _wall_brief(turn: Turn) -> str:
    """围墙等级分布，如 5[l1:3,l2:2]（逗号分隔，便于按行解析）"""
    walls = turn.walls()
    if not walls:
        return "0[]"
    counts: dict[int, int] = {}
    for wall in walls:
        counts[wall.level] = counts.get(wall.level, 0) + 1
    detail = ",".join(f"l{level}:{count}" for level, count in sorted(counts.items()))
    return f"{len(walls)}[{detail}]"


def _bag_brief(turn: Turn) -> str:
    """可控制角色的背包合并统计，如 bag=stone:6,iron:2（G11）"""
    counts: dict[str, int] = {}
    for unit in turn.controllable():
        for item in unit.backpack:
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return "bag=-"
    detail = ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))
    return f"bag={detail}"


def _enemy_brief(turn: Turn, payload: dict[str, Any]) -> str:
    """敌方可见信息（G1）

    接口文档：敌方只有基地与围墙全图可见，其他单位要进视野才显示，所以
    `enemy_towers=0` 要配合 `enemy_visible` 一起看——是真的没有，还是没看见。
    敌方积分报文里不提供，取不到就打 `-`。
    """
    enemies = [enemy for enemy in turn.enemies if enemy.is_alive]
    towers = sum(1 for enemy in enemies if enemy.kind in TOWER_TYPES)
    walls = sum(1 for enemy in enemies if enemy.kind == WALL)
    score = (payload.get("teamEnemy") or {}).get("totalScore")
    score_text = str(int(score)) if isinstance(score, (int, float)) else "-"
    return (
        f"enemy_score={score_text} enemy_towers={towers} "
        f"enemy_walls={walls} enemy_visible={len(enemies)}"
    )


def _task_brief(turn: Turn) -> str:
    """任务点与当前任务描述（自进化类任务的进度全看这两个字段）"""
    parts = []
    for task in turn.player_tasks:
        state = "可接" if task.is_valid else f"冷却{task.cold_down_rounds}"
        parts.append(
            f"{task.task_type or '任务'}"
            f"@({task.task_position.x},{task.task_position.y})"
            f"{state}/{task.timeout_rounds}回合"
        )
    phase = f'phase="{turn.phase_task}"' if turn.phase_task else "phase=-"
    return f"tasks=[{' '.join(parts) or '-'}] {phase}"


def _failed_brief(turn: Turn, last_commands: dict[str, Any]) -> str:
    """上一回合执行失败的角色 + 该角色当时下的动作（G8 的降级方案）

    判题系统不回失败原因，但把"失败的角色ID"和"上一回合我们给他下的动作"
    拼起来，就能把"堵路/距离不足/目标被抢"的范围缩小到具体动作类型。
    """
    failed = []
    for role_id, success in sorted(turn.last_action_results.items()):
        if success:
            continue
        command = last_commands.get(str(role_id)) or {}
        action = str(command.get("action") or "?")
        failed.append(f"{role_id}:{action}")
    return f"fail=[{' '.join(failed)}]"


def _idle_brief(turn: Turn) -> str:
    """武器空转（G12）：无人操控 vs 有人操控但射程内没目标

    两者性质不同：前者是回防失败（角色没到位），后者是射程覆盖不足。
    """
    controllers = [unit.pos for unit in turn.controllable()]
    idle_man = 0
    idle_target = 0
    for weapon in turn.weapons():
        if not any(distance(pos, weapon.pos) <= 1 for pos in controllers):
            idle_man += 1
            continue
        reach = weapon.range_of_attack()
        if not any(
            robot.is_alive and distance(weapon.pos, robot.pos) <= reach
            for robot in turn.robots
        ):
            idle_target += 1
    return f"idle_man={idle_man} idle_target={idle_target}"


def _role_order(role_id: Any) -> tuple[int, int, str]:
    """角色ID排序键：数字ID按数值排，非数字ID排在后面（日志不该因脏ID报错）"""
    text = str(role_id)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def _command_brief(command: dict[str, Any]) -> str:
    """单条指令的摘要，如 collect→(25,3)、build→(11,22) railgun、submitAnswer"""
    brief = str(command.get("action") or "?")
    targets = command.get("targetPos") or []
    if targets:
        brief += "→" + ",".join(f"({pos['x']},{pos['y']})" for pos in targets)
    if command.get("name"):
        brief += f" {command['name']}"
    if "num" in command:
        brief += f" x{command['num']}"
    if command.get("taskAnswer"):
        # 交卷内容也留一小段（任务执行成败就在这里），全量内容不打印
        brief += f' "{_truncate(str(command["taskAnswer"]), ANSWER_PREVIEW)}"'
    return brief


def _actions_brief(commands: dict[str, dict[str, Any]]) -> str:
    """本回合下发的关键动作（角色ID:动作+目标）"""
    if not commands:
        return "-"
    return " ".join(
        f"{role_id}:{_command_brief(command)}"
        for role_id, command in sorted(commands.items(), key=lambda item: _role_order(item[0]))
    )


def _gold_spent(
    commands: dict[str, dict[str, Any]],
    turn: Turn | None = None,
) -> int:
    """本回合 build/buy 的理论花费（G9）

    按任务书价格表算：建塔 25 金、围墙只花石材不花金币、买道具按
    `weaponShopList`（取不到时用兜底价）。实际扣款以下一回合 `gold=` 差值为准，
    两者不一致就说明有指令被拒。
    """
    total = 0
    for command in commands.values():
        action = command.get("action")
        if action == "build":
            if command.get("name") in TOWER_TYPES:
                total += WEAPON_BUILD_COST
        elif action == "buy":
            name = str(command.get("name") or "")
            price = None
            if turn is not None:
                for item in turn.weapon_shop:
                    if str(item.get("name") or "") == name:
                        raw = item.get("price")
                        if isinstance(raw, (int, float)) and raw >= 0:
                            price = int(raw)
                        break
            if price is None:
                price = ITEM_PRICE_FALLBACK.get(name, 0)
            total += price * int(command.get("num") or 1)
    return total


class _Telemetry:
    """跨回合统计（只写日志，不参与决策）

    `brain` 的决策是无状态的，这里只为"对战分析"攒三样东西：
    **击杀/损失**（对比上一回合的单位集合）、**每日总账**（金币峰值、建造/升级/
    交易次数、任务接取与交卷、积分增长、基地掉血）。
    """

    def __init__(self) -> None:
        self.round_no = 0
        self.is_day = True
        self.robot_ids: set[int] = set()
        self.unit_ids: set[int] = set()
        self.commands: dict[str, Any] = {}
        self.kills = 0
        self.losses = 0
        self.gold_peak = 0
        self.last_gold = 0
        self.last_hp = 0
        self.last_score = 0
        self.day_started = False
        self._reset_day()

    def _reset_day(self) -> None:
        self.builds = 0
        self.upgrades = 0
        self.sells = 0
        self.task_accepts = 0
        self.task_subs = 0
        self.day_gold_start = 0
        self.day_score_start = 0
        self.day_hp_start = 0
        self.day_hp_lost = 0
        self.day_kills = 0
        self.day_losses = 0

    def observe(self, turn: Turn) -> int | None:
        """每个请求开头调用：结算上一回合战果并累计当日数据

        返回:
            上一回合的回合号；首回合（或回合号不连续）时返回 None
        """
        previous = self.round_no
        station = turn.station()
        hp = station.health if station else 0

        robots = {robot.robot_id for robot in turn.robots if robot.is_alive}
        units = {unit.unit_id for unit in turn.ours if unit.is_alive}
        self.kills = 0
        self.losses = 0
        if previous and turn.round_no == previous + 1:
            # 天亮时残余机器人会被系统清场，跨昼夜的差值不是击杀
            if not turn.is_day and not self.is_day:
                self.kills = len(self.robot_ids - robots)
            self.losses = len(self.unit_ids - units)

        if not self.day_started:
            self.day_started = True
            self.day_gold_start = turn.gold
            self.day_score_start = turn.total_score
            self.day_hp_start = hp
            self.gold_peak = turn.gold

        self.gold_peak = max(self.gold_peak, turn.gold)
        self.day_kills += self.kills
        self.day_losses += self.losses
        if self.day_hp_start:
            # 以当天的开局血量为基准，记录当天最大掉血（升级回满不算掉血）
            self.day_hp_lost = max(self.day_hp_lost, self.day_hp_start - hp)

        self.robot_ids = robots
        self.unit_ids = units
        self.last_gold = turn.gold
        self.last_hp = hp
        self.last_score = turn.total_score
        self.round_no = turn.round_no
        self.is_day = turn.is_day
        return previous if previous and turn.round_no == previous + 1 else None

    def note_commands(self, commands: dict[str, dict[str, Any]]) -> None:
        """记录本回合的指令（供当日总账统计）"""
        for command in commands.values():
            action = command.get("action")
            if action == "build":
                self.builds += 1
            elif action == "buy":
                self.upgrades += 1
            elif action == "sell":
                self.sells += 1
            elif action == "acceptTask":
                self.task_accepts += 1
            elif action == "submitAnswer":
                self.task_subs += 1
        self.commands = commands

    def round_end_brief(self, req_id: int, round_no: Any, turn: Turn) -> str:
        """上一回合的战果结算行（G6/G12）"""
        return (
            f"round_end id={req_id} round={round_no} gold={turn.gold} "
            f"{_hp_brief(turn)} towers={_tower_brief(turn)} walls={_wall_brief(turn)} "
            f"kills={self.kills}/{len(turn.robots)} loses={self.losses} "
            f"{_idle_brief(turn)}"
        )

    def day_summary_brief(self, req_id: int, day: int, turn: Turn) -> str:
        """每日总账（模板 §1.2 时间线 / §6 失分点）"""
        return (
            f"day_summary id={req_id} day={day} gold_peak={self.gold_peak} "
            f"gold_final={self.last_gold} "
            f"build={self.builds} upgrade={self.upgrades} sell={self.sells} "
            f"task_accept={self.task_accepts} task_done={self.task_subs} "
            f"score_gain={self.last_score - self.day_score_start} "
            f"hp_lost={self.day_hp_lost} kills={self.day_kills} "
            f"loses={self.day_losses}"
        )

    def start_new_day(self, turn: Turn) -> None:
        """跨过一天边界：把当日总账清零重新累计"""
        self._reset_day()
        self.day_started = True
        self.day_gold_start = turn.gold
        self.day_score_start = turn.total_score
        station = turn.station()
        self.day_hp_start = station.health if station else 0


class Handler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def do_POST(self) -> None:
        """处理POST请求（不区分路径，`/` 与 `/action` 均处理）"""
        global _request_id
        _request_id += 1
        req_id = _request_id
        round_no: Any = "?"
        started = time.perf_counter()

        try:
            # 1. 读取请求体
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)

            # 2. 解析JSON
            payload = json.loads(raw.decode("utf-8"))
            round_no = payload.get("roundNo", 0)
            team_info = payload.get("teamOur", {})
            turn = Turn.load(payload)

            # 2.1 上一回合的战果结算 + 每日总账（都基于当前报文，不依赖缓存决策）
            previous = telemetry.observe(turn)
            if previous is not None:
                LOGGER.info("%s", telemetry.round_end_brief(req_id, previous, turn))
            day_index = (round_no - 1) // ROUNDS_PER_DAY + 1
            if round_no > 1 and (round_no - 1) % ROUNDS_PER_DAY == 0:
                LOGGER.info("%s", telemetry.day_summary_brief(
                    req_id, day_index - 1, turn,
                ))
                telemetry.start_new_day(turn)

            # 3. 记录本回合的资源与任务概览（请求全量报文只在 debug.log 里留档）
            LOGGER.info(
                "request_decoded id=%d round=%d team=%s team_type=%s roles=%d "
                "gold=%d score=%d base=%s towers=%s walls=%s robots=%s mines=%d "
                "%s %s %s %s %s",
                req_id, round_no, team_info.get("teamId", "?"),
                turn.team_type, len(turn.ours),
                turn.gold, turn.total_score, _base_brief(turn),
                _tower_brief(turn), _wall_brief(turn), _robot_brief(turn),
                sum(
                    1 for kind in turn.zones.values()
                    if kind in (STONE_MINE, IRON_MINE, COPPER_MINE)
                ),
                _task_brief(turn), _failed_brief(turn, telemetry.commands),
                _hp_brief(turn), _enemy_brief(turn, payload), _bag_brief(turn),
            )

            # 4. 任务期间的沙盒输出是任务成败的唯一线索，单独留一行预览
            if turn.last_cmd_result:
                LOGGER.info("sandbox_result id=%d round=%d text=%s",
                            req_id, round_no, _one_line(turn.last_cmd_result))

            # 5. 记录完整的格式化请求（DEBUG级别，只写 debug.log）
            LOGGER.debug("=" * 80)
            LOGGER.debug("REQUEST round %d:", round_no)
            LOGGER.debug(json.dumps(payload, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

            # 6. 调用决策引擎（返回指令与可选的LLM prompt）
            response, llm_prompt = decide(payload)

            # 7. 构建完整响应（executeCmd 仅在自进化任务期间有内容）
            sandbox_cmd = sandbox_command(payload)
            full_response = {
                "roleCommandMap": response,
                "prompt": llm_prompt,
                "executeCmd": sandbox_cmd,  # 沙盒命令
            }

            # 8. 记录策略完成（含决策耗时、理论花费与关键动作）
            telemetry.note_commands(response)
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.info("strategy_done id=%d round=%d commands=%d elapsed=%.2fms "
                        "sandbox=%s gold_spent=%d actions=%s",
                        req_id, round_no, len(response), elapsed_ms,
                        "下发" if sandbox_cmd else "空闲",
                        _gold_spent(response, turn), _actions_brief(response))
            if elapsed_ms > DECISION_BUDGET_WARN_MS:
                LOGGER.warning("decision slow at round %s: %.2fms",
                               round_no, elapsed_ms)

            # 9. 编码响应
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

            # 10. 记录完整的格式化响应（DEBUG级别，只写 debug.log）
            LOGGER.debug("RESPONSE round %d:", round_no)
            LOGGER.debug(json.dumps(full_response, ensure_ascii=False, indent=2))
            LOGGER.debug("=" * 80)

        except Exception:
            # 11. 异常处理：返回空指令
            LOGGER.exception("decision failed at round %s", round_no)
            full_response = {
                "roleCommandMap": {},
                "prompt": "",
                "executeCmd": "",
            }
            body = json.dumps(full_response, ensure_ascii=False).encode("utf-8")

        # 12. 发送HTTP响应
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """屏蔽默认的HTTP日志"""
        return


# 跨回合统计（模块级单例；一局比赛一个进程）
telemetry = _Telemetry()


def serve(port: int) -> None:
    """启动HTTP服务器"""
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()
