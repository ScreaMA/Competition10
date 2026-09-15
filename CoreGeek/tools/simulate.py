#!/usr/bin/env python3
"""本地对局模拟器：不用判题系统也能把客户端跑起来看行为。

用法:
    python CoreGeek/tools/simulate.py                    # 跑 1 天（130 回合）
    python CoreGeek/tools/simulate.py --rounds 400
    python CoreGeek/tools/simulate.py --task-ready 10    # 第 10 回合就派发任务

对应设计文档V2 §10.3。

它做三件事：

1. 维护一份**会演进**的地图状态：金币随卖矿增加、机器人按夜出现并被"击杀"、
   任务按接/交/超时结束、基地按机器人数量掉血。
2. 每回合调用 `brain.decide()`（而不是走 HTTP，这样能直接看到日志），
   并校验响应格式。
3. 跑完打印 `analyze_log` 的摘要——**不需要真的打一场比赛**就能看出
   开局建造节奏、金币有没有冻结、任务链路有没有走通。

局限（刻意不做的）：不做真实的移动碰撞与攻击结算，机器人是"按回合掉血"
的近似模型。它的用途是**验证客户端的决策链路与日志**，不是复刻判题器。
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 同目录的 analyze_log


ROUNDS_PER_DAY = 130   # 一天 = 白天 70 回合 + 夜晚 60 回合（任务书 §4.2）
DAY_ROUNDS = 70
NIGHT_ROUNDS = 60

BEIJING_TASK = (
    "请阅读task_1_beijing.md，获取任务信息\n"
    "# 自进化任务 A-1：查询北京文化遗产\n"
    "系统提供了一个 API 服务（运行在 `http://localhost:8899`），"
    "API 文档在 `API_DOCS.md` 中。\n"
    "## 任务要求\n"
    '{"city":"北京","total_count":<总记录条数>,'
    '"world_heritage_count":<保护级别为"世界遗产"的数量>,'
    '"types":["<所有不重复的遗产类型，顺序不限>"],'
    '"oldest_era":"<年代最早的遗产名称>"}\n'
)

# 沙盒回包：模拟"第一次 recon 完成、第二次 query 拿到答案"
RECON_RESULT = (
    "[exitCode:0]\n"
    "[RECON] root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api "
    "task=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md "
    "docs=1 ws= scripts=0 py=1\n"
    "[SCAN] files=4 dirs=1 py=yes sh=no timed_out=no api_calls=0 hits=0\n"
    "[DONE] step=recon elapsed=0.80s root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api\n"
)
ANSWER_RESULT = (
    "[exitCode:0]\n"
    "[QUERY] root=/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api has_task=yes\n"
    "[API] url=http://localhost:8899/api/v1/heritage/search?location=北京&limit=100 "
    "status=200 auth=yes n=42\n"
    "[SCAN] api_calls=1 hits=42 tried=200/api/v1/heritage/search\n"
    "[DATA] n=42 total=42 pages=1\n"
    '[ANSWER] {"city":"北京","total_count":42,"world_heritage_count":6,'
    '"types":["古遗址","古建筑"],"oldest_era":"周口店遗址"}\n'
    "[PROFILE] base=http://localhost:8899 path=/api/v1/heritage/search "
    'param=location auth="Authorization: Bearer heritage-key" target=北京 fields=5\n'
    "[DONE] step=query elapsed=2.10s calls=1 records=42 fields=5\n"
)


def build_base_payload(team: str = "challenger", vendor: bool = True) -> dict:
    """构造一份接近 `docs/request.txt` 的开局报文"""
    roles = [
        {"id": 10013 if team == "challenger" else 20013,
         "pos": {"x": 30, "y": 10}, "roleType": "station", "health": 1500,
         "attackPower": 0, "attackRange": 0, "level": 1,
         "backPackCapability": 0, "backpack": []},
        {"id": 10010 if team == "challenger" else 20010,
         "pos": {"x": 26, "y": 8}, "roleType": "worker", "health": 220,
         "attackPower": 0, "attackRange": 0, "backPackCapability": 100,
         "backpack": []},
        {"id": 10011 if team == "challenger" else 20011,
         "pos": {"x": 27, "y": 6}, "roleType": "pioneer", "health": 200,
         "attackPower": 0, "attackRange": 0, "backPackCapability": 40,
         "backpack": []},
        {"id": 10012 if team == "challenger" else 20012,
         "pos": {"x": 25, "y": 6}, "roleType": "worker", "health": 220,
         "attackPower": 0, "attackRange": 0, "backPackCapability": 100,
         "backpack": []},
    ]
    prefix = "challenger" if team == "challenger" else "defender"
    zones = [
        {"neutralType": "stone", "pos": {"x": 20, "y": 14}},
        {"neutralType": "stone", "pos": {"x": 14, "y": 3}},
        {"neutralType": "iron", "pos": {"x": 25, "y": 22}},
        {"neutralType": "copper", "pos": {"x": 8, "y": 24}},
        *([{"neutralType": "vendor", "pos": {"x": 33, "y": 16}}] if vendor else []),
        {"neutralType": "weaponShop", "pos": {"x": 25, "y": 20}},
        {"neutralType": f"{prefix}TaskPoint1", "pos": {"x": 23, "y": 14}},
        {"neutralType": f"{prefix}TaskPoint2", "pos": {"x": 26, "y": 17}},
        {"neutralType": f"{prefix}TaskPoint2", "pos": {"x": 27, "y": 17}},
    ]
    return {
        "roundNo": 1,
        "mapInfo": {"width": 41, "height": 32, "zones": zones},
        "teamOur": {
            "type": team, "teamId": "sim", "teamName": "sim",
            "goldNum": 75, "totalScore": 0,
            "playerTasks": [
                {"taskType": "自进化类1", "taskPosition": {"x": 23, "y": 14},
                 "coldDownRounds": 0, "scoreReward": 80, "goldReward": 80,
                 "isValid": True, "timeoutRounds": 15},
                {"taskType": "自进化类2", "taskPosition": {"x": 26, "y": 17},
                 "coldDownRounds": 0, "scoreReward": 80, "goldReward": 80,
                 "isValid": True, "timeoutRounds": 15},
            ],
            "roles": roles,
        },
        "teamEnemy": {"roles": [
            {"id": 20013 if team == "challenger" else 10013,
             "pos": {"x": 10, "y": 22}, "roleType": "station", "health": 1500,
             "attackPower": 0, "attackRange": 0, "level": 1},
        ]},
        "robot": {"roles": []},
        "phaseTask": "",
        "lastRoundRoleActionResults": {},
        "lastSummonTreasureResult": 0,
        "llmResp": "",
        "worldNews": {"officialNews": "今日无重大新闻", "folkLegends": "石门三钥"},
        "lastCmdResult": "",
        "vendorShopList": [
            {"name": "stone", "price": 1}, {"name": "iron", "price": 3},
            {"name": "copper", "price": 5},
        ],
        "weaponShopList": [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
            {"name": "WeaponUpgradeVoucher2", "price": 150},
            {"name": "WallUpgradeVoucher1", "price": 20},
            {"name": "StationUpgradeVoucher1", "price": 100},
            {"name": "WallFixer", "price": 10},
            {"name": "Medicine", "price": 10},
        ],
        "errors": [],
    }


class Simulator:
    """一个"判题器"的最小近似

    只结算决策链路**看得到**的那部分状态：金币、建筑、任务、机器人血量。
    不做移动与碰撞的真实结算——那需要复刻整个判题器，不是这个工具的用途。
    """

    def __init__(
        self, rounds: int = 130, team: str = "challenger", vendor: bool = True
    ) -> None:
        self.payload = build_base_payload(team, vendor=vendor)
        self.rounds = rounds
        self.gold = 75
        self.task_ids: dict[int, int] = {}
        self.sandbox_queue: list[str] = []
        self.sandbox_issued = 0
        self.errors = 0
        self.submissions = 0
        self.robot_id = 30000

    # --- 主循环 ---

    def run(self) -> None:
        from agent.brain import decide

        for round_no in range(1, self.rounds + 1):
            payload = self._payload_for(round_no)
            response = decide(payload)
            self._check(response)
            self._apply(round_no, response)

    # --- 报文 ---

    def _payload_for(self, round_no: int) -> dict:
        payload = copy.deepcopy(self.payload)
        payload["roundNo"] = round_no
        payload["teamOur"]["goldNum"] = self.gold
        payload["lastCmdResult"] = (
            self.sandbox_queue.pop(0) if self.sandbox_queue else ""
        )
        if round_no > 1:
            payload["lastRoundRoleActionResults"] = {
                str(unit["id"]): True for unit in payload["teamOur"]["roles"]
            }
        return payload

    # --- 响应校验与结算 ---

    def _check(self, response: dict) -> None:
        """响应必须是合法报文（接口文档 §2）"""
        assert set(response) == {"roleCommandMap", "prompt", "executeCmd"}, response
        assert isinstance(response["executeCmd"], str)
        for key, command in response["roleCommandMap"].items():
            assert isinstance(key, str) and key.isdigit(), key
            assert command.get("action"), command
            if command["action"] == "attack":
                assert command.get("controllerId"), command

    def _apply(self, round_no: int, response: dict) -> None:
        commands = response["roleCommandMap"]
        roles = {str(r["id"]): r for r in self.payload["teamOur"]["roles"]}

        for unit_id, command in commands.items():
            role = roles.get(unit_id)
            if role is None:
                continue
            self._apply_command(round_no, role, command)

        if response["executeCmd"]:
            self._queue_sandbox(round_no)

        # 夜战：机器人生成 / 我方攻击结算 / 机器人推进与啃基地
        self._resolve_night(round_no, response["roleCommandMap"])
        self._spawn_wave(round_no)

        # 任务点冷却每回合递减（任务书 §5：任务结束后 30 回合刷新）
        for task in self.payload["teamOur"]["playerTasks"]:
            task["coldDownRounds"] = max(0, int(task.get("coldDownRounds", 0)) - 1)

        # 每 10 回合发一点钱，模拟采集/贩卖带来的收入
        if round_no % 10 == 0:
            self.gold += 20

    def _apply_command(self, round_no: int, role: dict, command: dict) -> None:
        action = command.get("action")
        role_id = role["id"]
        if action == "move":
            target = command["targetPos"][0]
            if self._walkable(target):
                role["pos"] = {"x": target["x"], "y": target["y"]}
            return
        if action in ("move", "attack", "remove", "drop", "summonTreasure"):
            return
        if action == "use":
            self._apply_use(role, command)
            return
        if action == "build":
            self.gold = max(0, self.gold - (1 if command.get("name") == "wall" else 25))
            name = command.get("name") or "wall"
            target = command["targetPos"][0]
            if name == "wall":
                if not role["backpack"]:
                    return
                role["backpack"].pop()
                self._add_building(40000 + len(self.payload["teamOur"]["roles"]),
                                   target, "wall")
            else:
                self._add_building(10020 + role_id % 100, target, name)
        elif action == "collect":
            spot = command["targetPos"][0]
            role["backpack"] = role["backpack"][:99] + [
                self._mine_kind(spot) or "stone"
            ]
        elif action == "sell":
            price = {"stone": 1, "iron": 3, "copper": 5}.get(command.get("name"), 1)
            count = int(command.get("num") or 1)
            if role["backpack"].count(command.get("name")) >= count:
                for _ in range(count):
                    role["backpack"].remove(command.get("name"))
                self.gold += price * count
        elif action == "buy":
            price = next(
                (i["price"] for i in self.payload["weaponShopList"]
                 if i["name"] == command.get("name")), 0
            )
            if self.gold >= price:
                self.gold -= price
                role["backpack"].append(command.get("name"))
        elif action == "acceptTask":
            self.task_ids[role_id] = round_no
            self.payload["phaseTask"] = BEIJING_TASK
            self.sandbox_issued = 0  # 新任务从"第一次侦察"重新开始
        elif action == "submitAnswer":
            self.submissions += 1
            self.payload["phaseTask"] = ""
            self.payload["teamOur"]["totalScore"] += 80
            self.gold += 80
            self.payload["teamOur"]["playerTasks"] = [
                {**t, "coldDownRounds": 30} for t in self.payload["teamOur"]["playerTasks"]
            ]

    # --- 夜战（近似模型）---

    def _round_in_day(self, round_no: int) -> int:
        return (round_no - 1) % ROUNDS_PER_DAY + 1

    def _is_night(self, round_no: int) -> bool:
        return self._round_in_day(round_no) > DAY_ROUNDS

    def _spawn_wave(self, round_no: int) -> None:
        """白天最后一个回合结束时生成夜里的机器人

        放在这里而不是"夜晚第一回合"，是因为真实判题器**夜晚第一回合的请求里
        就能看到机器人**——晚一回合生成会让客户端那一整个回合白站一晚。
        """
        if self._round_in_day(round_no) != DAY_ROUNDS:
            return
        day = (round_no - 1) // ROUNDS_PER_DAY + 1
        robots = self.payload["robot"]["roles"]
        for index in range(2 + day):
            self.robot_id += 1
            robots.append({
                "id": self.robot_id,
                "pos": {"x": 2 + index, "y": 30},
                "roleType": "smallRobot" if index % 3 else "middleRobot",
                "health": 40 if index % 3 else 60,
                "abnormalState": "",
                "targetTeam": self.payload["teamOur"]["type"],
            })

    def _resolve_night(self, round_no: int, commands: dict) -> None:
        """机器人生成 → 我方攻击结算 → 机器人推进 → 天亮清场

        刻意做得很粗（不判遮挡、不算溅射），目的是让日志里的
        `kills=` / `station_damage=` / `idle_weapon=` 有真实数据可跑，
        而不是复刻判题器。
        """
        roles = self.payload["teamOur"]["roles"]
        robots = self.payload["robot"]["roles"]

        if not self._is_night(round_no):
            if robots:  # 天亮清场（任务书 §4.7.3）
                robots.clear()
            return

        # 我方攻击：落点上的机器人直接判死（近似）
        hit = set()
        for command in commands.values():
            if command.get("action") != "attack":
                continue
            for spot in command.get("targetPos") or []:
                hit.add((spot["x"], spot["y"]))
        robots[:] = [
            r for r in robots
            if (r["pos"]["x"], r["pos"]["y"]) not in hit
        ]

        # 机器人朝基地推进一步；贴到基地就啃
        station = next((r for r in roles if r["roleType"] == "station"), None)
        if station is None:
            return
        base = station["pos"]
        for robot in robots:
            dx = (base["x"] > robot["pos"]["x"]) - (base["x"] < robot["pos"]["x"])
            dy = (base["y"] > robot["pos"]["y"]) - (base["y"] < robot["pos"]["y"])
            robot["pos"] = {"x": robot["pos"]["x"] + dx, "y": robot["pos"]["y"] + dy}
            if max(abs(robot["pos"]["x"] - base["x"]),
                   abs(robot["pos"]["y"] - base["y"])) <= 3:
                station["health"] = max(0, station["health"] - 5)

    def _queue_sandbox(self, round_no: int) -> None:
        """模拟沙盒回包：每个任务的第 1 次下发侦察、第 2 次给出答案

        真实沙盒第一次 recon 只回报目录结构，query 才能拿到答案；这里按同样的
        节奏回包，用来验证"recon → query → submit"这条链路真的走得通。
        """
        self.sandbox_issued += 1
        self.sandbox_queue.append(
            RECON_RESULT if self.sandbox_issued == 1 else ANSWER_RESULT
        )

    def _apply_use(self, role: dict, command: dict) -> None:
        """结算 `use`：升级券 / 修复包 / 药剂

        少了这一步，"买了券但没生效"会被误判成客户端的 bug——客户端其实
        每回合都在正确地 `use→(x,y) <券>`，只是模拟器没结算。
        """
        name = str(command.get("name") or "")
        targets = command.get("targetPos") or []
        spot = targets[0] if targets else None
        level = {"WeaponUpgradeVoucher1": 2, "WeaponUpgradeVoucher2": 3,
                 "WallUpgradeVoucher1": 2, "WallUpgradeVoucher2": 3,
                 "StationUpgradeVoucher1": 2, "StationUpgradeVoucher2": 3}
        kind_of = {"2": {"gatling", "railgun", "rocket", "wall", "station"},
                   "3": {"gatling", "railgun", "rocket", "wall", "station"}}

        if name in level and spot is not None:
            for unit in self.payload["teamOur"]["roles"]:
                if unit["pos"] == spot and unit["roleType"] in kind_of[str(level[name])]:
                    unit["level"] = level[name]
                    if "health" in unit:
                        unit["health"] = {1: 1000, 2: 1500, 3: 2000}.get(
                            level[name], unit["health"]
                        )
        elif name == "Medicine":
            role["health"] = {"worker": 220, "pioneer": 200}.get(
                role["roleType"], role["health"]
            )

        if name in role["backpack"]:
            role["backpack"].remove(name)

    def _add_building(self, unit_id: int, target: dict, kind: str) -> None:
        """把新建成的建筑加到地图上"""
        self.payload["teamOur"]["roles"].append({
            "id": unit_id, "pos": {"x": target["x"], "y": target["y"]},
            "roleType": kind, "health": 1000,
            "attackPower": 10 if kind != "wall" else 0,
            "attackRange": 6 if kind != "wall" else 0,
            "level": 1, "backPackCapability": 0, "backpack": [],
        })

    def _walkable(self, target: dict) -> bool:
        """目标格是不是空地（不是中立元素、在图内）"""
        if not (0 <= target["x"] < 41 and 0 <= target["y"] < 32):
            return False
        spot = {"x": target["x"], "y": target["y"]}
        for zone in self.payload["mapInfo"]["zones"]:
            if zone["pos"] == spot:
                return False
        for unit in self.payload["teamOur"]["roles"]:
            if unit["pos"] == spot:
                return False
        return True

    def _mine_kind(self, spot: dict) -> str | None:
        for zone in self.payload["mapInfo"]["zones"]:
            if zone["pos"] == spot and zone["neutralType"] in ("stone", "iron", "copper"):
                return zone["neutralType"]
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="本地对局模拟")
    parser.add_argument("--rounds", type=int, default=ROUNDS_PER_DAY)
    parser.add_argument("--team", default="challenger",
                        choices=("challenger", "defender"))
    parser.add_argument("--log", default="", help="日志文件（默认打到 stdout）")
    parser.add_argument("--no-vendor", action="store_true",
                        help="地图上不放小贩（复现：没有小贩则矿石卖不掉、金币回不来）")
    args = parser.parse_args()

    # 文件收 DEBUG（含 task_dump 全量任务日志），stdout 只收 INFO；
    # 与 main3.py 的双通道配置一致，模拟出来的日志和真机同构。
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handlers: list[logging.Handler] = []
    if args.log:
        file_handler = logging.FileHandler(args.log, encoding="utf-8", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    handlers.append(stream)
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    started = time.perf_counter()
    simulator = Simulator(
        rounds=args.rounds, team=args.team, vendor=not args.no_vendor
    )
    simulator.run()
    elapsed = time.perf_counter() - started

    print("=" * 60)
    print(f"模拟 {args.rounds} 回合完成，用时 {elapsed:.2f}s"
          f"（平均 {elapsed / args.rounds * 1000:.2f}ms/回合）")
    print(f"最终金币 {simulator.gold}，交卷 {simulator.submissions} 次，"
          f"任务分数 {simulator.payload['teamOur']['totalScore']}")
    if args.log:
        from analyze_log import analyze, print_summary

        stats = analyze(Path(args.log))
        if stats:
            print_summary(stats, Path(args.log))


if __name__ == "__main__":
    main()
