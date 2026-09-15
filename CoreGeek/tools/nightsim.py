#!/usr/bin/env python3
"""夜防模拟器：把"三个角色 + 三座塔 + 一群机器人"跑起来，只看**开火机制**。

用法（在 CoreGeek 目录下）::

    python tools/nightsim.py                     # 默认：三塔三人在基地同一侧
    python tools/nightsim.py --spread            # 一人被隔在基地另一侧（实测形态）
    python tools/nightsim.py --rounds 40 --verbose

它回答的是实测报文里那个问题：**三座塔为什么长期只有两座在开火？**

    R71    attack=0   首夜第一回合，谁都没到位
    R72-79 attack=1   只有火箭就位
    R80+   attack=2   开拓者到位了，却拿不到任何指令（`idle_ids=20011`）
                      ——72 个夜战回合里它只开了 3 次火

与 `simulate.py` 的分工：那个跑**全局**（经济/任务/防守的联动），这个只跑
**夜间开火**这一段，所以能把机器人逼近、角色换位这些变量单独拧出来看。

机器人模型是近似的（朝基地走、被挡就停），不结算伤害——因为要验证的是
"我们有没有把该开的火开出去"，不是"打死了几只"。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import protocol                                    # noqa: E402
from agent.protocol import Pos, Turn, distance                # noqa: E402
from agent.strategy import defense                            # noqa: E402
from agent.world import World                                 # noqa: E402

# 实测报文的战场：基地原点 (30,9)（footprint 占 (30,9)(31,9)(30,10)(31,10)）
BASE = Pos(30, 9)

#: 实测报文里那三座塔：挤在基地左侧一列、两两相邻，朝西（来敌方向）。
#: 这就是 `tower_sites` 现在摆出来的样子——试过"摊开"（打分偏好两两不相邻），
#: 三种来敌方向下稳态都掉到每回合只开 2 炮，所以退回去了：塔要靠人操控，
#: 摆得开不等于守得住，人走过去是要花回合的。
TOWERS = (("rocket", Pos(29, 8)), ("railgun", Pos(29, 9)), ("gatling", Pos(29, 10)))

#: 三人在基地**同一侧**（都贴着塔，理想情况）
CHARS_TIGHT = ((20010, Pos(29, 11)), (20011, Pos(28, 11)), (20012, Pos(30, 8)))

#: 实测形态：开拓者(20011)在基地**另一侧**——它离加特林只有 1 格（所以
#: `manned` 把它算成"在操控"），但按就近配对会被配到西侧那两座塔中的一座，
#: 而基地正好挡在中间。
CHARS_SPREAD = ((20010, Pos(29, 11)), (20011, Pos(31, 11)), (20012, Pos(30, 8)))

ROBOT_KINDS = {"smallRobot": (5, 3, 40), "middleRobot": (10, 3, 60),
               "largeRobot": (20, 3, 500), "bossRobot": (40, 3, 800)}


@dataclass
class Report:
    """逐回合的开火情况"""

    rows: list[tuple[int, int, tuple[str, ...], tuple[int, ...]]] = field(
        default_factory=list
    )

    #: 前多少个回合算"跑位期"——角色从任务点/矿区走回塔边要好几步，
    #: 实测报文里开拓者从任务点回到基地花了 6 个回合
    SETTLE_ROUNDS = 10

    @property
    def steady(self) -> list[tuple[int, int, tuple[str, ...], tuple[int, ...]]]:
        """进入稳态之后的回合（跑位期不算）"""
        return self.rows[self.SETTLE_ROUNDS:]

    def summary(self) -> str:
        if not self.steady:
            return "没有稳态回合"
        counts = [row[1] for row in self.steady]
        return ("稳态每回合开火数：最少 %d / 最多 %d / 众数 %d"
                % (min(counts), max(counts),
                   max(set(counts), key=counts.count)))


class NightSim:
    """一夜的开火推演"""

    def __init__(
        self,
        *,
        base: Pos = BASE,
        towers=TOWERS,
        chars=CHARS_TIGHT,
        robot_count: int = 70,
        robot_origin: Pos = Pos(20, 20),
    ):
        self.base = base
        self.tower_spec = tuple(towers)
        self.char_spec = tuple(chars)
        self.robot_count = robot_count
        self.robot_origin = robot_origin
        self.robot_positions: list[Pos] = []
        self.report = Report()

    # --- 组装 ---

    def _payload(self, round_no: int) -> dict:
        bx, by = self.base.x, self.base.y
        roles = [
            # 基地：`pos` 是左上角，原点 (bx,by) => 左上角 (bx, by+1)
            _role(20013, "station", bx, by + 1, health=1500),
        ]
        for index, (kind, pos) in enumerate(self.tower_spec):
            roles.append(_role(20090 + index, kind, pos.x, pos.y, level=1))
        for unit_id, pos in self.char_spec:
            kind = "pioneer" if unit_id == 20011 else "worker"
            roles.append(_role(unit_id, kind, pos.x, pos.y))

        robots = [
            _robot(30000 + index, pos.x, pos.y)
            for index, pos in enumerate(self.robot_positions)
        ]
        return {
            "roundNo": round_no,
            "mapInfo": {"width": 41, "height": 32, "zones": []},
            "teamOur": {"type": "defender", "teamId": "4334", "teamName": "T",
                        "goldNum": 100, "totalScore": 0, "playerTasks": [],
                        "roles": roles},
            "teamEnemy": {"roles": []},
            "robot": {"roles": robots},
            "phaseTask": "", "lastRoundRoleActionResults": {},
            "lastSummonTreasureResult": 0, "llmResp": "",
            "worldNews": {"officialNews": "", "folkLegends": ""},
            "lastCmdResult": "", "vendorShopList": [], "weaponShopList": [],
            "errors": [],
        }

    def _spawn(self) -> None:
        """机器人从一侧压过来（一排排铺开，朝基地走）"""
        self.robot_positions = []
        per_row = 10
        for index in range(self.robot_count):
            self.robot_positions.append(
                Pos(self.robot_origin.x + index % per_row,
                    self.robot_origin.y + index // per_row)
            )

    def _advance_robots(self, blocked: set[Pos]) -> None:
        """机器人朝基地走一格；被挡住就停在原地（细节无所谓，逼近就行）"""
        moved: list[Pos] = []
        taken = set(self.robot_positions) | blocked
        for pos in self.robot_positions:
            best, best_key = pos, (distance(pos, self.base),)
            for dxy in protocol.NEIGHBOUR_OFFSETS:
                nxt = Pos(pos.x + dxy[0], pos.y + dxy[1])
                if nxt in taken or nxt in blocked:
                    continue
                key = (distance(nxt, self.base), nxt.x, nxt.y)
                if key < best_key:
                    best, best_key = nxt, key
            taken.discard(pos)
            taken.add(best)
            moved.append(best)
        self.robot_positions = moved

    # --- 推演 ---

    def run(self, rounds: int = 30, verbose: bool = False) -> Report:
        self._spawn()
        positions = {unit_id: pos for unit_id, pos in self.char_spec}
        for index, (_, pos) in enumerate(self.tower_spec):
            positions[20090 + index] = pos

        for round_no in range(1, rounds + 1):
            # 把上一回合的移动结果灌回去
            self.char_spec = tuple(
                (unit_id, positions[unit_id]) for unit_id, _ in self.char_spec
            )
            turn = Turn.load(self._payload(round_no))
            world = World.load(turn)
            actions = defense.night_actions(world, frozenset())

            firing = [a for a in actions if a.role == "fire"]
            firing_units = tuple(sorted(str(a.unit.unit_id) for a in firing))
            fired_towers = self._fired_towers(actions, turn)
            idle_units = tuple(
                sorted(u.unit_id for u in turn.characters()
                       if u.unit_id not in {a.unit.unit_id for a in actions})
            )
            self.report.rows.append((round_no, len(firing), fired_towers, idle_units))
            if verbose:
                print("  R%-3d attack=%d towers=%s idle=%s"
                      % (round_no, len(firing), ",".join(fired_towers) or "-",
                         ",".join(str(i) for i in idle_units) or "-"))

            # 应用移动
            for action in actions:
                if action.command and action.command.get("action") == "move":
                    spot = action.command["targetPos"][0]
                    positions[action.unit.unit_id] = Pos(spot["x"], spot["y"])

            blocked = {p for _, p in self.tower_spec}
            blocked |= {Pos(self.base.x + dx, self.base.y + dy)
                        for dx in (0, 1) for dy in (0, 1)}
            blocked |= set(positions.values())
            self._advance_robots(blocked)

        return self.report

    def _fired_towers(self, actions, turn) -> tuple[str, ...]:
        """这一回合**真的开火**的是哪几座塔

        靠 `controllerId` 反查角色站在哪座塔旁边——`manned` 那种"有人在旁边"
        不算数：实测报文里它长期报 3，而实际只有 2 座在开火。
        """
        out = []
        controllers = {a.unit.unit_id: a.unit for a in actions if a.role == "fire"}
        for tower in turn.towers():
            for unit in controllers.values():
                if distance(unit.pos, tower.pos) <= 1:
                    out.append("%s@%d,%d" % (tower.kind, tower.pos.x, tower.pos.y))
                    break
        return tuple(out)


def _role(unit_id: int, kind: str, x: int, y: int, **extra) -> dict:
    role = {
        "id": unit_id, "pos": {"x": x, "y": y}, "roleType": kind,
        "health": {"station": 1500, "pioneer": 200, "worker": 220}.get(kind, 1000),
        "attackPower": 0, "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 40,
        "backpack": [],
    }
    if kind in ("gatling", "railgun", "rocket"):
        # 射程/攻击力按任务书 §4.5.1 的 level1
        role.update({"level": 1,
                     "attackRange": {"gatling": 3, "railgun": 6, "rocket": 10}[kind],
                     "attackPower": {"gatling": 10, "railgun": 10, "rocket": 20}[kind]})
    if kind == "station":
        role["level"] = 1
    role.update(extra)
    return role


def _robot(robot_id: int, x: int, y: int, kind: str = "smallRobot") -> dict:
    return {"id": robot_id, "pos": {"x": x, "y": y}, "roleType": kind,
            "health": ROBOT_KINDS[kind][2], "abnormalState": "",
            "targetTeam": "defender"}


def main() -> int:
    parser = argparse.ArgumentParser(description="夜防开火模拟")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--robots", type=int, default=70)
    parser.add_argument("--spread", action="store_true",
                        help="开拓者在基地另一侧（实测形态）")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    chars = CHARS_SPREAD if args.spread else CHARS_TIGHT
    sim = NightSim(chars=chars, towers=TOWERS, robot_count=args.robots)
    print("布局: %s" % ("开拓者在另一侧（实测）" if args.spread else "三人同侧"))
    print("塔: %s" % " ".join("%s@%d,%d" % (k, p.x, p.y) for k, p in sim.tower_spec))
    print("人: %s" % " ".join("%d@%d,%d" % (i, p.x, p.y) for i, p in chars))
    report = sim.run(args.rounds, args.verbose)

    print()
    print(report.summary())
    print()
    print("  R    attack  开火的塔                          拿不到指令的角色")
    for row in report.rows:
        if not row[1] and row[0] > 5:
            continue
        if row[0] % 5 == 0 or row[0] <= 3:
            print("  %-4d %-7d %-32s %s"
                  % (row[0], row[1], ",".join(row[2]) or "-",
                     ",".join(str(i) for i in row[3]) or "-"))

    steady = [r[1] for r in report.steady]
    ok = bool(steady) and min(steady) >= 3
    print()
    print("结论: 稳态 %s ⇒ %s"
          % (sorted(set(steady)), "三塔都在开火" if ok else "**有塔没开火**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
