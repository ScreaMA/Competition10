"""防御策略用例：塔位可达性、墙线、三种武器的目标选择、夜间配位。

对应设计文档V2 第 8 章与 §7.4。
"""

from __future__ import annotations

import pytest

from agent.protocol import Pos, Turn
from agent.strategy import defense
from agent.world import World


def _world(payload_factory, role_factory, base=(20, 10), **kwargs):
    roles = [role_factory(10013, "station", *base)]
    roles.extend(kwargs.pop("roles", []))
    roles.extend(kwargs.pop("towers", []))
    return World.load(Turn.load(payload_factory(roles=roles, **kwargs)))


def _tower(role_factory, unit_id, kind, x, y, level=1, **kw):
    return role_factory(unit_id, kind, x, y, level=level, **kw)


# ==========================================================================
# 塔位与可达性（战术参考 T2）
# ==========================================================================


def test_tower_sites_are_within_reach_of_base(payload_factory, role_factory):
    world = _world(payload_factory, role_factory)
    sites = defense.tower_sites(world)
    assert 1 <= len(sites) <= 3
    for site in sites:
        assert world.turn.is_land(site)
        assert site not in world.turn.occupied()


def test_tower_sites_empty_when_full(payload_factory, role_factory):
    """三座塔齐了就不再建（任务书 §4.5.1：全局同时最多 3 座）"""
    world = _world(
        payload_factory, role_factory,
        towers=[
            _tower(role_factory, 10020, "gatling", 19, 8),
            _tower(role_factory, 10030, "railgun", 21, 8),
            _tower(role_factory, 10040, "rocket", 19, 11),
        ],
    )
    assert defense.tower_sites(world) == ()


def test_layout_ok_detects_blocked_tower(payload_factory, role_factory):
    """建成预演：塔在隔离区里（没人走得到它旁边）时返回 False

    这是 `战术参考/修改建议.md` 的 T2——聊天记录里"炮塔把路堵住然后有一个
    炮塔碰不到"的坑，是唯一"会导致整局失效"级的问题：塔在，但没人能操控。
    """
    turn = Turn.load(payload_factory(roles=[role_factory(10013, "station", 20, 10)]))
    # 一条横贯全图的封锁线（y=20），把基地（下方）与塔（上方）彻底隔开
    barrier = tuple(Pos(x, 20) for x in range(0, 41))
    isolated = Pos(20, 25)
    assert defense.layout_ok(turn, barrier + (isolated,)) is False
    # 没有那条封锁线时同一座塔是可达的
    assert defense.layout_ok(turn, (isolated,)) is True


def test_layout_ok_passes_for_open_layout(payload_factory, role_factory):
    turn = Turn.load(payload_factory(roles=[role_factory(10013, "station", 20, 10)]))
    assert defense.layout_ok(turn, (Pos(18, 10), Pos(22, 10), Pos(20, 13))) is True


def test_next_tower_type_rebuilds_the_destroyed_kind(payload_factory, role_factory):
    """按 `TOWER_LOADOUT` 补缺：火箭被打掉后补的仍是火箭（射程 10 优先级最高）"""
    world = _world(
        payload_factory, role_factory,
        towers=[
            _tower(role_factory, 10020, "gatling", 19, 8),
            _tower(role_factory, 10030, "railgun", 21, 8),
        ],
    )
    assert defense.next_tower_type(world.turn) == "rocket"


def test_tower_sites_prefer_enemy_side(payload_factory, role_factory):
    """候选塔位优先朝敌方来向（用可见的敌方基地推算）"""
    world = _world(
        payload_factory, role_factory, base=(20, 10),
        enemies=[role_factory(20013, "station", 38, 10)],  # 东边
    )
    sites = defense.tower_sites(world)
    assert sites
    # 第一个候选应该落在基地右侧
    assert sites[0].x > world.origin.x


# ==========================================================================
# 墙线
# ==========================================================================


def test_wall_sites_are_on_second_ring(payload_factory, role_factory):
    """围墙取基地周围第二圈（第一圈会把基地围死、角色出不去）"""
    world = _world(payload_factory, role_factory)
    sites = defense.wall_sites(world)
    assert sites
    for site in sites:
        offset = world.offset_of(site)
        assert max(abs(offset.x), abs(offset.y)) == 2


def test_wall_sites_skip_existing(payload_factory, role_factory):
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(40000, "wall", 18, 8)],
    )
    assert Pos(18, 8) not in defense.wall_sites(world)


# ==========================================================================
# 三种武器的目标选择
# ==========================================================================


def _armed_turn(payload_factory, role_factory, robot_factory, kind, level, robots,
                **tower_kw):
    tower = _tower(role_factory, 10040, kind, 20, 10, level=level, **tower_kw)
    return Turn.load(payload_factory(
        roles=[role_factory(10013, "station", 25, 25), tower],
        robots=[robot_factory(i, x, y, k) for i, (x, y, k) in enumerate(robots)],
    ))


def test_rocket_targets_count_equals_level(payload_factory, role_factory, robot_factory):
    turn = _armed_turn(payload_factory, role_factory, robot_factory, "rocket", 3,
                       [(15, 10, "smallRobot"), (25, 10, "middleRobot"),
                        (20, 15, "largeRobot")])
    tower = turn.towers()[0]
    targets = defense.fire_targets(turn, tower)
    assert len(targets) == 3  # 导弹枚数 = 等级（接口文档 §2.2）


def test_rocket_in_cooldown_does_not_fire(payload_factory, role_factory, robot_factory):
    """火箭发射后进入 3 回合冷却（任务书 §4.5.4）"""
    turn = _armed_turn(payload_factory, role_factory, robot_factory, "rocket", 1,
                       [(15, 10, "smallRobot")], cooldown=2)
    tower = turn.towers()[0]
    assert tower.cooldown == 2
    assert defense.fire_targets(turn, tower) == ()


def test_gatling_targets_stay_in_cone(payload_factory, role_factory, robot_factory):
    """加特林多目标必须在同一 90° 锥内，否则整次攻击非法"""
    from agent import grid

    turn = _armed_turn(payload_factory, role_factory, robot_factory, "gatling", 3,
                       [(23, 10, "smallRobot"), (23, 11, "middleRobot"),
                        (23, 12, "largeRobot"), (13, 10, "bossRobot")])
    tower = turn.towers()[0]
    targets = defense.fire_targets(turn, tower)
    assert len(targets) == 3
    anchor = targets[0]
    for other in targets[1:]:
        assert grid.within_cone(tower.pos, anchor, other, 90)


def test_gatling_pads_with_anchor_when_not_enough_targets(
    payload_factory, role_factory, robot_factory
):
    """目标不够时用锚点补齐：重复落点只是"打同一个地方"，不是指令错误"""
    turn = _armed_turn(payload_factory, role_factory, robot_factory, "gatling", 3,
                       [(23, 10, "smallRobot")])
    targets = defense.fire_targets(turn, turn.towers()[0])
    assert len(targets) == 3
    assert targets[0] == targets[1] == targets[2]


def test_railgun_picks_piercing_line(payload_factory, role_factory, robot_factory):
    """电磁狙击炮选"弹道上伤害总量最大"的落点（穿透机制）

    穿透只有在"能量还没耗尽"时才有意义（任务书 §4.5.4：能量按造成的伤害量
    扣减），所以这里给机器人 5 点血——10 点能量足以穿透两台。
    """
    turn = Turn.load(payload_factory(
        roles=[role_factory(10013, "station", 25, 25),
               _tower(role_factory, 10040, "railgun", 20, 10, level=1)],
        robots=[
            robot_factory(1, 26, 10, "smallRobot", health=5),
            robot_factory(2, 24, 10, "smallRobot", health=5),
            robot_factory(3, 15, 14, "smallRobot", health=5),
        ],
    ))
    targets = defense.fire_targets(turn, turn.towers()[0])
    assert len(targets) == 1
    # 正东方向 (24,10)(26,10) 排着两台，一条弹道打两个；其他方向只有一台
    assert targets[0] == Pos(26, 10)


def test_no_targets_returns_empty(payload_factory, role_factory, robot_factory):
    turn = _armed_turn(payload_factory, role_factory, robot_factory, "gatling", 1, [])
    assert defense.fire_targets(turn, turn.towers()[0]) == ()


# ==========================================================================
# 夜间配位
# ==========================================================================


def test_night_actions_fire_when_controller_adjacent(
    payload_factory, role_factory, robot_factory
):
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10010, "worker", 19, 10)],
        towers=[_tower(role_factory, 10020, "gatling", 20, 10)],
        robots=[robot_factory(1, 22, 10, "smallRobot")],
    )
    actions = defense.night_actions(world)
    fires = [a for a in actions if a.role == "fire"]
    assert len(fires) == 1
    assert fires[0].command["action"] == "attack"
    assert fires[0].command["controllerId"] == "10010"


def test_night_actions_approach_when_far(payload_factory, role_factory, robot_factory):
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10010, "worker", 30, 30)],
        towers=[_tower(role_factory, 10020, "gatling", 20, 10)],
        robots=[robot_factory(1, 22, 10, "smallRobot")],
    )
    actions = defense.night_actions(world)
    assert actions[0].role == "approach"
    assert actions[0].command["action"] == "move"


def test_night_pairing_prefers_already_positioned(
    payload_factory, role_factory, robot_factory
):
    """已经在武器旁边的角色不该被换走（避免每晚重新跑位）"""
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10010, "worker", 19, 10),
               role_factory(10012, "worker", 35, 35)],
        towers=[_tower(role_factory, 10020, "gatling", 20, 10)],
        robots=[robot_factory(1, 22, 10, "smallRobot")],
    )
    actions = {a.unit.unit_id: a for a in defense.night_actions(world)}
    assert actions[10010].role == "fire"


def test_excluded_units_do_not_get_weapons(
    payload_factory, role_factory, robot_factory
):
    """任务中的开拓者不参与夜间配位（离开任务点即任务结束）"""
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10011, "pioneer", 19, 10)],
        towers=[_tower(role_factory, 10020, "gatling", 20, 10)],
        robots=[robot_factory(1, 22, 10, "smallRobot")],
    )
    actions = defense.night_actions(world, exclude=frozenset({10011}))
    assert all(a.unit.unit_id != 10011 for a in actions)


# ==========================================================================
# 天黑前回防与换塔（"角色没在操作炮塔清理小怪"的两条根因）
# ==========================================================================


def _gun(role_factory, unit_id, x, y, kind="gatling", level=1, **kw):
    return role_factory(unit_id, kind, x, y, level=level,
                        attackRange=kw.pop("attackRange", 6), **kw)


def test_rounds_until_night(payload_factory, role_factory):
    """白天第 1 回合还剩 70 回合，最后 1 回合还剩 1，夜晚是 0"""
    def left(round_no):
        turn = Turn.load(payload_factory(
            round_no=round_no, roles=[role_factory(10013, "station", 20, 10)]))
        return defense.rounds_until_night(turn)

    assert left(1) == 70
    assert left(70) == 1
    assert left(71) == 0
    assert left(130) == 0


def test_dusk_recall_sends_far_worker_home(payload_factory, role_factory):
    """天快黑且路不够走 ⇒ 放下手里的活先回塔位

    回归："角色没在操作炮塔清理小怪"最直接的一条根因——白天角色在十几格外的
    矿区，**就位逻辑是天黑之后才启动的**，等它走回来塔已经空了好几个回合。
    """
    world = _world(
        payload_factory, role_factory, round_no=65,
        roles=[role_factory(10010, "worker", 5, 5)],
        towers=[_gun(role_factory, 10020, 20, 12)],
    )
    command = defense.dusk_recall(world, world.turn.workers()[0], set())
    assert command is not None and command["action"] == "move"


def test_dusk_recall_leaves_nearby_worker_alone(payload_factory, role_factory):
    """路够走就别提前收工——角色该干活干活"""
    world = _world(
        payload_factory, role_factory, round_no=30,
        roles=[role_factory(10010, "worker", 5, 5)],
        towers=[_gun(role_factory, 10020, 20, 12)],
    )
    assert defense.dusk_recall(world, world.turn.workers()[0], set()) is None


def test_dusk_recall_does_nothing_at_night(payload_factory, role_factory):
    world = _world(
        payload_factory, role_factory, round_no=75,
        roles=[role_factory(10010, "worker", 5, 5)],
        towers=[_gun(role_factory, 10020, 20, 12)],
    )
    assert defense.dusk_recall(world, world.turn.workers()[0], set()) is None


def test_controller_switches_to_tower_with_targets(
    payload_factory, role_factory, robot_factory
):
    """自己的塔够不着时，去换一座**有目标**的塔，别守一整晚

    配位是按距离硬配的：站在加特林（射程 3）旁边的角色，哪怕火箭（射程 10）
    那边正有目标，也只会站着不动——日志上就是 `idle_target` 长期 > 0。
    """
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10010, "worker", 13, 12)],   # 贴着加特林
        towers=[
            _gun(role_factory, 10020, 14, 12, attackRange=2),   # 够不着
            _gun(role_factory, 10040, 10, 12, kind="rocket", attackRange=20),  # 够得着
        ],
        robots=[robot_factory(30001, 6, 12, "smallRobot")],
    )
    actions = {a.unit.unit_id: a for a in defense.night_actions(world)}
    action = actions[10010]
    assert action.role == "switch", action.role
    assert action.command["action"] == "move"


def test_controller_fires_when_own_tower_has_targets(
    payload_factory, role_factory, robot_factory
):
    world = _world(
        payload_factory, role_factory,
        roles=[role_factory(10010, "worker", 13, 12)],
        towers=[_gun(role_factory, 10020, 14, 12, attackRange=6)],
        robots=[robot_factory(30001, 16, 12, "smallRobot")],
    )
    actions = {a.unit.unit_id: a for a in defense.night_actions(world)}
    assert actions[10010].role == "fire"
    assert actions[10010].command["action"] == "attack"


# ==========================================================================
# 三人三塔：配对要看可达性（实测 2.log R71–R212）
# ==========================================================================
#
# 实测报文里这条链路的形态：基地原点 (30,9)（footprint 占 (30,9)(31,9)(30,10)(31,10)），
# 三座塔挤在基地左侧一列 (29,8)(29,9)(29,10)，三个人分别在
#   (29,11) 加特林旁边、(30,11) 基地东南角、(30,8) 火箭旁边。
# 结果是**三座塔长期只有两座在开火**：开拓者在 (30,11) 待了 70+ 个回合没拿到
# 任何指令（`idle_ids=20011`），72 个夜战回合只开了 3 次火；首夜基地被打掉
# 1415 血、三塔全毁。


def _war(payload_factory, role_factory, robot_factory, *, chars, robots):
    """复刻实测战场：基地 (30,9) + 左列三塔 + 给定位置的三个人"""
    return _world(
        payload_factory, role_factory, base=(30, 10),
        towers=[
            _tower(role_factory, 20090, "rocket", 29, 8),
            _tower(role_factory, 20091, "railgun", 29, 9),
            _tower(role_factory, 20092, "gatling", 29, 10),
        ],
        roles=[
            role_factory(uid, "pioneer" if uid == 20011 else "worker", x, y)
            for uid, (x, y) in chars
        ],
        robots=[robot_factory(30000 + i, x, y) for i, (x, y) in enumerate(robots)],
    )


def _paired(world):
    return {
        c.unit_id: w
        for w, c in defense._pair(
            world, list(world.turn.characters()), list(world.turn.towers())
        )
    }


def test_pairing_prefers_a_tower_the_controller_can_reach(
    payload_factory, role_factory, robot_factory
):
    """配对**先看够不够得着**，再看离得近不近

    只按距离贪心（平局用 id 兜底）会先把加特林派给 id 更小的工人，再把基地
    **另一侧**那两座塔之一派给开拓者。而开拓者这时已经被机器人封在基地东南角
    ——它唯一的落脚点就是身旁那座加特林。配对一旦配错，它整晚都动不了。
    """
    # 把开拓者之外的每一条出路都堵上：它只能留在 (30,11)
    world = _war(
        payload_factory, role_factory, robot_factory,
        chars=[(20010, (29, 11)), (20011, (30, 11)), (20012, (30, 8))],
        robots=[(29, 12), (30, 12), (31, 11), (31, 12)],
    )
    paired = _paired(world)
    assert paired[20011].kind == "gatling", (
        "开拓者被配到了走不过去的塔：%s" % paired[20011].kind
    )
    # 工人 (29,11) 在基地西侧，西侧那两座它都走得过去
    assert paired[20010].kind in ("rocket", "railgun"), paired[20010].kind


def test_unreachable_controller_switches_instead_of_standing_still(
    payload_factory, role_factory, robot_factory
):
    """配到走不过去的塔时要**换一座**，不是原地站着

    实测里缺的就是这条：开拓者在基地东侧、被配到西侧的塔，绕行要 6 步，
    机器人一压过来路就断了 —— 72 个夜战回合只开了 3 次火。
    """
    # 塔旁边留一个够得着的目标，但把开拓者封在东南角
    world = _war(
        payload_factory, role_factory, robot_factory,
        chars=[(20010, (29, 11)), (20011, (30, 11)), (20012, (30, 8))],
        robots=[(29, 12), (30, 12), (31, 11), (31, 12), (27, 6)],
    )
    actions = {a.unit.unit_id: a for a in defense.night_actions(world)}
    action = actions[20011]
    assert action.command is not None, "开拓者被空转了"
    # 要么直接开火（够得着的那座），要么往它走——总之不能站着
    assert action.role in ("fire", "switch", "approach"), action.role


def test_three_towers_all_fire_in_steady_state(
    payload_factory, role_factory, robot_factory
):
    """三人三塔、机器人压过来时，**三座塔都要开火**

    这是实测报文里最刺眼的一条：`manned=3` 长期报 3，而实际只有 2 座在开火
    ——`manned` 数的是"有人站在旁边"，不是"这座塔打了"。
    """
    import nightsim

    sim = nightsim.NightSim(chars=nightsim.CHARS_SPREAD, robot_count=70)
    report = sim.run(rounds=25)
    steady = [row[1] for row in report.steady]
    assert steady, report.summary()
    assert min(steady) >= 3, (
        "稳态里有回合没打满三座：%s（%s）" % (sorted(set(steady)), report.summary())
    )


# ==========================================================================
# 围墙不能把自己人封死（实测 2.log：八段墙铺满三座塔西侧的全部落脚点）
# ==========================================================================


def _wall_world(payload_factory, role_factory, base=(30, 10)):
    """基地 + 三座塔（用 `tower_sites` 选出来的真实布局）"""
    world = _world(payload_factory, role_factory, base=base)
    sites = defense.tower_sites(world)
    roles = [role_factory(20090 + i, kind, p.x, p.y, level=1,
                          attackRange=6, attackPower=10)
             for i, (kind, p) in enumerate(zip(("rocket", "railgun", "gatling"), sites))]
    return _world(payload_factory, role_factory, base=base, towers=roles), sites


def test_walls_never_sit_on_a_building(payload_factory, role_factory):
    """围墙不能建在已经占用的格子上（**守不变式**，不是当前布局的回归）

    旧实现只跳过"已有的墙"和基地 footprint，**不管塔**——塔在 `ours` 里、
    不在 `mapInfo.zones` 里，只查 `zones` 会以为那格是空地。当前这套塔位恰好
    没撞上（所以这条在修复前也是绿的），但塔位一旦变化就会撞上：验"把三座塔摊开"
    那一版时，选出来的墙位里就有一格正压在炮台上。
    """
    world, _ = _wall_world(payload_factory, role_factory)
    walls = defense.wall_sites(world)
    assert walls, "没有给出任何墙位"
    blocked = set()
    for unit in world.turn.ours:
        if unit.is_alive and unit.kind not in ("worker", "pioneer"):
            blocked |= set(world.turn.footprint(unit))
    assert not (set(walls) & blocked), sorted(
        (p.x, p.y) for p in set(walls) & blocked
    )


def test_walls_leave_every_tower_maneuverable(payload_factory, role_factory):
    """八段墙全部建起来之后，**每座塔都还有落脚点**

    实测报文里八段墙正好铺在 `(28,8)(28,9)(28,10)`——那是三座塔（x=29）西侧的
    全部落脚点，railgun 的可达落脚点直接归零：塔从西面彻底够不着，角色只能从
    北/南绕，机器人一压就断（`2.log` R81 开拓者被闷在 `(30,11)`，72 个夜战
    回合只开了 3 次火）。

    判据是"至少留 `WALL_KEEP_STANDS` 个"而不是"有就行"：只留一格的话，
    一台机器人走过去堵上，这座塔整晚就哑了。
    """
    from agent import grid

    # `getattr` 兜底：这条用例的价值在于"修好之前会红"，所以判据要写成
    # 独立于被测常量——不然旧代码上是 `AttributeError` 而不是断言失败，
    # 看上去像用例自己坏了。
    keep = getattr(defense, "WALL_KEEP_STANDS", 2)

    world, sites = _wall_world(payload_factory, role_factory)
    assert len(sites) == 3
    walls = defense.wall_sites(world)
    assert len(walls) == defense.WALL_TARGET_SEGMENTS

    roles = [role_factory(30000 + i, "wall", p.x, p.y, level=1)
             for i, p in enumerate(walls)]
    built = _world(payload_factory, role_factory, base=(30, 10),
                   towers=[role_factory(20090 + i, kind, p.x, p.y, level=1,
                                        attackRange=6, attackPower=10)
                           for i, (kind, p) in enumerate(zip(
                               ("rocket", "railgun", "gatling"), sites))],
                   roles=roles)
    turn = built.turn
    blocked = set()
    for unit in turn.ours:
        if unit.is_alive and unit.kind not in ("worker", "pioneer"):
            blocked |= set(turn.footprint(unit))
    station = turn.station()
    reach = grid.reachable_set(turn, [station.pos], blocked, limit=25)

    for tower in turn.towers():
        stands = grid.stand_cells(turn, tower.pos)
        left = [c for c in stands if c in reach]
        assert len(left) >= keep, (
            "%s@%d,%d 只剩 %d 个落脚点（%s）"
            % (tower.kind, tower.pos.x, tower.pos.y, len(left),
               [(c.x, c.y) for c in left])
        )


# ==========================================================================
# 塔位必须朝**主轴**（敌人的出生点在斜对角，横竖两维都会命中）
# ==========================================================================


def _side_of_offset(pos: Pos, origin: Pos) -> str:
    dx, dy = pos.x - origin.x, pos.y - origin.y
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "up" if dy > 0 else "down"


@pytest.mark.parametrize(
    "enemy", [(8, 22), (8, 4), (20, 24), (36, 14), (38, 4), (20, 4), (2, 30)],
)
def test_towers_face_the_dominant_enemy_axis(
    payload_factory, role_factory, enemy
):
    """塔位要压在**主轴**那一侧，不能摆到次要方向去

    两个出生基地在地图对角（左上 vs 右下），敌基地几乎总是斜的：横竖两维都
    算"来敌方向"。只把它们当成等价，谁排前面就由环数、坐标这些与敌情无关的
    因素决定——实测 10 个方位里有 4 个把塔摆到了次要那一边。

    这条同时钉住"**方向是推出来的、不是写死的**"：换一套敌我位置，塔位自动
    跟着翻。
    """
    origin = Pos(30, 9)                     # station 左上角 (30,10)
    world = _world(
        payload_factory, role_factory, base=(30, 10),
        enemies=[role_factory(9001, "station", *enemy)],
    )
    dom = _side_of_offset(Pos(*enemy), Pos(origin.x, origin.y + 1))
    sites = defense.tower_sites(world)
    assert sites, "没有给出塔位"
    sides = {_side_of_offset(p, origin) for p in sites}
    assert dom in sides, "主轴是 %s，塔却摆到了 %s" % (dom, sorted(sides))
