"""防御策略用例：塔位可达性、墙线、三种武器的目标选择、夜间配位。

对应设计文档V2 第 8 章与 §7.4。
"""

from __future__ import annotations

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
