"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。

策略概览：
    白天：工人优先建造武器工事（火箭发射台/电磁狙击炮/加特林，射程优先），
          再采集石头建造围墙（先封敌方来路那一侧）；
          围墙建完后用富余资源换取金币和武器升级（金币阶梯：
          武器升级券 > 围墙升级券，不让金币在手里睡着）。
          开拓者优先完成自进化类任务（任务点领取 + 沙盒作答，
          描述里没给文件名时先在任务根目录里探测沙盒、认不出再退到全盘，
          顺带把任务文件都读回来
          缓存备用，下一个任务点就能即时交卷），
          有任务在身时不退回基地，任务点冷却期间白天也守在下一个会开放的
          任务点旁等它开放（省掉"回基地再折返"的来回），天黑前再回防；
          天黑前工人回防到武器旁，但火力/围墙不达标时先抢建，
          角色不会整回合空转。
    夜晚：每个角色操控一座武器攻击机器人，优先攻击威胁最高的目标；
          射程内没有目标、或者这一回合没摊上武器的角色去堵围墙缺口
          （石头砌不起墙时用身体堵，见 `_plug_wall_gap`）。

本模块为无状态决策：每回合从 `Turn` 重新解析地图与单位状态，
不依赖任何跨回合的战场状态，可自动适应矿区刷新、单位移动与视野变化。
任务答案同理，只认执行器按任务描述在沙盒里取到的数据（`[SOLUTION]` 段，
见 `_task_answer`），取不到数据就不提交；解析不到时再退到纯缓存
`_TASK_ANSWER_CACHE`（内容全部来自执行器的产出），未命中就走原来的
执行流程；LLM 建议也从请求里的 `llmResp` 现解析成有界计划（`_llm_plan`），
建议与指令出自同一套决策函数。
任务看门狗（`_TASK_WATCH`）同样只保留最近一回合的观察值（任务标识 + 回合号
+ 沙盒输出），回合号不连续就从头计数，因此它描述的是"当前这一局这个任务"；
沙盒反复回读同一份文件、任务超时交不上卷、同一份答案交满三次仍没被
Judge 放行、或者连续几个回合取数全失败（404 循环）时由它止损
（见 `_task_abandoned`），不再让开拓者被一个拿不到答案的任务永久占死。
"""

import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .grid import next_step, get_neighbors, cells_in_range
from .protocol import (
    Turn,
    Unit,
    PlayerTask,
    Pos,
    distance,
    # 单位类型
    WORKER,
    PIONEER,
    STATION,
    LAND,
    GATLING,
    RAILGUN,
    ROCKET,
    WALL,
    TOWER_TYPES,
    # 建筑满血表（任务书4.5.1，与日志共用）
    station_full_health,
    wall_full_health,
    # 矿石
    STONE_MINE,
    IRON_MINE,
    COPPER_MINE,
    WALL_MATERIAL,
    # 中立单位
    VENDOR,
    WEAPON_SHOP,
    # 任务点
    CHALLENGER_TASK_1,
    CHALLENGER_TASK_2,
    DEFENDER_TASK_1,
    DEFENDER_TASK_2,
    # 时间
    DAY_ROUNDS,
    ROUNDS_PER_DAY,
    # 建造成本
    WEAPON_BUILD_COST,
    # 指令构建
    move_command,
    collect_command,
    build_command,
    remove_command,
    attack_command,
    sell_command,
    buy_command,
    use_command,
    accept_task_command,
    submit_answer_command,
    station_footprint,
)

# 策略常量
# 武器建造顺序：射程优先（火箭 10 > 电磁狙击炮 6 > 加特林 3）。
# 复盘里敌方开局就建射程 10 的火箭发射台，我方却先建射程 3 的加特林，
# 机器人一路走到基地跟前才开始挨打；塔位本来就按方位分散（`_calc_tower_sites`），
# 让射程最远的先落地，等于把敌方来路更早罩进火力网——复盘的原话就是
# "武器优先级表（rocket 优先）…避免低射程 gatling"。
# 塔型与塔位在这里一一绑定（`_worker_day_logic` 按下标取），
# "说建哪座塔"与"建的是哪种武器"因此不会再脱节。
#
# 战术参考（T1，已完成仅存档）：聊天记录建议"2 导弹 1 电磁 或 3 导弹"，
# 但同型号能否重复建造待确认（任务书只写"自由选择搭配"，没有例子），
# 所以当前仍是三种各一座、按射程排序。确认可重复建造后改成
# `(ROCKET, ROCKET, RAILGUN)` 即可。
TOWER_LOADOUT = (ROCKET, RAILGUN, GATLING)  # 武器建造顺序（按射程由远及近）
STONE_BATCH = 3  # 工人采集石头的批次大小（越小围墙越早开工）
WALL_BUILD_PRIORITY = 1000  # 围墙建造优先级
SELL_BATCH = 10  # 卖给小贩的矿石批次大小
# 石材的零散出售下限（S2）：围墙配额铺满、武器塔也建满之后，背包里的石材再攒
# 批就没有意义了——攒着的结果是金币继续躺着（复盘里 stone 从 1 块堆到 3 块、
# 金币从 R6 起恒 0 到 R17，工人背着石头空转）。这时攒够 SELL_THRESHOLD 块就
# 顺路卖给小贩；还有塔/墙要建时石材照旧留给建造（见 `_build_backlog`）。
SELL_THRESHOLD = 2
# 纯收入矿石（铁/铜）的出售下限（S2）：铁/铜不参与砌墙，攒批没有任何好处，
# 手里有一块就变现一块。石材另算——它既是收入也是围墙材料，攒够一批再卖
# 更省回合（见 `_trade_logic`）。复盘里防守方的背包一直堆着 iron/copper、
# 金币却从 R6/R8 起冻结到 R17，就是"只采不卖"卡在攒批上。
MINERAL_SELL_THRESHOLD = 1
# 一段围墙只要石头*1（任务书4.5.1），所以手里有石头就该立刻变成墙，
# 不必攒够 STONE_BATCH 那么多段再一起铺。
WALL_STONE_COST = 1

# 金币见底线（经济回路）：低于这个值时不再等凑够一批矿石，手里有什么就卖什么。
# 复盘里 R6 建完第三座塔后 gold=0 冻结 13 个回合，背包里 stone:6 既不卖也不
# 建墙，全盘停摆——建造、升级、买券全部停摆，只能靠卖矿把金币重新转起来。
LOW_GOLD_THRESHOLD = WEAPON_BUILD_COST
# 建造分支的预算下限：余额不足一座塔的钱时不下发建造指令（`_gold_left` 算的是
# 扣掉本回合已发指令后的余额），免得同一回合里两条 build 只有一条能结算成功。
MIN_GOLD_FOR_BUILD = WEAPON_BUILD_COST

# 建筑满血表在 protocol.py（日志与决策共用，任务书 4.5.1 节）：
# 基地 1500/3000/4500，围墙 1000/1500/2000。判断"残血"靠这张表反查。

# 基地升级券（战术参考 T3）：商店售价 100/150 金，
# "基地快没血了给基地用一下升级券，这样血可以回满还顺便升级了"——
# 等于用一张券同时买到"满血复活"和"更高的血量上限"，性价比极高。
STATION_UPGRADE_VOUCHER = "StationUpgradeVoucher1"   # level1->level2
STATION_UPGRADE_VOUCHER2 = "StationUpgradeVoucher2"  # level2->level3
STATION_UPGRADE_GOLD = 100
STATION_UPGRADE_GOLD2 = 150
STATION_UPGRADE_VOUCHERS = {
    1: (STATION_UPGRADE_VOUCHER, STATION_UPGRADE_GOLD),
    2: (STATION_UPGRADE_VOUCHER2, STATION_UPGRADE_GOLD2),
}
# 基地血量低于满血的这个比例时才考虑用券（用早了浪费，用晚了基地已经没了）
STATION_LOW_HP_RATIO = 0.6

# 围墙修复包（战术参考 T4）：售价仅 10 金，使用后把目标围墙回满血
# （任务书 4.6.3："目标坐标所在围墙回满血"；聊天里提到 3×3 能一次奶 5 段墙，
#  所以站位要挑"3×3 内残血墙最多"的位置，两种口径都能吃到最大收益）。
WALL_FIXER = "WallFixer"
WALL_FIXER_GOLD = 10
WALL_FIXER_RADIUS = 1  # 覆盖范围半径（3×3）
WALL_REPAIR_MIN_TARGETS = 3  # 至少 3 段残血墙才值得跑一趟
DUSK_ROUNDS = 5  # 天黑前提前回防的回合数
MIN_TOWERS_BEFORE_NIGHT = 2  # 入夜前的最低火力：不足时优先抢建而不是回防待命
# 开局回合数：每个游戏日的前几个回合内塔数有硬下限，不受 LLM 计划影响
OPENING_ROUNDS = 2
# 开局的塔数下限：复盘里"首日 3 回合只落地 1 座塔、R2 整回合零建造"，
# 第 2 座塔拖到 R3 才开工，火力成型远慢于敌方（敌方 R3 单回合双建）。
# 把"开局两回合内塔数 ≥ 2"写成硬阈值，不再看当天 LLM 计划的心情；
# 塔已经有 2 座时这条下限不起作用，金币仍可按计划留给升级券。
OPENING_MIN_TOWERS = 2
# 入夜前的最低围墙段数：复盘里首夜防线只有一座光塔、零段围墙，这里要求
# 临天黑时再抢铺一段（手里有石材才抢建，没石材仍然按原策略回防）
MIN_WALLS_BEFORE_NIGHT = 2
# 防守方每天至少要保证铺好的围墙段数：复盘里防守方整天零围墙、正面毫无阻挡，
# 机器人直接贴脸打基地，而同一局的进攻方反倒把来路封得严严实实。
# LLM 计划把墙压到 0 时防守方仍按下限留出石材（见 `_wall_target`）。
DEFENDER_WALL_QUOTA = 1
# 防守方开局防线计划（S3，即复盘建议里的 WALL_PLAN）：每个游戏日的前
# WALL_PLAN_ROUNDS 个回合里，分管经济的工人照着 `_calc_wall_order` 生成的防线
# 坐标（先来敌方向、再上左下右）铺够 WALL_PLAN_SEGMENTS 段墙，塔由另一名工人
# 照建。这段窗口里"砌墙"的优先级高于采石闲逛与凑批卖矿——围墙是防守方唯一的
# 正面屏障，而复盘里塔位优先的建造分支把工人一直占在基地旁等金币，首段围墙
# 因此拖到金币花光才开工（PK589649 落到 R10、PK589653 全程 0 段），PK589697
# 更是拖到 R16 才立起第一段、全程只有 1 段，基地裸奔到终局。
# 窗口取 10 个回合而不是 5：采石点离基地常有十来格，往返一趟就是好几回合，
# 5 个回合里连第一段的料都攒不齐，防线自然一直不成型。
WALL_PLAN_ROUNDS = 10
WALL_PLAN_SEGMENTS = 4
# 兼容旧名：测试与外部脚本仍按 `WALL_FIRST_ROUND` 引用（改名时漏改调用方）
WALL_FIRST_ROUND = WALL_PLAN_ROUNDS
WEAPON_UPGRADE_VOUCHER = "WeaponUpgradeVoucher1"  # 武器升级券（level1->level2）
WEAPON_UPGRADE_VOUCHER2 = "WeaponUpgradeVoucher2"  # 武器升级券2（level2->level3）
WALL_UPGRADE_VOUCHER = "WallUpgradeVoucher1"  # 围墙升级券（level1->level2）
WALL_UPGRADE_VOUCHER2 = "WallUpgradeVoucher2"  # 围墙升级券2（level2->level3）
UPGRADE_GOLD = 100  # 购买一张武器升级券所需金币
UPGRADE_GOLD2 = 150  # 武器升级券2 的金币（任务书4.6.3）
WALL_UPGRADE_GOLD = 20  # 围墙升级券的金币（任务书4.6.3）
WALL_UPGRADE_GOLD2 = 30  # 围墙升级券2 的金币（任务书4.6.3）

# 建筑升级券：当前等级 -> (券名, 兜底价格)。任务书4.6.3的售价是 武器券1=100、
# 武器券2=150、围墙券1=20、围墙券2=30，满级（level3）没有对应条目。
# 表格同时决定"升哪座、用哪张券"，塔型那套"说的与做的对不上"在这里同样不成立。
WEAPON_UPGRADE_VOUCHERS = {
    1: (WEAPON_UPGRADE_VOUCHER, UPGRADE_GOLD),
    2: (WEAPON_UPGRADE_VOUCHER2, UPGRADE_GOLD2),
}
WALL_UPGRADE_VOUCHERS = {
    1: (WALL_UPGRADE_VOUCHER, WALL_UPGRADE_GOLD),
    2: (WALL_UPGRADE_VOUCHER2, WALL_UPGRADE_GOLD2),
}

# === V4 日计划（基地布局 + 每日任务队列）===
# 参照战术参考里对手的 V4 逻辑：每天开局先做一次规划（数破洞、算当天要挖几座
# 石矿、生成目标队列、估一遍耗时），三个角色再照计划执行，而不是每回合各自
# 就地贪心。这里落成"能算、能执行"的那部分，阈值都取自那份逻辑的原话。
#
# 基地布局（以基地 2x2 占地为锚点、按来敌方位定向，基地在左上/右下时自动
# 反转）：外墙 = 基地外围第二圈（`_wall_ring`，即 6x6 可建造区的外框）；
# 站位 = 墙内一圈的四个角 R1(前上)/R2(前下)/R3(后上)/K0(后下)。站在墙内圈
# 的角上时，一个 3×3 修复包正好盖住 5 段外墙（角上 1 段 + 两条边各 2 段）
# ——这就是那份逻辑里"一次修复 5 个墙"的由来；基地贴地图边时有一侧的墙在
# 图外，只剩 4 段。R3 入夜前要先退到 K0：R3 正对敌方进场的方向，站在那儿
# 会被进场的机器人先手。
#
# 炮位仍按 `_calc_tower_sites` 的"三面分散 + 可达性校验"选：V4 的"正面两座
# + 背面一座"图的是炮火压住来路，而当前实现已经用 `_enemy_sides` 把第一座塔
# 压在来路上，并用 BFS 保证每座塔都有人走得到（聊天记录里的"炮塔堵路"），
# 再按那个布局重排等于把既有回归测试锁定的塔位推倒重来。
STONE_PER_MINE = 10  # 一座矿采空要 10 个回合（任务书4.2：每个矿采集10次后消失，每次得1个）
STONE_RESERVE_MIN = 5  # Rmin：背包里始终留 5 块石材的保底
STONE_PLAN_MAX = 2  # AR 上限：一天最多规划 2 座石矿（第三座连来回的路都走不完）
QUEUE_TARGETS = 5  # 目标队列长度（默认 5 个铜矿，按 AR 从头替换成石矿）
WALL_BUILD_ROUNDS = 1  # 建一段围墙占 1 个回合（V4 修墙路径的时间口径）
WEAK_WALL_RATIO = 0.5  # 一级墙血量低于一半就算"已经破了"：可拆穿走捷径（拆掉再补）
DAY_PLAN_LIMIT = DAY_ROUNDS  # Tall 的上限：白天只有 70 个回合

# 生命药剂（任务书4.6.3：使用者回满血，10 金）。白天血量偏低时备一剂，
# 夜里血量掉到 50 以下立刻喝掉——这只救小人自己，不救墙也不救基地。
MEDICINE = "Medicine"
MEDICINE_GOLD = 10
MEDICINE_HP = 80  # 白天：血量低于 80 时补一剂（V4 的额外目标）
MEDICINE_HP_NIGHT = 50  # 夜晚：血量 <= 50 时喝（V4）
STATION_CRITICAL_HP = 150  # 夜晚：基地血量 < 150 时用基地升级券（V4）
CARRIER_LOW_HP = 40  # 夜晚：持券者自己血量 < 40 时也用一次（V4）
FIXER_STOCK = 3  # 围墙修复包的常备量（V4：不足就补齐）
FIXER_STOCK_FIRST = 2  # 第 3 天第一次补给只买 2 个（V4："天数=3则购买2个"）
FIXER_RESTOCK_DAY = 3  # 开拓者从第 3 天起负责补给修复包（V4）
FIXER_CARRIER_DAY = 4  # 铜矿工人从第 4 天起自带 3 个修复包（V4）
FIXER_NIGHT_HP = 100  # 夜晚：正面的人（R1/R2）发现自己那片墙 < 100 血时补一次
FIXER_NIGHT_HP_WEAK = 150  # 夜晚：后排的人（R3）负责的墙 < 150 血时补一次
SELF_FIXER_HP = 30  # 自身血量 < 30 时也补一次（V4："如果自身血量<30，也使用一次"）
PIONEER_WEAPON_VOUCHERS = 3  # 开拓者一次备 3 张武器升级券（三座塔各一张，V4）
PIONEER_WALL_VOUCHERS = 10  # 开拓者一次备 10 张围墙升级券（V4）

# 报文没给武器商店价格表时的商品兜底价（正式售价见任务书4.6.3）
ITEM_FALLBACK_PRICE = {
    name: price
    for vouchers in (
        WEAPON_UPGRADE_VOUCHERS,
        WALL_UPGRADE_VOUCHERS,
        STATION_UPGRADE_VOUCHERS,  # 基地券漏在这里会让"金币不足"判断失效
    )
    for name, price in vouchers.values()
}
# 消耗品不在券表里，兜底价单独补上：查不到价格时按 0 算会让"金币不足"的
# 判断失效（0 金也敢下单，回合结算时才失败），先按任务书4.6.3 的售价兜底。
ITEM_FALLBACK_PRICE[WALL_FIXER] = WALL_FIXER_GOLD
ITEM_FALLBACK_PRICE[MEDICINE] = MEDICINE_GOLD
# 金币闲置熔断线：手里攥着够再建两座塔的金币时，不允许再把塔数配额压到满编
# 以下（复盘里"金币连续多回合冻结在 50，无塔无墙无升级"就是这么来的）。
# 一座塔 25 金换 10 点火力和一段射程，比攒到 100 金升一级划算得多，
# 所以金币越积越多时优先把它变成塔，而不是留在手里。
GOLD_FLUSH_TOWERS = WEAPON_BUILD_COST * 2

# 可卖给小贩的矿石（按优先级排序，石矿既是围墙材料也是主要收入来源）
SELLABLE_MINES = (STONE_MINE, IRON_MINE, COPPER_MINE)
# 纯收入矿石：只用来换金币，不参与砌墙。围墙配额没铺满时经济分支整段被跳过，
# 铁/铜必须有一条独立于围墙的变现通道，否则防守方的金币会一直冻结
# （三场复盘里 gold 从 R6/R8/R9 起一路为 0 直到 R17，背包里却一直躺着矿石）。
INCOME_MINES = (IRON_MINE, COPPER_MINE)
# 负责"矿石换金币"的工人（V4 里的铜矿工人）的采集顺序：铜是队列里的默认
# 目标（V4："默认为5个铜矿"），铁作次选——铁矿在世界事件里会连着两天采不了，
# 铜矿没有这个风险；石材只作兜底。
ECONOMY_MINE_ORDER = (COPPER_MINE, IRON_MINE, STONE_MINE)

# 任务点排序权重：报文缺少 timeoutRounds 时用最大值，不抢占"临期优先"
TASK_TIMEOUT_UNKNOWN = 10 ** 9

# 基地四个方位（用于让武器塔分散布防，顺序仅用于同分时的稳定排序）
TOWER_SIDES = ("up", "left", "down", "right")

# 机器人威胁等级：数值越大越优先处理（与LLM prompt中的提示保持一致）
ROBOT_THREAT = {
    "bossRobot": 3,
    "largeRobot": 2,
    "middleRobot": 1,
    "smallRobot": 0,
}

# 自进化任务：从任务描述中识别需要在沙盒里读取的文件名
# 只认ASCII字符，避免把“请阅读”这类描述文字一起吃进文件名
TASK_FILE_PATTERN = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:md|txt|json|csv|log)")
# 沙盒输出中的任务标识前缀，用于确认输出属于当前任务
TASK_MARKER = "[TASK]"
# 沙盒输出中的答案结束标记：它之后的诊断信息（目录列表等）永远不会被当成答案
TASK_END_MARKER = "[TASK_END]"
# 沙盒探测标记：任务描述里没给文件名时先探一次沙盒，这个标记下的输出
# 只是文件路径清单，`_task_answer` 永远不会把它当成答案提交（见 `_sandbox_probe`）
TASK_PROBE_MARKER = "[TASK_PROBE]"
# 系统文档与库树（S2）：复盘 PK590252 的 R14/R16 两次读回来的
# /usr/share/doc/uom-se-1.0.4/README.md 就在这里——探测命令放宽到 `*.md`
# 之后，全盘 find 先命中的恰恰是这些库文档，任务正文一次都没读到。自进化
# 任务的文件与接口文档都不在这些树里，找任务文件与找接口文档都不扫它们。
TASK_SYSTEM_PRUNE = (
    "/usr/share/doc", "/usr/share/man", "/usr/share/info", "/usr/share/sgml",
    "/usr/lib", "/usr/include", "/usr/src",
)
# 沙盒里搜任务文件时跳过的虚拟目录：进程/内核/设备文件系统里不会有任务文件，
# 却会让全盘 find 变慢并刷出一堆 Permission denied
TASK_FIND_PRUNE = ("/proc", "/sys", "/dev") + TASK_SYSTEM_PRUNE
# 任务根目录（S2）：自进化任务是按"步-接口"分目录摆的一套同构任务，任务文件
# 与它自己的接口文档就在同一个目录树里。复盘 PK590252 里任务正文的真身是
# /tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md，而全盘
# 扫描（同一局的 R13 直接 `[TIMEOUT]`）既慢又要碰运气。探测、回读与执行器
# 的搜索都先锚在这里，根目录里没有才退到全盘——沙盒版本不同、任务文件摆在
# 别处时照样找得到。
TASK_ROOTS = ("/tmp/selfEvolutionTask", "/tmp/selfEvolution")
# 沙盒探测的次数上限（S2，敌方常量 max_probe_attempts=2）：只有任务描述里
# 连文件名都没给时才探测。第一次探测没认出任务文件，说明根目录里那套命名
# 对不上（或输出被截断），再探第二次；两次都没认出来就改用执行器按文件名
# 特征自己找，不再反复扫根目录——复盘里 R12-R17 六回合的沙盒空转、R14/R16
# 两次逐字相同的输出，都是"同一件事反复做"。
TASK_PROBE_LIMIT = 2
# 任务文件的文件名特征：描述里点名的那一份，以及任务目录里的同构任务
# （任务书5.3节的例子是 task_1_beijing / task_2_shanghai 这一套）
TASK_FILE_NAMES = ("task*", "spec*")
# 探测沙盒时放宽一档：描述里连文件名都没有时，任何文档都可能是任务说明
TASK_PROBE_NAMES = ("task*", "spec*", "*.md")
# 任务文件的扩展名闸门（与 TASK_FILE_PATTERN 的扩展名一致）。
# 沙盒里文件名带 task 前缀的不止任务书本身：Debian 的 docbook 样式表
# /usr/share/sgml/docbook/xsl-stylesheets-<版本>/html/task.xsl 同样命中
# `task*`。三场复盘（PK589253/589255/589257）里沙盒每次回读回来的都是这份
# 三万三千多字符的样式表，任务正文一次都没读到，开拓者的 phase 因此卡了
# 6~7 个回合、任务分全丢（"沙盒读错文件而非没执行"）。名字像任务文件、
# 扩展名却不是文档的一律不算任务文件。
TASK_FILE_EXTS = (".md", ".txt", ".json", ".csv", ".log")
# find 命中后执行的动作：只列出路径（探测与回读任务文件都用它）
TASK_FIND_PRINT = "-print"
# 任务文件分段标记：读任务文件时顺带把沙盒里的任务文件都读回来，
# 每份用这两个标记包起来，`_task_file` 据此认出沙盒里的真实文件名
TASK_FILE_MARKER = "[TASK_FILE]"
TASK_FILE_END = "[TASK_EOF]"
# 单份任务文件最多读回的行数、一次最多回读的份数，
# 避免一条命令的输出把响应体撑大
TASK_FILE_LIMIT = 60
TASK_FILE_MAX = 12

# 任务答案的产出方式（自进化闭环）：沙盒里的任务文件是"任务描述"，不是答案。
# 复盘里 R12/R14/R16/R18 四次 submitAnswer 交的全是任务原文
# （"# 自进化任务 A-1：查询北京文化遗产 ## 任务背景…"），Judge 一次都没放行，
# 任务分与任务金币全丢。答案必须由"按任务描述执行沙盒命令"产出：
# 执行器（`_task_executor`）读任务文件与沙盒里的接口文档，照文档给出的地址
# 真实调用本地接口取数，把取到的数据打成 `[SOLUTION]` 段。
TASK_SOLUTION_MARKER = "[SOLUTION]"  # 答案段落开头，后跟任务文件名
TASK_SOLUTION_END = "[/SOLUTION]"
TASK_DATA_MARKER = "[API]"  # 真实取数的证据：只有请求成功才会打印
TASK_API_FAIL_MARKER = "[APIFAIL]"  # 取数失败也留一行诊断（URL + 异常类型）
TASK_SCAN_MARKER = "[SCAN]"  # 找到多少任务文件/接口文档/可用地址 + 有没有抠到鉴权 Key
TASK_ANSWER_MIN_LEN = 4  # 答案最短长度（任务原文动辄几千字，这条挡住空答）
TASK_ECHO_RUN = r"[一-鿿]{6,}"  # 任务描述里的中文长句（复读判定用）
TASK_API_DEFAULT = "http://localhost:8899"  # 沙盒内的本地接口
TASK_API_TIMEOUT = 1  # 单次取数超时（秒），整条沙盒命令限时 15 秒
TASK_API_MAX_CALLS = 8  # 一条命令里最多请求几次（本地接口，失败也是立刻返回）
# 候选地址里固定排在最前面的条数（S1，见执行器的 `rotate`）：文档给出的样例
# 地址优先级最高，每回合都该先试；候选清单被 `TASK_API_MAX_CALLS` 截断时，
# 轮转只发生在这一小段之后的兜底候选上。
TASK_API_KEEP = 2
TASK_API_TIME_BUDGET = 8  # 取数阶段的时间上限（秒），留出找文件与回读的余量
TASK_EXEC_DIR_BUDGET = 4000  # 全盘找文件时最多进几个目录（防止 walk 慢过 15 秒）
TASK_API_QUERY_MAX = 2  # 每个接口地址最多试几个查询词
TASK_API_BODY_LIMIT = 400  # 单个响应体最多带回的字符数
TASK_SOLVE_MAX = 4  # 一次最多解几份任务文件（当前这份排第一）
TASK_API_PATH_SUFFIXES = ("/", "/api", "/docs")  # 文档没给样例时先试这几个
TASK_API_DOC_NAMES = (r"api", r"doc", r"readme", r"\.md$")  # 接口文档的文件名特征
# 全盘兜底找接口文档时的文件名特征（S1，复盘 PK591809）：任务根目录里的
# `.md` 首先命中的就是任务文件自己（任务书就是 `.md`，同样落在
# `TASK_API_DOC_NAMES` 里），所以近处搜到的"文档"可能一份接口说明都不含，
# 取数地址与鉴权 Key 都读不到——`[SCAN] docs=2 urls=1 key=no` 那个形态里，
# `urls` 只剩本地接口那一条兜底，取数只能一路 404 到止损。手上这几份文档里
# 一个本地接口地址都没有时，到全盘再找一遍（见执行器里的这段），但只认
# 名字里写着"接口/文档"的那些：比 `\.md$` 窄得多，免得又把 /usr 下的库文档
# 与任务文件自己捞回来占满名额。
TASK_DOC_WIDE_NAMES = (
    r"api", r"interface", r"swagger", r"openapi", r"readme", r"doc",
    r"接口", r"说明", r"文档",
)
# 接口鉴权（S3，复盘 PK590918/PK590917）：接口文档的样例里写着该带哪个头，而
# 执行器一直只发 `Accept`，R16 的 `[APIFAIL] ... HTTPError 401 => missing
# 'Authorization' header` 就是这么来的——同一回合对手已经带着 Bearer 取到数
# （`__API status=OK ... auth=Bearer`）。文档里 Key 的写法各家不同
# （`Authorization: Bearer sk-xxx`、`X-API-Key: xxx`、`api_key=xxx`、表格里的
# `| API Key | xxx |`），这里按几种常见写法把 Key 抠出来随请求一起发；抠不到
# 就照旧裸请求——多带一个头不影响本就无需鉴权的接口，少带一个头则必然 401。
TASK_API_KEY_MIN_LEN = 6  # Key 至少这么长：更短的串多半是行文里的词，不是 Key
TASK_API_KEY_PATTERNS = (
    # `Authorization: Bearer <key>` / `Authorization=<key>`（`Bearer` 可有可无）。
    # 分隔符里带上 `|`：接口文档常把请求头写成表格的一行
    # （`| Authorization | Bearer sk-xxx |`），只认 `:`/`=` 时这一行整个漏掉，
    # Key 抠不出来 -> `[SCAN] ... key=no` -> 裸请求 -> 401 缺 Authorization 头。
    # `Bearer` 两边的引号/反引号一并吃掉：文档写成 ``Bearer `sk-xxx` ``
    # （markdown 行内代码）时，值前面还挂着一个反引号，旧写法从反引号起头、
    # 匹配不上捕获组。
    r"authorization[\"']?\s*[:=：|]\s*[\"'`\s]*(?:bearer\s+)?[\"'`\s]*"
    r"([A-Za-z0-9._~+/=\-]{%d,})" % TASK_API_KEY_MIN_LEN,
    # `X-API-Key: <key>` / `api_key=<key>` / `token：<key>` / 表格里的 `| API Key | <key> |`
    r"(?:x-api-key|api[-_ ]?key|apikey|access[-_ ]?token|token)[\"']?\s*[:=：|]\s*[\"'`\s]*"
    r"([A-Za-z0-9._~+/=\-]{%d,})" % TASK_API_KEY_MIN_LEN,
)
# 文档里的占位写法（`Authorization: Bearer <你的 API Key>` 这类），抠出来也不是 Key；
# 后半列是表格里常见的一格说明文字，别把它当成 Key 发出去
TASK_API_KEY_PLACEHOLDERS = frozenset({
    "your_api_key", "your-api-key", "your_apikey", "yourkey", "your_key",
    "api_key", "apikey", "api-key", "key", "token", "access_token",
    "xxx", "xxxx", "xxxxxx", "placeholder", "example",
    "required", "optional", "string", "header", "bearer",
})
TASK_API_AUTH_HEADER = "Authorization"  # 文档写 Bearer 的那一种（对手用的也是它）
TASK_API_KEY_HEADER = "X-API-Key"  # 另一种常见写法，两个一起带上
TASK_API_BEARER = "Bearer "
TASK_EXEC_PRUNE = ("/proc", "/sys", "/dev", "/run")  # 全盘找文件时跳过的虚拟目录
# 找接口文档时额外跳过的系统文档树（S2）：执行器要按"读文档 -> 拼地址"取数，
# 而全盘捞回来的文档里最先命中的往往是库自带的说明（复盘 PK590252 的 R14/R16
# 两次读回来的都是 /usr/share/doc/uom-se-1.0.4/README.md，两万三千多字符，
# 与任务毫无关系），照着它拼出来的地址自然取不到数。任务根目录与任务文件
# 所在目录优先，这些系统文档树直接不扫。
TASK_DOC_PRUNE = TASK_EXEC_PRUNE + TASK_SYSTEM_PRUNE

# 任务止损（S1）：自进化任务的闭环是"下发沙盒命令 -> 取数 -> submitAnswer"，
# 沙盒里读不到任务正文、或者每回合回读回来的都是同一份文件时，这个环永远
# 合不上。复盘里 PK589649/589653 的沙盒从 R11 起连续 6~7 个回合返回逐字相同
# 的输出（exitCode:0 但没有取数证据），开拓者被读文件死循环占死，任务分丢光、
# 这名劳动力也一起白搭。这里给任务四条止损线，到线就放弃任务、把开拓者还给
# 战斗调度（见 `_task_abandoned`）：
#   - 同一份沙盒输出连续出现 TASK_LOOP_LIMIT 次（读文件循环）
#   - 任务已经占用开拓者 TASK_TIMEOUT_ROUNDS 个回合（任务书：单个任务时限 15 回合）
#   - 同一份答案交满 TASK_SUBMIT_LIMIT 次仍没被放行（见下面的"提交闸门"）
#   - 连续 TASK_API_FAIL_LIMIT 个回合取数全失败（见 `TASK_API_FAIL_LIMIT`）
# 超时线取 10 而不是任务书的 15：真能解出答案的任务在收到第二条沙盒输出的
# 回合就交卷了（答案缓存命中时更快），拖到第 10 个回合还交不上卷的任务，
# 剩下的 5 个回合同样交不上，不如早点把开拓者还给战斗调度。
TASK_LOOP_LIMIT = 3
TASK_TIMEOUT_ROUNDS = 10
# 兼容旧名：测试与外部脚本仍按 `TASK_TIMEOUT` 引用（改名时漏改调用方）
TASK_TIMEOUT = TASK_TIMEOUT_ROUNDS

# 取数连败止损（S1，复盘 PK590916/PK591014）：自进化任务的接口地址是靠
# "读沙盒里的接口文档 -> 拼地址"猜出来的，猜不中时沙盒每回合都返回一串
# `[APIFAIL] ... HTTPError 404`（一条命令 `TASK_API_MAX_CALLS` 次机会全打光，
# 日志上的 `fail=8` 就是它）。这种"取数全失败"与读文件死循环不同：每回合的
# 地址清单都在变（执行器与 LLM 给的 `CMD:` 交替下发），沙盒输出逐字相同这条
# 判据（`repeats`）永远到不了线——复盘里开拓者 R10 接任务后 watch 一路
# r0/t1→r2/t3→r1/t6，直到 `TASK_TIMEOUT_ROUNDS` 才兜底，整段任务窗都白等在
# 任务点上（idle_man 涨到 3）。这里再给一条"连续取数失败"的止损线：连续
# TASK_API_FAIL_LIMIT 个白天回合取数全失败、手里又攒不出一份答卷时就放弃
# 任务，把开拓者还给战斗调度——继续试下去只是把同一批猜错的地址再试一遍。
# 取 6 与 `TASK_LLM_MAX_PROMPTS` 对齐：LLM 兜底是文档写明的正解，得让它把
# 6 次求助跑完再判死；又比 `TASK_TIMEOUT_ROUNDS`（10）早收手，把省下的几个
# 回合还给战斗调度（`_task_abandoned` 里这两条线是或的关系）。
TASK_API_FAIL_LIMIT = 6

# 沙盒命令被整条掐掉时判题器留在 `lastCmdResult` 里的标记（S1）：一条命令限时
# 15 秒，被判题器 kill 掉的那一回合输出里没有任何取数证据（`[API]`/`[APIFAIL]`
# 都没有），`[SCAN]` 也可能还没轮到打印就被掐了——这种输出与"读文件死循环"
# 长得一模一样。旧判据于是把它归进读文件循环：输出逐字相同、`repeats` 每回合
# 累加，接上任务后的第 3 个回合就撞 `TASK_LOOP_LIMIT` 被熔断（复盘 PK590252 的
# R13 整条命令就是 `[TIMEOUT]`）。可超时说明的是"这条命令做得太多"，不是
# "读不出新东西"：该按取数连败那条更宽的线走（`TASK_API_FAIL_LIMIT`），把重试
# 与 LLM 兜底留给它，下一回合换一条更省时间的命令去取数（见 `_task_retry_budget`）。
TASK_TIMEOUT_MARKER = "[TIMEOUT]"
# 上一回合超时之后，这一回合执行器搜目录的上限（S3 的"读题去重"）：全盘 walk
# 是整条命令里最慢的一步，超时那一回合已经证明了这一点。重试回合把上限压到
# 这么小，让命令在 15 秒内跑到"调接口取数"那一步——题干上一回合已经读过一遍，
# 重试该把预算花在取数上，而不是再扫一遍同一棵目录树（`TASK_EXEC_DIR_BUDGET`）。
TASK_EXEC_RETRY_DIR_BUDGET = 400

# 提交闸门（S1）：复盘 PK590252 的 R17 提交的是
# "/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md"
# ——开拓者把"该读哪个文件"当成了"文件里问的答案"，Judge 判 0 分，任务 2
# 也跟着连锁未接。整个答案就是一条带文档扩展名的路径时一律不提交：按任务
# 描述取数解出来的答案不会长成文件路径的样子（判定见 `_task_path_answer`）。
# 同一份答案最多提交 TASK_SUBMIT_LIMIT 次（S1，敌方常量 max_submit_attempts=3）：
# 交完卷任务还留在 phaseTask 里，说明 Judge 没放行；同一个答案再交第三遍
# 同样放行不了，只会继续占着开拓者。到线就按止损处理（见 `_task_abandoned`），
# 把开拓者还给战斗调度。
TASK_SUBMIT_LIMIT = 3

# === LLM 解任务 ===
# 沙盒里的接口只能靠"读文档 -> 拼地址"去猜，猜不中时答案区永远是空的，
# 开拓者就卡在任务点耗到超时（复盘 #42：R12–R17 六次沙盒输出都是同一份
# 任务文件、从未提交，任务分 0）。所以再加一条兜底：把**任务描述 + 沙盒里
# 捞到的接口文档**交给 LLM，让它给出确切做法——
#   接口文档：明确写着"自进化任务期间调用 LLM 不占用每个游戏日的 3 次额度"，
#   这正是"用 Agent 自进化解题"这条路的正解。
# 回复约定两个前缀（只认第一个命中的）：
#   `CMD: <命令>`    -> 下一回合把这条命令原样丢进沙盒（15 秒限时）
#   `ANSWER: <答案>` -> 直接交卷，不再绕沙盒
TASK_LLM_CMD_PREFIX = "CMD:"
TASK_LLM_ANSWER_PREFIX = "ANSWER:"
TASK_LLM_MAX_PROMPTS = 6  # 同一个任务最多求助几次，避免整段任务都耗在提问上
TASK_LLM_EVIDENCE_LIMIT = 3000  # 喂给 LLM 的沙盒输出上限（任务文件+接口文档）
TASK_LLM_ANSWER_MIN_LEN = 2  # 比 `TASK_ANSWER_MIN_LEN` 更宽：LLM 可能只给一个数
# 单个任务跨回合的 LLM 交互状态：token -> {prompts, pending_cmd, cmd_round, answer}
_TASK_LLM_STATE: dict[str, dict[str, Any]] = {}

# 任务答案缓存：任务文件名 -> 沙盒执行产出的答案（`[SOLUTION]` 段的内容）
# 任务书5.3节要求"根据任务1探索的内容形成固定SOP或者SKILL，实现Agent自进化"，
# 积分又是"任务奖励 + 5 × 标准回合数 / (完成回合 - 接取回合)"（任务书第六章），
# 交得越早分越高。执行器一次会把沙盒里的任务文件都试着解一遍，解出来的答案
# 存进这里，后续任务点领到同一份任务时，开拓者不必再等一个来回的沙盒输出，
# 接取后下一回合就能直接作答（复盘里敌方就是靠答案缓存秒交，两次提交各拿 155 分）。
# 这是纯缓存：没有命中的任务仍然走"下发沙盒命令 -> 下一回合读输出"的原路径。
_TASK_ANSWER_CACHE: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class TaskWatch:
    """任务看门狗：最近一回合的那次观察（不是跨回合的战场状态）

    字段:
        token: 当时进行中的任务标识（`_task_token`）
        round_no: 记下这条观察的回合号
        output: 当时那份属于本任务的沙盒输出（没有输出时为空串）
        rounds: 这个任务已经占用开拓者的回合数
        repeats: 当前这份输出已经连续出现了几次（沙盒真在取数的回合不累计，
            见 `_task_fetch_failed`；上一回合跑的是 LLM 给的命令时同样不累计，
            见 `_llm_command_round`）
        probes: 到这个回合为止发出去的沙盒探测命令数（见 `TASK_PROBE_LIMIT`）
        submits: 到这个回合为止打算交上去的答卷数（见 `TASK_SUBMIT_LIMIT`）
        fails: 到这个回合为止连续取数全失败的回合数（见 `TASK_API_FAIL_LIMIT`）
    """

    token: str
    round_no: int
    output: str
    rounds: int
    repeats: int
    probes: int
    submits: int
    fails: int


# 任务看门狗（模块级单例，只存最近一回合的观察值）：每回合由 `decide` 用当前
# 报文刷新一次，任务分支与 `sandbox_command` 只读不改。换任务、换局、或者回合号
# 不连续时从头计数（见 `_watch_task`），所以它只描述"当前这一局这个任务"，
# 不是跨回合累积的战场缓存。放弃的判据见 `_task_abandoned`。
_TASK_WATCH: TaskWatch | None = None

# 是否提交LLM策略咨询prompt（可用环境变量 LLM_PROMPT=0 关闭）
# 接口文档：每队每个游戏日的 LLM 调用额度为 3 次（自进化任务期间不计入），
# 额度在当日首回合重置，所以每个游戏日最多咨询 LLM_PROMPT_PER_DAY 次
LLM_PROMPT_ENABLED = os.getenv("LLM_PROMPT", "1") != "0"
LLM_PROMPT_PER_DAY = 3  # 每个游戏日用满的咨询次数（接口文档的每日额度）

# LLM 建议里可以被决策层执行的部分：只开放有限几个"旋钮"，让建议和指令
# 出自同一套决策函数，而不是各说各话（复盘里的"LLM建议与指令脱节"）。
LLM_PLAN_LINE = re.compile(r"PLAN\s*[:：]\s*(?P<body>[^\r\n]*)", re.IGNORECASE)
LLM_PLAN_ITEM = re.compile(r"([A-Za-z_]+)\s*=\s*([A-Za-z0-9]+)")
# 提示词里的占位写法用尖括号：LLM 照抄模板时不会被解析成"tower=0"这类误读
LLM_PLAN_TEMPLATE = (
    "PLAN: tower=<1-3> wall=<0-3> upgrade=<on|off>"
    " defend=<left|right|up|down>"
)
LLM_PLAN_TRUE = ("on", "true", "yes", "1")
LLM_PLAN_FALSE = ("off", "false", "no", "0")
# 计划里塔数的下限：误读成 0 会让白天完全不设防，宁可保守也不接受
LLM_MIN_TOWERS = 1
LLM_MAX_TOWERS = len(TOWER_LOADOUT)
LLM_MAX_WALLS = 3


@dataclass(frozen=True, slots=True)
class LlmPlan:
    """LLM 建议中被采纳的部分（全部有界，且都有保守默认值）

    字段:
        tower: 白天要保证建成的武器塔数量（LLM_MIN_TOWERS..LLM_MAX_TOWERS）
        wall: 优先铺好的围墙段数（0..LLM_MAX_WALLS），0 表示没有配额
        upgrade: 是否允许白天花金币买武器升级券
        defend: 优先布防的方位（up/left/down/right），None 表示按默认顺序

    默认值等价于"完全按客户端原策略执行"，所以 LLM 不回复 PLAN 行、
    回复里字段缺失或值越界时，策略与改造前完全一致。
    """

    tower: int = LLM_MAX_TOWERS
    wall: int = 0
    upgrade: bool = True
    defend: str | None = None


LLM_PLAN_DEFAULT = LlmPlan()

# 沙盒输出中的错误特征：命中说明任务文件没读到，不能当作答案提交
TASK_ERROR_MARKERS = ("No such file", "Permission denied", "Is a directory")

# 接口错误响应体的特征（S2）：`[API]` 只证明"这一次请求有响应"，不证明
# "响应就是答案"。接口把错误包成 200 的 JSON（`{"status":"error",...}`）、
# 或者沙盒命令自己抛了异常时，取数器照样会把这段正文打进 `[SOLUTION]` 段，
# 上一版就原样交了上去——复盘 PK590557 的 R14/R16/R18 三次 submitAnswer
# 交的正是接口文档原文与 401/404 的错误 JSON，Judge 一次都没放行。
#
# 只认"看着就是错误"的形态，正常数据不会被误伤：错误键（`"error"` 为非空值）、
# 错误状态值、4xx/5xx 状态码、状态码后跟 HTTP 状态短语、Python 异常名与回溯。
# 特别地，HTTP 状态短语必须带状态码（`404 Not Found`）才判定——"故宫"的
# 英文是 Forbidden City，裸词匹配会把一份完全正确的答案拦下来。
TASK_ERROR_BODY = re.compile(
    # 负向断言里也要吃掉空白：`"error"\s*:\s*(?!...)` 里的 `\s*` 会回溯成零个
    # 字符，断言于是盯着空格而不是值看，`{"error": null}`（值就是空的）会被
    # 误判成错误体，把一份正常答案拦下来。
    r'"error"\s*:\s*(?!\s*(?:null\b|\[\s*\]|\{\s*\}|""|\'\'))|'
    r'"status"\s*:\s*"?'
    r"(?:err|fail|unauthor|forbidden|not[ _-]?found|invalid|denied|bad)|"
    r'"(?:status|code)"\s*:\s*"?[45]\d\d\b|'
    r"\b[45]\d\d\s+(?:Bad Request|Unauthorized|Forbidden|Not Found|"
    r"Method Not Allowed|Internal Server Error|Bad Gateway|Service Unavailable)\b|"
    r"\b(?:InvalidURL|HTTPError|URLError|SocketTimeout)\b|"
    r"Traceback \(most recent call last\)|"
    # 沙盒命令自己的报错（PK590921 的 R16，也是 PK590882 的 R16）：`jq` 之类的
    # 工具在沙盒里根本不存在，命令的输出于是是一行 `jq: command not found`，
    # 它既没有错误键也没有状态码，上面几条形态一道都拦不住，会被当成"取到的
    # 数"交上去。`cat` 读一个没找到的文件（同 R16 的
    # `cat: task_1_beijing.md: No such file or directory`）与 curl 写响应体
    # 失败（`Failed writing body`）同理。
    #
    # `command not found` 必须挂在工具名前缀上（`jq: command not found` /
    # `bash: line 3: curl: command not found`），而且工具名只能紧贴着它：
    # 这几个字本身是合法英文，放开了匹配会把一份写着
    # "command not found 是我的歌名" 的正常答案拦下来。`No such file or
    # directory` 则是整句固定搭配，本身不会出现在答案数据里；任务描述里的
    # `No such file`（见 `TASK_ERROR_MARKERS`）因为少了 "or directory" 不会被
    # 这条命中，两处判定互不干扰。
    r"^(?:\S+\s*)?:\s*command not found\b|"
    r"\s\S+:\s*command not found\b|"
    r"\bNo such file or directory\b|"
    r"\bEndpoint not found\b|"
    r"\bFailed writing body\b|"
    # 脚本跑不起来时的 stderr（S1）：命令调的是沙盒里的现成脚本（`./check`
    # 这类校验脚本）时，脚本本身有毛病的话输出里一行取数结果都没有，只有
    # `/bin/sh^M: bad interpreter`（CRLF 行尾，见 `_crlf_safe_command`，复盘
    # PK591009 的 R14）、`sh: 1: ./check: not found`（脚本不在/没有执行位）或
    # `unexpected EOF while looking for matching`（脚本里引号不配对）。这几句
    # 都是 shell 自己的诊断，正常取数结果不会长成这样，命中即不提交。
    r"\bbad interpreter\b|"
    r"\bcannot execute\b|"
    r"\bunexpected EOF while looking for matching\b|"
    r"\b\S+:\s*\d+:\s*\S+:\s*not found\b",
    re.IGNORECASE | re.MULTILINE,
)

# 答案与沙盒里读到的文档原文重合的判定（S1）：`TASK_ERROR_BODY` 挡的是"错误体"，
# `_task_echo` 挡的是"复读任务描述"，但复盘 PK590851 的 R13 交上去的是沙盒里
# 那份**接口文档**的原文（"# 国家文化遗产数字档案查询系统 — API 参考文档…"）
# ——执行器把 `/docs` 这类地址取回来的文档正文当成了取数结果打进 `[SOLUTION]`
# 段，四道闸门一道都没拦住，Judge 判 0 分；PK590847 的 R13–R16 沙盒里回读的
# 也一直是同一份文档。这份文档既不在任务描述里，也不是错误体，只能靠"它长什么
# 样"来认：执行器读到的接口文档、`_task_dump` 回读的任务文件，开头都留一行指纹，
# 答案里出现这份指纹就说明交的是文档原文（判定见 `_task_text_answer`）。
TASK_DOC_MARKER = "[DOC]"  # 执行器读到的接口文档开头（诊断行，供答案闸门比对）
TASK_TEXT_HINT = 120  # 文档指纹取开头这些字符（空白归一化后）
TASK_TEXT_HINT_MIN = 30  # 指纹短于这个长度不作数：太短的串容易误伤正常答案
# 答案"长得像文档"的判定（S2）：`_task_text_answer` 拿沙盒输出里的文档指纹
# 比对，而指纹只在执行器跑过的那一轮里才有（`[DOC]` 行、`[TASK_FILE]` 段）。
# 走 LLM 那条路时（`CMD: <命令>` 的输出就是答案）输出里没有指纹，一条
# `cat 接口文档.md` 的输出会被原样交上去——复盘 PK590836 的 R15 交的正是
# `# 国家文化遗产数字档案查询系统 — API 参考文档…` 这份文档原文，Judge 判 0，
# 且每交一次就烧掉一次提交额度。文档有自己的长相：首行是 Markdown 标题
# （`# 标题`），正文里还常点名"参考文档""版本"这类字样。答案是一段数据
# （`{"city":"北京"}` / `故宫`），不会长成这样，所以这条闸门只认"首行就是
# 标题"这一种形态，误伤面小到可以忽略。
TASK_DOC_HEAD = re.compile(r"^\s{0,3}#{1,6}\s")
TASK_DOC_WORDS = ("参考文档", "接口文档", "API 文档", "使用说明", "文档版本", "版本历史")

# 各阵营的任务点类型
_TASK_POINTS_BY_TEAM = {
    "challenger": (CHALLENGER_TASK_1, CHALLENGER_TASK_2),
    "defender": (DEFENDER_TASK_1, DEFENDER_TASK_2),
}


def decide(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    """
    主决策函数
    输入: 判题系统发来的游戏状态
    输出: (角色ID字符串: 指令字典, 提交给LLM的prompt)
    """
    turn = Turn.load(payload)
    # 任务看门狗：记录本回合的任务与沙盒输出，判断任务是否已经该止损
    _watch_task(turn)
    # 上回合执行器解出来的任务答案按文件名缓存，后续任务一到手就能直接作答
    _remember_task_answers(turn.last_cmd_result)
    # 上一回合 LLM 的回复：任务期间的 CMD/ANSWER 落进当前任务的状态
    _consume_task_reply(turn, payload)
    # 上一回合的LLM建议解析成有界计划，和指令生成器共用（解析不出来时是默认计划）
    plan = _llm_plan(payload)
    commands: dict[int, dict[str, Any]] = {}

    if turn.is_day:
        _decide_day(turn, commands, plan)
    else:
        _decide_night(turn, commands)

    # 任务期间优先用"任务求助"prompt（接口文档：任务期的 LLM 调用不占每日额度），
    # 没有任务或任务不需要求助时才是策略咨询
    prompt = _task_prompt(turn, payload) or _generate_strategy_prompt(turn, payload, plan)

    # 转换key为字符串
    return {str(key): value for key, value in commands.items()}, prompt


def task_brief(turn: Turn, sandbox_sent: bool = False) -> str:
    """任务链路的单行状态（对战分析靠它定位"为什么没交卷"）

    复盘里"任务 0 分"只能看到"沙盒在跑"，看不到卡在哪一步。这一行把整条链路
    摊开，全部是 `k=v`（日志按行正则解析）：

        phase=...      当前任务描述（截断）
        state=...      答案判定结果（见 `_task_answer_with_reason` 的原因码）
        watch=r2/t3    看门狗：同一份沙盒输出重复 r 次 / 任务已占用 t 回合
        abandoned=yes  是否已止损（开拓者被放回战斗调度）
        sandbox=发送   本回合是否下发了沙盒命令
        api=1 fail=3   上一份沙盒输出里取数成功 / 取数失败（`[APIFAIL]`）次数
        solution=yes   输出里有没有 `[SOLUTION]` 段
        cache=hit      答案缓存里有没有当前任务文件的答案
        llm=ask2/cmd   LLM 求助了几次 / 是否已拿到待执行命令或直接答案

    参数:
        turn: 当前回合信息
        sandbox_sent: 本回合是否真的下发了沙盒命令（server 侧知道）

    返回:
        单行状态字符串；没有任务时返回 `phase=- state=no_task`
    """
    if not turn.phase_task:
        return "phase=- state=no_task"

    _, reason = _task_answer_with_reason(turn)
    token = _task_token(turn.phase_task)
    watch = _TASK_WATCH
    repeats = rounds = fails = 0
    if watch is not None and watch.token == token:
        repeats, rounds, fails = watch.repeats, watch.rounds, watch.fails
    result = turn.last_cmd_result or ""
    llm_state = _TASK_LLM_STATE.get(token, {})
    phase = " ".join(str(turn.phase_task).split())[:60]
    flags = ""
    if llm_state.get("pending_cmd"):
        flags += "/cmd"
    if llm_state.get("answer"):
        flags += "/answer"
    return (
        f'phase="{phase}" state={reason} watch=r{repeats}/t{rounds}/f{fails} '
        f"abandoned={'yes' if _task_abandoned(turn) else 'no'} "
        f"sandbox={'发送' if sandbox_sent else '空闲'} "
        f"api={result.count(TASK_DATA_MARKER)} "
        f"fail={result.count(TASK_API_FAIL_MARKER)} "
        f"solution={'yes' if TASK_SOLUTION_MARKER in result else 'no'} "
        f"cache={'hit' if _cached_answer(turn) is not None else 'miss'} "
        f"llm=ask{int(llm_state.get('prompts') or 0)}{flags}"
    )


def sandbox_command(payload: dict[str, Any]) -> str:
    """生成提交给沙盒执行的命令（自进化任务期间使用）

    判题系统仅在接受任务到任务结束期间允许执行沙盒命令，命令的输出会在
    下一回合通过请求的 `lastCmdResult` 字段返回，再由 `_task_answer`
    解析成 `submitAnswer` 的答案。

    返回:
        需要提交给沙盒执行的shell命令；非任务期间或已有答案时返回空字符串
    """
    return _sandbox_command(Turn.load(payload))


# === 白天决策 ===


def _decide_day(
    turn: Turn,
    commands: dict[int, dict[str, Any]],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> None:
    """白天策略: 建造、采集、任务

    所有分支跑完后还有一个兜底（`_idle_gather`）：白天还没有任何指令的角色
    就近采一铲矿，保证不会整回合零动作。

    参数:
        turn: 当前回合信息
        commands: 指令输出字典（角色ID -> 指令）
        plan: 本回合的LLM计划（默认计划等价于原有策略）
    """
    # 计算需要建造的位置（塔位排序会参考敌我相对位置与LLM指定的布防方位）
    tower_sites = _calc_tower_sites(turn, plan.defend)
    wall_order = _calc_wall_order(turn)

    # 统计已建造的武器和围墙
    standing_towers = {unit.pos for unit in turn.weapons()}
    standing_walls = {unit.pos for unit in turn.walls()}
    occupied = turn.occupied_cells()

    # 计算缺少的建筑
    towers_missing = [pos for pos in tower_sites if pos not in standing_towers]
    walls_missing = [pos for pos in wall_order if pos not in standing_walls]

    # 过滤掉这一回合不能施工的位置（己方单位之外，敌方单位与机器人也算占用）
    free_towers = _buildable_sites(turn, towers_missing, occupied)
    free_walls = _buildable_sites(turn, walls_missing, occupied)

    # 已分配的位置（防止多个角色走向同一位置）
    claimed: set[Pos] = set()

    # 天黑前留出回防时间，避免夜晚武器无人操控而空转；
    # 但防线没达标时（火力不足或围墙不足两段）先抢建：多一座塔、多一段墙
    # 比多一个站在武器旁待命的角色更能提升夜晚防御
    dusk = _rounds_to_night(turn) <= DUSK_ROUNDS
    must_build = dusk and _needs_last_build(turn, free_towers, free_walls)

    # 为每个工人分配任务
    for worker in turn.workers():
        if dusk and not must_build:
            _fall_back_to_weapons(turn, worker, claimed, commands)
            continue
        _worker_day_logic(
            turn, worker, tower_sites, free_towers, free_walls, claimed,
            commands, plan,
        )

    # 开拓者行为（任务、宝藏）
    for pioneer in turn.pioneers():
        _pioneer_day_logic(
            turn, pioneer, tower_sites, wall_order, claimed, commands,
        )

    # 兜底：走到这里还没有任何指令的角色就近采一铲矿，保证白天不会整回合
    # 零动作（复盘里的"三个单位原地小步挪动、金币冻结"）。黄昏回防与任务
    # 待命是有意为之的"原地不动"，不在此列（见 `_idle_gather`）。
    if not dusk:
        for unit in turn.controllable():
            _idle_gather(turn, unit, claimed, commands)


def _buildable_sites(
    turn: Turn,
    sites: list[Pos],
    occupied: frozenset[Pos],
) -> list[Pos]:
    """筛掉这一回合不能施工的格子（下发建造指令前校验占用）

    建造指令落在被占住的格子上会直接判失败（任务书4.5.4节），这一回合的
    金币与施工都白费。己方单位之外还要避开敌方单位与机器人：`occupied_cells`
    只统计我方单位，机器人踩在塔位/墙位上时同样建不起来（复盘建议的
    "下发前校验占用并自动改最近空位"）。

    参数:
        turn: 当前回合信息
        sites: 待建的塔位或围墙位（已按优先级排序）
        occupied: 己方单位占据的格子

    返回:
        这一回合真正可以施工的位置（保持原有先后顺序）
    """
    blocked = set(occupied)
    for enemy in turn.enemies:
        if enemy.is_alive:
            blocked.update(turn.footprint(enemy))
    for robot in turn.robots:
        if robot.is_alive:
            blocked.add(robot.pos)
    return [pos for pos in sites if pos not in blocked]


def _idle_gather(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """兜底：没有任何目标的工人就近采一铲矿

    决策的各条分支都会尽量给角色安排动作，但任务点够不到、武器还没建成、
    地图上一座矿都采不了时，角色会整回合没有任何指令（复盘里的"角色原地
    挪位、金币连续多回合冻结"）。这里做最后一道兜底——按 石→铁→铜 就近
    采集，采不到就不下指令，交给下一回合重新判断。

    不打扰的情况:
        - 本回合已经有指令的角色（决策层已经给了更优先的动作）
        - 开拓者：`collect` 是工人专属动作（任务书 4.4，见 `_go_mine`），
          支它去矿边只会换来一条 `[COMMAND_ERROR]`（PK590881 的 R11–R15）；
          而且任务进行中的开拓者要留在任务点周围一格内、有任务可领的开拓者
          要赶去接任务（接任务、交任务是主要得分来源，见 `_pioneer_day_logic`），
          都轮不到采矿
        - 黄昏（调用方不调用）：回防到武器旁待命比多采一铲矿更重要，
          武器要有角色操控才会开火

    参数:
        turn: 当前回合信息
        unit: 待兜底的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    if unit.unit_id in commands:
        return
    if unit.kind != WORKER:
        return
    for mine_type in SELLABLE_MINES:
        if _go_mine(turn, unit, mine_type, claimed, commands):
            return


def _gold_left(turn: Turn, commands: dict[int, dict[str, Any]]) -> int:
    """本回合还能支配的金币（扣掉已经发出去、回合结算时才扣款的指令）

    同一回合里我们会依次给每个角色下指令，但金币要等回合结算才真正减少，
    `turn.gold` 从头到尾都是回合开始时的余额。手里只够一座塔的钱时，
    两名工人会各自下发一条 build，后一条注定失败——复盘里的"下了建造指令、
    下回合金币没扣、塔也没出现"就是这么来的。买券同理：同一回合里两名工人
    各买一张券同样会超支，所以购买指令的金额（按报文里的商店售价算）
    也一并扣掉。

    参数:
        turn: 当前回合信息
        commands: 本回合已经发出的指令

    返回:
        扣掉已发建造/购买指令后的余额
    """
    spent = 0
    for command in commands.values():
        if command.get("action") == "build" and command.get("name") in TOWER_TYPES:
            spent += WEAPON_BUILD_COST
        elif command.get("action") == "buy":
            spent += _item_price(
                turn,
                str(command.get("name") or ""),
                int(command.get("num") or 1),
            )
    return turn.gold - spent


def _item_price(turn: Turn, name: str, num: int = 1) -> int:
    """商品的总价（优先用报文里的武器商店售价，查不到时用任务书兜底价）

    "买不买得起"要按判题系统当前给的售价算：接口文档的 `weaponShopList`
    就是 `{name, price}` 清单，判题系统调价后不会再用老价格下单。
    清单里没有这件商品（报文没给价格表、或商店当天不卖）时退回任务书4.6.3
    的正式售价，兜底也为空时按 0 算——宁可高估余额，也不凭空扣钱。

    参数:
        turn: 当前回合信息
        name: 商品名称（如 WeaponUpgradeVoucher1）
        num: 数量

    返回:
        总价（金币）
    """
    for item in turn.weapon_shop:
        if str(item.get("name") or "") != name:
            continue
        price = item.get("price")
        if isinstance(price, (int, float)) and price >= 0:
            return int(price) * num
    return ITEM_FALLBACK_PRICE.get(name, 0) * num


def _worker_day_logic(
    turn: Turn,
    worker: Unit,
    tower_sites: tuple[Pos, ...],
    towers_missing: list[Pos],
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> None:
    """工人白天逻辑

    优先级: 建造武器工事 > 用背包里的石头建围墙 > 换金币/升级武器/升级围墙
            (经济分工) > 采集石头 > 施工用不上的石头卖给小贩 > 采集任意矿石

    任何分支最后都会落到"采集/交易"上，保证工人每回合都有产出，
    不会出现整回合没有任何指令的空转。

    建造位一旦认领（`claimed`）就归该工人：多个工人会分头去建不同的塔/
    围墙段，而不是几个人同时奔着同一个位置去，白走一趟还互相挡路。
    认领只是"排队"不是"独占"：一个空位都挑不到时仍然跟着已经有人赶去的塔走
    （见 `_tower_picks`），不会整队掉头去采集。

    `plan` 是LLM建议落下来的有界计划（见 `_llm_plan`）：今天要保证几座塔、
    先铺几段围墙、能不能买升级券。默认计划与改造前的行为完全一致；计划里的
    塔数还要再经 `_tower_target` 做一次金币闲置熔断，金币富余时不会被压低，
    围墙段数同理要走一遍 `_wall_target`（防守方有下限）。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        tower_sites: 武器工事的规划位置（按建造顺序）
        towers_missing: 尚未建造的武器工事位置
        walls_missing: 尚未建造的围墙位置
        claimed: 已被其他角色占用的目标集合，用于避免多个角色争抢同一格
        commands: 指令输出字典（角色ID -> 指令）
        plan: 本回合的LLM计划
    """
    economy = _is_economy_worker(turn, worker)
    # 围墙配额还没铺满时先施工、不急着变现，对应复盘建议的
    # "先补第 2 座炮塔，再沿进攻路径铺 2 段围墙"（防守方还有下限，见 `_wall_target`）
    wall_quota = len(turn.walls()) < _wall_target(turn, plan)

    # 当天的任务规划（V4）：破洞数、今天要挖几座石矿、目标队列、Tall 富余时
    # 顺路补的额外目标。计划每回合现算（模块本身不存跨回合状态），下面的
    # 采集顺序与额外目标都从这里取。
    day = _day_plan(turn, worker)
    # V4 的站位分工：0=R1 开拓者 / 1=R2 石工 / 2=R3 铜矿工人。
    # 铜矿工人只管挖矿与变现，不砌墙、不拆墙（V4："对于铜矿工人，我们不需要
    # 执行新建墙的步骤，并且其没有拆墙能力"）。
    role = _stand_role(turn, worker)

    # 防守方的开局防线（S3）：塔位优先的建造分支会把工人一直占在基地旁等金币，
    # 首段围墙因此要等到金币花光（R8~R10）才开工，甚至整局一段都没有。这里让
    # 分管经济的工人在开局窗口内跑完整的防线计划——按 `_calc_wall_order` 给出的
    # 坐标（先来敌方向）陆续铺到 `WALL_PLAN_SEGMENTS` 段，塔交给另一名工人照建
    # （只剩一名工人时不动——一双手还是先建塔）。
    if (
        role == 1  # V4：砌墙是石工（R2）的活，铜矿工人(R3)不新建墙
        and len(turn.workers()) >= 2
        and turn.team_type == "defender"
        and len(turn.walls()) < WALL_PLAN_SEGMENTS
        and _day_round(turn) < WALL_PLAN_ROUNDS
        and _early_wall(turn, worker, walls_missing, claimed, commands)
    ):
        return

    # 优先建造武器（塔数由 `_tower_target` 决定：计划配额 + 金币闲置熔断）。
    # 余额按 `_gold_left` 算：本回合已经发出去的建造指令结算时才扣款，
    # 只够一座塔的钱时第二个工人不该再下一条注定失败的 build。
    if (
        towers_missing
        and len(turn.weapons()) < _tower_target(turn, plan)
        and _gold_left(turn, commands) >= MIN_GOLD_FOR_BUILD
    ):
        # 就近认领: 每个工人挑离自己最近的那座塔，两个工人自然分头开工，
        # 而不是都盯着建造顺序表里的第一座（都挤过去的结果是另一座塔整局没人管）
        picks = _tower_picks(worker, tower_sites, towers_missing, claimed)
        picks = _retry_sites(turn, worker, picks)
        if picks:
            for _, index, site in picks:
                weapon_type = TOWER_LOADOUT[index % len(TOWER_LOADOUT)]
                if _build_or_walk(turn, worker, site, weapon_type, claimed, commands):
                    # 认领建造位: 其他工人改去下一座塔,不会几个人挤在同一个位置上
                    claimed.add(site)
                    return
            # 所有塔位这一回合都走不通（无路可走或被抢占）时不再空手过回合，
            # 而是继续往下走：有石头就建围墙，否则去采集。复盘里工人连续多回合
            # 只下 move、金币零增长，就是这里直接返回造成的。

    # 把富余资源换成战力（武器升级 > 围墙升级 > 卖矿换金币）：
    # 围墙建完时人人有责；围墙没建完时由分工里的"经济工人"负责，
    # 否则要等近二十段围墙全部铺完才会花钱，金币会闲置一整天
    # （围墙配额还没铺满时先铺墙，金币留到围墙立起来再花）
    if (not walls_missing or economy) and not wall_quota:
        if _upgrade_weapon_with_gold(
            turn, worker, claimed, commands, allow=plan.upgrade,
        ):
            return
        # 基地残血时优先保命：一张券同时买到"回满血"和"更高血量上限"
        # （战术参考 T3，聊天记录："基地快没血了给基地用一下升级券"）
        if _upgrade_station_when_low(
            turn, worker, claimed, commands, allow=plan.upgrade,
        ):
            return
        # 围墙被打残时用 10 金的修复包回满，比拆了重建省石材也省回合
        # （战术参考 T4；10 金的修复包排在 20 金的围墙升级券之前）
        if _repair_walls(
            turn, worker, claimed, commands, allow=plan.upgrade,
        ):
            return
        # 武器线花剩下的钱换成围墙升级券：复盘里三座武器与武器券都齐了以后
        # 金币再没有任何出口（"金币连续多回合冻结"），围墙是防守方唯一的正面屏障
        if _upgrade_wall_with_gold(
            turn, worker, claimed, commands, allow=plan.upgrade,
        ):
            return
        if _trade_logic(turn, worker, claimed, commands):
            return
        if not walls_missing:
            # V4 额外目标：防线已经建完，正事只剩当天那份队列，顺路的事
            # （补围墙修复包 / 补生命药剂 / 把矿石卖给小贩）排在同一层
            if _plan_extra_action(turn, worker, day, claimed, commands):
                return
            # 手里还没有可卖的矿石: 继续采集,攒够一批再换金币
            _gather_logic(turn, worker, claimed, commands, plan=day)
            return

    # 检查背包里的石头数量
    stones = worker.backpack.count(WALL_MATERIAL)

    # 金币见底时先把"围墙用不上"的矿石变现：铁/铜是纯收入，不参与砌墙。
    # 围墙配额没铺满时上面的经济分支整段被跳过（`not wall_quota` 那道门），
    # 而下面的采集分支又会先就地补石材，背包里的铁/铜于是一直变不成金币
    # ——三场复盘里防守方的金币从 R6/R8/R9 起冻结到 R17，工人背包里却始终
    # 躺着可卖的矿石。这一步只认铁/铜，石材仍然留给围墙（见 `_trade_logic`）。
    if _gold_critical(turn) and _trade_logic(
        turn, worker, claimed, commands, minerals=INCOME_MINES,
    ):
        return

    # 如果旁边有矿且石头不足,采集
    # （负责矿石变现的工人跳过这一步：否则它会一直就地采石，
    #   永远轮不到铁/铜，矿种分工就落空了；但围墙配额没铺满时全员先采石）
    if _mine_order(turn, worker, prefer_stone=wall_quota, plan=day)[0] == STONE_MINE:
        mine = _adjacent_mine(turn, worker, STONE_MINE)
        if mine is not None and stones < _stone_reserve(turn, plan, wall_quota):
            commands[worker.unit_id] = collect_command(mine)
            claimed.add(mine)
            return

    # 如果有石头,去建造围墙（认领成功就记进 claimed：几名工人自然分头铺不同
    # 段，而不是都奔向优先级最高的那一段，白走一趟还互相挡路；位置都被其他
    # 角色占住时继续往下走,别空转）
    if role != 2 and stones >= WALL_STONE_COST:
        sites = _retry_sites(
            turn, worker, [site for site in walls_missing if site not in claimed],
        )
        if sites and _build_or_walk(turn, worker, sites[0], WALL, claimed, commands):
            claimed.add(sites[0])
            return

    # 没石头(或暂时没位置建): 金币见底时先把背包里的矿石变现，再谈采集
    # （采集排在卖矿前面的旧顺序让矿石一直躺在背包里：复盘里 gold=0 冻结 13 个
    #   回合、stone:6 从头到尾没卖出去，建造/升级/买券跟着一起停摆）
    if _gold_critical(turn) and _trade_logic(turn, worker, claimed, commands):
        return

    # V4 额外目标：正事（修墙、挖队列里的矿）都排不上时，按当天的优先级
    # 顺路多做一件事（把矿卖给小贩 / 补生命药剂 / 补围墙修复包），
    # 再做原来的兜底（矿种由 `_gather_logic` 按队列顺序顺延）
    if _plan_extra_action(turn, worker, day, claimed, commands):
        return

    # 还是没着落: 就近采矿; 采不到就把背包里的矿石卖掉腾地方
    if _gather_logic(
        turn, worker, claimed, commands, prefer_stone=wall_quota, plan=day,
    ):
        return
    if _trade_logic(turn, worker, claimed, commands):
        return

    # 最后一招（V4）：正事一件都办不成时，拆开挡在路上的残墙开路
    # （一级墙血量低于一半就算"已经破了"，拆穿它走捷径，过去后再补回来）。
    # 铜矿工人没有拆墙能力，这一步跳过。
    if role != 2:
        _demolish_weak_wall(
            turn, worker, _nearest_zone(turn, STONE_MINE, worker.pos), commands,
        )


def _gold_critical(turn: Turn) -> bool:
    """金币是否已经见底（不足一座塔的造价）

    见底时经济回路成为第一优先级：建造、升级、买券都要求手里有金币，而金币
    只能靠卖矿换（任务书4.6.1）。复盘里 R6 建完第三座塔后 gold=0 连续冻结
    13 个回合、背包里的矿石一直没卖出去，全盘停摆——这时矿石留在背包里
    没有任何价值，先变现才有翻盘的可能。
    """
    return turn.gold < LOW_GOLD_THRESHOLD


def _build_backlog(turn: Turn) -> bool:
    """还有没有比"卖矿换金币"更优先的建造任务

    武器塔没满编（一座 25 金）、或者防守方的围墙配额还没铺够时，石材与金币都
    该留给建造：手里那几块石头是砌墙的料，卖了就得再去采一趟。两件事都做完
    之后石材继续攒批就只是让金币躺着（S2：复盘里 stone 从 1 块堆到 3 块、
    gold 从 R6 恒 0 到 R17），这时才按 `SELL_THRESHOLD` 零散变现。

    参数:
        turn: 当前回合信息

    返回:
        True 表示还有更优先的建造任务，石材不零散出售
    """
    if len(turn.weapons()) < LLM_MAX_TOWERS:
        return True
    return (
        turn.team_type == "defender"
        and len(turn.walls()) < DEFENDER_WALL_QUOTA
    )


def _wall_gap(turn: Turn, plan: LlmPlan) -> int:
    """围墙配额还欠几段（换算成石材块数，一段墙一块石头）

    防守方每天至少 `DEFENDER_WALL_QUOTA` 段（见 `_wall_target`），还欠的
    段数就是这一趟采石至少要带回的量。
    """
    return max(0, _wall_target(turn, plan) - len(turn.walls())) * WALL_STONE_COST


def _stone_reserve(
    turn: Turn,
    plan: LlmPlan,
    wall_quota: bool,
) -> int:
    """这一趟采石要攒到几块才回基地施工

    默认攒够 `STONE_BATCH` 一批（一趟来回多带几块，省得来回跑）。但围墙配额
    还欠着时只留够"还欠的段数"就回去开工：攒批的代价是首段围墙被拖后很久，
    而 `wall_quota` 期间经济线整段被压住（见 `_worker_day_logic` 的经济分支），
    一拖就是十几个回合——复盘里防守方首段围墙直到 R15 才出现（PK589253），
    另外两场更是全程 0 段，金币从 R6/R8 冻结到 R17。
    """
    if not wall_quota:
        return STONE_BATCH
    return max(WALL_STONE_COST, _wall_gap(turn, plan))


def _early_wall(
    turn: Turn,
    worker: Unit,
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """防守方开局防线：手里有石材就先砌墙，否则先去采石材（S3）

    只在"防守方的开局防线还没铺够、且还在开局窗口内"时由分管经济的工人执行
    （见 `_worker_day_logic`，窗口与段数见 `WALL_PLAN_ROUNDS`/
    `WALL_PLAN_SEGMENTS`）。塔位优先的建造分支会让工人一直在基地旁等金币，
    首段围墙因此拖到金币花光才开工——复盘里 PK589649 的首段围墙落到 R10、
    PK589653 全程 0 段、PK589697 拖到 R16 且只有 1 段，机器人直接贴脸打基地。

    砌墙的位置取自 `_calc_wall_order`（即复盘建议里的 WALL_PLAN）：按基地坐标
    生成、先封敌方来路，而不是"顺路在采石点旁随手砌一段"——复盘里
    (32,12)/(33,12) 那两段正是贴着采石点砌的，来敌方向反而留了口子。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        walls_missing: 尚未建造（且这一回合能施工）的围墙位置
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（砌墙或采石）
    """
    # 金币见底、手里又攥着纯收入矿石时先变现（这个问题留给下面的卖矿分支，
    # 跑去石矿转一圈只会把最后一点收入也拖后）
    if _gold_critical(turn) and any(
        worker.backpack.count(kind) for kind in INCOME_MINES
    ):
        return False

    # 手里有石头: 一段围墙只要一块石头，直接开工
    if worker.backpack.count(WALL_MATERIAL) >= WALL_STONE_COST:
        sites = _retry_sites(
            turn, worker, [site for site in walls_missing if site not in claimed],
        )
        if sites and _build_or_walk(turn, worker, sites[0], WALL, claimed, commands):
            claimed.add(sites[0])
            return True
        return False

    # 手里还没有石材: 去石矿采一铲（背包满了时留给下面的卖矿分支腾地方）
    return _go_mine(turn, worker, STONE_MINE, claimed, commands)


# === 每日任务规划（V4）===


@dataclass(frozen=True, slots=True)
class DayPlan:
    """当天的任务规划（V4：每天算一次，三个角色照着执行）

    字段:
        holes: 还差的墙位（C，按建造顺序）
        stone_mines: 今天要挖几座石矿（AR）
        queue: 目标队列（AR 个石矿打头，其余是铜矿）
        repair_rounds: 修墙路径的回合数（PT）
        end_point: 修墙路径的终点（PLE，也是这一天干完活要回来的地方）
        route_rounds: 走完队列再回 PLE 的估算回合数（LTmin）
        total_rounds: Tall = PT + LTmin
        extras: Tall 还有富余时按优先级补上的额外目标

    这是每回合现算的纯函数结果（本模块不存跨回合状态），所以"今天的计划"
    永远按当前局面重算：矿区刷新、围墙被打掉、小人夜里换了位置都会立刻反映
    到队列与耗时估算上。
    """

    holes: tuple[Pos, ...]
    stone_mines: int
    queue: tuple[str, ...]
    repair_rounds: int
    end_point: Pos | None
    route_rounds: int
    total_rounds: int
    extras: tuple[str, ...]


def _wall_holes(turn: Turn) -> tuple[Pos, ...]:
    """还差的墙位（C：应该存在的墙 - 当前已有的墙）

    "应该存在"按 `_calc_wall_order` 的规划算（基地外围第二圈，入口那一格
    不算破洞），"当前已有"按坐标逐格比对——别处顺手多砌的墙不会把正面的
    破洞抵消掉，V4 的"算出破洞数量"要的就是正面还缺几段。
    """
    built = {wall.pos for wall in turn.walls()}
    return tuple(pos for pos in _calc_wall_order(turn) if pos not in built)


def _stone_demand(turn: Turn, holes: tuple[Pos, ...]) -> int:
    """AR：今天要挖几座石矿（V4 的 ceil((C + Rmin - BR) / 10)，上限 2）

    C 是破洞数（`_wall_holes`），Rmin 是背包里要留的保底石材
    （`STONE_RESERVE_MIN`），BR 是手里的石材（两名工人的背包一起算：谁采的
    都算数）。一座矿采空是 `STONE_PER_MINE` 个回合（任务书4.2：每个矿采集
    10 次后消失，每次得 1 个），所以除以 10 得到的是"要挖几座矿"。
    上限 2：白天只有 70 个回合，第三座矿连来回的路都走不完。
    """
    held = sum(worker.backpack.count(WALL_MATERIAL) for worker in turn.workers())
    need = len(holes) + STONE_RESERVE_MIN - held
    if need <= 0:
        return 0
    return min(STONE_PLAN_MAX, -(-need // STONE_PER_MINE))  # 向上取整


def _work_queue(turn: Turn, stone_mines: int) -> tuple[str, ...]:
    """目标队列：默认 5 个铜矿，按 AR 从头替换成石矿（V4）

    石料是防线材料、铜是收入，所以石矿永远排在队首：手里没石材时先补破洞，
    挖够了再换铜矿变现（V4 原话："默认为5个铜矿，按照AR的数量替换队列前的
    元素"）。
    """
    stone_part = (STONE_MINE,) * min(stone_mines, QUEUE_TARGETS)
    return stone_part + (COPPER_MINE,) * (QUEUE_TARGETS - len(stone_part))


def _queue_order(turn: Turn, plan: DayPlan | None = None) -> tuple[str, ...]:
    """把目标队列摊成这一回合的矿种顺序（队列打头，其余矿种照旧兜底）

    队列里没有的矿种仍然排在后头：地图上只剩铁矿时石工照样采铁（空转比
    采错矿更糟），只是优先级排在队列之后。调用方已经算过当天的计划时把
    `plan` 传进来，省得把破洞数与队列再算一遍。

    参数:
        turn: 当前回合信息
        plan: 当天的计划（None 时现算一份）

    返回:
        按优先级排序的矿种元组
    """
    queue = plan.queue if plan is not None else _work_queue(
        turn, _stone_demand(turn, _wall_holes(turn)),
    )
    # 铜矿工人的队列里没有石材（V4：他不砌墙，也就不需要石料）：兜底链同样
    # 把石头放到最后——"挖石头可能反而有反作用"，只有实在没别的矿可挖时才捡
    tail = SELLABLE_MINES if STONE_MINE in queue else INCOME_MINES + (STONE_MINE,)
    return tuple(dict.fromkeys(queue + tail))


def _queue_targets(
    turn: Turn,
    queue: tuple[str, ...],
    origin: Pos,
) -> tuple[Pos, ...]:
    """队列里每个目标对应的矿点坐标（依次就近取，同类型可落到不同的矿点）

    一条队列里可能有 5 个铜矿、地图上却只有 2 座：取过的矿点不重复用，
    取不满就少算几个目标（V4 的口径也是"剔掉走不到的路径"）。
    """
    targets: list[Pos] = []
    used: set[Pos] = set()
    here = origin
    for mine_type in queue:
        mines = [pos for pos in turn.get_mines(mine_type) if pos not in used]
        if not mines:
            continue
        nearest = min(mines, key=lambda pos: (distance(here, pos), pos.x, pos.y))
        used.add(nearest)
        targets.append(nearest)
        here = nearest
    return tuple(targets)


def _route_rounds(
    start: Pos,
    targets: tuple[Pos, ...],
    end: Pos | None,
    dwell: int,
) -> int:
    """走完一串目标再回到 end 要几个回合（V4 的时间口径）

    八方向移动下两点之间的路程就是切比雪夫距离（任务书4.5.4），每个目标还要
    停下来干 `dwell` 个回合（挖矿 10 个回合、砌墙 1 个回合）。顺序按贪心
    最近邻排——V4 的原话也是"两两算好距离做成表格"，地图小、目标少，够用。

    参数:
        start: 出发位置
        targets: 依次要跑的目标（矿点或墙位）
        end: 干完活回到的位置（PLE），没有时不再算回程
        dwell: 每个目标上停留的回合数

    返回:
        估计的回合数
    """
    rounds = 0
    here = start
    remaining = list(targets)
    while remaining:
        nxt = min(remaining, key=lambda pos: (distance(here, pos), pos.x, pos.y))
        remaining.remove(nxt)
        rounds += distance(here, nxt) + dwell
        here = nxt
    if targets and end is not None:
        rounds += distance(here, end)
    return rounds


def _copper_return_plan(turn: Turn, worker: Unit) -> tuple[int, Pos | None]:
    """铜矿工人一天的收尾：PLE 是小贩（V4）

    铜矿工人不砌墙也不拆墙（没有施工任务），所以他的 PT 只是"从最后一站回到
    落脚点"的路程，PLE 直接取小贩——V4 原话："对于铜矿工人…我们直接将其 PLE
    设置为小贩，并计算小贩到 K0 点的时间为 PT"。第 4 天起的采购日再把武器商店
    接在小贩之后（V4：武器商店必须在一天的最后、且在小贩之后）。

    参数:
        turn: 当前回合信息
        worker: 铜矿工人

    返回:
        (PT, PLE) 二元组；地图上没有小贩时退回自己的站位
    """
    vendor = _nearest_zone(turn, VENDOR, worker.pos)
    if vendor is None:
        return 0, _stand_for(turn, worker)

    rounds = 0
    last = vendor
    if (
        _game_day(turn) >= FIXER_CARRIER_DAY
        and worker.backpack.count(WALL_FIXER) < FIXER_STOCK
    ):
        shop = _nearest_zone(turn, WEAPON_SHOP, vendor)
        if shop is not None:
            rounds += distance(vendor, shop) + WALL_BUILD_ROUNDS
            last = shop

    hold = _dusk_stand(turn, worker)
    if hold is not None:
        rounds += distance(last, hold)
    return rounds, vendor


def _repair_plan(
    turn: Turn,
    worker: Unit,
    holes: tuple[Pos, ...],
) -> tuple[int, Pos | None]:
    """修墙路径的回合数（PT）与终点（PLE，V4）

    修墙是石工一天里的第一件事：从当前位置把破洞一个个补上、再回到自己的
    站位（R2）——所以 PLE 就是 R2 站位，也是这一天挖完矿要回来落脚的地方
    （V4："以 PLE 为起点、R2 位置为终点执行修墙"）。没有破洞时 PT=0，
    PLE 照旧取 R2。

    铜矿工人（R3）不施工：他的 PLE 与 PT 走 `_copper_return_plan`。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        holes: 还差的墙位

    返回:
        (PT, PLE) 二元组
    """
    if _stand_role(turn, worker) == 2:
        return _copper_return_plan(turn, worker)

    stand = _stand_for(turn, worker)
    if not holes:
        return 0, stand
    return _route_rounds(worker.pos, holes, stand, WALL_BUILD_ROUNDS), stand


def _plan_extras(
    turn: Turn,
    worker: Unit,
    queue: tuple[str, ...],
    total_rounds: int,
) -> tuple[str, ...]:
    """Tall 还有富余时按优先级补上的额外目标（V4 的额外目标优先级表）

    顺序（V4 原话）:
        小贩（背包里有可卖的矿，或今天的队列里有铜矿——反正要去挖铜）>
        生命药剂（自身血量 < 80）>
        武器商店（第 3 天起、背包里的围墙修复包不足常备量）>
        铁矿 > 石头
    时间已经贴到白天的回合数上限（Tall >= 70）时一个也不加：先把当天的正事
    （修墙 + 队列里的矿）做完，顺路的事留到明天。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        queue: 当天的目标队列
        total_rounds: Tall（修墙 + 队列的估算回合数）

    返回:
        额外目标元组（按优先级排列；一个都排不上时为空）
    """
    if total_rounds >= DAY_PLAN_LIMIT:
        return ()

    copper = _stand_role(turn, worker) == 2
    extras: list[str] = []
    # 铜矿工人的 PLE 本身就是小贩（`_copper_return_plan`），不必再单列一个目标
    if not copper and (
        COPPER_MINE in queue
        or any(worker.backpack.count(kind) for kind in SELLABLE_MINES)
    ):
        extras.append(VENDOR)
    if worker.health < MEDICINE_HP:
        extras.append(MEDICINE)
    if worker.backpack.count(WALL_FIXER) < FIXER_STOCK:
        # 天数门槛与 `_stock_fixers` 保持一致：石工第 3 天起、铜工第 4 天起；
        # 计划层先按门槛列出来，免得把"今天买不到的东西"排进优先级
        restock_day = FIXER_CARRIER_DAY if copper else FIXER_RESTOCK_DAY
        if _game_day(turn) >= restock_day:
            extras.append(WEAPON_SHOP)
    # 铁矿/石头排在最后：这两个矿种本来就在 `_queue_order` 的兜底链里
    # （队列里没有时按 铜->铁->石 顺延），列在这里是为了让计划本身完整。
    # 铜矿工人不挖石头（V4："挖石头可能反而有反作用"）。
    extras.append(IRON_MINE)
    if not copper:
        extras.append(STONE_MINE)
    return tuple(extras)


def _day_plan(turn: Turn, worker: Unit) -> DayPlan:
    """当天的任务规划（V4：数破洞 -> 算 AR -> 修墙路径 -> 目标队列 -> 额外目标）

    Tall = PT + LTmin 是"修完墙 + 走完队列"的估算回合数，V4 用它判断计划排不
    排得下：目标一个一个往队列里加（Dnum 循环），只有加完仍然 Tall < 70 才留下
    ——否则会出现"计划里排了 5 座矿、一天却只跑得完 2 座"的假计划，队列本身
    是挖矿优先级（`_queue_order` 按它排矿种），排进去却干不完会让工人一直往
    远矿跑。第一个目标无论如何都留（至少得有一件事做）。

    计划不落库，每回合现算，所以夜里换了位置、矿区刷新、围墙被打掉都会立刻
    反映进来。
    """
    holes = _wall_holes(turn)
    # V4 的分工：石工（R2）按 AR 挖石料，铜矿工人（R3）整条队列都是铜/铁
    # ——"挖石头可能反而有反作用"（石料归石工管，铜工去挖石只会白占回合）
    copper = _stand_role(turn, worker) == 2
    stone_mines = 0 if copper else _stone_demand(turn, holes)
    repair_rounds, end_point = _repair_plan(turn, worker, holes)
    origin = end_point if end_point is not None else worker.pos

    queue: list[str] = []
    route_rounds = 0
    for kind in _work_queue(turn, stone_mines):
        candidate = tuple(queue + [kind])
        candidate_rounds = _route_rounds(
            origin, _queue_targets(turn, candidate, origin),
            end_point, STONE_PER_MINE,
        )
        if queue and repair_rounds + candidate_rounds >= DAY_PLAN_LIMIT:
            break  # 加到这一件就超预算了：V4 的"Dnum=n 成立、n+1 不成立"
        queue.append(kind)
        route_rounds = candidate_rounds

    total_rounds = repair_rounds + route_rounds

    # V4 的 n−1 回退：背包里压着一批卖不掉的矿石、计划却排不下"去小贩"时，
    # 把队列目标一个个砍掉，直到腾得出这一趟（V4："Dnum=n 通过、n+1 不通过，
    # 但 n 时无法再额外添加一个到达小贩后售卖的环节，则需要将 n 再次-1"）。
    # 至少留一个目标：砍到没事做不如先去把矿卖掉。
    carried = sum(worker.backpack.count(kind) for kind in SELLABLE_MINES)
    while (
        len(queue) > 1
        and carried >= SELL_BATCH
        and VENDOR not in _plan_extras(turn, worker, tuple(queue), total_rounds)
    ):
        queue.pop()
        route_rounds = _route_rounds(
            origin, _queue_targets(turn, tuple(queue), origin),
            end_point, STONE_PER_MINE,
        )
        total_rounds = repair_rounds + route_rounds

    return DayPlan(
        holes=holes,
        stone_mines=stone_mines,
        queue=tuple(queue),
        repair_rounds=repair_rounds,
        end_point=end_point,
        route_rounds=route_rounds,
        total_rounds=total_rounds,
        extras=_plan_extras(turn, worker, tuple(queue), total_rounds),
    )


def _plan_extra_action(
    turn: Turn,
    worker: Unit,
    day: DayPlan,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """按当天的额外目标优先级，能做的那件就做（V4）

    正事（修墙、挖队列里的矿、建塔）都排不上时才轮到这里。V4 的五个额外
    目标里，铁矿/石头由 `_gather_logic` 按队列兜底链处理（同一件事不写两遍），
    所以这里只认前三件：
        VENDOR      -> 把背包里的矿石卖给小贩（顺路变现）
        MEDICINE    -> 血量 < 80 时补一剂生命药剂（背包里有就喝，没有就去买）
        WEAPON_SHOP -> 补围墙修复包（夜里不能施工，这是唯一能救墙的东西）

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        day: 当天的计划
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令
    """
    for target in day.extras:
        if target == VENDOR and _trade_logic(turn, worker, claimed, commands):
            return True
        if target == MEDICINE and _take_medicine(turn, worker, claimed, commands):
            return True
        if (
            target == WEAPON_SHOP
            and _shop_is_last_stop(turn, worker)
            and _stock_fixers(turn, worker, claimed, commands)
        ):
            return True
    return False


def _shop_is_last_stop(turn: Turn, worker: Unit) -> bool:
    """去武器商店是不是"当天最后一站"（V4 的先后硬约束）

    V4 要求商店必须排在一天的最后、且在小贩之后（"必须一天中的最后时间抵达
    武器商店位于小贩之后"）。判断口径：剩下的白天回合刚好只够"走到商店 +
    回落脚点"——再多就说明还有正事（挖矿、修墙、卖矿）没干完，这一回合不
    该往商店跑。顺路（已经站在店旁）时不设门槛。
    """
    shop = _nearest_zone(turn, WEAPON_SHOP, worker.pos)
    if shop is None:
        return False
    if distance(worker.pos, shop) <= 1:
        return True
    hold = _dusk_stand(turn, worker) or _stand_for(turn, worker)
    cost = distance(worker.pos, shop)
    if hold is not None:
        cost += distance(shop, hold)
    return _rounds_to_night(turn) <= cost + 1


def _take_medicine(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """血量偏低时备一剂生命药剂：背包里有就喝，没有就去武器商店买

    V4 把"买药"列进额外目标（自身血量 < 80），夜里掉到 50 以下时由
    `_night_medicine` 喝掉（任务书4.6.3：生命药剂 10 金，使用者回满血）。
    血量健康时不做——金币优先给建造与升级，药剂只是保命。
    """
    if worker.health >= MEDICINE_HP:
        return False
    if MEDICINE in worker.backpack:
        commands[worker.unit_id] = use_command(MEDICINE)
        return True
    return _go_buy_item(
        turn, worker, claimed, commands, MEDICINE,
        _item_price(turn, MEDICINE, 1), allow=True,
    )


def _stock_fixers(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """按天数补给围墙修复包（V4）：不足常备量时去武器商店补齐

    夜里不能施工，围墙挨打后唯一的救急手段就是这个 10 金的修复包（一次把
    3×3 里的墙奶回来），所以要在白天备足。V4 的补给节奏：石工从第 3 天起
    补货（"天数=3则购买2个"），铜矿工人从第 4 天起自带 3 个；开拓者那份
    在 `_pioneer_shopping_list` 里（他没有天数门槛，缺了随时补）。
    背包里够了就不买——10 金一张，够用就行。

    参数:
        turn: 当前回合信息
        unit: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（购买或走向武器商店）
    """
    day = _game_day(turn)
    workers = turn.workers()
    # 第 2 名工人就是 V4 的铜矿工人（第 4 天起自带 3 个），其余角色（石工、
    # 只剩一名工人时的那个）从第 3 天起补货，第一次只买 2 个
    copper = len(workers) >= 2 and unit.unit_id != workers[0].unit_id
    if day < (FIXER_CARRIER_DAY if copper else FIXER_RESTOCK_DAY):
        return False

    held = unit.backpack.count(WALL_FIXER)
    want = FIXER_STOCK_FIRST if day < FIXER_CARRIER_DAY else FIXER_STOCK
    if held >= want:
        return False

    return _go_buy_item(
        turn, unit, claimed, commands, WALL_FIXER,
        _item_price(turn, WALL_FIXER, want - held), allow=True, num=want - held,
    )


def _wall_at(turn: Turn, pos: Pos) -> Unit | None:
    """坐标上的围墙（没有时返回 None）"""
    for wall in turn.walls():
        if wall.pos == pos:
            return wall
    return None


def _demolish_weak_wall(
    turn: Turn,
    unit: Unit,
    target: Pos | None,
    commands: dict[int, dict[str, Any]],
) -> bool:
    """拆开挡路的残墙（V4）：血量低于一半的一级墙就算"已经破了"，可以拆穿走捷径

    V4 的路径口径是"血量 < 50% 的一级墙视为可通过，通过时拆掉、过完再补回来"。
    拆墙不返还石材（任务书4.5.1），所以只在真的没路可走时用最后一招：挑一格
    紧挨着自己、又比现在更靠近目标的残墙拆掉，下一回合就能从缺口穿过去。
    白天才做（夜里拆墙也走不了，还要留人在武器旁）。

    参数:
        turn: 当前回合信息
        unit: 当前决策的单位
        target: 想去的目标（矿点等），None 时不做
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达拆除指令
    """
    if target is None or not turn.is_day:
        return False

    here = distance(unit.pos, target)
    for pos in get_neighbors(unit.pos):
        wall = _wall_at(turn, pos)
        if wall is None or wall.level > 1:
            continue
        if wall.health >= _wall_full_health(wall.level) * WEAK_WALL_RATIO:
            continue
        if distance(pos, target) >= here:
            continue
        commands[unit.unit_id] = remove_command(pos)
        return True
    return False


def _tower_picks(
    worker: Unit,
    tower_sites: tuple[Pos, ...],
    sites_missing: list[Pos],
    claimed: set[Pos],
) -> list[tuple[int, int, Pos]]:
    """工人这一回合可以奔的塔位（就近排序,没人认领的优先）

    先挑没人认领的塔位：第一个工人朝塔位赶路时就会把它认领下来，其他人自然
    分头去建别的塔。但"认领"只是排队，不是独占——一个空位都挑不到时（塔位被
    队友认领完了，或者只剩最后一座塔），允许跟着已经有人赶去的塔位走，
    谁先到谁施工。

    复盘里的 R2 型空过回合就是这么来的：`towers_missing` 里剩下的塔位全被
    队友认领，后一名工人挑不到任何塔位，整回合被派去采矿——计划承诺的塔在
    指令里一条都看不到，金币 50 闲置到天亮（issue #34：PK586411/536/619/647
    连续四场复发）。跟着走最多两人奔同一座塔（到场的那个开工，另一个下一回合
    自然改去别处），比整队掉头去采集划算得多。

    参数:
        worker: 当前决策的工人（用于按距离排序）
        tower_sites: 武器工事的规划位置（按建造顺序）
        sites_missing: 尚未建成、且这一回合没被占住的塔位
        claimed: 已被其他角色认领的目标集合

    返回:
        (距离, 塔位下标, 坐标) 列表，距离升序；没有待建塔位时为空
    """
    free = [
        (distance(worker.pos, site), index, site)
        for index, site in enumerate(tower_sites)
        if site in sites_missing and site not in claimed
    ]
    if not free:
        free = [
            (distance(worker.pos, site), index, site)
            for index, site in enumerate(tower_sites)
            if site in sites_missing
        ]
    return sorted(free, key=lambda pick: pick[:2])


def _retry_sites(
    turn: Turn,
    unit: Unit,
    candidates: list[Any],
) -> list[Any]:
    """上一回合的动作失败时跳过第一个候选,换一处重试

    任务书4.5.4节的碰撞规则下,移动/建造会因为"目标点被夺取"而失败;
    下一回合原地重复同一条指令往往还是失败,换一个建造位重试才能把建造
    推进下去。

    参数:
        turn: 当前回合信息
        unit: 当前决策的单位
        candidates: 按优先级排序的候选建造位

    返回:
        本回合实际可用的候选列表；上一回合没失败时原样返回
    """
    if len(candidates) > 1 and turn.action_failed(unit.unit_id):
        return candidates[1:]
    return candidates


def _is_economy_worker(turn: Turn, worker: Unit) -> bool:
    """该工人是否负责"矿石换金币"这条经济线

    分工按角色ID顺序静态划定，不依赖任何跨回合缓存：两名工人时第1名管石材
    与围墙、第2名管矿石变现，于是两名工人不会一起挤在同一个矿点上，经济也
    总有人推进；只剩一名工人（夜里阵亡后常见）时它兼顾两件事，否则金币会
    一直闲置到围墙圈建完。

    参数:
        turn: 当前回合信息
        worker: 待判断的工人

    返回:
        True 表示该工人负责卖矿换金币与武器升级
    """
    workers = turn.workers()
    if len(workers) <= 1:
        return True
    return worker.unit_id != workers[0].unit_id


def _mine_order(
    turn: Turn,
    worker: Unit,
    *,
    prefer_stone: bool = False,
    plan: DayPlan | None = None,
) -> tuple[str, ...]:
    """该工人本回合的采集矿种顺序（V4：按当天的目标队列排）

    队列是"今天要采什么"的那份规划（`_day_plan`）：AR 个石矿打头、其余是
    铜矿。两名工人时仍按分工走——第 1 名工人（石工）照着队列采，第 2 名
    工人（V4 的铜矿工人）以铜矿为主矿、铁作次选；只剩一名工人时它一个人
    兼两摊，队列里有什么先采什么（石矿打头，防线优先）。

    围墙配额还欠着时（`prefer_stone`）全员石材优先：那时候只有石材能让
    首段围墙立起来，多采一铲铁没有意义（老行为，见 `_wall_target`）。

    队列里没有的矿种仍然排在后头（`SELLABLE_MINES` 兜底）：地图上只剩
    铁矿时石工照样采铁，空转比采错矿更糟。

    参数:
        turn: 当前回合信息
        worker: 待判断的工人
        prefer_stone: 为 True 时全员石材优先（围墙配额还没铺满时用）
        plan: 当天的计划（None 时现算一份）

    返回:
        按优先级排序的矿种元组
    """
    if prefer_stone:
        return SELLABLE_MINES
    if len(turn.workers()) < 2:
        return _queue_order(turn, plan)
    if _is_economy_worker(turn, worker):
        # 第 2 名工人是 V4 的铜矿工人：铜是队列里的默认目标，铁作次选
        return ECONOMY_MINE_ORDER
    return _queue_order(turn, plan)


def _pioneer_day_logic(
    turn: Turn,
    pioneer: Unit,
    tower_sites: tuple[Pos, ...],
    wall_order: tuple[Pos, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """开拓者白天逻辑

    优先级: 维持进行中的任务（含提交答案，命中答案缓存时接取后即交卷；
            等答案期间也要有就地动作，见 `_task_wait_logic`）
            > 前往任务点领取任务（本回合走不动就原地等，不退回去跟随武器塔）
            > 守候正在冷却的任务点（白天守在下一个会开放的任务点旁，天黑前回防）
            > 跟随武器塔（只在没有任何任务点时，或者任务已被看门狗放弃时）

    任务止损（S1）：进行中的任务连续多回合没有任何进展（沙盒反复回读同一份
    文件）或已经占满 `TASK_TIMEOUT_ROUNDS` 个回合时，`_task_abandoned` 判定放弃，
    开拓者不再守任务点、`_sandbox_command` 也不再下发读文件命令，直接回基地
    跟队——离开任务点周围一格会让判题系统强制结束这个任务。
    止损前先把手里攒出来的答卷交掉、止损后转去另一个还开着的任务点
    （见下面的止损分支）：复盘里"答案到手却跟着任务一起被放弃"（PK591011 的
    R13、PK591537 的 R14）与"任务2 全程可接却没人接"（PK591595 的 R17/R18）
    都是这一步白丢的分。

    任务规则（任务书5章）:
        - 开拓者需在己方任务点周围一格内领取任务
        - 领取后离开任务点周围一格会导致任务强制结束
        - 任务点2占据两格，站在任意一格旁边都算"在任务点周围一格内"
          （判定见 `_task_distance`）
        - 任务结束后需要等待冷却，冷却期内 isValid 为 false
        - 自进化类任务需在沙盒中取数后作答，答案经 `submitAnswer` 提交

    任务点的选择是"临期优先、其次就近"：单个任务只有 15 回合时限，
    先去快过期的那个才能把两个任务的分数都拿到手。两个任务点会按这个
    顺序依次尝试，最优先的那个走不通时退而先去下一个，不会整局放弃任务。
    两个任务点都在冷却时，开拓者守在"下一个会开放的那个"旁边等它开放
    （`_next_task_position`），天黑前按返程路费提前退回基地操控武器
    （`_task_wait_margin`）——任务分是主要得分来源，守点比回基地待命划算。

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        tower_sites: 武器工事规划位置（备用，供后续扩展）
        wall_order: 围墙建造顺序，用于避免开拓者占住建造点
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    # 0. 任务已被看门狗放弃（沙盒反复回读同一份文件 / 超时交不上卷）:
    #    不再守任务点、也不再下发沙盒命令，直接回基地跟队——离开任务点周围
    #    一格会让判题系统强制结束这个任务，开拓者因此回到战斗调度，
    #    而不是被一个永远拿不到答案的任务占死（S1）
    if turn.phase_task and _task_abandoned(turn):
        task_pos = _nearest_task_position(turn, pioneer.pos)
        watch = _TASK_WATCH
        # 止损归止损，手里已经攒出来的答卷先交掉（S1）：LLM 直接给的答案
        # （`_llm_direct_answer`）与答案缓存都不依赖沙盒，而"放弃"这一步排在
        # "提交"前面——答案到手的当回合正好撞上止损线时（复盘 PK591011 的
        # R13、PK591537 的 R14 都是沙盒卡住后被直接放弃），这份答案就跟着
        # 任务一起被丢掉，白丢一次得分机会。提交次数照旧受 `TASK_SUBMIT_LIMIT`
        # 约束（`watch.submits` 已经超线时不再交），不会把提交额度刷爆；
        # 夜里不交卷（开拓者要操控武器，任务也已经被强制结束）
        if (
            task_pos is not None
            and turn.is_day
            and watch is not None
            and watch.submits <= TASK_SUBMIT_LIMIT
            and _task_distance(turn, pioneer.pos, task_pos) <= 1
        ):
            answer = _task_answer(turn) or _cached_answer(turn)
            if answer is not None:
                commands[pioneer.unit_id] = submit_answer_command(answer)
                return
        # 止损之后开拓者也不是就此收工：另一个还开着的任务点照样去领
        # （S2，复盘里任务2 全程"可接/15回合"却没人接——被止损的开拓者直接
        # 回基地跟队，剩下的白天再没碰过任务）。刚放弃的这个点要排除掉，
        # 回头再接上等于没止损（看门狗按任务标识计数，同一份沙盒还会照旧卡住）
        if (
            task_pos is not None
            and _task_distance(turn, pioneer.pos, task_pos) <= 1
            and _accept_task(turn, pioneer, claimed, commands, skip=task_pos)
        ):
            return
        _pioneer_follow_weapons(turn, pioneer, wall_order, claimed, commands)
        return

    # 1. 任务进行中: 留在任务点周围（离开会强制结束任务），拿到沙盒输出后作答
    if turn.phase_task:
        task_pos = _nearest_task_position(turn, pioneer.pos)
        if task_pos is not None:
            if _task_distance(turn, pioneer.pos, task_pos) > 1:
                target = _nearest_task_cell(turn, pioneer.pos, task_pos)
                step = _step_toward(turn, pioneer, target, claimed)
                if step is not None:
                    commands[pioneer.unit_id] = move_command(step)
                return
            # 沙盒命令的输出上一回合才返回，这里按任务标识取出本任务的答案；
            # 沙盒里之前已经读过同一个任务文件时直接交卷，不必再等一个来回
            answer = _task_answer(turn) or _cached_answer(turn)
            if answer is not None:
                commands[pioneer.unit_id] = submit_answer_command(answer)
                return
            # 还没等到答案：这一回合也得给开拓者一个就地动作，别让它零指令空转
            # （S1：复盘里开拓者 R11-R17 一条指令都没有，idle_man 一路涨到 3）
            _task_wait_logic(turn, pioneer, task_pos, claimed, commands)
            return

    # 2. 有可接取的任务: 按优先级依次尝试领取
    # 之前的实现只试最优先的那个任务点：那一格被挡住/绕不过去时开拓者就整回合
    # 放弃任务、跑去跟随武器塔，两个任务点（合计160分+160金币）都会白白过期。
    if any(task.is_valid for task in turn.player_tasks):
        # 这一步走不动时留在原地等下一个回合（同伴让开、路就通了），
        # 而不是掉头回基地跟随武器塔——那等于往相反方向走，下一回合再往外走，
        # 来回打转永远到不了任务点（复盘里的"开拓者整局在基地附近徘徊、
        # 从未靠近任务点"，两处任务点合计160分+160金币一直没人领）
        _accept_task(turn, pioneer, claimed, commands)
        return

    # 3. 任务点都在冷却中: 白天守在下一个会开放的任务点旁等它开放，天黑前再退回基地
    #    （复盘里敌方 r14 交完第一个任务、r17 就接上第二个：任务点冷却结束即可续做。
    #      旧实现只在冷却剩 3 回合时才往外走，其余回合先回基地跟随武器塔——
    #      任务点离基地十几格，一天来回一趟就是二十多个回合，两个任务点
    #      因此常常只赶得上一个）
    if _rounds_to_night(turn) > _task_wait_margin(turn, pioneer):
        task_pos = _next_task_position(turn, pioneer.pos)
        if task_pos is not None and _head_to_task(
            turn, pioneer, task_pos, claimed, commands,
        ):
            return

    # 4. 没有可接取的任务: 收工后去武器商店补货（V4），再跟随武器塔为夜晚做准备
    if _pioneer_shopping(turn, pioneer, claimed, commands):
        return
    _pioneer_follow_weapons(turn, pioneer, wall_order, claimed, commands)


def _pioneer_shopping_list(turn: Turn) -> tuple[tuple[str, int], ...]:
    """开拓者的采购清单（商品, 要备的数量），顺序即优先级（V4）

    V4 把"围墙与武器升级"这件事交给开拓者：任务做完/任务点冷却时顺路把券
    和修复包买齐，工人腾出手去跑当天的矿物队列。优先级是 V4 的原话：
        3×武器升级券1 > 1×基地升级券1 > 围墙修复包（不足 3 个时补齐）
        > 10×围墙升级券1 > 3×武器升级券2 > 基地升级券2
    券按"当前等级用得上"的那张列：武器/基地还在 level1 时只需要第一张，
    已经到 level2 才轮到第二张（买早了用不上，还占背包）。

    参数:
        turn: 当前回合信息

    返回:
        (商品名, 要备的数量) 元组，按优先级排列
    """
    station = turn.station()
    weapons = turn.weapons()
    items: list[tuple[str, int]] = []
    if any(weapon.level == 1 for weapon in weapons):
        items.append((WEAPON_UPGRADE_VOUCHER, PIONEER_WEAPON_VOUCHERS))
    if station is not None and station.level == 1:
        items.append((STATION_UPGRADE_VOUCHER, 1))
    items.append((WALL_FIXER, FIXER_STOCK))
    if any(wall.level == 1 for wall in turn.walls()):
        items.append((WALL_UPGRADE_VOUCHER, PIONEER_WALL_VOUCHERS))
    if any(weapon.level == 2 for weapon in weapons):
        items.append((WEAPON_UPGRADE_VOUCHER2, PIONEER_WEAPON_VOUCHERS))
    if station is not None and station.level == 2:
        items.append((STATION_UPGRADE_VOUCHER2, 1))
    return tuple(items)


def _pioneer_shopping(
    turn: Turn,
    pioneer: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """开拓者收工后去武器商店补货（V4 的采购优先级）

    按 `_pioneer_shopping_list` 的顺序一件件试：背包里还缺几件就买几件，
    买不起（或不在店旁、天快黑了）就试下一件，一件都办不成时不下指令，
    交给下面的"跟随武器塔"——夜里武器要有人操控，采购不能把开拓者留在
    地图另一头。

    塔还没建齐时给工人留一座塔的钱（`reserve`）：券可以晚一天买，火力
    成型晚一天就可能被机器人贴脸打基地。

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（购买或走向武器商店）
    """
    reserve = WEAPON_BUILD_COST if len(turn.weapons()) < LLM_MAX_TOWERS else 0
    for item, want in _pioneer_shopping_list(turn):
        held = pioneer.backpack.count(item)
        if held >= want:
            continue
        if _go_buy_item(
            turn, pioneer, claimed, commands, item,
            _item_price(turn, item, want - held),
            allow=True, reserve=reserve, num=want - held,
        ):
            return True
    return False


def _task_wait_logic(
    turn: Turn,
    pioneer: Unit,
    task_pos: Pos,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """任务进行中、答案还没到时的就地动作（S1）

    自进化任务的答案要等下一回合的沙盒输出，而任务期间开拓者必须留在任务点
    周围一格内（离开会强制结束任务），于是它常常整回合什么都不做——复盘里
    "开拓者 20011 R11-R17 无任何指令、idle_man 升至 3"，一个可操控单位整段
    白天白搭。这里给它安排一条就地能做的动作：在"仍然落在任务点周围一格内"
    的相邻格里挪一步，方向朝最近的武器塔（不离开任务圈，顺带为夜晚操控武器
    省一段路）。走不通时不下指令——原地待命比乱走安全（任务点的有效性只看
    距离），但只要挪得动，这名角色就不会再零指令空转。

    这里**不安排采集**：任务书 4.4 的指令表里 `collect` 只许工人用，开拓者
    发了也是整条被驳回（复盘 PK590881 的 R11–R15 就是这么把任务窗口耗掉的，
    见 `_go_mine`）——"顺手采一铲"的收益远小于白白丢掉的一整个任务回合。

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者（此刻已在任务点周围一格内）
        task_pos: 进行中任务的任务点坐标
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    # 在任务圈内朝最近的武器塔挪一步（采集是工人专属动作，见上面的说明）
    weapons = turn.weapons()
    if not weapons:
        return
    nearest = min(
        weapons,
        key=lambda weapon: (
            distance(pioneer.pos, weapon.pos), weapon.pos.x, weapon.pos.y,
        ),
    )
    here = distance(pioneer.pos, nearest.pos)
    build_sites = _reserved_build_sites(turn)
    blocked = turn.blocked(pioneer)

    best: Pos | None = None
    for pos in get_neighbors(pioneer.pos):
        if not turn.land(pos) or pos in blocked or pos in claimed:
            continue
        # 白天不占建造点：站上去会让那座塔/那段墙整局建不起来
        if pos in build_sites:
            continue
        # 只在任务点周围一格内挪动，离开就会让任务强制结束
        if _task_distance(turn, pos, task_pos) > 1:
            continue
        if distance(pos, nearest.pos) >= here:
            continue
        if best is None or (distance(pos, nearest.pos), pos.x, pos.y) < (
            distance(best, nearest.pos), best.x, best.y,
        ):
            best = pos

    if best is not None:
        commands[pioneer.unit_id] = move_command(best)
        claimed.add(best)


def _pioneer_follow_weapons(
    turn: Turn,
    pioneer: Unit,
    wall_order: tuple[Pos, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """开拓者跟随武器塔（没有任务可做，或任务已被看门狗放弃时）

    武器工事要有角色操控才会开火（任务书4.4节），所以没有任务时开拓者守在
    最近的武器旁待命；`wall_order` 用来避开围墙的建造点——站上去会把那一格
    占住，那段围墙整局都建不起来（见 `_valid_stand_cells`）。

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        wall_order: 围墙建造顺序，用于避免开拓者占住建造点
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    weapons = turn.weapons()
    if not weapons:
        return

    # 找到最近的武器塔
    nearest_weapon = min(weapons, key=lambda w: distance(pioneer.pos, w.pos))

    # 如果已经在武器旁边且不在围墙建造点上,不动
    if distance(pioneer.pos, nearest_weapon.pos) <= 1:
        if pioneer.pos not in wall_order:
            return

    # 否则向武器塔靠近（但只在基地周围移动）
    step = _step_toward(turn, pioneer, nearest_weapon.pos, claimed, inside_only=True)
    if step is not None:
        commands[pioneer.unit_id] = move_command(step)


def _head_to_task(
    turn: Turn,
    pioneer: Unit,
    task_pos: Pos,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """走向任务点，到了就领取任务

    到达判定用任务点的全部格子（见 `_task_distance`）：任务点2占据两格，
    开拓者站在另一格旁边同样可以领取，不必再绕到报文给出的那一格。

    返回:
        True 表示本回合已下达指令（领取或移动）
    """
    if _task_distance(turn, pioneer.pos, task_pos) <= 1:
        commands[pioneer.unit_id] = accept_task_command()
        return True

    step = _step_toward(
        turn, pioneer, _nearest_task_cell(turn, pioneer.pos, task_pos), claimed,
    )
    if step is None:
        return False

    commands[pioneer.unit_id] = move_command(step)
    return True


def _accept_task(
    turn: Turn,
    pioneer: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    skip: Pos | None = None,
) -> bool:
    """按优先级依次尝试领取可接取的任务点

    顺序是"临期优先、其次就近"（`timeoutRounds` 小的先做）：单个任务只有
    15 回合时限（任务书5章），先去快过期的那个，两个任务的分数才都有机会
    拿到手；走不通的那个退而试下一个，不会整局放弃任务。

    `skip` 是"刚被看门狗放弃的那个任务点"（止损分支传入）：止损之后开拓者
    回头再把同一个任务点接上等于没止损——看门狗按任务标识计数，同一个任务的
    沙盒还会照旧卡住，那十来个回合等于白等第二遍。任务点2占两格（`_task_cells_of`），
    所以按 `_task_distance` 判定"是不是同一个任务点"，而不是逐格比对坐标。

    参数:
        turn: 当前回合信息
        pioneer: 当前决策的开拓者
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        skip: 不要接的任务点（刚放弃的那个）；没有时传 None

    返回:
        True 表示本回合已下达指令（领取或走向某个任务点）
    """
    ordered = sorted(turn.player_tasks, key=lambda task: (
        task.timeout_rounds if task.timeout_rounds > 0 else TASK_TIMEOUT_UNKNOWN,
        distance(pioneer.pos, task.task_position),
        task.task_position.x,
        task.task_position.y,
    ))
    for task in ordered:
        if not task.is_valid:
            continue
        if skip is not None and _task_distance(
            turn, task.task_position, skip,
        ) <= 1:
            continue
        if _head_to_task(turn, pioneer, task.task_position, claimed, commands):
            return True
    return False


def _task_point_cells(turn: Turn) -> tuple[Pos, ...]:
    """己方任务点在地图上占据的全部格子

    任务点2占据两格（任务书4.6.2节），报文与地图里都可能只给出其中一格，
    因此把 playerTasks 的坐标与地图上本阵营任务点的坐标合并去重。
    """
    kinds = _TASK_POINTS_BY_TEAM.get(turn.team_type, ())
    cells = [task.task_position for task in turn.player_tasks]
    cells += [pos for pos, kind in turn.zones.items() if kind in kinds]
    return tuple(dict.fromkeys(cells))


def _task_cells_of(turn: Turn, task_pos: Pos) -> tuple[Pos, ...]:
    """某个任务点占据的格子：报文给出的那一格 + 与它相邻的同阵营任务点格子"""
    return (task_pos,) + tuple(
        cell for cell in _task_point_cells(turn)
        if cell != task_pos and distance(cell, task_pos) <= 1
    )


def _task_distance(turn: Turn, origin: Pos, task_pos: Pos) -> int:
    """origin 到该任务点的距离（到它的任意一格）"""
    return min(distance(origin, cell) for cell in _task_cells_of(turn, task_pos))


def _nearest_task_cell(turn: Turn, origin: Pos, task_pos: Pos) -> Pos:
    """任务点里离 origin 最近的一格（两格任务点走最近的那格）"""
    return min(
        _task_cells_of(turn, task_pos),
        key=lambda cell: (distance(origin, cell), cell.x, cell.y),
    )


def _next_task_position(turn: Turn, origin: Pos) -> Pos | None:
    """下一个会开放的任务点坐标（冷却剩余最少，其次就近）

    任务点冷却期间开拓者守在它旁边等开放（见 `_pioneer_day_logic`），
    守错任务点会白白错过另一个更早开放的任务点。报文没有 playerTasks 时
    退回地图上的己方任务点（`_nearest_task_position`）。
    """
    if not turn.player_tasks:
        return _nearest_task_position(turn, origin)
    return min(
        turn.player_tasks,
        key=lambda task: (
            task.cold_down_rounds,
            distance(origin, task.task_position),
            task.task_position.x,
            task.task_position.y,
        ),
    ).task_position


def _task_wait_margin(turn: Turn, pioneer: Unit) -> int:
    """开拓者守在任务点旁时，天黑前要留出的返程回合数

    任务点离基地十几格，按"回基地的路费 + 提前回防的回合数"留足返程时间，
    否则只顾守任务点会让夜晚的武器没人操控（任务书4.4节：武器要有角色
    操控才会开火）。开拓者本来就在基地旁守点时，退化成 `DUSK_ROUNDS`。
    """
    station = turn.station()
    if station is None:
        return DUSK_ROUNDS
    return distance(pioneer.pos, station.pos) + DUSK_ROUNDS


def _trade_logic(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    minerals: tuple[str, ...] = SELLABLE_MINES,
) -> bool:
    """资源交易逻辑：把背包里多余的矿石卖给小贩换金币

    小贩收购价随世界新闻波动（任务书4.6.1节），卖出所得可用于购买升级券。
    背包还有空间时攒够一批再卖；背包已经满了就先卖掉手头最多的那种矿腾地方，
    既换到金币又避免工人因为塞满背包而无法采集。小贩离得太远、跑一趟回不来
    时不出门，先就近采集，等靠近了再卖（见 `_can_return_before_dusk`）。

    金币见底（低于一座塔的造价）时不再等凑够一批：手里有多少卖多少。复盘里
    R6 建完第三座塔后 gold=0 冻结 13 个回合，而背包里 stone:6 一直躺在背包里
    ——等凑够 `SELL_BATCH` 的话，这点矿石永远变不成钱，经济也就永远转不起来。
    铁/铜这类纯收入矿石同样不等攒批（`MINERAL_SELL_THRESHOLD`）：它们不参与
    砌墙，留在背包里只是占地方，金币没见底也该有几块卖几块。
    石材在"建造线已经走完"时同样不等攒批（`SELL_THRESHOLD`，见 `_build_backlog`）。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        minerals: 这次允许卖出的矿种（默认全部可卖；围墙配额没铺满时
                  只卖铁/铜这类纯收入矿石，石材留给围墙）

    返回:
        True 表示本回合已下达指令（贩卖或走向小贩）
    """
    if not minerals:
        return False
    quantities = [
        (mine_type, worker.backpack.count(mine_type))
        for mine_type in minerals
    ]
    # 数量最多的那种优先卖；数量相同时取 minerals 里靠前的那种
    mine_type, amount = max(
        quantities,
        key=lambda item: (item[1], -minerals.index(item[0])),
    )
    # 背包满了就卖一批腾地方,否则等攒够一批再卖；金币见底时有几块卖几块。
    # 铁/铜是纯收入矿石（不参与砌墙），攒批没有任何好处——有几块卖几块，
    # 每回合都能有一笔进账（S2：复盘里"只采不卖、金币冻结在 0"）。
    # 石材另算：还有塔/墙要建时它留着砌墙（继续攒批），建造线走完之后攒批就
    # 只是让金币接着躺着了，攒够 `SELL_THRESHOLD` 块就变现。
    batch = SELL_BATCH
    if worker.backpack_full or _gold_critical(turn):
        batch = 1
    elif mine_type in INCOME_MINES:
        batch = MINERAL_SELL_THRESHOLD
    elif not _build_backlog(turn):
        batch = SELL_THRESHOLD
    if amount < batch:
        return False

    vendor = _nearest_zone(turn, VENDOR, worker.pos)
    if vendor is None:
        return False

    # 已在小贩旁边: 直接贩卖
    if distance(worker.pos, vendor) <= 1:
        commands[worker.unit_id] = sell_command(mine_type, amount)
        return True

    # 否则走向小贩（路太远、天黑前回不来时不出这趟门）
    if not _can_return_before_dusk(turn, worker, vendor):
        return False
    step = _step_toward(turn, worker, vendor, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True

    return False


def _upgrade_weapon_with_gold(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow: bool = True,
) -> bool:
    """把富余金币换成武器升级券并用于武器（level1 -> level2 -> level3）

    任务书4.6.3节：升级券在武器商店购买，需在目标武器周围一格内使用，
    升级后武器恢复到满血，攻击力与射程同时提升。level2 的武器还能用
    升级券2 再升一级（level3 的火箭发射台是全图射程），金币因此总有下一级
    可升，不会卡在"三座都到 level2 之后钱没处花"。

    因为升级券先买后用、跨回合存在背包里，这里按背包内容分两步走：
    背包里已有券就直接去武器旁使用，否则到武器商店购买。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        allow: 是否允许花金币买券（LLM计划里"今天不买"时为 False，
               已经买好的券仍然照用不误）

    返回:
        True 表示本回合已下达指令（购买/使用/移动），调用方应直接返回
    """
    options = _upgrade_options(
        worker, turn.weapons(), WEAPON_UPGRADE_VOUCHERS, claimed,
    )
    # 武器线是金币的第一去处：不留储备，够一张券的钱就买
    return _spend_on_upgrade(
        turn, worker, claimed, commands, options, allow=allow, reserve=0,
    )


def _upgrade_wall_with_gold(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow: bool = True,
) -> bool:
    """把武器线花剩下的金币换成围墙升级券并用于围墙（level1 -> level2 -> level3）

    复盘里三场都出现"金币连续多回合冻结在 50 甚至更久、无建造无购买"：
    三座武器建完、武器券也买过之后，金币再没有任何出口（586322/586323/586377）。
    围墙是防守方唯一的正面屏障，升级券只要 20 金（任务书4.6.3），
    升级后围墙还会回满血，正好接住这笔闲钱。

    金币要先扣掉武器线的储备（`_gold_reserve`）：一张武器券 100~150 金
    换 10 点攻击力与一段射程，比一面围墙多 500 血划算得多，不能把攒着
    买武器券的钱花在围墙上。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        allow: 是否允许花金币买券（LLM计划里"今天不买"时为 False，
               已经买好的券仍然照用不误）

    返回:
        True 表示本回合已下达指令（购买/使用/移动），调用方应直接返回
    """
    options = _upgrade_options(
        worker, turn.walls(), WALL_UPGRADE_VOUCHERS, claimed,
    )
    return _spend_on_upgrade(
        turn, worker, claimed, commands, options,
        allow=allow, reserve=_gold_reserve(turn, worker, claimed),
    )


def _upgrade_options(
    worker: Unit,
    buildings: tuple[Unit, ...],
    vouchers: dict[int, tuple[str, int]],
    claimed: set[Pos],
) -> list[tuple[int, str, Unit]]:
    """把"还能升级的建筑"排成候选表（最便宜的升级优先，其次离工人最近）

    满级的建筑、以及已经被其他角色认领的建筑都不在候选里（`claimed` 同一
    回合内共享，两个工人不会挤到同一座建筑旁边）。价格用任务书4.6.3 的
    售价：报文里没有券的价格表，只有武器的 `weaponShopList` 才有。

    参数:
        worker: 当前决策的工人（用于按距离排序）
        buildings: 待筛选的建筑（武器或围墙）
        vouchers: 升级券表（当前等级 -> (券名, 价格)）
        claimed: 已被其他角色占用的目标集合

    返回:
        (券价, 券名, 建筑) 三元组列表，价格升序；没有可升级的建筑时为空
    """
    options: list[tuple[int, str, Unit]] = []
    for building in buildings:
        entry = vouchers.get(building.level)
        if entry is None or building.pos in claimed:
            continue
        voucher, price = entry
        options.append((price, voucher, building))
    options.sort(key=lambda option: (
        option[0],
        distance(worker.pos, option[2].pos),
        option[2].pos.x,
        option[2].pos.y,
    ))
    return options


def _gold_reserve(turn: Turn, worker: Unit, claimed: set[Pos]) -> int:
    """买围墙券之前要留出的金币储备（武器线还没花完的钱）

    复盘建议的"金币优先转化战力"在这里落成顺序：还差武器塔（每座 25 金）
    时先留一座塔的钱，塔齐了就留"下一张武器券"的钱，武器满编满级之后
    储备为 0——此时金币可以放心换成围墙券，不会再有金币躺在手里。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合

    返回:
        买围墙券前至少要留下的金币
    """
    if len(turn.weapons()) < LLM_MAX_TOWERS:
        return WEAPON_BUILD_COST
    options = _upgrade_options(
        worker, turn.weapons(), WEAPON_UPGRADE_VOUCHERS, claimed,
    )
    return options[0][0] if options else 0


def _spend_on_upgrade(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    options: list[tuple[int, str, Unit]],
    *,
    allow: bool,
    reserve: int,
) -> bool:
    """把券用在对应的建筑上，没有券时（金币够、计划允许）去武器商店买一张

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        options: `_upgrade_options` 给出的候选表（价格升序）
        allow: 是否允许今天买券（False 时已经买好的券仍然照用不误）
        reserve: 买券前要留下的金币储备（围墙券不能把武器券的钱花掉）

    返回:
        True 表示本回合已下达指令（使用、购买或移动）
    """
    # 没有可升级的建筑（武器还没建、围墙还没铺、或者都已经满级）时无从下手
    if not options:
        return False

    # 1. 身上有券: 去对应建筑旁使用（背包里可能同时躺着两张券，级别不同）
    for _, voucher, building in options:
        if voucher in worker.backpack:
            return _use_voucher_at(
                turn, worker, claimed, commands, voucher, building.pos,
            )

    # 2. 金币足够且计划允许: 去武器商店买最便宜的那张券
    #    （背包满了买不了，券进不来，先卖矿腾地方）
    _, voucher, _ = options[0]
    return _go_buy_item(
        turn, worker, claimed, commands, voucher,
        _item_price(turn, voucher), allow=allow, reserve=reserve,
    )


def _go_buy_item(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    item: str,
    price: int,
    *,
    allow: bool,
    reserve: int = 0,
    num: int = 1,
) -> bool:
    """去武器商店买一件道具：已经在店旁就直接买，否则走过去

    背包满时买不进来（任务书4.6.3：背包空间不足则购买失败），先由调用方
    去卖矿腾地方；金币不足或计划不允许时不出门。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        item: 道具名（如 WallFixer / StationUpgradeVoucher1）
        price: 这一单的总价（数量按 num 算，来自 weaponShopList，取不到时用兜底价）
        allow: 是否允许本回合花钱买
        reserve: 买之前要留下的金币（给更高优先级的支出留钱）
        num: 一次买几件（修复包这类消耗品按常备量一次补齐）

    返回:
        True 表示本回合已下达指令（购买或移动）
    """
    if not allow or worker.backpack_full:
        return False
    if _gold_left(turn, commands) < reserve + price:
        return False

    shop = _nearest_zone(turn, WEAPON_SHOP, worker.pos)
    if shop is None:
        return False

    if distance(worker.pos, shop) <= 1:
        commands[worker.unit_id] = buy_command(item, num)
        return True

    if not _can_return_before_dusk(turn, worker, shop):
        return False
    step = _step_toward(turn, worker, shop, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True
    return False


def _station_full_health(level: int) -> int:
    """基地该等级的满血值（任务书4.5.1：level1/2/3 = 1500/3000/4500）"""
    return station_full_health(level)


def _upgrade_station_when_low(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow: bool = True,
) -> bool:
    """基地残血时用基地升级券：回满血 + 顺便升级（战术参考 T3）

    聊天记录原话："基地快没血了给基地用一下升级券，这样血可以回满还顺便升级"。
    一张 100/150 金的券同时买到"满血复活"和"更高的血量上限"，是守卫基地
    最划算的一笔支出；满级（level3）或血量还健康时不做。

    券名与价格取自任务书4.6.3（StationUpgradeVoucher1/2 = 100/150 金），
    `docs/request.txt` 的 weaponShopList 里也确认在售。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        allow: 是否允许本回合买券（已有的券不受限制）

    返回:
        True 表示本回合已下达指令（用券、买券或移动）
    """
    station = turn.station()
    if station is None or not station.is_alive or station.level >= 3:
        return False
    if station.health > _station_full_health(station.level) * STATION_LOW_HP_RATIO:
        return False

    options = _upgrade_options(
        worker, (station,), STATION_UPGRADE_VOUCHERS, claimed,
    )
    if not options:
        return False
    return _spend_on_upgrade(
        turn, worker, claimed, commands, options, allow=allow, reserve=0,
    )


def _wall_full_health(level: int) -> int:
    """围墙该等级的满血值（任务书4.5.1：level1/2/3 = 1000/1500/2000）"""
    return wall_full_health(level)


def _damaged_walls(turn: Turn) -> tuple[Unit, ...]:
    """掉血的围墙（报文只给 health，满血值按等级查任务书4.5.1）"""
    return tuple(
        wall for wall in turn.walls()
        if wall.health < _wall_full_health(wall.level)
    )


def _best_repair_spot(
    turn: Turn,
    damaged: tuple[Unit, ...],
    origin: Pos,
) -> Unit | None:
    """挑一段"周围残血墙最密集"的围墙作为修复目标

    任务书 4.6.3 写的是"目标坐标所在围墙回满血"（单体口径），聊天记录说的是
    "3×3 一次奶 5 段"（群体口径）。两者取交集：目标定在残血墙最密集处——
    单体口径下奶到最该奶的那一段，群体口径下一次覆盖最多段
    （截图 `新版站位-角落WallFixer.png` 的角落站位正是这个意思）。

    参数:
        turn: 当前回合信息
        damaged: 残血围墙
        origin: 决策单位的当前位置（同分时取更近的）

    返回:
        修复目标围墙；没有残血墙时返回 None
    """
    if not damaged:
        return None

    occupied = {wall.pos for wall in damaged}

    def coverage(wall: Unit) -> int:
        """以该墙为中心 WALL_FIXER_RADIUS 范围内还有多少段残血墙（含自己）"""
        return sum(
            1 for pos in occupied
            if distance(wall.pos, pos) <= WALL_FIXER_RADIUS
        )

    return min(
        damaged,
        key=lambda wall: (
            -coverage(wall),                 # 覆盖越多越好
            distance(origin, wall.pos),      # 越近越好
            wall.health,                     # 血越少越优先
            wall.pos.x, wall.pos.y,
        ),
    )


def _repair_walls(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    allow: bool = True,
) -> bool:
    """用围墙修复包把残血围墙回满（战术参考 T4）

    围墙被打残后原本只能拆了重建（费石材又费回合），修复包只要 10 金
    （任务书4.6.3），把目标围墙直接回满血。残血墙少于
    `WALL_REPAIR_MIN_TARGETS` 段时不值得跑一趟——拆了重建更省。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        allow: 是否允许本回合买修复包

    返回:
        True 表示本回合已下达指令（使用、购买或移动）
    """
    damaged = _damaged_walls(turn)
    if len(damaged) < WALL_REPAIR_MIN_TARGETS:
        return False

    target = _best_repair_spot(turn, damaged, worker.pos)
    if target is None:
        return False

    # 手里没有修复包: 先去武器商店买一张（背包满了买不进来，交给卖矿分支）
    if WALL_FIXER not in worker.backpack:
        return _go_buy_item(
            turn, worker, claimed, commands, WALL_FIXER,
            _item_price(turn, WALL_FIXER), allow=allow,
        )

    # 修复包要站在待修复围墙一格范围内使用（任务书4.6.3）
    if distance(worker.pos, target.pos) <= 1:
        commands[worker.unit_id] = use_command(WALL_FIXER, target.pos)
        claimed.add(target.pos)
        return True

    if not _can_return_before_dusk(turn, worker, target.pos):
        return False
    step = _step_toward(turn, worker, target.pos, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True
    return False


def _use_voucher_at(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    voucher: str,
    target: Pos,
) -> bool:
    """在 target 上使用背包里的券：已经在旁边就直接用，否则走过去

    券先买后用、跨回合存在背包里（任务书4.6.3节：只有站在目标建筑周围
    一格内使用才会生效），所以"走过去"这一步本身就是本回合的指令。
    路太远、跑一趟赶不回天黑前时不出门，免得夜晚的武器没人操控。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        voucher: 要使用的券
        target: 券的目标建筑坐标

    返回:
        True 表示本回合已下达指令（使用或移动）
    """
    if distance(worker.pos, target) <= 1:
        commands[worker.unit_id] = use_command(voucher, target)
        claimed.add(target)
        return True

    if not _can_return_before_dusk(turn, worker, target):
        return False
    step = _step_toward(turn, worker, target, claimed)
    if step is not None:
        commands[worker.unit_id] = move_command(step)
        return True
    return False


def _rounds_to_night(turn: Turn) -> int:
    """距离天黑还剩多少回合（含当前回合）"""
    day_round = (turn.round_no - 1) % ROUNDS_PER_DAY
    return DAY_ROUNDS - day_round


def _can_return_before_dusk(turn: Turn, unit: Unit, target: Pos) -> bool:
    """去 target 办完事，还来得及在天黑前回到基地吗

    采集和建造都在基地旁边，来回一两回合就够；卖矿、买升级券却要跑到地图
    另一头，跑远了回不来就会让夜晚的武器没人操控（任务书4.4节：武器要有
    角色操控才会开火）。这里按"往返路费 + 提前回防的回合数"做个粗算，
    路太远就不出这趟门。

    参数:
        turn: 当前回合信息
        unit: 准备出门的单位
        target: 目的地坐标

    返回:
        True 表示这一趟来回之后还剩回防时间
    """
    return _rounds_to_night(turn) > 2 * distance(unit.pos, target) + DUSK_ROUNDS


def _needs_last_build(
    turn: Turn,
    towers_missing: list[Pos],
    walls_missing: list[Pos],
) -> bool:
    """天黑前是否还要抢建（入夜前的火力/防线预算检查）

    只在天黑前的最后几个回合使用。火力不够又买得起塔时先补塔，手里有石头
    却还没铺够 `MIN_WALLS_BEFORE_NIGHT` 段围墙时先补墙——首夜只有一座光塔、
    零段围墙时防线没有纵深（复盘里的"首夜崩盘"）。这两件事都在基地旁边
    完成，做完再回防也来得及，比整队提前回防更划算。

    参数:
        turn: 当前回合信息
        towers_missing: 尚未建造（且未被占据）的武器位置
        walls_missing: 尚未建造（且未被占据）的围墙位置

    返回:
        True 表示本回合应当继续建造而不是回防
    """
    if (
        towers_missing
        and len(turn.weapons()) < MIN_TOWERS_BEFORE_NIGHT
        and turn.gold >= WEAPON_BUILD_COST
    ):
        return True

    if walls_missing and len(turn.walls()) < MIN_WALLS_BEFORE_NIGHT:
        return any(WALL_MATERIAL in worker.backpack for worker in turn.workers())

    return False


def _fall_back_to_weapons(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """天黑前把角色召回最近的武器旁待命

    武器工事必须由角色操控才会开火（任务书4.4节），提前回防可以避免
    夜晚首个回合武器无人操控而白白空转。

    V4 的站位优先：自己那个退守位（`_dusk_stand`，R3 是 K0）正好挨着某座
    武器时先回站位——站在那里照样能操控武器，而站位在墙内一圈的角上，
    围墙修复包的 3×3 一次能奶 5 段墙（V4 的"角落站位"），夜里补墙最顺手。
    站位挨不着武器（或者走不过去）时退回原来的"就近武器"。

    参数:
        turn: 当前回合信息
        unit: 待召回的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    weapons = turn.weapons()
    if not weapons:
        return

    stand = _dusk_stand(turn, unit)
    if stand is not None and any(
        distance(stand, weapon.pos) <= 1 for weapon in weapons
    ):
        if distance(unit.pos, stand) <= 1:
            return
        step = _step_toward(turn, unit, stand, claimed)
        if step is not None:
            commands[unit.unit_id] = move_command(step)
            return

    nearest = min(
        weapons,
        key=lambda w: (distance(unit.pos, w.pos), w.pos.x, w.pos.y),
    )
    if distance(unit.pos, nearest.pos) <= 1:
        return

    step = _step_toward(turn, unit, nearest.pos, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)


def _plug_wall_gap(
    turn: Turn,
    unit: Unit,
    holes: tuple[Pos, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    weapon: Unit | None = None,
) -> bool:
    """用身体堵住围墙缺口（T6 人肉城墙）

    石材不够、或者墙夜里被拆掉时，`_wall_holes` 里的每一格都是机器人直通
    基地的门——墙砌不起来，人站上去也一样：角色占住的格子不可通行，机器人
    要么先把这个肉盾打掉，要么绕开（见战术参考的 `人肉城墙堵缺口.png`）。
    缺口按 `_calc_wall_order` 的顺序给，先堵的就是敌方来路那一段；规划好的
    入口不在里面（那是自己人进出的通道，堵上会把队友关在外头）。

    站位要站到缺口那一格**本身**，所以这里不用 `_step_toward`（它落在目标
    的邻格上），直接按 A* 往那一格走（`next_step`）。

    `weapon` 不是 None 时只挑"那一格还挨着这座武器"的缺口：这类角色手里有
    武器要操控（八方向距离 1 才算操控得到），站远了武器就没人管了，所以只有
    "站在缺口上照样够得着武器"的格子才轮得到它——塔位与外墙本来就贴在一起
    （`_side_corridor` 的墙角位与 `_wall_ring` 的外圈只差一格），这种格子通常
    就在武器旁边。

    已经在缺口上时返回 True 但不下指令：不动就是堵着，而调用方拿到 True
    就不会再把它支使到别处去。

    只在夜里用（调用方是 `_decide_night`）：白天那几个缺口既是工人进出取矿的
    路，也是当天要砌墙的施工位，站上去只会把自家人堵在里面、让墙更晚立起来。

    参数:
        turn: 当前回合信息
        unit: 待调度的角色
        holes: 本回合的围墙缺口（`_wall_holes`，按建造顺序）
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        weapon: 该角色负责的武器；没有武器时传 None（全按优先级挑）

    返回:
        True 表示这一回合的站位已经定了（正走向某个缺口，或已经堵在上面）
    """
    if not holes:
        return False

    if weapon is None:
        cells: list[Pos] = list(holes)
    else:
        hole_set = set(holes)
        cells = [pos for pos in get_neighbors(weapon.pos) if pos in hole_set]

    # 已经在缺口上就别再挪窝：站着不动就是堵着，再挑一格反而会把门让开
    if unit.pos in cells:
        claimed.add(unit.pos)
        return True

    for pos in cells:
        if pos in claimed:
            continue
        step = next_step(turn, unit, pos)
        if step is None or step in claimed:
            continue
        claimed.add(step)
        commands[unit.unit_id] = move_command(step)
        return True
    return False


def _use_item_if_held(
    unit: Unit,
    item: str,
    commands: dict[int, dict[str, Any]],
    target: Pos | None = None,
) -> bool:
    """背包里有这件道具就用掉（没有时不下指令）

    这些道具都是"用完就没、效果立即生效"的救急品（生命药剂、围墙修复包、
    升级券），所以调用方不需要记"用过没有"：用完之后条件自然不再成立
    （血回满了、墙回满了、基地升级了），本模块因此不用存任何跨回合状态。
    """
    if item not in unit.backpack:
        return False
    commands[unit.unit_id] = use_command(item, target)
    return True


def _night_medicine(unit: Unit, commands: dict[int, dict[str, Any]]) -> bool:
    """夜晚血量 <= 50 时喝一剂生命药剂（V4）

    任务书4.6.3：生命药剂 10 金，使用者回满血。喝下去血量就满了，条件自然
    不再成立，所以不会每回合都在喝药——只有真的挨打到快没血才用。
    """
    if unit.health > MEDICINE_HP_NIGHT:
        return False
    return _use_item_if_held(unit, MEDICINE, commands)


def _night_station_voucher(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """基地血量 < 150（或持券者自己 < 40）时用基地升级券（V4）

    夜里没有工人施工，基地掉到 150 以下就只剩挨打了；这张券等于一次"满血
    复活 + 升级"（任务书4.5.1：升级后建筑回到满血）。持券的人自己快没了时
    也用它——人倒下背包里的券也就用不上了（V4："持有券的小人血量低于40时"）。

    参数:
        turn: 当前回合信息
        unit: 当前决策的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达指令（使用或走向基地）
    """
    station = turn.station()
    if station is None or not station.is_alive or station.level >= 3:
        return False
    if station.health >= STATION_CRITICAL_HP and unit.health >= CARRIER_LOW_HP:
        return False

    entry = STATION_UPGRADE_VOUCHERS.get(station.level)
    if entry is None:
        return False
    voucher = entry[0]
    if voucher not in unit.backpack:
        return False

    # 已经在基地旁就直接用；否则往基地走（基地快没了，值得离开武器一趟）
    if distance(unit.pos, station.pos) <= 1:
        commands[unit.unit_id] = use_command(voucher, station.pos)
        return True
    step = _step_toward(turn, unit, station.pos, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)
        return True
    return False


def _night_wall_fixer(turn: Turn, unit: Unit, commands: dict[int, dict[str, Any]]) -> bool:
    """夜晚用围墙修复包补墙（V4：靠站位把 3×3 覆盖到的墙一次奶满）

    修复包要站在目标围墙一格范围内用（任务书4.6.3），所以只有当自己身边
    就有残墙时才动手——不动窝就能补，武器照旧有人操控。阈值按 V4 的分工：
    正面的人（R1/R2）盯自己那一片墙，血量低于 `FIXER_NIGHT_HP` 就补一次；
    后排的人（R3）负责次要方向的墙，阈值放宽到 `FIXER_NIGHT_HP_WEAK`；
    自身血量低于 `SELF_FIXER_HP` 时也补一次（人快没了，把身前的墙奶回来）。

    补完墙就满血了，条件自然不再成立（不会每回合都在奶同一段墙）。

    参数:
        turn: 当前回合信息
        unit: 当前决策的角色
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达使用指令
    """
    if WALL_FIXER not in unit.backpack:
        return False

    # 正面的人（R1 开拓者 / R2 石工）盯自己那一片墙（阈值 100），
    # 后排的人（R3 铜工）负责次要方向的墙（阈值 150）
    threshold = (
        FIXER_NIGHT_HP if _stand_role(turn, unit) <= 1 else FIXER_NIGHT_HP_WEAK
    )
    if unit.health < SELF_FIXER_HP:
        threshold = _wall_full_health(1)  # 自己快没了：身边有残墙就补，不看阈值

    damaged = tuple(
        wall for wall in turn.walls()
        if wall.health < threshold
        and distance(unit.pos, wall.pos) <= WALL_FIXER_RADIUS
    )
    if not damaged:
        return False

    target = _best_repair_spot(turn, damaged, unit.pos)
    if target is None:
        return False
    return _use_item_if_held(unit, WALL_FIXER, commands, target.pos)


def _night_support(
    turn: Turn,
    unit: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """夜晚的救急动作（V4）：生命药剂 > 基地升级券 > 围墙修复包

    开火之前先看一眼要不要救急：血量 <= 50 先喝药（人倒下了谁也救不了），
    基地 < 150 或有券的人自己 < 40 时用基地升级券，站位旁的墙快破了补一次
    修复包。三件事都只在真的危急时触发，触发后条件自然消失，所以武器照旧
    有人操控、不会整夜都在做后勤。

    参数:
        turn: 当前回合信息
        unit: 当前决策的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）

    返回:
        True 表示本回合已下达救急指令（调用方不要再下攻击/移动指令）
    """
    if _night_medicine(unit, commands):
        return True
    if _night_station_voucher(turn, unit, claimed, commands):
        return True
    return _night_wall_fixer(turn, unit, commands)


# === 夜晚决策 ===


def _decide_night(turn: Turn, commands: dict[int, dict[str, Any]]) -> None:
    """夜晚策略: 操控武器攻击

    除了操控武器，夜里还有一件闲事：**人肉城墙**（T6）——围墙被拆掉、或
    石材不够还没砌起来时，`_wall_holes` 给出的那几格就是机器人直通基地的
    门，与其空着，不如让这一回合腾得出手的角色站上去（见 `_plug_wall_gap`）。
    """
    claimed: set[Pos] = set()
    # 围墙缺口整场夜只算一次：`_wall_holes` 按 `_calc_wall_order` 的顺序给，
    # 先敌方来路那一段，与白天"先砌哪一段"是同一套判断
    holes = _wall_holes(turn)

    # 为每个武器配对一个操控角色
    paired: set[int] = set()
    for controller, weapon in _pair_controllers_and_weapons(turn):
        paired.add(controller.unit_id)
        # 救急优先（V4）：喝药 / 基地升级券 / 补墙只在真危急时触发，
        # 触发后条件立刻消失，武器不会因此整夜没人操控
        if _night_support(turn, controller, claimed, commands):
            continue

        # 检查操控角色是否在武器旁边
        if distance(controller.pos, weapon.pos) <= 1:
            # 武器在冷却中,跳过
            if weapon.cooldown > 0:
                continue

            # 寻找攻击目标
            target = _find_attack_target(turn, weapon)
            if target is not None:
                commands[weapon.unit_id] = attack_command(controller.unit_id, [target])
                continue

            # 射程内还没有目标（机器人多半在路上）：趁机挪到武器旁那格缺口上
            # ——站在缺口上照样操控得到武器，却把门堵上了（T6）
            _plug_wall_gap(turn, controller, holes, claimed, commands, weapon)
            continue

        # 操控角色不在武器旁边,向武器移动（顺路能堵缺口时优先站缺口）
        if _plug_wall_gap(turn, controller, holes, claimed, commands, weapon):
            continue
        step = _step_toward(turn, controller, weapon.pos, claimed)
        if step is not None:
            commands[controller.unit_id] = move_command(step)

    # 人肉城墙（T6）：这一回合没摊上武器的角色（塔被拆了、或者本来就没塔）
    # 去堵优先级最高的那个缺口，而不是原地空转（复盘里 idle_man 一路涨到 3）
    for unit in turn.controllable():
        if unit.unit_id in paired or unit.unit_id in commands:
            continue
        _plug_wall_gap(turn, unit, holes, claimed, commands)


def _pair_controllers_and_weapons(turn: Turn) -> list[tuple[Unit, Unit]]:
    """为武器配对操控角色

    配对策略:
        - 每轮取"距离最近的 操控者-武器 组合"，配对后双方一起移出候选，
          再继续配对剩下的，直到角色或武器用完
        - 距离相同时按角色ID、武器ID排序，保证结果稳定

    按距离就近配对可以让角色少跑路，夜晚首个回合更容易全部就位开火。

    返回:
        [(操控角色, 武器工事), ...] 配对列表
    """
    candidates = sorted(
        (
            (distance(controller.pos, weapon.pos),
             controller.unit_id, weapon.unit_id, controller, weapon)
            for controller in turn.controllable()
            for weapon in turn.weapons()
        ),
        key=lambda item: item[:3],
    )

    pairs: list[tuple[Unit, Unit]] = []
    paired_controllers: set[int] = set()
    paired_weapons: set[int] = set()
    for _, controller_id, weapon_id, controller, weapon in candidates:
        if controller_id in paired_controllers or weapon_id in paired_weapons:
            continue
        paired_controllers.add(controller_id)
        paired_weapons.add(weapon_id)
        pairs.append((controller, weapon))

    return pairs


def _find_attack_target(turn: Turn, weapon: Unit) -> Pos | None:
    """为武器寻找攻击目标（优先机器人）

    参数:
        turn: 当前回合信息
        weapon: 待操控的武器工事

    返回:
        攻击目标坐标；无可攻击目标时返回 None

    优先级:
        1. 攻击我方的存活机器人（先按威胁分级，同级取最近的一台）
        2. 视野内的敌方单位（选择最近的一个）
    """
    weapon_range = weapon.range_of_attack()

    # 优先攻击机器人
    robots = turn.alive_robots_targeting_me()
    targets_in_range = [
        robot for robot in robots
        if distance(weapon.pos, robot.pos) <= weapon_range
    ]

    if targets_in_range:
        # 大型/BOSS机器人威胁更高，优先处理；同级再取最近的
        nearest = min(targets_in_range, key=lambda r: (
            -ROBOT_THREAT.get(r.kind, 0),
            distance(weapon.pos, r.pos),
            r.robot_id,
        ))
        return nearest.pos

    # 没有机器人,尝试攻击敌方单位
    enemies = turn.enemies
    enemy_targets = [
        enemy for enemy in enemies
        if enemy.is_alive and distance(weapon.pos, enemy.pos) <= weapon_range
    ]

    if enemy_targets:
        nearest = min(enemy_targets, key=lambda e: (
            distance(weapon.pos, e.pos),
            e.unit_id,
        ))
        return nearest.pos

    return None


# === 辅助函数 ===


def _adjacent_mine(turn: Turn, unit: Unit, mine_type: str) -> Pos | None:
    """查找相邻的指定类型矿点"""
    mines = turn.get_mines(mine_type)
    adjacent = [
        mine for mine in mines
        if unit.pos != mine and distance(unit.pos, mine) <= 1
    ]
    if not adjacent:
        return None
    # 返回最近的矿点
    return min(adjacent, key=lambda m: (distance(unit.pos, m), m.x, m.y))


def _nearest_zone(turn: Turn, zone_type: str, origin: Pos) -> Pos | None:
    """查找离指定位置最近的中立元素（小贩、武器商店等）"""
    positions = [
        pos for pos, kind in turn.zones.items() if kind == zone_type
    ]
    if not positions:
        return None
    return min(positions, key=lambda pos: (distance(origin, pos), pos.x, pos.y))


def _nearest_task_position(turn: Turn, origin: Pos) -> Pos | None:
    """查找离指定位置最近的己方任务点坐标

    优先使用 playerTasks 中的任务点信息，缺失时回退到地图 zones 中
    本阵营的任务点类型。
    """
    positions = [task.task_position for task in turn.player_tasks]
    if not positions:
        kinds = _TASK_POINTS_BY_TEAM.get(turn.team_type, ())
        positions = [
            pos for pos, kind in turn.zones.items() if kind in kinds
        ]
    if not positions:
        return None
    return min(positions, key=lambda pos: (distance(origin, pos), pos.x, pos.y))


# === 自进化任务（沙盒） ===


def _task_output(turn: Turn) -> str:
    """上一回合沙盒输出里属于当前任务的那一份（没有时为空串）

    判题系统把沙盒命令的输出放在 `lastCmdResult` 里，但那份输出可能属于上一个
    任务（命令刚下发、或者任务刚换）。这里按任务标识认领：带本任务标识
    （`TASK_MARKER`，或描述里没给文件名时的 `TASK_PROBE_MARKER`）才算数。
    """
    if not turn.phase_task or not turn.last_cmd_result:
        return ""
    token = _task_token(turn.phase_task)
    if (
        f"{TASK_MARKER}{token}" in turn.last_cmd_result
        or f"{TASK_PROBE_MARKER}{token}" in turn.last_cmd_result
    ):
        return turn.last_cmd_result
    return ""


def _task_fetch_failed(output: str) -> bool:
    """这一份沙盒输出是不是"执行器跑过了、却一次数都没取到"（S1）

    执行器的每次请求都会留一行：成功打 `[API]`（`TASK_DATA_MARKER`），失败打
    `[APIFAIL]`（`TASK_API_FAIL_MARKER`）。输出里一条取数记录都没有时，分四种：

        1. 有 `[APIFAIL]`：地址猜错了——沙盒该做的都做了（读文档、拼地址、
           逐个请求），卡住的是候选地址。
        2. 连 `[APIFAIL]` 也没有，输出里也看不出执行器开过工：沙盒这一回合
           根本没走到取数那一步（只把任务文件读回来就交差了），那是读文件死循环。
        3. 连 `[APIFAIL]` 也没有，但 `[SCAN]` 在（它是执行器开工后打的第一行）：
           执行器跑过了，却连一次请求都没发出去。这既不是读文件死循环（沙盒
           没在干读文件那件事），也不是"猜错了地址"（一个地址都没试），而是
           执行器自己没跑出候选：任务文件没找到、或者取过数的文件为空，
           `files` 是空的，取数循环整段跳过。复盘 PK591684 的 R11–R13 与
           PK591772 的 R12–R13 就是这个形态：日志上 `[SCAN] docs=2 urls=2
           key=yes`、`exitCode:0` 全都正常，可 `api=0` 且一行 `[APIFAIL]`
           都没有。

    第 3 条要单列，是因为它的止损线不该按读文件死循环那条走：读文件死循环
    已经把该读的都读了，再等下去只是把同一份文件再读一遍；而"执行器没找到
    任务文件"是这一回合的输入不巧，下一回合候选地址会轮转（`rotate`）、LLM
    兜底也可能给出别的取数命令，还有救。旧判据把它归进第 2 条，`repeats`
    于是每回合累加，接上任务后的第 3 个回合就撞上 `TASK_LOOP_LIMIT` 被熔断
    （PK591684 的 R13：`watch=r3/t4/f3`，超时线与取数连败线都还没到），
    主张的"读题成功后必须调 API"因此一次都没轮上。

    第 4 条是"整条命令被掐掉"（`TASK_TIMEOUT_MARKER`，PK590252 的 R13）：
    kill 发生在取数证据打出来之前时，输出同样一行取数记录都没有，可它既不是
    读文件循环（沙盒没有在反复读同一份文件），也不是"猜错了地址"（一个地址
    都还没试）——是这条命令做得太多、时间花在了找文件上（见 `_task_retry_budget`）。

    第 1/3/4 条的止损线都一样（见 `_task_abandoned`）：读文件死循环按同一份输出
    重复几次熔断（`TASK_LOOP_LIMIT`），它们按连续几回合取不到数熔断
    （`TASK_API_FAIL_LIMIT`，比前者宽，正是为了把几个回合留给重试与 LLM 兜底）。
    判据在 `_watch_task` 里用来把这几类的 `repeats` 按住不动——否则它们
    会被当成读文件循环提前熔断。

    参数:
        output: 上一回合属于本任务的沙盒输出

    返回:
        True 表示这一份输出里没有任何取数证据、也不是"在读同一份文件"的死循环
        （取数全失败 / 执行器没发出请求 / 整条命令被掐掉）
    """
    if TASK_DATA_MARKER in output:
        return False  # 取到数了（哪怕后来又失败），这一回合算有进展
    return (
        TASK_API_FAIL_MARKER in output          # 试过地址，全失败
        or TASK_SCAN_MARKER in output           # 执行器跑过，却一次请求都没发
        or TASK_TIMEOUT_MARKER in output        # 整条命令被掐了，取数还没轮到
    )


def _task_retry_budget(turn: Turn) -> int:
    """这一回合执行器搜目录的上限（上一回合超时过就压小）

    全盘 walk 是沙盒命令里最慢的一步（`TASK_EXEC_DIR_BUDGET` 就是为它设的上限），
    上一回合整条命令被掐掉（`TASK_TIMEOUT_MARKER`）说明预算还是花超了。取数
    证据是任务唯一的进展凭据，重试回合把搜目录的上限压到 `TASK_EXEC_RETRY_DIR_BUDGET`，
    命令就能在 15 秒内跑到"调接口取数"那一步：题干与接口文档上一回合已经读过
    一遍，这一回合不该再扫一遍目录树（报告 S3 的"读题去重"）。

    判据看整份 `lastCmdResult`（不像看门狗那样按任务标识认领）：超时是"沙盒这
    一回合很慢"的证据，不是某个任务的观察值；被掐掉的输出可能连任务标识都没
    打出来（判题器整份替换成 `[TIMEOUT]`），按任务标识认领的话这种最该压预算的
    回合反而漏掉。没有超时观察时返回原上限，行为与改造前一致。
    """
    if TASK_TIMEOUT_MARKER in (turn.last_cmd_result or ""):
        return TASK_EXEC_RETRY_DIR_BUDGET
    return TASK_EXEC_DIR_BUDGET


def _watch_task(turn: Turn) -> None:
    """刷新任务看门狗（每回合由 `decide` 调用一次）

    看门狗只回答一个问题："这个任务还有没有进展"。判题系统不回任务状态，
    报文里能看到的只有 `lastCmdResult` 里属于本任务的沙盒输出，所以这里只记
    最近一回合的观察值（`_TASK_WATCH`），并且只在回合号连续时往上累计：
    换任务、换局、回合号跳变都从这一回合重新计数，不会把别的一局的观察带进来。

    三个计数（都从报文本身推出来，不额外存东西）:
        - `probes`：上一回合的输出是探测输出（带 `TASK_PROBE_MARKER`），
          说明那一次探测已经发出去过了（S2 的次数上限）。
        - `submits`：这一回合手里有能交的答案，且是白天（夜里不交卷），
          说明这一回合就会交一次卷（S1 的提交次数上限）。交完卷任务仍然
          留在 `phaseTask` 里，下一回合就会再数一次。
        - `fails`：这一回合的沙盒输出里既没有取数证据（`TASK_DATA_MARKER`），
          手里也攒不出一份答卷，说明这一趟取数又白跑了（S1 的取数连败上限）。
          只数白天的回合：夜里开拓者本来就要回防，任务已经被强制结束。

    `repeats`（同一份输出连续出现了几次，读文件死循环的判据）只在沙盒**没在
    取数**的回合上累计：见下面的说明与 `_task_fetch_failed`。

    参数:
        turn: 当前回合信息
    """
    global _TASK_WATCH
    if not turn.phase_task:
        _TASK_WATCH = None
        return

    token = _task_token(turn.phase_task)
    output = _task_output(turn)
    probes = 1 if TASK_PROBE_MARKER in output else 0
    submits = 1 if turn.is_day and (
        _task_answer(turn) is not None or _cached_answer(turn) is not None
    ) else 0
    # 取数失败：有本任务的沙盒输出、里面却没有取数证据，且这一回合交不出卷。
    # 输出为空（命令刚下发、还没回结果）不算失败——那只是还没到看结果的时候。
    failed = 1 if (
        turn.is_day and output and TASK_DATA_MARKER not in output and not submits
    ) else 0
    previous = _TASK_WATCH
    if (
        previous is None
        or previous.token != token
        or turn.round_no != previous.round_no + 1
    ):
        # 新任务（或接不上上一回合的观察）：这一份输出算第 1 次出现
        _TASK_WATCH = TaskWatch(
            token, turn.round_no, output, 1, 1 if output else 0, probes, submits,
            failed,
        )
        return

    # 又读到同一份输出说明这一回合没有任何进展，往上累计；换了新输出则重新数。
    # 但"沙盒执行器跑过了、只是没取到数"不算没有进展（S1）：地址是照文档猜的，
    # 而猜法是确定性的——同一份任务文件每回合算出同一批候选地址、撞同一批
    # 404，输出因此逐字相同，`repeats` 于是必然到线：复盘 PK591771/PK591786
    # 里 R11 下发沙盒命令、R12–R14 三回合回读同一份输出，任务就在 R14 被熔断
    # （watch=r3/t4，而 `rounds` 才 4、`fails` 才 3，两条更宽的止损线都还没到）。
    # 连一次请求都没发出去的那种也一样（判据 3，见 `_task_fetch_failed`）：
    # PK591684 的 R11–R13 沙盒回读同一份 `[SCAN] docs=2 urls=2 key=yes`、
    # `api=0`，同样在第 3 个回合被熔断，重试与 LLM 兜底都没轮上。
    # 这两类都该归"取数连败"（`TASK_API_FAIL_LIMIT`）管：它比读文件循环线宽，
    # LLM 兜底（`TASK_LLM_MAX_PROMPTS`）才有机会把六次求助跑完。
    # 上一回合跑的是 LLM 给的那条命令时同理（S1，复盘 PK591806 的 R11–R17）：
    # 沙盒里跑的不是执行器，输出里当然没有 `[SCAN]`——那是"这一回合没派执行器
    # 去取数"，不是"读文件读不出新东西"。按读文件死循环熔断的话，任务会在
    # LLM 的六次求助还没跑完、执行器一次取数都没试过的时候就被放弃
    # （PK591783 的 R13：`api=0` 恒 0 却已经 abandoned=yes，任务分与这名
    # 劳动力一起丢掉）。这一回合照旧计入取数连败（`fails` 与输出无关），
    # 由更宽的 `TASK_API_FAIL_LIMIT` 兜底，任务不会因此永远挂在任务点上。
    # 整条命令被掐掉的那种同理（判据 4，见 `_task_fetch_failed`）：超时只是
    # 说明这条命令做得太多，下一回合压小搜目录上限重试即可（`_task_retry_budget`），
    # 不该在第 3 个回合就把任务判死。
    if not output:
        repeats = 0
    elif _task_fetch_failed(output):
        repeats = 1
    elif _llm_command_round(turn, previous.round_no):
        repeats = 1
    elif output == previous.output:
        repeats = previous.repeats + 1
    else:
        repeats = 1
    # 取数连败是"连续"计数：只要有一回合取到数（或有卷可交）就从头数
    fails = previous.fails + 1 if failed else 0
    _TASK_WATCH = TaskWatch(
        token, turn.round_no, output, previous.rounds + 1, repeats,
        previous.probes + probes, previous.submits + submits, fails,
    )


def _task_abandoned(turn: Turn) -> bool:
    """当前任务是不是已经被看门狗放弃（只读，不刷新观察值）

    四条止损线（见 `TASK_LOOP_LIMIT` / `TASK_TIMEOUT_ROUNDS` /
    `TASK_SUBMIT_LIMIT` / `TASK_API_FAIL_LIMIT`）：同一份沙盒输出连续出现了
    `TASK_LOOP_LIMIT` 次、任务已经占用了 `TASK_TIMEOUT_ROUNDS` 个回合、
    同一份答案已经交满 `TASK_SUBMIT_LIMIT` 次还没被 Judge 放行，或者连续
    `TASK_API_FAIL_LIMIT` 个回合取数全失败。复盘里开拓者就是被"每回合回读
    同一份任务文件"的死循环占死的（PK589649 的 R11–R17、PK589653 的 R12–R17），
    上交的答案还可能是错的（PK590252 的 R17 交的是文件路径），任务分拿不到，
    这名劳动力也一起白搭；另一批复盘（PK590916/PK591014）里卡住的则不是
    死循环而是取数连败——沙盒每回合都换一批 404 的地址，`repeats` 到不了线，
    只能靠取数连败这条（或超时线）兜底。

    第一条线只认"沙盒没在取数"的那种重复（见 `_task_fetch_failed`）：猜地址
    猜错的回合输出同样逐字相同（猜法是确定性的），但它有取数失败的诊断在手，
    该走第二条的取数连败线而不是被当成读文件循环提前熔断（PK591771/PK591786
    的 R11–R14 就是这么在接任务后的第 4 个回合被放弃的，`rounds` 才 4、
    `fails` 才 3，两条更宽的线都还没到）；执行器跑过、却连一次请求都没发出去
    的回合同理（PK591684 的 R11–R13，`[SCAN] docs=2 key=yes` 而 `api=0`），
    `repeats` 同样被按住，交给取数连败那条线；整条命令被掐掉的回合同样按住
    `repeats`（`TASK_TIMEOUT_MARKER`，PK590252 的 R13），下一回合用压小的
    搜目录上限重试（见 `_task_retry_budget`）。

    观察值必须是本回合或上一回合记下的（`sandbox_command` 排在 `decide` 之前
    调用时，看到的是上一回合那条），回合号对不上就当作没有观察，免得把别的
    一局的观察套到当前任务上。
    """
    watch = _TASK_WATCH
    if watch is None or not turn.phase_task:
        return False
    if watch.token != _task_token(turn.phase_task):
        return False
    if turn.round_no not in (watch.round_no, watch.round_no + 1):
        return False
    return (
        watch.repeats >= TASK_LOOP_LIMIT
        or watch.rounds >= TASK_TIMEOUT_ROUNDS
        # `submits` 把本回合这一次也算在内，所以第 TASK_SUBMIT_LIMIT+1 次
        # 才止损——也就是最多交满 TASK_SUBMIT_LIMIT 次
        or watch.submits > TASK_SUBMIT_LIMIT
        # 取数连败：沙盒每回合都换一批猜错的地址，输出不重样，`repeats` 到不了
        # 线，靠这条把开拓者要回来（见 `TASK_API_FAIL_LIMIT`）
        or watch.fails >= TASK_API_FAIL_LIMIT
    )


def _task_probe_done(turn: Turn) -> bool:
    """当前任务的沙盒探测次数是不是已经用满（只读，见 `TASK_PROBE_LIMIT`）

    探测只用来"描述里没给文件名时认一下沙盒里的任务文件叫什么"，认不出来
    说明那套命名对不上，再扫一遍根目录只是把同一份输出再拿一次。用满次数
    之后 `_sandbox_command` 直接把任务根目录交给执行器，由它按文件名特征
    自己找（沙盒里真正干活的是执行器，探测只是给它指个路）。

    参数:
        turn: 当前回合信息

    返回:
        True 表示这个任务不再探测，直接执行
    """
    watch = _TASK_WATCH
    if watch is None or not turn.phase_task:
        return False
    if watch.token != _task_token(turn.phase_task):
        return False
    if turn.round_no not in (watch.round_no, watch.round_no + 1):
        return False
    return watch.probes >= TASK_PROBE_LIMIT


def _task_llm_state(turn: Turn) -> dict[str, Any]:
    """当前任务的 LLM 交互状态（按任务标识存，跨回合保留）

    键是任务标识（`_task_token`，含任务描述与序号），任务结束换新任务时
    自然换一份状态，不会把上一个任务的答案带过来。
    """
    return _TASK_LLM_STATE.setdefault(
        _task_token(turn.phase_task),
        {"prompts": 0, "pending_cmd": "", "cmd_round": 0, "answer": ""},
    )


def _llm_command_round(turn: Turn, round_no: int) -> bool:
    """`round_no` 那一回合的沙盒跑的是不是 LLM 给的那条命令（S1）

    `_llm_task_command` 下发 `CMD:` 时把回合号记在任务状态里（`cmd_round`），
    这里据此认出"上一回合的沙盒输出是那条命令的结果"。与执行器的输出相比，
    它天然没有 `[SCAN]`——判据在 `_watch_task` 里用来把这类回合从"读文件
    死循环"里摘出来（见那里的说明）。只读状态，不新建：没有求助记录的任务
    不必因为看一眼多出一份空状态。
    """
    if not turn.phase_task:
        return False
    state = _TASK_LLM_STATE.get(_task_token(turn.phase_task))
    if not state:
        return False
    # `cmd_round` 只在真的下发过命令时才非 0：回合号 0（还没记录）不算命中，
    # 免得状态里的默认值把第一回合认成"跑过 LLM 的命令"
    stamp = int(state.get("cmd_round") or 0)
    return stamp > 0 and stamp == round_no


def _task_prompt(turn: Turn, payload: dict[str, Any]) -> str:
    """任务卡住时向 LLM 求助的 prompt（任务期间不占每日额度）

    只在"接了任务、还没有答案或待执行命令、并且已经拿到沙盒输出"时发：
    任务文件与接口文档要先从沙盒捞回来，LLM 才有东西可看。同一个任务最多问
    `TASK_LLM_MAX_PROMPTS` 次——问不出结果就该止损，别把整个任务窗耗在提问上。

    参数:
        turn: 当前回合信息
        payload: 原始请求（取沙盒输出等字段）

    返回:
        要提交给 LLM 的 prompt；本回合不需要求助时返回空串
    """
    if not turn.phase_task or _task_abandoned(turn):
        return ""
    state = _task_llm_state(turn)
    if state["answer"] or state["pending_cmd"]:
        return ""
    if state["prompts"] >= TASK_LLM_MAX_PROMPTS:
        return ""

    evidence = (turn.last_cmd_result or "").strip()
    if not evidence:
        return ""  # 沙盒还没吐回任务文件/接口文档，先让执行器去捞

    state["prompts"] += 1
    return "\n".join([
        "你在替我解一道《未来战争》的自进化类任务，你只能通过沙盒里的一条 shell 命令取数。",
        f"任务描述：{turn.phase_task}",
        "",
        "沙盒已经捞回来的内容（任务文件、接口文档、目录清单）：",
        evidence[:TASK_LLM_EVIDENCE_LIMIT],
        "",
        "约束：沙盒无法访问外网，本地接口在 http://localhost:8899；",
        "接口要鉴权时，按接口文档里的写法带上请求头（如 Authorization: Bearer，"
        "Key 就在文档里）；返回 401/403 说明头没带对，别把错误信息当答案。",
        "一条命令限时 15 秒，一回合只能发一条命令，命令的 stdout 会原样回到我这里。",
        "请只回一行，二选一：",
        "CMD: <一条能在沙盒里直接跑出答案的 shell 命令，只输出答案本身>",
        "ANSWER: <你已经能确定答案时，直接给答案>",
    ])


def _consume_task_reply(turn: Turn, payload: dict[str, Any]) -> None:
    """把上一回合 LLM 的回复（llmResp）落进当前任务的状态

    只认 `CMD:` / `ANSWER:` 两个前缀；两条都出现时优先 `CMD`——真去沙盒取数
    才算解出来，LLM 凭文档直接给的答案只当兜底。

    `CMD:` 给的命令先过一道 `_shell_command_ok` 的体检（引号成对、单行、
    不带 heredoc）：拼不出合法命令的回复直接丢掉，本回合改走执行器自己取数
    （`_sandbox_command` 的兜底路径），而不是拿一整个回合去换一条注定报语法错
    的命令；状态里没落下命令时 `_task_prompt` 下一回合会再问一次，问到
    `TASK_LLM_MAX_PROMPTS` 次为止（复盘 PK590881 的 R14 就是被一条引号不配对
    的命令耗掉了一个回合，PK591011 的 R13/R16 则是被 heredoc 耗掉的）。

    体检之后还有一道鉴权闸门（`_task_command_auth_ok`，S1）：接口已经回过
    "缺 Authorization 头"（复盘 PK591784 的 R16 401）而命令里一点鉴权材料都
    没有时同样丢掉——执行器的请求会带上文档里的 Key，比这条必然 401 的命令
    更接近答案。
    """
    reply = str(payload.get("llmResp") or "")
    if not turn.phase_task or not reply:
        return
    state = _task_llm_state(turn)
    for line in reply.splitlines():
        text = line.strip()
        if text.startswith(TASK_LLM_CMD_PREFIX):
            command = text[len(TASK_LLM_CMD_PREFIX):].strip()
            if (
                command
                and _shell_command_ok(command)
                and _task_command_auth_ok(command, turn.last_cmd_result or "")
            ):
                state["pending_cmd"] = command
                return
    for line in reply.splitlines():
        text = line.strip()
        if text.startswith(TASK_LLM_ANSWER_PREFIX):
            answer = text[len(TASK_LLM_ANSWER_PREFIX):].strip()
            if answer:
                state["answer"] = answer
                return


# heredoc 在"一回合一条命令"里永远收不了尾（S2，复盘 PK591011 的 R13/R16）。
# heredoc 的结束符必须独占一行，而 `CMD:` 只给得出一行命令：包装时另起一行
# 接上去的 `echo "[TASK_END]"` 与末尾的 `:` 会被当成 heredoc 正文一起吞掉，
# bash 只回一句 `here-document at line 0 delimited by end-of-file (wanted
# 'EOF')`（PK591011 的 R16 就是这条）。更糟的是这行报错出在解析阶段，整条
# 命令一个字都不会执行——连开头的任务标识都没打印，`_task_answer` 只能判
# `no_marker`（同一场的 R13：sandbox=发送但 state=no_marker），这一个任务回合
# 连同那一次的提交机会一起白费。`<<<`（here-string）是另一回事：它当场就有
# 内容，单行也跑得起来，照旧放行（`(?<!<)` 挡住的是 `<<<` 里从第二个 `<`
# 起算的那一对，否则 here-string 会被误判成 heredoc）。
LLM_HEREDOC_PATTERN = re.compile(r"(?<!<)<<(?![<=])")


def _shell_command_ok(command: str) -> bool:
    """LLM 给的沙盒命令能不能直接交给 bash 跑（引号成对、单行、不带 heredoc）

    LLM 的回复是一行 `CMD: <命令>`，会被原样拼进沙盒命令里。少一个配对的
    引号时 bash 整条报 `unexpected EOF while looking for matching '"'`：
    复盘 PK590881 的 R14 就是这么白丢一个回合的——沙盒输出里连任务标识都
    没有，`_task_answer` 只能判 `no_marker`，下一回合从头再来。

    体检看三件事：
        - 单/双引号各自成对（不区分转义）
        - 不含换行（换行会把命令拆成多行，末尾的 `[TASK_END]` 会被卷进命令体）
        - 不含 heredoc 重定向（结束符独占一行，单行命令里收不了尾，末标记
          与退出码一起被吞掉，见 `LLM_HEREDOC_PATTERN`）

    成对性不看转义，`echo "a\\"b"` 这类合法写法会被一并挡掉；heredoc 也一样，
    `grep -o 'a<<b' file` 这类把 `<<` 当数据的命令会被误伤。两者都按同一个
    取舍办：宁可漏放一条，交给执行器自己去取数（`_sandbox_command` 的兜底
    路径），也不拿一个回合去赌一条可能跑不起来的命令。

    参数:
        command: LLM 给的取数命令（单行）

    返回:
        True 表示这条命令可以拼进沙盒命令
    """
    if not command or "\n" in command or "\r" in command:
        return False
    if LLM_HEREDOC_PATTERN.search(command):
        return False
    return all(command.count(quote) % 2 == 0 for quote in ('"', "'"))


# 命令里"带了鉴权材料"的样子（S1，见 `_task_command_auth_ok`）：执行器发的是
# `Authorization` / `X-API-Key` 两个头，文档里常见的还有 `Bearer <key>`、
# `api_key=<key>`、`token=<key>`，以及把 Key 直接拼进查询串的写法（`?key=`）。
TASK_AUTH_REQUEST_PATTERN = re.compile(
    r"authorization|x-api-key|api[-_]?key|access[-_]?token|\btoken\b|\bbearer\b|[?&]key=",
    re.I,
)
# 命令里出现本地接口的形态（沙盒里的接口就在 `TASK_API_DEFAULT` 那个主机上）：
# 带不带 `http://` 前缀、带不带端口都算。
TASK_LOCAL_API_PATTERN = re.compile(
    r"(?:https?://)?(?:localhost|127\.0\.0\.1)(?::\d+)?",
    re.I,
)
# 沙盒证据里"这个接口要 Authorization 头、而刚才没带"的证词（S1）：接口在
# 401/403 的正文里点名自己要哪个头——复盘 PK591784 的 R16 回的就是
# `[APIFAIL] ... HTTPError 401 => {"error":"Authentication failed: Missing
# 'Authorization' header"}`。头名与状态码要落在同一行（诊断行就是这么打的），
# 免得把文档里两处不相干的说法拼成一句。没有这份证词时无从知道接口要不要
# 鉴权，命令照原样放行。
TASK_AUTH_DEMAND_PATTERN = re.compile(
    r"(?:authorization|x-api-key)[^\n]{0,80}?(?:\b40[13]\b|missing|required|invalid)"
    r"|(?:\b40[13]\b|missing|required|invalid)[^\n]{0,80}?(?:authorization|x-api-key)",
    re.I,
)


def _task_command_auth_ok(command: str, evidence: str) -> bool:
    """LLM 给的取数命令会不会"没带鉴权就去调接口"（S1）

    执行器发出的每个请求都按文档里的写法带上了鉴权头（`api_key` +
    `request_headers`），而 LLM 给的 `CMD:` 是原样下发的：它见过接口文档
    （`_task_prompt` 把沙盒证据一并喂过去，也提醒过"401/403 是头没带对"），
    却未必会把头抄进命令里。这种命令必然换来一句
    `401 ... Missing 'Authorization' header`（复盘 PK591784 的 R16），答案拿
    不到，这一个任务回合也白搭。

    判死要同时满足三件事，缺一不可：
        - 命令确实在调本地接口（`TASK_LOCAL_API_PATTERN`）——读文件、跑校验
          脚本这类不碰接口的命令与鉴权无关；
        - 命令里一点认证材料都没有（`TASK_AUTH_REQUEST_PATTERN`）——照文档
          抄了 `Bearer` / `api_key=` / `token=` 的都算带了，带得对不对由接口
          说了算，这里不判；
        - 沙盒证据里有"这个接口要 Authorization 头"的证词
          （`TASK_AUTH_DEMAND_PATTERN`）——接口自己说了要头，才谈得上"没带
          必然 401"。

    三条都中的命令丢掉（与 `_shell_command_ok` 同一个取舍：宁可漏放一条，
    也不拿一个回合去赌一条注定失败的请求），本回合改走执行器兜底
    （`_sandbox_command`）——执行器会按文档把 Key 抠出来带上、并轮转候选地址，
    比再发一条注定 401 的请求更接近答案。`_task_prompt` 下一回合还会再问一次，
    次数照旧受 `TASK_LLM_MAX_PROMPTS` 约束。

    参数:
        command: LLM 给的取数命令（已过 `_shell_command_ok` 体检）
        evidence: 上一回合的沙盒输出（接口文档与 `[APIFAIL]` 诊断都在里面）

    返回:
        True 表示这条命令可以下发
    """
    if not TASK_LOCAL_API_PATTERN.search(command):
        return True  # 不碰接口的命令与鉴权无关
    if TASK_AUTH_REQUEST_PATTERN.search(command):
        return True  # 带过鉴权材料，放行给接口判
    return TASK_AUTH_DEMAND_PATTERN.search(evidence) is None


# 沙盒里被调用的脚本可能是 CRLF 行尾（S1）：任务自带的校验脚本按 Windows 换行
# 落地时，内核 exec 会把 `\r` 一起读进 shebang，整条命令只换来一行
# `/bin/sh^M: bad interpreter`——复盘 PK591009 的 R14 就是这么白烧掉一个任务
# 回合的（沙盒输出里没有任何取数结果，答案区只能是空的，任务分继续挂零）。
# 命令是 LLM 给的、脚本是沙盒里的，两头都不归我们管，只能在下发前把脚本的
# 行尾归一化，见 `_crlf_safe_command`。
LLM_SCRIPT_PROGRAMS = (
    "sh", "bash", "dash", "zsh", "sudo", "env", "exec", "command",
    "python", "python3",
)
# 沙盒自己的路径不碰：`/bin/cat`、`/usr/bin/awk` 这些是沙盒的工具，不是任务
# 脚本，归一化它们既没意义又可能把沙盒改坏
LLM_SCRIPT_SYSTEM = (
    "/bin/", "/sbin/", "/usr/", "/etc/", "/lib/", "/proc/", "/sys/", "/dev/",
    "/var/",
)
# 认得出是脚本的后缀：`./check` 这类校验脚本通常没有后缀，所以空后缀也算
LLM_SCRIPT_EXTS = ("", ".sh", ".bash", ".py", ".pl", ".rb")
# 带这些字符的写法一律不碰：拼进前置片段会破坏命令本身
LLM_SCRIPT_BAD = frozenset("'\"`\\*?[]$&|;<>(){}\n\r\t")


def _script_wrapper(token: str) -> bool:
    """这个词是不是"跑脚本的方式"而不是脚本本身（解释器 / 前缀命令 / 选项）

    `/bin/sh ./check`、`python3 /tmp/x.py`、`sudo ./check` 里的第一个词都是
    解释器或前缀命令，真正要跑的东西在后一个词上；`-x` 这类选项同理，跳过它们
    才找得到脚本本身。
    """
    if token in LLM_SCRIPT_PROGRAMS:
        return True
    if token.startswith("-"):
        return True
    return (
        token.startswith(LLM_SCRIPT_SYSTEM)
        and os.path.basename(token) in LLM_SCRIPT_PROGRAMS
    )


def _script_path(token: str) -> bool:
    """这个词看着是不是沙盒里的一个脚本（见 `_script_in_command`）

    三件事: 带路径前缀（`./`、`../`、绝对路径）、不是沙盒自己的工具路径、
    后缀在 `LLM_SCRIPT_EXTS` 里（`./check` 这类校验脚本没有后缀，所以空后缀也算）。
    """
    if not token.startswith(("./", "../", "/")):
        return False
    if token.startswith(LLM_SCRIPT_SYSTEM):
        return False
    if any(char in LLM_SCRIPT_BAD for char in token):
        return False
    return os.path.splitext(token)[1].lower() in LLM_SCRIPT_EXTS


# 命令分段符：`a && b`、`a; b`、`a | b` 里的每一段都是一条独立的命令，脚本可能
# 出现在任何一段的开头（`cd /tmp/selfEvolutionTask/1-x && ./check`）。
LLM_COMMAND_SPLIT = re.compile(r"&&|\|\||;|\|")


def _script_in_command(command: str) -> str:
    """命令里第一个被执行到的沙盒脚本（没有则返回空串）

    跳过解释器/前缀命令与选项之后看每一段命令的第一个词：`./check`、
    `../check.sh`、`/tmp/selfEvolutionTask/1-x/check.py` 是"跑沙盒里的一个脚本"，
    而归一化只对脚本有意义——`cat ./task.md` 里的路径是数据文件（给取数结果做
    sed 只会改坏答案），`curl` 后面的地址同理，这些一概不碰。

    分段是必要的（S2，复盘 PK591806 的 R14）：LLM 给的命令常写成
    `cd /tmp/selfEvolutionTask/1-x && ./check`，而 `cd` 既不是解释器也不是
    脚本，旧实现盯着整条命令的第一个词看，脚本因此一次都没被认出来——`./check`
    照旧以 CRLF 落地，只换来一行 `/bin/sh^M: bad interpreter`。每一段只看开头
    那一个词（连同它前面的解释器/选项），`grep -o 'x' ./task.md` 这类"脚本路径
    只是参数"的写法照旧不碰。

    参数:
        command: LLM 给的沙盒命令（单行）

    返回:
        可以安全做行尾归一化的脚本路径；找不到时返回空串
    """
    for segment in LLM_COMMAND_SPLIT.split(command):
        tokens = segment.strip().split()
        index = 0
        while index < len(tokens) and _script_wrapper(tokens[index]):
            index += 1
        if index >= len(tokens):
            continue
        token = tokens[index]
        if _script_path(token):
            return token
    return ""


def _crlf_safe_command(command: str) -> str:
    """把命令里调用的沙盒脚本转成 LF 行尾再执行（S1）

    CRLF 的脚本会以 `/bin/sh^M: bad interpreter` 收场（复盘 PK591009 的 R14：
    这一条把任务窗里的一个回合整段烧掉）。前置片段先 `[ -f ]` 判存在：LLM 猜
    的路径未必真有那个文件，这种情况原命令照跑，行为与改造前一致；`sed -i`
    本身失败（只读挂载、没有 sed）也只是归一化没生效，不影响后面的命令。

    参数:
        command: LLM 给的沙盒命令（单行）

    返回:
        带行尾归一化前置片段的命令；没有脚本可归一化时原样返回
    """
    script = _script_in_command(command)
    if not script:
        return command
    # 命令是“先 cd 再跑脚本”的写法时（`cd /tmp/selfEvolutionTask/1-x && ./check`），
    # 前置片段排在整条命令最前面，那时工作目录还没切过去：`[ -f ./check ]` 判的
    # 是沙盒的工作目录（`/`），脚本明明在，归一化却静默跳过，`./check` 照旧以
    # CRLF 落地、只换来一行 `/bin/sh^M: bad interpreter`（S2，复盘 PK591806 的
    # R14）。`_script_dir` 把脚本之前那个 `cd <目录>` 记下来，用它补全相对路径
    # 再去判存在（见 `_script_dir`）。
    path = _script_dir(command, script) or script
    # 正则里的 `\r` 不能直接写在 sed 表达式里（POSIX 没定义，个别 sed 当成
    # 字母 r，那样会把每行末尾的 r 都删掉、把脚本改坏），改用 printf 生成一个
    # 真正的回车字符拼进表达式。变量名带 `CRLF_` 前缀，避开命令自己可能用到的
    # 短名字（`P`、`CR` 这种在 LLM 给的命令里并不罕见）
    return (
        f"CRLF_P='{path}'; CRLF_CR=$(printf '\\r'); "
        f"[ -f \"$CRLF_P\" ] && "
        f"sed -i \"s/$CRLF_CR\\$//\" \"$CRLF_P\" 2>/dev/null; "
        f"{command}"
    )


# 命令里的 `cd <目录>`：目录参数只认不带引号的普通路径（带引号/变量的写法一律
# 不解析，宁可少补一个路径也不猜错），后面必须跟着一个命令分隔符才作数——
# `&&`、`;`、`||`、`|` 四种都算（见 `_script_dir`）。
LLM_CD_PATTERN = re.compile(r"cd\s+([^\s;&|'\"`]+)\s*(?:&&|;|\|\||\|)")


def _script_dir(command: str, script: str) -> str:
    """把命令里的脚本路径补全成绝对路径（没有可用的 `cd` 时返回空串）

    只看脚本出现之前的部分：脚本后面的 `cd` 跟这次调用没关系
    （`./check && cd /tmp`）。连续多次 `cd` 按顺序拼接，相对路径接在上一段
    后面（`cd /a && cd b && ./check` -> `/a/b/check`）；补出来的路径不存在时
    `_crlf_safe_command` 的 `[ -f ]` 会把归一化跳过，行为与改造前一致。

    参数:
        command: LLM 给的沙盒命令（单行）
        script: `_script_in_command` 认出来的脚本路径

    返回:
        脚本的绝对路径；补不出来时返回空串（调用方退回脚本原样）
    """
    if not script.startswith("./"):
        return ""  # 已经是 `../x` 这类带前缀的写法，或本来就短，不补
    at = command.find(script)
    if at < 0:
        return ""
    folder = ""
    for part in LLM_CD_PATTERN.findall(command[:at]):
        if part.startswith("/"):
            folder = part
        elif folder:
            folder = folder.rstrip("/") + "/" + part
        else:
            folder = part
    if not folder:
        return ""
    return folder.rstrip("/") + "/" + script[2:]


def _llm_task_command(turn: Turn) -> str:
    """把 LLM 给的取数命令包成一条沙盒命令（带任务标识，供下一回合取答案）

    包装方式和执行器一致：`[TASK]<标识>` 与 `[TASK_END]` 之间是答案区，
    末尾的 `:` 保证退出码为 0。

    命令调的是沙盒里的脚本时先做一遍行尾归一化（S1，见 `_crlf_safe_command`）：
    CRLF 的脚本只会换来一行 `/bin/sh^M: bad interpreter`，一个任务回合白搭。
    """
    state = _task_llm_state(turn)
    command = str(state.get("pending_cmd") or "")
    if not command:
        return ""
    state["pending_cmd"] = ""
    state["cmd_round"] = turn.round_no
    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    return (
        f'echo "{marker}"; {_crlf_safe_command(command)}\n'
        f'echo "{TASK_END_MARKER}"; :'
    )


def _llm_direct_answer(turn: Turn) -> str | None:
    """LLM 直接给出的答案（不需要沙盒，幂等：一直保留到任务结束）"""
    if not turn.phase_task:
        return None
    answer = str(_task_llm_state(turn).get("answer") or "").strip()
    if len(answer) < TASK_LLM_ANSWER_MIN_LEN:
        return None
    if _task_echo(answer, turn.phase_task):
        return None  # 把任务原文当答案交上去 = 又一次 0 分
    if _task_text_answer(answer, turn.last_cmd_result):
        return None  # 把喂给 LLM 的那份文档原文抄回来，同样不是答案（S1）
    if _task_doc_body(answer):
        return None  # LLM 抄的是一份文档的正文，不是一个答案（S2）
    if _task_path_answer(answer):
        return None  # "ANSWER: <任务文件的路径>" 同样不是答案（S1）
    # LLM 把命令的报错当成了答案（PK590921 的 R14/R16）：`ANSWER:` 那一行写的
    # 可以是它刚跑完的命令的报错原文（404 JSON 由 `_task_error_body` 拦），
    # 也可以是 `jq: command not found` 这行工具报错——后者既没有错误键也没有
    # 状态码，不在这里挡一道就会被当成答案交上去（`_llm_command_answer` 有
    # 同样的闸门，两条取答案的路不能只有一条装了）。
    if _task_error_body(answer):
        return None
    return answer


def _llm_command_answer(turn: Turn, region: str) -> str | None:
    """LLM 指定的取数命令跑完后的输出（只在紧接着的那一回合认）"""
    if not turn.phase_task:
        return None
    state = _task_llm_state(turn)
    sent_round = int(state.get("cmd_round") or 0)
    if not sent_round or turn.round_no != sent_round + 1:
        return None
    answer = region.strip()
    if len(answer) < TASK_LLM_ANSWER_MIN_LEN:
        return None
    if any(bad in answer for bad in TASK_ERROR_MARKERS):
        return None
    if _task_echo(answer, turn.phase_task):
        return None
    if _task_error_body(answer):
        return None  # 命令把接口的错误提示打了出来，这一趟同样没取到数
    if _task_text_answer(answer, turn.last_cmd_result):
        return None  # 命令把沙盒里的文档原文打了出来（`cat 文档`），不是答案
    if _task_doc_body(answer):
        return None  # 命令把一份文档的正文打了出来（`cat 文档`），不是答案（S2）
    if _task_path_answer(answer):
        return None  # 命令只把任务文件的路径打了出来，不算取到数
    return answer


def _sandbox_command(turn: Turn) -> str:
    """任务期间需要提交给沙盒执行的命令

    自进化类任务的任务文件是"任务描述"（"请阅读task_1_beijing.md，获取任务
    信息"），答案要按描述在沙盒里取数才能得到。命令带上任务标识，便于下一
    回合确认输出属于当前任务；已经拿到本任务的答案后就不再重复执行。

    答案区里跑的是执行器（`_task_executor`）：它读任务文件与沙盒里的接口
    文档，照文档给出的地址真实调用本地接口取数，把取到的数据打成
    `[SOLUTION]` 段。取不到数据就什么都不打——`_task_answer` 只认带取数
    证据的答案，宁可这一回合不提交，也不把任务原文当成答案交上去
    （复盘里 R12/R14/R16/R18 四次 submitAnswer 交的全是任务原文，任务分
    恒为 0，两个任务点合计 160 分 + 160 金币全部丢掉）。

    描述里连文件名都没有时（"请按沙盒里的任务说明作答"这类），先按
    `_sandbox_probe` 探一次沙盒，下一回合从探测结果里认出文件名再走上面的
    读文件流程；探到 `TASK_PROBE_LIMIT` 次还没认出文件就改用执行器，把任务
    根目录（`TASK_ROOTS`）交给它自己找——不再反复扫同一棵目录树。

    答案区之后依次是工作目录诊断与任务文件回读（`[TASK_FILE]` 分段，供
    `_task_file` 认出沙盒里的真实文件名、给执行器圈定候选任务文件）。
    这两段都排在 `TASK_END_MARKER` 之后，永远不会被当成答案。

    任务已经被看门狗放弃（读文件死循环 / 超时）时返回空串：继续下发读文件
    命令只会把同一个循环再跑一遍，开拓者却已经被放回去干别的了。
    """
    if (
        not turn.phase_task
        or _task_answer(turn) is not None
        or _cached_answer(turn) is not None
        or _task_abandoned(turn)
    ):
        return ""

    # LLM 给了取数命令就优先跑它：一回合只能发一条命令，它比"继续猜地址"更准
    llm_command = _llm_task_command(turn)
    if llm_command:
        return llm_command

    # 描述里没给文件名时，用上一回合的探测结果找；还没探过就先探一次
    target = _task_file(turn.phase_task) or _task_file(turn.last_cmd_result)
    if target is None and not _task_probe_done(turn):
        return _sandbox_probe(turn)
    if target is None:
        # 探测次数已经用满（见 `TASK_PROBE_LIMIT`）：把任务根目录直接交给执行器，
        # 由它在里面按文件名特征找任务文件与接口文档。再探一次只是把同一份
        # 路径清单再拿一遍（复盘里 R14/R16 两次输出逐字相同就是这么来的），
        # 而执行器本来就有"找文件 + 读文档 + 取数"的完整流程。
        target = TASK_ROOTS[0]

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    scan = "pwd; ls -a -- . 2>&1 | head -40"
    # 执行器片段以 heredoc 结束符收尾，换行后再接诊断与回读；
    # 末尾的 `:` 保证整条命令的退出码为 0：输出带 `[exitCode:0]` 才会被
    # `_task_answer` 采纳，而执行器里的取数失败时退出码可能是非 0 的。
    # 回合号作为候选地址的轮转量传进去（S1）：取数的地址是照文档猜的，而每回合
    # 算出来的候选清单逐字相同，不轮转就永远只试最前面那 `TASK_API_MAX_CALLS` 条
    # （复盘里沙盒连着几回合输出逐字相同、api 恒 0，重试等于没重试）。
    # 搜目录的上限同理按上一回合的观察给（S3）：上一回合整条命令被超时掐掉时，
    # 这一回合少扫一点目录、把时间留给取数那几步（见 `_task_retry_budget`）。
    return (
        f'echo "{marker}";'
        f" {_task_executor(target, turn.round_no, _task_retry_budget(turn))}"
        f'\necho "{TASK_END_MARKER}"; {scan};'
        f' {_task_dump(turn)}; :'
    )


def _task_executor(
    task_path: str,
    offset: int = 0,
    dir_budget: int = TASK_EXEC_DIR_BUDGET,
) -> str:
    """在沙盒里执行自进化任务的命令片段（读任务文件 -> 调接口取数 -> 打答案）

    沙盒里只有基础 shell 与 python，命令一回合只能下一发、限时 15 秒，
    所以取数脚本一次跑完：找任务文件与接口文档、按文档里的样例地址调用
    本地接口、把响应体打成 `[SOLUTION]` 段。

    参数:
        task_path: 任务描述里点名的任务文件（沙盒路径或文件名）
        offset: 候选地址的轮转量（S1，通常传当前回合号）：候选清单比
            `TASK_API_MAX_CALLS` 长，每回合算出来的清单又逐字相同，不轮转
            就永远只试最前面那几条（见执行器里的 `rotate`）
        dir_budget: 搜目录的上限（通常 `TASK_EXEC_DIR_BUDGET`；上一回合整条
            命令被掐掉时压小，见 `_task_retry_budget`）

    返回:
        可直接拼进沙盒命令的 shell 片段
    """
    script = (
        TASK_EXECUTOR
        .replace("__TASK_PATH__", repr(task_path))
        .replace("__BASE__", repr(TASK_API_DEFAULT))
        .replace("__TIMEOUT__", str(TASK_API_TIMEOUT))
        .replace("__MAX_CALLS__", str(TASK_API_MAX_CALLS))
        .replace("__KEEP__", str(TASK_API_KEEP))
        .replace("__OFFSET__", str(offset))
        .replace("__TIME_BUDGET__", str(TASK_API_TIME_BUDGET))
        .replace("__DIR_BUDGET__", str(dir_budget))
        .replace("__QUERY_MAX__", str(TASK_API_QUERY_MAX))
        .replace("__BODY_LIMIT__", str(TASK_API_BODY_LIMIT))
        .replace("__SOLVE_MAX__", str(TASK_SOLVE_MAX))
        .replace("__DOC_NAMES__", repr(TASK_API_DOC_NAMES))
        .replace("__WIDE_DOC_NAMES__", repr(TASK_DOC_WIDE_NAMES))
        .replace("__SUFFIXES__", repr(TASK_API_PATH_SUFFIXES))
        .replace("__PRUNE__", repr(TASK_EXEC_PRUNE))
        .replace("__ROOTS__", repr(TASK_ROOTS))
        .replace("__DOC_PRUNE__", repr(TASK_DOC_PRUNE))
        .replace("__SOLUTION__", repr(TASK_SOLUTION_MARKER))
        .replace("__SOLUTION_END__", repr(TASK_SOLUTION_END))
        .replace("__DATA__", repr(TASK_DATA_MARKER))
        .replace("__FAIL__", repr(TASK_API_FAIL_MARKER))
        .replace("__SCAN__", repr(TASK_SCAN_MARKER))
        .replace("__DOC__", repr(TASK_DOC_MARKER))
        .replace("__TEXT_HINT__", str(TASK_TEXT_HINT))
        .replace("__KEY_PATTERNS__", repr(TASK_API_KEY_PATTERNS))
        .replace("__KEY_MIN__", str(TASK_API_KEY_MIN_LEN))
        .replace("__KEY_PLACEHOLDERS__", repr(TASK_API_KEY_PLACEHOLDERS))
        .replace("__AUTH_HEADER__", repr(TASK_API_AUTH_HEADER))
        .replace("__KEY_HEADER__", repr(TASK_API_KEY_HEADER))
        .replace("__BEARER__", repr(TASK_API_BEARER))
    )
    # 沙盒的解释器叫 python3 或 python，挑一个能用的（挑不到时脚本不会执行，
    # 答案区为空 -> 这一回合不提交，下一回合重来）。
    # 片段以 heredoc 结束符收尾且不带换行：调用方必须换行后再接别的命令
    # （结束符要独占一行，直接接 `;` 会让后一条命令变成脚本的一部分）
    # `-u` 让 stdout 无缓冲（S1）：`[SCAN]` / `[API]` / `[APIFAIL]` 这几行是
    # "这一回合到底有没有去取数"的唯一凭据，而整条沙盒命令限时 15 秒、超时
    # 会被判题器直接掐掉（管道里的 Python 默认按块缓冲，被 kill 时缓冲区里
    # 的内容一起丢掉）。丢了凭证的回合在输出里"看不出执行器开过工"，看门狗
    # 于是按读文件死循环计数，接上任务后的第 3 个回合就熔断——报告里
    # "读题成功、api=0、watch r0→r3 后放弃"的形态正是这样漏掉了执行器其实
    # 已经跑过、只是没跑完的那几行
    return (
        'for P in python3 python; do command -v "$P" >/dev/null 2>&1 && break;'
        f" done; $P -u - <<'PYEOF' 2>/dev/null\n{script}\nPYEOF"
    )


# 沙盒执行器：占位符由 `_task_executor` 按当前任务填好。
# 之所以要"执行"而不是"读文件"，是因为任务文件里写的是任务要求（"查询北京
# 文化遗产"），答案在文档给出的本地接口里；直接把任务文件的内容交上去
# 等于答非所问（复盘里就是这么丢掉全部任务分的）。
TASK_EXECUTOR = '''\
import os
import re
import time
import urllib.parse
import urllib.request

TASK_PATH = __TASK_PATH__
BASE = __BASE__
TIMEOUT = __TIMEOUT__
MAX_CALLS = __MAX_CALLS__
KEEP = __KEEP__
OFFSET = __OFFSET__
TIME_BUDGET = __TIME_BUDGET__
DIR_BUDGET = __DIR_BUDGET__
QUERY_MAX = __QUERY_MAX__
BODY_LIMIT = __BODY_LIMIT__
SOLVE_MAX = __SOLVE_MAX__
DOC_NAMES = __DOC_NAMES__
SUFFIXES = __SUFFIXES__
PRUNE = __PRUNE__
ROOTS = __ROOTS__
DOC_PRUNE = __DOC_PRUNE__
WIDE_DOC_NAMES = __WIDE_DOC_NAMES__
SOLUTION = __SOLUTION__
SOLUTION_END = __SOLUTION_END__
DATA = __DATA__
FAIL = __FAIL__
SCAN = __SCAN__
DOC = __DOC__
TEXT_HINT = __TEXT_HINT__
KEY_PATTERNS = __KEY_PATTERNS__
KEY_MIN = __KEY_MIN__
KEY_PLACEHOLDERS = __KEY_PLACEHOLDERS__
AUTH_HEADER = __AUTH_HEADER__
KEY_HEADER = __KEY_HEADER__
BEARER = __BEARER__
SKIP_WORDS = ("http", "https", "localhost", "task", "spec", "md", "txt", "json", "api")


def read(path):
    """读文件，读不到就返回空串（沙盒里权限与路径都不可控）"""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def fetch(url, headers):
    """调用接口并把响应体截断返回（失败时打一行诊断，绝不抛异常打断整条命令）

    失败诊断要留在输出里：复盘里沙盒"执行了但没答案"时，日志上看不到任何
    原因（旧实现把异常吞掉、命令又带 `2>/dev/null`），只能靠猜。

    headers 由 `request_headers` 按文档里抠出来的 Key 拼好（S3）：鉴权头是
    401 与 200 之间唯一的差别。
    """
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8", "replace").strip()[:BODY_LIMIT]
    except Exception as exc:
        # 异常名后面带上 HTTP 状态码：复盘里只有 `[APIFAIL] ... HTTPError`，
        # 分不清 401（缺鉴权）还是 404（地址不对），下一次修复只能靠猜（S1）
        code = getattr(exc, "code", "")
        name = type(exc).__name__
        status = "%s %s" % (name, code) if code else name
        line = "%s %s" % (url, status)
        # 错误响应体里常常写着"缺什么"（401 的鉴权格式、404 的可用地址），
        # 压成一行跟在后面：下一轮照着改地址/补鉴权，不必再猜（S1）。
        # 必须和状态码打在同一行：`[APIFAIL]` 的行数就是"取数失败了几次"
        # （见 `task_brief` 的 fail 计数），一次失败拆成两行会让计数翻倍
        try:
            detail = " ".join(exc.read().decode("utf-8", "replace").split())
        except Exception:
            detail = ""
        if detail:
            line += " => " + detail[:BODY_LIMIT]
        print(FAIL, line)
        return ""


def api_key(text):
    """文档里写明的接口 Key（没写、或只写了占位符时返回空串）

    接口文档的样例里通常直接给出该带的头（`Authorization: Bearer sk-xxx`、
    `X-API-Key: xxx`、`api_key=xxx`，或表格里的 `| API Key | xxx |`），照抄
    下来就能过鉴权（S3，复盘 PK590918/PK590917 的接口一直 401：`missing
    'Authorization' header`）。命中第一条能过体检的值就返回；一条都挑不出来
    时返回空串，调用方照旧裸请求。
    """
    for pattern in KEY_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            value = match.group(1).strip("`\\\"'.,")
            if len(value) < KEY_MIN or len(set(value)) < 3:
                continue  # 太短，或 `xxxxxx` 这类占位，都不是 Key
            if value.lower() in KEY_PLACEHOLDERS or value.lower().startswith("http"):
                continue  # 文档里的 `Authorization: Bearer YOUR_API_KEY`／换行后接着的地址
            return value
    return ""


def request_headers(key):
    """请求头：有 Key 就同时带上 Bearer 与 X-API-Key（S3）

    两种写法在鉴权接口里都常见，文档也未必把两种都写上；多带一个对方不认识
    的头不影响正常请求，而一条命令的请求名额有限（`TASK_API_MAX_CALLS`），
    不拿它去试第二种写法——试错的那次注定 401，还得再花一次请求补回来。
    """
    headers = {"Accept": "*/*"}
    if key:
        headers[AUTH_HEADER] = BEARER + key
        headers[KEY_HEADER] = key
    return headers


def find_files(patterns, limit, roots=ROOTS, prune=PRUNE):
    """按文件名特征在沙盒里找文件（任务文件与接口文档都在沙盒深处）

    搜索默认锚在任务根目录（S2）：任务文件与它自己的接口文档就摆在同一棵
    目录树里，而全盘 walk 是这里最慢的一步——复盘 PK590252 的 R13 整条命令
    就是被它拖到 `[TIMEOUT]` 的。根目录里一个都没找到时，调用方拿同一套
    上限退到全盘（`roots=("/",)`），沙盒版本不同也照样找得到。

    两个上限都要兜住：找到够数就停，进的目录太多也停（沙盒命令整体限时
    15 秒，宁可少找几个也不能超时）。
    """
    found = []
    seen = set()
    visited = 0
    for start in roots:
        if not os.path.isdir(start):
            continue
        for root, dirs, files in os.walk(start):
            visited += 1
            if visited > DIR_BUDGET:
                return found
            dirs[:] = [d for d in dirs if os.path.join(root, d) not in prune]
            for name in files:
                path = os.path.join(root, name)
                if path in seen:
                    continue
                if not any(re.search(pattern, name, re.I) for pattern in patterns):
                    continue
                seen.add(path)
                found.append(path)
                if len(found) >= limit:
                    return found
    return found


def task_docs(roots):
    """沙盒里文件名像任务文件的那些（先看任务根目录，再退到全盘）"""
    patterns = (r"^(task|spec).*\\.(md|txt|json)$",)
    return find_files(patterns, 24, roots, PRUNE)


def task_files():
    """待解的任务文件：描述里点名的那份排第一（答案只认它）"""
    wanted = os.path.basename(TASK_PATH) if TASK_PATH else ""
    named = []
    others = []
    for path in task_docs(ROOTS) or task_docs(("/",)):
        if wanted and os.path.basename(path) == wanted:
            named.append(path)
        else:
            others.append(path)
    if TASK_PATH and os.path.isfile(TASK_PATH) and TASK_PATH not in named:
        named.insert(0, TASK_PATH)
    return named + others


# 主机名（authority）里允许出现的字符：RFC 3986 的那一套。反引号、引号、
# 全角标点、中文都不在其中——它们只会来自文档的行文，不是地址的一部分。
HOST_SAFE = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".-_~%!$&()*+,;=:@[]"
)


def cut_host(text):
    """在主机名的第一个非法字符处截断（S1）

    文档把地址写在句子里时，杂质常常紧跟在端口后面：`http://localhost:8899`），API`
    里连一个 `/` 都没有，整段会被 urlsplit 当成 netloc，而 `quote` 只覆盖
    path/query，反引号与全角标点原样进了 URL，urlopen 于是抛 InvalidURL
    ——复盘 PK590836/PK590849 里 R12–R17 连续多个回合的
    `[APIFAIL] http://localhost:8899`），API InvalidURL` 就是这一条：只剥两端
    的标点救不了它（末尾是 ASCII 的 `API`，没得剥）。
    主机名只允许 `HOST_SAFE` 里的字符，第一个非法字符连同它后面的行文一起丢掉
    ——`），API` 是文档的句子，不是地址。

    只看 authority（`//` 之后到第一个 `/?#` 之前）：`?city=北京` 这类中文
    查询词不在 authority 里，它由 `refine_url` 的 quote 做百分号编码，照旧可用。
    """
    start = text.find("//")
    if start < 0:
        return text
    start += 2
    end = len(text)
    for sep in ("/", "?", "#"):
        pos = text.find(sep, start)
        if pos >= 0:
            end = min(end, pos)
    for index in range(start, end):
        if text[index] not in HOST_SAFE:
            return text[:index]
    return text


def refine_url(raw):
    """把文档里抓到的地址整成 urlopen 能吃的形式（整不出来就返回空串）

    文档是中文的，地址常写在句子中间或反引号里，尾随的全角标点、引号会让
    urllib 直接抛 `InvalidURL`——复盘 #67 里 R12–R17 连续 6 回合
    `APIFAIL ... ），API InvalidURL`（URL 含反引号+中文）就是这么来的，
    任务因此 8 个回合读不到题面、最终 0 分。这里做三件事：

    1. 剥掉两端的标点/引号/括号（含全角）
    2. 主机名里混进来的杂质（反引号、全角标点、中文）在第一个非法字符处截断
       （见 `cut_host`）
    3. 路径与查询里的非 ASCII 字符（如 `?city=北京`）按 UTF-8 百分号编码
    """
    text = raw.strip()
    trim = "`'\\\"、，。；：？！,.;:!?)]}>（）【】《》“”‘’"
    while text and text[-1] in trim:
        text = text[:-1]
    while text and text[0] in trim:
        text = text[1:]
    text = cut_host(text)
    while text and text[-1] in trim:
        text = text[:-1]
    if not text:
        return ""
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    path = urllib.parse.quote(parts.path, safe="/%:@&=+$,-_.!~*'()")
    query = urllib.parse.quote(parts.query, safe="=&%:@+$,-_.!~*'()")
    url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, ""))
    # 拼完再体检一次：urlopen 能吃的地址全是可打印 ASCII。走到这里还带非
    # ASCII，说明上面哪一步没盖住，这个地址干脆不试——宁可不取数，也不让它
    # 再去撞一次 InvalidURL，把一个回合的沙盒白烧掉。
    if any(ord(char) < 33 or ord(char) > 126 for char in url):
        return ""
    return url


def endpoints(doc_text):
    """接口文档里的调用样例：本地接口优先，其次才是文档里抓到的其他地址

    沙盒内的接口就在 BASE 上（TASK_API_DEFAULT），而 find_files(DOC_NAMES)
    从全盘捞回来的文档里什么外链都有。旧实现把抓到的外链排在本地接口前面，
    MAX_CALLS 被这些在无网沙盒里调不通的地址耗光，真正能取数的本地接口
    一次都没被请求到，答案区永远是空的——三场复盘里"沙盒执行了（exitCode:0）
    却拿不到答案"就是这么来的。
    """
    urls = []
    for raw in re.findall(r"https?://[^\\s<>)\\]}]+", doc_text):
        raw = refine_url(raw)
        if raw and raw not in urls:
            urls.append(raw)
    local = [url for url in urls if "localhost" in url or "127.0.0.1" in url]
    picked = local or [BASE] + [url for url in urls if url != BASE]
    return picked


def queries(text, name):
    """查询关键词：文件名里的英文词最可靠（task_1_beijing.md -> beijing）

    文件名按非字母切成词干里的词再取：`task_1_beijing.md` 给的是 `beijing`
    （文档问的正是城市名），而不是整段词干 `task_1_beijing`。整段文件名不是
    一个查询值——它拼进样例地址只会 404（复盘 PK590884/PK590920 的
    `[APIFAIL] http://localhost:8899/task_1_alpha` 就是这么来的），而
    `QUERY_MAX` 只有两个名额，它先占掉一个就把文档里真正有用的词挤出去了。

    词干里的中文词同样算数（S1）：任务文件写成 `task_1_北京.md` 这类中文名时，
    按非字母切词的旧写法一个查询词都取不出来，只好退到文档正文里随手挑的英文
    词（`GET`、`heritage` 这类模板里的路径名），拼进样例地址必然取不到数——
    与整段文件名当查询词是同一个坑，只是换了种形态。中文词干排在英文词干
    之后、文档正文之前：正文里的英文词是最后的选择，也是误伤面最大的一档。
    """
    stem = os.path.splitext(name or "")[0]
    # 文件名与词干本身不算查询词：正文里再提到一次这个文件名时同样跳过
    whole = {value.lower() for value in (name, stem) if value}
    values = []
    for token in re.split(r"[^A-Za-z]+", stem):
        if len(token) > 1 and token.lower() not in SKIP_WORDS:
            values.append(token)
    # 中文词干：连取最长的一段（`{2,8}` 顺带截断过长的名字），
    # 它是这一段里唯一能当查询值的来源
    for token in re.findall(r"[一-鿿]{2,8}", stem):
        if token not in whole:
            values.append(token)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,30}", text):
        if token.lower() in SKIP_WORDS or token.lower() in whole:
            continue
        values.append(token)
    picked = []
    for value in values:
        if value not in picked:
            picked.append(value)
    return picked[:QUERY_MAX]


def swap_path(url, value):
    """样例地址末尾那段数字换成任务自己的查询词（换不出来时返回空串）

    接口文档给的调用样例常常是个模板：`http://localhost:8899/api/task/1`
    里的 `1` 是"第几个样例"，照抄下来只会得到一行
    `Endpoint not found:/api/task/1`（复盘 PK591787 的 R16 就是这条），而把
    查询词接在样例后面（`.../api/task/1/beijing`）同样不是文档里那个接口。
    这时的正解是把末尾那段数字换成自己的查询词（`.../api/task/beijing`）。
    末尾不是数字（样例已经写成真实路径，或压根没有路径）时返回空串，
    调用方不加这条候选，取数名额照旧留给原来那几条。
    """
    match = re.match(r"^(.*/)(\\d+)([?#].*)?$", url)
    if not match:
        return ""
    return match.group(1) + value + (match.group(3) or "")


def candidates(text, name, urls):
    """这一份任务要试的调用地址：文档样例 + 把样例里的查询词换成任务自己的

    样例地址的改法两种都算：带查询串的换查询串里的值，末尾是数字的换整段
    路径（见 `swap_path`）。后一种排在清单末尾：它只在样例确实是个模板时才
    有意义，而清单会被 `TASK_API_MAX_CALLS` 截断——真到截断那一步，该先
    保住的是文档里那几条样例与根地址那几个兜底后缀。

    样例里的查询值还常常是空的：文档写成 `.../weather?city=<城市名>` 时，
    `endpoints` 的地址正则在 `<` 处截断，抓到的就是 `.../weather?city=`；
    文档本来就写成空值（`?city=`）时同样是这个形状。旧写法要求 `=` 后面
    "至少有一个字符"（`[^&/]+`），空值样例因此一个带查询词的候选都生不出来
    ——请求照原样发出去，问的是空查询词，接口只会回一行取数失败的诊断
    （复盘里"读题成功、却一个回合接一个回合取不到数"的又一种成因）。

    查询词里的非 ASCII（`task_1_北京.md` -> `北京`，见 `queries`）先做百分号
    编码再拼（S1）：`urlopen` 只吃 ASCII 地址，中文照原样拼进路径或查询串，
    请求发不出去、只会换来一行 `[APIFAIL] ... UnicodeEncodeError`——取数名额
    白烧一次，这一回合照旧 `api=0`。替换一律走 lambda 而不是替换串：`re.sub`
    的替换串会把组引用（反斜杠加数字）当成语法，拼进去的查询词不该有这种副作用。
    """
    urls = urls or [BASE]
    values = [urllib.parse.quote(value, safe="") for value in queries(text, name)]
    out = []
    for url in urls[:2]:
        if url not in out:
            out.append(url)
        for value in values:
            variant = (
                re.sub(r"=([^&/]*)", lambda _match: "=" + value, url, count=1)
                if "=" in url
                else url.rstrip("/") + "/" + value
            )
            if variant not in out:
                out.append(variant)
    for suffix in SUFFIXES:
        variant = BASE.rstrip("/") + suffix
        if variant not in out:
            out.append(variant)
    for url in urls[:2]:
        for value in values:
            variant = swap_path(url, value)
            if variant and variant not in out:
                out.append(variant)
    return out


def rotate(items, keep, offset):
    """把候选地址里"兜底的那一段"按回合轮转（前 keep 条固定不动，S1）

    自进化任务的接口地址是靠"读文档 -> 拼地址"猜的，而猜法是确定性的：同一份
    任务文件每回合算出来的候选清单逐字相同，被 MAX_CALLS 截断后每回合试的还是
    同一批地址、撞同一批 404，沙盒输出因此逐字相同——复盘 PK591771/PK591786
    的 R11–R14 就是这样：读题成功（key=yes、docs=2）但 api 恒 0、fail=8，
    重试了几个回合等于把同一批猜错的地址又试了一遍。

    文档给出的样例地址优先级最高（`candidates` 把它们排在清单最前面），前 keep
    条每回合照旧先试；其余候选按回合号轮转，转上几个回合整份清单都能覆盖到。

    offset 为 0（拿不到回合号）或清单不比 keep 长时原样返回。
    """
    if offset <= 0 or keep >= len(items):
        return items
    head, tail = items[:keep], items[keep:]
    shift = offset % len(tail)
    return head + tail[shift:] + tail[:shift]


files = task_files()
# 接口文档的搜索范围（S2）：任务根目录与各任务文件所在目录优先，系统文档树
# （/usr/share/doc 这类）整段跳过——复盘 PK590252 的 R14/R16 两次读回来的
# /usr/share/doc/uom-se-1.0.4/README.md 就是从那里捞的，照着它拼地址自然
# 取不到数。根目录里一份文档都没有时才退到全盘（仍然带着 DOC_PRUNE）。
doc_roots = list(ROOTS)
for path in files[:SOLVE_MAX]:
    folder = os.path.dirname(path)
    if folder and folder not in doc_roots:
        doc_roots.append(folder)
doc_files = find_files(DOC_NAMES, 6, doc_roots, DOC_PRUNE)
if not doc_files:
    doc_files = find_files(DOC_NAMES, 6, ("/",), DOC_PRUNE)
doc_text = "\\n".join(read(path) for path in doc_files)
# 近处这几份文档里连一个本地接口地址都没抓到（S1，复盘 PK591809 的
# `[SCAN] docs=2 urls=1 key=no`）时，到全盘再找一遍接口文档并进来：任务
# 根目录里的 `.md` 首先命中的是任务文件自己（任务书就是 `.md`），上面那条
# `if not doc_files` 的兜底因此永远触发不了——接口文档不在任务树里时一次都
# 读不到，`endpoints` 只剩本地接口那一行兜底，取数只能一路 404 到止损
# （`api=0`、答案区永远为空）。只认 `WIDE_DOC_NAMES` 那几种名字，避免把
# 无关的 `.md` 再捞一堆回来。
if not re.search(r"https?://(?:localhost|127\\.0\\.0\\.1)", doc_text):
    wide = [
        path for path in find_files(WIDE_DOC_NAMES, 6, ("/",), DOC_PRUNE)
        if path not in doc_files
    ]
    if wide:
        doc_files += wide
        doc_text += "\\n" + "\\n".join(read(path) for path in wide)
urls = endpoints(doc_text)
# 鉴权 Key 先从接口文档里找，找不到再退到任务文件（S3）：文档写的是"该带哪个
# 头"，偶尔也有把 Key 直接写在题面里的。两处都没有时 headers 里只剩 Accept，
# 照旧裸请求——不发一个没有 Key 的 Authorization 头。
key = api_key(doc_text) or api_key("\\n".join(read(path) for path in files[:SOLVE_MAX]))
headers = request_headers(key)
print(SCAN, "tasks=%d docs=%d urls=%d key=%s" % (
    len(files), len(doc_files), len(urls), "yes" if key else "no"))
# 接口文档的开头各打一行（DOC）：文档页被当成"取数结果"取回来时（`/docs`
# 这类地址返回的就是文档本身），决策侧靠这些指纹认出"答案就是文档原文"
# （见 `_task_text_answer`）。指纹必须排在 `[SOLUTION]` 段之前，取数失败时
# 答案区为空、闸门也不会跟着失效。
for path in doc_files:
    hint = " ".join(read(path).split())[:TEXT_HINT]
    if hint:
        print(DOC, hint)

deadline = time.time() + TIME_BUDGET
calls = 0
solutions = []
for path in files[:SOLVE_MAX]:
    name = os.path.basename(path)
    text = read(path)
    bodies = []
    for url in rotate(candidates(text, name, urls), KEEP, OFFSET):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        calls += 1
        body = fetch(url, headers)
        if body:
            print(DATA, url, "=>", len(body))
            bodies.append(body)
    if bodies:
        solutions.append((name, bodies))

# 一份任务文件都没找到时（`files` 为空：任务描述点名的文件名在沙盒里对不上，
# 任务目录里也没有 task*/spec* 文档），上面的循环整段跳过，这一回合就成了
# “执行器跑过了、却连一次请求都没发”：输出里 `[SCAN] tasks=0 ... api=0
# fail=0` 全都正常，答案区却是空的——复盘 PK591684 的 R11-R13、PK591772 的
# R12-R13 与 PK591783 的 R11-R12 都是这个形态，任务一路卡到止损，取数一次
# 都没试过。接口文档（`urls`）与本地接口（`BASE`）本来就在手里，没有任务
# 文件照样得把候选地址试一遍：`[API]` / `[APIFAIL]` 这两行是“读题之后真的
# 去调了 API”的唯一凭据，也是看门狗判断该按哪条止损线走的依据
# （见 `_task_fetch_failed` 的判据 3）。
if not calls:
    name = os.path.basename(TASK_PATH or "")
    # 查询词的来源：任务描述里点名的那份任务文件（`queries`：task_1_beijing.md
    # -> beijing）。沙盒里没找到这份文件时，它是唯一还握在手里的查询词来源，
    # 漏传就等于拿文档正文里随手挑的英文词去填样例地址（`?city=` 那类模板
    # 只有填对了才取得到数）。目录名不是文件名，不传。
    query = name if re.search(r"\\.(md|txt|json|csv|log)$", name, re.I) else ""
    if not query:
        # 任务描述里没点名文件时 `TASK_PATH` 可能是任务根目录：那不是一份任务
        # 文件，答案段只能挂一个占位名（决策侧按“描述里没给文件名”取第一段）
        name = "task"
    bodies = []
    for url in rotate(candidates(doc_text, query, urls), KEEP, OFFSET):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        calls += 1
        body = fetch(url, headers)
        if body:
            print(DATA, url, "=>", len(body))
            bodies.append(body)
    if bodies:
        solutions.append((name, bodies))

for name, bodies in solutions:
    print(SOLUTION + name)
    for body in bodies:
        print(body)
    print(SOLUTION_END)
'''


def _sandbox_find(
    names: tuple[str, ...],
    action: str,
    exts: tuple[str, ...] = TASK_FILE_EXTS,
    root: str = "/",
) -> str:
    """沙盒里按文件名查找的 find 片段（默认从根目录起全盘找）

    复盘里沙盒的工作目录就是 `/`，`ls -a -- .` 只有 bin/dev/etc/home/lib/
    lib64/proc/sbin/tmp/usr，而任务文件并不在 `/` 的前三层里：旧实现把搜索
    限定在 `find . -maxdepth 3` 加两个猜出来的目录（`/tmp/selfEvolutionTask`、
    `/tmp/selfEvolution`），任务文件一次都没被找到，开拓者整个任务周期
    卡在任务点。这里按文件名全盘找，只跳过 `TASK_FIND_PRUNE` 里的虚拟目录，
    读不到文件的目录由 `2>/dev/null` 静音；`root` 用来把搜索锚到任务根目录
    （S2，见 `_sandbox_find_first`）。

    命中还要过一道扩展名闸门（`exts`）：只有"名字像任务文件、且扩展名是
    文档"的才算任务文件，`task.xsl` 这类同名样式表被挡在外面（见
    `TASK_FILE_EXTS`；三场复盘里回读回来的正是它）。

    参数:
        names: 文件名通配（如 `task*`），多个通配之间是"或"关系
        action: 命中后执行的动作（`TASK_FIND_PRINT` 只列路径，内容由调用方按需读取）
        exts: 扩展名白名单，命中文件必须以后缀之一结尾
        root: 搜索起点（默认全盘；任务根目录见 `TASK_ROOTS`）

    返回:
        可直接拼进沙盒命令的 find 片段
    """
    prune = " -o ".join(f'-path "{path}"' for path in TASK_FIND_PRUNE)
    wanted = " -o ".join(f'-name "{name}"' for name in names)
    docs = " -o ".join(f'-name "*{ext}"' for ext in exts)
    return (
        f'find {root} \\( {prune} \\) -prune -o -type f \\( {wanted} \\)'
        f" -a \\( {docs} \\) {action} 2>/dev/null"
    )


def _sandbox_find_first(
    names: tuple[str, ...],
    action: str,
    limit: int,
    exts: tuple[str, ...] = TASK_FILE_EXTS,
) -> str:
    """先在任务根目录里找，找不到才退到全盘（S2）

    任务文件与它自己的接口文档就摆在 `TASK_ROOTS` 那棵目录树里，而全盘 find
    是沙盒命令里最慢的一步——复盘 PK590252 的 R13 整条命令直接 `[TIMEOUT]`，
    R14/R16 两次又把 `/usr/share/doc` 下的库文档当成任务文件读了回来。这里
    把根目录里的命中当成主路径，只有那里一个都没找到（沙盒版本不同、任务
    文件摆在别处）才扫全盘，既快又不会先捞到无关文件。

    参数:
        names: 文件名通配（如 `task*`）
        action: 命中后执行的动作（通常只列路径）
        limit: 最多带回几条（防止一条命令的输出把响应体撑大）
        exts: 扩展名白名单

    返回:
        可直接拼进沙盒命令的 shell 片段（输出与单条 find 一致：一行一个路径）
    """
    near = "; ".join(
        _sandbox_find(names, action, exts, root=path) for path in TASK_ROOTS
    )
    far = _sandbox_find(names, action, exts)
    return (
        f"found=$({{ {near}; }} | head -{limit}); "
        f'if [ -z "$found" ]; then found=$({far} | head -{limit}); fi; '
        f'echo "$found"'
    )


def _task_dump(turn: Turn) -> str:
    """读回沙盒里全部任务文件的沙盒命令（排在答案结束标记之后）

    自进化类任务是一整套同构任务（任务书的例子是"查询北京/上海/广州天气"），
    一次把沙盒里的任务文件都读回来，下一个任务点就不用再花一个来回等输出
    （见 `_TASK_ANSWER_CACHE`）。只读文件名像任务文件的那几个，
    免得把沙盒里的无关文档一起吃回来占用输出行数。

    上一回合一份任务正文都没回读到（输出里没有 `TASK_FILE_MARKER`）放宽一档，
    连 `*.md` 一起扫：沙盒里任务文件的实际命名未必和任务描述里写的一致，
    卡在一个文件名上反复空转不如把候选都摊开（探测输出里的文件名同样会
    被 `_task_file` 认出来，下一回合就能直接读中意的那份）。回读段里的
    候选同样要过 `TASK_FILE_EXTS` 的扩展名闸门，所以"没回读到任务正文"
    与"回读到的全是 task.xsl 这类无关文件"是同一个信号——两种情况都换用
    放宽后的命令重试，最多退这两步（`TASK_FILE_NAMES` -> `TASK_PROBE_NAMES`）。

    参数:
        turn: 当前回合信息（用上一回合的沙盒输出判断要不要放宽）
    """
    names = TASK_FILE_NAMES
    if turn.last_cmd_result and TASK_FILE_MARKER not in turn.last_cmd_result:
        names = TASK_PROBE_NAMES
    return (
        f"for f in $({_sandbox_find_first(names, TASK_FIND_PRINT, TASK_FILE_MAX)});"
        f' do echo "{TASK_FILE_MARKER}$f";'
        f' cat -- "$f" 2>/dev/null | head -{TASK_FILE_LIMIT};'
        f' echo "{TASK_FILE_END}"; done'
    )


def _sandbox_probe(turn: Turn) -> str:
    """任务描述里找不到文件名时，探测沙盒里的任务文件

    把沙盒里文件名像任务文件的那些连同完整路径一起打印出来，下一回合
    `_sandbox_command` 就能从输出里认出该读哪个文件。探测输出带
    `TASK_PROBE_MARKER`：`_task_answer` 只认 `TASK_MARKER`，
    所以文件路径清单永远不会被当成答案提交；工作目录与目录列表排在
    路径清单之后，只作诊断线索。

    搜索先用任务根目录（`_sandbox_find_first`）：任务文件就在那里时不必扫全盘，
    一条命令的 15 秒限时因此不会耗在无谓的 walk 上（复盘 PK590252 的 R13
    整条命令就是被全盘 find 拖到 `[TIMEOUT]` 的）。

    参数:
        turn: 当前回合信息

    返回:
        需要提交给沙盒执行的探测命令
    """
    token = _task_token(turn.phase_task)
    # 只按文件名找（task*/spec*，再退到 *.md）：描述里连文件名都没给，
    # 任何文档都可能是任务说明，但沙盒里的其他内容不该被当成任务文件读回来
    return (
        f'echo "{TASK_PROBE_MARKER}{token}"; '
        f"{_sandbox_find_first(TASK_PROBE_NAMES, TASK_FIND_PRINT, 40)}; "
        f"pwd; ls -a -- . 2>&1 | head -40; "
        f'echo "{TASK_END_MARKER}"; :'
    )


def _task_file(phase_task: str) -> str | None:
    """从任务描述里找出需要在沙盒中读取的文件名"""
    match = TASK_FILE_PATTERN.search(phase_task)
    return match.group(0) if match else None


def _task_token(phase_task: str) -> str:
    """任务短标识：长任务描述只会用到开头几个可打印字符"""
    return re.sub(r"\W+", "", phase_task)[:16]


def _task_answer(turn: Turn) -> str | None:
    """从上一回合的沙盒输出中解析当前任务的答案

    输出格式约定为 "[exitCode:N]\\n<输出>"（见接口文档），因此只有执行成功
    且带有本任务标识的输出才会被当作答案，避免答非所问或复用上一个任务的结果。
    答案取任务标识到 `TASK_END_MARKER` 之间、`[SOLUTION]` 段里的内容。

    四道闸门保证交上去的是答案本体（复盘里 4 次 submitAnswer 交的全是
    任务描述、PK590252 的 R17 交的又是文件路径、PK590557 的 R14/R16/R18
    交的是接口文档原文与错误 JSON，Judge 一次都没放行）：
        1. 答案区里必须出现过真实取数的证据（`TASK_DATA_MARKER`）——
           执行器取不到数据时答案区是空的，这一回合就不提交；
        2. 答案里不能出现任务描述里的中文长句（`_task_echo`）；
        3. 答案不能是接口的错误响应体（`_task_error_body`）：那说明这次取数
           其实失败了，只是失败信息被当成了正文；
        4. 答案不能是沙盒里那份文档的原文（`_task_text_answer`）：PK590851 的
           R13 交的就是接口文档原文，它是"取数取到了文档页"而不是答案；
        5. 答案不能是一条文件路径（`_task_path_answer`）：那说明开拓者把
           "该读哪个文件"当成了答案。
    走 LLM 那条路时输出里没有文档指纹可比，另有一道按"文档长什么样"判定的
    闸门（`_task_doc_body`，PK590836 的 R15 交的是 API 文档正文）。
    命中错误特征的输出（文件不存在等）同样不能提交：错误答案既拿不到分，
    又白白消耗任务冷却，所以宁可这一回合不提交，等下一条沙盒输出。
    """
    answer, _ = _task_answer_with_reason(turn)
    return answer


def _task_answer_with_reason(turn: Turn) -> tuple[str | None, str]:
    """答案判定的结果与原因码（原因码供 `task_brief` 写进日志）

    复盘里"任务没交卷"只能靠人翻沙盒输出猜原因，这里把判定过程本身变成
    可解析的字段：`ok` / `llm_answer` / `no_marker`（本任务的沙盒输出还没到）/
    `exit_nonzero`（命令失败）/ `no_api_data`（取不到数）/ `error_in_output` /
    `short_or_missing` / `echo_task_text`（答案就是任务原文）/
    `error_body`（答案是接口的错误响应体）/ `doc_text`（答案是沙盒里那份文档的
    原文）/ `path_answer`（答案是一条文件路径）/ `doc_body`（答案是 Markdown
    文档的正文，见 `_task_doc_body`）。
    """
    if not turn.phase_task:
        return None, "no_task"

    # 1. LLM 已经直接给出答案（不用绕沙盒）
    direct = _llm_direct_answer(turn)
    if direct is not None:
        return direct, "llm_answer"

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    result = turn.last_cmd_result
    if marker not in result:
        return None, "no_marker"
    if "[exitCode:0]" not in result:
        return None, "exit_nonzero"

    region = result.split(marker, 1)[1].split(TASK_END_MARKER, 1)[0]

    # 2. 上一回合跑的是 LLM 指定的取数命令：标记之间的输出本身就是答案
    llm_output = _llm_command_answer(turn, region)
    if llm_output is not None:
        return llm_output, "llm_cmd_output"

    if TASK_DATA_MARKER not in region:
        return None, "no_api_data"  # 没取到数据：沙盒里只有任务原文
    if any(bad in region for bad in TASK_ERROR_MARKERS):
        return None, "error_in_output"

    answer = _solution_answer(region, turn)
    if answer is None or len(answer) < TASK_ANSWER_MIN_LEN:
        return None, "short_or_missing"
    if _task_echo(answer, turn.phase_task):
        return None, "echo_task_text"
    if _task_error_body(answer):
        return None, "error_body"  # 交上去的是接口的错误提示，不是答案
    if _task_text_answer(answer, result):
        return None, "doc_text"  # 交上去的是沙盒里那份文档的原文，不是答案
    if _task_doc_body(answer):
        return None, "doc_body"  # 交上去的是一份 Markdown 文档的正文（S2）
    if _task_path_answer(answer):
        return None, "path_answer"  # 交上去的是一条路径：文件里问的答案还没拿到
    return answer, "ok"


def _task_path_answer(answer: str) -> bool:
    """答案是不是一条文件路径（S1：交文件路径 = 又一次 0 分）

    复盘 PK590252 的 R17 提交的正是
    "/tmp/selfEvolutionTask/1-fixed-step/1-unknown-api/task_1_beijing.md"
    ——开拓者找到了任务文件，却把"该读哪个文件"当成了答案交上去。整个答案
    就是一条带文档扩展名的路径（`TASK_FILE_PATTERN` 的全匹配）时才判定命中：
    "查询北京文化遗产"这类任务解出来的答案是一段数据，不会长成路径的样子，
    而多行/带空格的答案天然不会全匹配。

    参数:
        answer: 待提交的答案内容

    返回:
        True 表示这条答案看着就是文件路径，不能提交
    """
    text = answer.strip()
    return "\n" not in text and TASK_FILE_PATTERN.fullmatch(text) is not None


def _task_error_body(answer: str) -> bool:
    """答案是不是接口返回的错误体或异常回溯（S2）

    取数器把"有响应的正文"直接打进 `[SOLUTION]` 段，而错误响应往往也是
    200 + 一段 JSON（`{"status":"error","message":"missing query"}`，见
    `TASK_ERROR_BODY`），于是一段错误提示就变成了"答案"。复盘 PK590557 的
    R14/R16/R18 三次 submitAnswer 交的正是接口文档原文与 401/404 的错误
    JSON，Judge 全部判 0；更糟的是每交一次就消耗一次提交额度，真正的答案
    取到时反而可能已经撞上 `TASK_SUBMIT_LIMIT` 被看门狗放弃。

    这里不做"答案应该长什么样"的正面判定（任务千变万化），只排掉一眼能看出
    是错误体的那几种形态；命中时不提交，`_sandbox_command` 下一回合照常
    重跑取数命令。

    参数:
        answer: 待提交的答案内容

    返回:
        True 表示这条答案是错误体的正文，不能提交
    """
    return TASK_ERROR_BODY.search(answer) is not None


def _task_doc_body(answer: str) -> bool:
    """答案是不是一份 Markdown 文档的正文（S2）

    `_task_text_answer` 靠沙盒输出里的文档指纹判定"复读文档"，而指纹来自
    执行器（`[DOC]` 行）与 `_task_dump` 的回读段（`[TASK_FILE]`），只在执行器
    跑过的那一轮里才有。LLM 给的 `CMD: <命令>` 跑完之后，输出区里就是命令的
    原始 stdout，没有任何指纹可比——一条 `cat 接口文档.md` 的输出于是被原样
    当成答案交上去（复盘 PK590836 的 R15：`# 国家文化遗产数字档案查询系统 —
    API 参考文档…`，Judge 判 0）。这里按"文档长什么样"补一道闸门：

        - 首行是 Markdown 标题（`# 标题`，允许前三格缩进）
        - 并且正文不止一行，或者标题里就点名了"参考文档""版本"这类字样

    两条同时成立才拦。答案是一段取数结果（JSON、短字符串、一条记录），
    不会以 `# ` 起头又接着写好些行；反过来，单行的 `# xxx` 也放行，免得把
    某个恰好以井号开头的短答案误伤掉。

    参数:
        answer: 待提交的答案内容

    返回:
        True 表示这条答案是一份文档的正文，不能提交
    """
    lines = [line for line in answer.splitlines() if line.strip()]
    if not lines or not TASK_DOC_HEAD.match(lines[0]):
        return False
    return len(lines) > 1 or any(word in lines[0] for word in TASK_DOC_WORDS)


def _task_text_hints(result: str) -> list[str]:
    """沙盒输出里那些文档的开头（供 `_task_text_answer` 比对）

    两处来源：
        - `[DOC]` 行：执行器读到的接口文档开头（见 `TASK_EXECUTOR`）；
        - `[TASK_FILE]<路径>` 与 `[TASK_EOF]` 之间：`_task_dump` 回读的任务文件
          正文（第一行是路径，后面才是内容）。

    只取"开头"是有意的：原样复读的答案一定以文档开头起头，而正常取到的数据
    不会整段等于某份文档的开头，误伤的窗口因此小到可以忽略。

    参数:
        result: 报文的 `lastCmdResult`（沙盒输出）

    返回:
        文档开头的字符串列表（顺序按输出里出现的先后）
    """
    hints = []
    for line in result.splitlines():
        if line.startswith(TASK_DOC_MARKER):
            hints.append(line[len(TASK_DOC_MARKER):])
    for chunk in result.split(TASK_FILE_MARKER)[1:]:
        body = chunk.split(TASK_FILE_END, 1)[0]
        _, _, text = body.partition("\n")  # 第一行是文件路径，后面才是正文
        if text.strip():
            hints.append(text)
    return hints


def _task_text_answer(answer: str, result: str) -> bool:
    """答案是不是沙盒里某份文档（任务文件/接口文档）的原文（S1）

    复盘 PK590851 的 R13 把接口文档原文交了上去（0 分），PK590847 的 R13–R16
    沙盒里回读的也一直是同一份文档：`_task_echo` 只认任务描述里的中文长句，
    文档不在任务描述里；`TASK_ERROR_BODY` 只认错误体，文档也不是错误——两道
    闸门都拦不住，答案就这么交上去了。这里拿"沙盒里那些文档的开头"当指纹，
    命中就不提交（下一回合照常重跑取数命令，等真正的数据）。

    指纹与答案都先做空白归一化再比对：文档从沙盒里 `cat` 回来时行首缩进、
    换行位置未必与取回的那一份逐字相同，而"是不是同一段文字"才是要判的东西。

    判据只有输出里带得出的那几份文档：执行器把接口文档的开头打成了 `[DOC]` 行、
    `_task_dump` 把任务文件读了回来（`[TASK_FILE]` 段），两者都没有的输出
    （比如 LLM 自己给的一条 `cat 文档` 命令）这里比不了，交给上面几道闸门。

    参数:
        answer: 待提交的答案内容
        result: 报文的 `lastCmdResult`（文档与文档指纹都在里面）

    返回:
        True 表示这条答案是沙盒里文档的原文，不能提交
    """
    head = " ".join(answer.split())
    if not head:
        return False
    for text in _task_text_hints(result):
        hint = " ".join(text.split())[:TASK_TEXT_HINT]
        if len(hint) >= TASK_TEXT_HINT_MIN and hint in head:
            return True
    return False


def _solution_answer(region: str, turn: Turn) -> str | None:
    """从执行器的输出里取出当前任务那一段 `[SOLUTION]` 的内容

    执行器会把沙盒里的任务文件都试着解一遍（解出来的存进答案缓存，见
    `_remember_task_answers`），每一份打成 `[SOLUTION]<文件名>` 到
    `[/SOLUTION]` 的一段。这里只取与当前任务同名的那个：任务描述里没点名
    文件时取第一段（那正是探测结果里认出来的那一份）。

    参数:
        region: 沙盒输出里任务标识与 `TASK_END_MARKER` 之间的内容
        turn: 当前回合信息

    返回:
        当前任务的答案内容；没有可用的答案段时返回 None
    """
    blocks: dict[str, str] = {}
    for chunk in region.split(TASK_SOLUTION_MARKER)[1:]:
        path, _, body = chunk.partition("\n")
        name = path.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            blocks[name] = body.split(TASK_SOLUTION_END, 1)[0].strip()

    target = _task_file(turn.phase_task)
    if target is not None:
        return blocks.get(target.replace("\\", "/").rsplit("/", 1)[-1]) or None
    return next(iter(blocks.values()), None)


def _task_echo(answer: str, phase_task: str) -> bool:
    """答案是不是在复读任务原文

    任务描述里的中文长句出现在答案里，说明交上去的是任务文件的内容而不是
    执行结果（复盘里的 4 次 0 分提交都是这个形态）。答案来自接口取数，
    正常不会整句重复任务描述。
    """
    return any(run in answer for run in re.findall(TASK_ECHO_RUN, phase_task))


def _remember_task_answers(result: str) -> None:
    """把执行器解出来的答案按任务文件名记进答案缓存

    `_sandbox_command` 的执行器会把沙盒里的任务文件都试着解一遍，每份的答案
    用 `[SOLUTION]`（后跟文件名）与 `[/SOLUTION]` 分段；这里把"文件名 -> 答案"
    存下来，下一个任务点领到同一份任务时就能省掉一个来回的沙盒执行
    （见 `_cached_answer`）。

    缓存只增不改（`setdefault`）：已经记下的答案不会被后来的输出覆盖。

    只有"这一份之前出现过取数证据（`[API]`）"的答案才进缓存：执行器取不到数
    时也会把任务文件打成 `[SOLUTION]` 段（沙盒里本来就有这份文件），
    把它缓存下来等于把任务原文背下来，下一个任务一到手就被当成答案交上去
    ——这正是复盘里"四次 submitAnswer 交的全是任务描述"的成因之一。

    沙盒里那份文档的原文同样不进缓存（`_task_text_answer`）：`/docs` 这类
    地址把文档页当正文返回时，`[API]` 证据是有的，但缓存下来的仍然是文档
    ——下一个任务点一到手就会把它当答案秒交（PK590851 的 R13 正是这么交的）。
    文档正文（`_task_doc_body`）同理。

    参数:
        result: 报文的 `lastCmdResult`（上回合沙盒命令的输出）
    """
    chunks = result.split(TASK_SOLUTION_MARKER)
    evidence = chunks[0]
    for chunk in chunks[1:]:
        path, _, body = chunk.partition("\n")
        answer = body.split(TASK_SOLUTION_END, 1)[0].strip()
        name = path.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if (
            name
            and answer
            and TASK_DATA_MARKER in evidence
            and not _task_text_answer(answer, result)
            and not _task_doc_body(answer)
        ):
            _TASK_ANSWER_CACHE.setdefault(name, answer)
        evidence += TASK_SOLUTION_MARKER + chunk


def _cached_answer(turn: Turn) -> str | None:
    """当前任务在答案缓存里的答案（沙盒里之前解出来的同名任务）

    积分 = 任务奖励 + 5 × 标准回合数 / (完成回合 - 接取回合)（任务书第六章），
    完成回合差越小分越高。缓存命中时开拓者在任务进行中的第一个回合就能交卷，
    把回合差压到 1（复盘里敌方两次 cached submit 各得 155 分，我们则要重新
    执行一遍沙盒、回合差至少 2）。

    任务描述里没点名文件时返回 None：探测出来的文件名与任务描述的对应关系
    不确定，宁可多花一个来回执行一次，也不拿别的任务的答案去作答。

    缓存里那条答案本身也要过 `_task_path_answer`、`_task_error_body` 与
    `_task_doc_body` 三道闸门：缓存是在执行器输出上直接建的
    （`_remember_task_answers`），同一份"答案"从这里出去同样可能是一条文件
    路径、一段接口错误提示或者一份文档的正文——提交闸门只在 `_task_answer`
    里拦一道的话，这条路就绕过去了。
    """
    target = _task_file(turn.phase_task)
    if target is None:
        return None
    answer = _TASK_ANSWER_CACHE.get(target.replace("\\", "/").rsplit("/", 1)[-1])
    if answer is None or _task_path_answer(answer) or _task_error_body(answer):
        return None
    if _task_doc_body(answer):
        return None  # 缓存里那条"答案"是一份文档的正文，同样不能交（S2）
    return answer


def _go_mine(
    turn: Turn,
    unit: Unit,
    mine_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """让单位去采矿（只有工人能干这活）

    任务书 4.4 的指令表里 `collect` 的可用角色一栏只写了工人，别的角色下达
    采集会被判题系统整条打回。复盘 PK590881 的 R11–R15 连续 5 个回合
    `[COMMAND_ERROR] role 20011 (pioneer) wants collect, but only worker can
    do this action`：开拓者被支去采石，矿一块没采到（任务角色还得留在任务点
    周围一格内，根本走不到矿边），任务窗口也一起耗光了。这里做最后一道闸门，
    非工人一律不下发采集/走向矿点，调用方按"这名角色干不了这活"另作安排。
    """
    if unit.kind != WORKER or unit.backpack_full:
        return False

    mines = [
        m for m in turn.get_mines(mine_type)
        if m not in claimed
    ]
    if not mines:
        return False

    # 按距离排序
    mines.sort(key=lambda m: (distance(unit.pos, m), m.x, m.y))

    for mine in mines:
        # 如果已经相邻,采集
        if unit.pos != mine and distance(unit.pos, mine) <= 1:
            commands[unit.unit_id] = collect_command(mine)
            claimed.add(mine)
            return True

        # 否则向矿点移动
        step = _step_toward(turn, unit, mine, claimed)
        if step is not None:
            commands[unit.unit_id] = move_command(step)
            return True

    return False


def _gather_logic(
    turn: Turn,
    worker: Unit,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
    *,
    prefer_stone: bool = False,
    plan: DayPlan | None = None,
) -> bool:
    """保证工人每回合都有产出：按矿种分工就近采集

    石材是围墙材料，优先采；两名工人时按"矿种互补"分工（见 `_mine_order`），
    一名采石、一名采铁/铜换金币，不会一起挤在同一个矿点上；分工里负责的矿种
    附近没有（或已被其他角色占住）时退回其他矿种，保证不会整回合没有指令。

    参数:
        turn: 当前回合信息
        worker: 当前决策的工人
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
        prefer_stone: 为 True 时全员石材优先（围墙配额还没铺满时用）
        plan: 当天的计划（为 None 时由 `_mine_order` 现算一份）

    返回:
        True 表示本回合已下达采集或移动指令
    """
    for mine_type in _mine_order(
        turn, worker, prefer_stone=prefer_stone, plan=plan,
    ):
        if _go_mine(turn, worker, mine_type, claimed, commands):
            return True
    return False


def _build_or_walk(
    turn: Turn,
    unit: Unit,
    target: Pos,
    building_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """如果在建造位置旁边则建造,否则向目标移动

    返回:
        True 表示本回合已下达指令（建造或移动）；False 表示无路可走，
        调用方应把建造位留给其他角色而不是空占着
    """
    # 已经在目标旁边(但不是目标本身),执行建造
    if unit.pos != target and distance(unit.pos, target) <= 1:
        commands[unit.unit_id] = build_command(target, building_type)
        claimed.add(target)
        return True

    # 否则向目标移动
    step = _step_toward(turn, unit, target, claimed)
    if step is not None:
        commands[unit.unit_id] = move_command(step)
        return True
    return False


def _step_toward(
    turn: Turn,
    unit: Unit,
    target: Pos,
    claimed: set[Pos],
    *,
    inside_only: bool = False,
) -> Pos | None:
    """计算向目标移动的下一步

    先把目标周围可停留的格子按优先级试一遍（见 `_valid_stand_cells`）；
    落脚点都能走到、只是最顺路的那一步被队友认领时，退而走"朝目标最近的
    一步"（见 `_closest_step`），而不是判定目标不可达。
    """
    # 计算可停留的格子
    stand_cells = _valid_stand_cells(turn, unit, target, claimed, inside_only)
    reachable = False

    for stand in stand_cells:
        # 已经在目标位置
        if stand == unit.pos:
            return None

        # 计算路径
        step = next_step(turn, unit, stand)
        if step is None:
            continue
        reachable = True
        if step in claimed:
            continue

        claimed.add(step)
        return step

    # 落脚点走得到、第一步只是被队友认领时，不能就此认定"目标走不通"：
    # 旧实现返回 None，调用方于是改做别的事（工人掉头去采矿、开拓者原地发呆），
    # 复盘里的"计划说建塔、指令里一条 build 都没有"就是这么来的（issue #34）。
    if reachable:
        return _closest_step(turn, unit, target, claimed, inside_only)
    return None


def _closest_step(
    turn: Turn,
    unit: Unit,
    target: Pos,
    claimed: set[Pos],
    inside_only: bool,
) -> Pos | None:
    """退路：在本体可走、没人认领的相邻格里挑一个离目标最近的

    只作 `_step_toward` 的兜底用：A* 找到的落脚点都可达，但通往它们的第一步
    落在队友这一回合认领的格子上（同一回合多个角色会一起抢相邻格）时，
    仍然朝目标方向走一步，而不是原地判定"走不通"。

    目标已经不比当前格更近的相邻格一律不选：等距或变远的移动只是折返
    （复盘里"工人 (30,7)→(31,6)→(30,7) 两回合原地打转"），
    这时返回 None、由调用方按"原地待命"处理。白天同样不踩武器塔/围墙的
    建造点（站上去会把那一格占住，见 `_valid_stand_cells`）。

    参数:
        turn: 当前回合信息
        unit: 移动的单位
        target: 目标位置
        claimed: 已被其他角色认领的格子
        inside_only: 为 True 时只保留基地周围1格范围内的格子

    返回:
        本回合要移动到的格子；没有更近的可站位置时返回 None
    """
    station = turn.station()
    footprint = station_footprint(station.pos) if station else ()
    blocked = turn.blocked(unit)
    build_sites = _reserved_build_sites(turn) if turn.is_day else frozenset()
    here = distance(unit.pos, target)

    best: Pos | None = None
    for pos in get_neighbors(unit.pos):
        if not turn.land(pos) or pos in blocked or pos in claimed:
            continue
        if pos in build_sites:
            continue
        if inside_only and _footprint_distance(pos, footprint) > 1:
            continue
        if distance(pos, target) >= here:
            continue
        if best is None or (distance(pos, target), pos.x, pos.y) < (
            distance(best, target), best.x, best.y,
        ):
            best = pos

    if best is not None:
        claimed.add(best)
    return best


def _valid_stand_cells(
    turn: Turn,
    unit: Unit,
    target: Pos,
    claimed: set[Pos],
    inside_only: bool = False,
) -> list[Pos]:
    """计算目标周围可停留的格子

    参数:
        turn: 当前回合信息
        unit: 移动的单位
        target: 目标位置（建造点、矿点、武器或小贩坐标）
        claimed: 已被其他单位声明的格子（防止多个角色在同一回合争抢同一格）
        inside_only: 为 True 时只保留基地周围1格范围内的格子，
                     用于避免开拓者跑去远处挡住工人的建造路线

    返回:
        按离基地距离升序排列的可停留格子列表（优先靠近基地）

    逻辑:
        1. 取目标位置的八方向相邻格子
        2. 过滤掉非陆地、被阻挡（建筑/单位/机器人/中立元素）以及已被占用的格子
        3. 白天避开武器塔/围墙的建造点（角色占住建造位会让建筑永远建不起来）；
           真的无处落脚时才退回旧行为允许站在建造点上（目标本身是建造点时），
           否则角色会因为"相邻格全是建造位"而无处落脚，反而建不起来
        4. inside_only 为 True 时进一步限制在基地周围
        5. 按离基地的切比雪夫距离排序
    """
    station = turn.station()
    footprint = station_footprint(station.pos) if station else ()
    blocked = turn.blocked(unit)

    # 目标的八方向相邻格子
    neighbors = get_neighbors(target)

    cells = [
        pos for pos in neighbors
        if turn.land(pos)
        and pos not in blocked
        and (pos == unit.pos or pos not in claimed)
        and (
            not inside_only
            or _footprint_distance(pos, footprint) <= 1
        )
    ]

    # 白天避开建造点: 角色站上去会把这一格占住,武器/围墙就再也建不起来了。
    # 走去施工（目标本身就是建造点）时旧实现会整体放行建造点，于是角色顺路
    # 站到别的塔位/墙位上，另一名工人这一回合就建不成（复盘里的"下了建造指令、
    # 下回合金币没扣、塔也没出现"）；只有实在无处落脚时才退回旧行为。
    if turn.is_day:
        build_sites = _reserved_build_sites(turn)
        outside = [pos for pos in cells if pos not in build_sites]
        if outside or target not in build_sites:
            cells = outside
        else:
            cells = [pos for pos in cells if pos != target]

    # 按离基地的距离排序（优先靠近基地）
    cells.sort(key=lambda pos: (_footprint_distance(pos, footprint), pos.x, pos.y))
    return cells


def _reserved_build_sites(turn: Turn) -> frozenset[Pos]:
    """本回合规划中的建造点（武器塔 + 围墙）

    这些格子要留给施工：一旦被角色占住，建造点就会被判为"已占用"而从
    待建列表里消失，对应的塔或围墙整局都建不起来。
    """
    return frozenset(_calc_tower_sites(turn)) | frozenset(_calc_wall_order(turn))


def _footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    """计算点到footprint的最小距离"""
    if not footprint:
        return 0
    return min(distance(pos, cell) for cell in footprint)


def _bfs_reachable(
    turn: Turn,
    starts: frozenset[Pos],
    goals: frozenset[Pos],
    blocked: set[Pos],
) -> bool:
    """从 starts 出发能否走到 goals 里任意一格（八方向BFS，blocked 视为墙）

    只在塔位校验里用，所以单独写一个小 BFS，不复用 A*（A* 要顺带算代价与
    路径，这里只关心连通性）。

    参数:
        turn: 当前回合信息（取地图边界）
        starts: 起点集合
        goals: 目标格集合
        blocked: 不可通行格

    返回:
        True 表示至少有一个目标格可达
    """
    if not starts or not goals:
        return False
    frontier = [pos for pos in starts if pos not in blocked]
    seen = set(frontier)
    while frontier:
        current = frontier.pop()
        if current in goals:
            return True
        for neighbor in get_neighbors(current):
            if neighbor in seen or neighbor in blocked:
                continue
            if not turn.land(neighbor):
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return False


def _tower_sites_reachable(turn: Turn, sites: tuple[Pos, ...]) -> bool:
    """校验塔位布局：三座塔都建成后，每座塔旁是否仍有可达的操控落脚点

    塔本身是障碍物（任务书4.1：建筑阻挡移动）。塔位选错会把基地 6×6 区域的
    通道切断，出现"塔在、但操控者走不过去"的情况——聊天记录原话：
    "很容易出现炮塔把路堵住然后有一个炮塔碰不到的情况"，那样等于白扔 25 金，
    夜里还少一门火力。

    做法:
        1. 把候选布局里的三座塔临时视为已建成（加入阻挡集）
        2. 对每座塔，取它八邻域里可通行、且不被其它塔占住的格子作为候选落脚点
        3. 用 BFS 校验"至少有一个角色能走到其中某格"

    参数:
        turn: 当前回合信息
        sites: 候选塔位布局（未建成，按建造顺序）

    返回:
        True 表示每座塔都存在可达的操控位
    """
    if not sites:
        return True

    movers = turn.controllable()
    blocked: set[Pos] = {pos for pos, kind in turn.zones.items() if kind != LAND}
    blocked.update(sites)  # 候选塔位按"已经建成"处理
    blocked.update(turn.occupied_cells())
    for enemy in turn.enemies:
        if enemy.is_alive:
            blocked.update(turn.footprint(enemy))
    for robot in turn.robots:
        if robot.is_alive:
            blocked.add(robot.pos)
    # 角色自己占的格子不算障碍（它们会走开）
    for unit in movers:
        blocked.discard(unit.pos)

    starts = frozenset(unit.pos for unit in movers)
    if not starts:
        station = turn.station()
        if station is None:
            return True
        starts = frozenset(
            neighbor
            for cell in station_footprint(station.pos)
            for neighbor in get_neighbors(cell)
            if neighbor not in blocked and turn.land(neighbor)
        )

    for site in sites:
        stands = frozenset(
            neighbor for neighbor in get_neighbors(site)
            if neighbor not in blocked and turn.land(neighbor)
        )
        if not stands:
            return False  # 这座塔被邻居格堵死，谁都站不到旁边
        if not _bfs_reachable(turn, starts, stands, blocked):
            return False
    return True


def _calc_tower_sites(
    turn: Turn,
    preferred: str | None = None,
) -> tuple[Pos, ...]:
    """计算武器塔建造位置（基地周围一圈）

    取基地 2x2 占地周围距离为1且可通行的格子，按上/左/下/右四个方位各取
    一个代表点，再按"先朝敌方来路、后朝地图内侧"的顺序取前三个，使三座
    武器覆盖不同方向而不挤在基地同一侧，并优先罩住敌人来的那一侧
    （复盘指出塔位只按地图空间选，没参考敌方来路）。
    分别对应加特林、电磁狙击炮、火箭发射台。

    参数:
        turn: 当前回合信息
        preferred: LLM 计划指定的布防方位（up/left/down/right），None 表示自动

    返回:
        按建造顺序排列的武器塔位置（最多3个）
    """
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 基地footprint周围距离为1的格子，按方位分组
    groups: dict[str, list[Pos]] = {side: [] for side in TOWER_SIDES}
    for cell in footprint:
        for neighbor in get_neighbors(cell):
            if neighbor in footprint or not turn.land(neighbor):
                continue
            groups[_side_of(neighbor, xmin, xmax, ymax)].append(neighbor)

    # 每个方位取最居中的一个格子作为代表，再按"敌人在哪边"排序
    enemy_sides = _enemy_sides(turn)
    sites = [
        (side, _side_representative(side, cells))
        for side, cells in groups.items()
        if cells
    ]
    sites.sort(key=lambda item: (
        _side_priority(item[0], enemy_sides, preferred),
        -_side_room(item[0], turn, xmin, xmax, ymin, ymax),
        TOWER_SIDES.index(item[0]),
    ))

    # 按优先级取前3个位置；若这个布局会把某座塔堵到"没人能操控"，
    # 换下一个组合重试（四方位取三，最多 4 种组合，穷举成本可忽略）
    ranked = [pos for _, pos in sites]
    for combo in combinations(ranked, 3):
        layout = tuple(combo)
        if _tower_sites_reachable(turn, layout):
            return layout
    # 所有组合都不可达（例如基地被围墙围死）时退回原顺序，行为与改造前一致
    return tuple(ranked[:3])


def _enemy_sides(turn: Turn) -> frozenset[str]:
    """敌方主力大致来自基地的哪几个方位

    优先用可见的敌方基地定位来路（敌方单位都是从它出发的），看不到敌方基地
    时退回最近的敌方单位。两个都看不到时返回空集合，塔位排序退回原来的
    "朝地图内侧空间"，策略与改造前完全一致。

    参数:
        turn: 当前回合信息

    返回:
        方位集合（up/left/down/right 的子集）
    """
    station = turn.station()
    if station is None:
        return frozenset()

    visible = [enemy for enemy in turn.enemies if enemy.is_alive]
    candidates = [unit for unit in visible if unit.kind == STATION] or visible
    if not candidates:
        return frozenset()

    enemy = min(
        candidates,
        key=lambda unit: (distance(unit.pos, station.pos), unit.pos.x, unit.pos.y),
    )
    dx = enemy.pos.x - station.pos.x
    dy = enemy.pos.y - station.pos.y

    sides = set()
    if dx < 0:
        sides.add("left")
    elif dx > 0:
        sides.add("right")
    if dy < 0:
        sides.add("down")
    elif dy > 0:
        sides.add("up")
    return frozenset(sides)


def _side_priority(
    side: str,
    enemy_sides: frozenset[str],
    preferred: str | None,
) -> int:
    """塔位排序的第一权重：朝敌方来路 > LLM 指定 > 其他

    敌方来路是实打实的坐标推算（`_enemy_sides`：可见的敌方基地，看不到基地
    时取最近的敌方单位），LLM 的 `defend` 只是照着局面猜的方位。复盘里计划
    一路写死 `defend=up`，敌方基地却在我方左下方，第一座塔因此压在没人来的
    那一侧（"defend 方位按敌我坐标推算，替换写死的 defend=up"）。看不到敌方
    单位时 `enemy_sides` 为空，LLM 指定的方位照旧优先。
    """
    if side in enemy_sides:
        return 0
    if preferred is not None and side == preferred:
        return 1
    return 2


def _side_of(pos: Pos, xmin: int, xmax: int, ymax: int) -> str:
    """判断外围格子位于基地的哪一侧（上边=ymax+1，下边=ymin-1）"""
    if pos.x < xmin:
        return "left"
    if pos.x > xmax:
        return "right"
    if pos.y > ymax:
        return "up"
    return "down"


def _side_representative(side: str, cells: list[Pos]) -> Pos:
    """取某个方位上最居中的候选格子，避免武器全部偏向一侧的角落"""
    if side in ("left", "right"):
        middle = (min(pos.y for pos in cells) + max(pos.y for pos in cells)) / 2
        return min(cells, key=lambda pos: (abs(pos.y - middle), pos.x, pos.y))

    middle = (min(pos.x for pos in cells) + max(pos.x for pos in cells)) / 2
    return min(cells, key=lambda pos: (abs(pos.x - middle), pos.x, pos.y))


def _side_room(
    side: str,
    turn: Turn,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
) -> int:
    """方位朝向地图内侧的空间：空间越大越可能迎来敌人"""
    if side == "left":
        return xmin
    if side == "right":
        return turn.width - 1 - xmax
    if side == "up":
        return turn.height - 1 - ymax
    return ymin


def _wall_side_order(turn: Turn) -> tuple[str, ...]:
    """围墙的铺设优先顺序：先封敌方来路，其余按"上 -> 左 -> 下 -> 右"

    复用塔位那套方位判定（`_enemy_sides`），让围墙与炮塔压在同一侧：来路先被
    墙压窄、再被塔罩住，机器人只能顶着火力拆墙（复盘里"防守方 0 段围墙、正面
    无任何阻挡"，以及"用墙把来路压缩进塔的射程"）。

    看不到敌方单位时各方位的优先级相同，顺序退化成"上 -> 左 -> 下 -> 右"，
    与改造前完全一致。

    参数:
        turn: 当前回合信息

    返回:
        四个方位的铺设顺序（up/left/down/right 的一个排列）
    """
    enemy_sides = _enemy_sides(turn)
    return tuple(sorted(
        TOWER_SIDES,
        key=lambda side: (
            _side_priority(side, enemy_sides, None),
            TOWER_SIDES.index(side),
        ),
    ))


def _wall_ring(turn: Turn) -> dict[str, list[Pos]]:
    """基地第二圈的四个方位格子表（即 WALL_PLAN 的坐标来源，S3）

    同一格不会落在两条边上，所以四张表拼起来正好是环绕基地的一圈。
    超出地图或落在非陆地上的格子由 `_calc_wall_order` 统一过滤。
    """
    station = turn.station()
    if station is None:
        return {}

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    return {
        # 上边（从右到左）
        "up": [Pos(x, ymax + 2) for x in range(xmax + 2, xmin - 3, -1)],
        # 左边（从上到下）
        "left": [Pos(xmin - 2, y) for y in range(ymax + 1, ymin - 2, -1)],
        # 下边（从左到右）
        "down": [Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)],
        # 右边（从下到上）
        "right": [Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)],
    }


def _wall_entrance(turn: Turn, by_side: dict[str, list[Pos]]) -> Pos | None:
    """围墙入口：开在威胁最小那一侧的中间格

    旧实现把入口写死在地图右下角，敌方从右侧/下侧来时，围墙正好在来敌方向留了
    一个口子（复盘里"围墙留口/方向错位"）。入口跟着 `_wall_side_order` 的最后
    一位走——那正是"最不可能来敌人"的那一侧，来敌方向的墙因此始终是封死的。

    参数:
        turn: 当前回合信息
        by_side: `_wall_ring` 给出的四方位格子表

    返回:
        入口坐标；没有基地或该侧没有格子时返回 None（不开口子）
    """
    order = _wall_side_order(turn)
    if not order:
        return None
    cells = by_side.get(order[-1]) or []
    if not cells:
        return None
    return cells[len(cells) // 2]


def _calc_wall_order(turn: Turn) -> tuple[Pos, ...]:
    """计算围墙建造顺序（基地周围第二圈，即复盘建议里的 WALL_PLAN）

    按 `_wall_side_order` 给出的方位顺序（先敌方来路、其余上左下右）环绕基地
    铺一圈围墙——坐标由基地位置现算（`_wall_ring`），因此来敌方向永远排在最
    前面，而不是"顺路在采石点旁随手砌一段"（复盘里 (32,12)/(33,12) 那两段正是
    贴着采石点砌的，来敌方向反而留了口子）。入口开在威胁最小的一侧
    （`_wall_entrance`），超出地图或落在非陆地上的点会被过滤掉。
    """
    station = turn.station()
    if station is None:
        return ()

    by_side = _wall_ring(turn)
    order = [
        pos
        for side in _wall_side_order(turn)
        for pos in by_side[side]
    ]
    entrance = _wall_entrance(turn, by_side)

    return tuple(
        pos for pos in order
        if pos != entrance
        and turn.land(pos)
        and 0 <= pos.x < turn.width
        and 0 <= pos.y < turn.height
    )


# === 基地布局（V4）===

# 背面方位表：正面（主要来敌方向）的对面就是后排，R3 与 K0 在这一侧
_OPPOSITE_SIDE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass(frozen=True, slots=True)
class BaseLayout:
    """基地布局锚点（V4：按来敌方位定向，基地在左上/右下时自动反转）

    字段:
        front: 主要来敌方位（up/left/down/right）
        back: 背面方位（front 的反面）
        corners: 墙内一圈的四个角，顺序为 R1(前上)/R2(前下)/R3(后上)/K0(后下)；
                 格子不在地图上（基地贴地图边）时为 None

    站位分工（V4）：R1 是开拓者，R2 是石工（第 1 名工人，正面下角），
    R3 是铜工（第 2 名工人，后上角）；K0 是入夜前 R3 退守的安全位。
    布局里的三个炮位（正面两座、背面一座）不进这个对象：塔位仍由
    `_calc_tower_sites` 按"三面分散 + 可达性校验"选（理由见模块开头的
    V4 常量说明）。
    """

    front: str
    back: str
    corners: tuple[Pos | None, Pos | None, Pos | None, Pos | None]

    @property
    def hold(self) -> Pos | None:
        """K0：R3 入夜前的安全位"""
        return self.corners[3]


def _side_corridor(
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    side: str,
) -> tuple[Pos, ...]:
    """墙内一圈（离基地一格）在某一侧的四个格子，从"头"排到"尾"

    槽位含义（V4 布局）：0=小人站位、1=炮位、2=炮位/安全位、3=小人站位。
    竖边（left/right）按 y 从大到小排、横边（up/down）按 x 从小到大排，
    这样"头"永远是布局意义上的前上角（R1）、"尾"是前下角（R2），基地在
    哪个角落都不用另写一套坐标。
    """
    if side in ("left", "right"):
        x = xmin - 1 if side == "left" else xmax + 1
        return tuple(Pos(x, y) for y in (ymax + 1, ymax, ymin, ymin - 1))
    y = ymax + 1 if side == "up" else ymin - 1
    return tuple(Pos(x, y) for x in (xmin - 1, xmin, xmax, xmax + 1))


def _base_layout(turn: Turn) -> BaseLayout | None:
    """算出 V4 布局的锚点（没有基地时返回 None）

    来敌方位沿用围墙那套判定（`_wall_side_order` 的第一位就是先封的那一侧），
    所以"正面"永远是敌人来的方向；基地贴地图边时图外的格子由 `turn.land`
    过滤掉，拿不到坐标的角在 `corners` 里是 None。

    参数:
        turn: 当前回合信息

    返回:
        布局锚点；没有基地时返回 None
    """
    station = turn.station()
    if station is None:
        return None

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    front = _wall_side_order(turn)[0]
    back = _OPPOSITE_SIDE[front]
    front_cells = _side_corridor(xmin, xmax, ymin, ymax, front)
    back_cells = _side_corridor(xmin, xmax, ymin, ymax, back)

    corners = tuple(
        pos if turn.land(pos) else None
        for pos in (front_cells[0], front_cells[3], back_cells[0], back_cells[2])
    )
    return BaseLayout(front=front, back=back, corners=corners)


def _stand_role(turn: Turn, unit: Unit) -> int:
    """该角色在 V4 布局里占哪个站位（0=R1 开拓者 / 1=R2 石工 / 2=R3 铜工）

    分工与 `_stand_for` 共用一套判定：开拓者是 R1，第 1 名工人是 R2（正面
    下角），第 2 名工人是 R3（后上角）；只剩一名工人时它一个人兼两摊，按
    R2 算（修墙更要紧）。
    """
    if unit.kind == PIONEER:
        return 0
    workers = turn.workers()
    if len(workers) < 2 or (workers and unit.unit_id == workers[0].unit_id):
        return 1
    return 2


def _stand_for(turn: Turn, unit: Unit) -> Pos | None:
    """该角色在 V4 布局里的站位（R1 开拓者 / R2 石工 / R3 铜工）

    站位不在地图上（基地贴地图边）时返回 None，调用方退回原来的"就近站位"。

    参数:
        turn: 当前回合信息
        unit: 待判断的角色

    返回:
        该角色的站位坐标；没有基地或站位不在图内时返回 None
    """
    layout = _base_layout(turn)
    if layout is None:
        return None
    return layout.corners[_stand_role(turn, unit)]


def _dusk_stand(turn: Turn, unit: Unit) -> Pos | None:
    """入夜前的退守位（V4）：R3（铜矿工人）先退到 K0，其他人回自己的站位

    K0 在后排炮位旁边，又正对敌方进场方向的身后（V4："R3 位置会毒死 E1
    进入，因此入夜时小人 R3 需要先站在 K0 位置"）；K0 拿不到坐标时退回它
    自己的站位 R3。没有布局时返回 None，调用方按"就近武器"处理。
    """
    layout = _base_layout(turn)
    if layout is None:
        return None
    if _stand_role(turn, unit) == 2:
        return layout.hold if layout.hold is not None else _stand_for(turn, unit)
    return _stand_for(turn, unit)


def _game_day(turn: Turn) -> int:
    """当前是第几个游戏日（从 1 开始，与 `_day_round` 同一套算法）"""
    return (turn.round_no - 1) // ROUNDS_PER_DAY + 1


# === LLM 策略咨询 ===


def _llm_plan(payload: dict[str, Any]) -> LlmPlan:
    """把上一回合的 LLM 建议（llmResp）解析成有界的作战计划

    复盘里反复出现的"LLM建议与指令脱节"：建议内容从来没进过指令生成器。
    这里只认 prompt 约定的 `PLAN:` 行，逐键做白名单 + 范围钳制——LLM 只能在
    既有策略的旋钮里做选择，不能凭空发明动作：

        tower   今天要保证建成的塔数（钳到 LLM_MIN_TOWERS..LLM_MAX_TOWERS）
        wall    今天优先铺好的围墙段数（钳到 0..LLM_MAX_WALLS）
        upgrade 今天是否允许买武器升级券
        defend  优先布防的方位（up/left/down/right）

    解析不出来时（没写 PLAN 行、字段缺失、值越界）逐项退回默认值，默认值
    等价于改造前的策略，所以 LLM 不配合也不会让决策变差。建议里先复述了
    prompt 的占位模板、再给出真正的计划时，按"先解析出来的值优先"合并，
    模板本身（`tower=<1-3>` 这类尖括号写法）解析不出任何字段。

    计划每回合都从 `llmResp` 现解析（不落任何跨回合缓存），因此它的生效
    窗口与"报文里带着 LLM 回复"的回合一致，回复消失后自动回到默认计划；
    塔位/塔数/围墙配额这些旋钮的作用期正好是开局那几天，与回复到达的时机
    吻合。

    参数:
        payload: 判题系统原始请求（取 llmResp 字段）

    返回:
        解析后的计划；无法解析时返回 LLM_PLAN_DEFAULT
    """
    text = str(payload.get("llmResp") or "")
    # 建议里可能先复述了 prompt 的占位模板、再给出真正的计划，
    # 因此逐行扫所有 PLAN 行，按"先解析出来的值优先"合并，别被模板带偏
    items: dict[str, str] = {}
    for match in LLM_PLAN_LINE.finditer(text):
        for key, value in LLM_PLAN_ITEM.findall(match.group("body")):
            items.setdefault(key.lower(), value.lower())
    if not items:
        return LLM_PLAN_DEFAULT

    def _bounded(key: str, default: int, low: int, high: int) -> int:
        raw = items.get(key)
        if raw is None or not raw.isdigit():
            return default
        return min(max(int(raw), low), high)

    upgrade = LLM_PLAN_DEFAULT.upgrade
    if items.get("upgrade") in LLM_PLAN_TRUE:
        upgrade = True
    elif items.get("upgrade") in LLM_PLAN_FALSE:
        upgrade = False

    defend = items.get("defend")
    if defend not in TOWER_SIDES:
        defend = None

    return LlmPlan(
        tower=_bounded(
            "tower", LLM_PLAN_DEFAULT.tower, LLM_MIN_TOWERS, LLM_MAX_TOWERS,
        ),
        wall=_bounded("wall", LLM_PLAN_DEFAULT.wall, 0, LLM_MAX_WALLS),
        upgrade=upgrade,
        defend=defend,
    )


def _opening_round(turn: Turn) -> bool:
    """是否还在开局回合（每个游戏日的前 `OPENING_ROUNDS` 个回合）

    按"每个游戏日"而不是"整局"计算：夜里塔可能被拆掉，第二天开局同样要
    先把火力补回下限，否则白天又要在没有塔的情况下空转好几个回合。
    """
    return (turn.round_no - 1) % ROUNDS_PER_DAY < OPENING_ROUNDS


def _day_round(turn: Turn) -> int:
    """当前是这一天的第几个回合（从 0 开始，与 `_opening_round` 同一套算法）"""
    return (turn.round_no - 1) % ROUNDS_PER_DAY


def _tower_target(turn: Turn, plan: LlmPlan) -> int:
    """本回合要保证建成的武器塔数量（含金币闲置熔断与开局下限）

    正常情况下就是 LLM 计划里的 `tower`（默认满编 3 座）；但金币已经攒到
    `GOLD_FLUSH_TOWERS`（够再建两座塔）时一律提到满编：复盘里"金币 75 只花
    25、余下 50 连躺三个回合"的根因就是计划把塔数配额压低后金币再没有出口。
    金币留在手里不产生任何防御力，宁可多建一座塔。

    开局的前 `OPENING_ROUNDS` 个回合还额外有 `OPENING_MIN_TOWERS` 的硬下限：
    复盘里"首日 3 回合只落地 1 座塔、R2 整回合零建造"，第 2 座塔拖到 R3 才
    开工，而敌方同一局是单回合双建——火力成型的快慢不该由当天 LLM 计划决定。
    塔数已经达标时这条下限不起作用，金币仍然可以按计划留给升级券。

    熔断只在金币富余时生效（阈值高于单座造价），所以 LLM 仍然可以为
    "留钱买升级券"而少建一座塔；`upgrade` 开关与围墙配额都不受影响。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        白天要保证建成的塔数（LLM_MIN_TOWERS..LLM_MAX_TOWERS）
    """
    if turn.gold >= GOLD_FLUSH_TOWERS:
        return LLM_MAX_TOWERS
    if _opening_round(turn):
        return max(plan.tower, OPENING_MIN_TOWERS)
    return plan.tower


def _pending_tower_sites(turn: Turn, plan: LlmPlan) -> list[tuple[int, Pos]]:
    """执行层这一回合真正还要建的塔位（下标 -> 坐标，保持建造顺序）

    判定与 `_decide_day` 完全一致：`_calc_tower_sites` 规划的位置，去掉已经
    建成的塔、再去掉这一回合被占住的格子（`_buildable_sites`）。下标是塔位在
    完整规划里的序号——武器类型由 `TOWER_LOADOUT` 与下标绑定（见
    `_worker_day_logic`），带上序号才能把"要建哪座塔"与"建的是哪种武器"对上。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        (塔位下标, 坐标) 列表；没有待建塔位时为空
    """
    sites = _calc_tower_sites(turn, plan.defend)[:_tower_target(turn, plan)]
    built = {unit.pos for unit in turn.weapons()}
    buildable = set(_buildable_sites(
        turn,
        [site for site in sites if site not in built],
        turn.occupied_cells(),
    ))
    return [
        (index, site)
        for index, site in enumerate(sites)
        if site in buildable
    ]


def _tower_site_brief(turn: Turn, plan: LlmPlan) -> str:
    """本回合还要建的塔位坐标与对应武器（写进 prompt 的"最近可建位"）

    塔型由 `TOWER_LOADOUT` 与塔位下标绑定，这里把执行层真正要建的
    "坐标 -> 武器"直接摊给 LLM 看：建议里的布防方位因此有具体坐标可对，
    也让复盘里"LLM 说补建第 2 座高伤塔、实际建的却是另一种武器"这类
    "说的与做的对不上"不会再发生（塔型不受 LLM 文字左右，只会被照实告知）。

    已经建成的塔位不再出现在清单里（`_pending_tower_sites`）：复盘里 LLM
    照着一份陈旧的塔位清单反复建议"再建一座火箭炮于(29,9)"，而那一格上一个
    回合就已经建成了同款武器——塔位清单必须以执行层的当前状态为准。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        形如 "rocket(12,23)、railgun(10,22)" 的待建塔位清单；
        没有待建塔位时给出说明
    """
    pending = _pending_tower_sites(turn, plan)
    if not pending:
        return "暂无可用塔位"
    return "、".join(
        f"{TOWER_LOADOUT[index % len(TOWER_LOADOUT)]}({site.x},{site.y})"
        for index, site in pending
    )


def _tower_built_brief(turn: Turn) -> str:
    """已经建成的武器塔清单（写进 prompt，杜绝"重复建议已建成的塔位"）"""
    weapons = turn.weapons()
    if not weapons:
        return "暂无"
    return "、".join(
        f"{weapon.kind}({weapon.pos.x},{weapon.pos.y})" for weapon in weapons
    )


def _plan_summary(turn: Turn, plan: LlmPlan) -> str:
    """把本回合的既定计划写成一句人话

    计划出自 `_calc_tower_sites`/任务排序等同一套决策函数，LLM 因此可以对
    具体数字提意见，而不是和指令生成器各说各话。塔数与围墙段数报的是
    `_tower_target`/`_wall_target`（含金币闲置熔断、开局下限与防守方下限），
    塔位清单也来自同一套 `_calc_tower_sites`（且只列还没建成的塔位，
    见 `_pending_tower_sites`），布防方位由 `_defend_brief` 按同一套方位判定
    给出，所以 LLM 看到的就是执行层真正要建的座数/段数/坐标——复盘建议的
    "prompt 注入最近可建位坐标，消除'先移动、下回合再建'的一回合延迟"。
    """
    return (
        f"武器目标 {_tower_target(turn, plan)} 座（现有 {len(turn.weapons())} 座："
        f"{_tower_built_brief(turn)}）；"
        f"待建塔位 {_tower_site_brief(turn, plan)}；"
        f"优先铺围墙 {_wall_target(turn, plan)} 段（现有 {len(turn.walls())} 段）；"
        f"富余金币 {_gold_brief(plan)}；"
        f"布防方位 {_defend_brief(turn, plan)}"
    )


def _gold_brief(plan: LlmPlan) -> str:
    """富余金币的去处（写进 prompt 的金币计划，与执行层同一套优先级）

    复盘建议"提示词显式加'前期不存金币'规则"：金币优先变成武器升级券
    （100~150 金，攻击力与射程一起涨），武器满编满级之后买围墙升级券
    （20~30 金，围墙上限与血量一起涨，升级还会回满血）。
    这里把执行层真正会走的顺序摊给 LLM，"金币留着手不用"因而不再是一种建议。
    """
    if not plan.upgrade:
        return "今天不买券，金币留作他用"
    return (
        f"先武器升级券（{UPGRADE_GOLD}~{UPGRADE_GOLD2}金），"
        f"武器满级后围墙升级券（{WALL_UPGRADE_GOLD}~{WALL_UPGRADE_GOLD2}金），"
        "不存金币"
    )


def _wall_target(turn: Turn, plan: LlmPlan) -> int:
    """本回合要保证铺好的围墙段数（防守方有下限）

    正常情况下就是 LLM 计划里的 `wall`；但防守方在此基础上至少保证
    `DEFENDER_WALL_QUOTA` 段：防守方考的是扛住进攻，正面没有围墙时机器人会
    直接贴脸打基地（复盘里"防守方围墙 0 段、正面无任何阻挡、射程只覆盖基地
    贴脸区"）。进攻方不受影响，仍然完全按 LLM 计划走（0 段就是不铺）。

    配额管的是"今天要铺几段"这个目标：配额没铺满时工人优先采石、先施工，
    矿石不急着变现、金币也留到围墙立起来再花（见 `_worker_day_logic`）。
    地图上既没有石矿、背包里也没有存货时限额自动失效——没有石材就铺不出墙，
    再扣着经济线只会让矿石卖不掉、金币闲置。

    参数:
        turn: 当前回合信息
        plan: 本回合的LLM计划

    返回:
        要保证铺好的围墙段数（0..LLM_MAX_WALLS）
    """
    if turn.team_type == "defender":
        # 没有石材来源（地图上没石矿、背包里也没存货）时围墙根本无从铺起，
        # 这时不能因为"还差一段墙"把经济线扣住——矿石卖不掉就是金币闲置
        has_stone = bool(turn.stone_mines()) or any(
            WALL_MATERIAL in unit.backpack for unit in turn.workers()
        )
        if has_stone:
            return max(plan.wall, DEFENDER_WALL_QUOTA)
    return plan.wall


def _defend_brief(turn: Turn, plan: LlmPlan) -> str:
    """本回合实际优先布防的方位（与 `_side_priority` 用同一套判定）

    敌方来路已知时以它为准，计划里的 `defend` 只在看不到敌方单位时才生效，
    这样 prompt 报出去的方位就是执行层真正会用的顺序——复盘里"prompt 说布防
    up、塔却按别的方位排"这类建议与执行脱节不会再出现。
    """
    sides = [side for side in TOWER_SIDES if side in _enemy_sides(turn)]
    if sides:
        brief = "敌方来路 " + "/".join(sides)
        if plan.defend is not None and plan.defend not in sides:
            brief += f"（计划里的 {plan.defend} 不在来路上，不采用）"
        return brief
    if plan.defend is not None:
        return f"{plan.defend}（按计划，当前看不到敌方单位）"
    return "按地图内侧空间选择（当前看不到敌方单位）"


def _task_brief(turn: Turn) -> str:
    """可接任务点的坐标/奖励/剩余回合/距离，供LLM判断值不值得去做任务

    距离按"开拓者（没有开拓者时按基地）到任务点的棋盘距离"算：复盘里 LLM
    两次以"任务点距离远、风险未知"为由建议放弃任务，而两个任务点离我方基地
    只有 11~13 格（586377），把距离直接写进 prompt 就不会再凭感觉放弃
    两个任务点合计的 160 分+160 金币。
    """
    valid = [task for task in turn.player_tasks if task.is_valid]
    if not valid:
        return ""
    origin = _task_origin(turn)
    items = [
        f"({task.task_position.x},{task.task_position.y}){task.score_reward}分"
        + (f"/剩{task.timeout_rounds}回合" if task.timeout_rounds > 0 else "")
        + (f"/距我{distance(origin, task.task_position)}格" if origin else "")
        for task in valid[:2]
    ]
    return "：" + "；".join(items)


def _task_origin(turn: Turn) -> Pos | None:
    """算任务点距离的起点：开拓者（去领任务的就是它），没有开拓者时退回基地"""
    pioneers = turn.pioneers()
    if pioneers:
        return pioneers[0].pos
    station = turn.station()
    return station.pos if station else None


def _generate_strategy_prompt(
    turn: Turn,
    payload: dict[str, Any],
    plan: LlmPlan = LLM_PLAN_DEFAULT,
) -> str:
    """生成提交给LLM的策略咨询prompt（用满每个游戏日的调用额度）

    接口文档规定每个游戏日有LLM调用次数限制（errorCode=5：每队每个游戏日
    3 次，在当日首回合重置），因此只在每天的前 `LLM_PROMPT_PER_DAY` 个回合
    各请求一次，并带上上一回合的LLM回复作为上下文。复盘里"只在第 1 回合
    调用过 LLM，R2~R4 的 prompt 为空、指令退化为无目标移动"，只问一次
    等于把额度白白浪费掉，这里把当日额度用满。

    prompt 里一并给出"本回合既定计划"并要求最后回一行 `PLAN:`：建议因此
    能落到具体数字上，回复里的 PLAN 行由 `_llm_plan` 解析回指令生成器
    （复盘里的"策略与执行脱节"）。

    参数:
        turn: 当前回合信息
        payload: 判题系统原始请求（取 llmResp 作为上下文）
        plan: 上一回合LLM建议解析出的计划，用于说明当前策略

    返回:
        需要提交给LLM的prompt；本回合不需要咨询时返回空字符串
    """
    if not LLM_PROMPT_ENABLED:
        return ""
    # 当日前几个回合（第1、2、3回合与131、132、133回合...）
    if (turn.round_no - 1) % ROUNDS_PER_DAY >= LLM_PROMPT_PER_DAY:
        return ""

    robots = turn.alive_robots_targeting_me()
    weapons = turn.weapons()
    previous = str(payload.get("llmResp") or "").strip()

    lines = [
        "你是《未来战争》塔防对战的策略顾问。以下是当前局面，请给出本回合的作战建议。",
        f"回合 {turn.round_no}（第{(turn.round_no - 1) // ROUNDS_PER_DAY + 1}天白天）",
        f"阵营: {turn.team_type}，金币: {turn.gold}，积分: {turn.total_score}",
        f"我方武器: {len(weapons)}/3 座，围墙: {len(turn.walls())} 段",
        f"可控制角色: {len(turn.controllable())} 个",
        f"来袭机器人: {len(robots)} 个"
        f"（小型/中型/大型/BOSS尽量优先处理大型与BOSS）",
        f"可领取任务点: {sum(1 for t in turn.player_tasks if t.is_valid)} 个"
        f"{_task_brief(turn)}",
        f"本回合既定计划: {_plan_summary(turn, plan)}",
        "请用不超过5行中文说明：优先建造或升级什么、角色如何站位、富余金币怎么花。",
        # 任务不在可拨动的旋钮里：开拓者按"临期优先"自动去任务点领取并作答
        # （`_pioneer_day_logic`），prompt 因此不再问"是否值得做任务"——
        # 复盘里 LLM 反复建议"放弃任务"，客户端却照旧去领，建议与执行对不上
        # （586322/586377）。把权责讲清楚，建议才和指令对得上。
        "开拓者会自动前往任务点领取任务并作答，不需要建议放弃任务；",
        "只有最后一行 PLAN 的字段会改变客户端指令，其余文字建议仅供参考。",
        "最后一行必须输出作战计划（值越界会被忽略）："
        f"{LLM_PLAN_TEMPLATE}",
    ]
    if previous:
        lines.insert(1, f"上一回合LLM建议: {previous[:500]}")

    return "\n".join(lines)
