"""决策模块：游戏策略的核心实现。

对应设计文档 3.5 节。

策略概览：
    白天：工人优先建造武器工事（火箭发射台/电磁狙击炮/加特林，射程优先），
          再采集石头建造围墙（先封敌方来路那一侧）；
          围墙建完后用富余资源换取金币和武器升级（金币阶梯：
          武器升级券 > 围墙升级券，不让金币在手里睡着）。
          开拓者优先完成自进化类任务（任务点领取 + 沙盒作答，
          描述里没给文件名时先全盘探测沙盒，顺带把任务文件都读回来
          缓存备用，下一个任务点就能即时交卷），
          有任务在身时不退回基地，任务点冷却期间白天也守在下一个会开放的
          任务点旁等它开放（省掉"回基地再折返"的来回），天黑前再回防；
          天黑前工人回防到武器旁，但火力/围墙不达标时先抢建，
          角色不会整回合空转。
    夜晚：每个角色操控一座武器攻击机器人，优先攻击威胁最高的目标。

本模块为无状态决策：每回合从 `Turn` 重新解析地图与单位状态，
不依赖任何跨回合的战场状态，可自动适应矿区刷新、单位移动与视野变化。
任务答案同理，只认执行器按任务描述在沙盒里取到的数据（`[SOLUTION]` 段，
见 `_task_answer`），取不到数据就不提交；解析不到时再退到纯缓存
`_TASK_ANSWER_CACHE`（内容全部来自执行器的产出），未命中就走原来的
执行流程；LLM 建议也从请求里的 `llmResp` 现解析成有界计划（`_llm_plan`），
建议与指令出自同一套决策函数。
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
# 报文没给武器商店价格表时的商品兜底价（正式售价见任务书4.6.3）
ITEM_FALLBACK_PRICE = {
    name: price
    for vouchers in (WEAPON_UPGRADE_VOUCHERS, WALL_UPGRADE_VOUCHERS)
    for name, price in vouchers.values()
}
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
# 负责"矿石换金币"的工人的采集顺序：铁/铜是纯收入来源，石材只作兜底
ECONOMY_MINE_ORDER = (IRON_MINE, COPPER_MINE, STONE_MINE)

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
# 沙盒里搜任务文件时跳过的虚拟目录：进程/内核/设备文件系统里不会有任务文件，
# 却会让全盘 find 变慢并刷出一堆 Permission denied
TASK_FIND_PRUNE = ("/proc", "/sys", "/dev")
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
TASK_ANSWER_MIN_LEN = 4  # 答案最短长度（任务原文动辄几千字，这条挡住空答）
TASK_ECHO_RUN = r"[一-鿿]{6,}"  # 任务描述里的中文长句（复读判定用）
TASK_API_DEFAULT = "http://localhost:8899"  # 沙盒内的本地接口
TASK_API_TIMEOUT = 1  # 单次取数超时（秒），整条沙盒命令限时 15 秒
TASK_API_MAX_CALLS = 8  # 一条命令里最多请求几次（本地接口，失败也是立刻返回）
TASK_API_TIME_BUDGET = 8  # 取数阶段的时间上限（秒），留出找文件与回读的余量
TASK_EXEC_DIR_BUDGET = 4000  # 全盘找文件时最多进几个目录（防止 walk 慢过 15 秒）
TASK_API_QUERY_MAX = 2  # 每个接口地址最多试几个查询词
TASK_API_BODY_LIMIT = 400  # 单个响应体最多带回的字符数
TASK_SOLVE_MAX = 4  # 一次最多解几份任务文件（当前这份排第一）
TASK_API_PATH_SUFFIXES = ("/", "/api", "/docs")  # 文档没给样例时先试这几个
TASK_API_DOC_NAMES = (r"api", r"doc", r"readme", r"\.md$")  # 接口文档的文件名特征
TASK_EXEC_PRUNE = ("/proc", "/sys", "/dev", "/run")  # 全盘找文件时跳过的虚拟目录

# 任务答案缓存：任务文件名 -> 沙盒执行产出的答案（`[SOLUTION]` 段的内容）
# 任务书5.3节要求"根据任务1探索的内容形成固定SOP或者SKILL，实现Agent自进化"，
# 积分又是"任务奖励 + 5 × 标准回合数 / (完成回合 - 接取回合)"（任务书第六章），
# 交得越早分越高。执行器一次会把沙盒里的任务文件都试着解一遍，解出来的答案
# 存进这里，后续任务点领到同一份任务时，开拓者不必再等一个来回的沙盒输出，
# 接取后下一回合就能直接作答（复盘里敌方就是靠答案缓存秒交，两次提交各拿 155 分）。
# 这是纯缓存：没有命中的任务仍然走"下发沙盒命令 -> 下一回合读输出"的原路径。
_TASK_ANSWER_CACHE: dict[str, str] = {}

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
    # 上回合执行器解出来的任务答案按文件名缓存，后续任务一到手就能直接作答
    _remember_task_answers(turn.last_cmd_result)
    # 上一回合的LLM建议解析成有界计划，和指令生成器共用（解析不出来时是默认计划）
    plan = _llm_plan(payload)
    commands: dict[int, dict[str, Any]] = {}

    if turn.is_day:
        _decide_day(turn, commands, plan)
    else:
        _decide_night(turn, commands)

    prompt = _generate_strategy_prompt(turn, payload, plan)

    # 转换key为字符串
    return {str(key): value for key, value in commands.items()}, prompt


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
    """兜底：没有任何目标的角色就近采一铲矿

    决策的各条分支都会尽量给角色安排动作，但任务点够不到、武器还没建成、
    地图上一座矿都采不了时，角色会整回合没有任何指令（复盘里的"角色原地
    挪位、金币连续多回合冻结"）。这里做最后一道兜底——按 石→铁→铜 就近
    采集，采不到就不下指令，交给下一回合重新判断。

    不打扰的情况:
        - 本回合已经有指令的角色（决策层已经给了更优先的动作）
        - 任务进行中的开拓者：任务要求它留在任务点周围一格内，
          任何移动都可能让任务强制结束
        - 有任务可领的开拓者：接任务、交任务是主要得分来源，被兜底支去采矿
          等于把开拓者从任务点上拽走（它的任务优先级最高，见 `_pioneer_day_logic`）
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
    if unit.kind == PIONEER and (
        turn.phase_task or any(task.is_valid for task in turn.player_tasks)
    ):
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
            # 手里还没有可卖的矿石: 继续采集,攒够一批再换金币
            _gather_logic(turn, worker, claimed, commands)
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
    if _mine_order(turn, worker, prefer_stone=wall_quota)[0] == STONE_MINE:
        mine = _adjacent_mine(turn, worker, STONE_MINE)
        if mine is not None and stones < _stone_reserve(turn, plan, wall_quota):
            commands[worker.unit_id] = collect_command(mine)
            claimed.add(mine)
            return

    # 如果有石头,去建造围墙（认领成功就记进 claimed：几名工人自然分头铺不同
    # 段，而不是都奔向优先级最高的那一段，白走一趟还互相挡路；位置都被其他
    # 角色占住时继续往下走,别空转）
    if stones >= WALL_STONE_COST:
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

    # 还是没着落: 就近采矿; 采不到就把背包里的矿石卖掉腾地方
    if _gather_logic(turn, worker, claimed, commands, prefer_stone=wall_quota):
        return
    _trade_logic(turn, worker, claimed, commands)


def _gold_critical(turn: Turn) -> bool:
    """金币是否已经见底（不足一座塔的造价）

    见底时经济回路成为第一优先级：建造、升级、买券都要求手里有金币，而金币
    只能靠卖矿换（任务书4.6.1）。复盘里 R6 建完第三座塔后 gold=0 连续冻结
    13 个回合、背包里的矿石一直没卖出去，全盘停摆——这时矿石留在背包里
    没有任何价值，先变现才有翻盘的可能。
    """
    return turn.gold < LOW_GOLD_THRESHOLD


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
) -> tuple[str, ...]:
    """该工人本回合的采集矿种顺序

    "矿种互补"只在还有另一名工人兜底采石材时成立：同一时刻最多一名工人去
    采铁/铜换金币，其余人继续采石材保证围墙不停工；只剩一名工人时它必须
    石材优先（围墙是防守的根本），所以退回默认顺序。

    参数:
        turn: 当前回合信息
        worker: 待判断的工人
        prefer_stone: 为 True 时全员石材优先（围墙配额还没铺满时用）

    返回:
        按优先级排序的矿种元组
    """
    if (
        prefer_stone
        or len(turn.workers()) < 2
        or not _is_economy_worker(turn, worker)
    ):
        return SELLABLE_MINES
    return ECONOMY_MINE_ORDER


def _pioneer_day_logic(
    turn: Turn,
    pioneer: Unit,
    tower_sites: tuple[Pos, ...],
    wall_order: tuple[Pos, ...],
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> None:
    """开拓者白天逻辑

    优先级: 维持进行中的任务（含提交答案，命中答案缓存时接取后即交卷）
            > 前往任务点领取任务（本回合走不动就原地等，不退回去跟随武器塔）
            > 守候正在冷却的任务点（白天守在下一个会开放的任务点旁，天黑前回防）
            > 跟随武器塔（只在没有任何任务点时才做）

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

    # 2. 有可接取的任务: 按优先级依次尝试领取
    # 之前的实现只试最优先的那个任务点：那一格被挡住/绕不过去时开拓者就整回合
    # 放弃任务、跑去跟随武器塔，两个任务点（合计160分+160金币）都会白白过期。
    valid_tasks = [task for task in turn.player_tasks if task.is_valid]
    if valid_tasks:
        ordered = sorted(valid_tasks, key=lambda task: (
            task.timeout_rounds if task.timeout_rounds > 0 else TASK_TIMEOUT_UNKNOWN,
            distance(pioneer.pos, task.task_position),
            task.task_position.x,
            task.task_position.y,
        ))
        for task in ordered:
            if _head_to_task(turn, pioneer, task.task_position, claimed, commands):
                return
        # 这一步走不动时留在原地等下一个回合（同伴让开、路就通了），
        # 而不是掉头回基地跟随武器塔——那等于往相反方向走，下一回合再往外走，
        # 来回打转永远到不了任务点（复盘里的"开拓者整局在基地附近徘徊、
        # 从未靠近任务点"，两处任务点合计160分+160金币一直没人领）
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

    # 4. 没有可接取的任务: 跟随武器塔,为夜晚操控武器做准备
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
    # 背包满了就卖一批腾地方,否则等攒够一批再卖；金币见底时有几块卖几块
    batch = 1 if (worker.backpack_full or _gold_critical(turn)) else SELL_BATCH
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
        price: 售价（来自 weaponShopList，取不到时用兜底价）
        allow: 是否允许本回合花钱买
        reserve: 买之前要留下的金币（给更高优先级的支出留钱）

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
        commands[worker.unit_id] = buy_command(item)
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

    参数:
        turn: 当前回合信息
        unit: 待召回的角色
        claimed: 已被其他角色占用的目标集合
        commands: 指令输出字典（角色ID -> 指令）
    """
    weapons = turn.weapons()
    if not weapons:
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


# === 夜晚决策 ===


def _decide_night(turn: Turn, commands: dict[int, dict[str, Any]]) -> None:
    """夜晚策略: 操控武器攻击"""
    claimed: set[Pos] = set()

    # 为每个武器配对一个操控角色
    for controller, weapon in _pair_controllers_and_weapons(turn):
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

        # 操控角色不在武器旁边,向武器移动
        step = _step_toward(turn, controller, weapon.pos, claimed)
        if step is not None:
            commands[controller.unit_id] = move_command(step)


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
    读文件流程。

    答案区之后依次是工作目录诊断与任务文件回读（`[TASK_FILE]` 分段，供
    `_task_file` 认出沙盒里的真实文件名、给执行器圈定候选任务文件）。
    这两段都排在 `TASK_END_MARKER` 之后，永远不会被当成答案。
    """
    if (
        not turn.phase_task
        or _task_answer(turn) is not None
        or _cached_answer(turn) is not None
    ):
        return ""

    # 描述里没给文件名时，用上一回合的探测结果找；还没探过就先探一次
    target = _task_file(turn.phase_task) or _task_file(turn.last_cmd_result)
    if target is None:
        return _sandbox_probe(turn)

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    scan = "pwd; ls -a -- . 2>&1 | head -40"
    # 执行器片段以 heredoc 结束符收尾，换行后再接诊断与回读；
    # 末尾的 `:` 保证整条命令的退出码为 0：输出带 `[exitCode:0]` 才会被
    # `_task_answer` 采纳，而执行器里的取数失败时退出码可能是非 0 的
    return (
        f'echo "{marker}"; {_task_executor(target)}'
        f'\necho "{TASK_END_MARKER}"; {scan};'
        f' {_task_dump(turn)}; :'
    )


def _task_executor(task_path: str) -> str:
    """在沙盒里执行自进化任务的命令片段（读任务文件 -> 调接口取数 -> 打答案）

    沙盒里只有基础 shell 与 python，命令一回合只能下一发、限时 15 秒，
    所以取数脚本一次跑完：找任务文件与接口文档、按文档里的样例地址调用
    本地接口、把响应体打成 `[SOLUTION]` 段。

    参数:
        task_path: 任务描述里点名的任务文件（沙盒路径或文件名）

    返回:
        可直接拼进沙盒命令的 shell 片段
    """
    script = (
        TASK_EXECUTOR
        .replace("__TASK_PATH__", repr(task_path))
        .replace("__BASE__", repr(TASK_API_DEFAULT))
        .replace("__TIMEOUT__", str(TASK_API_TIMEOUT))
        .replace("__MAX_CALLS__", str(TASK_API_MAX_CALLS))
        .replace("__TIME_BUDGET__", str(TASK_API_TIME_BUDGET))
        .replace("__DIR_BUDGET__", str(TASK_EXEC_DIR_BUDGET))
        .replace("__QUERY_MAX__", str(TASK_API_QUERY_MAX))
        .replace("__BODY_LIMIT__", str(TASK_API_BODY_LIMIT))
        .replace("__SOLVE_MAX__", str(TASK_SOLVE_MAX))
        .replace("__DOC_NAMES__", repr(TASK_API_DOC_NAMES))
        .replace("__SUFFIXES__", repr(TASK_API_PATH_SUFFIXES))
        .replace("__PRUNE__", repr(TASK_EXEC_PRUNE))
        .replace("__SOLUTION__", repr(TASK_SOLUTION_MARKER))
        .replace("__SOLUTION_END__", repr(TASK_SOLUTION_END))
        .replace("__DATA__", repr(TASK_DATA_MARKER))
    )
    # 沙盒的解释器叫 python3 或 python，挑一个能用的（挑不到时脚本不会执行，
    # 答案区为空 -> 这一回合不提交，下一回合重来）。
    # 片段以 heredoc 结束符收尾且不带换行：调用方必须换行后再接别的命令
    # （结束符要独占一行，直接接 `;` 会让后一条命令变成脚本的一部分）
    return (
        'for P in python3 python; do command -v "$P" >/dev/null 2>&1 && break;'
        f" done; $P - <<'PYEOF' 2>/dev/null\n{script}\nPYEOF"
    )


# 沙盒执行器：占位符由 `_task_executor` 按当前任务填好。
# 之所以要"执行"而不是"读文件"，是因为任务文件里写的是任务要求（"查询北京
# 文化遗产"），答案在文档给出的本地接口里；直接把任务文件的内容交上去
# 等于答非所问（复盘里就是这么丢掉全部任务分的）。
TASK_EXECUTOR = '''\
import os
import re
import time
import urllib.request

TASK_PATH = __TASK_PATH__
BASE = __BASE__
TIMEOUT = __TIMEOUT__
MAX_CALLS = __MAX_CALLS__
TIME_BUDGET = __TIME_BUDGET__
DIR_BUDGET = __DIR_BUDGET__
QUERY_MAX = __QUERY_MAX__
BODY_LIMIT = __BODY_LIMIT__
SOLVE_MAX = __SOLVE_MAX__
DOC_NAMES = __DOC_NAMES__
SUFFIXES = __SUFFIXES__
PRUNE = __PRUNE__
SOLUTION = __SOLUTION__
SOLUTION_END = __SOLUTION_END__
DATA = __DATA__
SKIP_WORDS = ("http", "https", "localhost", "task", "spec", "md", "txt", "json", "api")


def read(path):
    """读文件，读不到就返回空串（沙盒里权限与路径都不可控）"""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def fetch(url):
    """调用接口并把响应体截断返回（失败返回空串，绝不抛异常打断整条命令）"""
    try:
        request = urllib.request.Request(url, headers={"Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read().decode("utf-8", "replace").strip()[:BODY_LIMIT]
    except Exception:
        return ""


def find_files(patterns, limit):
    """按文件名特征在沙盒里找文件（任务文件与接口文档都在沙盒深处）

    全盘 walk 是这里最慢的一步，所以两个上限都要兜住：找到够数就停，
    进的目录太多也停（沙盒命令整体限时 15 秒，宁可少找几个也不能超时）。
    """
    found = []
    seen = set()
    visited = 0
    for root, dirs, files in os.walk("/"):
        visited += 1
        if visited > DIR_BUDGET:
            break
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in PRUNE]
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


def task_files():
    """待解的任务文件：描述里点名的那份排第一（答案只认它）"""
    wanted = os.path.basename(TASK_PATH) if TASK_PATH else ""
    named = []
    others = []
    for path in find_files((r"^(task|spec).*\\.(md|txt|json)$",), 24):
        if wanted and os.path.basename(path) == wanted:
            named.append(path)
        else:
            others.append(path)
    if TASK_PATH and os.path.isfile(TASK_PATH) and TASK_PATH not in named:
        named.insert(0, TASK_PATH)
    return named + others


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
        raw = raw.strip().strip("\\"'").rstrip(".,;:!?、。）])")
        if raw and raw not in urls:
            urls.append(raw)
    local = [url for url in urls if "localhost" in url or "127.0.0.1" in url]
    picked = local or [BASE] + [url for url in urls if url != BASE]
    return picked


def queries(text, name):
    """查询关键词：文件名里的英文词最可靠（task_1_beijing.md -> beijing）"""
    values = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,30}", name or ""):
        if token.lower() not in SKIP_WORDS:
            values.append(token)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,30}", text):
        if token.lower() not in SKIP_WORDS:
            values.append(token)
    picked = []
    for value in values:
        if value not in picked:
            picked.append(value)
    return picked[:QUERY_MAX]


def candidates(text, name, urls):
    """这一份任务要试的调用地址：文档样例 + 把样例里的查询词换成任务自己的"""
    urls = urls or [BASE]
    out = []
    for url in urls[:2]:
        if url not in out:
            out.append(url)
        for value in queries(text, name):
            variant = (
                re.sub(r"=([^&/]+)", "=" + value, url, count=1)
                if "=" in url
                else url.rstrip("/") + "/" + value
            )
            if variant not in out:
                out.append(variant)
    for suffix in SUFFIXES:
        variant = BASE.rstrip("/") + suffix
        if variant not in out:
            out.append(variant)
    return out


doc_text = "\\n".join(read(path) for path in find_files(DOC_NAMES, 6))
urls = endpoints(doc_text)

deadline = time.time() + TIME_BUDGET
calls = 0
solutions = []
for path in task_files()[:SOLVE_MAX]:
    name = os.path.basename(path)
    text = read(path)
    bodies = []
    for url in candidates(text, name, urls):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        calls += 1
        body = fetch(url)
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
) -> str:
    """沙盒里按文件名全盘查找的 find 片段

    复盘里沙盒的工作目录就是 `/`，`ls -a -- .` 只有 bin/dev/etc/home/lib/
    lib64/proc/sbin/tmp/usr，而任务文件并不在 `/` 的前三层里：旧实现把搜索
    限定在 `find . -maxdepth 3` 加两个猜出来的目录（`/tmp/selfEvolutionTask`、
    `/tmp/selfEvolution`），任务文件一次都没被找到，开拓者整个任务周期
    卡在任务点。这里改成从根目录起全盘按文件名找，只跳过 `TASK_FIND_PRUNE`
    里的虚拟目录，读不到文件的目录由 `2>/dev/null` 静音。

    命中还要过一道扩展名闸门（`exts`）：只有"名字像任务文件、且扩展名是
    文档"的才算任务文件，`task.xsl` 这类同名样式表被挡在外面（见
    `TASK_FILE_EXTS`；三场复盘里回读回来的正是它）。

    参数:
        names: 文件名通配（如 `task*`），多个通配之间是"或"关系
        action: 命中后执行的动作（`TASK_FIND_PRINT` 只列路径，内容由调用方按需读取）
        exts: 扩展名白名单，命中文件必须以后缀之一结尾

    返回:
        可直接拼进沙盒命令的 find 片段
    """
    prune = " -o ".join(f'-path "{path}"' for path in TASK_FIND_PRUNE)
    wanted = " -o ".join(f'-name "{name}"' for name in names)
    docs = " -o ".join(f'-name "*{ext}"' for ext in exts)
    return (
        f'find / \\( {prune} \\) -prune -o -type f \\( {wanted} \\)'
        f" -a \\( {docs} \\) {action} 2>/dev/null"
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
        f'for f in $({_sandbox_find(names, TASK_FIND_PRINT)}'
        f" | head -{TASK_FILE_MAX});"
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
        f"{_sandbox_find(TASK_PROBE_NAMES, TASK_FIND_PRINT)} | head -40; "
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

    两道闸门保证交上去的不是任务原文（复盘里 4 次 submitAnswer 交的全是
    任务描述，Judge 一次都没放行）：
        1. 答案区里必须出现过真实取数的证据（`TASK_DATA_MARKER`）——
           执行器取不到数据时答案区是空的，这一回合就不提交；
        2. 答案里不能出现任务描述里的中文长句（`_task_echo`）。
    命中错误特征的输出（文件不存在等）同样不能提交：错误答案既拿不到分，
    又白白消耗任务冷却，所以宁可这一回合不提交，等下一条沙盒输出。
    """
    if not turn.phase_task:
        return None

    marker = f"{TASK_MARKER}{_task_token(turn.phase_task)}"
    result = turn.last_cmd_result
    if marker not in result or "[exitCode:0]" not in result:
        return None

    region = result.split(marker, 1)[1].split(TASK_END_MARKER, 1)[0]
    if TASK_DATA_MARKER not in region:
        return None  # 没取到数据：沙盒里只有任务原文，不能当答案交上去
    if any(bad in region for bad in TASK_ERROR_MARKERS):
        return None

    answer = _solution_answer(region, turn)
    if answer is None or len(answer) < TASK_ANSWER_MIN_LEN:
        return None
    if _task_echo(answer, turn.phase_task):
        return None
    return answer


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

    参数:
        result: 报文的 `lastCmdResult`（上回合沙盒命令的输出）
    """
    for chunk in result.split(TASK_SOLUTION_MARKER)[1:]:
        path, _, body = chunk.partition("\n")
        answer = body.split(TASK_SOLUTION_END, 1)[0].strip()
        name = path.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if name and answer:
            _TASK_ANSWER_CACHE.setdefault(name, answer)


def _cached_answer(turn: Turn) -> str | None:
    """当前任务在答案缓存里的答案（沙盒里之前解出来的同名任务）

    积分 = 任务奖励 + 5 × 标准回合数 / (完成回合 - 接取回合)（任务书第六章），
    完成回合差越小分越高。缓存命中时开拓者在任务进行中的第一个回合就能交卷，
    把回合差压到 1（复盘里敌方两次 cached submit 各得 155 分，我们则要重新
    执行一遍沙盒、回合差至少 2）。

    任务描述里没点名文件时返回 None：探测出来的文件名与任务描述的对应关系
    不确定，宁可多花一个来回执行一次，也不拿别的任务的答案去作答。
    """
    target = _task_file(turn.phase_task)
    if target is None:
        return None
    return _TASK_ANSWER_CACHE.get(target.replace("\\", "/").rsplit("/", 1)[-1])


def _go_mine(
    turn: Turn,
    unit: Unit,
    mine_type: str,
    claimed: set[Pos],
    commands: dict[int, dict[str, Any]],
) -> bool:
    """让单位去采矿"""
    if unit.backpack_full:
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

    返回:
        True 表示本回合已下达采集或移动指令
    """
    for mine_type in _mine_order(turn, worker, prefer_stone=prefer_stone):
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


def _calc_wall_order(turn: Turn) -> tuple[Pos, ...]:
    """计算围墙建造顺序（基地周围第二圈）

    按 `_wall_side_order` 给出的方位顺序（先敌方来路、其余上左下右）环绕基地
    铺一圈围墙，并在右下角留一个入口供角色进出。超出地图或落在非陆地上的点
    会被过滤掉。
    """
    station = turn.station()
    if station is None:
        return ()

    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # 矩形四条边各自的格子（同一格不会落在两条边上）
    by_side: dict[str, list[Pos]] = {
        # 上边（从右到左）
        "up": [Pos(x, ymax + 2) for x in range(xmax + 2, xmin - 3, -1)],
        # 左边（从上到下）
        "left": [Pos(xmin - 2, y) for y in range(ymax + 1, ymin - 2, -1)],
        # 下边（从左到右）
        "down": [Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)],
        # 右边（从下到上）
        "right": [Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)],
    }
    order = [
        pos
        for side in _wall_side_order(turn)
        for pos in by_side[side]
    ]

    # 留一个入口（右下角）
    entrance = Pos(xmax + 2, ymin - 1)

    return tuple(
        pos for pos in order
        if pos != entrance
        and turn.land(pos)
        and 0 <= pos.x < turn.width
        and 0 <= pos.y < turn.height
    )


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
