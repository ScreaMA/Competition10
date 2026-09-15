"""年代排序表：与 `agent/strategy/task/scripts.py` 里的 `BUILTIN_ERA` 必须一致。

那份表写在注入沙盒的脚本模板里（是一个 raw string 的内部，导不出来），所以这里
必须抄一份。**抄的东西会漂**，所以 `tests/test_tasksandbox.py` 里有一条用例把
两边的表逐项比对——漂了就红。

任务原文对 `oldest_era` 的要求是"年代最早的遗产**名称**"。日志里明确记着
"比较 era 时按此历史顺序，不要按字符串字典序"——所以这张表是判分口径的一部分。
"""

from __future__ import annotations

#: 年代 -> 排序权重（越小越早）
ERA_ORDER: dict[str, int] = {
    "旧石器时代": 0, "新石器时代": 1, "夏": 2, "商": 3, "商周": 3, "周": 4,
    "西周": 4, "东周": 4, "春秋": 5, "战国": 6, "秦": 7, "汉": 8, "西汉": 8,
    "东汉": 8, "三国": 10, "魏晋": 11, "晋": 11, "南北朝": 13, "隋": 14,
    "唐": 15, "五代": 16, "宋": 17, "北宋": 17, "南宋": 17, "辽": 18, "金": 18,
    "元": 19, "明": 20, "清": 21, "近现代": 22, "近代": 22, "现代": 22,
}


def era_rank(text: str) -> int:
    """年代的排序权重（未知年代排最后；退化时取串里的第一个数字）

    与沙盒脚本里的 `era_rank` 同口径：先查表，再按"表里的键是不是文本的子串"
    匹配（"明" 命中 "明代"），最后才退回数字解析。
    """
    text = str(text or "")
    if text in ERA_ORDER:
        return ERA_ORDER[text]
    for key, rank in ERA_ORDER.items():
        if key and key in text:
            return rank
    digits = "".join(c for c in text if c.isdigit())
    if digits:
        import re

        match = re.search(r"-?\d+", text)
        if match:
            return int(match.group(0))
    return 9999
