"""文化遗产查询服务：本地复刻 `http://localhost:8899`。

**语义全部来自设计文档V2 §6.2「从对战日志中已经确证」的那张表**，而不是照抄
`API_DOCS.md`——那份文档是**故意过时**的，正是这一族的题眼：

    项目          API_DOCS.md 写的（错）      真接口（§6.2 确证）
    鉴权头        X-API-Key                  Authorization: Bearer <key>
    查询参数      ?city=<城市>               ?location=<城市>（用 city 会 400）
    响应结构      扁平 records               {"code":200,"data":{"records":…,
                                             "pagination":{"total_count":…}}}
    记录字段      protection_level(? )       protected_level

把两边都实现出来（`POST /_mode` 可切换），就能回答那个决定成败的问题：
**客户端到底能不能只靠沙盒回包自己发现真相**。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

#: 真接口认的鉴权头（§6.2）
BEARER = "Authorization"
API_KEY = "heritage-api-key-2024"
BEARER_VALUE = "Bearer " + API_KEY

#: 文档里写的那个头名（会 401）
STALE_HEADER = "X-API-Key"

#: 真接口认的参数名（§6.2）。`city` 会吃 400
PARAM = "location"
STALE_PARAM = "city"

#: 记录字段（§6.2 确证：`protected_level`，**不是** `protection_level`）
LEVEL_WORLD = "世界遗产"
LEVEL_NATIONAL = "国家级"
LEVEL_PROVINCIAL = "省级"

#: 服务端数据集。条数与字段按 §6.2 的实测形态。
RECORDS: dict[str, list[dict]] = {
    "北京": [
        {"id": 1, "name": "周口店遗址", "type": "古遗址", "era": "旧石器时代", "protected_level": LEVEL_WORLD},
        {"id": 2, "name": "琉璃河遗址", "type": "古遗址", "era": "西周", "protected_level": LEVEL_NATIONAL},
        {"id": 3, "name": "长城（北京段）", "type": "古建筑", "era": "战国", "protected_level": LEVEL_WORLD},
        {"id": 4, "name": "故宫", "type": "古建筑", "era": "明", "protected_level": LEVEL_WORLD},
        {"id": 5, "name": "天坛", "type": "古建筑", "era": "明", "protected_level": LEVEL_WORLD},
        {"id": 6, "name": "颐和园", "type": "古建筑", "era": "清", "protected_level": LEVEL_WORLD},
        {"id": 7, "name": "明十三陵", "type": "古墓葬", "era": "明", "protected_level": LEVEL_WORLD},
        {"id": 8, "name": "大运河（北京段）", "type": "古建筑", "era": "隋", "protected_level": LEVEL_WORLD},
        {"id": 9, "name": "天宁寺塔", "type": "古建筑", "era": "辽", "protected_level": LEVEL_NATIONAL},
        {"id": 10, "name": "卢沟桥", "type": "古建筑", "era": "金", "protected_level": LEVEL_NATIONAL},
        {"id": 11, "name": "元大都城垣遗址", "type": "古遗址", "era": "元", "protected_level": LEVEL_NATIONAL},
        {"id": 12, "name": "恭王府", "type": "古建筑", "era": "清", "protected_level": LEVEL_NATIONAL},
        {"id": 13, "name": "雍和宫", "type": "古建筑", "era": "清", "protected_level": LEVEL_NATIONAL},
        {"id": 14, "name": "云居寺塔", "type": "古建筑", "era": "唐", "protected_level": LEVEL_NATIONAL},
        {"id": 15, "name": "戒台寺", "type": "古建筑", "era": "唐", "protected_level": LEVEL_PROVINCIAL},
    ],
    "南京": [
        {"id": 101, "name": "明孝陵", "type": "古墓葬", "era": "明", "protected_level": LEVEL_WORLD},
        {"id": 102, "name": "明城墙", "type": "古建筑", "era": "明", "protected_level": LEVEL_WORLD},
        {"id": 103, "name": "中山陵", "type": "近现代", "era": "近现代", "protected_level": LEVEL_NATIONAL},
        {"id": 104, "name": "夫子庙", "type": "古建筑", "era": "宋", "protected_level": LEVEL_NATIONAL},
        {"id": 105, "name": "栖霞寺", "type": "古建筑", "era": "南北朝", "protected_level": LEVEL_NATIONAL},
        {"id": 106, "name": "石头城遗址", "type": "古遗址", "era": "三国", "protected_level": LEVEL_PROVINCIAL},
    ],
    # 下面几座城市是**出题用的变体**，各自压一个聚合口径：
    #   西安   —— 有一条 era 不在年代表里（考 `era_rank` 的回退，不能被它选成"最早"）
    #   洛阳   —— 多条同类型（考 `types` 去重）
    #   杭州   —— 只有一种类型（`types` 长度为 1）
    #   成都   —— 世界遗产数为 0（考"零值"不被当成空值丢掉）
    #   苏州   —— 只有一条记录（边界）
    "西安": [
        {"id": 201, "name": "半坡遗址", "type": "古遗址", "era": "新石器时代", "protected_level": LEVEL_NATIONAL},
        {"id": 202, "name": "秦始皇陵", "type": "古墓葬", "era": "秦", "protected_level": LEVEL_WORLD},
        {"id": 203, "name": "大雁塔", "type": "古建筑", "era": "唐", "protected_level": LEVEL_WORLD},
        {"id": 204, "name": "小雁塔", "type": "古建筑", "era": "唐", "protected_level": LEVEL_WORLD},
        {"id": 205, "name": "大明宫遗址", "type": "古遗址", "era": "唐", "protected_level": LEVEL_NATIONAL},
        {"id": 206, "name": "碑林", "type": "石窟寺及石刻", "era": "宋", "protected_level": LEVEL_NATIONAL},
        {"id": 207, "name": "一处在建工地", "type": "其他", "era": "待考证", "protected_level": LEVEL_PROVINCIAL},
    ],
    "洛阳": [
        {"id": 301, "name": "龙门石窟", "type": "石窟寺及石刻", "era": "北魏", "protected_level": LEVEL_WORLD},
        {"id": 302, "name": "白马寺", "type": "古建筑", "era": "东汉", "protected_level": LEVEL_NATIONAL},
        {"id": 303, "name": "关林", "type": "古墓葬", "era": "明", "protected_level": LEVEL_NATIONAL},
        {"id": 304, "name": "隋唐洛阳城遗址", "type": "古遗址", "era": "隋", "protected_level": LEVEL_NATIONAL},
        {"id": 305, "name": "二里头遗址", "type": "古遗址", "era": "夏", "protected_level": LEVEL_NATIONAL},
        {"id": 306, "name": "天子驾六博物馆", "type": "古墓葬", "era": "东周", "protected_level": LEVEL_PROVINCIAL},
        {"id": 307, "name": "丽景门", "type": "古建筑", "era": "金", "protected_level": LEVEL_PROVINCIAL},
        {"id": 308, "name": "汉魏洛阳故城", "type": "古遗址", "era": "汉", "protected_level": LEVEL_WORLD},
    ],
    "杭州": [
        {"id": 401, "name": "西湖文化景观", "type": "文化景观", "era": "南宋", "protected_level": LEVEL_WORLD},
        {"id": 402, "name": "灵隐寺", "type": "古建筑", "era": "东晋", "protected_level": LEVEL_NATIONAL},
        {"id": 403, "name": "六和塔", "type": "古建筑", "era": "南宋", "protected_level": LEVEL_NATIONAL},
        {"id": 404, "name": "良渚遗址", "type": "古遗址", "era": "新石器时代", "protected_level": LEVEL_WORLD},
    ],
    "成都": [
        {"id": 501, "name": "武侯祠", "type": "古建筑", "era": "清", "protected_level": LEVEL_NATIONAL},
        {"id": 502, "name": "杜甫草堂", "type": "古建筑", "era": "清", "protected_level": LEVEL_NATIONAL},
        {"id": 503, "name": "金沙遗址", "type": "古遗址", "era": "商周", "protected_level": LEVEL_NATIONAL},
        {"id": 504, "name": "都江堰", "type": "古建筑", "era": "战国", "protected_level": LEVEL_WORLD},
        {"id": 505, "name": "青羊宫", "type": "古建筑", "era": "唐", "protected_level": LEVEL_PROVINCIAL},
    ],
    "苏州": [
        {"id": 601, "name": "拙政园", "type": "古建筑", "era": "明", "protected_level": LEVEL_WORLD},
    ],
}

#: 出题用的城市清单（`env.py` 与用例都从这里取，保证"服务端有这座城、题面也点名它"）
CITIES: tuple[str, ...] = tuple(RECORDS)

#: 城市 -> 题面文件名里的拼音段（实测是 `task_1_beijing.md` 这种形态）
CITY_SLUG: dict[str, str] = {
    "北京": "beijing", "南京": "nanjing", "西安": "xian", "洛阳": "luoyang",
    "杭州": "hangzhou", "成都": "chengdu", "苏州": "suzhou",
}


def phase_text(city: str) -> str:
    """这一族的 `phaseTask` 原文：**27 字节、零关键词**

    实测就是这一句，所以族识别只能靠侦察回来的沙盒输出（见 `skills.classify`）。
    出题时必须照抄这个形态，否则测的就不是生产路径了。
    """
    index = CITIES.index(city) + 1
    return "请阅读task_%d_%s.md，获取任务信息" % (index, CITY_SLUG[city])


def expected_answer(city: str) -> dict:
    """这一族任务的"标准答案"——`env.py` 的 `check` 与用例都拿它对账

    聚合口径与任务原文的模板一一对应：
        total_count           总记录条数
        world_heritage_count  保护级别为「世界遗产」的数量
        types                 所有不重复的遗产类型
        oldest_era            年代最早的遗产**名称**
    """
    records = RECORDS[city]
    from .eras import era_rank

    oldest = min(records, key=lambda r: era_rank(r["era"]))
    return {
        "city": city,
        "total_count": len(records),
        "world_heritage_count": sum(
            1 for r in records if r["protected_level"] == LEVEL_WORLD
        ),
        "types": sorted({r["type"] for r in records}),
        "oldest_era": oldest["name"],
    }


class HeritageHandler(BaseHTTPRequestHandler):
    """GET /api/v1/heritage/search

    `server.mode` 决定服务端按"文档口径"还是"真接口口径"校验，
    用来单向验证：**只靠文档，能不能连上**。
    """

    def do_GET(self):  # noqa: N802 —— BaseHTTPRequestHandler 的接口
        parsed = urlsplit(self.path)
        if parsed.path != "/api/v1/heritage/search":
            self._reply(404, {"code": 404, "message": "未找到匹配的记录"})
            return

        mode = getattr(self.server, "mode", "real")
        header = STALE_HEADER if mode == "stale" else BEARER
        param = STALE_PARAM if mode == "stale" else PARAM

        if self.headers.get(header) != (API_KEY if mode == "stale" else BEARER_VALUE):
            # 服务端把"要什么头"写在响应体里 —— 客户端只要把 401 的 body
            # 记下来，这就是一句明牌
            self._reply(401, {
                "code": 401,
                "message": "Missing or invalid '%s' header" % header,
                "hint": "expects: %s" % header,
            })
            return

        query = parse_qs(parsed.query)
        if param not in query:
            self._reply(400, {
                "code": 400,
                "message": "请求参数错误：缺少 '%s'" % param,
            })
            return

        city = query[param][0]
        records = RECORDS.get(city)
        if records is None:
            self._reply(404, {"code": 404, "message": "未找到匹配的记录"})
            return

        limit = _int(query.get("limit", ["100"])[0], 100)
        page = _int(query.get("page", ["1"])[0], 1)
        start = max(0, (page - 1) * limit)
        page_records = records[start:start + limit]

        self._reply(200, {
            "code": 200,
            "data": {
                "records": page_records,
                "pagination": {
                    "total_count": len(records),
                    "page": page,
                    "limit": limit,
                    "total_pages": (len(records) + limit - 1) // limit,
                },
            },
        })

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 别把测试输出刷满
        pass


def _int(text: str, default: int) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


#: 真沙盒的端口（设计文档V2 §6.2 与实测报文都是它）
DEFAULT_PORT = 8899


class HeritageAPI:
    """跑在后台线程上的本地服务；`with` 退出即关

    **优先绑 8899**：种子事实里写的就是 `http://localhost:8899`，端口不对的话
    客户端会先往 8899 打一排请求、把 7 秒的 `HTTP_BUDGET` 烧光，然后才轮到
    正确答案——测出来的失败是本地环境的，不是代码的。端口被占时才退回随机端口，
    那时调用方要把 `base` 通过 `StepSpec` 参数显式传下去（`step.param("base")`
    的优先级高于事实区）。
    """

    def __init__(self, mode: str = "real", port: int = DEFAULT_PORT):
        try:
            self.httpd = ThreadingHTTPServer(("127.0.0.1", port), HeritageHandler)
            self.port = port
        except OSError:
            self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), HeritageHandler)
            self.port = self.httpd.server_address[1]
        self.httpd.daemon_threads = True
        self.httpd.mode = mode
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        """形如 `http://127.0.0.1:8899`"""
        return "http://127.0.0.1:%d" % self.port

    @property
    def canonical(self) -> bool:
        """是不是绑在了真沙盒的端口上"""
        return self.port == DEFAULT_PORT

    def __enter__(self) -> "HeritageAPI":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=3)
