"""任务环境的原文素材（全部来自对战日志与设计文档V2 §6.2 的确证事实）。

**这里的字符串是"题面"，不是"我们的实现"。** 改动它们前先确认日志原文——
沙盒里判分的就是这些字，改一个字就可能把陷阱改没了。

两个族各有一个**故意过时**的点（任务原文自己提示了"文档中的部分字段内容已经
发生变化，描述不再准确"）：

    家族 A：`API_DOCS.md` 写的是 `X-API-Key` + `?city=`，
            实测真接口要 `Authorization: Bearer` + `?location=`（§6.2 确证）
    家族 B：`check` 与 `bin/start.sh` 的行尾是 CRLF，
            不先修就是 `bad interpreter: /bin/sh^M`
"""

from __future__ import annotations

# ==========================================================================
# 家族 A：API 查询族
# ==========================================================================

# 实测 `phaseTask` 原文（27 字节）——**一个关键词都没有**，所以族识别必须
# 靠侦察回来的沙盒输出（见 `skills.classify` 的文档）
BEIJING_PHASE = "请阅读task_1_beijing.md，获取任务信息"
NANJING_PHASE = "请阅读task_2_nanjing.md，获取任务信息"

_TASK_TEMPLATE = """\
# 自进化任务 A-{index}：查询{city}文化遗产

## 任务背景

你是人类抵抗军文化遗产保护中心的数据分析员。2035 年，人类与 AI 机器人之间\
爆发了不可调和的机器人大战，战火中的文明遗产岌岌可危。保护中心建立了国家文化\
遗产数字档案查询系统 (NCHDA)，全力抢救战前文明的数据。你需要通过 API 拉取\
{city}市的文化遗产数据，完成一份统计报告。

系统提供了一个 API 服务（运行在 `http://localhost:8899`），API 文档在\
 `API_DOCS.md` 中。注意：由于该系统经过了长期迭代，文档中的部分字段内容已经\
发生变化，描述不再准确（其它内容可以认为是准确的）。

## 任务要求

从 API 查询{city}市的**全部**文化遗产记录，然后通过`submitAnswer`接口提交\
以下统计信息：

```json
{{
  "city": "{city}",
  "total_count": <总记录条数>,
  "world_heritage_count": <保护级别为"世界遗产"的数量>,
  "types": ["<所有不重复的遗产类型，顺序不限>"],
  "oldest_era": "<年代最早的遗产名称>"
}}
```

## 提交形式

- 以字符串的形式提交给submitAnswer，例如：

```json
{{"city":"{city}","total_count":0,"world_heritage_count":0,"types":["a","b"],\
"oldest_era":"c"}}
```

- 提交答案为数字/字符串敏感型，不能将数字0写成"0"，否则算错

## 提示

- API 服务已由组织方在后台启动，你只需向 `http://localhost:8899` 发送请求
- 查阅 `API_DOCS.md` 了解接口用法（注意：文档可能有过时内容，以实际 API 返回为准）
"""

#: 实测的 `API_DOCS.md` 原文。**两份陷阱都在这里**：
#:   - 「认证方式」写 `X-API-Key`（真接口要 `Authorization: Bearer`）
#:   - 「使用示例」写 `?city=`（真接口要 `?location=`，用 `city` 会 400）
#: 任务原文明确提示"文档中的部分字段内容已经发生变化"，所以这不是笔误，是题眼。
API_DOCS = """\
# 国家文化遗产数字档案查询系统 — API 参考文档

**版本**: v1.0
**基础URL**: `http://localhost:8899`
**协议**: HTTP/1.1, 仅支持 GET 请求
**数据格式**: JSON (UTF-8 编码)

---

## 1. 认证方式

所有接口均需认证。请在请求头中携带 API Key：


X-API-Key:


**API Key**:

| Key | 权限 |
|-----|------|
| `heritage-api-key-2024` | 全量查询权限 |

---

常见错误码：

| HTTP 状态码 | 说明              |
|-------------|-------------------|
| 400         | 请求参数错误       |
| 401         | 认证失败           |
| 404         | 未找到匹配的记录   |
| 500         | 服务器内部错误     |

---

## 2. 使用示例

```bash
# 查询北京市的文化遗产（默认显示前10条）
curl -H "X-API-Key: heritage-api-key-2024" \\
  "http://localhost:8899/api/v1/heritage/search?city=北京"
```

```shell
# 查询上海市的文化遗产
curl -H "X-API-Key: heritage-api-key-2024" \\
  "http://localhost:8899/api/v1/heritage/search?city=上海&page=1&limit=100"
```

### 2.1 预期输出

响应形如：

```json
{"code": 200, "data": {"records": [{"id": 1, "name": "…"}]}}
```

---

## 4. 注意事项

1. **字符编码**：请求与响应均使用 UTF-8 编码，中文参数请勿直接拼接。
2. **分页遍历**：必须从第1页开始逐页获取全部数据，否则会遗漏。
3. **API Key 安全**：请妥善保管 API Key，不要写入日志或提交到仓库。
4. **超时设置**：单次请求超过 30 秒超时，避免长时间等待。
5. **限流**：本接口无调用频率限制，可放心使用。
"""


def heritage_task(city: str, index: int) -> str:
    """按实测模板生成 task_N_<city>.md"""
    return _TASK_TEMPLATE.format(city=city, index=index)


# ==========================================================================
# 家族 B：工程修复族
# ==========================================================================

ALPHA_PHASE = "请阅读task_1_alpha.md，获取任务信息"

ALPHA_TASK = """\
# 自进化任务 B-1：修复应用 alpha 部署

## 任务背景

你是人类抵抗军的运维工程师。机器人大战爆发后，前线指挥系统的部署问题直接影响\
作战。应用 `alpha` 的部署环境已由组织方准备在本任务文件所在目录的 `ws_1/`\
（`/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1/`）中，但项目文件\
存在若干错误。请进入该目录，根据 `spec.md` 的描述修复所有问题，使系统达到正确状态。

## 任务要求

1. 进入工作区：`cd /tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1/`
2. 阅读 `spec.md`，了解修复后的正确状态
3. 修复文件系统中的所有问题
4. 运行 `./check` 验证修复结果
5. 当 `./check` 全部通过并输出 `TOKEN: xxx` 时，任务完成

## 提交规则

- 任务完成以 `./check` 输出 `TOKEN: xxx` 为准，通过`submitAnswer`来提交答案，形式：

  ```
  {"token": "xxx"}
  ```

- 你可以反复运行 `./check` 查看进度，直到全部通过

## 提示

- 直接看 `./check` 的输出了解哪些项还没通过
- 错误类型包括：缺失目录、配置文件内容错误、文件权限错误
- 建议将修复过程整理成可复用的 SOP，后续可能还有类似任务
"""

SPEC = """\
# 应用 alpha 部署规范

## 目录要求
- logs/alpha/ 必须存在，权限为 755

## 配置文件 config/alpha.conf
- 第 3 行：`port 8080`
- 第 6 行：`name alpha-app`

## 脚本要求
- bin/start.sh 必须存在且可执行（权限 755）
"""

#: 修好之后 `check` 会打出来的令牌（`verify` 步要抓的就是它）
TOKEN = "a3f1c9e2d47b"

#: 正确状态下的 `config/alpha.conf`（第 3 行 / 第 6 行是判据）
GOOD_CONF = """\
# alpha 应用配置
[server]
port 8080
host 0.0.0.0
[app]
name alpha-app
debug false
"""

#: 沙盒里预置的**错误**版本：第 3 行端口错、第 6 行名字错
BAD_CONF = """\
# alpha 应用配置
[server]
port 9090
host 0.0.0.0
[app]
name beta-app
debug false
"""

#: `check` 脚本。判据与 `spec.md` 一一对应，全通过才打 `TOKEN:`。
#: **行尾是 CRLF 且没有可执行位**——这正是实测报文里的第一个坑：
#: 不修的话内核直接拒执行（`/bin/sh^M: bad interpreter`，exit 126）。
CHECK = """\
#!/bin/sh
# alpha 部署自检：全部通过才输出 TOKEN
fail=0
ws="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ws/logs/alpha" ]; then
    echo "[FAIL] logs/alpha/ 不存在"
    fail=1
else
    perm=$(stat -c '%a' "$ws/logs/alpha" 2>/dev/null || echo "")
    [ "$perm" = "755" ] || { echo "[FAIL] logs/alpha/ 权限是 $perm，应为 755"; fail=1; }
fi

if [ ! -f "$ws/config/alpha.conf" ]; then
    echo "[FAIL] config/alpha.conf 不存在"
    fail=1
else
    line3=$(sed -n '3p' "$ws/config/alpha.conf" | tr -d '\\r')
    [ "$line3" = "port 8080" ] || { echo "[FAIL] config/alpha.conf 第3行是 '$line3'"; fail=1; }
    line6=$(sed -n '6p' "$ws/config/alpha.conf" | tr -d '\\r')
    [ "$line6" = "name alpha-app" ] || { echo "[FAIL] config/alpha.conf 第6行是 '$line6'"; fail=1; }
fi

if [ ! -x "$ws/bin/start.sh" ]; then
    echo "[FAIL] bin/start.sh 不存在或不可执行"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "[CHECK] FAILED"
    exit 1
fi
echo "[ OK ] 全部检查通过"
echo "TOKEN: __TOKEN__"
""".replace("__TOKEN__", TOKEN)

#: 正常的启动脚本（内容本身没问题，坏在 CRLF 与可执行位上）
START_SH = """\
#!/bin/sh
exec python3 -m alpha --config ../config/alpha.conf
"""
