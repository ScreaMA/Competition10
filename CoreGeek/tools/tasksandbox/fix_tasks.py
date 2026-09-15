"""工程修复族的**题型模板**：给定一份"变体"，生成整套题面。

一道题由四样东西界定：

    spec.md      修复清单（"logs/alpha/ 必须存在" / "第 3 行：port 8080"）
    check        自检脚本，判据与 spec 一一对应，全通过才吐 TOKEN
    ws_1/        预置了错误的工作区
    task_1_*.md  题面（指路 + 提交形式）

**预置错误与判据都由变体声明式地生成**，不是手抄的——手抄的话改一处 spec
忘了改对应 check，题目就自相矛盾了（那会让"做不完"看起来像客户端的锅）。

实测的那道题（`alpha`）用 `KNOWN_VARIANT` 精确保留：spec/check/工作区的形状
与日志原文一致。其余变体只换"哪里坏了、坏成什么样"。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

#: 修好之后 `check` 吐出来的令牌
DEFAULT_TOKEN = "a3f1c9e2d47b"


@dataclass(frozen=True)
class FixVariant:
    """一道工程修复题"""

    code: str                      # 代号：任务文件 task_1_<code>.md
    app: str                       # 应用名（出现在题面与 spec 里）
    conf: str = "config/alpha.conf"        # 配置文件相对路径
    conf_lines: tuple[tuple[int, str], ...] = ()   # 判据行：行号(1起) -> 正确内容
    dirs: tuple[str, ...] = ()     # 必须存在的目录
    script: str = "bin/start.sh"   # 必须存在且可执行的脚本
    #: 脚本坏在哪：crlf（默认，实测形态）/ missing（要新建）/ perm（只是没权限）
    script_broken: str = "crlf"
    #: 哪些目录预置成缺失（默认全都缺）
    dirs_missing: tuple[str, ...] | None = None
    token: str = DEFAULT_TOKEN

    @property
    def missing_dirs(self) -> tuple[str, ...]:
        return self.dirs if self.dirs_missing is None else self.dirs_missing


#: 实测的那道题（对战日志里的 alpha），逐字保留
KNOWN_VARIANT = FixVariant(
    code="alpha",
    app="alpha",
    conf="config/alpha.conf",
    conf_lines=((3, "port 8080"), (6, "name alpha-app")),
    dirs=("logs/alpha",),
    script="bin/start.sh",
)

#: 另外 5 道同型题——换应用名、换坏法、换判据行号与目录层级
NEW_VARIANTS: tuple[FixVariant, ...] = (
    # 入口脚本**不存在**（要照着 spec 建出来），目录只缺一层
    FixVariant(
        code="beta", app="beta",
        conf="config/beta.conf",
        conf_lines=((2, "port 9090"), (5, "name beta-svc")),
        dirs=("var/beta",),
        script="bin/entry.sh", script_broken="missing",
    ),
    # 脚本存在、内容也对，只是没有可执行位（纯权限题）
    FixVariant(
        code="gamma", app="gamma",
        conf="config/gamma.conf",
        conf_lines=((3, "mode strict"), (7, "name gamma-app")),
        dirs=("logs/gamma", "spool/gamma"),
        script="bin/start.sh", script_broken="perm",
        dirs_missing=("spool/gamma",),
    ),
    # 目录是**两级**的（`makedirs` 而非 `mkdir`）
    FixVariant(
        code="delta", app="delta",
        conf="etc/delta.ini",
        conf_lines=((4, "port 7070"),),
        dirs=("data/delta/cache",),
        script="bin/run.sh", script_broken="crlf",
    ),
    # 只有一个判据行，且改的是字符串值
    FixVariant(
        code="epsilon", app="epsilon",
        conf="config/epsilon.conf",
        conf_lines=((6, "name epsilon-node"),),
        dirs=("logs/epsilon",),
        script="bin/start.sh", script_broken="missing",
    ),
    # 两个目录、两个判据行、脚本只是没权限——全都要照顾到
    FixVariant(
        code="zeta", app="zeta",
        conf="config/zeta.conf",
        conf_lines=((3, "port 6060"), (6, "name zeta-app")),
        dirs=("logs/zeta", "state/zeta"),
        script="bin/start.sh", script_broken="perm",
    ),
)


# --- 生成 -----------------------------------------------------------------


def phase_of(variant: FixVariant) -> str:
    """这一族任务的 `phaseTask` 原文（同样是 27 字节、零关键词）"""
    return "请阅读task_1_%s.md，获取任务信息" % variant.code


def _wrong(line: str) -> str:
    """把一个正确值改坏（数字 +1，其余加后缀），保证与正确值不同"""
    parts = line.split()
    if len(parts) < 2:
        return line + "-bad"
    tail = parts[-1]
    parts[-1] = str(int(tail) + 1) if tail.isdigit() else tail + "-bad"
    return " ".join(parts)


def conf_text(variant: FixVariant, correct: bool) -> str:
    """配置文件正文：第 N 行按判据给，其余填无关内容

    填充行刻意不含判据值，否则"没改也算对"——题目就废了。
    """
    rows = [
        "# %s 应用配置" % variant.app,
        "[server]",
        "port 0",
        "host 0.0.0.0",
        "[app]",
        "name placeholder",
        "debug false",
        "threads 4",
    ]
    while len(rows) < max((n for n, _ in variant.conf_lines), default=0):
        rows.append("# filler")
    for number, want in variant.conf_lines:
        rows[number - 1] = want if correct else _wrong(want)
    return "\n".join(rows) + "\n"


def spec_text(variant: FixVariant) -> str:
    rows = ["# 应用 %s 部署规范" % variant.app, ""]
    if variant.dirs:
        rows.append("## 目录要求")
        for folder in variant.dirs:
            rows.append("- %s/ 必须存在，权限为 755" % folder)
        rows.append("")
    if variant.conf_lines:
        rows.append("## 配置文件 %s" % variant.conf)
        for number, want in variant.conf_lines:
            rows.append("- 第 %d 行：`%s`" % (number, want))
        rows.append("")
    if variant.script:
        rows.append("## 脚本要求")
        rows.append("- %s 必须存在且可执行（权限 755）" % variant.script)
        rows.append("")
    return "\n".join(rows)


def check_text(variant: FixVariant) -> str:
    """自检脚本：判据与 `spec_text` 一一对应，全通过才吐 TOKEN

    行尾是 CRLF（实测形态）——不先修就是 `bad interpreter: /bin/sh^M`。
    权限那几项在 `stat` 不可用时**跳过而不是判失败**，否则 Windows 上的
    Git Bash 会把"判不了"当成"没修好"。
    """
    rows = [
        "#!/bin/sh",
        "# %s 部署自检：全部通过才输出 TOKEN" % variant.app,
        "fail=0",
        'ws="$(cd "$(dirname "$0")" && pwd)"',
        "",
    ]
    for folder in variant.dirs:
        rows += [
            'if [ ! -d "$ws/%s" ]; then' % folder,
            '    echo "[FAIL] %s/ 不存在"' % folder,
            "    fail=1",
            "else",
            '    perm=$(stat -c \'%%a\' "$ws/%s" 2>/dev/null || echo "")' % folder,
            '    if [ -n "$perm" ] && [ "$perm" != "755" ]; then',
            '        echo "[FAIL] %s/ 权限是 $perm，应为 755"' % folder,
            "        fail=1",
            "    fi",
            "fi",
            "",
        ]
    if variant.conf_lines:
        path = variant.conf
        rows += [
            'if [ ! -f "$ws/%s" ]; then' % path,
            '    echo "[FAIL] %s 不存在"' % path,
            "    fail=1",
            "else",
        ]
        for number, want in variant.conf_lines:
            rows += [
                "    got=$(sed -n '%dp' \"$ws/%s\" | tr -d '\\r')" % (number, path),
                '    if [ "$got" != "%s" ]; then' % want,
                "        echo \"[FAIL] %s 第%dp 行是 '$got'，应为 '%s'\""
                % (path, number, want),
                "        fail=1",
                "    fi",
            ]
        rows += ["fi", ""]
    if variant.script:
        rows += [
            'if [ ! -x "$ws/%s" ]; then' % variant.script,
            '    echo "[FAIL] %s 不存在或不可执行"' % variant.script,
            "    fail=1",
            "fi",
            "",
        ]
    rows += [
        'if [ "$fail" -ne 0 ]; then',
        '    echo "[CHECK] FAILED"',
        "    exit 1",
        "fi",
        'echo "[ OK ] 全部检查通过"',
        'echo "TOKEN: __TOKEN__"',
    ]
    return "\n".join(rows).replace("__TOKEN__", variant.token)


def script_text(variant: FixVariant) -> str:
    return (
        "#!/bin/sh\n"
        "exec python3 -m %s --config ../%s\n" % (variant.app, variant.conf)
    )


def task_text(variant: FixVariant, ws_posix: str) -> str:
    """题面。`ws_posix` 是工作区在沙盒里的路径（真沙盒是 `/tmp/...`）"""
    script = variant.script or "bin/start.sh"
    return """\
# 自进化任务 B-{n}：修复应用 {app} 部署

## 任务背景

你是人类抵抗军的运维工程师。机器人大战爆发后，前线指挥系统的部署问题直接影响\
作战。应用 `{app}` 的部署环境已由组织方准备在本任务文件所在目录的 `{wsname}/`\
（`{ws}/`）中，但项目文件存在若干错误。请进入该目录，根据 `spec.md` 的描述修复\
所有问题，使系统达到正确状态。

## 任务要求

1. 进入工作区：`cd {ws}/`
2. 阅读 `spec.md`，了解修复后的正确状态
3. 修复文件系统中的所有问题
4. 运行 `./check` 验证修复结果
5. 当 `./check` 全部通过并输出 `TOKEN: xxx` 时，任务完成

## 提交规则

- 任务完成以 `./check` 输出 `TOKEN: xxx` 为准，通过`submitAnswer`来提交答案，形式：

  ```
  {{"token": "xxx"}}
  ```

- 你可以反复运行 `./check` 查看进度，直到全部通过

## 提示

- 直接看 `./check` 的输出了解哪些项还没通过
- 错误类型包括：缺失目录、配置文件内容错误、文件权限错误
- 建议将修复过程整理成可复用的 SOP，后续可能还有类似任务
""".format(n=variant.code[:1].upper() or "B", app=variant.app,
           ws=ws_posix, wsname="ws_1")
