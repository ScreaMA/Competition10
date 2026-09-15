"""把客户端生成的沙盒命令真正跑起来。

`scripts.build()` 产出的是 `sh` 包裹的 Python heredoc：

    for P in python3 python; do …; done; $P -u - <<'PYEOF' 2>&1
    <python>
    PYEOF

有 POSIX shell 时（Linux，或 Windows 上的 Git Bash）就按沙盒的样子**原样跑整条
命令**——连包装层、heredoc 引号、`2>&1` 重定向一起验证。只有连 shell 都找不到
时才退化成"抠出 Python 正文直接执行"（那一步查不出包装层的毛病）。

三种错误都会在这一层原样暴露：
    - 生成脚本的**签名/语法**错（沙盒里只表现为"这一步没有输出"）
    - 参数拼装错（blob 里的 base/ws/spec 指错地方）
    - 路径拼接错（相对路径在错误的 cwd 下解析）
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.strategy.task import scripts
from agent.strategy.task.memory import MEMORY, StepSpec

HEREDOC = re.compile(r"<<'PYEOF' 2>&1\n(.*)\nPYEOF\Z", re.S)
MARKER = re.compile(r"^\[([A-Z][A-Z0-9_]*)\]\s?(.*)$")
KV = re.compile(r'([a-zA-Z_][\w.]*)=(?:"([^"]*)"|(\S+))')


class StepResult:
    """一步的执行结果：原始输出 + 解析出来的标记"""

    def __init__(self, step: str, command: str, output: str):
        self.step = step
        self.command = command
        self.output = output
        self.markers: dict[str, str] = {}
        self.tags: list[str] = []
        for line in output.splitlines():
            match = MARKER.match(line)
            if match:
                tag, body = match.group(1), match.group(2).strip()
                self.tags.append(tag)
                self.markers.setdefault(tag, body)

    def has(self, tag: str) -> bool:
        return tag in self.markers

    def kv(self, key: str, default: str = "") -> str:
        """取 `TAG.key` 形式的字段（`[CHECK] ok=no code=1` -> `CHECK.code`）"""
        tag, _, name = key.partition(".")
        for found, quoted, plain in KV.findall(self.markers.get(tag, "")):
            if found == name:
                return quoted or plain
        return default

    def answer(self) -> dict[str, Any] | None:
        """`[ANSWER]` 里的 JSON（这一族要提交的东西）"""
        import json

        body = self.markers.get("ANSWER")
        if not body:
            return None
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None

    def token(self) -> str | None:
        body = self.markers.get("TOKEN")
        return body.split()[0] if body else None

    def errors(self) -> list[str]:
        """脚本崩了的痕迹（`TypeError` / `Traceback`）—— 沙盒里只会表现为
        "这一步没有输出"，本地要把它们捞出来"""
        found = []
        if "Traceback" in self.output:
            found.append("Traceback")
        for match in re.finditer(r"^(\w+Error): (.*)$", self.output, re.M):
            found.append("%s: %s" % (match.group(1), match.group(2)[:80]))
        return found


def body_of(command: str) -> str:
    match = HEREDOC.search(command)
    if not match:
        raise ValueError("命令不是预期的 heredoc 形式：%r" % command[:80])
    return match.group(1)


def _posix_shell() -> str | None:
    """可用的 POSIX shell（Windows 上由 Git Bash 提供）"""
    for name in ("sh", "bash"):
        found = shutil.which(name)
        if found:
            return found
    return None


_PYTHON3_SHIM = None


def _python_shim_dir() -> str:
    """造一个只有 `python3` 的目录，塞在 PATH 最前面

    客户端那条包装是 `for P in python3 python; do command -v "$P" …; done`——
    在 Windows 上 `python3` 常常解析到 Microsoft Store 的占位程序，`command -v`
    找得到、跑起来却什么都不干，整条命令就静默空转。垫一个真的上去，
    包装脚本本身就能原样跑。
    """
    global _PYTHON3_SHIM
    if _PYTHON3_SHIM:
        return _PYTHON3_SHIM
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="tasksandbox-bin-"))
    shim = folder / "python3"
    shim.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable.replace("\\", "/"),
                    encoding="utf-8", newline="")
    try:
        shim.chmod(0o755)
    except OSError:
        pass
    _PYTHON3_SHIM = str(folder)
    return _PYTHON3_SHIM


def execute(command: str, cwd: Path) -> str:
    """把客户端生成的那条命令**原样**跑起来，返回 stdout+stderr

    有 POSIX shell 时走 `sh -c "<整条命令>"` —— 连 `for P in python3 python …`
    那层包装、heredoc 引号、`2>&1` 重定向都一起验证（就是沙盒里的执行方式）。
    没有 shell 时才退化成"抠出 Python 正文直接跑"，那一步查不出包装层的问题。

    `PYTHONIOENCODING=utf-8`：标记正文里有中文，不钉死编码的话 Windows 会按
    本地代码页写 stdout（那是**脚手架**的问题，不是脚本的）。
    """
    shell = _posix_shell()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if shell:
        env["PATH"] = _python_shim_dir() + os.pathsep + env.get("PATH", "")
        # **落成脚本文件再执行，不要用 `sh -c "<整条命令>"`。**
        # MSYS2 版的 sh（Git Bash）会对 `-c` 的参数再做一层引号/转义处理，
        # heredoc 正文里的反斜杠被吃掉一层——参数 blob 里的 Windows 路径
        # 当场变成 `C:\Users`，`json.loads` 直接 `Invalid \escape` 崩掉。
        # 写文件跑没有这一层，Linux 上两种写法等价。
        script_path = Path(cwd) / "_task_cmd.sh"
        script_path.write_text(command, encoding="utf-8", newline="")
        proc = subprocess.run(
            [shell, str(script_path)], cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return proc.stdout.decode("utf-8", "replace").replace("\r\n", "\n")

    script = Path(cwd) / "_task_step.py"
    script.write_text(body_of(command), encoding="utf-8", newline="")
    proc = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def step_command(
    name: str,
    *,
    phase_task: str = "",
    facts: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    check_output: str = "",
) -> str:
    """生成一步的沙盒命令

    `facts` 缺省取**当前 `MEMORY` 的事实区**——生产路径就是这样
    （`solver._execute` 传的是 `self.memory.facts`）。内置的种子事实
    （鉴权头、参数名…）都在那里，不带上就等于把内置知识关掉了。
    """
    step = StepSpec(name)
    if params:
        step = step.with_params(**params)
    return scripts.build(
        step,
        phase_task=phase_task,
        facts=dict(MEMORY.facts) if facts is None else facts,
        check_output=check_output,
    )


def run_step(
    name: str,
    sandbox,
    *,
    phase_task: str = "",
    facts: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    check_output: str = "",
    cwd: Path | None = None,
) -> StepResult:
    """生成一步的命令并在沙盒里执行

    `cwd` 默认取任务目录：客户端的 `find_task_dir` 会从 `.` 扫起，工作目录
    就是它唯一的"落脚点"（本地没有 `/tmp/selfEvolutionTask`）。
    """
    command = step_command(
        name, phase_task=phase_task, facts=facts, params=params,
        check_output=check_output,
    )
    where = Path(cwd) if cwd is not None else Path(sandbox.api_dir)
    return StepResult(name, command, execute(command, where))
