"""Claude执行器：调用本地Claude Code CLI完成代码修改。

对应设计文档 11.3 / 11.5 节。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from dispatcher import Task

LOGGER = logging.getLogger(__name__)

# INFO 日志中保留的 Claude 回复长度
SUMMARY_LOG_LENGTH = 500

# 让 Claude Code 进入非交互打印模式的参数；该模式下提示词可直接从 stdin 读入
PRINT_MODE_FLAGS = ("-p", "--print")


def _shorten(text: str, limit: int = SUMMARY_LOG_LENGTH) -> str:
    """截断长文本，保留长度信息"""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(共{len(text)}字符)"


def _single_line(prompt: str) -> str:
    """把多行提示词压成单行（降级路径专用）"""
    return " ".join(prompt.split())


@dataclass
class ExecResult:
    """一次Claude执行的结果"""

    ok: bool
    returncode: int
    output: str
    duration: float
    error: str = ""


class ClaudeExecutor:
    """封装Claude Code CLI的调用"""

    def __init__(
        self,
        executable: str = "claude",
        work_dir: str = ".",
        timeout: int = 600,
        extra_args: list[str] | None = None,
    ) -> None:
        self.executable = executable
        self.work_dir = work_dir
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    @classmethod
    def from_config(cls, config: dict) -> "ClaudeExecutor":
        claude = config.get("claude") or {}
        return cls(
            executable=str(claude.get("executable") or "claude"),
            work_dir=str(claude.get("work_dir") or "."),
            timeout=int(claude.get("timeout") or 600),
            extra_args=list(claude.get("extra_args") or []),
        )

    def _resolve(self) -> str:
        """解析可执行文件路径"""
        found = shutil.which(self.executable)
        if found is None:
            LOGGER.warning("claude executable %r not found on PATH", self.executable)
        return found or self.executable

    def _build_invocation(self, task: Task) -> tuple[list[str], str | None]:
        """构造命令行与 stdin 内容，返回 (command, stdin_text)。

        提示词默认从 stdin 送入，而不是作为 argv 的一个元素：Issue 提示词是多行
        文本（标题、正文、要求各占一行），而 Windows 上 `claude` 通常解析到 npm
        生成的 `claude.cmd` 垫片，命令行参数在经 cmd.exe 解析时会被第一个换行截断
        ——Claude 只会看到提示词的第一行，标题和正文全部丢失，于是「没改代码」。
        走 stdin 还能顺带绕开 Windows 约 32K 的命令行长度上限。
        """
        command = [self._resolve(), *self.extra_args]
        if any(flag in self.extra_args for flag in PRINT_MODE_FLAGS):
            return command, task.prompt
        # 没配置 -p/--print 时 CLI 会进入交互模式（本身就会挂住），这里仅保证提示词
        # 仍能完整送达，不额外改变原有行为
        LOGGER.warning("%s missing in extra_args, prompt passed as a single-line argument",
                       "/".join(PRINT_MODE_FLAGS))
        return [*command, _single_line(task.prompt)], None

    def _kill_tree(self, process: subprocess.Popen) -> None:
        """终止整个进程树

        Windows 上 `claude` 解析到的是 npm 生成的 .cmd 垫片，直接 kill 只会杀掉
        cmd.exe，真正在跑的 node 子进程会继续运行（实测 timeout=600s 的任务跑了
        1111s 才结束）。必须用 taskkill /T 连子孙进程一起杀。
        """
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True, check=False,
            )
            return
        try:
            process.kill()
        except OSError:
            LOGGER.exception("failed to kill claude process")

    def run(self, task: Task) -> ExecResult:
        """执行任务，返回执行结果"""
        command, stdin_text = self._build_invocation(task)
        LOGGER.info(
            "executing claude for issue #%d (timeout=%ds, prompt=%d chars/%d lines)",
            task.issue_number, self.timeout,
            len(task.prompt), task.prompt.count("\n") + 1,
        )
        LOGGER.debug("command: %s", " ".join(command))

        started = time.time()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.work_dir,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            duration = time.time() - started
            LOGGER.exception("failed to launch claude")
            return ExecResult(
                ok=False, returncode=-1, output="", duration=duration, error=str(exc),
            )

        timed_out = False
        try:
            stdout_text, stderr_text = process.communicate(
                input=stdin_text, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(process)
            try:
                stdout_text, stderr_text = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                stdout_text, stderr_text = "", ""

        duration = time.time() - started
        output = (stdout_text or "").strip()
        error = (stderr_text or "").strip()

        if timed_out:
            LOGGER.error("claude timed out after %.1fs", duration)
            return ExecResult(
                ok=False, returncode=-1, output=output, duration=duration,
                error=f"timeout after {self.timeout}s",
            )

        ok = process.returncode == 0
        LOGGER.info(
            "claude finished: rc=%s ok=%s duration=%.1fs output=%d chars",
            process.returncode, ok, duration, len(output),
        )
        if output:
            # 把 Claude 的结论记入日志，便于复盘“为什么这样改/为什么没改”
            LOGGER.info("claude summary: %s", _shorten(output))
            LOGGER.debug("claude output:\n%s", output)
        if error:
            LOGGER.warning("claude stderr: %s", _shorten(error))
        return ExecResult(
            ok=ok,
            returncode=process.returncode if process.returncode is not None else -1,
            output=output,
            duration=duration,
            error=error,
        )
