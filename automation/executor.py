"""Claude执行器：调用本地Claude Code CLI完成代码修改。

对应设计文档 11.3 / 11.5 节。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

from dispatcher import Task

LOGGER = logging.getLogger(__name__)


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

    def run(self, task: Task) -> ExecResult:
        """执行任务，返回执行结果"""
        command = [self._resolve(), *self.extra_args, task.prompt]
        LOGGER.info("executing claude for issue #%d (timeout=%ds)",
                    task.issue_number, self.timeout)
        LOGGER.debug("command: %s", " ".join(command))

        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            LOGGER.error("claude timed out after %.1fs", duration)
            return ExecResult(
                ok=False,
                returncode=-1,
                output=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                duration=duration,
                error=f"timeout after {self.timeout}s",
            )
        except OSError as exc:
            duration = time.time() - started
            LOGGER.exception("failed to launch claude")
            return ExecResult(
                ok=False, returncode=-1, output="", duration=duration, error=str(exc),
            )

        duration = time.time() - started
        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        ok = completed.returncode == 0
        LOGGER.info(
            "claude finished: rc=%d ok=%s duration=%.1fs output=%d chars",
            completed.returncode, ok, duration, len(output),
        )
        if error:
            LOGGER.debug("claude stderr: %s", error[:2000])
        return ExecResult(
            ok=ok,
            returncode=completed.returncode,
            output=output,
            duration=duration,
            error=error,
        )
