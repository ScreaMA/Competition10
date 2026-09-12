#!/usr/bin/env python3
"""自动化系统主程序：整合Issue监控、任务调度、Claude执行与Git推送。

对应设计文档 11.3 / 11.6 节。

用法:
    pip install pyyaml requests
    python automation_main.py                 # 按配置轮询
    python automation_main.py --once          # 只处理一轮
    python automation_main.py --config my.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dispatcher import Dispatcher
from executor import ClaudeExecutor
from git_pusher import GitPusher
from issue_monitor import GitHubClient, IssueMonitor, load_token

LOGGER = logging.getLogger("automation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "automation_config.yaml"

# 从 PR 链接中解析编号，例如 https://github.com/o/r/pull/3 -> 3
PULL_URL_RE = re.compile(r"/pull/(\d+)")


def pull_number(url: str) -> int | None:
    """从PR链接中解析PR编号，解析失败返回 None"""
    match = PULL_URL_RE.search(url or "")
    return int(match.group(1)) if match else None


def setup_logging(config: dict[str, Any]) -> None:
    """配置日志输出"""
    logging_config = config.get("logging") or {}
    level = getattr(
        logging, str(logging_config.get("level") or "INFO").upper(), logging.INFO,
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file = logging_config.get("file")
    if log_file:
        path = Path(str(log_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_config(path: Path) -> dict[str, Any]:
    """加载YAML配置"""
    if not path.is_file():
        raise SystemExit(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    LOGGER.info("config loaded from %s", path)
    return config


class Automation:
    """自动化系统：轮询Issue -> 调用Claude -> 推送PR"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        github = config.get("github") or {}
        automation = config.get("automation") or {}

        self.poll_interval = int(automation.get("poll_interval") or 300)
        self.auto_create_pr = bool(automation.get("auto_create_pr", True))
        self.comment_on_issue = bool(automation.get("comment_on_issue", True))
        self.branch_prefix = str(automation.get("branch_prefix") or "auto-fix")
        # PR合并后是否把本地代码同步到远端最新
        self.auto_sync = bool(automation.get("auto_sync", True))
        # 同步时是否顺带删除远端已合并分支
        self.delete_remote_branch = bool(
            automation.get("delete_remote_branch_after_merge", False)
        )
        self._main_branch: str | None = None

        self.token = load_token(str(github.get("token_file") or ""))
        self.client = GitHubClient(
            token=self.token,
            owner=str(github.get("repo_owner") or ""),
            repo=str(github.get("repo_name") or ""),
            proxy=str(github.get("proxy") or ""),
        )

        self.monitor = IssueMonitor.from_config(config, self.client)
        self.dispatcher = Dispatcher.from_config(config)
        self.executor = ClaudeExecutor.from_config(config)
        self.pusher = GitPusher.from_config(
            config, self.client, self.token, repo_dir=PROJECT_ROOT,
        )
        # Claude 的工作目录同样以项目根目录为基准
        if self.executor.work_dir in (".", "./"):
            self.executor.work_dir = str(PROJECT_ROOT)

    # === 单个Issue的处理 ===

    def handle(self, issue) -> bool:
        """处理一个Issue，返回是否成功"""
        task = self.dispatcher.dispatch(issue)
        LOGGER.info("start processing %s", task.describe())

        result = self.executor.run(task)
        if not result.ok:
            LOGGER.error("claude execution failed for issue #%d: %s",
                         issue.number, result.error or f"rc={result.returncode}")
            self.monitor.mark_processed(issue.number, "exec_failed", result.error)
            if self.comment_on_issue:
                self.pusher.comment_issue(
                    issue.number,
                    f"自动化系统执行Claude失败（rc={result.returncode}）：\n"
                    f"```\n{result.error[:1000]}\n```",
                )
            return False

        if not self.auto_create_pr:
            LOGGER.info("auto_create_pr disabled, stop after execution")
            self.monitor.mark_processed(issue.number, "executed", "pr skipped")
            return True

        pr_url = self.pusher.publish(task, summary=result.output)
        extra: dict[str, Any] = {"branch": task.branch_name}
        number = pull_number(pr_url)
        if number:
            extra["pr"] = number
        self.monitor.mark_processed(
            issue.number, "published" if pr_url else "no_change", pr_url, **extra,
        )
        return bool(pr_url)

    # === 与远端同步 ===

    def main_branch(self) -> str:
        """主干分支名（缓存，避免每轮多一次API调用）"""
        if self._main_branch is None:
            try:
                self._main_branch = self.client.default_branch()
            except Exception:
                LOGGER.exception("获取默认分支失败，暂按 main 处理")
                self._main_branch = "main"
        return self._main_branch

    def sync_repository(self) -> int:
        """检查已创建的PR是否已合并，合并后把本地代码同步到远端最新

        每轮轮询开始时调用。只有快进（fast-forward）才会更新本地主干，
        本地领先或分叉、工作区不干净时保持现状并在下一轮重试。

        返回:
            本轮完成同步（合并或关闭）的PR数量
        """
        if not self.auto_sync:
            return 0

        synced = 0
        for issue_number, record in self.monitor.pending_pull_requests():
            number = record.get("pr") or pull_number(str(record.get("detail") or ""))
            if not number:
                continue
            try:
                pull = self.client.get_pull_request(int(number))
            except Exception:
                LOGGER.exception("查询PR #%s 失败（issue #%d），下轮重试",
                                 number, issue_number)
                continue

            branch = str((pull.get("head") or {}).get("ref")
                         or record.get("branch") or "")
            if pull.get("merged"):
                LOGGER.info("PR #%s（issue #%d）已合并，同步本地主干 %s",
                            number, issue_number, self.main_branch())
                if self.pusher.sync_main(self.main_branch(), branch,
                                         self.delete_remote_branch):
                    self.monitor.mark_synced(issue_number, f"PR #{number} merged")
                    synced += 1
                else:
                    LOGGER.warning("PR #%s 已合并但本地同步未完成，下轮重试", number)
            elif str(pull.get("state") or "") == "closed":
                LOGGER.info("PR #%s（issue #%d）已关闭未合并，不再跟踪",
                            number, issue_number)
                self.monitor.mark_synced(issue_number, f"PR #{number} closed")
            else:
                LOGGER.debug("PR #%s 仍处于 %s 状态", number, pull.get("state"))

        if synced:
            LOGGER.info("本轮同步了 %d 个已合并PR", synced)
        return synced

    # === 主循环 ===

    def run_once(self) -> int:
        """处理一轮：先同步已合并的改动，再处理新Issue"""
        try:
            self.sync_repository()
        except Exception:
            # 同步失败不影响本轮Issue处理
            LOGGER.exception("代码同步失败，继续处理Issue")

        candidates = self.monitor.fetch_candidates()
        handled = 0
        for issue in candidates:
            try:
                if self.handle(issue):
                    handled += 1
            except Exception:
                LOGGER.exception("unexpected failure on issue #%d", issue.number)
                self.monitor.mark_processed(issue.number, "error", "unexpected")
        LOGGER.info("round finished, %d/%d issues handled", handled, len(candidates))
        return handled

    def run_forever(self) -> None:
        """按配置的间隔持续轮询"""
        LOGGER.info("automation started (poll_interval=%ds)", self.poll_interval)
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                LOGGER.info("interrupted, exiting")
                return
            except Exception:
                # 异常不影响后续轮询
                LOGGER.exception("polling round failed")
            time.sleep(self.poll_interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitHub Issue 自动化处理系统")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="配置文件路径（默认 automation/automation_config.yaml）")
    parser.add_argument("--once", action="store_true",
                        help="只执行一轮，不进入轮询循环")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 统一以项目根目录为工作目录，便于相对路径解析
    os.chdir(PROJECT_ROOT)

    config = load_config(Path(args.config))
    setup_logging(config)

    try:
        automation = Automation(config)
    except Exception as exc:
        LOGGER.error("failed to initialise automation: %s", exc)
        raise SystemExit(1)

    if args.once:
        automation.run_once()
    else:
        automation.run_forever()


if __name__ == "__main__":
    main()
