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
        # 本地与远端分叉时的处理方式：rebase(默认) / merge / ff-only
        self.sync_strategy = str(automation.get("sync_strategy") or "rebase")
        # 是否自动合并自己创建的PR（免人工审批）
        self.auto_merge = bool(automation.get("auto_merge", False))
        # 自动合并方式：squash(默认) / merge / rebase
        self.auto_merge_method = str(automation.get("auto_merge_method") or "squash")
        # 自动合并的最大尝试次数（PR刚创建时GitHub可能还没算完可合并性）
        self.max_merge_attempts = int(automation.get("max_merge_attempts") or 5)
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

        # 记录任务开始前就已存在的未提交文件，避免把别人的WIP卷进本次提交
        pre_dirty = self.pusher.dirty_paths()
        if pre_dirty:
            LOGGER.warning("工作区已有 %d 个未提交文件，本次提交将跳过它们：%s",
                           len(pre_dirty), ", ".join(pre_dirty[:5]))

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

        pr_url = self.pusher.publish(task, summary=result.output, pre_dirty=pre_dirty)
        extra: dict[str, Any] = {"branch": task.branch_name}
        number = pull_number(pr_url)
        if number:
            extra["pr"] = number
        self.monitor.mark_processed(
            issue.number, "published" if pr_url else "no_change", pr_url, **extra,
        )

        # 开了 auto_merge 就立即尝试合并并同步，免人工审批
        if pr_url and self.auto_merge:
            self.merge_pull_request(
                issue.number, self.monitor.record(issue.number), task.branch_name,
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
        """检查自己创建的PR：能合并的自动合并，已合并的把本地代码同步到远端最新

        每轮轮询开始时调用。PR的处理顺序：
            1. 已合并  -> 同步本地主干（快进/rebased，见 sync_strategy）
            2. 已关闭未合并 -> 停止跟踪
            3. 仍开放且开启 auto_merge -> 调API自动合并，成功后立即同步
            4. 仍开放且未开启 auto_merge -> 留待人工审批

        返回:
            本轮完成同步的PR数量
        """
        if not (self.auto_sync or self.auto_merge):
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
                if not self.auto_sync:
                    self.monitor.mark_synced(issue_number, f"PR #{number} merged")
                    continue
                LOGGER.info("PR #%s（issue #%d）已合并，同步本地主干 %s",
                            number, issue_number, self.main_branch())
                if self.pusher.sync_main(self.main_branch(), branch,
                                         self.delete_remote_branch,
                                         self.sync_strategy):
                    self.monitor.mark_synced(issue_number, f"PR #{number} merged")
                    synced += 1
                else:
                    LOGGER.warning("PR #%s 已合并但本地同步未完成，下轮重试", number)
            elif str(pull.get("state") or "") == "closed":
                LOGGER.info("PR #%s（issue #%d）已关闭未合并，不再跟踪",
                            number, issue_number)
                self.monitor.mark_synced(issue_number, f"PR #{number} closed")
            elif self.auto_merge:
                # 仍开放：自动合并，无需人工审批
                if self.merge_pull_request(issue_number, record, branch):
                    synced += 1
            else:
                LOGGER.debug("PR #%s 仍处于 %s 状态（auto_merge 未开启）",
                             number, pull.get("state"))

        if synced:
            LOGGER.info("本轮同步了 %d 个PR", synced)
        return synced

    def merge_pull_request(self, issue_number: int, record: dict[str, Any],
                           branch: str = "") -> bool:
        """自动合并PR，成功后立即把本地代码同步到远端

        合并失败（例如GitHub还没算完可合并性、存在冲突）不会中断流程：
        记录尝试次数，达到 max_merge_attempts 后停止重试并在Issue中说明，
        交由人工处理。

        返回:
            True 表示已合并并完成本地同步
        """
        number = int(record.get("pr") or pull_number(str(record.get("detail") or "")) or 0)
        if not number:
            return False
        branch = branch or str(record.get("branch") or "")
        attempts = int(record.get("merge_attempts") or 0) + 1

        try:
            self.client.merge_pull_request(number, self.auto_merge_method)
        except Exception as exc:
            message = str(exc)
            if attempts >= self.max_merge_attempts:
                LOGGER.warning("PR #%s 自动合并失败（已尝试%d次）：%s",
                               number, attempts, message)
                self.monitor.update_record(
                    issue_number, status="merge_failed",
                    merge_attempts=attempts, merge_error=message[:300],
                )
                if self.comment_on_issue:
                    self.pusher.comment_issue(
                        issue_number,
                        f"自动合并 PR #{number} 失败（已重试{attempts}次），需要人工处理：\n"
                        f"```\n{message[:1000]}\n```",
                    )
            else:
                LOGGER.warning("PR #%s 暂不可合并（第%d次尝试），下轮重试：%s",
                               number, attempts, message)
                self.monitor.update_record(issue_number, merge_attempts=attempts)
            return False

        LOGGER.info("PR #%s（issue #%d）已自动合并（%s）",
                    number, issue_number, self.auto_merge_method)

        if not self.auto_sync:
            self.monitor.mark_synced(issue_number, f"PR #{number} auto-merged")
            return True

        if self.pusher.sync_main(self.main_branch(), branch,
                                 self.delete_remote_branch, self.sync_strategy):
            self.monitor.mark_synced(issue_number, f"PR #{number} auto-merged")
            if self.comment_on_issue:
                self.pusher.comment_issue(
                    issue_number,
                    f"已自动合并 PR #{number} 并把本地代码同步到最新。",
                )
            return True

        LOGGER.warning("PR #%s 已合并，但本地同步未完成，下轮重试", number)
        return False

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
