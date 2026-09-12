"""Git操作器：创建分支、提交、推送、创建PR并回评Issue。

对应设计文档 11.3 / 11.5 节。

注意：自动化系统遵循“创建PR而非直接推送到主干”的原则，
所有修改都推送到独立的 auto-fix 分支，等待人工审核合并。
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dispatcher import Task
from issue_monitor import GitHubClient

LOGGER = logging.getLogger(__name__)


@dataclass
class GitResult:
    """一次git命令的执行结果"""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class GitPusher:
    """封装仓库上的git操作与PR创建"""

    def __init__(
        self,
        client: GitHubClient,
        repo_dir: str | Path = ".",
        repo_url: str = "",
        token: str = "",
        user_name: str = "Auto-Fix Bot",
        user_email: str = "bot@competition10.local",
        remote: str = "origin",
        proxy: str = "",
    ) -> None:
        self.client = client
        self.repo_dir = Path(repo_dir)
        self.repo_url = repo_url
        self.token = token
        self.user_name = user_name
        self.user_email = user_email
        self.remote = remote
        self.proxy = proxy

    @classmethod
    def from_config(cls, config: dict, client: GitHubClient, token: str,
                    repo_dir: str | Path = ".") -> "GitPusher":
        github = config.get("github") or {}
        git = config.get("git") or {}
        return cls(
            client=client,
            repo_dir=repo_dir,
            repo_url=str(github.get("repo_url") or ""),
            token=token,
            user_name=str(git.get("user_name") or "Auto-Fix Bot"),
            user_email=str(git.get("user_email") or "bot@competition10.local"),
            # git.proxy 可单独覆盖，未设置时复用 github.proxy
            proxy=str(git.get("proxy") or github.get("proxy") or ""),
        )

    # === 基础操作 ===

    def _git(self, *args: str, check: bool = False) -> GitResult:
        """执行git命令"""
        command = ["git"]
        # 直连 GitHub 不通时，git 也需要走代理
        if self.proxy:
            command += ["-c", f"http.proxy={self.proxy}"]
        command += list(args)
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            LOGGER.exception("failed to run git")
            return GitResult(-1, "", str(exc))

        result = GitResult(
            completed.returncode,
            (completed.stdout or "").strip(),
            (completed.stderr or "").strip(),
        )
        LOGGER.debug("git %s -> rc=%d", " ".join(args), result.returncode)
        if check and not result.ok:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {result.stderr or result.stdout}"
            )
        return result

    def current_branch(self) -> str:
        """获取当前分支名"""
        result = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout or "main"

    def has_changes(self) -> bool:
        """工作区是否存在未提交的改动"""
        return bool(self._git("status", "--porcelain").stdout)

    def changed_files(self) -> list[str]:
        """列出改动的文件"""
        lines = self._git("status", "--porcelain").stdout.splitlines()
        return [line[3:].strip() for line in lines if line.strip()]

    def _auth_url(self) -> str:
        """生成带token的推送地址（仅用于push，不落盘）"""
        if not self.repo_url or not self.token:
            return ""
        url = self.repo_url
        if url.startswith("https://"):
            return url.replace(
                "https://", f"https://x-access-token:{self.token}@", 1,
            )
        return ""

    # === 主流程 ===

    def create_branch(self, branch_name: str) -> None:
        """基于当前HEAD创建并切换到工作分支"""
        LOGGER.info("creating branch %s", branch_name)
        self._git("checkout", "-B", branch_name, check=True)

    def commit_all(self, message: str) -> bool:
        """提交所有改动，无改动时返回False"""
        if not self.has_changes():
            LOGGER.warning("no changes to commit")
            return False
        self._git("add", "-A", check=True)
        self._git(
            "-c", f"user.name={self.user_name}",
            "-c", f"user.email={self.user_email}",
            "commit", "-m", message,
            check=True,
        )
        LOGGER.info("committed: %s", message)
        return True

    def push(self, branch_name: str) -> None:
        """推送分支到远程仓库"""
        LOGGER.info("pushing branch %s", branch_name)
        auth_url = self._auth_url()
        if auth_url:
            self._git("push", auth_url, f"{branch_name}:{branch_name}", check=True)
        else:
            # 未配置repo_url/token时，依赖本地已配置的凭据
            self._git("push", "-u", self.remote, branch_name, check=True)

    def open_pull_request(self, task: Task, base: str, summary: str = "") -> str:
        """创建Pull Request，返回PR链接"""
        body = (
            f"由自动化系统根据 Issue #{task.issue_number} 生成。\n\n"
            f"**Issue标题**: {task.title}\n\n"
            f"**改动文件**:\n"
            + "\n".join(f"- `{name}`" for name in self.changed_files())
        )
        if summary:
            body += f"\n\n**执行摘要**（Claude 自述）:\n\n```\n{summary[:2000]}\n```"
        body += "\n\n> 请人工审核后再合并。"

        try:
            pr = self.client.create_pull_request(
                title=task.commit_message,
                head=task.branch_name,
                base=base,
                body=body,
            ) or {}
        except Exception:
            LOGGER.exception("failed to create pull request for %s", task.branch_name)
            return ""
        url = str(pr.get("html_url") or "")
        LOGGER.info("pull request created: %s", url)
        return url

    def comment_issue(self, issue_number: int, body: str) -> None:
        """在Issue中回评"""
        try:
            self.client.comment_issue(issue_number, body)
            LOGGER.info("commented on issue #%d", issue_number)
        except Exception:
            LOGGER.exception("failed to comment on issue #%d", issue_number)

    def rollback(self, branch_name: str, original_branch: str) -> None:
        """失败回滚：切回原分支并删除工作分支"""
        LOGGER.warning("rolling back branch %s", branch_name)
        self._git("checkout", original_branch)
        self._git("branch", "-D", branch_name)

    # === 与远端同步 ===

    def sync_main(self, main_branch: str, merged_branch: str = "",
                  delete_remote_branch: bool = False,
                  strategy: str = "rebase") -> bool:
        """把本地主干同步到远端最新（PR合并后调用）

        参数:
            main_branch: 主干分支名（如 main）
            merged_branch: 刚被合并的工作分支，同步后顺带清理本地分支
            delete_remote_branch: 是否同时删除远端已合并分支
            strategy: 本地与远端分叉时的处理方式
                - "rebase": 把本地提交重放到远端之上（默认，保持线性历史）
                - "merge": 生成一个合并提交
                - "ff-only": 只允许快进，分叉时保持现状

        返回:
            True 表示本地已与远端主干一致（前进了，或本来就一致）
            False 表示未同步（工作区脏、拉取失败、冲突已回滚等），调用方可下轮重试

        安全约束（重要）:
            - 优先快进（--ff-only）
            - 需要 rebase/merge 时，一旦冲突立即 --abort 回滚，绝不留下半成品
            - 工作区不干净时直接跳过，避免把未提交的改动卷进/丢失
            - 清理本地分支用 `git branch -d`，未合并的分支会拒绝删除
        """
        if self.has_changes():
            LOGGER.warning("工作区有未提交改动，跳过同步")
            return False

        fetch = self._git("fetch", self.remote)
        if not fetch.ok:
            LOGGER.warning("fetch 失败，跳过同步: %s",
                           fetch.stderr or fetch.stdout)
            return False

        if self.current_branch() != main_branch:
            LOGGER.info("切换到 %s 以同步远端提交", main_branch)
            if not self._git("checkout", main_branch).ok:
                LOGGER.warning("切换分支失败，跳过同步")
                return False

        remote_ref = f"{self.remote}/{main_branch}"
        before = self._git("rev-parse", "HEAD").stdout

        if not self._git("merge", "--ff-only", remote_ref).ok:
            if strategy == "ff-only":
                LOGGER.warning("本地 %s 与 %s 已分叉（strategy=ff-only），保持现状",
                               main_branch, remote_ref)
                return False
            if strategy == "merge":
                merged = self._git("merge", "--no-edit", remote_ref)
                abort_args = ("merge", "--abort")
            else:
                merged = self._git("rebase", remote_ref)
                abort_args = ("rebase", "--abort")

            if not merged.ok:
                self._git(*abort_args)
                LOGGER.warning("与 %s 同步出现冲突，已回滚（strategy=%s）：%s",
                               remote_ref, strategy,
                               (merged.stderr or merged.stdout).splitlines()[-1:])
                return False
            LOGGER.info("本地提交已按 %s 方式叠放到 %s 之上", strategy, remote_ref)

        after = self._git("rev-parse", "HEAD").stdout
        if before == after:
            LOGGER.info("本地 %s 已是最新 (%s)", main_branch, after[:8])
            return True

        LOGGER.info("已同步 %s：%s -> %s", main_branch, before[:8], after[:8])

        if merged_branch and merged_branch != main_branch:
            # 用 -d（安全删除）：只有确实已并入主干才会删除，否则保留并告警
            removed = self._git("branch", "-d", merged_branch)
            if removed.ok:
                LOGGER.info("已清理本地分支 %s", merged_branch)
            else:
                LOGGER.warning("本地分支 %s 未删除：%s", merged_branch,
                               removed.stderr or removed.stdout)
        if delete_remote_branch and merged_branch:
            target = self._auth_url() or self.remote
            self._git("push", target, "--delete", merged_branch)
        return True

    # === 组合动作 ===

    def publish(self, task: Task, summary: str = "") -> str:
        """提交、推送、创建PR并回评Issue，返回PR链接

        参数:
            task: 待执行的任务
            summary: Claude 的执行摘要，会写入PR正文与Issue回评，便于人工复盘

        失败时回滚到原始分支，保证下一次任务从干净的起点开始。
        """
        original_branch = self.current_branch()
        self.create_branch(task.branch_name)
        try:
            if not self.commit_all(task.commit_message):
                self.rollback(task.branch_name, original_branch)
                detail = (
                    f"\n\n<details><summary>执行摘要（Claude 自述）</summary>\n\n"
                    f"```\n{summary[:2000]}\n```\n</details>"
                    if summary else ""
                )
                self.comment_issue(
                    task.issue_number,
                    f"自动化系统未检测到代码改动（分支 `{task.branch_name}` 已回滚），"
                    f"未创建PR。请确认Issue描述是否需要代码变更。{detail}",
                )
                return ""

            self.push(task.branch_name)
            base = self.client.default_branch()
            pr_url = self.open_pull_request(task, base, summary)
            if pr_url:
                self.comment_issue(
                    task.issue_number,
                    f"自动化系统已提交修改，PR: {pr_url}\n\n请人工审核后合并。",
                )
            else:
                self.comment_issue(
                    task.issue_number,
                    f"代码已推送到分支 `{task.branch_name}`，但PR创建失败，请手动创建。",
                )
            # 回到原分支，便于处理下一个Issue
            self._git("checkout", original_branch)
            return pr_url
        except Exception as exc:
            LOGGER.exception("publish failed for %s", task.branch_name)
            self.rollback(task.branch_name, original_branch)
            self.comment_issue(
                task.issue_number,
                f"自动化处理失败：{exc}\n请人工检查后重试。",
            )
            return ""
