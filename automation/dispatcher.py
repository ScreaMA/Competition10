"""任务调度器：将Issue转换为Claude任务，生成分支名与提示词。

对应设计文档 11.3 / 11.5 节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from issue_monitor import Issue

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """你正在处理仓库 {repo} 的Issue #{number}。

标题：{title}

内容：
{body}

要求：
1. 只修改完成该Issue所必需的代码，保持现有代码风格与结构。
2. 修改完成后不要执行git提交或推送，提交与推送由自动化系统负责。
3. 最后用中文简要说明你做了哪些改动。
"""


@dataclass
class Task:
    """一个由Issue转换而来的自动化任务"""

    issue_number: int
    title: str
    body: str
    branch_name: str
    prompt: str
    commit_message: str
    created_at: str

    def describe(self) -> str:
        return (
            f"Task(issue=#{self.issue_number}, branch={self.branch_name}, "
            f"title={self.title!r})"
        )


class Dispatcher:
    """Issue -> Task 转换器"""

    def __init__(
        self,
        repo_full_name: str,
        branch_prefix: str = "auto-fix",
        commit_prefix: str = "[Auto-Fix]",
    ) -> None:
        self.repo_full_name = repo_full_name
        self.branch_prefix = branch_prefix.strip("/") or "auto-fix"
        self.commit_prefix = commit_prefix

    @classmethod
    def from_config(cls, config: dict) -> "Dispatcher":
        github = config.get("github") or {}
        automation = config.get("automation") or {}
        git = config.get("git") or {}
        owner = github.get("repo_owner") or ""
        name = github.get("repo_name") or ""
        return cls(
            repo_full_name=f"{owner}/{name}".strip("/"),
            branch_prefix=str(automation.get("branch_prefix") or "auto-fix"),
            commit_prefix=str(git.get("commit_prefix") or "[Auto-Fix]"),
        )

    def build_branch_name(self, issue: Issue, now: datetime | None = None) -> str:
        """生成分支名，形如 auto-fix/issue-123-20260912-143052"""
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        return f"{self.branch_prefix}/issue-{issue.number}-{stamp}"

    def build_prompt(self, issue: Issue) -> str:
        """从Issue内容提取交给Claude的提示词"""
        body = (issue.body or "").strip() or "(Issue正文为空，请根据标题判断需求)"
        return PROMPT_TEMPLATE.format(
            repo=self.repo_full_name or "(unknown repo)",
            number=issue.number,
            title=issue.title,
            body=body,
        ).strip()

    def build_commit_message(self, issue: Issue) -> str:
        """生成提交信息，形如 [Auto-Fix] #123: 优化武器建造顺序"""
        title = issue.title.strip().replace("\n", " ")
        return f"{self.commit_prefix} #{issue.number}: {title}"

    def dispatch(self, issue: Issue) -> Task:
        """将Issue转换为Task"""
        task = Task(
            issue_number=issue.number,
            title=issue.title,
            body=issue.body,
            branch_name=self.build_branch_name(issue),
            prompt=self.build_prompt(issue),
            commit_message=self.build_commit_message(issue),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        LOGGER.info("dispatched %s", task.describe())
        return task
