"""Issue监控器：轮询GitHub Issues，按标签过滤需要处理的任务。

对应设计文档 11.3 / 11.5 节。

使用GitHub REST API v3（通过requests访问），并记录已处理的Issue，
避免同一个Issue被重复处理。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
DEFAULT_TIMEOUT = 30

# 每轮默认只处理最新的几个 issue（见 `IssueMonitor.max_candidates`）
DEFAULT_MAX_ISSUES = 3

# token 文件候选路径（相对于 automation 目录）
_TOKEN_CANDIDATES = (
    "github_token.txt",
    "githubtoken.txt",
    "../github_token.txt",
    "../githubtoken.txt",
)


@dataclass
class Issue:
    """一个待处理的GitHub Issue"""

    number: int
    title: str
    body: str
    labels: tuple[str, ...] = ()
    state: str = "open"
    html_url: str = ""
    user: str = ""

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Issue":
        return cls(
            number=int(raw["number"]),
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
            labels=tuple(
                str(label.get("name") if isinstance(label, dict) else label)
                for label in raw.get("labels") or ()
            ),
            state=str(raw.get("state") or "open"),
            html_url=str(raw.get("html_url") or ""),
            user=str((raw.get("user") or {}).get("login") or ""),
        )

    @property
    def summary(self) -> str:
        """Issue 的单行摘要"""
        return f"#{self.number} [{', '.join(self.labels) or '-'}] {self.title}"


def load_token(path_value: str) -> str:
    """读取GitHub Token

    优先使用配置文件中指定的路径；若不存在，则依次尝试 automation 目录
    及项目根目录下的常见文件名。也支持直接通过环境变量 GITHUB_TOKEN 提供。
    """
    env_token = os.getenv("GITHUB_TOKEN")
    if env_token:
        return env_token.strip()

    candidates = [os.getenv("GITHUB_TOKEN_FILE") or "", path_value, *_TOKEN_CANDIDATES]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                LOGGER.info("token loaded from %s", path)
                return token
    raise FileNotFoundError(
        "GitHub token not found; checked: "
        + ", ".join(str(c) for c in candidates if c)
    )


class GitHubClient:
    """GitHub REST API 的最小封装"""

    def __init__(self, token: str, owner: str, repo: str, proxy: str = "") -> None:
        self.owner = owner
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "competition10-automation",
        })
        # 直连 GitHub 不通时可通过配置走本地代理
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
            LOGGER.info("github api using proxy %s", proxy)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """发送API请求并返回JSON结果"""
        url = path if path.startswith("http") else f"{API_ROOT}{path}"
        response = self.session.request(
            method, url, params=params, json=payload, timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub API {method} {path} failed: "
                f"{response.status_code} {response.text[:500]}"
            )
        if not response.content:
            return None
        return response.json()

    # === Issue ===

    def list_issues(self, labels: list[str] | None = None,
                    state: str = "open") -> list[Issue]:
        """列出仓库中的Issue（不含Pull Request）"""
        params: dict[str, Any] = {"state": state, "per_page": 100}
        if labels:
            params["labels"] = ",".join(labels)
        raw = self.request(
            "GET", f"/repos/{self.owner}/{self.repo}/issues", params=params,
        ) or []
        # 带有 pull_request 字段的条目实际是PR，需要过滤掉
        return [
            Issue.load(item) for item in raw
            if "pull_request" not in item
        ]

    def get_issue(self, number: int) -> Issue:
        """获取单个Issue"""
        raw = self.request("GET", f"/repos/{self.owner}/{self.repo}/issues/{number}")
        return Issue.load(raw)

    def get_pull_request(self, number: int) -> dict[str, Any]:
        """获取单个Pull Request（含 merged/state/head.ref 等字段）"""
        return self.request(
            "GET", f"/repos/{self.owner}/{self.repo}/pulls/{number}"
        ) or {}

    def comment_issue(self, number: int, body: str) -> Any:
        """在Issue下添加评论"""
        return self.request(
            "POST",
            f"/repos/{self.owner}/{self.repo}/issues/{number}/comments",
            payload={"body": body},
        )

    # === Pull Request ===

    def create_pull_request(self, title: str, head: str, base: str,
                            body: str = "") -> dict[str, Any]:
        """创建Pull Request"""
        return self.request(
            "POST",
            f"/repos/{self.owner}/{self.repo}/pulls",
            payload={"title": title, "head": head, "base": base, "body": body},
        )

    def merge_pull_request(self, number: int, method: str = "squash",
                           commit_title: str = "") -> dict[str, Any]:
        """合并Pull Request

        参数:
            method: merge / squash / rebase
        返回:
            {"merged": true, "sha": "..."}；不可合并时抛 RuntimeError（含状态码与原因）
        """
        payload: dict[str, Any] = {"merge_method": method}
        if commit_title:
            payload["commit_title"] = commit_title
        return self.request(
            "PUT",
            f"/repos/{self.owner}/{self.repo}/pulls/{number}/merge",
            payload=payload,
        ) or {}

    def default_branch(self) -> str:
        """获取仓库默认分支名"""
        raw = self.request("GET", f"/repos/{self.owner}/{self.repo}")
        return str((raw or {}).get("default_branch") or "main")


class ProcessedStore:
    """已处理Issue的记录（防重复处理）"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {"issues": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("processed store %s unreadable, starting fresh", self.path)
            self._data = {"issues": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def seen(self, issue_number: int) -> bool:
        """该Issue是否已处理过"""
        return str(issue_number) in self._data.get("issues", {})

    def record(self, issue_number: int) -> dict[str, Any]:
        """获取某个Issue的处理记录"""
        return self._data.get("issues", {}).get(str(issue_number), {})

    def mark(self, issue_number: int, status: str, detail: str = "", **extra: Any) -> None:
        """记录处理结果（extra 可带 pr / branch 等附加字段）"""
        entry = {
            "status": status,
            "detail": detail,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        entry.update(extra)
        self._data.setdefault("issues", {})[str(issue_number)] = entry
        self._save()

    def pending_pull_requests(self) -> list[tuple[int, dict[str, Any]]]:
        """返回「已创建PR但还没确认合并同步」的 (issue编号, 记录) 列表"""
        pending: list[tuple[int, dict[str, Any]]] = []
        for key, entry in self._data.get("issues", {}).items():
            if entry.get("status") != "published" or entry.get("synced"):
                continue
            try:
                pending.append((int(key), entry))
            except (TypeError, ValueError):
                continue
        return sorted(pending)

    def mark_synced(self, issue_number: int, detail: str = "") -> None:
        """标记该Issue对应的PR已完成同步（合并或关闭）"""
        entry = self._data.setdefault("issues", {}).setdefault(str(issue_number), {})
        entry["synced"] = True
        entry["synced_time"] = datetime.now().isoformat(timespec="seconds")
        if detail:
            entry["sync_detail"] = detail
        self._save()

    def update(self, issue_number: int, **fields: Any) -> None:
        """就地把字段合并进已有记录（保留原有字段）"""
        entry = self._data.setdefault("issues", {}).setdefault(str(issue_number), {})
        entry.update(fields)
        self._save()


@dataclass
class IssueMonitor:
    """Issue监控器"""

    client: GitHubClient
    labels: list[str] = field(default_factory=list)
    ignore_labels: list[str] = field(default_factory=list)
    store: ProcessedStore | None = None
    # 每轮最多处理几个 issue（0 表示不限制）：复盘流水线会积压十几个分析，
    # 全跑完要几十分钟，而结论只看最近几场
    max_candidates: int = DEFAULT_MAX_ISSUES

    @classmethod
    def from_config(cls, config: dict[str, Any], client: GitHubClient) -> "IssueMonitor":
        github = config.get("github") or {}
        automation = config.get("automation") or {}
        state_file = automation.get("state_file") or ".processed_issues.json"
        return cls(
            client=client,
            labels=list(github.get("issue_labels") or []),
            ignore_labels=list(github.get("ignore_labels") or []),
            store=ProcessedStore(state_file),
            max_candidates=int(
                automation.get("max_issues_per_round", DEFAULT_MAX_ISSUES)
            ),
        )

    def should_process(self, issue: Issue) -> bool:
        """判断Issue是否需要处理"""
        if issue.state != "open":
            return False
        if self.store is not None and self.store.seen(issue.number):
            return False
        if self.ignore_labels and set(issue.labels) & set(self.ignore_labels):
            return False
        # 未配置标签过滤时，所有开放的Issue都处理
        if not self.labels:
            return True
        return bool(set(issue.labels) & set(self.labels))

    def fetch_candidates(self) -> list[Issue]:
        """拉取需要处理的Issue列表（窗口固定为最新的 `max_candidates` 个开放Issue）

        注意: GitHub API 的 `labels` 参数是“与”语义（要求Issue同时带全部标签），
        而配置中的 issue_labels 是“或”语义（命中任意一个即处理），
        因此这里拉取全部开放Issue，再在本地过滤。

        顺序（先截断、后过滤）：**先按编号从新到旧固定取最新的那几个**
        （`automation.max_issues_per_round`，默认 3），再在它们中间做
        "已处理/命中忽略标签"的过滤。窗口不滑动：这 3 个处理完之后本轮就没有
        候选，不会顺延到次新的一批，更早的历史 issue 也不会被翻出来处理。
        """
        try:
            issues = self.client.list_issues()
        except Exception:
            LOGGER.exception("failed to list issues")
            return []

        # 新的在前：编号越大越新（同一批 issue 的编号单调递增）
        issues.sort(key=lambda issue: issue.number, reverse=True)
        if self.max_candidates > 0 and len(issues) > self.max_candidates:
            LOGGER.info("窗口固定为最新的 %d 个 issue，其余 %d 个不处理",
                        self.max_candidates, len(issues) - self.max_candidates)
            issues = issues[: self.max_candidates]

        candidates = [issue for issue in issues if self.should_process(issue)]
        for issue in candidates:
            LOGGER.info("candidate: %s", issue.summary)
        LOGGER.info("fetched %d issues, %d to process",
                    len(issues), len(candidates))
        return candidates

    def mark_processed(self, issue_number: int, status: str, detail: str = "",
                       **extra: Any) -> None:
        """记录Issue已处理"""
        if self.store is not None:
            self.store.mark(issue_number, status, detail, **extra)

    def record(self, issue_number: int) -> dict[str, Any]:
        """获取Issue的处理记录"""
        return self.store.record(issue_number) if self.store is not None else {}

    def pending_pull_requests(self) -> list[tuple[int, dict[str, Any]]]:
        """已创建PR但尚未确认合并同步的Issue列表"""
        return self.store.pending_pull_requests() if self.store is not None else []

    def mark_synced(self, issue_number: int, detail: str = "") -> None:
        """标记PR已同步"""
        if self.store is not None:
            self.store.mark_synced(issue_number, detail)

    def update_record(self, issue_number: int, **fields: Any) -> None:
        """更新Issue处理记录的附加字段（如自动合并尝试次数）"""
        if self.store is not None:
            self.store.update(issue_number, **fields)
