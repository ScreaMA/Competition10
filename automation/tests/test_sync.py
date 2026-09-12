"""PR合并后同步本地代码的功能测试。

覆盖:
    - PR链接解析、状态记录的 pending/mark_synced 语义
    - Automation.sync_repository 对“已合并/未合并/已关闭/同步失败”四种PR的处理
    - GitPusher.sync_main 在真实git仓库上的快进、脏工作区、本地领先三种情形
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from automation_main import Automation, pull_number
from git_pusher import GitPusher
from issue_monitor import ProcessedStore

SYNCED = "synced"


# === 纯逻辑 ===


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/ScreaMA/Competition10/pull/3", 3),
        ("https://github.com/o/r/pull/123/files", 123),
        ("", None),
        ("https://github.com/o/r/issues/3", None),
    ],
)
def test_pull_number(url, expected):
    """PR链接解析"""
    assert pull_number(url) == expected


def test_pending_pull_requests(tmp_path):
    """只有 published 且未同步的记录才算待跟踪"""
    store = ProcessedStore(tmp_path / "state.json")
    store.mark(1, "no_change", "")
    store.mark(2, "published", "https://github.com/o/r/pull/3", pr=3, branch="b2")
    store.mark(3, "published", "https://github.com/o/r/pull/4", pr=4, branch="b3")
    store.mark_synced(3, "PR #4 merged")

    pending = store.pending_pull_requests()
    assert [issue for issue, _ in pending] == [2]
    assert pending[0][1]["pr"] == 3
    assert store.record(3).get(SYNCED) is True


def test_pending_pull_requests_survives_reload(tmp_path):
    """状态落盘后可再次读取（轮询是跨进程/跨轮次的）"""
    path = tmp_path / "state.json"
    ProcessedStore(path).mark(2, "published", "https://github.com/o/r/pull/3", pr=3)

    assert [issue for issue, _ in ProcessedStore(path).pending_pull_requests()] == [2]


# === sync_repository 的四种PR状态 ===


class FakeClient:
    def __init__(self, pulls: dict[int, dict]) -> None:
        self.pulls = pulls
        self.queried: list[int] = []

    def get_pull_request(self, number: int) -> dict:
        self.queried.append(number)
        return self.pulls.get(number, {})

    def default_branch(self) -> str:
        return "main"


class FakePusher:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[str, str, bool, str]] = []

    def sync_main(self, main_branch: str, merged_branch: str = "",
                  delete_remote_branch: bool = False,
                  strategy: str = "rebase") -> bool:
        self.calls.append((main_branch, merged_branch, delete_remote_branch, strategy))
        return self.result


class FakeMonitor:
    def __init__(self, pending: list[tuple[int, dict]]) -> None:
        self._pending = pending
        self.synced: list[tuple[int, str]] = []

    def pending_pull_requests(self):
        return list(self._pending)

    def mark_synced(self, issue_number: int, detail: str = "") -> None:
        self.synced.append((issue_number, detail))


def _automation(pending, pulls, pusher_result=True, auto_sync=True,
                delete_remote=False) -> tuple[Automation, FakePusher, FakeMonitor, FakeClient]:
    """构造只装配了同步所需依赖的 Automation（不读配置、不联网）"""
    automation = object.__new__(Automation)
    pusher = FakePusher(pusher_result)
    monitor = FakeMonitor(pending)
    client = FakeClient(pulls)
    automation.pusher = pusher
    automation.monitor = monitor
    automation.client = client
    automation.auto_sync = auto_sync
    automation.delete_remote_branch = delete_remote
    automation.sync_strategy = "rebase"
    automation._main_branch = None
    return automation, pusher, monitor, client


def test_sync_when_pr_merged():
    """PR已合并 -> 快进同步并标记完成"""
    pending = [(2, {"pr": 3, "branch": "auto-fix/issue-2-x",
                    "detail": "https://github.com/o/r/pull/3"})]
    automation, pusher, monitor, _ = _automation(
        pending, {3: {"merged": True, "state": "closed",
                      "head": {"ref": "auto-fix/issue-2-x"}}},
    )

    assert automation.sync_repository() == 1
    assert pusher.calls == [("main", "auto-fix/issue-2-x", False, "rebase")]
    assert monitor.synced == [(2, "PR #3 merged")]


def test_sync_retries_when_fast_forward_blocked():
    """已合并但本地不能快进（脏工作区/分叉）-> 不标记完成，下轮重试"""
    pending = [(2, {"pr": 3, "branch": "b"})]
    automation, pusher, monitor, _ = _automation(
        pending, {3: {"merged": True, "state": "closed", "head": {"ref": "b"}}},
        pusher_result=False,
    )

    assert automation.sync_repository() == 0
    assert pusher.calls  # 尝试过
    assert monitor.synced == []


def test_open_pr_is_left_alone():
    """PR仍在审核 -> 不做任何动作"""
    pending = [(2, {"pr": 3, "branch": "b"})]
    automation, pusher, monitor, client = _automation(
        pending, {3: {"merged": False, "state": "open", "head": {"ref": "b"}}},
    )

    assert automation.sync_repository() == 0
    assert pusher.calls == []
    assert monitor.synced == []
    assert client.queried == [3]


def test_closed_without_merge_stops_tracking():
    """PR被关闭但未合并 -> 停止跟踪，但不同步代码"""
    pending = [(2, {"pr": 3, "branch": "b"})]
    automation, pusher, monitor, _ = _automation(
        pending, {3: {"merged": False, "state": "closed", "head": {"ref": "b"}}},
    )

    assert automation.sync_repository() == 0
    assert pusher.calls == []
    assert monitor.synced == [(2, "PR #3 closed")]


def test_pr_number_falls_back_to_url():
    """状态里没有pr编号时，从PR链接中解析"""
    pending = [(2, {"detail": "https://github.com/o/r/pull/7"})]
    automation, pusher, monitor, client = _automation(
        pending, {7: {"merged": True, "state": "closed", "head": {"ref": "b7"}}},
    )

    assert automation.sync_repository() == 1
    assert client.queried == [7]
    assert pusher.calls[0][1] == "b7"


def test_sync_can_be_disabled():
    """auto_sync=false 时完全不动本地代码"""
    pending = [(2, {"pr": 3, "branch": "b"})]
    automation, pusher, monitor, client = _automation(
        pending, {3: {"merged": True, "state": "closed"}}, auto_sync=False,
    )

    assert automation.sync_repository() == 0
    assert pusher.calls == [] and client.queried == []


# === 真实 git 仓库上的 sync_main ===


def _git(cwd: Path, *args: str) -> str:
    """在指定目录执行git命令（带上测试用的提交者身份）"""
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    })
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True, env=env,
    )
    return (completed.stdout or "").strip()


@pytest.fixture
def repo_pair(tmp_path):
    """构造 origin(bare) + 本地克隆，本地已经在 main 上落后远端一个提交"""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "--initial-branch=main")

    local = tmp_path / "local"
    local.mkdir()
    _git(local, "init", "--initial-branch=main")
    (local / "code.txt").write_text("v1\n", encoding="utf-8")
    _git(local, "add", "-A")
    _git(local, "commit", "-m", "init")
    _git(local, "remote", "add", "origin", str(origin))
    _git(local, "push", "-u", "origin", "main")

    # 另开一份克隆模拟“别人把PR合并到了远端”
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(origin), str(other))
    (other / "code.txt").write_text("v2\n", encoding="utf-8")
    _git(other, "commit", "-am", "remote update")
    _git(other, "push", "origin", "main")
    return local


def test_sync_main_fast_forwards(repo_pair):
    """远端有更新时快进本地主干"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)

    assert pusher.sync_main("main") is True
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "v2\n"
    assert pusher.current_branch() == "main"


def test_sync_main_is_noop_when_up_to_date(repo_pair):
    """已经是最新时返回True但不产生变化"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    pusher.sync_main("main")
    pusher.sync_main("main")  # 第二次没有新提交
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "v2\n"


def test_sync_main_skips_dirty_worktree(repo_pair):
    """工作区有未提交改动时跳过，绝不覆盖"""
    (repo_pair / "code.txt").write_text("local edit\n", encoding="utf-8")
    pusher = GitPusher(client=None, repo_dir=repo_pair)

    assert pusher.sync_main("main") is False
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "local edit\n"


def test_sync_main_ff_only_refuses_divergence(repo_pair):
    """strategy=ff-only：本地领先远端时保持现状"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    (repo_pair / "local.txt").write_text("local ahead\n", encoding="utf-8")
    _git(repo_pair, "add", "-A")
    _git(repo_pair, "commit", "-m", "local ahead")

    assert pusher.sync_main("main", strategy="ff-only") is False
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "v1\n"


def test_sync_main_rebases_local_commits(repo_pair):
    """默认rebase：本地提交重放到远端之上，两边改动都保留且历史线性"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    (repo_pair / "local.txt").write_text("local\n", encoding="utf-8")
    _git(repo_pair, "add", "-A")
    _git(repo_pair, "commit", "-m", "local commit")

    assert pusher.sync_main("main") is True
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "v2\n"  # 远端改动
    assert (repo_pair / "local.txt").read_text(encoding="utf-8") == "local\n"

    log = _git(repo_pair, "log", "--oneline").splitlines()
    assert "local commit" in log[0]      # 本地提交在最上面
    assert "remote update" in log[1]     # 下面紧跟着远端提交
    assert pusher.has_changes() is False


def test_sync_main_merge_strategy_creates_merge_commit(repo_pair):
    """strategy=merge：生成合并提交，两边改动都保留"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    (repo_pair / "local.txt").write_text("local\n", encoding="utf-8")
    _git(repo_pair, "add", "-A")
    _git(repo_pair, "commit", "-m", "local commit")

    assert pusher.sync_main("main", strategy="merge") is True
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "v2\n"
    assert (repo_pair / "local.txt").read_text(encoding="utf-8") == "local\n"
    assert _git(repo_pair, "log", "--merges", "--oneline")


def test_sync_main_aborts_on_conflict(repo_pair):
    """rebase冲突：回滚到原状态，不留下半成品"""
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    (repo_pair / "code.txt").write_text("local version\n", encoding="utf-8")
    _git(repo_pair, "add", "-A")
    _git(repo_pair, "commit", "-m", "conflicting commit")
    head_before = _git(repo_pair, "rev-parse", "HEAD")

    assert pusher.sync_main("main") is False
    assert _git(repo_pair, "rev-parse", "HEAD") == head_before
    assert (repo_pair / "code.txt").read_text(encoding="utf-8") == "local version\n"
    # rebase 状态已清理，可以直接继续工作
    assert not (repo_pair / ".git" / "rebase-merge").exists()
    assert not (repo_pair / ".git" / "rebase-apply").exists()


def test_sync_main_cleans_only_merged_local_branch(repo_pair):
    """已合并的工作分支被清理，未合并的保留（用 branch -d 而非 -D）"""
    _git(repo_pair, "branch", "merged-work")

    _git(repo_pair, "checkout", "-b", "unmerged-work")
    (repo_pair / "wip.txt").write_text("wip\n", encoding="utf-8")
    _git(repo_pair, "add", "-A")
    _git(repo_pair, "commit", "-m", "unmerged work")
    _git(repo_pair, "checkout", "main")

    pusher = GitPusher(client=None, repo_dir=repo_pair)
    assert pusher.sync_main("main", "merged-work") is True
    assert pusher.sync_main("main", "unmerged-work") is True

    branches = _git(repo_pair, "branch", "--format=%(refname:short)").split()
    assert "merged-work" not in branches
    assert "unmerged-work" in branches


def test_sync_main_cleans_branch_when_already_up_to_date(repo_pair):
    """主干已是最新时也要清理传入的已合并分支

    回归：连续两个PR都合并时，处理第二个PR时主干可能已经是最新（第一个PR
    同步时就拉过了），早期实现在“已是最新”分支提前返回，导致该PR的工作
    分支永远不被清理。
    """
    pusher = GitPusher(client=None, repo_dir=repo_pair)
    assert pusher.sync_main("main") is True       # 先把主干拉到远端最新
    _git(repo_pair, "branch", "merged-later")     # 在最新点上再开一个已合并分支

    assert pusher.sync_main("main", "merged-later") is True

    branches = _git(repo_pair, "branch", "--format=%(refname:short)").split()
    assert "merged-later" not in branches
