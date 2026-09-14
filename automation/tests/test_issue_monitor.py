"""Issue 候选筛选测试：每轮只处理最新的 N 个。"""

from __future__ import annotations

from issue_monitor import DEFAULT_MAX_ISSUES, Issue, IssueMonitor, ProcessedStore


class FakeClient:
    """只实现 list_issues 的假客户端"""

    def __init__(self, issues: list[Issue]) -> None:
        self._issues = issues

    def list_issues(self, labels=None, state: str = "open") -> list[Issue]:
        return list(self._issues)


def _issue(number: int, labels: tuple[str, ...] = ("auto-fix",)) -> Issue:
    return Issue(number=number, title=f"issue {number}", body="", labels=labels)


def _monitor(tmp_path, issues: list[Issue], **kwargs) -> IssueMonitor:
    return IssueMonitor(
        client=FakeClient(issues),
        labels=["auto-fix"],
        store=ProcessedStore(tmp_path / "state.json"),
        **kwargs,
    )


def test_fetch_candidates_keeps_only_newest(tmp_path):
    """只保留最新的 3 个（乱序返回也要按编号从新到旧）"""
    issues = [_issue(number) for number in range(1, 7)]  # #1..#6

    picked = _monitor(tmp_path, list(reversed(issues))).fetch_candidates()

    assert [issue.number for issue in picked] == [6, 5, 4]


def test_fetch_candidates_skips_processed_and_ignored(tmp_path):
    """截断之后再过滤：最新 3 个里被处理过 / 命中忽略标签的都不在候选里"""
    monitor = _monitor(tmp_path, [
        _issue(9),
        _issue(8, labels=("wontfix",)),
        _issue(7),
        _issue(6),  # 排在最新 3 个之外，根本不会被看到
    ])
    monitor.mark_processed(7, "published", "")

    picked = monitor.fetch_candidates()

    assert [issue.number for issue in picked] == [9]


def test_fetch_candidates_never_looks_back(tmp_path):
    """最新 3 个都处理过时本轮没有候选，不往历史回溯（只看最新 3 个）"""
    monitor = _monitor(tmp_path, [_issue(n) for n in range(1, 8)])
    for number in (7, 6, 5):
        monitor.mark_processed(number, "published", "")

    assert monitor.fetch_candidates() == []


def test_fetch_candidates_limit_can_be_disabled(tmp_path):
    """max_candidates=0 表示不限制"""
    monitor = _monitor(tmp_path, [_issue(n) for n in range(1, 6)], max_candidates=0)

    assert len(monitor.fetch_candidates()) == 5


def test_from_config_reads_limit():
    """配置项 max_issues_per_round 生效；没配时默认 3"""
    config = {"github": {}, "automation": {"max_issues_per_round": 2}}
    assert IssueMonitor.from_config(config, client=None).max_candidates == 2

    assert IssueMonitor.from_config(
        {"github": {}}, client=None,
    ).max_candidates == DEFAULT_MAX_ISSUES == 3
