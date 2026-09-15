"""沙盒子系统：自进化任务的观测、计划、执行与反馈。

对应设计文档V2 第 6 章。

导出的唯一入口是 `solver.TaskSolver.plan()`：给定世界与本回合的沙盒观测，
返回一个 `TaskPlan`（该不该接任务、要不要下发 executeCmd、要不要提交答案）。
"""

from .memory import Memory, TaskRun, RunState
from .sandbox import SandboxOutput, parse_output, fingerprint
from .solver import TaskPlan, TaskSolver, Action

__all__ = [
    "Memory",
    "TaskRun",
    "RunState",
    "SandboxOutput",
    "parse_output",
    "fingerprint",
    "TaskPlan",
    "TaskSolver",
    "Action",
]
