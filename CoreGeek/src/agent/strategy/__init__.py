"""策略层：白天/夜晚的确定性决策与自进化任务链路。

对应设计文档V2 第 6–8 章。

模块边界就是测试边界：`economy` / `defense` / `task.solver` 都能脱离 HTTP
服务器单独构造输入测试，`brain.decide(payload)` 是唯一的集成面。
"""
