"""《未来战争》参赛客户端 agent 包。

模块划分（对应设计文档V2 §3.1）:

    protocol.py   规则常量表、报文解析、指令构造（无策略）
    grid.py       几何与寻路：切比雪夫距离、A*、可达性、锥形判定
    world.py      世界模型：基地原点、敌方来向、可建造区在线学习
    telemetry.py  回合遥测：逐回合增量（击杀/掉血/损失/空转），只服务日志
    brain.py      编排器：decide(payload) -> response
    server.py     HTTP 服务 + 决策超时保护
    strategy/
        economy.py  采集、贩卖、建造执行、金币分配
        defense.py  塔位与墙线规划、夜晚火力控制
        task/       自进化任务子系统（沙盒、技能库、状态机）

约束：**零第三方依赖**，只用标准库。判题与沙盒环境都不能假设有 pip。
"""
