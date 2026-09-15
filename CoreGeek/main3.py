#!/usr/bin/env python3
"""程序入口：解析参数，配置日志，启动HTTP服务器。

对应设计文档V2 §3.1。

判题系统的调用方式是 `bash run.sh <port>`（`run.sh` 负责挑解释器再 exec 到这里）。
本文件只做四件事，**不含任何策略逻辑**：

1. 校验端口参数
2. 切工作目录到本文件所在目录，并把 `src/` 加进 `sys.path`
3. 把 stdout / stderr 统一成 UTF-8（中文日志不能在判题器侧变成乱码）
4. 配置双通道日志：INFO 进 stdout（判题器会收走），DEBUG 进 `debug.log`

`debug.log` 是复盘流水线（`automation/`）唯一的输入，格式见
`tools/analyze_log.py` 的模块文档。
"""

import logging
import os
import sys
from pathlib import Path


def main() -> None:
    # 1. 参数校验
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main3.py <port>")

    port = int(sys.argv[1])

    # 2. 设置工作目录
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    # 3. 统一使用UTF-8输出，避免中文乱码
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # 4. 配置日志
    # INFO级别输出到stdout（判题器收走的信息），DEBUG级别完整进 debug.log
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    debug_handler = logging.FileHandler("debug.log", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler, debug_handler])

    # 5. 启动服务器（绑定成功后的 "listening" 日志由 server.serve 打，
    #    这里不重复打一遍——重复的行会让复盘工具把一次启动数成两次）
    from agent.server import serve

    serve(port)


if __name__ == "__main__":
    main()

