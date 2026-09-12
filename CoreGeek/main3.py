#!/usr/bin/env python3
"""程序入口：解析参数，配置日志，启动HTTP服务器。

对应设计文档 3.1 节。
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
    # INFO级别输出到stdout（简要信息）
    # DEBUG级别输出到文件（完整请求响应）
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    # 添加文件处理器记录DEBUG日志
    debug_handler = logging.FileHandler("debug.log", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler, debug_handler])

    # 5. 启动服务器
    from agent.server import serve

    logging.info("listening on 0.0.0.0:%d", port)
    serve(port)


if __name__ == "__main__":
    main()
