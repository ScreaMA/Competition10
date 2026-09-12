#!/bin/bash
# run.sh - 启动客户端脚本（判题系统调用方式: bash run.sh <port>）
# 用法: bash run.sh <port>
# 说明: 与 Demo 一致，入口为同目录下的 main3.py

if [ $# -ne 1 ]; then
    echo "Usage: bash run.sh <port>" >&2
    exit 1
fi

PORT="$1"

# 端口校验（判题系统分配的端口不做范围限制，只校验是合法端口号）
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: invalid port: $PORT" >&2
    exit 1
fi

# 解析可用的 python 解释器（优先 python3，回退 python）
# 注意: 不能只看命令是否存在（Windows 上 python3 可能只是应用商店的占位程序），
#       必须真正执行一次并校验版本 >= 3.11
PYTHON_BIN=""
for candidate in python3 python; do
    if ! command -v "$candidate" > /dev/null 2>&1; then
        continue
    fi
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
            > /dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
    echo "Warning: $candidate 不可用或版本低于 3.11，尝试下一个解释器" >&2
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3.11 or higher is required (tried: python3, python)." >&2
    exit 1
fi

cd "$(dirname "$0")" || exit 1

# 启动客户端
exec "$PYTHON_BIN" main3.py "$PORT"
