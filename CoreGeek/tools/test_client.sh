#!/bin/bash
# test_client.sh - 快速联调：启动客户端 → 发送真实报文 → 校验响应
# 用法: bash CoreGeek/tools/test_client.sh [port]

set -u

PORT="${1:-8000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLIENT_DIR="$ROOT/CoreGeek"
SAMPLE="$ROOT/docs/request.txt"
RESPONSE_FILE="$ROOT/response.json"
LOG_FILE="$ROOT/test_client.log"

if [ ! -f "$SAMPLE" ]; then
    echo "Error: 缺少测试报文 $SAMPLE" >&2
    exit 1
fi

if ! command -v curl > /dev/null 2>&1; then
    echo "Error: 需要 curl 才能运行本脚本" >&2
    exit 1
fi

# 1. 启动客户端（后台）
cd "$CLIENT_DIR" || exit 1
bash run.sh "$PORT" > "$LOG_FILE" 2>&1 &
PID=$!
trap 'kill "$PID" 2> /dev/null' EXIT

# 2. 等待启动
sleep 3

if ! kill -0 "$PID" 2> /dev/null; then
    echo "❌ 客户端启动失败，日志如下：" >&2
    tail -20 "$LOG_FILE" >&2
    exit 1
fi

# 3. 发送测试请求（判题系统实际发往 POST /，这里两种路径都验证）
echo "--- POST / ---"
curl -s -X POST "http://localhost:$PORT/" \
    -H "Content-Type: application/json" \
    -d @"$SAMPLE" -o "$RESPONSE_FILE" -w "http_status=%{http_code}\n"

echo "--- POST /action ---"
curl -s -X POST "http://localhost:$PORT/action" \
    -H "Content-Type: application/json" \
    -d @"$SAMPLE" -o /dev/null -w "http_status=%{http_code}\n"

# 4. 检查响应
if [ -f "$RESPONSE_FILE" ] && [ -s "$RESPONSE_FILE" ]; then
    echo "✅ 收到响应:"
    if command -v python3 > /dev/null 2>&1; then
        python3 -m json.tool "$RESPONSE_FILE"
    else
        cat "$RESPONSE_FILE"
        echo
    fi
else
    echo "❌ 未收到响应" >&2
fi

echo "--- 客户端日志（末尾20行） ---"
tail -20 "$LOG_FILE"
