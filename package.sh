#!/bin/bash
# package.sh - 打包脚本（Linux/Mac）

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 打包文件名
PACKAGE_NAME="CoreGeek_${TIMESTAMP}.tar.gz"

# 清理临时文件
cd CoreGeek
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -f debug.log

# 打包
cd ..
tar -czf "$PACKAGE_NAME" CoreGeek/

echo "Package created: $PACKAGE_NAME"
ls -lh "$PACKAGE_NAME"
