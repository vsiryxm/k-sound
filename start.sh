#!/bin/bash
# k-sound · 一键启动（本机 localhost:8300）
set -e
cd "$(dirname "$0")"
mkdir -p jobs

# 服务就绪后自动打开浏览器（后台子进程，不阻塞 uvicorn）
( sleep 1.5; open "http://127.0.0.1:8300" 2>/dev/null || true ) &

exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8300
