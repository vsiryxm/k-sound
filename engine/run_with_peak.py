#!/usr/bin/env python3
"""k-sound · 子命令峰值内存包装器

用法：python run_with_peak.py <cmd> [args...]
作为父进程运行任意命令（如 demucs），结束后打印其峰值 RSS：
    [peak_rss] <MB>
（macOS 的 ru_maxrss 单位为字节；仅测量本包装进程的直接子进程，天然隔离。）
"""
import resource
import subprocess
import sys

r = subprocess.run(sys.argv[1:])
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"[peak_rss] {peak / (1024 * 1024):.0f} MB", flush=True)
sys.exit(r.returncode)
