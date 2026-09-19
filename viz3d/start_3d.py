#!/usr/bin/env python
"""启动本地服务器并打开三维可视化页面。

为什么需要本地服务器
--------------------
CesiumJS 用 Web Worker 做地形与三维瓦片的解析，而 Worker 是通过
``blob:`` URL 创建的。浏览器的同源策略禁止 ``file://`` 页面创建可用的
blob Worker，于是会抛出::

    Uncaught NetworkError: Failed to execute 'importScripts' on
    'WorkerGlobalScope': The script at 'blob:null/...' failed to load.

结果是页面永远停在加载动画上——**双击 HTML 打开是不行的**，这不是页面
的 bug，是 Cesium 的固有限制（几乎所有基于 WebGL + Worker 的库都如此）。

因此需要一个 HTTP 服务。本脚本起一个最小化的本地静态服务器（只服务
``viz3d/`` 目录，仅监听 127.0.0.1，不对外暴露），然后自动打开浏览器。

用法::

    python viz3d/start_3d.py
    python viz3d/start_3d.py --port 8123 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = "chengdu_evtol_3d.html"


class _Handler(http.server.SimpleHTTPRequestHandler):
    """静态文件处理器：静音日志 + 禁用缓存（便于反复重新导出数据）。"""

    def log_message(self, fmt, *args):
        # 只记录错误，避免每个资源请求都刷屏
        if args and str(args[1]).startswith(("4", "5")):
            print(f"  [HTTP {args[1]}] {args[0]}")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def _free_port(preferred: int) -> int:
    """优先用指定端口；被占用时自动找一个空闲端口。"""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="启动三维可视化本地服务器")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--no-browser", action="store_true", help="只起服务，不自动打开浏览器")
    args = ap.parse_args(argv)

    missing = [f for f in (PAGE, "evtol_data.js", "cesium_token.js")
               if not (HERE / f).exists()]
    if missing:
        print("缺少以下文件，请先运行 scripts/export_3d.py：")
        for m in missing:
            print("   -", m)
        return 1

    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}/{PAGE}"

    handler = partial(_Handler, directory=str(HERE))
    # 允许端口复用，避免上一次没退干净时启动失败
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)

    print("=" * 66)
    print("  成都市 eVTOL 起降场选址 · 三维可视化")
    print("=" * 66)
    print(f"  地址: {url}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 66)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
