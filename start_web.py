from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from webui import create_app


def _open_browser_when_ready(url: str, health_url: str, timeout: float = 30.0) -> None:
    """Wait for the local server before opening the browser."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    print(f"网页服务未能在 {timeout:.0f} 秒内就绪，请检查启动窗口中的错误：{url}")


def _lan_address() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 HYROX 姿态分析网页")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--access-token", default=os.environ.get("POSE_WEB_ACCESS_TOKEN", ""), help="分享访问令牌")
    args = parser.parse_args()

    local_url = f"http://127.0.0.1:{args.port}"
    token_query = f"?access_token={args.access_token}" if args.access_token else ""
    browser_url = f"{local_url}/{token_query}"
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(browser_url, f"{local_url}/healthz"),
            daemon=True,
        ).start()
    print(f"本机访问：{browser_url}")
    lan_ip = _lan_address()
    if args.host in {"0.0.0.0", "::"} and lan_ip:
        print(f"局域网访问：http://{lan_ip}:{args.port}/{token_query}")
    print("关闭此窗口即可停止服务。")
    create_app(access_token=args.access_token).run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
