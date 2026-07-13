from __future__ import annotations

import argparse
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLOUDFLARED = PROJECT_ROOT / "tools" / "cloudflared.exe"
PUBLIC_URL_PATTERN = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")


def _wait_for_server(url: str, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError("本地网页服务未能启动")


def _write_public_shortcuts(url: str) -> None:
    (PROJECT_ROOT / "公网访问地址.txt").write_text(url + "\n", encoding="utf-8")
    shortcut = "[InternetShortcut]\n" f"URL={url}\n" "IconFile=%SystemRoot%\\System32\\SHELL32.dll\n" "IconIndex=220\n"
    (PROJECT_ROOT / "打开网页.url").write_text(shortcut, encoding="utf-8")
    (PROJECT_ROOT / "打开公网网页.url").write_text(shortcut, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 HYROX 临时公网分享")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--cloudflared", type=Path, default=DEFAULT_CLOUDFLARED)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--protected", action="store_true", help="为临时分享链接添加随机访问口令")
    parser.add_argument("--access-token", default=os.environ.get("POSE_WEB_ACCESS_TOKEN", ""))
    args = parser.parse_args()

    cloudflared = args.cloudflared.resolve()
    if not cloudflared.exists():
        print(f"未找到 Cloudflare Tunnel 程序：{cloudflared}")
        print("请先按《网页版使用说明》安装 cloudflared。")
        return 2

    token = args.access_token or (secrets.token_urlsafe(18) if args.protected else "")
    origin = f"http://127.0.0.1:{args.port}"
    server_environment = os.environ.copy()
    server_environment["POSE_TRUST_PROXY"] = "1"
    server_environment["POSE_SECURE_COOKIES"] = "1"
    server = subprocess.Popen(
        [
            sys.executable,
            "start_web.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--no-browser",
            *(["--access-token", token] if token else []),
        ],
        cwd=PROJECT_ROOT,
        env=server_environment,
    )
    tunnel: subprocess.Popen[str] | None = None
    try:
        _wait_for_server(f"{origin}/healthz")
        tunnel = subprocess.Popen(
            [str(cloudflared), "tunnel", "--url", origin, "--no-autoupdate"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert tunnel.stdout is not None
        opened = False
        for line in tunnel.stdout:
            print(line, end="")
            match = PUBLIC_URL_PATTERN.search(line)
            if match and not opened:
                public_url = f"{match.group(0)}/"
                if token:
                    public_url += f"?access_token={token}"
                _write_public_shortcuts(public_url)
                print("\n公网分享地址（电脑和手机均可打开）：")
                print(public_url)
                if token:
                    print("链接包含临时访问口令，请勿转发给不信任的人。")
                else:
                    print("当前为匿名访问链接，请只在需要时运行；关闭此窗口即可停止公网访问。")
                print()
                if not args.no_browser:
                    webbrowser.open(public_url)
                opened = True
        return tunnel.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
        if server.poll() is None:
            server.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
