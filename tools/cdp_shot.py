"""用 CDP 驱动已开的 Chrome：打开页面 → 点若干元素 → 抓面板文字 + 截图。

给"验证前端改动"用：这台机器上已经有真的 Chrome，不必再下 playwright
的浏览器；websockets 已在环境里，不用装新库。

用法：
    python tools/cdp_shot.py <url> <逗号分隔的选择器> <输出前缀> [每次点击后等待毫秒]
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import urllib.parse
import urllib.request

import websockets

CDP_HTTP = "http://127.0.0.1:9222"


def new_tab(url: str) -> str:
    """开一个新标签页，返回它的 WebSocket 调试地址。"""
    req = urllib.request.Request(
        f"{CDP_HTTP}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["webSocketDebuggerUrl"]


class Session:
    """极简 CDP 会话：发一条命令、等它对应的那条回包。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.seq = 0

    async def call(self, method: str, **params):
        self.seq += 1
        mid = self.seq
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def eval(self, expr: str):
        r = await self.call(
            "Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True
        )
        return r.get("result", {}).get("value")


async def main() -> None:
    url = sys.argv[1]
    selectors = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
    prefix = sys.argv[3] if len(sys.argv) > 3 else "shot"
    wait_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

    async with websockets.connect(new_tab(url), max_size=64 * 1024 * 1024) as ws:
        s = Session(ws)
        await s.call("Page.enable")
        await s.call("Runtime.enable")
        await asyncio.sleep(wait_ms / 1000)

        for sel in selectors:
            js = f"""(() => {{
                const el = document.querySelector({json.dumps(sel)});
                if (!el) return 'MISSING ' + {json.dumps(sel)};
                el.click();
                return 'clicked ' + el.textContent.trim().slice(0, 24);
            }})()"""
            print(f"[click] {sel} -> {await s.eval(js)}")
            await asyncio.sleep(wait_ms / 1000)

        for name, expr in (
            ("panel", "document.querySelector('.workspace-body')?.innerText ?? ''"),
            ("tabs", "document.querySelector('.tabs')?.innerText ?? ''"),
            ("bubbles", "[...document.querySelectorAll('.tab-bubble')].map(e=>e.textContent).join(',')"),
            ("notify", "[...document.querySelectorAll('.tab--notify')].map(e=>e.textContent.trim()).join('|')"),
        ):
            print(f"[{name}] {await s.eval(expr)!r}")

        shot = await s.call("Page.captureScreenshot", format="png")
        path = f"{prefix}.png"
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        print(f"[saved] {path}")


if __name__ == "__main__":
    asyncio.run(main())
