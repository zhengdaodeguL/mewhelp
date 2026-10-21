"""在已开的 Chrome 里模拟一个视口尺寸，然后跑一段 JS。

    python tools/cdp_emulate.py <url> <宽> <高> <js 表达式> [等待毫秒]

和 cdp_eval.py 的区别只有一条：先发 Emulation.setDeviceMetricsOverride。
窄屏问题（抽屉宽度、标签被裁）**必须**用真的视口量 —— 用 CSS 缩放
模拟时 vw 单位、媒体查询、min() 的基准都不一样，量出来的数是假的。
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.parse
import urllib.request

import websockets

CDP_HTTP = "http://127.0.0.1:9222"


def new_tab(url: str) -> str:
    req = urllib.request.Request(
        f"{CDP_HTTP}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["webSocketDebuggerUrl"]


async def main() -> None:
    url, width, height, expr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    wait_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 1500

    async with websockets.connect(new_tab(url), max_size=64 * 1024 * 1024) as ws:
        seq = 0

        async def call(method, **params):
            nonlocal seq
            seq += 1
            mid = seq
            await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                   deviceScaleFactor=1, mobile=False)
        await call("Page.enable")
        await call("Runtime.enable")
        await asyncio.sleep(wait_ms / 1000)
        r = await call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        out = r.get("result", {})
        print(json.dumps(out.get("value"), ensure_ascii=False, indent=1))
        if out.get("description"):
            print("EXCEPTION:", out.get("description"))


if __name__ == "__main__":
    asyncio.run(main())
