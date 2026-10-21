"""在已开的 Chrome 里跑一段 JS 并打印结果。用于验证前端行为。

    python tools/cdp_eval.py <url> <js 表达式> [等待毫秒]
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
    url = sys.argv[1]
    expr = sys.argv[2]
    wait_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

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

        await call("Page.enable")
        await call("Runtime.enable")
        await asyncio.sleep(wait_ms / 1000)
        r = await call(
            "Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True
        )
        out = r.get("result", {})
        print(json.dumps(out.get("value"), ensure_ascii=False, indent=1))
        if out.get("description"):
            print("EXCEPTION:", out.get("description"))


if __name__ == "__main__":
    asyncio.run(main())
