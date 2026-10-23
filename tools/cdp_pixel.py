"""像素级核对：截图 + 元素坐标 + 计算样式，一次性拿全。

    CDP_SETUP_JS='...' python tools/cdp_pixel.py <url> <选择器,...> <out.png> [等待毫秒]

为什么要这个：视觉模型对"这块是白底还是灰底"这类判断并不可靠（实测把
rgb(237,240,242) 的灰 pill 说成白色矩形）。颜色这种事应该直接量像素。

输出两段：
  · 每个选择器的 rect（CSS px，相对视口）与计算样式
  · 截图（deviceScaleFactor=1，所以像素坐标 == CSS 坐标）
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys

from _cdp import connect

PROPS = ("backgroundColor", "color", "borderTopWidth", "borderTopColor", "borderRadius",
         "padding", "fontSize", "lineHeight", "boxShadow")


async def main() -> None:
    url = sys.argv[1]
    sels = [s for s in sys.argv[2].split(",") if s]
    out = sys.argv[3]
    wait_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 1200

    async with connect(url) as s:
        await s.call("Emulation.setDeviceMetricsOverride", width=1440, height=900,
                     deviceScaleFactor=1, mobile=False)
        await s.navigate(url)
        await s.wait_ready()
        setup = os.environ.get("CDP_SETUP_JS", "").strip()
        if setup:
            print("[setup]", await s.eval(setup))
        await asyncio.sleep(wait_ms / 1000)

        expr = f"""(() => {{
            const props = {json.dumps(list(PROPS))};
            const out = {{}};
            for (const sel of {json.dumps(sels)}) {{
                const el = document.querySelector(sel);
                if (!el) {{ out[sel] = 'MISSING'; continue; }}
                const b = el.getBoundingClientRect();
                const cs = getComputedStyle(el);
                const style = {{}};
                for (const p of props) style[p] = cs[p];
                out[sel] = {{
                    rect: [Math.round(b.x), Math.round(b.y), Math.round(b.width), Math.round(b.height)],
                    style,
                    text: (el.textContent || '').trim().slice(0, 40),
                }};
            }}
            return JSON.stringify(out);
        }})()"""
        print(json.dumps(json.loads(await s.eval(expr)), ensure_ascii=False, indent=1))

        shot = await s.call("Page.captureScreenshot", format="png")
        with open(out, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        print(f"[saved] {out}")


if __name__ == "__main__":
    asyncio.run(main())
