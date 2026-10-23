"""用 CDP 驱动已开的 Chrome：打开页面 → 点若干元素 → 抓面板文字 + 截图。

给"验证前端改动"用：这台机器上已经有真的 Chrome，不必再下 playwright
的浏览器；websockets 已在环境里，不用装新库。

用法：
    python tools/cdp_shot.py <url> <逗号分隔的选择器> <输出前缀> [每次点击后等待毫秒]

    先跑一段准备 JS（比如发一条消息把对话铺开），用环境变量传：
    CDP_SETUP_JS='...' python tools/cdp_shot.py ...

每轮只留一个标签页（见 _cdp.pick）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys

from _cdp import connect


async def main() -> None:
    url = sys.argv[1]
    selectors = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
    prefix = sys.argv[3] if len(sys.argv) > 3 else "shot"
    wait_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

    async with connect(url) as s:
        setup = os.environ.get("CDP_SETUP_JS", "").strip()
        if setup:
            print(f"[setup] {await s.eval(setup)}")
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
