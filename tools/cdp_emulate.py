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

from _cdp import connect


async def main() -> None:
    url, width, height, expr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    wait_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 1500

    async with connect(url) as s:
        # 视口必须在导航**之后**再设一次。Page.navigate 会重建渲染上下文，
        # 之前设的 metrics 会被丢掉 —— 那样量出来的是真实窗口尺寸（1440），
        # 而媒体查询仍然按桌面走，于是"窄屏验证"实际什么都没验到。
        await s.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                     deviceScaleFactor=1, mobile=False)
        await s.navigate(url)
        await s.wait_ready()
        await asyncio.sleep(wait_ms / 1000)
        value = await s.eval(expr)
        print(json.dumps(value, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
