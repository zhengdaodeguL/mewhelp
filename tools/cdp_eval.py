"""在已开的 Chrome 里跑一段 JS 并打印结果。用于验证前端行为。

    python tools/cdp_eval.py <url> <js 表达式> [等待毫秒]
"""

from __future__ import annotations

import asyncio
import json
import sys

from _cdp import connect


async def main() -> None:
    url = sys.argv[1]
    expr = sys.argv[2]
    wait_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

    async with connect(url) as s:
        await asyncio.sleep(wait_ms / 1000)
        value = await s.eval(expr)
        print(json.dumps(value, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
