"""关掉所有普通页面标签，只留一个空白页。

    python tools/cdp_clean.py

CDP 驱动脚本（cdp_eval / cdp_shot / cdp_emulate）每次都会新开一个标签页。
一个页面就挂着整个 React 应用和一套 WebGL 风景层，开着三四个测试页
内存就上去了 —— 所以每轮测试之前先清一遍，只留一个。
"""

from __future__ import annotations

import json
import urllib.request

CDP_HTTP = "http://127.0.0.1:9222"


def main() -> None:
    with urllib.request.urlopen(f"{CDP_HTTP}/json/list") as resp:
        targets = json.loads(resp.read())

    pages = [t for t in targets if t.get("type") == "page"]
    closed = 0
    # 留最新的那个，关掉其余的。现在 cdp_* 脚本自己就会复用/清理，
    # 这个脚本只在手动收拾残留时用得上。
    for t in pages[:-1]:
        req = urllib.request.Request(f"{CDP_HTTP}/json/close/{t['id']}")
        try:
            urllib.request.urlopen(req).read()
            closed += 1
        except Exception as exc:  # noqa: BLE001 - 关不掉不该让测试流程中断
            print(f"skip {t['id']}: {exc}")

    print(f"closed {closed}, kept {min(len(pages), 1)}")


if __name__ == "__main__":
    main()
