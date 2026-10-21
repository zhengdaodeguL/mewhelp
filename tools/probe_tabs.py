"""验证「灵动式选项卡」的未读气泡到底会不会亮。

路径：起 CDP 会话打开应用 → 切到「图谱」标签（此刻"工作区"不在场）
     → 在页面里发一轮真问答（等价于用户打字，走前端真正的 SSE 客户端）
     → 轮询 DOM，看 .tab--notify / .tab-bubble 有没有出现。

气泡依赖**流式**路径（卡片随 SSE 逐个到达），所以必须走真问答，
不能直接改 React state。
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.parse
import urllib.request

import websockets

CDP_HTTP = "http://127.0.0.1:9222"
APP = "http://127.0.0.1:8642/"
QUESTION = sys.argv[1] if len(sys.argv) > 1 else "帮我查下订单 ORD-1002 到哪了"

ASK_JS = """
(async () => {
  const ta = document.querySelector('textarea');
  if (!ta) return 'NO_TEXTAREA';
  const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype, 'value').set;
  setter.call(ta, QUESTION_TEXT);
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise(r => setTimeout(r, 150));
  const form = ta.closest('form');
  if (form) { form.requestSubmit(); return 'submitted-form'; }
  const btn = [...document.querySelectorAll('button')]
      .find(b => b.textContent.includes('发送'));
  if (btn) { btn.click(); return 'clicked-send'; }
  ta.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  return 'dispatched-enter';
})()
"""

SNAP_JS = """
JSON.stringify({
  notifyIds: [...document.querySelectorAll('.tab--notify')].map(e => e.id),
  bubbles: [...document.querySelectorAll('.tab-bubble')]
      .map(e => e.parentElement.id + '=' + e.textContent),
  tabs: [...document.querySelectorAll('.tab')].map(e => e.innerText.replace(/\\n/g, '/')),
  streaming: !!document.querySelector('.bubble--streaming'),
})
"""


def new_tab(url: str) -> str:
    req = urllib.request.Request(
        f"{CDP_HTTP}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["webSocketDebuggerUrl"]


async def main() -> None:
    async with websockets.connect(new_tab(APP), max_size=64 * 1024 * 1024) as ws:
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

        async def ev(expr):
            r = await call(
                "Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True
            )
            res = r.get("result", {})
            if res.get("subtype") == "error" or "description" in res:
                print("   [EXC]", res.get("description"))
            return res.get("value")

        await call("Page.enable")
        await call("Runtime.enable")
        await asyncio.sleep(2.0)

        print("0) 起始标签：", await ev("document.querySelector('.tabs').innerText.replace(/\\n/g,' ')"))

        await ev("document.querySelector('#tab-graph').click()")
        await asyncio.sleep(2.2)
        print("1) 切到图谱：", (await ev(SNAP_JS)))

        print("2) 提问：", await ev(ASK_JS.replace("QUESTION_TEXT", json.dumps(QUESTION))))

        for i in range(1, 11):
            await asyncio.sleep(1.0)
            snap = await ev(SNAP_JS)
            print(f"   t+{i}s {snap}")
            data = json.loads(snap) if snap else {}
            if data.get("notifyIds") or (not data.get("streaming") and i >= 4):
                break

        await ev("document.querySelector('#tab-cards').click()")
        await asyncio.sleep(0.8)
        print("3) 切回工作区：", await ev(SNAP_JS))

        await asyncio.sleep(0.4)
        print("   工作区卡片数：", await ev(
            "document.querySelectorAll('.card-stack .card').length"))


if __name__ == "__main__":
    asyncio.run(main())
