"""诊断：为什么在「图谱」标签下发问，工作区卡片数一直是 0。

和 probe_tabs.py 走同一条路径（真页面、真 SSE），但把每一秒的
bubble / streaming / card 数都打出来，并收集页面的 console.error ——
用来区分"后端没推 tool 事件"和"前端收到了却没落成卡片"。
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

SNAP = """JSON.stringify((function(){
  var bubbles = document.querySelectorAll('.bubble');
  var last = bubbles[bubbles.length - 1];
  return {
    streaming: !!document.querySelector('.bubble--streaming'),
    bubbleCount: bubbles.length,
    last: last ? (last.innerText || '').slice(0, 90) : '',
    errs: (window.__errs || []).slice(0, 6),
    cards: document.querySelectorAll('.card-stack .card').length,
    tabs: Array.prototype.map.call(document.querySelectorAll('.tab'), function (e) {
      return (e.innerText || '').split(String.fromCharCode(10)).join('/');
    }),
    notify: Array.prototype.map.call(document.querySelectorAll('.tab--notify'), function (e) {
      return e.id;
    })
  };
})())"""

ASK = """(async () => {
  const ta = document.querySelector('textarea');
  const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype, 'value').set;
  setter.call(ta, QUESTION_TEXT);
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise(function (r) { setTimeout(r, 200); });
  const btns = [].slice.call(document.querySelectorAll('.composer-box button'));
  const send = btns.find(function (b) { return b.textContent.includes('发送'); });
  if (!send) return 'no-send-button';
  send.click();
  return 'clicked';
})()"""

HOOK = """window.__errs = [];
window.addEventListener('error', function (e) { window.__errs.push(String(e.message)); });
var origError = console.error;
console.error = function () {
  window.__errs.push([].slice.call(arguments).map(String).join(' '));
  origError.apply(console, arguments);
};
var origFetch = window.fetch;
window.fetch = function (input, init) {
  window.__fetches = window.__fetches || [];
  window.__fetches.push(String(input && input.url ? input.url : input));
  return origFetch.apply(this, arguments);
};"""


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
            if "description" in res:
                print("   [EXC]", res.get("description", "")[:400])
            return res.get("value")

        await call("Page.enable")
        await call("Runtime.enable")
        # 必须尽早注入：页面脚本跑起来之前没钩住 fetch，就看不到请求有没有发出去
        await call("Page.addScriptToEvaluateOnNewDocument", source=HOOK)
        await call("Page.reload")
        await asyncio.sleep(2.5)

        print("session:", await ev("localStorage.getItem('mewhelp.session_id')"))
        await ev("document.querySelector('#tab-graph').click()")
        await asyncio.sleep(1.6)
        print("ask:", await ev(ASK.replace("QUESTION_TEXT", json.dumps(QUESTION))))
        for i in range(1, 9):
            await asyncio.sleep(1.0)
            print(f"t+{i}s", await ev(SNAP))
        print("fetches:", await ev("JSON.stringify(window.__fetches || [])"))
        await ev("document.querySelector('#tab-cards').click()")
        await asyncio.sleep(1.0)
        print("切换到工作区后:", await ev(SNAP))


if __name__ == "__main__":
    asyncio.run(main())
