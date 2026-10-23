"""CDP 驱动的公共底座：会话、求值、以及"复用同一个标签页"。

    from _cdp import connect

    async with connect(url) as s:
        await s.eval("document.title")

为什么要有这个文件：之前三个脚本各自 `PUT /json/new` 开新标签页，跑一轮
截图验证就攒下三四个页面。每个页面都挂着整套 React 应用 + WebGL 风景层，
内存很快上去。现在统一成：**同源已有页面就复用，没有才新开，且新开时
顺手关掉其它页面**。测试全程只留一个页面。
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request

import websockets

CDP_HTTP = "http://127.0.0.1:9222"


def _list() -> list[dict]:
    with urllib.request.urlopen(f"{CDP_HTTP}/json/list") as resp:
        return json.loads(resp.read())


def _close(target_id: str) -> None:
    req = urllib.request.Request(f"{CDP_HTTP}/json/close/{target_id}")
    try:
        urllib.request.urlopen(req).read()
    except Exception:  # noqa: BLE001 - 关不掉不该中断测试
        pass


def _new(url: str) -> dict:
    req = urllib.request.Request(
        f"{CDP_HTTP}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def pick(url: str) -> dict:
    """挑一个页面来用：同源的就复用，否则新开一个并关掉多余的。

    同源的判据是 scheme + host + port，不是完整 URL —— 我们只关心
    "这个标签页里已经跑着同一个应用"，具体路由每次都会 Page.navigate 过去。
    """
    origin = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(url))
    pages = [t for t in _list() if t.get("type") == "page"]
    same = [t for t in pages if t.get("url", "").startswith(origin)]

    target = same[0] if same else _new(url)
    for t in pages:
        if t["id"] != target["id"]:
            _close(t["id"])
    return target


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

    async def navigate(self, url: str) -> None:
        await self.call("Page.navigate", url=url)

    async def wait_ready(self, timeout_ms: int = 15000) -> None:
        """等 React 真的挂上 DOM 再交还控制权。

        只等 readyState 不够：那只是 HTML 到了，`#root` 里还是空的 ——
        调用方紧接着去查 `.composer-input` 会拿到 null，然后误以为
        "这个页面没有输入框"。判据用"应用根节点里有元素"。
        """
        waited = 0
        while waited < timeout_ms:
            state = await self.eval(
                "document.readyState + ':' + (document.querySelector('#root')?.childElementCount ?? 0)"
            )
            if isinstance(state, str) and state.startswith("complete:") and state != "complete:0":
                return
            await asyncio.sleep(0.1)
            waited += 100
        raise RuntimeError(f"{timeout_ms}ms 内页面没有渲染出来")


class _Conn:
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self.session: Session | None = None

    async def __aenter__(self) -> Session:
        target = pick(self.url)
        self.ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024)
        self.session = Session(self.ws)
        await self.session.call("Page.enable")
        await self.session.call("Runtime.enable")
        # 把标签页推到前台并模拟"窗口有焦点"。
        #
        # 后台标签页里 document.visibilityState 是 hidden，Chrome 会把
        # 动画时钟冻住 —— 实测 `.app` 的首屏入场动画停在 t=0，
        # transform: scale(1.012)、opacity: 0。于是截图截到的是**入场前的
        # 半透明缩放帧**，量出来的布局也是被 scale 过的（390 的视口量出
        # 395 宽）。这不是界面 bug，是测量工具没让页面"活"起来。
        await self.session.call("Page.bringToFront")
        await self.session.call("Emulation.setFocusEmulationEnabled", enabled=True)
        # 复用旧页面时它可能停在上一次测试的对话上，统一导航回干净的入口
        await self.session.navigate(self.url)
        await self.session.wait_ready()
        return self.session

    async def __aexit__(self, *exc) -> None:
        if self.ws is not None:
            await self.ws.close()


def connect(url: str) -> _Conn:
    return _Conn(url)


async def sleep_ms(ms: int) -> None:
    await asyncio.sleep(ms / 1000)
