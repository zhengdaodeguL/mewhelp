import { useEffect, useState } from 'react'
import { checkHealth } from './api/client'
import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { MessageList } from './components/MessageList'
import { Workspace } from './components/Workspace'
import { useChat } from './hooks/useChat'
import { useMediaQuery } from './hooks/useMediaQuery'

/** 断点与 CSS 里的保持一致：宽屏右栏常驻，窄屏变抽屉。 */
const NARROW = '(max-width: 1024px)'

export default function App() {
  const { messages, cards, isStreaming, send, stop, reset, clearCards } = useChat()
  const [online, setOnline] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const isNarrow = useMediaQuery(NARROW)

  // 只用来点亮页头的状态灯；拿不到就当离线，不打扰用户
  useEffect(() => {
    const controller = new AbortController()
    let timer: number | undefined

    const poll = async () => {
      const { ok } = await checkHealth(controller.signal)
      setOnline(ok)
    }
    void poll()
    timer = window.setInterval(poll, 30_000)

    return () => {
      controller.abort()
      if (timer) window.clearInterval(timer)
    }
  }, [])

  // 抽屉只在窄屏存在；转回宽屏时把它复位，免得下次变窄又莫名其妙弹开
  useEffect(() => {
    if (!isNarrow) setDrawerOpen(false)
  }, [isNarrow])

  return (
    <div className={`app${isNarrow && drawerOpen ? ' app--drawer-open' : ''}`}>
      <div className="pane-chat">
        <Header
          online={online}
          onReset={() => {
            reset()
            setDrawerOpen(false)
          }}
          showWorkspaceToggle={isNarrow}
          workspaceOpen={drawerOpen}
          workspaceCount={cards.length}
          onToggleWorkspace={() => setDrawerOpen((v) => !v)}
        />
        <MessageList messages={messages} />
        <Composer isStreaming={isStreaming} onSend={send} onStop={stop} />
      </div>

      {/* 窄屏抽屉打开时的遮罩：点空白处收起 */}
      {isNarrow && drawerOpen && (
        <button
          type="button"
          className="drawer-scrim"
          aria-label="收起工作区"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      <div className={`pane-workspace${isNarrow ? ' pane-workspace--drawer' : ''}`}>
        <Workspace
          cards={cards}
          onClear={clearCards}
          onClose={isNarrow ? () => setDrawerOpen(false) : undefined}
        />
      </div>
    </div>
  )
}
