import { useEffect, useState } from 'react'
import { checkHealth } from './api/client'
import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { MessageList } from './components/MessageList'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, isStreaming, send, stop, reset } = useChat()
  const [online, setOnline] = useState(false)

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

  return (
    <div className="app">
      <Header online={online} onReset={reset} />
      <MessageList messages={messages} />
      <Composer isStreaming={isStreaming} onSend={send} onStop={stop} />
    </div>
  )
}
