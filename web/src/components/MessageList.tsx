import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'
import { MessageBubble } from './MessageBubble'

/** 等待首字时的三点动画。 */
function TypingIndicator() {
  return (
    <div className="row row--assistant">
      <div className="avatar" aria-hidden="true">
        M
      </div>
      <div className="bubble bubble--assistant bubble--typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  )
}

interface Props {
  messages: ChatMessage[]
}

export function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const last = messages[messages.length - 1]
  // 最后一条是空的流式气泡 -> 还没吐字，用三点动画占位
  const waitingFirstToken = Boolean(last?.streaming && last.content === '')

  useEffect(() => {
    // 每次消息变化都贴底，跟真实聊天软件一致
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <div className="messages" role="log" aria-live="polite" aria-label="对话记录">
      <div className="messages-inner">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {waitingFirstToken && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
