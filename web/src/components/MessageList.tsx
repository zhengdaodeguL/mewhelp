import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../types'
import { IconArrowDown } from './icons'
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
  canRetry: boolean
  onRetry: () => void
}

export function MessageList({ messages, canRetry, onRetry }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)

  const last = messages[messages.length - 1]
  // 最后一条是空的流式气泡 -> 还没吐字，用三点动画占位
  const waitingFirstToken = Boolean(last?.streaming && last.content === '')
  // "重问"只给最后一条答复：中间那些重问没有意义
  const lastAssistant = messages.map((m) => m.role).lastIndexOf('assistant')

  const scrollToBottom = useCallback((smooth = true) => {
    bottomRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'end' })
  }, [])

  // 只在用户本来就贴着底时才自动跟随。他往上翻着看历史时被拽回底部，
  // 是聊天界面里最恼人的一类交互。
  useEffect(() => {
    if (atBottom) scrollToBottom()
  }, [messages, atBottom, scrollToBottom])

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
  }

  return (
    <div className="messages-wrap">
      <div
        className="messages"
        ref={scrollRef}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-label="对话记录"
      >
        <div className="messages-inner">
          {messages.map((m, i) => (
            <MessageBubble
              key={m.id}
              message={m}
              canRetry={canRetry && i === lastAssistant}
              onRetry={onRetry}
            />
          ))}
          {waitingFirstToken && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {!atBottom && (
        <button
          type="button"
          className="scroll-bottom"
          aria-label="回到底部"
          title="回到底部"
          onClick={() => {
            setAtBottom(true)
            scrollToBottom()
          }}
        >
          <IconArrowDown />
        </button>
      )}
    </div>
  )
}
