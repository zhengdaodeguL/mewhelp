import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage, Rating } from '../types'
import { IconArrowDown } from './icons'
import { MessageBubble } from './MessageBubble'

/** 等待首字时的占位：三点 + 两条微光条，把"内容正在取"画成形状。 */
function TypingIndicator() {
  return (
    <div className="row row--assistant">
      <div className="avatar avatar--thinking" aria-hidden="true">
        M
      </div>
      <div className="bubble bubble--assistant bubble--typing" aria-label="正在查询">
        <div className="typing-dots">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
        <span className="typing-line" />
        <span className="typing-line typing-line--short" />
      </div>
    </div>
  )
}

interface Props {
  messages: ChatMessage[]
  canRetry: boolean
  onRetry: () => void
  query?: string
  /** 查找定位到的那一条（会闪一下） */
  flashId?: string
  /** 正在被引用的那一条 */
  quotedId?: string
  starredIds: string[]
  onRate: (message: ChatMessage, rating: Rating) => void
  onStar: (message: ChatMessage) => void
  onQuote: (message: ChatMessage) => void
  /** 生成中：页头下方那条进度光带 */
  streaming: boolean
}

export function MessageList({
  messages,
  canRetry,
  onRetry,
  query = '',
  flashId = '',
  quotedId = '',
  starredIds,
  onRate,
  onStar,
  onQuote,
  streaming,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)
  const [scrolled, setScrolled] = useState(false)
  //: 用户往上翻时"已经开始看不到"的那一条下标。用它算出漏了几条 ——
  //: 只说"有新消息"，用户不知道错过了多少，于是多半懒得点。
  const hiddenFrom = useRef(0)

  const last = messages[messages.length - 1]
  // 最后一条是空的流式气泡 -> 还没吐字，用三点动画占位
  const waitingFirstToken = Boolean(last?.streaming && last.content === '')
  // "重问"只给最后一条答复：中间那些重问没有意义
  const lastAssistant = messages.map((m) => m.role).lastIndexOf('assistant')
  const unseen = atBottom ? 0 : Math.max(0, messages.length - hiddenFrom.current)

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
    // 80px 的容差：滚动事件的小数像素抖动会让"刚好贴底"反复横跳
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    setScrolled(el.scrollTop > 4)
    setAtBottom((was) => {
      if (!nearBottom && was) hiddenFrom.current = messages.length
      return nearBottom
    })
  }

  return (
    <div className="messages-wrap">
      {streaming && <div className="stream-bar" aria-hidden="true" />}

      <div
        className={`messages${scrolled ? ' messages--scrolled' : ''}`}
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
              query={query}
              flash={Boolean(flashId) && flashId === m.id}
              quoted={Boolean(quotedId) && quotedId === m.id}
              starred={Boolean(m.turnId) && starredIds.includes(m.turnId as string)}
              onRate={onRate}
              onStar={onStar}
              onQuote={onQuote}
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
          aria-label="回到最新"
          title="回到最新"
          onClick={() => {
            setAtBottom(true)
            scrollToBottom()
          }}
        >
          <IconArrowDown width={14} height={14} />
          {unseen > 0 ? (
            <span className="scroll-bottom-count">{unseen} 条新消息</span>
          ) : (
            '回到最新'
          )}
        </button>
      )}
    </div>
  )
}
