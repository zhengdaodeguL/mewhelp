import type { ChatMessage } from '../types'

/** 逐字接收中的光标 —— 让用户知道"还在说"，而不是以为卡住了。 */
function StreamingCursor() {
  return <span className="cursor" aria-hidden="true" />
}

interface Props {
  message: ChatMessage
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  if (message.kind === 'notice') {
    return (
      <div className="notice-row">
        <span className="notice">{message.content}</span>
      </div>
    )
  }

  return (
    <div className={`row ${isUser ? 'row--user' : 'row--assistant'}`}>
      {!isUser && (
        <div className="avatar" aria-hidden="true">
          M
        </div>
      )}
      <div className={`bubble ${isUser ? 'bubble--user' : 'bubble--assistant'}`}>
        {/* 空内容 + streaming = 还在等首字；气泡留白由 CSS 撑住，避免高度跳动 */}
        {message.content}
        {message.streaming && <StreamingCursor />}
      </div>
    </div>
  )
}
