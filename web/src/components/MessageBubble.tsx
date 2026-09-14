import type { ChatMessage } from '../types'
import { useCopy } from '../hooks/useCopy'
import { IconCheck, IconCopy, IconRefresh } from './icons'
import { RichText } from './RichText'

/** 逐字接收中的光标 —— 让用户知道"还在说"，而不是以为卡住了。 */
function StreamingCursor() {
  return <span className="cursor" aria-hidden="true" />
}

interface Props {
  message: ChatMessage
  /** 只有最后一条答复才给"重问"——中间那些重问没有意义 */
  canRetry: boolean
  onRetry: () => void
}

export function MessageBubble({ message, canRetry, onRetry }: Props) {
  const isUser = message.role === 'user'
  const { copied, copy } = useCopy()

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

      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'bubble--user' : 'bubble--assistant'}`}>
          {/* 用户输入原样显示（他打什么就是什么）；
              助手答复按轻量 Markdown 排版 —— 段落、表格、加粗 */}
          {isUser ? message.content : <RichText text={message.content} />}
          {message.streaming && <StreamingCursor />}
        </div>

        {/* 消息工具条：平时藏起来，悬停或聚焦时才出现，不抢对话本身 */}
        {message.content && !message.streaming && (
          <div className={`msg-actions ${isUser ? 'msg-actions--user' : ''}`}>
            <button
              type="button"
              className="icon-btn"
              title={copied ? '已复制' : '复制'}
              aria-label={copied ? '已复制' : '复制这条消息'}
              onClick={() => void copy(message.content)}
            >
              {copied ? <IconCheck /> : <IconCopy />}
            </button>
            {!isUser && canRetry && (
              <button
                type="button"
                className="icon-btn"
                title="重问一次"
                aria-label="重问一次"
                onClick={onRetry}
              >
                <IconRefresh />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
