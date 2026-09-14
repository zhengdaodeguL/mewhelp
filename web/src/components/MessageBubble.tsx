import type { ChatMessage, Rating } from '../types'
import { useCopy } from '../hooks/useCopy'
import {
  IconCheck,
  IconCopy,
  IconRefresh,
  IconStar,
  IconThumbDown,
  IconThumbUp,
} from './icons'
import { RichText, highlight } from './RichText'

/** 逐字接收中的光标 —— 让用户知道"还在说"，而不是以为卡住了。 */
function StreamingCursor() {
  return <span className="cursor" aria-hidden="true" />
}

interface Props {
  message: ChatMessage
  /** 只有最后一条答复才给"重问"——中间那些重问没有意义 */
  canRetry: boolean
  onRetry: () => void
  /** 查找关键词（非空时把命中处标黄） */
  query?: string
  /** 被查找定位到的那一条，闪一下 */
  flash?: boolean
  starred: boolean
  onRate: (message: ChatMessage, rating: Rating) => void
  onStar: (message: ChatMessage) => void
}

export function MessageBubble({
  message,
  canRetry,
  onRetry,
  query = '',
  flash = false,
  starred,
  onRate,
  onStar,
}: Props) {
  const isUser = message.role === 'user'
  const { copied, copy } = useCopy()

  if (message.kind === 'notice') {
    return (
      <div className="notice-row">
        <span className="notice">{message.content}</span>
      </div>
    )
  }

  // 反馈和收藏都要靠这个句柄（后端叫 request_id）。没有它的消息
  // （开场白、正在生成的那条）就只给复制和重问。
  const ratable = Boolean(message.turnId) && !message.streaming

  return (
    <div
      // 查找跳转要能滚到这一条，所以需要一个稳定的锚点。
      // 用消息 id 而不是下标：下标会随新消息插入整体错位。
      id={`msg-${message.id}`}
      className={`row ${isUser ? 'row--user' : 'row--assistant'}${flash ? ' row--flash' : ''}`}
    >
      {!isUser && (
        <div className="avatar" aria-hidden="true">
          M
        </div>
      )}

      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'bubble--user' : 'bubble--assistant'}`}>
          {/* 用户输入原样显示（他打什么就是什么）；
              助手答复按轻量 Markdown 排版 —— 段落、表格、加粗 */}
          {isUser ? highlight(message.content, query) : <RichText text={message.content} query={query} />}
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

            {/* 收藏是本地视图，对用户和助手的消息都开放 —— 一句话和一句答复
                都可能是"我想回头再看一眼"的那条 */}
            {ratable && (
              <button
                type="button"
                className="icon-btn"
                title={starred ? '取消收藏' : '收藏这条'}
                aria-label={starred ? '取消收藏' : '收藏这条消息'}
                aria-pressed={starred}
                onClick={() => onStar(message)}
              >
                <IconStar filled={starred} />
              </button>
            )}

            {/* 赞/踩只给助手答复：用户不需要给自己的话打分 */}
            {ratable && !isUser && (
              <>
                <button
                  type="button"
                  className={`icon-btn${message.rating === 'up' ? ' icon-btn--on' : ''}`}
                  title={message.rating === 'up' ? '已标记有帮助' : '这条有帮助'}
                  aria-label="这条答复有帮助"
                  aria-pressed={message.rating === 'up'}
                  onClick={() => onRate(message, 'up')}
                >
                  <IconThumbUp />
                </button>
                <button
                  type="button"
                  className={`icon-btn${message.rating === 'down' ? ' icon-btn--down' : ''}`}
                  title={message.rating === 'down' ? '已反馈没解决' : '这条没解决我的问题'}
                  aria-label="这条答复没解决问题"
                  aria-pressed={message.rating === 'down'}
                  onClick={() => onRate(message, 'down')}
                >
                  <IconThumbDown />
                </button>
              </>
            )}

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
