import { useEffect, useRef, useState } from 'react'
import { QuickReplies } from './QuickReplies'

const MAX_CHARS = 2000

interface Props {
  isStreaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}

export function Composer({ isStreaming, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // 输入框随内容长高，但不超过 6 行 —— 超过就内部滚动，不能把对话框挤没
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  const canSend = text.trim().length > 0 && !isStreaming

  const submit = () => {
    if (!canSend) return
    onSend(text)
    setText('')
  }

  const handleQuickPick = (question: string) => {
    if (isStreaming) return
    onSend(question)
  }

  return (
    <div className="composer">
      <QuickReplies disabled={isStreaming} onPick={handleQuickPick} />
      <div className="composer-box">
        <textarea
          ref={textareaRef}
          className="composer-input"
          value={text}
          rows={1}
          maxLength={MAX_CHARS}
          placeholder="描述你的问题，例如：帮我查下订单 ORD-1001 到哪了"
          aria-label="输入消息"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter 发送，Shift+Enter 换行 —— 和主流聊天软件一致
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {isStreaming ? (
          <button type="button" className="send send--stop" onClick={onStop} aria-label="停止生成">
            停止
          </button>
        ) : (
          <button type="button" className="send" disabled={!canSend} onClick={submit} aria-label="发送">
            发送
          </button>
        )}
      </div>
      <p className="composer-hint">Enter 发送 · Shift + Enter 换行</p>
    </div>
  )
}
