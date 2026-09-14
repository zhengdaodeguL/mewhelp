import { useEffect, useRef, useState, type RefObject } from 'react'
import type { OrderBrief } from '../api/client'
import { IconSend, IconStop } from './icons'
import { OrderPicker } from './OrderPicker'
import { QuickReplies } from './QuickReplies'

const MAX_CHARS = 2000

interface Props {
  isStreaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  /** 交给上层，好让 Ctrl/⌘+K 能聚焦到输入框 */
  inputRef: RefObject<HTMLTextAreaElement | null>
  /** 快捷问题只是"不知道怎么开口"时的引子，聊起来之后就该让位 */
  showQuickReplies: boolean
}

export function Composer({ isStreaming, onSend, onStop, inputRef, showQuickReplies }: Props) {
  const [text, setText] = useState('')
  const innerRef = useRef<HTMLTextAreaElement>(null)

  // 输入框随内容长高，但不超过 6 行 —— 超过就内部滚动，不能把对话框挤没
  useEffect(() => {
    const el = innerRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  // Esc 停止生成 —— 流式输出时最想按的一个键
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isStreaming) {
        e.preventDefault()
        onStop()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isStreaming, onStop])

  const canSend = text.trim().length > 0 && !isStreaming

  const submit = () => {
    if (!canSend) return
    onSend(text)
    setText('')
  }

  /**
   * 选中订单：只把话**填**进输入框，不直接发出去。
   * 用户点了单之后往往还想补一句（"这单能退吗"），替他发掉等于替他做决定。
   */
  const pickOrder = (order: OrderBrief) => {
    setText(`帮我查下订单 ${order.order_id} 的状态`)
    innerRef.current?.focus()
  }

  return (
    <div className="composer">
      <OrderPicker disabled={isStreaming} onPick={pickOrder} />
      {showQuickReplies && <QuickReplies disabled={isStreaming} onPick={onSend} />}

      <div className="composer-box">
        <textarea
          ref={(el) => {
            innerRef.current = el
            inputRef.current = el
          }}
          className="composer-input"
          value={text}
          rows={1}
          maxLength={MAX_CHARS}
          placeholder="描述你的问题，例如：帮我查下订单 ORD-1001 到哪了"
          aria-label="输入消息"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter 发送，Shift+Enter 换行 —— 和主流聊天软件一致。
            // isComposing 判断是为了中文输入法：选词时的回车不能当成发送。
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
        />

        {isStreaming ? (
          <button type="button" className="send send--stop" onClick={onStop} title="停止生成（Esc）">
            <IconStop />
            停止
          </button>
        ) : (
          <button type="button" className="send" disabled={!canSend} onClick={submit}>
            <IconSend />
            发送
          </button>
        )}
      </div>

      <p className="composer-hint">
        <kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行 · <kbd>Esc</kbd> 停止 ·{' '}
        <kbd>Ctrl</kbd>+<kbd>K</kbd> 聚焦输入 · <kbd>Ctrl</kbd>+<kbd>F</kbd> 查找
      </p>
    </div>
  )
}
