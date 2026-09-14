import { useEffect, useRef, useState, type RefObject } from 'react'
import type { OrderBrief } from '../api/client'
import type { ChatMessage } from '../types'
import { IconClose, IconSend, IconStop } from './icons'
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
  /** 正在引用的那条消息（引用提问） */
  quote: ChatMessage | null
  onClearQuote: () => void
}

export function Composer({
  isStreaming,
  onSend,
  onStop,
  inputRef,
  showQuickReplies,
  quote,
  onClearQuote,
}: Props) {
  const [text, setText] = useState('')
  const innerRef = useRef<HTMLTextAreaElement>(null)

  // 输入框随内容长高，但不超过 6 行 —— 超过就内部滚动，不能把对话框挤没
  useEffect(() => {
    const el = innerRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  // 选中"引用"之后立刻聚焦：引用是"我要针对这句说点什么"，
  // 光标不跳到输入框里，用户还得再点一下 —— 那一下就是白给的
  useEffect(() => {
    if (quote) innerRef.current?.focus()
  }, [quote])

  // Esc 停止生成 —— 流式输出时最想按的一个键。
  // 引用条存在时 Esc 先取消引用（就近原则），再按一次才是停止。
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (isStreaming) {
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
    // 引用条不在这里清 —— 清不清由上层决定（它还要决定这条引用要不要
    // 跟着请求一起发出去）。这里只管把输入框腾空。
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

      {quote && (
        <div className="quote-bar">
          <span className="quote-label">
            {quote.role === 'user' ? '引用我的上一句' : '引用上一条答复'}
          </span>
          <span className="quote-text">{quote.content.replace(/\s+/g, ' ').slice(0, 90)}</span>
          <button
            type="button"
            className="icon-btn"
            title="取消引用"
            aria-label="取消引用"
            onClick={onClearQuote}
          >
            <IconClose width={13} height={13} />
          </button>
        </div>
      )}

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
        <kbd>Ctrl</kbd>+<kbd>K</kbd> 命令面板 · <kbd>Ctrl</kbd>+<kbd>F</kbd> 查找
      </p>
    </div>
  )
}
