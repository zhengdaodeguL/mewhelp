import { useEffect, useRef, useState, type RefObject } from 'react'
import type { OrderBrief } from '../api/client'
import type { ChatMessage } from '../types'
import { IconArrowUp, IconClose, IconStop } from './icons'
import { OrderPicker } from './OrderPicker'

const MAX_CHARS = 2000

interface Props {
  isStreaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  /** 交给上层，好让 Ctrl/⌘+K 能聚焦到输入框 */
  inputRef: RefObject<HTMLTextAreaElement | null>
  /** 正在引用的那条消息（引用提问） */
  quote: ChatMessage | null
  onClearQuote: () => void
}

/**
 * 输入区：一块大圆角面板，里面分两层 —— 上面写字，下面放操作。
 *
 * 上一版是"一条输入框 + 一颗带文字的发送按钮"并排。两种形态的区别不在
 * 好看，而在**输入区是这块界面的主入口还是附属件**：并排的窄条读起来像
 * 一个表单字段，分层的面板读起来像一张"可以往里写的纸"。客服场景里
 * 用户 90% 的时间是在这里打字，它值得占住底部那一整块。
 *
 * 发送键收成圆形上箭头：文字"发送"在禁用态下最容易被压成灰糊一片，
 * 而一个圆形的形状在禁用态依然清楚是"送出去的键"。禁用时用中性灰底
 * 而不是降透明度 —— 理由和上一版一致：禁用是"现在不能用"，不是
 * "这里没有东西"。
 *
 * 快捷问题已经搬去首屏空态（EmptyState）。它们本来就是"不知道怎么开口"
 * 时的引子，而对话一旦开始，这一排就一直挤在输入框上面占地方。
 */
export function Composer({ isStreaming, onSend, onStop, inputRef, quote, onClearQuote }: Props) {
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

      <div className={`composer-box${quote ? ' composer-box--quoted' : ''}`}>
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

        <textarea
          ref={(el) => {
            innerRef.current = el
            inputRef.current = el
          }}
          className="composer-input"
          value={text}
          rows={1}
          maxLength={MAX_CHARS}
          placeholder="问点什么…"
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

        <div className="composer-foot">
          <span className="composer-hint">
            <kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行 · <kbd>Ctrl</kbd>+
            <kbd>K</kbd> 命令面板
          </span>

          {isStreaming ? (
            <button
              type="button"
              className="send send--stop"
              onClick={onStop}
              title="停止生成（Esc）"
              aria-label="停止生成"
            >
              <IconStop width={14} height={14} />
            </button>
          ) : (
            <button
              type="button"
              className="send"
              disabled={!canSend}
              onClick={submit}
              title="发送（Enter）"
              aria-label="发送"
            >
              <IconArrowUp width={16} height={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
