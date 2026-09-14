import { useEffect, useRef } from 'react'
import { IconArrowDown, IconArrowUp, IconClose, IconSearch } from './icons'

/**
 * 会话内查找。
 *
 * 长对话只有两个痛点：找不到之前说过的那句、和翻不回去。前者靠这一条解决，
 * 后者靠已有的"回到底部"。
 *
 * 命中数显示成 "2 / 7" 而不是只给一个总数 —— 用户需要知道"还有几条"，
 * 才知道要不要继续按。而且它明确回答了"到底找没找到"（0 时给 0）。
 */

interface Props {
  query: string
  /** 命中总数（消息条数） */
  count: number
  /** 当前在第几条（从 1 开始；没有命中时是 0） */
  index: number
  onQuery: (value: string) => void
  onStep: (delta: number) => void
  onClose: () => void
}

export function FindBar({ query, count, index, onQuery, onStep, onClose }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  // 打开就聚焦：这个面板的存在意义就是"立刻开始打字"
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  return (
    <div className="findbar" role="search">
      <IconSearch width={14} height={14} />
      <input
        ref={inputRef}
        className="findbar-input"
        value={query}
        placeholder="在本次对话里查找"
        aria-label="在本次对话里查找"
        onChange={(e) => onQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            onStep(e.shiftKey ? -1 : 1)
          } else if (e.key === 'Escape') {
            e.preventDefault()
            onClose()
          }
        }}
      />
      <span className="findbar-count" aria-live="polite">
        {query ? (count === 0 ? '没有匹配' : `${index} / ${count}`) : ''}
      </span>
      <button
        type="button"
        className="icon-btn"
        title="上一个（Shift+Enter）"
        aria-label="上一个匹配"
        disabled={count === 0}
        onClick={() => onStep(-1)}
      >
        <IconArrowUp width={14} height={14} />
      </button>
      <button
        type="button"
        className="icon-btn"
        title="下一个（Enter）"
        aria-label="下一个匹配"
        disabled={count === 0}
        onClick={() => onStep(1)}
      >
        <IconArrowDown width={14} height={14} />
      </button>
      <button type="button" className="icon-btn" title="关闭（Esc）" aria-label="关闭查找" onClick={onClose}>
        <IconClose width={14} height={14} />
      </button>
    </div>
  )
}
