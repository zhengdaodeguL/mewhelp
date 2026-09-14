import { useState } from 'react'
import { relativeTime } from '../format'
import type { MemoryItem } from '../types'
import { IconPin, IconPlus, IconTrash } from './icons'

/**
 * 记忆面板 —— 系统记住的关于"我"的事。
 *
 * 这个面板的核心价值不是"展示"，而是**可控**：用户能看到系统记住了什么、
 * 能自己加一条、能删掉记错的、能钉住不想丢的。
 *
 * 「能看见 + 能改」是记忆功能能不能被信任的分水岭。一个只在后台悄悄记、
 * 用户无从查看也无从纠正的"记忆系统"，在用户眼里就是"它怎么知道我的地址"
 * 的惊吓；同样的能力摆在明面上，感觉就是"它挺懂我"。
 */

interface Props {
  items: MemoryItem[]
  busy: boolean
  error: string
  onAdd: (text: string) => void
  onTogglePin: (item: MemoryItem) => void
  onDelete: (item: MemoryItem) => void
}

export function MemoryPanel({ items, busy, error, onAdd, onTogglePin, onDelete }: Props) {
  const [draft, setDraft] = useState('')

  const submit = () => {
    const text = draft.trim()
    if (!text || busy) return
    onAdd(text)
    setDraft('')
  }

  return (
    <div className="memory">
      <div className="memory-add">
        <input
          className="memory-input"
          value={draft}
          maxLength={60}
          placeholder="让我记住一件事，比如「我穿 L 码」"
          aria-label="添加一条记忆"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <button
          type="button"
          className="send send--small"
          disabled={!draft.trim() || busy}
          onClick={submit}
        >
          <IconPlus width={13} height={13} />
          记住
        </button>
      </div>

      {error && <p className="memory-error">{error}</p>}

      {items.length === 0 ? (
        <div className="panel-empty">
          <p>
            还没有记住任何事。
            <br />
            你在对话里说过的尺码、称呼、联系偏好，会被自动记在这里；
            <br />
            也可以在上面自己写一条。
          </p>
        </div>
      ) : (
        <ul className="memory-list">
          {items.map((item) => (
            <li key={item.id} className={`memory-item${item.pinned ? ' memory-item--pinned' : ''}`}>
              <div className="memory-line">
                <span className="memory-kind">{item.kindLabel}</span>
                <span className="memory-text">{item.text}</span>
              </div>
              <div className="memory-meta">
                <span>{item.source === 'manual' ? '我自己写的' : '对话里记住的'}</span>
                {item.hits > 0 && (
                  <>
                    <span className="session-dot" aria-hidden="true" />
                    {/* 命中次数是"这条记忆有没有价值"唯一的客观依据 ——
                        它真的帮系统补全过用户没说清的话，才算数。 */}
                    <span>帮上忙 {item.hits} 次</span>
                  </>
                )}
                <span className="session-dot" aria-hidden="true" />
                <time dateTime={item.updatedAt}>{relativeTime(item.updatedAt)}</time>
              </div>
              <div className="memory-actions">
                <button
                  type="button"
                  className={`icon-btn${item.pinned ? ' icon-btn--on' : ''}`}
                  title={item.pinned ? '取消钉住' : '钉住（不会被自动淘汰）'}
                  aria-pressed={item.pinned}
                  aria-label={item.pinned ? `取消钉住「${item.text}」` : `钉住「${item.text}」`}
                  onClick={() => onTogglePin(item)}
                >
                  <IconPin width={13} height={13} filled={item.pinned} />
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  title="忘记这条"
                  aria-label={`忘记「${item.text}」`}
                  onClick={() => onDelete(item)}
                >
                  <IconTrash width={13} height={13} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
