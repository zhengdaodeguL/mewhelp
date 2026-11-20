import { Fragment, useMemo, useState } from 'react'
import { dayGroup, relativeTime, turnsToRounds } from '../format'
import { useNow } from '../hooks/useNow'
import type { SessionBrief } from '../types'
import { SessionSkeleton } from './Skeleton'
import { IconClose, IconPencil, IconPlus, IconSearch, IconTrash } from './icons'

/**
 * 左侧历史栏。
 *
 * 设计上只做一件事：让用户**认出**"哪一次对话是我要回去的那次"。
 * 所以每条给的是标题（第一句原话）、轮数、相对时间、最后一句答复的开头，
 * 而不是摘要、意图、命中率这些 —— 后者是给人分析用的，不是给人找东西用的。
 *
 * 三个交互细节值得单说：
 *
 *   * **按自然日分组**（今天 / 昨天 / 本周 / 更早）。分组的价值是定位：
 *     "昨天那通"比"列表第 7 条"好找得多。
 *   * **删除要二次确认，而且是就地确认**，不用 `window.confirm`：
 *     原生弹窗在演示/嵌入式场景里经常被浏览器拦掉，点一下什么都不发生，
 *     用户会以为"删不掉"。就地确认还顺带告诉了他删的是哪一条。
 *   * **操作收进常显的更多菜单**，和打开会话的按钮分开。
 *     悬停不出现新的点击目标；重命名/删除都必须先主动打开菜单。
 */

interface Props {
  sessions: SessionBrief[]
  activeId: string
  loading: boolean
  onSelect: (sessionId: string) => void
  onNew: () => void
  onRename: (sessionId: string, title: string) => void
  onDelete: (sessionId: string) => void
  /** 窄屏下是抽屉，需要一个关闭入口 */
  onClose?: () => void
}

export function HistoryRail({
  sessions,
  activeId,
  loading,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onClose,
}: Props) {
  const [query, setQuery] = useState('')
  const [editing, setEditing] = useState<string>('')
  const [draft, setDraft] = useState('')
  const [confirming, setConfirming] = useState<string>('')
  // 相对时间要自己往前走。不重新渲染的话，开着页面十分钟，"刚刚"还是"刚刚"。
  const now = useNow()

  // 搜索在**本地**过滤：一次最多几十条，服务端往返换不来什么，
  // 却会让每敲一个字都闪一下加载态。
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return sessions
    return sessions.filter(
      (s) => s.title.toLowerCase().includes(q) || s.preview.toLowerCase().includes(q),
    )
  }, [sessions, query])

  const startRename = (s: SessionBrief) => {
    setConfirming('')
    setEditing(s.sessionId)
    setDraft(s.title)
  }

  const commitRename = () => {
    if (editing) onRename(editing, draft.trim())
    setEditing('')
  }

  let lastGroup = ''

  return (
    <section className="rail" aria-label="历史对话">
      <header className="rail-head">
        <span className="rail-title">历史对话</span>
        <button type="button" className="ghost-btn rail-new" onClick={onNew} title="开一次新对话">
          <IconPlus width={13} height={13} />
          新对话
        </button>
        {onClose && (
          <button
            type="button"
            className="icon-btn rail-close"
            aria-label="收起历史栏"
            title="收起"
            onClick={onClose}
          >
            <IconClose />
          </button>
        )}
      </header>

      <div className="rail-search">
        <IconSearch width={13} height={13} />
        <input
          className="rail-search-input"
          value={query}
          placeholder="搜索历史对话"
          aria-label="搜索历史对话"
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            type="button"
            className="icon-btn rail-search-clear"
            aria-label="清空搜索"
            onClick={() => setQuery('')}
          >
            <IconClose width={12} height={12} />
          </button>
        )}
      </div>

      <div className="rail-list">
        {loading && sessions.length === 0 ? (
          // 第一次加载：先画出"将有四行"的轮廓，数据回来时位置不跳
          <>
            <SessionSkeleton />
            <SessionSkeleton />
            <SessionSkeleton />
            <SessionSkeleton />
          </>
        ) : visible.length === 0 ? (
          <p className="rail-empty">
            {sessions.length === 0
              ? '还没有历史对话。聊几句就会出现在这里。'
              : '没有匹配的对话'}
          </p>
        ) : (
          visible.map((s, i) => {
            const active = s.sessionId === activeId
            const group = dayGroup(s.updatedAt, now)
            const showGroup = group !== lastGroup
            lastGroup = group
            return (
              <Fragment key={s.sessionId}>
                {showGroup && <div className="session-group">{group}</div>}
                <div
                  className={`session${active ? ' session--active' : ''}`}
                  style={{ '--i': i } as React.CSSProperties}
                >
                  {editing === s.sessionId ? (
                    <input
                      className="session-rename"
                      value={draft}
                      autoFocus
                      aria-label="会话标题"
                      onChange={(e) => setDraft(e.target.value)}
                      onBlur={commitRename}
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitRename()
                        if (e.key === 'Escape') setEditing('')
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      className="session-select"
                      aria-current={active ? 'true' : undefined}
                      onClick={() => onSelect(s.sessionId)}
                    >
                      <span className="session-title" title={s.title}>{s.title}</span>
                      <span className="session-meta">
                        <span>{turnsToRounds(s.turns)}</span>
                        <span className="session-dot" aria-hidden="true" />
                        <time dateTime={s.updatedAt}>{relativeTime(s.updatedAt, now)}</time>
                      </span>
                      {s.preview && <span className="session-preview">{s.preview}</span>}
                    </button>
                  )}

                  {confirming === s.sessionId ? (
                    <div className="session-confirm" onClick={(e) => e.stopPropagation()}>
                      <span>删除这次对话？</span>
                      <button
                        type="button"
                        className="mini-btn mini-btn--danger"
                        onClick={() => {
                          onDelete(s.sessionId)
                          setConfirming('')
                        }}
                      >
                        删除
                      </button>
                      <button type="button" className="mini-btn" onClick={() => setConfirming('')}>
                        取消
                      </button>
                    </div>
                  ) : (
                    editing !== s.sessionId && (
                      <details
                        className="session-menu"
                        onKeyDown={(e) => {
                          if (e.key === 'Escape') {
                            e.currentTarget.open = false
                            e.currentTarget.querySelector('summary')?.focus()
                          }
                        }}
                        onBlur={(e) => {
                          if (!e.currentTarget.contains(e.relatedTarget)) e.currentTarget.open = false
                        }}
                      >
                        <summary aria-label={`「${s.title}」更多操作`} title="更多操作">···</summary>
                        <div className="session-menu-items">
                        <button
                          type="button"
                          aria-label={`重命名「${s.title}」`}
                          onClick={(e) => {
                            e.currentTarget.closest('details')?.removeAttribute('open')
                            startRename(s)
                          }}
                        >
                          <IconPencil width={13} height={13} /> 重命名
                        </button>
                        <button
                          type="button"
                          aria-label={`删除「${s.title}」`}
                          onClick={(e) => {
                            e.currentTarget.closest('details')?.removeAttribute('open')
                            setConfirming(s.sessionId)
                          }}
                        >
                          <IconTrash width={13} height={13} /> 删除
                        </button>
                        </div>
                      </details>
                    )
                  )}
                </div>
              </Fragment>
            )
          })
        )}
      </div>

      <footer className="rail-foot">
        <span className="rail-foot-text">对话存在服务端，换台设备也在</span>
      </footer>
    </section>
  )
}
