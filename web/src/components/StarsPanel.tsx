import type { ChatMessage } from '../types'
import { IconStar } from './icons'

/**
 * 收藏面板：我星标过的消息。
 *
 * 和"划重点"是同一个动作：一段长长的对话里，真正需要回头看一眼的往往只有
 * 两三句（订单号、一条规则、一句结论）。收藏把它们集中到一栏，比让用户
 * 用滚轮在几十条气泡里找要实际得多。
 *
 * 面板只展示、不编辑 —— 想看全文就点「跳到」回到对话里的那一条。
 */

interface Props {
  /** 当前会话里被收藏的消息（按收藏时间倒序由上层给） */
  items: ChatMessage[]
  onJump: (messageId: string) => void
  onUnstar: (message: ChatMessage) => void
}

/** 卡片里放不下整段答复，截到两三行；要全的就跳回对话。 */
function preview(text: string, limit = 120): string {
  const line = text.replace(/\s+/g, ' ').trim()
  return line.length > limit ? `${line.slice(0, limit)}…` : line
}

export function StarsPanel({ items, onJump, onUnstar }: Props) {
  if (items.length === 0) {
    return (
      <div className="panel-empty">
        <p>
          还没有收藏。
          <br />
          把鼠标移到任意一条消息上，点右边的星星
          <br />
          就能把它收在这里，方便以后回看。
        </p>
      </div>
    )
  }

  return (
    <ul className="star-list">
      {items.map((m) => (
        <li key={m.id} className="star-item">
          <div className="star-head">
            <span className={`star-role star-role--${m.role}`}>
              {m.role === 'user' ? '我' : '客服'}
            </span>
            <div className="star-actions">
              <button type="button" className="mini-btn" onClick={() => onJump(m.id)}>
                跳到
              </button>
              <button
                type="button"
                className="icon-btn icon-btn--on"
                title="取消收藏"
                aria-label="取消收藏"
                onClick={() => onUnstar(m)}
              >
                <IconStar width={13} height={13} filled />
              </button>
            </div>
          </div>
          <p className="star-text">{preview(m.content)}</p>
        </li>
      ))}
    </ul>
  )
}
