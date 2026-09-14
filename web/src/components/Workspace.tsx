import { useRef, type ReactNode } from 'react'
import { useCopy } from '../hooks/useCopy'
import type { ChatMessage, MemoryItem } from '../types'
import { cardToText, formatAmount, statusTone, type WorkspaceCard } from '../workspace'
import { MemoryPanel } from './MemoryPanel'
import { StarsPanel } from './StarsPanel'
import {
  IconBox,
  IconCheck,
  IconCopy,
  IconDoc,
  IconMemory,
  IconStar,
  IconTruck,
} from './icons'

/**
 * 右侧工作区：三个标签页。
 *
 *   工作区 —— 这一轮对话"办的事"：订单、物流轨迹、政策依据
 *   记忆   —— 系统记住的关于我的事（可看、可加、可删）
 *   收藏   —— 我星标过的消息
 *
 * 三者放在同一个面板里而不是各开一个抽屉，是因为它们回答的是同一个问题：
 * "这次对话留下了什么"。分成三处之后，用户要先想"我要找的东西属于哪一类"
 * 才能去点那个按钮 —— 而记忆和收藏这两个概念的区别，用户第一眼是分不清的。
 *
 * 这里只放**用户该看的东西**。意图、节点路径、证据分数、工具名、耗时都是
 * 内部信息，一律不进这个面板（开发者走 /docs 和 /api/trace/{request_id}）。
 */

export type WorkspaceTab = 'cards' | 'memory' | 'stars'

const TABS: ReadonlyArray<{ key: WorkspaceTab; label: string; Icon: typeof IconBox }> = [
  { key: 'cards', label: '工作区', Icon: IconBox },
  { key: 'memory', label: '记忆', Icon: IconMemory },
  { key: 'stars', label: '收藏', Icon: IconStar },
]

type CardKind = WorkspaceCard['kind']

const KIND_META: Record<CardKind, { label: string; Icon: typeof IconBox }> = {
  order: { label: '订单', Icon: IconBox },
  logistics: { label: '物流', Icon: IconTruck },
  source: { label: '依据', Icon: IconDoc },
  action: { label: '处理结果', Icon: IconCheck },
}

/** 卡片外壳：统一的头（类型标签 + 图标 + 复制）与配色。 */
function CardShell({
  kind,
  children,
  headExtra,
}: {
  kind: CardKind
  children: ReactNode
  headExtra?: ReactNode
}) {
  const { label, Icon } = KIND_META[kind]
  return (
    <article className={`card card--${kind}`}>
      <header className="card-head">
        <span className="card-kind">
          <Icon width={13} height={13} />
          {label}
        </span>
        {headExtra}
      </header>
      {children}
    </article>
  )
}

/** 每张卡片右上角的复制按钮 —— 拿到订单信息的下一步多半是粘到别处。 */
function CardCopy({ card }: { card: WorkspaceCard }) {
  const { copied, copy } = useCopy()
  return (
    <button
      type="button"
      className="icon-btn card-copy"
      title={copied ? '已复制' : '复制'}
      aria-label={copied ? '已复制' : '复制卡片内容'}
      onClick={() => void copy(cardToText(card))}
    >
      {copied ? <IconCheck /> : <IconCopy />}
    </button>
  )
}

function StatusPill({ label }: { label: string }) {
  return <span className={`pill pill--${statusTone(label)}`}>{label}</span>
}

function OrderView({ card }: { card: Extract<WorkspaceCard, { kind: 'order' }> }) {
  const total = formatAmount(card.totalAmount, card.currency)
  return (
    <CardShell
      kind="order"
      headExtra={
        <>
          <span className="card-id">{card.orderId}</span>
          <StatusPill label={card.statusLabel} />
          <CardCopy card={card} />
        </>
      }
    >
      {card.items.length > 0 && (
        <ul className="item-list">
          {card.items.map((it, i) => (
            <li key={`${it.name}-${i}`} className="item">
              <span className="item-name">{it.name}</span>
              <span className="item-qty">×{it.qty}</span>
              <span className="item-price">{formatAmount(it.price, card.currency) ?? '—'}</span>
            </li>
          ))}
        </ul>
      )}

      <dl className="kv">
        {total && (
          <>
            <dt>合计</dt>
            <dd className="kv-strong">{total}</dd>
          </>
        )}
        {card.address && (
          <>
            <dt>收货地址</dt>
            <dd>{card.address}</dd>
          </>
        )}
      </dl>
    </CardShell>
  )
}

function LogisticsView({ card }: { card: Extract<WorkspaceCard, { kind: 'logistics' }> }) {
  return (
    <CardShell
      kind="logistics"
      headExtra={
        <>
          <span className="card-id">{card.orderId}</span>
          <StatusPill label={card.statusLabel} />
          <CardCopy card={card} />
        </>
      }
    >
      {(card.carrier || card.trackingNo) && (
        <dl className="kv kv--tight">
          {card.carrier && (
            <>
              <dt>承运</dt>
              <dd>{card.carrier}</dd>
            </>
          )}
          {card.trackingNo && (
            <>
              <dt>运单号</dt>
              <dd className="kv-mono">{card.trackingNo}</dd>
            </>
          )}
        </dl>
      )}

      {card.steps.length > 0 && (
        <ol className="timeline">
          {card.steps.map((s, i) => (
            <li key={i} className="timeline-item">
              <span className="timeline-dot" aria-hidden="true" />
              <div className="timeline-body">
                <p className="timeline-text">{s.text}</p>
                {s.time && <time className="timeline-time">{s.time}</time>}
              </div>
            </li>
          ))}
        </ol>
      )}

      {card.note && <p className="card-note">{card.note}</p>}
    </CardShell>
  )
}

function SourceView({ card }: { card: Extract<WorkspaceCard, { kind: 'source' }> }) {
  return (
    <CardShell
      kind="source"
      headExtra={
        <>
          <span className="card-title">{card.title}</span>
          <CardCopy card={card} />
        </>
      }
    >
      <p className="card-snippet">{card.snippet}</p>
    </CardShell>
  )
}

function ActionView({ card }: { card: Extract<WorkspaceCard, { kind: 'action' }> }) {
  return (
    <CardShell kind="action" headExtra={<CardCopy card={card} />}>
      {card.orderId && <span className="card-id card-id--own">{card.orderId}</span>}
      <p className="card-snippet">{card.text}</p>
    </CardShell>
  )
}

function Card({ card }: { card: WorkspaceCard }) {
  switch (card.kind) {
    case 'order':
      return <OrderView card={card} />
    case 'logistics':
      return <LogisticsView card={card} />
    case 'source':
      return <SourceView card={card} />
    case 'action':
      return <ActionView card={card} />
  }
}

function CardsView({ cards }: { cards: WorkspaceCard[] }) {
  const summary = cards.reduce<Record<string, number>>((acc, c) => {
    acc[c.kind] = (acc[c.kind] ?? 0) + 1
    return acc
  }, {})

  if (cards.length === 0) {
    return (
      <div className="workspace-empty">
        <div className="grid-canvas" aria-hidden="true" />
        <p className="workspace-empty-text">
          对话里查到的订单、物流轨迹和政策依据
          <br />
          会陈列在这里
        </p>
      </div>
    )
  }

  return (
    <>
      <div className="workspace-tallies">
        {(Object.keys(KIND_META) as CardKind[])
          .filter((k) => summary[k])
          .map((k) => (
            <span key={k} className={`tally tally--${k}`}>
              {KIND_META[k].label} {summary[k]}
            </span>
          ))}
      </div>
      <div className="card-stack">
        {cards.map((c) => (
          <Card key={c.id} card={c} />
        ))}
      </div>
    </>
  )
}

interface Props {
  tab: WorkspaceTab
  onTab: (tab: WorkspaceTab) => void
  cards: WorkspaceCard[]
  onClearCards: () => void
  memories: MemoryItem[]
  memoryBusy: boolean
  memoryError: string
  onAddMemory: (text: string) => void
  onToggleMemoryPin: (item: MemoryItem) => void
  onDeleteMemory: (item: MemoryItem) => void
  starred: ChatMessage[]
  onJumpTo: (messageId: string) => void
  onUnstar: (message: ChatMessage) => void
  /** 窄屏下工作区是抽屉，需要一个关闭入口 */
  onClose?: () => void
}

export function Workspace({
  tab,
  onTab,
  cards,
  onClearCards,
  memories,
  memoryBusy,
  memoryError,
  onAddMemory,
  onToggleMemoryPin,
  onDeleteMemory,
  starred,
  onJumpTo,
  onUnstar,
  onClose,
}: Props) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

  const counts: Record<WorkspaceTab, number> = {
    cards: cards.length,
    memory: memories.length,
    stars: starred.length,
  }

  /** 标签之间用左右方向键切换（WAI-ARIA 的 tabs 约定）。 */
  const onKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
    e.preventDefault()
    const next = (index + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length
    onTab(TABS[next].key)
    tabRefs.current[next]?.focus()
  }

  return (
    <section className="workspace" aria-label="工作区">
      <header className="workspace-head">
        <div className="tabs" role="tablist" aria-label="右侧面板">
          {TABS.map(({ key, label, Icon }, i) => (
            <button
              key={key}
              ref={(el) => {
                tabRefs.current[i] = el
              }}
              type="button"
              role="tab"
              id={`tab-${key}`}
              aria-selected={tab === key}
              aria-controls={`tabpanel-${key}`}
              tabIndex={tab === key ? 0 : -1}
              className={`tab${tab === key ? ' tab--active' : ''}`}
              onClick={() => onTab(key)}
              onKeyDown={(e) => onKeyDown(e, i)}
            >
              <Icon width={14} height={14} />
              {label}
              {counts[key] > 0 && <span className="tab-count">{counts[key]}</span>}
            </button>
          ))}
        </div>

        <div className="workspace-actions">
          {tab === 'cards' && cards.length > 0 && (
            <button type="button" className="ghost-btn" onClick={onClearCards}>
              清空
            </button>
          )}
          {onClose && (
            <button type="button" className="ghost-btn" onClick={onClose} aria-label="收起工作区">
              收起
            </button>
          )}
        </div>
      </header>

      <div
        className="workspace-body"
        role="tabpanel"
        id={`tabpanel-${tab}`}
        aria-labelledby={`tab-${tab}`}
      >
        {tab === 'cards' && <CardsView cards={cards} />}
        {tab === 'memory' && (
          <MemoryPanel
            items={memories}
            busy={memoryBusy}
            error={memoryError}
            onAdd={onAddMemory}
            onTogglePin={onToggleMemoryPin}
            onDelete={onDeleteMemory}
          />
        )}
        {tab === 'stars' && (
          <StarsPanel items={starred} onJump={onJumpTo} onUnstar={onUnstar} />
        )}
      </div>
    </section>
  )
}
