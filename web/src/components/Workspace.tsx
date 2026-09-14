import type { ReactNode } from 'react'
import { useCopy } from '../hooks/useCopy'
import { cardToText, formatAmount, statusTone, type WorkspaceCard } from '../workspace'
import { IconBox, IconCheck, IconCopy, IconDoc, IconTruck } from './icons'

/**
 * 右侧工作区：把这一轮对话"办的事"摆出来 —— 订单、物流轨迹、政策依据。
 *
 * 这里只放**用户该看的东西**。意图、节点路径、证据分数、工具名、耗时都是
 * 内部信息，一律不进这个面板（开发者走 /docs 和 /api/trace/{request_id}）。
 */

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

interface Props {
  cards: WorkspaceCard[]
  onClear: () => void
  /** 窄屏下工作区是抽屉，需要一个关闭入口 */
  onClose?: () => void
}

export function Workspace({ cards, onClear, onClose }: Props) {
  const summary = cards.reduce<Record<string, number>>((acc, c) => {
    acc[c.kind] = (acc[c.kind] ?? 0) + 1
    return acc
  }, {})

  return (
    <section className="workspace" aria-label="工作区">
      <header className="workspace-head">
        <h2 className="workspace-title">工作区</h2>
        {cards.length > 0 ? (
          <div className="workspace-tallies">
            {(Object.keys(KIND_META) as CardKind[])
              .filter((k) => summary[k])
              .map((k) => (
                <span key={k} className={`tally tally--${k}`}>
                  {KIND_META[k].label} {summary[k]}
                </span>
              ))}
          </div>
        ) : (
          <span className="workspace-count" />
        )}
        <div className="workspace-actions">
          {cards.length > 0 && (
            <button type="button" className="ghost-btn" onClick={onClear}>
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

      <div className="workspace-body">
        {cards.length === 0 ? (
          <div className="workspace-empty">
            <div className="grid-canvas" aria-hidden="true" />
            <p className="workspace-empty-text">
              对话里查到的订单、物流轨迹和政策依据
              <br />
              会陈列在这里
            </p>
          </div>
        ) : (
          <div className="card-stack">
            {cards.map((c) => (
              <Card key={c.id} card={c} />
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
