import { formatAmount, type WorkspaceCard } from '../workspace'

/**
 * 右侧工作区：把这一轮对话"办的事"摆出来 —— 订单、物流轨迹、政策依据。
 *
 * 这里只放**用户该看的东西**。意图、节点路径、证据分数、工具名、耗时都是
 * 内部信息，一律不进这个面板（开发者走 /docs 和 /api/trace/{request_id}）。
 */

interface CardProps<T> {
  card: T
}

function OrderView({ card }: CardProps<Extract<WorkspaceCard, { kind: 'order' }>>) {
  const total = formatAmount(card.totalAmount, card.currency)
  return (
    <article className="card">
      <header className="card-head">
        <span className="card-kind">订单</span>
        <span className="card-id">{card.orderId}</span>
        <span className="pill">{card.statusLabel}</span>
      </header>

      {card.items.length > 0 && (
        <ul className="item-list">
          {card.items.map((it, i) => (
            <li key={`${it.name}-${i}`} className="item">
              <span className="item-name">{it.name}</span>
              <span className="item-qty">×{it.qty}</span>
              <span className="item-price">
                {formatAmount(it.price, card.currency) ?? '—'}
              </span>
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
    </article>
  )
}

function LogisticsView({ card }: CardProps<Extract<WorkspaceCard, { kind: 'logistics' }>>) {
  return (
    <article className="card">
      <header className="card-head">
        <span className="card-kind">物流</span>
        <span className="card-id">{card.orderId}</span>
        <span className="pill">{card.statusLabel}</span>
      </header>

      {(card.carrier || card.trackingNo) && (
        <dl className="kv">
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
    </article>
  )
}

function SourceView({ card }: CardProps<Extract<WorkspaceCard, { kind: 'source' }>>) {
  return (
    <article className="card card--source">
      <header className="card-head">
        <span className="card-kind">依据</span>
        <span className="card-title">{card.title}</span>
      </header>
      <p className="card-snippet">{card.snippet}</p>
    </article>
  )
}

function ActionView({ card }: CardProps<Extract<WorkspaceCard, { kind: 'action' }>>) {
  return (
    <article className="card card--action">
      <header className="card-head">
        <span className="card-kind">处理结果</span>
        {card.orderId && <span className="card-id">{card.orderId}</span>}
      </header>
      <p className="card-snippet">{card.text}</p>
    </article>
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
  return (
    <section className="workspace" aria-label="工作区">
      <header className="workspace-head">
        <h2 className="workspace-title">工作区</h2>
        <span className="workspace-count">{cards.length > 0 ? `${cards.length} 项` : ''}</span>
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
            {/* 呼应参考图的透视网格：暗示"这里是块画布，等着放东西" */}
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
