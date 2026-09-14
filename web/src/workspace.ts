/**
 * 工作区卡片：把后端推来的原始数据整理成**用户看得懂**的卡片。
 *
 * 两条原则：
 *   1. **按数据形状判断，不按工具名判断。** 工具名是内部实现，改个名字不该
 *      让界面塌掉；而"有 items 数组"就是订单、"有 events 数组"就是物流轨迹，
 *      这是数据本身的性质。
 *   2. **不搬运内部字段。** 分数、chunk_id、工具名、耗时一律不进卡片。
 *
 * 这里是纯函数，不碰 DOM —— 便于单测，也便于 node 侧直接跑。
 */

export interface OrderItem {
  name: string
  qty: number
  price: number
}

export interface OrderCard {
  id: string
  kind: 'order'
  orderId: string
  statusLabel: string
  items: OrderItem[]
  totalAmount: number | null
  currency: string
  address: string | null
}

export interface LogisticsStep {
  time: string | null
  text: string
}

export interface LogisticsCard {
  id: string
  kind: 'logistics'
  orderId: string
  statusLabel: string
  carrier: string | null
  trackingNo: string | null
  steps: LogisticsStep[]
  note: string | null
}

export interface SourceCard {
  id: string
  kind: 'source'
  title: string
  snippet: string
}

/** 取消 / 退款 / 改地址这类**动了业务状态**的操作结果。 */
export interface ActionCard {
  id: string
  kind: 'action'
  orderId: string | null
  text: string
}

export type WorkspaceCard = OrderCard | LogisticsCard | SourceCard | ActionCard

// ---------------------------------------------------------------
// 取值助手：后端字段可能缺失，任何一处都不该让整张卡片渲染失败
// ---------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function str(v: unknown): string | null {
  return typeof v === 'string' && v.trim() ? v : null
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : []
}

/** 取第一个非空字符串 —— 用于在几个候选字段里挑"人话"文案。 */
function firstText(...values: unknown[]): string | null {
  for (const v of values) {
    const s = str(v)
    if (s) return s
  }
  return null
}

/** 金额显示：整数不拖小数点，有角分才保留两位。 */
export function formatAmount(amount: number | null, currency: string): string | null {
  if (amount === null) return null
  const symbol = currency === 'CNY' || !currency ? '¥' : `${currency} `
  const text = Number.isInteger(amount) ? String(amount) : amount.toFixed(2)
  return `${symbol}${text}`
}

// ---------------------------------------------------------------
// 状态色阶
// ---------------------------------------------------------------

/** 状态标签的语义色。同一件事在不同单上长得一样，用户才不用重读。 */
export type StatusTone = 'warn' | 'info' | 'active' | 'ok' | 'muted'

const TONE_RULES: ReadonlyArray<readonly [readonly string[], StatusTone]> = [
  [['待付款', '待处理', '待发货'], 'warn'],
  [['备货', '处理中', '拦截'], 'info'],
  [['已发货', '运输', '派送', '已揽收'], 'active'],
  [['已签收', '已完成', '成功'], 'ok'],
  [['已取消', '已关闭', '失败'], 'muted'],
]

/**
 * 由中文状态文案推出色阶。
 *
 * 刻意按**文案**而不是英文 status 判断：界面拿到的就是文案，
 * 英文状态码是后端实现细节，不该为了配色再漏一份到前端。
 */
export function statusTone(label: string): StatusTone {
  for (const [keywords, tone] of TONE_RULES) {
    if (keywords.some((k) => label.includes(k))) return tone
  }
  return 'muted'
}

// ---------------------------------------------------------------
// 构建
// ---------------------------------------------------------------

/**
 * 工具结果 → 卡片。
 *
 * 认不出来就返回 null（宁可不显示，也不要往工作区里塞一张半成品卡片）。
 */
export function cardFromToolResult(toolName: string, result: unknown): WorkspaceCard | null {
  if (!isRecord(result)) return null
  const orderId = str(result.order_id)

  // 物流轨迹：有 events 数组
  if (orderId && Array.isArray(result.events)) {
    const steps: LogisticsStep[] = arr(result.events).map((raw) => {
      const e: Record<string, unknown> = isRecord(raw) ? raw : {}
      return {
        time: firstText(e.time, e.at, e.timestamp),
        text: firstText(e.text, e.desc, e.description, e.status_label) ?? '—',
      }
    })
    const latest = firstText(result.latest)
    if (latest && steps.length === 0) steps.push({ time: null, text: latest })

    return {
      id: `logistics:${orderId}:${toolName}`,
      kind: 'logistics',
      orderId,
      statusLabel: firstText(result.status_label, result.status) ?? '—',
      carrier: str(result.carrier),
      trackingNo: firstText(result.tracking_no, result.waybill_no),
      steps,
      note: firstText(result.note),
    }
  }

  // 订单：有 items 数组
  if (orderId && Array.isArray(result.items)) {
    const items: OrderItem[] = arr(result.items).map((raw) => {
      const it: Record<string, unknown> = isRecord(raw) ? raw : {}
      return {
        name: firstText(it.name, it.title, it.sku) ?? '商品',
        qty: num(it.qty) ?? 1,
        price: num(it.price) ?? 0,
      }
    })
    return {
      id: `order:${orderId}`,
      kind: 'order',
      orderId,
      statusLabel: firstText(result.status_label, result.status) ?? '—',
      items,
      totalAmount: num(result.total_amount),
      currency: firstText(result.currency) ?? 'CNY',
      address: firstText(result.shipping_address, result.current_address, result.address),
    }
  }

  // 动了业务状态的操作（取消 / 退款 / 改地址）
  const text = firstText(result.refund_note, result.note, result.message)
  if (orderId || text) {
    const label = firstText(result.status_label)
    return {
      id: `action:${orderId ?? toolName}:${text ?? label ?? ''}`,
      kind: 'action',
      orderId,
      text: text ?? (label ? `订单当前状态：${label}` : '操作已受理。'),
    }
  }

  return null
}

/** 检索依据 → 卡片。同一篇只留一条，避免重复堆叠。 */
export function cardsFromEvidence(chunks: unknown): SourceCard[] {
  const seen = new Set<string>()
  const cards: SourceCard[] = []
  for (const raw of arr(chunks)) {
    if (!isRecord(raw)) continue
    const title = str(raw.title)
    const snippet = str(raw.snippet)
    if (!title || !snippet) continue
    if (seen.has(title)) continue
    seen.add(title)
    cards.push({ id: `source:${title}`, kind: 'source', title, snippet })
  }
  return cards
}

/**
 * 工具事件 → 卡片。
 *
 * 后端是**一次调用推一个 event**（data 是单个对象），但这里同时接受数组形态 ——
 * 契约哪天改成批量推，界面不至于悄悄变空。
 */
export function cardsFromTools(tools: unknown): WorkspaceCard[] {
  const list = Array.isArray(tools) ? tools : [tools]
  const cards: WorkspaceCard[] = []
  for (const raw of list) {
    if (!isRecord(raw)) continue
    if (raw.ok !== true) continue // 失败的不进工作区：卡片上没法体面地表达错误
    const name = str(raw.name) ?? ''
    const card = cardFromToolResult(name, raw.result)
    if (card) cards.push(card)
  }
  return cards
}

/** 新卡片并到列表最前面，同 id 覆盖（重复问同一单不该堆出一摞）。 */
export function mergeCards(existing: WorkspaceCard[], incoming: WorkspaceCard[]): WorkspaceCard[] {
  if (incoming.length === 0) return existing
  const fresh = new Map(incoming.map((c) => [c.id, c]))
  const kept = existing.filter((c) => !fresh.has(c.id))
  return [...fresh.values(), ...kept].slice(0, 20)
}

/**
 * 把卡片拍成一段纯文本。
 *
 * 给"复制"用 —— 用户拿到订单信息后的下一步，多半是粘到聊天窗口、
 * 工单系统或者备忘录里。让他用鼠标划选一张带排版的卡片是折磨。
 */
export function cardToText(card: WorkspaceCard): string {
  switch (card.kind) {
    case 'order': {
      const lines = [`订单 ${card.orderId}（${card.statusLabel}）`]
      for (const item of card.items) {
        const price = formatAmount(item.price, card.currency) ?? ''
        lines.push(`  ${item.name} ×${item.qty}  ${price}`.trimEnd())
      }
      const total = formatAmount(card.totalAmount, card.currency)
      if (total) lines.push(`  合计 ${total}`)
      if (card.address) lines.push(`  收货地址 ${card.address}`)
      return lines.join('\n')
    }
    case 'logistics': {
      const lines = [`物流 ${card.orderId}（${card.statusLabel}）`]
      if (card.carrier) lines.push(`  承运 ${card.carrier}`)
      if (card.trackingNo) lines.push(`  运单号 ${card.trackingNo}`)
      for (const step of card.steps) {
        lines.push(`  ${step.time ? `${step.time} ` : ''}${step.text}`)
      }
      if (card.note) lines.push(`  ${card.note}`)
      return lines.join('\n')
    }
    case 'source':
      return `${card.title}\n${card.snippet}`
    case 'action':
      return `${card.orderId ? `${card.orderId} ` : ''}${card.text}`
  }
}
