/**
 * 后端接口客户端。
 *
 * 分四组：
 *   - 对话：`streamChat`（POST /api/chat 读 SSE）、`checkHealth`
 *   - 订单快捷选择：`fetchOrders`
 *   - 会话历史：列表 / 恢复 / 改名 / 删除
 *   - 长期记忆：列表 / 新增 / 置顶 / 删除
 *
 * SSE 走 fetch + ReadableStream 而不是 EventSource —— 因为要 POST 请求体，
 * 而 EventSource 只支持 GET。解析规则在 ./sse.ts（纯函数，测试会直接跑它）。
 *
 * 所有读接口都遵守同一条纪律：**拿不到就返回空，不抛**。面板少一块比整页白屏好。
 */

import type { HistoryMessage, MemoryItem, SessionBrief, TurnOutcome } from '../types'
import { cardsFromEvidence, cardsFromTools, type WorkspaceCard } from '../workspace'
import { drainEvents, normalizeSse } from './sse'

const API_BASE = ''

export interface StreamCallbacks {
  /** 收到一段正文（逐字） */
  onToken: (text: string) => void
  /** 本轮新增的工作区卡片（订单 / 物流 / 依据） */
  onCards: (cards: WorkspaceCard[]) => void
  /** 用户档案有变化（全量）。没有档案的轮次不会推送这个回调。 */
  onMemory: (items: MemoryItem[]) => void
  /** 本轮的不透明句柄 —— 界面拿它做反馈与收藏 */
  onTurnId: (id: string) => void
  /** 本轮结束 */
  onDone: (outcome: TurnOutcome) => void
  /** 出错（网络/服务端），message 已是可直接展示给用户的人话 */
  onError: (message: string) => void
}

export interface ChatOptions {
  sessionId: string
  userId: string
  signal?: AbortSignal
}

export async function streamChat(
  message: string,
  options: ChatOptions,
  callbacks: StreamCallbacks,
): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        message,
        session_id: options.sessionId,
        user_id: options.userId,
      }),
      signal: options.signal,
    })
  } catch {
    callbacks.onError('网络连接不太顺畅，请检查网络后重试。')
    return
  }

  if (!response.ok || !response.body) {
    callbacks.onError('服务暂时不可用，请稍后再试。')
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finished = false

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      // stream: true —— 多字节中文被切在两个 chunk 之间时不能立刻解码
      buffer = normalizeSse(buffer + decoder.decode(value, { stream: true }))

      const { events, rest } = drainEvents(buffer)
      buffer = rest

      for (const { event, data } of events) {
        if (event === 'token') {
          const text = (data as { text?: string })?.text
          if (text) callbacks.onToken(text)
        } else if (event === 'evidence') {
          // 只取标题与原文摘录进工作区；分数/chunk_id 是内部口径，不渲染
          const cards = cardsFromEvidence((data as { chunks?: unknown })?.chunks)
          if (cards.length) callbacks.onCards(cards)
        } else if (event === 'tool') {
          const cards = cardsFromTools(data)
          if (cards.length) callbacks.onCards(cards)
        } else if (event === 'memory') {
          const items = memoryItems((data as { items?: unknown })?.items)
          callbacks.onMemory(items)
        } else if (event === 'done') {
          finished = true
          callbacks.onDone({ handoff: Boolean((data as { handoff?: boolean })?.handoff) })
        } else if (event === 'error') {
          finished = true
          callbacks.onError('服务出了点小问题，已为你转接人工，请稍候。')
        } else if (event === 'meta') {
          // meta 里绝大部分是意图/节点路径这类纯内部信息，前端刻意不消费；
          // 只有 request_id 是界面需要的 —— 反馈和收藏都要靠它当键。
          const id = (data as { request_id?: string })?.request_id
          if (id) callbacks.onTurnId(id)
        }
      }
    }
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') return
    callbacks.onError('回答中断了，请重试。')
    return
  }

  if (!finished) {
    // 流断了但没收到 done —— 仍要把已收到的内容收尾，避免气泡一直转
    callbacks.onDone({ handoff: false })
  }
}

export interface HealthInfo {
  ok: boolean
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthInfo> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal })
    return { ok: res.ok }
  } catch {
    return { ok: false }
  }
}

// ---------------------------------------------------------------
// 订单快捷选择
// ---------------------------------------------------------------

export interface OrderBrief {
  order_id: string
  status: string
  status_label: string
  total_amount: number
  currency: string
  item_count: number
  title: string
  created_at: string | null
}

export interface OrderList {
  orders: OrderBrief[]
  /** 这批单是不是"演示数据"（没有登录态时后端会标记） */
  demo: boolean
}

/**
 * 取当前用户的订单摘要，给快捷选择用。
 *
 * 拿不到就返回空列表 —— 选择器消失即可，不该因为它没加载出来就挡住整个对话。
 */
export async function fetchOrders(signal?: AbortSignal): Promise<OrderList> {
  try {
    const res = await fetch(`${API_BASE}/api/orders`, { signal })
    if (!res.ok) return { orders: [], demo: true }
    const body = (await res.json()) as Partial<OrderList>
    return {
      orders: Array.isArray(body.orders) ? body.orders : [],
      demo: body.demo !== false,
    }
  } catch {
    return { orders: [], demo: true }
  }
}

// ---------------------------------------------------------------
// 会话历史
// ---------------------------------------------------------------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

export function memoryItems(raw: unknown): MemoryItem[] {
  if (!Array.isArray(raw)) return []
  return raw.filter(isRecord).map((r) => ({
    id: Number(r.id) || 0,
    kind: str(r.kind) || 'preference',
    kindLabel: str(r.kind_label) || '记忆',
    text: str(r.text),
    source: str(r.source) || 'auto',
    pinned: r.pinned === true,
    hits: Number(r.hits) || 0,
    createdAt: str(r.created_at),
    updatedAt: str(r.updated_at),
  }))
}

function sessionBriefs(raw: unknown): SessionBrief[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter(isRecord)
    .map((r) => ({
      sessionId: str(r.session_id),
      title: str(r.title) || '未命名会话',
      turns: Number(r.turns) || 0,
      preview: str(r.preview),
      updatedAt: str(r.updated_at),
    }))
    .filter((s) => s.sessionId)
}

export async function fetchSessions(
  userId: string,
  signal?: AbortSignal,
): Promise<SessionBrief[]> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions?user_id=${encodeURIComponent(userId)}`,
      { signal },
    )
    if (!res.ok) return []
    return sessionBriefs(((await res.json()) as { sessions?: unknown }).sessions)
  } catch {
    return []
  }
}

export interface RestoredSession {
  ok: boolean
  title: string
  messages: HistoryMessage[]
}

/** 恢复一次历史对话。拿不到就返回空 —— 界面退化成"新会话"即可。 */
export async function fetchSession(
  sessionId: string,
  userId: string,
  signal?: AbortSignal,
): Promise<RestoredSession> {
  const empty: RestoredSession = { ok: false, title: '', messages: [] }
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
      { signal },
    )
    if (!res.ok) return empty
    const body = (await res.json()) as Record<string, unknown>
    if (body.ok !== true) return empty
    const rows = Array.isArray(body.messages) ? body.messages.filter(isRecord) : []
    return {
      ok: true,
      title: str(body.title),
      messages: rows.map((r) => ({
        role: r.role === 'user' ? 'user' : 'assistant',
        content: str(r.content),
        handoff: r.handoff === true,
        requestId: str(r.request_id),
      })),
    }
  } catch {
    return empty
  }
}

export async function renameSession(
  sessionId: string,
  title: string,
  userId: string,
): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      },
    )
    return res.ok && ((await res.json()) as { ok?: boolean }).ok === true
  } catch {
    return false
  }
}

export async function deleteSession(sessionId: string, userId: string): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    )
    return res.ok && ((await res.json()) as { ok?: boolean }).ok === true
  } catch {
    return false
  }
}

// ---------------------------------------------------------------
// 长期记忆
// ---------------------------------------------------------------

export async function fetchMemories(
  userId: string,
  signal?: AbortSignal,
): Promise<MemoryItem[]> {
  try {
    const res = await fetch(
      `${API_BASE}/api/memories?user_id=${encodeURIComponent(userId)}`,
      { signal },
    )
    if (!res.ok) return []
    return memoryItems(((await res.json()) as { items?: unknown }).items)
  } catch {
    return []
  }
}

/** 用户自己在面板里写一条备忘。写成功返回这条记忆。 */
export async function addMemory(userId: string, text: string): Promise<MemoryItem | null> {
  try {
    const res = await fetch(`${API_BASE}/api/memories`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, text }),
    })
    if (!res.ok) return null
    return memoryItems([((await res.json()) as { item?: unknown }).item])[0] ?? null
  } catch {
    return null
  }
}

export async function setMemoryPinned(
  userId: string,
  id: number,
  pinned: boolean,
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/memories/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, pinned }),
    })
    return res.ok
  } catch {
    return false
  }
}

export async function deleteMemory(userId: string, id: number): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/memories/${id}?user_id=${encodeURIComponent(userId)}`,
      { method: 'DELETE' },
    )
    return res.ok
  } catch {
    return false
  }
}

// ---------------------------------------------------------------
// 答复反馈
// ---------------------------------------------------------------

/**
 * 赞 / 踩一条答复。
 *
 * 点踩在后端会落到"问题池"，所以必须带上 `session_id` 和 `request_id` ——
 * 没有它们，运营只看到"有人不满意"，却不知道是哪一句答得不好。
 * 仓库里 `/api/feedback` 早就有了，界面一直没接上；这里把它补全。
 */
export async function sendFeedback(
  sessionId: string,
  requestId: string,
  thumbsDown: boolean,
  comment?: string,
): Promise<boolean> {
  if (!requestId) return false
  try {
    const res = await fetch(`${API_BASE}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        request_id: requestId,
        thumbs_down: thumbsDown,
        comment: comment ?? null,
      }),
    })
    return res.ok
  } catch {
    return false
  }
}
