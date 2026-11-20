/**
 * 后端接口客户端。
 *
 * 分五组：
 *   - 对话：`streamChat`（POST /api/chat 读 SSE）、`checkHealth`
 *   - 订单快捷选择：`fetchOrders`
 *   - 会话历史：列表 / 恢复 / 改名 / 删除
 *   - 长期记忆：列表 / 新增 / 置顶 / 删除
 *   - 知识节点图谱：`fetchKnowledgeGraph`
 *
 * SSE 走 fetch + ReadableStream 而不是 EventSource —— 因为要 POST 请求体，
 * 而 EventSource 只支持 GET。解析规则在 ./sse.ts（纯函数，测试会直接跑它）。
 *
 * 所有读接口都遵守同一条纪律：**拿不到就返回空，不抛**。面板少一块比整页白屏好。
 */

import type { HistoryMessage, MemoryItem, MessageStatus, SessionBrief, TurnOutcome } from '../types'
import { cardsFromEvidence, cardsFromHistory, cardsFromTools, type WorkspaceCard } from '../workspace'
import { drainEvents, normalizeSse } from './sse'

const API_BASE = ''

export interface AuthUser {
  id: string
  email: string
}

export async function currentUser(
  signal?: AbortSignal,
): Promise<{ user: AuthUser | null; error: string | null }> {
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, { signal, credentials: 'same-origin' })
    if (res.status === 401) return { user: null, error: null }
    if (!res.ok) return { user: null, error: '暂时无法连接服务，请检查网络后重试。' }
    const body = (await res.json()) as { authenticated?: boolean; user?: AuthUser | null }
    return { user: body.authenticated && body.user ? body.user : null, error: null }
  } catch {
    return { user: null, error: '暂时无法连接服务，请检查网络后重试。' }
  }
}

export async function authenticate(
  mode: 'login' | 'register',
  email: string,
  password: string,
): Promise<{ user: AuthUser | null; error: string }> {
  try {
    const res = await fetch(`${API_BASE}/api/auth/${mode}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ email, password }),
    })
    const body = (await res.json()) as { user?: AuthUser; detail?: string }
    return res.ok && body.user
      ? { user: body.user, error: '' }
      : { user: null, error: body.detail || '暂时无法登录，请稍后再试。' }
  } catch {
    return { user: null, error: '网络连接不太顺畅，请稍后再试。' }
  }
}

export async function logout(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/auth/logout`, {
      method: 'POST',
      credentials: 'same-origin',
    })
    return res.ok
  } catch {
    return false
  }
}

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
  requestId?: string
  /**
   * 引用提问：用户明确指向的那一句。
   *
   * 不是"多给点信息" —— 被引用的往往是**较早**的消息，它很可能已经滑出
   * 服务端的上下文窗口（`window_turns`），只靠会话历史根本看不到。
   * 所以它必须跟着这一次请求显式发出去。
   */
  quote?: string
  signal?: AbortSignal
}

export async function stopChat(turn: {
  sessionId: string; requestId: string; message: string; answer: string
}): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/chat/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        session_id: turn.sessionId, request_id: turn.requestId,
        message: turn.message.slice(0, 2000), answer: turn.answer,
      }),
    })
    return response.ok && ((await response.json()) as { ok?: boolean }).ok === true
  } catch {
    return false
  }
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
        request_id: options.requestId ?? '',
        quote: options.quote ?? '',
      }),
      credentials: 'same-origin',
      signal: options.signal,
    })
  } catch (err) {
    if (options.signal?.aborted || (err as Error)?.name === 'AbortError') return
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
          callbacks.onDone({
            handoff: Boolean((data as { handoff?: boolean })?.handoff),
            status: messageStatus((data as { status?: unknown })?.status),
          })
        } else if (event === 'error') {
          finished = true
          const message = (data as { message?: string })?.message
          callbacks.onError(message || '服务暂时不可用，请稍后再试。')
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
    callbacks.onDone({ handoff: false, status: 'interrupted' })
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
}

/**
 * 取当前用户的订单摘要，给快捷选择用。
 *
 * 拿不到就返回空列表 —— 选择器消失即可，不该因为它没加载出来就挡住整个对话。
 */
export async function fetchOrders(signal?: AbortSignal): Promise<OrderList> {
  try {
    const res = await fetch(`${API_BASE}/api/orders`, { signal })
    if (!res.ok) return { orders: [] }
    const body = (await res.json()) as Partial<OrderList>
    return { orders: Array.isArray(body.orders) ? body.orders : [] }
  } catch {
    return { orders: [] }
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

function messageStatus(value: unknown): MessageStatus {
  return value === 'verifying' || value === 'stopped' || value === 'interrupted' ? value : 'complete'
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

export async function fetchSessions(signal?: AbortSignal): Promise<SessionBrief[]> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions`,
      { signal, credentials: 'same-origin' },
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
  cards: WorkspaceCard[]
}

/** 恢复一次历史对话。拿不到就返回空 —— 界面退化成"新会话"即可。 */
export async function fetchSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<RestoredSession> {
  const empty: RestoredSession = { ok: false, title: '', messages: [], cards: [] }
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`,
      { signal, credentials: 'same-origin' },
    )
    if (!res.ok) return empty
    const body = (await res.json()) as Record<string, unknown>
    if (body.ok !== true) return empty
    const rows = Array.isArray(body.messages) ? body.messages.filter(isRecord) : []
    return {
      ok: true,
      title: str(body.title),
      cards: cardsFromHistory(rows),
      messages: rows.map((r) => ({
        role: r.role === 'user' ? 'user' : 'assistant',
        content: str(r.content),
        handoff: r.handoff === true,
        requestId: str(r.request_id),
        status: messageStatus(r.status),
      })),
    }
  } catch {
    return empty
  }
}

/** 正文 SSE 已结束，单独同步这一轮的落库/核验结果；不让光标为收尾继续转。 */
export async function waitForTurn(
  sessionId: string,
  requestId: string,
  signal: AbortSignal,
  onSnapshot: (session: RestoredSession, reply: HistoryMessage) => void,
): Promise<boolean> {
  const deadline = Date.now() + 30_000
  while (!signal.aborted && Date.now() < deadline) {
    const session = await fetchSession(sessionId, signal)
    if (signal.aborted) return false
    const reply = session.messages.find((m) => m.role === 'assistant' && m.requestId === requestId)
    if (reply) {
      onSnapshot(session, reply)
      if (reply.status !== 'verifying') return true
    }
    await new Promise<void>((resolve) => {
      const finish = () => {
        clearTimeout(timer)
        signal.removeEventListener('abort', finish)
        resolve()
      }
      const timer = setTimeout(finish, 250)
      signal.addEventListener('abort', finish, { once: true })
      if (signal.aborted) finish()
    })
  }
  return false
}

export async function renameSession(
  sessionId: string,
  title: string,
): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ title }),
      },
    )
    return res.ok && ((await res.json()) as { ok?: boolean }).ok === true
  } catch {
    return false
  }
}

export async function deleteSession(sessionId: string): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE', credentials: 'same-origin' },
    )
    return res.ok && ((await res.json()) as { ok?: boolean }).ok === true
  } catch {
    return false
  }
}

// ---------------------------------------------------------------
// 长期记忆
// ---------------------------------------------------------------

export async function fetchMemories(signal?: AbortSignal): Promise<MemoryItem[]> {
  try {
    const res = await fetch(
      `${API_BASE}/api/memories`,
      { signal, credentials: 'same-origin' },
    )
    if (!res.ok) return []
    return memoryItems(((await res.json()) as { items?: unknown }).items)
  } catch {
    return []
  }
}

/** 用户自己在面板里写一条备忘。写成功返回这条记忆。 */
export async function addMemory(text: string): Promise<MemoryItem | null> {
  try {
    const res = await fetch(`${API_BASE}/api/memories`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ text }),
    })
    if (!res.ok) return null
    return memoryItems([((await res.json()) as { item?: unknown }).item])[0] ?? null
  } catch {
    return null
  }
}

export async function setMemoryPinned(
  id: number,
  pinned: boolean,
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/memories/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ pinned }),
    })
    return res.ok
  } catch {
    return false
  }
}

export async function deleteMemory(id: number): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/api/memories/${id}`,
      { method: 'DELETE', credentials: 'same-origin' },
    )
    return res.ok
  } catch {
    return false
  }
}

// ---------------------------------------------------------------
// 知识节点图谱
// ---------------------------------------------------------------

export interface GraphLeaf {
  id: string
  label: string
  text: string
}

export interface GraphSection {
  id: string
  title: string
  sections: GraphSection[]
  leafs: GraphLeaf[]
}

export interface GraphDoc extends GraphSection {
  /** 文档名（doc_id），例如「退款政策」 */
  docId: string
}

export interface KnowledgeGraph {
  count: number
  docs: GraphDoc[]
}

/**
 * 取知识库的知识节点图谱。
 *
 * 拿不到就返回空树 —— 图谱退化为空状态即可，不该因为它没加载出来
 * 让整个工作区打不开。
 *
 * 但"空"有两种，界面要分开讲：**知识库本来就没有内容**（正常空状态）和
 * **这次请求失败了**（值得重试）。所以返回值上带一个 `failed`：调用方
 * 据此决定要不要出错误提示。返回值仍然遵守"不抛"的纪律 —— 多一个字段，
 * 不是多一种失败。
 */
export async function fetchKnowledgeGraph(
  signal?: AbortSignal,
): Promise<KnowledgeGraph & { failed: boolean }> {
  const empty: KnowledgeGraph = { count: 0, docs: [] }
  try {
    const res = await fetch(`${API_BASE}/api/knowledge/graph`, { signal })
    if (!res.ok) return { ...empty, failed: true }
    const body = (await res.json()) as Record<string, unknown>
    const rows = Array.isArray(body.docs) ? body.docs.filter(isRecord) : []
    return {
      count: Number(body.count) || 0,
      docs: rows.map((r) => ({
        id: str(r.id),
        docId: str(r.id),
        title: str(r.title) || '未命名',
        sections: graphSections(r.sections),
        leafs: graphLeafs(r.leafs),
      })),
      failed: false,
    }
  } catch {
    // 主动取消（切标签）不算失败：用户已经不看这个面板了
    if (signal?.aborted) return { ...empty, failed: false }
    return { ...empty, failed: true }
  }
}

function graphSections(raw: unknown): GraphSection[] {
  if (!Array.isArray(raw)) return []
  return raw.filter(isRecord).map((r) => ({
    id: str(r.id),
    title: str(r.title),
    sections: graphSections(r.sections),
    leafs: graphLeafs(r.leafs),
  }))
}

function graphLeafs(raw: unknown): GraphLeaf[] {
  if (!Array.isArray(raw)) return []
  return raw.filter(isRecord).map((r) => ({
    id: str(r.id),
    label: str(r.label),
    text: str(r.text),
  }))
}

// ---------------------------------------------------------------
// 答复反馈
// ---------------------------------------------------------------

/**
 * 赞 / 踩一条答复。
 *
 * 点踩必须带上 `session_id` 和 `request_id`：没有它们，后端拿不到"这次是哪句
 * 被踩了"，反馈就等于一句没有指向的"有人不满意"。至于点踩之后后端怎么处理，
 * 是内部流程，不在这里展开。
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
