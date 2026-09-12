/**
 * 后端接口客户端。
 *
 * 只用两个能力：
 *   - `streamChat`：POST /api/chat，读 SSE，逐字回调答复文本
 *   - `checkHealth`：GET /api/health，用于页头的「在线」状态灯
 *
 * SSE 走 fetch + ReadableStream 而不是 EventSource —— 因为要 POST 请求体，
 * 而 EventSource 只支持 GET。解析规则在 ./sse.ts（纯函数，测试会直接跑它）。
 */

import type { TurnOutcome } from '../types'
import { cardsFromEvidence, cardsFromTools, type WorkspaceCard } from '../workspace'
import { drainEvents, normalizeSse } from './sse'

const API_BASE = ''

export interface StreamCallbacks {
  /** 收到一段正文（逐字） */
  onToken: (text: string) => void
  /** 本轮新增的工作区卡片（订单 / 物流 / 依据） */
  onCards: (cards: WorkspaceCard[]) => void
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
        } else if (event === 'done') {
          finished = true
          callbacks.onDone({ handoff: Boolean((data as { handoff?: boolean })?.handoff) })
        } else if (event === 'error') {
          finished = true
          callbacks.onError('服务出了点小问题，已为你转接人工，请稍候。')
        }
        // meta：意图/节点路径，纯内部信息，前端刻意不消费
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
