/**
 * 会话状态机：消息列表 + 发送 + 流式收字 + 历史恢复。
 *
 * 界面上的所有状态都由这里派生，组件只负责画。这样"什么时候显示输入中""什么时候
 * 把气泡标记为完成"只有一个地方说了算，不会出现两个组件各判断一套的死角。
 *
 * 两条与"历史"有关的纪律：
 *
 * 1. **恢复会话时只认服务端**。消息不做本地持久化 —— 两份真相（本地一份、
 *    服务端一份）在改了接口之后必然对不上，而对不上的表现是"刷新之后少了
 *    几句"，极难复现。
 * 2. **切会话要中断正在进行的这一轮**。不中断的话，上一个会话的 token 会
 *    落到新会话的气泡里，看起来像串台。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchMemories, fetchSession, sendFeedback, stopChat, streamChat, waitForTurn } from '../api/client'
import type { ChatMessage, MemoryItem, Rating } from '../types'
import { mergeCards, startWorkspaceTurn, type WorkspaceCard } from '../workspace'
import { stopMessages } from '../chat-state'

function randomId(prefix: string): string {
  const rand =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
      : Math.random().toString(36).slice(2, 14)
  return `${prefix}_${rand}`
}

/**
 * 空对话就是空对话。
 *
 * 上一版这里塞了一条开场白消息（"你好，我是 MewHelp 客服助手 👋"），
 * 让它长得像一条真答复。问题在于它**占着最新一条的位置**：用户发出第一句，
 * 真消息把它顶上去，看起来像一句话凭空消失了；而且它和真答复同形同色，
 * 用户分不清"已经聊过"还是"还没开始"。
 *
 * 现在首屏交给 EmptyState（不属于对话流的另一块区域），消息列表老实为空。
 * 好处不只是视觉：`messages.length === 0` 从此就是"这个会话还没开始"
 * 的准确判据，不用再靠 id === 'greeting' 这种字符串去认。
 */

/** 转人工时补的那条提示 —— 恢复历史时也要补回来，否则用户会以为没转成。 */
const HANDOFF_NOTICE = '已为你转接人工客服，稍后会有同事继续跟进。'

function notice(content: string): ChatMessage {
  return { id: randomId('n'), role: 'assistant', kind: 'notice', content }
}

/** 服务端的历史消息 → 界面消息。 */
function fromHistory(rows: Array<{ role: string; content: string; handoff: boolean; requestId: string; status?: ChatMessage['status'] }>): ChatMessage[] {
  const out: ChatMessage[] = []
  rows.forEach((r, i) => {
    const role = r.role === 'user' ? 'user' : 'assistant'
    out.push({
      id: `h${i}-${role}`,
      role,
      kind: 'normal',
      content: r.content,
      status: r.status,
      // 句柄要带回来：历史里的消息也要能点赞、能收藏
      turnId: r.requestId || undefined,
    })
    if (role === 'assistant' && r.handoff) out.push({ ...notice(HANDOFF_NOTICE), id: `handoff-${r.requestId}` })
  })
  return out
}

interface Options {
  sessionId: string
  /** 当前正在引用的那条消息的正文（引用提问） */
  quote?: string
  /** 每轮结束后回调（用来刷新历史列表） */
  onTurnEnd?: () => void
  /** 服务端推来全量档案（用来更新记忆面板） */
  onMemory?: (items: MemoryItem[]) => void
}

export function useChat({ sessionId, quote, onTurnEnd, onMemory }: Options) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [cards, setCards] = useState<WorkspaceCard[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const activeTurnRef = useRef<{
    requestId: string; replyId: string; sessionId: string; message: string; answer: string
  } | null>(null)
  const syncsRef = useRef(new Map<string, AbortController>())
  const sessionRef = useRef(sessionId)
  sessionRef.current = sessionId
  const latestReplyRef = useRef('')
  //: 用户是否已经在本会话里发过消息。恢复历史是异步的，如果它比用户的手速慢，
  //: 回来的那一刻会把刚发出去的消息连同占位气泡一起抹掉 —— 用户看到自己打的字
  //: 凭空消失了。这个标记就是那道"晚到的响应不许覆盖新输入"的护栏。
  const sentRef = useRef(false)

  // 回调放进 ref：它们变化时不该触发"重新发送"或"重新恢复历史"
  const onTurnEndRef = useRef(onTurnEnd)
  const onMemoryRef = useRef(onMemory)
  useEffect(() => {
    onTurnEndRef.current = onTurnEnd
    onMemoryRef.current = onMemory
  }, [onTurnEnd, onMemory])

  const syncTurn = useCallback((sid: string, requestId: string, replyId: string) => {
    if (syncsRef.current.has(requestId)) return
    const controller = new AbortController()
    syncsRef.current.set(requestId, controller)
    void waitForTurn(sid, requestId, controller.signal, (session, reply) => {
      if (sessionRef.current !== sid) return
      setMessages((prev) => {
        const updated = prev.map((m) => m.id === replyId
          ? { ...m, content: reply.content, streaming: false, status: reply.status, turnId: requestId }
          : m)
        const noticeId = `handoff-${requestId}`
        if (reply.handoff && !updated.some((m) => m.id === noticeId)) {
          const index = updated.findIndex((m) => m.id === replyId)
          if (index !== -1) updated.splice(index + 1, 0, { ...notice(HANDOFF_NOTICE), id: noticeId })
        }
        return updated
      })
      if (latestReplyRef.current === replyId) setCards(session.cards)
    }).then(async (saved) => {
      if (controller.signal.aborted) return
      if (saved) {
        onTurnEndRef.current?.()
        const items = await fetchMemories(controller.signal)
        if (!controller.signal.aborted) onMemoryRef.current?.(items)
      } else if (sessionRef.current === sid) {
        setMessages((prev) => [...prev, notice('暂时未能确认这轮对话的保存结果，请稍后刷新历史。')])
      }
    }).finally(() => syncsRef.current.delete(requestId))
  }, [])

  /** 换会话：中断在途请求、清空工作区、把历史拉回来。 */
  useEffect(() => {
    const previous = activeTurnRef.current
    abortRef.current?.abort()
    abortRef.current = null
    activeTurnRef.current = null
    if (previous) {
      void stopChat(previous).then(() => syncTurn(previous.sessionId, previous.requestId, previous.replyId))
    }
    setIsStreaming(false)
    setCards([])
    setMessages([])
    sentRef.current = false

    if (!sessionId) return
    const controller = new AbortController()
    let alive = true
    void fetchSession(sessionId, controller.signal).then((restored) => {
      // 期间又切走了、或者用户已经开口了，就当没发生过 ——
      // 前者会把旧会话的消息画到新会话里，后者会抹掉用户刚打的字
      if (!alive || controller.signal.aborted || sentRef.current) return
      if (restored.ok && restored.messages.length > 0) {
        setMessages(fromHistory(restored.messages))
        setCards(restored.cards)
        restored.messages.forEach((message, i) => {
          if (message.role === 'assistant' && message.status === 'verifying') {
            syncTurn(sessionId, message.requestId, `h${i}-assistant`)
          }
        })
      }
    })
    return () => {
      alive = false
      controller.abort()
    }
  }, [sessionId, syncTurn])

  useEffect(() => () => {
    abortRef.current?.abort()
    if (activeTurnRef.current) void stopChat(activeTurnRef.current)
    for (const controller of syncsRef.current.values()) controller.abort()
  }, [])

  const send = useCallback(
    async (rawText: string) => {
      const text = rawText.trim()
      if (!text || isStreaming) return

      const userMessage: ChatMessage = {
        id: randomId('u'),
        role: 'user',
        kind: 'normal',
        content: text,
      }
      const replyId = randomId('a')
      const requestId = randomId('req')
      latestReplyRef.current = replyId
      const placeholder: ChatMessage = {
        id: replyId,
        role: 'assistant',
        kind: 'normal',
        content: '',
        streaming: true,
        turnId: requestId,
      }

      setMessages((prev) => [...prev, userMessage, placeholder])
      setIsStreaming(true)
      sentRef.current = true
      setCards(startWorkspaceTurn)

      const controller = new AbortController()
      abortRef.current = controller
      activeTurnRef.current = { requestId, replyId, sessionId, message: text, answer: '' }

      const patchReply = (updater: (msg: ChatMessage) => ChatMessage) => {
        if (controller.signal.aborted || sessionRef.current !== sessionId) return
        setMessages((prev) => prev.map((m) => (m.id === replyId ? updater(m) : m)))
      }

      await streamChat(
        text,
        { sessionId, requestId, quote, signal: controller.signal },
        {
          onToken: (piece) => {
            if (!controller.signal.aborted && activeTurnRef.current?.requestId === requestId) {
              activeTurnRef.current.answer += piece
            }
            patchReply((m) => ({ ...m, content: m.content + piece }))
          },
          onCards: (incoming) => {
            if (controller.signal.aborted || sessionRef.current !== sessionId) return
            setCards((prev) => mergeCards(prev, incoming))
          },
          onMemory: (items) => {
            if (controller.signal.aborted || sessionRef.current !== sessionId) return
            onMemoryRef.current?.(items)
          },
          onTurnId: (id) => {
            // 反馈和收藏都要用它当键，所以一拿到就挂到气泡上
            patchReply((m) => ({ ...m, turnId: id }))
          },
          onDone: ({ handoff, status }) => {
            if (controller.signal.aborted || sessionRef.current !== sessionId) return
            if (handoff) {
              // 转人工是真实业务结果，用户必须知道 —— 用一个独立提示条，不混进答复正文
              setMessages((prev) => [
                ...prev.map((m) => (m.id === replyId ? { ...m, streaming: false, status } : m)),
                { ...notice(HANDOFF_NOTICE), id: `handoff-${requestId}` },
              ])
            } else {
              patchReply((m) => ({ ...m, streaming: false, status }))
            }
            setIsStreaming(false)
            activeTurnRef.current = null
            syncTurn(sessionId, requestId, replyId)
          },
          onError: (message) => {
            if (controller.signal.aborted || sessionRef.current !== sessionId) return
            patchReply((m) => ({
              ...m,
              streaming: false,
              status: 'interrupted',
              content: m.content || message,
            }))
            setIsStreaming(false)
            activeTurnRef.current = null
            syncTurn(sessionId, requestId, replyId)
          },
        },
      )
    },
    [isStreaming, sessionId, quote, syncTurn],
  )

  /** 停止模型请求，保留这一轮；随后按同一个问答 ID 同步后端保存结果。 */
  const stop = useCallback(() => {
    const turn = activeTurnRef.current
    abortRef.current?.abort()
    abortRef.current = null
    activeTurnRef.current = null
    setMessages(stopMessages)
    setIsStreaming(false)
    if (turn) void stopChat(turn).then(() => syncTurn(turn.sessionId, turn.requestId, turn.replyId))
  }, [syncTurn])

  /** 重问上一条：把最后一条用户消息之后的对话丢掉，再原样问一次。 */
  const retryLast = useCallback(() => {
    if (isStreaming) return
    const lastUser = messages.map((m) => m.role).lastIndexOf('user')
    if (lastUser === -1) return
    const text = messages[lastUser].content
    setMessages((prev) => prev.slice(0, lastUser))
    void send(text)
  }, [isStreaming, messages, send])

  /**
   * 给一条答复点赞 / 点踩。
   *
   * 界面状态先改（点下去要立刻有反馈），再把评价发给后端。发失败不回滚 ——
   * 对用户来说，"我点了个踩但它没生效"没有任何可操作的补救，回滚反而显得像点错了。
   */
  const rate = useCallback(
    (message: ChatMessage, rating: Rating) => {
      const next = message.rating === rating ? undefined : rating
      setMessages((prev) =>
        prev.map((m) => (m.id === message.id ? { ...m, rating: next } : m)),
      )
      if (next && message.turnId) {
        void sendFeedback(sessionId, message.turnId, next === 'down')
      }
    },
    [sessionId],
  )

  /** 只清工作区，不动对话 —— 卡片是参考资料，用户可能想边聊边清。 */
  const clearCards = useCallback(() => setCards([]), [])

  return {
    messages,
    cards,
    isStreaming,
    canRetry: messages.some((m) => m.role === 'user'),
    send,
    stop,
    retryLast,
    rate,
    clearCards,
  }
}
