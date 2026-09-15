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
import { fetchSession, sendFeedback, streamChat } from '../api/client'
import type { ChatMessage, MemoryItem, Rating } from '../types'
import { mergeCards, type WorkspaceCard } from '../workspace'

function randomId(prefix: string): string {
  const rand =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
      : Math.random().toString(36).slice(2, 14)
  return `${prefix}_${rand}`
}

/** 开场白：让用户知道这是谁能干什么，而不是对着空白框发呆。 */
const GREETING: ChatMessage = {
  id: 'greeting',
  role: 'assistant',
  kind: 'normal',
  content: '你好，我是 MewHelp 客服助手 👋\n可以帮你查订单、查物流、了解退换货和会员规则，也可以直接帮你转人工。',
}

/** 转人工时补的那条提示 —— 恢复历史时也要补回来，否则用户会以为没转成。 */
const HANDOFF_NOTICE = '已为你转接人工客服，稍后会有同事继续跟进。'

function notice(content: string): ChatMessage {
  return { id: randomId('n'), role: 'assistant', kind: 'notice', content }
}

/** 服务端的历史消息 → 界面消息。 */
function fromHistory(rows: Array<{ role: string; content: string; handoff: boolean; requestId: string }>): ChatMessage[] {
  const out: ChatMessage[] = []
  rows.forEach((r, i) => {
    const role = r.role === 'user' ? 'user' : 'assistant'
    out.push({
      id: `h${i}-${role}`,
      role,
      kind: 'normal',
      content: r.content,
      // 句柄要带回来：历史里的消息也要能点赞、能收藏
      turnId: r.requestId || undefined,
    })
    if (role === 'assistant' && r.handoff) out.push(notice(HANDOFF_NOTICE))
  })
  return out
}

interface Options {
  userId: string
  sessionId: string
  /** 当前正在引用的那条消息的正文（引用提问） */
  quote?: string
  /** 每轮结束后回调（用来刷新历史列表） */
  onTurnEnd?: () => void
  /** 服务端推来全量档案（用来更新记忆面板） */
  onMemory?: (items: MemoryItem[]) => void
}

export function useChat({ userId, sessionId, quote, onTurnEnd, onMemory }: Options) {
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING])
  const [cards, setCards] = useState<WorkspaceCard[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
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

  /** 换会话：中断在途请求、清空工作区、把历史拉回来。 */
  useEffect(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
    setCards([])
    setMessages([GREETING])
    sentRef.current = false

    if (!sessionId || !userId) return
    const controller = new AbortController()
    let alive = true
    void fetchSession(sessionId, userId, controller.signal).then((restored) => {
      // 期间又切走了、或者用户已经开口了，就当没发生过 ——
      // 前者会把旧会话的消息画到新会话里，后者会抹掉用户刚打的字
      if (!alive || controller.signal.aborted || sentRef.current) return
      if (restored.ok && restored.messages.length > 0) {
        setMessages(fromHistory(restored.messages))
      }
    })
    return () => {
      alive = false
      controller.abort()
    }
  }, [sessionId, userId])

  useEffect(() => () => abortRef.current?.abort(), [])

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
      const placeholder: ChatMessage = {
        id: replyId,
        role: 'assistant',
        kind: 'normal',
        content: '',
        streaming: true,
      }

      setMessages((prev) => [...prev, userMessage, placeholder])
      setIsStreaming(true)
      sentRef.current = true

      const controller = new AbortController()
      abortRef.current = controller

      const patchReply = (updater: (msg: ChatMessage) => ChatMessage) => {
        setMessages((prev) => prev.map((m) => (m.id === replyId ? updater(m) : m)))
      }

      await streamChat(
        text,
        { sessionId, userId, quote, signal: controller.signal },
        {
          onToken: (piece) => {
            patchReply((m) => ({ ...m, content: m.content + piece }))
          },
          onCards: (incoming) => {
            setCards((prev) => mergeCards(prev, incoming))
          },
          onMemory: (items) => {
            onMemoryRef.current?.(items)
          },
          onTurnId: (id) => {
            // 反馈和收藏都要用它当键，所以一拿到就挂到气泡上
            patchReply((m) => ({ ...m, turnId: id }))
          },
          onDone: ({ handoff }) => {
            if (handoff) {
              // 转人工是真实业务结果，用户必须知道 —— 用一个独立提示条，不混进答复正文
              setMessages((prev) => [
                ...prev.map((m) => (m.id === replyId ? { ...m, streaming: false } : m)),
                notice(HANDOFF_NOTICE),
              ])
            } else {
              patchReply((m) => ({ ...m, streaming: false }))
            }
            setIsStreaming(false)
            onTurnEndRef.current?.()
          },
          onError: (message) => {
            patchReply((m) => ({
              ...m,
              streaming: false,
              kind: 'notice',
              content: message,
            }))
            setIsStreaming(false)
          },
        },
      )
    },
    [isStreaming, sessionId, userId, quote],
  )

  /**
   * 停止当前生成：把已收到的内容定格为最终答复，不再继续等。
   *
   * 一个字都没收到的那种"停止"要单独处理：占位气泡留着就是一个永久空白框 ——
   * 它没有重问按钮（重问挂在用户消息上），也没有任何可读内容，用户只会以为
   * 界面坏了。这种气泡直接撤掉，连同它对应的那条用户消息一起 —— 留着用户消息
   * 而拿走回复，看起来像"我说了话但没人理"。整轮撤掉，用户可以重打一次。
   *
   * （真模型下这个窗口是真实存在的：答复算完到第一个 token 之间还要跑一次
   * 记忆抽取，那段时间界面正好停在一个空气泡上。）
   */
  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (last?.streaming && !last.content) {
        const trimmed = prev.slice(0, -1)
        const prevMsg = trimmed[trimmed.length - 1]
        return prevMsg?.role === 'user' ? trimmed.slice(0, -1) : trimmed
      }
      return prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
    })
    setIsStreaming(false)
  }, [])

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
