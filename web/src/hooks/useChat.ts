/**
 * 会话状态机：消息列表 + 发送 + 流式收字。
 *
 * 界面上的所有状态都由这里派生，组件只负责画。这样"什么时候显示输入中""什么时候
 * 把气泡标记为完成"只有一个地方说了算，不会出现两个组件各判断一套的死角。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { streamChat } from '../api/client'
import type { ChatMessage } from '../types'
import { mergeCards, type WorkspaceCard } from '../workspace'

/** 会话 id 存在 localStorage：刷新页面后上下文还在，多轮追问才成立。 */
const SESSION_KEY = 'mewhelp.session_id'
const USER_KEY = 'mewhelp.user_id'

function randomId(prefix: string): string {
  const rand =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
      : Math.random().toString(36).slice(2, 14)
  return `${prefix}_${rand}`
}

function readOrCreate(key: string, prefix: string): string {
  try {
    const existing = window.localStorage.getItem(key)
    if (existing) return existing
    const created = randomId(prefix)
    window.localStorage.setItem(key, created)
    return created
  } catch {
    // 隐私模式下 localStorage 可能不可用 —— 退化成单次会话，不影响使用
    return randomId(prefix)
  }
}

/** 开场白：让用户知道这是谁能干什么，而不是对着空白框发呆。 */
const GREETING: ChatMessage = {
  id: 'greeting',
  role: 'assistant',
  kind: 'normal',
  content: '你好，我是 MewHelp 客服助手 👋\n可以帮你查订单、查物流、了解退换货和会员规则，也可以直接帮你转人工。',
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING])
  const [cards, setCards] = useState<WorkspaceCard[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('')
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    sessionIdRef.current = readOrCreate(SESSION_KEY, 'sess')
    userIdRef.current = readOrCreate(USER_KEY, 'user')
    return () => abortRef.current?.abort()
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
      const placeholder: ChatMessage = {
        id: replyId,
        role: 'assistant',
        kind: 'normal',
        content: '',
        streaming: true,
      }

      setMessages((prev) => [...prev, userMessage, placeholder])
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      const patchReply = (updater: (msg: ChatMessage) => ChatMessage) => {
        setMessages((prev) => prev.map((m) => (m.id === replyId ? updater(m) : m)))
      }

      await streamChat(
        text,
        {
          sessionId: sessionIdRef.current,
          userId: userIdRef.current,
          signal: controller.signal,
        },
        {
          onToken: (piece) => {
            patchReply((m) => ({ ...m, content: m.content + piece }))
          },
          onCards: (incoming) => {
            setCards((prev) => mergeCards(prev, incoming))
          },
          onDone: ({ handoff }) => {
            if (handoff) {
              // 转人工是真实业务结果，用户必须知道 —— 用一个独立提示条，不混进答复正文
              setMessages((prev) => [
                ...prev.map((m) => (m.id === replyId ? { ...m, streaming: false } : m)),
                {
                  id: randomId('n'),
                  role: 'assistant',
                  kind: 'notice',
                  content: '已为你转接人工客服，稍后会有同事继续跟进。',
                },
              ])
            } else {
              patchReply((m) => ({ ...m, streaming: false }))
            }
            setIsStreaming(false)
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
    [isStreaming],
  )

  /** 停止当前生成：把已收到的内容定格为最终答复，不再继续等。 */
  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)))
    setIsStreaming(false)
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setIsStreaming(false)
    try {
      window.localStorage.removeItem(SESSION_KEY)
    } catch {
      /* 忽略：清不掉也只是沿用旧会话 */
    }
    sessionIdRef.current = readOrCreate(SESSION_KEY, 'sess')
    setMessages([GREETING])
    setCards([])
  }, [])

  /** 只清工作区，不动对话 —— 卡片是参考资料，用户可能想边聊边清。 */
  const clearCards = useCallback(() => setCards([]), [])

  return { messages, cards, isStreaming, send, stop, reset, clearCards }
}
