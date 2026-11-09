/**
 * 身份：这个浏览器会话是谁、正在哪一次会话里。
 *
 * 两者都存在 localStorage 里：
 *   - user_id   —— "我是谁"。长期记忆挂在它下面，所以它必须跨会话稳定；
 *                  重新生成一个 id 等于换了一个人，用户会发现自己写的备忘全没了。
 *   - session_id —— "我正在哪次对话里"。多轮上下文挂在它下面。
 *
 * 抽成独立钩子而不是塞在 useChat 里，是因为"会话历史"和"长期记忆"两栏都要
 * 用到 user_id，而它们都必须在 useChat 之前拿到它 —— 依赖方向清楚了，
 * 才不会出现"先建连接后拿身份"这种顺序错。
 */

import { useCallback, useState } from 'react'


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

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    /* 存不下就只在本次会话生效 */
  }
}

export function useIdentity(accountId: string) {
  const sessionKey = `mewhelp.session_id.${accountId}`
  const [sessionId, setSessionIdState] = useState(() => readOrCreate(sessionKey, 'sess'))

  /** 切到另一个会话（点历史里的某一条）。 */
  const setSessionId = useCallback((id: string) => {
    if (!id) return
    write(sessionKey, id)
    setSessionIdState(id)
  }, [sessionKey])

  /** 开一次全新的会话。id 现生成，历史里会在第一条消息落地后才出现。 */
  const startNewSession = useCallback(() => {
    const id = randomId('sess')
    write(sessionKey, id)
    setSessionIdState(id)
    return id
  }, [sessionKey])

  return { sessionId, setSessionId, startNewSession }
}
