/**
 * 会话历史。
 *
 * 历史的真身在服务端（换设备、换浏览器都要在），这里只做"取回来 + 就地增删改"。
 * 改名和删除都**先打服务端再改本地**：反过来做（乐观更新）在失败时会把用户的
 * 历史列表和数据弄得不一致，而这类失败在演示环境里恰恰很常见（服务没起来）。
 *
 * 挂载时不主动拉：调用方在拿到 user_id 后调 `refresh()`。这样可以避免
 * "user_id 还没读出来就先发了一次空查询"。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteSession, fetchSessions, renameSession } from '../api/client'
import type { SessionBrief } from '../types'

export function useHistory(userId: string) {
  const [sessions, setSessions] = useState<SessionBrief[]>([])
  const [loading, setLoading] = useState(false)
  const inflight = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    if (!userId) return
    // 上一次还没回来就再来一次（连续两轮对话）时，取消旧的 ——
    // 不取消的话两次响应回来的顺序不确定，列表可能被旧数据覆盖回去。
    inflight.current?.abort()
    const controller = new AbortController()
    inflight.current = controller
    setLoading(true)
    try {
      const rows = await fetchSessions(userId, controller.signal)
      if (!controller.signal.aborted) setSessions(rows)
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }, [userId])

  useEffect(() => {
    void refresh()
    return () => inflight.current?.abort()
  }, [refresh])

  const rename = useCallback(
    async (sessionId: string, title: string) => {
      const ok = await renameSession(sessionId, title, userId)
      if (!ok) return false
      setSessions((prev) =>
        prev.map((s) => (s.sessionId === sessionId ? { ...s, title: title || s.title } : s)),
      )
      return true
    },
    [userId],
  )

  const remove = useCallback(
    async (sessionId: string) => {
      const ok = await deleteSession(sessionId, userId)
      if (!ok) return false
      setSessions((prev) => prev.filter((s) => s.sessionId !== sessionId))
      return true
    },
    [userId],
  )

  return { sessions, loading, refresh, rename, remove }
}
