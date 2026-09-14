/**
 * 长期记忆面板的状态。
 *
 * 数据的来源有两个，这里必须说清楚谁说了算：
 *
 *   - **服务端推来的全量**（每轮对话的 `memory` 事件）→ `replace()`
 *   - **面板上的操作**（新增 / 钉住 / 删除）→ 就地改本地，并打服务端
 *
 * 统一走"全量替换"而不是增量合并：增量要求前后端对"什么算一次变化"达成一致，
 * 一旦某次事件丢了，面板和真实记忆就会永久对不上，而且没有任何报错。
 * 全量替换最坏情况只是慢一拍。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { addMemory, deleteMemory, fetchMemories, setMemoryPinned } from '../api/client'
import type { MemoryItem } from '../types'

export function useMemory(userId: string) {
  const [items, setItems] = useState<MemoryItem[]>([])
  const [busy, setBusy] = useState(false)
  //: 还没拿到过第一批数据。用来决定显示骨架屏还是"还没有记住任何事" ——
  //: 两者差一个字，但对用户的意义完全相反（"在加载"vs"是空的"）。
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const inflight = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    if (!userId) return
    inflight.current?.abort()
    const controller = new AbortController()
    inflight.current = controller
    try {
      const rows = await fetchMemories(userId, controller.signal)
      if (!controller.signal.aborted) {
        setItems(rows)
        setLoading(false)
      }
    } finally {
      /* 取消时不动状态：新的一次请求会填上 */
    }
  }, [userId])

  useEffect(() => {
    void refresh()
    return () => inflight.current?.abort()
  }, [refresh])

  /** 服务端推来的全量档案 —— 直接替换，不做合并。 */
  const replace = useCallback((next: MemoryItem[]) => {
    setItems(next)
    setLoading(false)
  }, [])

  const add = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return false
      setBusy(true)
      setError('')
      const created = await addMemory(userId, trimmed)
      setBusy(false)
      if (!created) {
        setError('没能保存，稍后再试')
        return false
      }
      // 服务端算的 id 才是真的（它要去重、要按 (user_id, mem_key) 合并），
      // 所以这里用返回值而不是本地拼一条。
      setItems((prev) => {
        const rest = prev.filter((i) => i.id !== created.id)
        return [created, ...rest]
      })
      return true
    },
    [userId],
  )

  const togglePin = useCallback(
    async (item: MemoryItem) => {
      const next = !item.pinned
      setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, pinned: next } : i)))
      const ok = await setMemoryPinned(userId, item.id, next)
      if (!ok) {
        // 服务端没成功就翻回去 —— 留着一个假的"已钉住"，用户下次打开会发现
        // 它自己掉了，那比当场回弹更让人困惑。
        setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, pinned: !next } : i)))
        setError('操作没成功，稍后再试')
      }
    },
    [userId],
  )

  const remove = useCallback(
    async (item: MemoryItem) => {
      setItems((prev) => prev.filter((i) => i.id !== item.id))
      const ok = await deleteMemory(userId, item.id)
      if (!ok) {
        void refresh() // 删失败就把真实状态拉回来
        setError('删除没成功，稍后再试')
      }
      return ok
    },
    [userId, refresh],
  )

  return { items, loading, busy, error, refresh, replace, add, togglePin, remove }
}
