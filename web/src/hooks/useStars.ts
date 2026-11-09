/**
 * 收藏（星标）的消息。
 *
 * 为什么存在浏览器本地、而不是像会话历史那样放服务端：
 * 收藏是**我自己的便利贴**，它描述的是"我想再翻回来看这条"，属于个人视图；
 * 而聊天记录本身必须跨设备可见。两者的归属不同，存储位置也就不同。
 *
 * 键取 `turnId`（后端的不透明句柄）而不是消息在列表里的下标：
 * 下标会随新消息插入而整体错位，收藏会自己跳到别的消息上。
 */

import { useCallback, useEffect, useState } from 'react'


/** 上限。没有上限的本地列表会一直涨，而旧到看不见的收藏本来也没什么用。 */
const MAX_STARS = 200

function read(storageKey: string): string[] {
  try {
    const raw = window.localStorage.getItem(storageKey)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.filter((v) => typeof v === 'string') : []
  } catch {
    return []
  }
}

export function useStars(accountId: string) {
  const storageKey = `mewhelp.stars.${accountId}`
  const [ids, setIds] = useState<string[]>(() => read(storageKey))

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(ids))
    } catch {
      /* 存不下就只在本次会话生效 */
    }
  }, [ids, storageKey])

  const has = useCallback((turnId?: string) => Boolean(turnId && ids.includes(turnId)), [ids])

  const toggle = useCallback((turnId?: string) => {
    if (!turnId) return
    setIds((prev) =>
      prev.includes(turnId)
        ? prev.filter((id) => id !== turnId)
        : [turnId, ...prev].slice(0, MAX_STARS),
    )
  }, [])

  return { ids, has, toggle }
}
