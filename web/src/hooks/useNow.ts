import { useEffect, useState } from 'react'

/**
 * 一个每分钟（默认 30 秒）跳一次的"现在"。
 *
 * 存在的理由很具体：历史栏里写的是"3 分钟前"，如果时间戳只在数据变化时算一次，
 * 用户开着页面十分钟不动，那一行会一直显示"刚刚" —— 一个**慢慢变成谎话**的
 * 相对时间，比绝对时间还糟。
 *
 * 间隔 30 秒是刻意的：相对时间的粒度是"分钟"，秒级刷新既没必要，
 * 又会让整棵树每秒钟重渲染一次。
 */
export function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])

  return now
}
