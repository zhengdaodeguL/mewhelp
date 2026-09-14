/**
 * 轻量提示条（toast）。
 *
 * 为什么是一个模块级的订阅表，而不是 React Context 或者一路往下传的 prop：
 *
 *   * Context 要求把 Provider 包在最外层，然后每个想弹提示的组件都要 `useContext`
 *     —— 对一个"偶尔说一句话"的能力来说，这套仪式太重了；
 *   * 往下传 prop 会把 `toast` 穿过 Workspace → MemoryPanel、MessageList →
 *     MessageBubble 这些**纯粹是中间层**的组件，它们自己不弹任何东西。
 *
 * 副作用是全局可变状态 —— 但它的生命周期就是"弹一句话，1.8 秒后消失"，
 * 没有业务含义，也不会在组件之间同步任何需要一致性的东西。
 *
 * 纪律：只用它报告**已经发生的状态变化**（已记住、已删除、已导出、已反馈）。
 * 不要用它做持续性状态（"正在加载"该用骨架屏），也不要给"复制成功"弹 ——
 * 图标变对勾已经把话说完了。
 */

import { useEffect, useState } from 'react'

export type ToastTone = 'info' | 'ok' | 'error'

export interface ToastItem {
  id: number
  text: string
  tone: ToastTone
}

/** 自动消失时间。够读一行中文，又不至于挡着后面的操作。 */
const LIFETIME = 2600

/** 同时最多几条：再多就变成一堵墙，用户只会全部无视。 */
const MAX_VISIBLE = 3

let seq = 0
let items: ToastItem[] = []
const listeners = new Set<(next: ToastItem[]) => void>()

function emit(): void {
  for (const fn of listeners) fn(items)
}

function remove(id: number): void {
  items = items.filter((t) => t.id !== id)
  emit()
}

/** 弹一条。调用点通常在事件处理函数里，不返回值。 */
export function toast(text: string, tone: ToastTone = 'info'): void {
  const item: ToastItem = { id: ++seq, text, tone }
  items = [...items, item].slice(-MAX_VISIBLE)
  emit()
  window.setTimeout(() => remove(item.id), LIFETIME)
}

export function dismissToast(id: number): void {
  remove(id)
}

/** 订阅当前该显示的提示条。 */
export function useToasts(): ToastItem[] {
  const [list, setList] = useState<ToastItem[]>(items)
  useEffect(() => {
    listeners.add(setList)
    setList(items)
    return () => {
      listeners.delete(setList)
    }
  }, [])
  return list
}
