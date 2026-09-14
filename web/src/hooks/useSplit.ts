/**
 * 左右分栏的比例。
 *
 * 用**比例**而不是像素：窗口变宽时两栏一起变宽，是桌面应用该有的行为。
 * 存像素的话，用户在大屏上拉好的分栏，换到笔记本上会有一栏被挤成一条缝。
 *
 * 比例只存一次、只在一处算（`clamp`），拖拽、键盘、双击复位都走同一个入口 ——
 * 三个入口各写一遍边界判断，是"键盘能调到 0.05 而拖拽不能"这类不一致的来源。
 */

import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'mewhelp.split'

/** 任一栏都不允许窄到看不清内容。 */
export const MIN_RATIO = 0.2
export const MAX_RATIO = 0.8

/** 初始 46% 给对话、54% 给工作区 —— 工作区的卡片比气泡宽，多一点才不挤。 */
export const DEFAULT_RATIO = 0.46

/** 键盘微调步长；按住 Shift 走大步。 */
const NUDGE = 0.02
const NUDGE_BIG = 0.08

function clamp(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_RATIO
  return Math.min(MAX_RATIO, Math.max(MIN_RATIO, value))
}

function read(): number {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw) return clamp(Number(raw))
  } catch {
    /* 隐私模式：本次会话用默认值 */
  }
  return DEFAULT_RATIO
}

export function useSplit() {
  const [ratio, setRatioState] = useState(read)

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(ratio))
    } catch {
      /* 存不下就只在本次会话生效 */
    }
  }, [ratio])

  const setRatio = useCallback((value: number) => setRatioState(clamp(value)), [])

  const nudge = useCallback((delta: number, big = false) => {
    const step = big ? NUDGE_BIG : NUDGE
    setRatioState((prev) => clamp(prev + delta * step))
  }, [])

  const reset = useCallback(() => setRatioState(DEFAULT_RATIO), [])

  return { ratio, setRatio, nudge, reset }
}
