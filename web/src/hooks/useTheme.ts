import { useCallback, useEffect, useState } from 'react'

/**
 * 主题：跟随系统 / 浅色 / 深色，选择记在 localStorage。
 *
 * 关键点是**别让 JS 决定首屏颜色**：样式表按 `html[data-theme]` 出图，
 * 而 `data-theme` 由 index.html 里一段内联脚本在首次绘制前写好 ——
 * 否则默认主题是浅色的人，每次进页面都会先被闪一下白。
 */

export type ThemeChoice = 'system' | 'light' | 'dark'
export type ResolvedTheme = 'light' | 'dark'

const STORAGE_KEY = 'mewhelp.theme'

/** 三态循环，用于点一下就切换的按钮。 */
const NEXT: Record<ThemeChoice, ThemeChoice> = {
  system: 'light',
  light: 'dark',
  dark: 'system',
}

export function readThemeChoice(): ThemeChoice {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw === 'light' || raw === 'dark' || raw === 'system') return raw
  } catch {
    /* 隐私模式下 localStorage 不可用 —— 当作跟随系统 */
  }
  return 'system'
}

export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  if (choice !== 'system') return choice
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readThemeChoice)
  const [theme, setTheme] = useState<ResolvedTheme>(() => resolveTheme(readThemeChoice()))

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)')
    const sync = () => {
      const next = resolveTheme(choice)
      document.documentElement.dataset.theme = next
      setTheme(next)
    }
    sync()
    try {
      window.localStorage.setItem(STORAGE_KEY, choice)
    } catch {
      /* 存不下就只在本次会话生效 */
    }

    // 只有"跟随系统"才需要监听系统切换
    if (choice !== 'system') return
    mql.addEventListener('change', sync)
    return () => mql.removeEventListener('change', sync)
  }, [choice])

  const cycle = useCallback(() => setChoice((c) => NEXT[c]), [])

  return { choice, theme, setChoice, cycle }
}
