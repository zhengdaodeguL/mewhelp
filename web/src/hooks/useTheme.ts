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

/**
 * 浏览器界面那条颜色（`<meta name="theme-color">`）。
 *
 * 值和 styles.css 里 `--bg` 的两个取值一致 —— 界面的底是风景照片，
 * 顶部这条对不上时，看起来就像页面被切掉一条。
 * index.html 里的内联脚本也各写了一遍（它跑在首次绘制之前，来不及 import）；
 * 三处任何一处改了，另外两处要跟着改。
 */
const THEME_COLOR: Record<ResolvedTheme, string> = {
  light: '#e9f1f3',
  dark: '#17272f',
}

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
      // 浏览器界面那条颜色也要跟着走，否则用户在深色下切到浅色，
      // 页面亮了、浏览器那条还是深色
      document
        .querySelector('meta[name="theme-color"]')
        ?.setAttribute('content', THEME_COLOR[next])
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
