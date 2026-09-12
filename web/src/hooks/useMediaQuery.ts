import { useEffect, useState } from 'react'

/**
 * 订阅一个媒体查询。
 *
 * 用途很具体：宽屏下工作区是常驻右栏，窄屏下它必须变成抽屉 —— 这两种形态
 * 需要的 DOM 不一样（抽屉要有关闭按钮，常驻栏不要）。用 CSS 藏按钮能骗过
 * 眼睛，但会让屏幕阅读器念出一个用不上的"收起"。所以按断点真实地渲染。
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia(query).matches,
  )

  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    onChange()
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])

  return matches
}
