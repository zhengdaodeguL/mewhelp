import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * 复制到剪贴板，并给一小段"已复制"反馈。
 *
 * 除了 `navigator.clipboard`，还留了一条 `execCommand` 的退路：剪贴板 API
 * 只在安全上下文（https / localhost）可用，而这类服务经常被部署在内网 http 上。
 */
export function useCopy() {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current)
    },
    [],
  )

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const area = document.createElement('textarea')
      area.value = text
      area.style.position = 'fixed'
      area.style.opacity = '0'
      document.body.appendChild(area)
      area.select()
      try {
        document.execCommand('copy')
      } catch {
        /* 复制不了也不是致命错，静默即可 */
      }
      document.body.removeChild(area)
    }
    setCopied(true)
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setCopied(false), 1600)
  }, [])

  return { copied, copy }
}
