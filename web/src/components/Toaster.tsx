import { IconAlert, IconCheck, IconSpark } from './icons'
import { dismissToast, useToasts, type ToastTone } from '../toast'

/**
 * 提示条的渲染容器。挂在 App 最外层，全局只有一个。
 *
 * 带 `aria-live="polite"`：提示条是异步冒出来的，屏幕阅读器需要被通知。
 * 用 polite 而不是 assertive —— 它是"顺带说一句"，不该打断用户正在听的内容。
 */

const TONE_ICON: Record<ToastTone, typeof IconCheck> = {
  info: IconSpark,
  ok: IconCheck,
  error: IconAlert,
}

export function Toaster() {
  const items = useToasts()
  if (items.length === 0) return null

  return (
    <div className="toaster" role="status" aria-live="polite">
      {items.map((t) => {
        const Icon = TONE_ICON[t.tone]
        return (
          <button
            key={t.id}
            type="button"
            className={`toast toast--${t.tone}`}
            title="点击关闭"
            onClick={() => dismissToast(t.id)}
          >
            <Icon width={14} height={14} />
            {t.text}
          </button>
        )
      })}
    </div>
  )
}
