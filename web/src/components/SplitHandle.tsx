import { useCallback, useRef } from 'react'
import { MAX_RATIO, MIN_RATIO } from '../hooks/useSplit'

/**
 * 左右分栏之间的拖拽条。
 *
 * 三件容易被忽略、但缺了就会"能用但难用"的事：
 *
 * 1. **用 Pointer Events 而不是 mouse*。** 鼠标事件在触屏和手写笔上不触发；
 *    更要命的是没有指针捕获：鼠标移出这条 6px 宽的细缝就收不到 move 了，
 *    拖拽会在最需要它的时候断掉。所以监听挂在 window 上，而不是这个元素上。
 *
 * 2. **它是可聚焦的。** `role="separator"` + tabIndex + 方向键 ——
 *    键盘用户和"手抖拖不准"的人都靠它。只做鼠标拖拽等于把分栏宽度这个功能
 *    对一部分人关掉了。
 *
 * 3. **双击复位。** 拖到某个很别扭的比例之后，用户最想做的是"回到默认"，
 *    而不是反向猜着拖回去。
 *
 * 拖拽期间给 <body> 挂 resizing：光标的 col-resize 是次要的，
 * 主要是禁掉文本选中 —— 否则拖一次会选中满屏文字。
 */

interface Props {
  ratio: number
  onRatio: (value: number) => void
  /** 方向键的步进（比例值） */
  onNudge: (delta: number, big?: boolean) => void
  onReset: () => void
}

export function SplitHandle({ ratio, onRatio, onNudge, onReset }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  const startDrag = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const self = ref.current
      const panes = self?.parentElement
      if (!self || !panes) return
      // 阻止默认行为，否则拖拽会先被当成"选中文字"处理，拖起来一顿一顿的
      e.preventDefault()

      const rect = panes.getBoundingClientRect()
      const handleWidth = self.offsetWidth
      const usable = rect.width - handleWidth
      if (usable <= 0) return

      const onMove = (ev: PointerEvent) => {
        onRatio((ev.clientX - rect.left - handleWidth / 2) / usable)
      }
      const onUp = () => {
        window.removeEventListener('pointermove', onMove)
        window.removeEventListener('pointerup', onUp)
        window.removeEventListener('pointercancel', onUp)
        document.body.classList.remove('resizing')
      }

      document.body.classList.add('resizing')
      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
      window.addEventListener('pointercancel', onUp)
    },
    [onRatio],
  )

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      onNudge(-1, e.shiftKey)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      onNudge(1, e.shiftKey)
    } else if (e.key === 'Home') {
      e.preventDefault()
      onRatio(MIN_RATIO)
    } else if (e.key === 'End') {
      e.preventDefault()
      onRatio(MAX_RATIO)
    }
  }

  return (
    <div
      ref={ref}
      className="split"
      role="separator"
      aria-orientation="vertical"
      aria-label="调整对话区与工作区的宽度"
      aria-valuenow={Math.round(ratio * 100)}
      aria-valuemin={Math.round(MIN_RATIO * 100)}
      aria-valuemax={Math.round(MAX_RATIO * 100)}
      aria-valuetext={`对话区 ${Math.round(ratio * 100)}%`}
      tabIndex={0}
      title="拖动调整宽度（双击还原，方向键微调）"
      onPointerDown={startDrag}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
    >
      <span className="split-grip" aria-hidden="true" />
    </div>
  )
}
