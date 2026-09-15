/**
 * 转场基元：让"切换"看起来是**一件事退下去、另一件事浮上来**，
 * 而不是两块内容原地硬替。
 *
 * 两条独立的通道，各管各的场景：
 *
 *   `useCrossfade`  —— 交替换场（切标签、切会话）。新旧两块在场上重叠约 200ms，
 *                      旧的一块化开、新的一块从失焦里浮起来。用 React 状态驱动，
 *                      不依赖任何浏览器 API，所有浏览器行为一致。
 *   `withThemeWipe` —— 圆形揭示（切主题）。用 View Transitions API 把整页快照
 *                      做一次从指针位置扩散的圆形揭示。这是渐进增强：
 *                      不支持的浏览器直接切，体验不比原来差。
 *
 * 为什么不用 View Transitions 统一做交替换场：那需要把整页拍快照，
 * 而这个界面底下有一层一直在推镜的照片、面板上还有 backdrop-filter ——
 * 为了切一个标签去合成整屏，代价和收益不成比例。
 */

import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'

/** 用户是否要求减少动效。每处动效都要问一次，不能只在 CSS 里关。 */
export function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
  )
}

export interface Crossfade<T> {
  /** 现在该显示的那一个 */
  current: T
  /** 正在化开、马上就没了的那一个（没有就是 null） */
  leaving: T | null
}

/**
 * 交替换场。`value` 一变，旧值会作为 `leaving` 多留 `ms` 毫秒。
 *
 * 调用方把两个都渲染出来：`leaving` 放在绝对定位的 `.swap-out` 里，
 * `current` 放在正常流里的 `.swap-in` 里（key 绑到 current，才会重新触发入场动画）。
 */
export function useCrossfade<T>(value: T, ms = 200): Crossfade<T> {
  const [state, setState] = useState<Crossfade<T>>({ current: value, leaving: null })
  const timer = useRef<number | undefined>(undefined)
  //: 上一次的值。放在 ref 里而不是 state —— 它只是用来判断"变了没有"，
  //: 进 state 会多渲染一轮，而这一轮恰好卡在转场开始的那一帧上。
  const last = useRef(value)

  useEffect(() => {
    if (Object.is(last.current, value)) return
    const previous = last.current
    last.current = value

    if (prefersReducedMotion()) {
      setState({ current: value, leaving: null })
      return
    }

    setState({ current: value, leaving: previous })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(
      () => setState({ current: value, leaving: null }),
      ms,
    )
  }, [value, ms])

  useEffect(() => () => window.clearTimeout(timer.current), [])

  return state
}

/**
 * 退场延迟卸载。
 *
 * React 里没有"卸载动画"这回事：条件渲染一变成 false，元素当帧就从 DOM 里没了，
 * CSS 里写多少 exit 关键帧都没有机会跑。想让它退场，只能自己把"从 DOM 里拿掉"
 * 这件事推迟到动画之后 —— 这个 hook 就是那个推迟器。
 *
 * 返回的 `closing` 在退场这段时间里为 true，调用方把它变成 class，
 * 让 CSS 播出退场动画（并且必须把 `pointer-events` 关掉：
 * 命令面板的遮罩是 `inset: 0` 的一整层，留着它接点击等于界面卡死 200ms）。
 *
 * 退场途中又打开（`open` 回到 true）会把计时器取消、`closing` 撤回，
 * 元素自始至终没有被卸载过 —— 输入框里的字和焦点都还在。
 */
export function useDelayedUnmount(
  open: boolean,
  ms = 180,
): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open)
  const [closing, setClosing] = useState(false)

  useEffect(() => {
    if (open) {
      setClosing(false)
      setMounted(true)
      return
    }
    if (!mounted) return
    // 用户要求减少动效时不留这一段时间 —— 那 180ms 里元素还在，只是看不见
    if (prefersReducedMotion()) {
      setMounted(false)
      return
    }
    setClosing(true)
    const timer = window.setTimeout(() => {
      setMounted(false)
      setClosing(false)
    }, ms)
    return () => window.clearTimeout(timer)
  }, [open, mounted, ms])

  return { mounted, closing }
}

/** 最近一次指针位置（视口坐标）。圆形揭示要拿它当圆心。 */
let pointer = { x: 0, y: 0 }

export function lastPointer(): { x: number; y: number } {
  return pointer
}

export function trackPointer(): () => void {
  const onMove = (e: PointerEvent) => {
    pointer = { x: e.clientX, y: e.clientY }
  }
  window.addEventListener('pointermove', onMove, { passive: true })
  return () => window.removeEventListener('pointermove', onMove)
}

/**
 * 圆形揭示：从指针位置扩散一圈，把新主题"擦"出来。
 *
 * 用 `flushSync` 包住状态更新是必须的：View Transitions 要在回调**返回时**
 * 立即拍下新快照，而 React 默认是异步批量更新的 —— 不强制同步刷新，
 * 拍到的还是旧界面，整个转场就变成了一次无意义的原地闪。
 *
 * 类型上 `document.startViewTransition` 是必有的（进了 lib.dom），
 * 但运行时 Safari/Firefox 的旧版本并没有 —— 所以这里显式把类型放宽再判空，
 * 不能指望 TS 帮忙。
 */
type WipeHost = {
  startViewTransition?: (cb: () => void) => { ready: Promise<void> }
}

export function withThemeWipe(apply: () => void): void {
  const host = document as unknown as WipeHost
  if (typeof host.startViewTransition !== 'function' || prefersReducedMotion()) {
    apply()
    return
  }

  const { x, y } = lastPointer()
  const transition = host.startViewTransition(() => flushSync(apply))

  transition.ready
    .then(() => {
      // 半径取到"离圆心最远的那个角"，保证圆扩散到边缘时正好铺满
      const radius = Math.hypot(
        Math.max(x, window.innerWidth - x),
        Math.max(y, window.innerHeight - y),
      )
      document.documentElement.animate(
        {
          clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`],
        },
        {
          duration: 620,
          easing: 'cubic-bezier(0.22, 0.61, 0.36, 1)',
          pseudoElement: '::view-transition-new(root)',
        },
      )
    })
    .catch(() => {
      /* 快照被浏览器跳过时什么都不用做，新主题已经生效了 */
    })
}
