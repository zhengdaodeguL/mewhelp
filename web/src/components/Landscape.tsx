/**
 * 底层风景：五张自然照片轮流上场，每张都在极慢地推镜。
 *
 * 三条约束决定了这里的写法：
 *
 * 1. **首屏只加载一张。** 五张加起来 1.5MB，全塞进首帧会把白屏拖长一倍多。
 *    其余四张等首帧之后再预加载，且**解码完才允许轮到它** ——
 *    否则轮到一张还没下完的图，画面上会只剩色罩，风景"消失"一下再回来。
 *
 * 2. **每层的推镜方向不同**（见 styles.css 里的 nth-child）。五张都朝一个方向动，
 *    轮换就会被一眼看穿成一段循环动画，而不是一片在呼吸的景色。
 *
 * 3. **跟着指针轻微视差**。幅度只有 12px：它唯一的作用是让风景"在玻璃后面"，
 *    而不是"贴在屏幕上"。幅度大了会晕。
 *
 * 照片来自 Lorem Picsum（图源 Unsplash），Unsplash License 允许自由使用。
 * 文件名对应 id 见下面 PHOTOS 表，换图只要换文件 + 这张表。
 */

import { useEffect, useRef, useState } from 'react'
import { prefersReducedMotion } from '../transition'

interface Photo {
  src: string
  /** 每张照片的透明度不一样：亮的那几张要压得更狠，否则文字对比度会跳 */
  opacity: number
}

/**
 * 顺序是有讲究的：冷暖交替排开，相邻两张的色调差得越远，交叉淡入越明显。
 * 同色系放在一起（蓝 → 青 → 绿），轮换时几乎看不出换了图。
 */
const PHOTOS: Photo[] = [
  { src: '/scene/lake.webp', opacity: 0.5 }, // 蓝：湖 + 雪山
  { src: '/scene/canyon.webp', opacity: 0.42 }, // 暖：日落峡谷
  { src: '/scene/falls.webp', opacity: 0.45 }, // 青：林间瀑布
  { src: '/scene/meadow.webp', opacity: 0.45 }, // 金：黄昏草甸
  { src: '/scene/valley.webp', opacity: 0.52 }, // 绿：山谷公路
]

/** 每张停留多久。配上 2.4s 的交叉淡入，一轮约 45 秒 —— 慢到不会让人等。 */
const DWELL_MS = 8600

//: Safari 至今没有 requestIdleCallback，所以两套都要留。
type IdleWindow = Window & {
  requestIdleCallback?: (cb: () => void) => number
  cancelIdleCallback?: (handle: number) => void
}

export function Landscape() {
  const [live, setLive] = useState(0)
  //: 已经解码完、可以上场的下标。第一张默认就绪，因为**只有它不走下面的预加载**：
  //: 预加载那段的循环写成 `i && load(i)`，下标 0 会被跳过 —— 它的请求是本次渲染里
  //: 那行内联 `backgroundImage` 发出的。不在这里直接标成就绪，它永远等不到
  //: "加载完成"那一步，也就永远不会显示。
  //:
  //: 顺带记下真实时序，免得下次有人凭直觉写错（实测 1440×900 本地）：
  //: HTML/CSS/JS 在 14ms 起取、118ms 下完；**lake.webp 到 149ms 才开始请求** ——
  //: 它是 React 提交那行内联样式时被浏览器发现的，并不是"跟着 HTML 一起"
  //: （只有写在 HTML 里的东西才是）。另外四张在 236ms，走的是 idle 回调。
  //: 刻意为它加 <link rel=preload> 被否掉了：照片是氛围、界面才是内容，
  //: 让 300KB 的图去和渲染阻塞的 JS 抢带宽，换来的是首屏更晚出来。
  const [ready, setReady] = useState<ReadonlySet<number>>(() => new Set([0]))
  const readyRef = useRef(ready)

  useEffect(() => {
    readyRef.current = ready
  }, [ready])

  // 首帧之后再预加载其余四张。用 requestIdleCallback 是为了不和首屏抢带宽。
  useEffect(() => {
    let cancelled = false
    const load = (i: number) => {
      const img = new Image()
      img.src = PHOTOS[i].src
      const done = () => {
        if (!cancelled) setReady((prev) => new Set(prev).add(i))
      }
      // decode() 成功才是真的"能画出来了"：onload 之后仍可能有一帧空白，
      // 而空白期间轮到它上场，用户看到的就是风景"消失一下再回来"。
      if (typeof img.decode === 'function') img.decode().then(done, done)
      else img.onload = done
    }

    const idleWin = window as IdleWindow
    const start = () => PHOTOS.forEach((_, i) => i && load(i))
    const handle = idleWin.requestIdleCallback
      ? idleWin.requestIdleCallback(start)
      : window.setTimeout(start, 900)

    return () => {
      cancelled = true
      idleWin.cancelIdleCallback?.(handle)
      window.clearTimeout(handle)
    }
  }, [])

  // 轮换。只挑已经就绪的下标 —— 跳过没下完的那张，而不是停在那里等它。
  useEffect(() => {
    if (prefersReducedMotion()) return
    const timer = window.setInterval(() => {
      setLive((current) => {
        const readyNow = readyRef.current
        for (let step = 1; step <= PHOTOS.length; step++) {
          const next = (current + step) % PHOTOS.length
          if (readyNow.has(next)) return next
        }
        return current
      })
    }, DWELL_MS)
    return () => window.clearInterval(timer)
  }, [])

  // 指针视差：整层反向位移十几个像素。写在 CSS 变量上，由 .scene 消费。
  useEffect(() => {
    if (prefersReducedMotion()) return
    let raf = 0
    let tx = 0
    let ty = 0
    const apply = () => {
      raf = 0
      document.documentElement.style.setProperty('--scene-x', `${tx}px`)
      document.documentElement.style.setProperty('--scene-y', `${ty}px`)
    }
    const onMove = (e: PointerEvent) => {
      // 归一化到 -1~1 再乘幅度：屏幕上不管多大，位移都是同一个"手感"
      tx = (e.clientX / window.innerWidth - 0.5) * -14
      ty = (e.clientY / window.innerHeight - 0.5) * -14
      if (!raf) raf = window.requestAnimationFrame(apply)
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => {
      window.removeEventListener('pointermove', onMove)
      if (raf) window.cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <div className="scene" aria-hidden="true">
      <div className="scene-track">
        {PHOTOS.map((photo, i) => (
          <div
            key={photo.src}
            className={`scene-photo${i === live ? ' is-live' : ''}`}
            style={
              {
                // 没就绪的层不设图 —— 设了它会自己去请求，等于绕过上面的加载时机
                backgroundImage: ready.has(i) ? `url(${photo.src})` : undefined,
                '--scene-opacity': photo.opacity,
              } as React.CSSProperties
            }
          />
        ))}
      </div>
      <div className="scene-veil" />
      <div className="scene-vignette" />
      <div className="scene-grain" />
    </div>
  )
}
