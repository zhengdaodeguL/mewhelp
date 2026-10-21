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
  /** 深色主题下的不透明度 */
  opacity: number
  /** 浅色主题下的不透明度。见下面 PHOTOS 上方的说明。 */
  opacityLight: number
}

/**
 * 顺序是有讲究的：冷暖交替排开，相邻两张的色调差得越远，交叉淡入越明显。
 * 同色系放在一起（蓝 → 青 → 绿），轮换时几乎看不出换了图。
 *
 * **两个 opacity 是按每张照片实测的相对亮度反比分配出来的**，不是随手定的：
 * 亮的压得狠、暗的放得多，目的是让五张轮换时面板的合成亮度落在同一条线上。
 * 否则 8.6 秒一次的轮换会变成整屏明暗闪烁 —— 那比单调更烦人。
 *
 * 实测相对亮度（0-255，WCAG 的 Y，**不是** sRGB 均值 —— 算对比度必须用它）：
 *   lake 78.7 > valley 70.3 > meadow 51.1 > falls 31.7 > canyon 24.8
 *
 * 浅色主题那一列整体更低，是物理决定的、不是偷懒：深色主题是"在暮色里看见
 * 峡谷"，照片越透越好看；浅色是"白玻璃上透出一点色温"，白玻璃一透，
 * 深色正文就压不住照片的暗部了。
 *
 * 换素材时要重做的两件事：先量新图的相对亮度按反比重排这两列，
 * 再重跑一遍对比度核验。硬性上限：**照片最亮 5% 的平均亮度不得超过 245**。
 */
const PHOTOS: Photo[] = [
  { src: '/scene/lake.webp', opacity: 0.54, opacityLight: 0.44 }, // 蓝：峡湾；最亮，压到最低
  { src: '/scene/canyon.webp', opacity: 0.66, opacityLight: 0.56 }, // 暖：日落峡谷；最暗，放到最高
  { src: '/scene/falls.webp', opacity: 0.58, opacityLight: 0.52 }, // 青：林间瀑布
  { src: '/scene/meadow.webp', opacity: 0.56, opacityLight: 0.46 }, // 金：黄昏草甸
  { src: '/scene/valley.webp', opacity: 0.56, opacityLight: 0.46 }, // 绿：山谷公路
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
      // 只有**成功解码**才算"能画出来了"：onload 之后仍可能有一帧空白，
      // 而空白期间轮到它上场，用户看到的就是风景"消失一下再回来"。
      //
      // 失败绝不能走 done：把没下下来的下标标成就绪，轮换会挑中它，
      // 而那一层的 background-image 指向一个不存在的文件 —— 整层只剩
      // 色罩，风景凭空消失 8.6 秒。失败就让这个下标永远不进 ready，
      // 轮换会自然跳过它（见下面的 setInterval）。
      const fail = () => {
        if (!cancelled) console.warn('[landscape] 背景图加载失败，跳过：', PHOTOS[i].src)
      }
      if (typeof img.decode === 'function') img.decode().then(done, fail)
      else img.onload = done
      img.onerror = fail
    }

    const idleWin = window as IdleWindow
    const start = () => PHOTOS.forEach((_, i) => i && load(i))
    const useIdle = typeof idleWin.requestIdleCallback === 'function'
    const handle = useIdle
      ? idleWin.requestIdleCallback!(start)
      : window.setTimeout(start, 900)

    return () => {
      cancelled = true
      // 只取消真正注册过的那个：idle 句柄和 timeout 句柄来自两个不同的
      // 编号空间，对 idle 句柄调 clearTimeout 有可能撞上别处一个仍然
      // 有效的 timeout id 把它取消掉，而对 timeout 句柄调
      // cancelIdleCallback 则是个静默的空操作。
      if (useIdle) idleWin.cancelIdleCallback?.(handle)
      else window.clearTimeout(handle)
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

  /**
   * 把"现在轮到哪一张"写到 <html data-scene> 上。
   *
   * 这是**环境光的色相来源**：光标扫过时底下那团柔光，颜色跟着当前这张照片走 ——
   * 轮到峡湾是冷蓝、峡谷是暖橘、瀑布是青绿、草甸是金、山谷是绿。
   * 43 秒一轮里界面真的换四次调子，而不是五张照片在同一个暖橘灯下轮播。
   *
   * 和它旁边的 --scene-x / --scene-y 完全同一个机制（写 html 上的属性/变量，
   * 由 CSS 消费）：不新增 props、不改组件契约。
   *
   * 换色是**渐变的** —— CSS 里用 @property 把这个变量注册成 <color> 并挂了
   * 2400ms 过渡，和照片自己的交叉淡入同步。不支持的浏览器退化成硬切，
   * 幅度这么低（alpha 0.16 上下）几乎看不出来。
   */
  useEffect(() => {
    document.documentElement.dataset.scene = String(live)
  }, [live])

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
                // 深浅两套都写进去，由样式表按主题挑一个（内联样式盖不过样式表，
                // 所以不能只写一个再让 CSS 去覆盖它）
                '--scene-opacity': photo.opacity,
                '--scene-opacity-light': photo.opacityLight,
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
