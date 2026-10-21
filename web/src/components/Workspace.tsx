import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useCopy } from '../hooks/useCopy'
import { useCrossfade } from '../transition'
import type { KnowledgeGraph } from '../api/client'
import type { ChatMessage, MemoryItem } from '../types'
import { cardToText, formatAmount, statusTone, type WorkspaceCard } from '../workspace'
import { KnowledgePanel } from './KnowledgePanel'
import { MemoryPanel } from './MemoryPanel'
import { StarsPanel } from './StarsPanel'
import {
  IconBox,
  IconCheck,
  IconCopy,
  IconDoc,
  IconGraph,
  IconMemory,
  IconStar,
  IconTruck,
} from './icons'

/**
 * 右侧工作区：四个标签页。
 *
 *   工作区 —— 这一轮对话"办的事"：订单、物流轨迹、政策依据
 *   记忆   —— 系统记住的关于我的事（可看、可加、可删）
 *   收藏   —— 我星标过的消息
 *   图谱   —— 知识库本身的结构：文档 → 章节 → 小节，可按词筛
 *
 * 四者放在同一个面板里而不是各开一个抽屉，是因为它们回答的是同一个问题：
 * "这次对话留下了什么"。分成四处之后，用户要先想"我要找的东西属于哪一类"
 * 才能去点那个按钮 —— 而记忆和收藏这两个概念的区别，用户第一眼是分不清的。
 *
 * 这里只放**用户该看的东西**。意图、节点路径、证据分数、工具名、耗时都是
 * 内部信息 —— 连 HTTP 侧的开发者入口也一并关闭了（见后端 api/app.py），
 * 界面这一层更没有让它们进来的理由。
 */

export type WorkspaceTab = 'cards' | 'memory' | 'stars' | 'graph'

const TABS: ReadonlyArray<{ key: WorkspaceTab; label: string; Icon: typeof IconBox }> = [
  { key: 'cards', label: '工作区', Icon: IconBox },
  { key: 'memory', label: '记忆', Icon: IconMemory },
  { key: 'stars', label: '收藏', Icon: IconStar },
  { key: 'graph', label: '图谱', Icon: IconGraph },
]

type CardKind = WorkspaceCard['kind']

const KIND_META: Record<CardKind, { label: string; Icon: typeof IconBox }> = {
  order: { label: '订单', Icon: IconBox },
  logistics: { label: '物流', Icon: IconTruck },
  source: { label: '依据', Icon: IconDoc },
  action: { label: '处理结果', Icon: IconCheck },
}

/** 卡片外壳：统一的头（类型标签 + 图标 + 复制）与配色。 */
function CardShell({
  kind,
  children,
  headExtra,
}: {
  kind: CardKind
  children: ReactNode
  headExtra?: ReactNode
}) {
  const { label, Icon } = KIND_META[kind]
  return (
    <article className={`card card--${kind}`}>
      <header className="card-head">
        <span className="card-kind">
          <Icon width={13} height={13} />
          {label}
        </span>
        {headExtra}
      </header>
      {children}
    </article>
  )
}

/** 每张卡片右上角的复制按钮 —— 拿到订单信息的下一步多半是粘到别处。 */
function CardCopy({ card }: { card: WorkspaceCard }) {
  const { copied, copy } = useCopy()
  return (
    <button
      type="button"
      className="icon-btn card-copy"
      title={copied ? '已复制' : '复制'}
      aria-label={copied ? '已复制' : '复制卡片内容'}
      onClick={() => void copy(cardToText(card))}
    >
      {copied ? <IconCheck /> : <IconCopy />}
    </button>
  )
}

function StatusPill({ label }: { label: string }) {
  return <span className={`pill pill--${statusTone(label)}`}>{label}</span>
}

function OrderView({ card }: { card: Extract<WorkspaceCard, { kind: 'order' }> }) {
  const total = formatAmount(card.totalAmount, card.currency)
  return (
    <CardShell
      kind="order"
      headExtra={
        <>
          <span className="card-id">{card.orderId}</span>
          <StatusPill label={card.statusLabel} />
          <CardCopy card={card} />
        </>
      }
    >
      {card.items.length > 0 && (
        <ul className="item-list">
          {card.items.map((it, i) => (
            <li key={`${it.name}-${i}`} className="item">
              <span className="item-name">{it.name}</span>
              <span className="item-qty">×{it.qty}</span>
              <span className="item-price">{formatAmount(it.price, card.currency) ?? '—'}</span>
            </li>
          ))}
        </ul>
      )}

      <dl className="kv">
        {total && (
          <>
            <dt>合计</dt>
            <dd className="kv-strong">{total}</dd>
          </>
        )}
        {card.address && (
          <>
            <dt>收货地址</dt>
            <dd>{card.address}</dd>
          </>
        )}
      </dl>
    </CardShell>
  )
}

function LogisticsView({ card }: { card: Extract<WorkspaceCard, { kind: 'logistics' }> }) {
  return (
    <CardShell
      kind="logistics"
      headExtra={
        <>
          <span className="card-id">{card.orderId}</span>
          <StatusPill label={card.statusLabel} />
          <CardCopy card={card} />
        </>
      }
    >
      {(card.carrier || card.trackingNo) && (
        <dl className="kv kv--tight">
          {card.carrier && (
            <>
              <dt>承运</dt>
              <dd>{card.carrier}</dd>
            </>
          )}
          {card.trackingNo && (
            <>
              <dt>运单号</dt>
              <dd className="kv-mono">{card.trackingNo}</dd>
            </>
          )}
        </dl>
      )}

      {card.steps.length > 0 && (
        <ol className="timeline">
          {card.steps.map((s, i) => (
            <li key={i} className="timeline-item">
              <span className="timeline-dot" aria-hidden="true" />
              <div className="timeline-body">
                <p className="timeline-text">{s.text}</p>
                {s.time && <time className="timeline-time">{s.time}</time>}
              </div>
            </li>
          ))}
        </ol>
      )}

      {card.note && <p className="card-note">{card.note}</p>}
    </CardShell>
  )
}

function SourceView({ card }: { card: Extract<WorkspaceCard, { kind: 'source' }> }) {
  return (
    <CardShell
      kind="source"
      headExtra={
        <>
          <span className="card-title">{card.title}</span>
          <CardCopy card={card} />
        </>
      }
    >
      <p className="card-snippet">{card.snippet}</p>
    </CardShell>
  )
}

function ActionView({ card }: { card: Extract<WorkspaceCard, { kind: 'action' }> }) {
  return (
    <CardShell kind="action" headExtra={<CardCopy card={card} />}>
      {card.orderId && <span className="card-id card-id--own">{card.orderId}</span>}
      <p className="card-snippet">{card.text}</p>
    </CardShell>
  )
}

function Card({ card, index }: { card: WorkspaceCard; index: number }) {
  const inner = (() => {
    switch (card.kind) {
      case 'order':
        return <OrderView card={card} />
      case 'logistics':
        return <LogisticsView card={card} />
      case 'source':
        return <SourceView card={card} />
      case 'action':
        return <ActionView card={card} />
    }
  })()
  // 卡片按 index 依次入场：一次推出三张时，错开的那几十毫秒让"一起来"
  // 和"有先后"同时成立 —— 顺序信息是白送的
  return (
    <div className="card-slot" style={{ '--i': index } as React.CSSProperties}>
      {inner}
    </div>
  )
}

/**
 * 卡片的指向性倾斜。
 *
 * 鼠标在卡面上哪个位置，卡片就往哪个方向转一点点，高光也跟着移过去。
 * 幅度压得很小（最多 3 度）—— 这是"卡片知道你在这儿"，不是"卡片在跳舞"。
 *
 * 用事件委托在 `.card-stack` 上算，而不是给每张卡片挂监听：
 * 一次推出七八张卡片就是七八个监听器，而它们做的事完全一样。
 * 代价是要手动找最近的 `.card-slot` —— 用 closest 一行就够。
 */
const TILT_ENABLED =
  typeof window !== 'undefined' && window.matchMedia?.('(hover: hover) and (pointer: fine)').matches

function tiltCards(e: React.PointerEvent<HTMLDivElement>) {
  if (!TILT_ENABLED) return
  const slot = (e.target as HTMLElement).closest<HTMLElement>('.card-slot')
  const card = slot?.querySelector<HTMLElement>('.card')
  if (!slot || !card) return
  const box = slot.getBoundingClientRect()
  const px = (e.clientX - box.left) / box.width
  const py = (e.clientY - box.top) / box.height
  card.style.setProperty('--tilt-y', `${((px - 0.5) * 5.5).toFixed(2)}deg`)
  card.style.setProperty('--tilt-x', `${((0.5 - py) * 4.5).toFixed(2)}deg`)
  card.style.setProperty('--sheen-x', `${(px * 100).toFixed(1)}%`)
}

function untiltCards(e: React.PointerEvent<HTMLDivElement>) {
  for (const card of e.currentTarget.querySelectorAll<HTMLElement>('.card')) {
    card.style.removeProperty('--tilt-x')
    card.style.removeProperty('--tilt-y')
  }
}

function CardsView({ cards }: { cards: WorkspaceCard[] }) {
  const summary = cards.reduce<Record<string, number>>((acc, c) => {
    acc[c.kind] = (acc[c.kind] ?? 0) + 1
    return acc
  }, {})

  if (cards.length === 0) {
    return (
      <div className="workspace-empty">
        <div className="grid-canvas" aria-hidden="true" />
        <div className="grid-scan" aria-hidden="true" />
        <p className="workspace-empty-text">
          对话里查到的订单、物流轨迹和政策依据
          <br />
          会陈列在这里
        </p>
      </div>
    )
  }

  return (
    <>
      <div className="workspace-tallies">
        {(Object.keys(KIND_META) as CardKind[])
          .filter((k) => summary[k])
          .map((k) => (
            <span key={k} className={`tally tally--${k}`}>
              {KIND_META[k].label} {summary[k]}
            </span>
          ))}
      </div>
      <div className="card-stack" onPointerMove={tiltCards} onPointerLeave={untiltCards}>
        {cards.map((c, i) => (
          <Card key={c.id} card={c} index={i} />
        ))}
      </div>
    </>
  )
}

interface Props {
  tab: WorkspaceTab
  onTab: (tab: WorkspaceTab) => void
  cards: WorkspaceCard[]
  onClearCards: () => void
  memories: MemoryItem[]
  memoryLoading: boolean
  memoryBusy: boolean
  memoryError: string
  onAddMemory: (text: string) => void
  onToggleMemoryPin: (item: MemoryItem) => void
  onDeleteMemory: (item: MemoryItem) => void
  starred: ChatMessage[]
  onJumpTo: (messageId: string) => void
  onUnstar: (message: ChatMessage) => void
  /** 知识图谱（四个标签里唯一"挂上面板才去拉"的内容） */
  graph: KnowledgeGraph
  graphLoading: boolean
  graphError: string
  onRefreshGraph: () => void
  /** 窄屏下工作区是抽屉，需要一个关闭入口 */
  onClose?: () => void
}

export function Workspace({
  tab,
  onTab,
  cards,
  onClearCards,
  memories,
  memoryLoading,
  memoryBusy,
  memoryError,
  onAddMemory,
  onToggleMemoryPin,
  onDeleteMemory,
  starred,
  onJumpTo,
  onUnstar,
  graph,
  graphLoading,
  graphError,
  onRefreshGraph,
  onClose,
}: Props) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  //: 轨道的 ref。滑块的位置要拿它当原点量（见下面 measureTab 的说明）。
  const tabsRef = useRef<HTMLDivElement>(null)
  //: 活动标签底块的几何（x / 宽）。测出来，而不是"每个标签自己画背景"：
  //: 切换时看到的是同一块底色挪过去，而不是两个元素一灭一亮 —— 连续性。
  const [indicator, setIndicator] = useState({ x: 0, w: 0 })
  //: 滑块要飞的**距离**（像素）。灵动式拉伸的依据：飞得越远，途中被拉得
  //: 越长。距离本身不带方向 —— scaleX 不区分左右，拉的是"长度"。
  const [flight, setFlight] = useState(0)
  //: 飞行方向（-1 / 0 / 1），决定拉伸的原点在哪一端。
  const [flightDir, setFlightDir] = useState(1)
  //: 上一次测到的滑块几何。拉伸量要拿"新位置 - 旧位置"来算，而旧位置
  //: 不能从 indicator 这个 state 里读 —— 见下面 useLayoutEffect 里的说明。
  const lastRect = useRef({ x: 0, w: 0 })

  //: 标签内容的交叉转场。`current` 在正常流里，`leaving` 绝对定位盖在上面化开。
  const { current, leaving } = useCrossfade(tab, 200)

  const panel = (which: WorkspaceTab) => {
    if (which === 'cards') return <CardsView cards={cards} />
    if (which === 'stars') return <StarsPanel items={starred} onJump={onJumpTo} onUnstar={onUnstar} />
    if (which === 'graph')
      return <KnowledgePanel graph={graph} loading={graphLoading} error={graphError} onRefresh={onRefreshGraph} />
    return (
      <MemoryPanel
        items={memories}
        loading={memoryLoading}
        busy={memoryBusy}
        error={memoryError}
        onAdd={onAddMemory}
        onTogglePin={onToggleMemoryPin}
        onDelete={onDeleteMemory}
      />
    )
  }

  //: 拉伸的回落：飞行结束（transition 走完）后把 scaleX 归 1。
  //: 没有这一步，滑块会一直以"拉长后的长度"停在目标标签上 ——
  //: 拉伸是途中的形变，不是终态。
  useEffect(() => {
    if (flight === 0) return
    const t = window.setTimeout(() => setFlight(0), 280)
    return () => window.clearTimeout(t)
  }, [flight])

  const activeIndex = TABS.findIndex((t) => t.key === tab)

  /**
   * 滑块的位置**用 rect 相减算，不用 offsetLeft**。
   *
   * 滑块是绝对定位的，它的静态起点是轨道的**内容盒**；而 offsetLeft 是相对
   * 轨道的**内边距边**。轨道一旦有 padding（分段控件必须有，不然滑块会顶着
   * 轨道的圆角），两者就差了正好一个 padding —— 实测偏 3~3.5px，
   * 看起来像"选错了标签"。
   *
   * 这里改成：先取轨道自己的 border-box 左边 + 左边框宽，得到内边距边的位置，
   * 再用活动标签的 rect 减掉它。这个算法对 padding / border / 圆角都不敏感。
   */
  const measureTab = useCallback((): { x: number; w: number } | null => {
    const el = tabRefs.current[activeIndex]
    const track = tabsRef.current
    if (!el || !track) return null
    const origin =
      track.getBoundingClientRect().left + parseFloat(getComputedStyle(track).borderLeftWidth || '0')
    return { x: el.getBoundingClientRect().left - origin, w: el.offsetWidth }
  }, [activeIndex])

  useLayoutEffect(() => {
    const el = tabRefs.current[activeIndex]
    if (!el) return
    const next = measureTab()
    if (next) {
      // 灵动式拉伸：先把滑块按飞行距离拉长（它会以 260ms 的弹性曲线飞），
      // 再落到新位置。拉伸量封顶在标签宽的 45% —— 超过这个比例，滑块
      // 在两个标签之间就变成一条横贯的色带，"形体"反而没了。
      //
      // 起点必须读 lastRect 这个 ref，不能读 indicator 这个 state：
      // ResizeObserver 触发 setIndicator 时 effect 会重跑（依赖里有
      // indicator.x），而那条路径上 activeIndex 根本没变 —— 位置差为 0，
      // 于是把刚设的 flight 又清成 0。实测这就是"拉伸永远看不到"的原因：
      // 真正的切换那一帧设了 flight，下一次提交立刻被同一次 effect 重跑抹掉。
      const prev = lastRect.current
      const dist = Math.abs(next.x - prev.x)
      // 只有真的换了位置才设拉伸。挂载时的首次测量（prev.w 为 0）和
      // 计数变宽（x 不动）都不该触发 —— 它们也把 flight 写一遍，
      // 就会把正在飞的拉伸清掉。
      if (prev.w > 0 && dist > 0) {
        setFlight(Math.min(dist, next.w * 0.45))
        // 拉伸方向跟着飞行方向：往右飞从左端甩出去，往左飞从右端 ——
        // 不换原点的话，往左飞会从左侧拉出去，视觉上是"提前到了半格"
        setFlightDir(Math.sign(next.x - prev.x))
      }
      lastRect.current = next
      setIndicator(next)
    }

    /* 用 ResizeObserver 盯着**所有**标签，而不是只盯活动那个。
       原先这里挂的是 window 的 resize 监听，effect 依赖里虽然写了三个计数，
       但函数体只重新注册了监听、没有立即重测 —— 于是计数一出现标签就变宽、
       滑块却停在旧宽度上：实测标签 99.5px 配滑块 79px，计数徽章有一半悬在
       滑块外面。凡是"元素尺寸变了但没触发我列出的那个事件"的情况，
       都会以同样的方式失败；换成观察元素本身就不存在这个死角。

       为什么要连非活动标签一起盯：标签是并排的，前面的标签多出一个计数徽章，
       后面的标签会整体右移，而**它自己的宽度没变** —— 只观察活动标签的话，
       ResizeObserver 不会触发，滑块留在旧位置。实测「工作区」从 78.5px 涨到
       103.5px 时，「图谱」右移了 25px，滑块却一动不动，看起来像选错了一格。 */
    const observer = new ResizeObserver(() => {
      const rect = measureTab()
      if (rect) {
        // 位置没变、只是变宽（计数多了一位）—— 记录新宽度，但不碰 flight。
        lastRect.current = rect
        setIndicator(rect)
      }
    })
    for (const t of tabRefs.current) {
      if (t) observer.observe(t)
    }
    return () => observer.disconnect()
  }, [measureTab, activeIndex, indicator.x])

  const counts: Record<WorkspaceTab, number> = {
    cards: cards.length,
    memory: memories.length,
    stars: starred.length,
    graph: graph.count,
  }

  /**
   * 灵动式通知：不处于活动标签的内容一旦更新，那个标签就被"点亮"。
   *
   * 在 render 里做而不是 useEffect：内容到达和用户点标签可能落在同一次
   * 提交里，effect 要到提交之后才跑，那时 pending 已被 goTo 清掉，
   * 通知就永远出不来。render 期间"根据前一次渲染调整 state"是 React
   * 官方认可的模式，前提是能收敛 —— seen 一旦追上 counts 就不再触发。
   *
   * 首次看见（面板挂载后的第一批数据，比如记忆/图谱的初次拉取）只记录
   * 不点亮：那是"本来就在那里的东西"，不是"刚刚发生的事"。
   *
   * 判据是"首次装载停下来的那一刻"，不是"第一次渲染"。记忆和图谱都是挂载
   * 之后才去请求的，首帧计数必然是 0，数据一到就变成 N —— 拿首帧当基线，
   * 每次打开页面都会冒出一个假的 "+N"，像是在说"你刚错过了什么"，而用户
   * 什么都没错过。所以装载期间不碰基线，停下来的那次才定；此后（比如手动
   * 刷新图谱）再涨，就是真的新到了。
   */
  const seen = useRef<Partial<Record<WorkspaceTab, number>>>({})
  const [pending, setPending] = useState<Partial<Record<WorkspaceTab, number>>>({})
  for (const { key } of TABS) {
    const n = counts[key]
    // 装载中不写基线：这一批还不算"看见了"，等它定下来
    if (key === 'memory' ? memoryLoading : key === 'graph' ? graphLoading : false) continue
    const prev = seen.current[key]
    if (prev === n) continue
    seen.current[key] = n
    if (prev !== undefined && key !== tab && n > prev) {
      // 只记增量：徽章显示"新到了几条"，而不是总数 —— 总数在标签自己的
      // 计数里已经有了，徽章再说一遍就是同一句话讲两遍
      setPending((p) => ({ ...p, [key]: (p[key] ?? 0) + (n - prev) }))
    }
  }
  const goTo = useCallback(
    (key: WorkspaceTab) => {
      onTab(key)
      setPending((p) => {
        if (!(key in p)) return p
        const next = { ...p }
        delete next[key]
        return next
      })
    },
    [onTab],
  )

  /**
   * 计数变化的"闪一下"。
   *
   * 页头那个总计数（Header 的 badge--bump）弹，标签自己的计数不弹 ——
   * 用户的眼睛在右栏，他看到的是"数字变了"但没有"它动了"，两者对不上。
   *
   * 两条纪律和上面的未读点 alike 亮完全一致：
   *
   *   - **装载期间不记录基线。** 记忆和图谱是挂载之后才去请求的，首帧计数
   *     是 0、数据一到变成 N。如果装载中就把 0 记成基线，数据落地那一下
   *     就会被当成"数字变了"，标签在用户第一次点开它的时候闪一下 ——
   *     像是在说"你刚错过了 70 条"，而他什么都没错过。跳过装载中的那些
   *     渲染，基线落在"装载停下来"的那一刻，此后（比如手动刷新图谱）
   *     再变才是真的变了。
   *   - **每次变化都换一个 key。** 原先靠一个布尔标记 + 320ms 定时器清掉
   *     class；两次变化挨得近时第二次只是把 class 再写一遍，动画不会重放，
   *     而且定时器的清理和"用户恰好在闪的那 320ms 里切了标签"会互相打架。
   *     改成自增序号后，变化一次换一次 DOM 节点，动画自然从头播；
   *     动画本身是有限时长的，播完就停，不需要任何清理。
   */
  const [bumpTick, setBumpTick] = useState<Partial<Record<WorkspaceTab, number>>>({})
  //: 正在闪的标签。序号只是"换一个 key 重新挂载"，闪完必须把类摘掉 ——
  //: 否则那个计数永久带着动画类，"现在有没有在闪"这件事就再也读不出来了。
  //: 用 animationend 而不是定时器：时长写在 CSS 里，JS 不该再抄一份。
  const [flashing, setFlashing] = useState<Partial<Record<WorkspaceTab, boolean>>>({})
  const prevCounts = useRef<Partial<Record<WorkspaceTab, number>>>(
    // 首帧就把**已经定下来**的那些计数记成基线：记忆和图谱是挂载后才去
    // 请求的，装载期间不记（见下面的渲染期写入）—— 但「工作区」的卡片数
    // 首帧就是确定的，它要是一条都没变，等到第一次真变化时才有 prev 可比。
    Object.fromEntries(
      TABS.filter(({ key }) => (key === 'memory' ? !memoryLoading : key === 'graph' ? !graphLoading : true)).map(
        ({ key }) => [key, counts[key]],
      ),
    ),
  )
  //: 上一帧各标签是否在装载（见 effect 里"装载刚停"的判断）
  const loadingRef = useRef<Partial<Record<WorkspaceTab, boolean>>>({
    memory: memoryLoading,
    graph: graphLoading,
  })
  useEffect(() => {
    const changed: WorkspaceTab[] = []
    //: 这一帧还在装载的 key —— 上一帧装载、这一帧停下来，落地的是第一批
    //: 数据，不该闪（见下面的判断）。effect 里读不到"上一帧的 loading"，
    //: 所以自己记一份。
    const wasLoading = loadingRef.current
    loadingRef.current = {
      memory: memoryLoading,
      graph: graphLoading,
      cards: false,
      stars: false,
    }
    for (const { key } of TABS) {
      const loading = key === 'memory' ? memoryLoading : key === 'graph' ? graphLoading : false
      // 装载中不写基线：这一批还不算"数据变了"（理由见上）
      if (loading) continue
      const n = counts[key]
      const prev = prevCounts.current[key]
      prevCounts.current[key] = n
      // 刚停下来的那一次：落地的是第一批数据（0 → 70），它是这个标签的
      // 初始状态，不是"新到了 70 条"。prev === undefined 同理。
      if (prev !== undefined && prev !== n && !wasLoading[key]) changed.push(key)
    }
    if (changed.length === 0) return
    setBumpTick((t) => {
      const next = { ...t }
      for (const k of changed) next[k] = (next[k] ?? 0) + 1
      return next
    })
    setFlashing((f) => {
      const next = { ...f }
      for (const k of changed) next[k] = true
      return next
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [counts.cards, counts.memory, counts.stars, counts.graph, memoryLoading, graphLoading])

  const clearFlash = useCallback((key: WorkspaceTab) => {
    setFlashing((f) => {
      if (!f[key]) return f
      const next = { ...f }
      delete next[key]
      return next
    })
  }, [])

  /** 标签之间用左右方向键切换（WAI-ARIA 的 tabs 约定）。 */
  const onKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
    e.preventDefault()
    const next = (index + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length
    goTo(TABS[next].key)
    tabRefs.current[next]?.focus()
  }

  return (
    <section className="workspace" aria-label="工作区">
      <header className="workspace-head">
        <div className="tabs" role="tablist" aria-label="右侧面板" ref={tabsRef}>
          {indicator.w > 0 && (
            <span
              className="tab-indicator"
              aria-hidden="true"
              style={{
                width: indicator.w,
                transform: `translateX(${indicator.x}px) scaleX(${1 + flight / indicator.w})`,
                //: 拉伸的原点跟着飞行方向：往右飞从左端甩出去，往左飞从右端
                transformOrigin: flightDir >= 0 ? 'left center' : 'right center',
              }}
            />
          )}
          {TABS.map(({ key, label, Icon }, i) => {
            const notify = pending[key]
            return (
              <button
                key={key}
                ref={(el) => {
                  tabRefs.current[i] = el
                }}
                type="button"
                role="tab"
                id={`tab-${key}`}
                aria-selected={tab === key}
                aria-controls={`tabpanel-${key}`}
                tabIndex={tab === key ? 0 : -1}
                className={`tab${tab === key ? ' tab--active' : ''}${notify ? ' tab--notify' : ''}`}
                onClick={() => goTo(key)}
                onKeyDown={(e) => onKeyDown(e, i)}
                // 窄屏下文字会被藏起来（见 styles.css 的 1024px 媒体查询），
                // 这两个属性是那时唯一还能说出"这是哪个标签"的地方
                title={label}
                aria-label={counts[key] > 0 ? `${label}，${counts[key]} 条` : label}
              >
                <Icon width={14} height={14} />
                <span className="tab-label">{label}</span>
                {counts[key] > 0 && (
                  <span
                    // 序号变了就换一个节点：重挂载让 CSS 动画从头播，
                    // 不依赖"先清 class 再等一帧再加回来"这种时序把戏
                    key={`${key}-${bumpTick[key] ?? 0}`}
                    className={`tab-count${flashing[key] ? ' tab-count--bump' : ''}`}
                    onAnimationEnd={() => clearFlash(key)}
                  >
                    {counts[key]}
                  </span>
                )}
                {notify ? (
                  <span className="tab-bubble" aria-hidden="true">
                    {notify}
                  </span>
                ) : null}
              </button>
            )
          })}
        </div>

        <div className="workspace-actions">
          {tab === 'cards' && cards.length > 0 && (
            <button type="button" className="ghost-btn" onClick={onClearCards}>
              清空
            </button>
          )}
          {onClose && (
            <button type="button" className="ghost-btn" onClick={onClose} aria-label="收起工作区">
              收起
            </button>
          )}
        </div>
      </header>

      <div
        className="workspace-body"
        role="tabpanel"
        id={`tabpanel-${tab}`}
        aria-labelledby={`tab-${tab}`}
      >
        {/* 切标签不是一个"换页"动作，是同一块地方换了内容 ——
            所以旧的一块要先化开再让新的浮起来，两块在场上重叠 200ms。
            没有这一重叠，视觉上就只是"啪"地换了一下，人会下意识地
            重新找一遍"我刚才看的那条去哪了"。 */}
        <div className="swap-root">
          {leaving !== null && (
            <div className="swap-out" aria-hidden="true">
              {panel(leaving)}
            </div>
          )}
          <div className="swap-in" key={current}>
            {panel(current)}
          </div>
        </div>
      </div>
    </section>
  )
}
