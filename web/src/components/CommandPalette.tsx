import { useEffect, useMemo, useRef, useState } from 'react'
import {
  IconChat,
  IconClose,
  IconCommand,
  IconDownload,
  IconHistory,
  IconMemory,
  IconPanel,
  IconPlus,
  IconSearch,
  IconStar,
  IconSun,
} from './icons'

/**
 * 命令面板（Ctrl/⌘ + K）。
 *
 * 界面上的功能长到十几个之后，"记住它在哪个角落"本身就成了一种负担。
 * 命令面板把"找按钮"换成"打两个字" —— 而且它**不需要重新组织界面**：
 * 功能加一个条目就行，不用再往那一排图标里塞第 7 个按钮。
 *
 * 三个实现上的取舍：
 *
 * 1. **动作清单由 App 组装**，这个组件只负责"筛 + 选 + 键盘"。
 *    反过来（把动作写死在这里）会让这个文件依赖 App 的每一个回调，
 *    加一个动作就要改两个文件。
 * 2. **过滤是子序列匹配**，不是子串。中文里"新对话"和"对话"都该命中"对话"，
 *    而打"xd"也该能命中"新对话" —— 子串匹配做不到后者，用户就会觉得
 *    "我明明打了字它说没有"。
 * 3. **用绝对定位的 key 列表 + 键盘高亮**，不做虚拟滚动：三五十个动作的
 *    规模下，虚拟滚动是纯负担。
 */

export interface PaletteAction {
  id: string
  label: string
  /** 右侧那行灰字：说明它做什么，或者快捷键 */
  hint?: string
  /** 分组标题，用于在长列表里分开"动作"和"跳到某次对话" */
  group: string
  icon: 'chat' | 'plus' | 'search' | 'download' | 'theme' | 'panel' | 'memory' | 'star' | 'history'
  run: () => void
}

const ICONS = {
  chat: IconChat,
  plus: IconPlus,
  search: IconSearch,
  download: IconDownload,
  theme: IconSun,
  panel: IconPanel,
  memory: IconMemory,
  star: IconStar,
  history: IconHistory,
} as const

/** 子序列匹配：needle 的每个字符都要在 haystack 里按顺序出现。 */
export function fuzzyMatch(needle: string, haystack: string): boolean {
  const n = needle.trim().toLowerCase()
  if (!n) return true
  const h = haystack.toLowerCase()
  let i = 0
  for (const ch of h) {
    if (ch === n[i]) i += 1
    if (i === n.length) return true
  }
  return i === n.length
}

interface Props {
  actions: PaletteAction[]
  themeChoice: 'system' | 'light' | 'dark'
  onClose: () => void
  /** 正在退场（由 useDelayedUnmount 驱动）。这期间只播动画，不再接任何交互。 */
  closing?: boolean
}

export function CommandPalette({ actions, themeChoice, onClose, closing = false }: Props) {
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)

  const visible = useMemo(
    () => actions.filter((a) => fuzzyMatch(query, `${a.label} ${a.hint ?? ''} ${a.group}`)),
    [actions, query],
  )

  // 筛选结果变了就回到第一条：不然会出现"高亮停在一个已经不显示的项上"，
  // 按回车执行的是看不见的东西
  useEffect(() => {
    setActive(0)
  }, [query])

  useEffect(() => {
    listRef.current
      ?.querySelector('.palette-item--active')
      ?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const run = (action: PaletteAction | undefined) => {
    if (!action) return
    onClose()
    // 先关面板再执行：动作里可能有"聚焦输入框"，面板还开着会立刻被抢回焦点
    window.setTimeout(action.run, 0)
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((i) => (visible.length ? (i + 1) % visible.length : 0))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => (visible.length ? (i - 1 + visible.length) % visible.length : 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      run(visible[active])
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    }
  }

  let lastGroup = ''

  return (
    <div
      className={`palette-backdrop${closing ? ' is-closing' : ''}`}
      onClick={onClose}
      role="presentation"
      // 自己管键盘：焦点在输入框上，方向键从那里冒泡上来
      onKeyDown={onKeyDown}
    >
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="命令面板"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="palette-head">
          <IconCommand width={15} height={15} />
          <input
            className="palette-input"
            value={query}
            autoFocus
            placeholder="输入命令，比如「新对话」「导出」「收藏」"
            aria-label="搜索命令"
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="button" className="icon-btn" aria-label="关闭" onClick={onClose}>
            <IconClose width={14} height={14} />
          </button>
        </div>

        <div className="palette-list" ref={listRef}>
          {visible.length === 0 ? (
            <p className="palette-empty">没有匹配的命令</p>
          ) : (
            visible.map((a, i) => {
              const Icon = ICONS[a.icon]
              const showGroup = a.group !== lastGroup
              lastGroup = a.group
              return (
                <div key={a.id}>
                  {showGroup && <div className="palette-group">{a.group}</div>}
                  <button
                    type="button"
                    className={`palette-item${i === active ? ' palette-item--active' : ''}`}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => run(a)}
                  >
                    <Icon width={15} height={15} />
                    <span className="palette-label">{a.label}</span>
                    {a.hint && <span className="palette-hint">{a.hint}</span>}
                  </button>
                </div>
              )
            })
          )}
        </div>

        <div className="palette-foot">
          <span>
            <kbd>↑</kbd> <kbd>↓</kbd> 选择
          </span>
          <span>
            <kbd>Enter</kbd> 执行
          </span>
          <span>
            <kbd>Esc</kbd> 关闭
          </span>
          <span style={{ marginLeft: 'auto' }}>
            当前主题：{themeChoice === 'system' ? '跟随系统' : themeChoice === 'light' ? '浅色' : '深色'}
          </span>
        </div>
      </div>
    </div>
  )
}
