import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { checkHealth, currentUser, logout, type AuthUser } from './api/client'
import { AuthGate } from './components/AuthGate'
import { CommandPalette, type PaletteAction } from './components/CommandPalette'
import { Composer } from './components/Composer'
import { FindBar } from './components/FindBar'
import { Header } from './components/Header'
import { HistoryRail } from './components/HistoryRail'
import { MessageList } from './components/MessageList'
import { SplitHandle } from './components/SplitHandle'
import { Toaster } from './components/Toaster'
import { Workspace, type WorkspaceTab } from './components/Workspace'
import { useChat } from './hooks/useChat'
import { useHistory } from './hooks/useHistory'
import { useIdentity } from './hooks/useIdentity'
import { useGraph } from './hooks/useGraph'
import { useMediaQuery } from './hooks/useMediaQuery'
import { useMemory } from './hooks/useMemory'
import { useSplit } from './hooks/useSplit'
import { useStars } from './hooks/useStars'
import { useTheme } from './hooks/useTheme'
import { downloadText, exportFileName, sessionToMarkdown } from './session-export'
import { toast } from './toast'
import { trackPointer, useCrossfade, useDelayedUnmount, withThemeWipe } from './transition'
import type { ChatMessage, MemoryItem, Rating } from './types'
import { cardToText } from './workspace'

/** 断点与 CSS 里的保持一致：宽屏三栏常驻，窄屏左右两栏变抽屉。 */
const NARROW = '(max-width: 1024px)'

/** 查找定位之后闪一下的时长。够看到，短到不打扰。 */
const FLASH_MS = 1400

export default function App() {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [authError, setAuthError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    void currentUser(controller.signal).then(({ user: account, error }) => {
      if (controller.signal.aborted) return
      setUser(account)
      setAuthError(error ?? '')
      setAuthLoading(false)
    })
    return () => controller.abort()
  }, [])

  const signOut = useCallback(async () => {
    const signedOut = await logout()
    if (!signedOut) {
      toast('暂时无法退出登录，请检查网络后重试。', 'error')
      return
    }
    setUser(null)
  }, [])

  if (authLoading) {
    return <main className="auth-screen"><div className="auth-loading" aria-label="正在加载" /></main>
  }
  if (authError) {
    return (
      <main className="auth-screen">
        <section className="auth-panel" aria-labelledby="auth-error-title">
          <div className="auth-brand">
            <span className="auth-brand-mark" aria-hidden="true">M</span>
            <span>MewHelp</span>
          </div>
          <div className="auth-heading">
            <h1 id="auth-error-title">暂时无法连接</h1>
            <p className="auth-subtitle" role="alert">{authError}</p>
          </div>
          <button
            className="auth-submit auth-retry"
            type="button"
            onClick={() => {
              setAuthLoading(true)
              setAuthError('')
              void currentUser().then(({ user: account, error }) => {
                setUser(account)
                setAuthError(error ?? '')
                setAuthLoading(false)
              })
            }}
          >
            重新连接
          </button>
        </section>
      </main>
    )
  }
  if (!user) return <AuthGate onAuthenticated={setUser} />
  return <ChatApp key={user.id} accountId={user.id} email={user.email} onLogout={() => void signOut()} />
}

interface ChatAppProps {
  accountId: string
  email: string
  onLogout: () => void
}

function ChatApp({ accountId, email, onLogout }: ChatAppProps) {
  const isNarrow = useMediaQuery(NARROW)
  const { sessionId, setSessionId, startNewSession } = useIdentity(accountId)
  const { choice: themeChoice, cycle: cycleTheme, setChoice: setThemeChoice } = useTheme()
  const { ratio, setRatio, nudge, reset: resetRatio } = useSplit()
  const history = useHistory()
  const memory = useMemory()
  const stars = useStars(accountId)

  const [online, setOnline] = useState(false)
  const [railOpen, setRailOpen] = useState(true)
  const [wsOpen, setWsOpen] = useState(true)
  //: 窄屏下同一时刻只开一个抽屉 —— 两个都开在手机上什么都看不见
  const [drawer, setDrawer] = useState<'rail' | 'workspace' | null>(null)
  const [tab, setTab] = useState<WorkspaceTab>('cards')
  //: 图谱只在标签第一次被打开时才拉数据（理由见 useGraph）
  const graph = useGraph(useMemo(() => tab === 'graph', [tab]))
  const [findOpen, setFindOpen] = useState(false)
  const [findQuery, setFindQuery] = useState('')
  const [findIndex, setFindIndex] = useState(0)
  const [flashId, setFlashId] = useState('')
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [quote, setQuote] = useState<ChatMessage | null>(null)
  const [resizing, setResizing] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const flashTimer = useRef<number | undefined>(undefined)

  //: 换会话时整段对话交叉淡出淡入 —— 切会话是"换了一件事在聊"，
  //: 消息原地替换会让人以为界面卡了一下。
  const conversation = useCrossfade(sessionId, 240)

  //: 命令面板关闭时不要"啪"地消失。它是全屏浮层，硬切最显眼。
  const palette = useDelayedUnmount(paletteOpen, 180)

  //: 抽屉遮罩同理，时长跟抽屉的滑出（220ms）对齐
  const scrim = useDelayedUnmount(isNarrow && drawer !== null, 220)

  const chat = useChat({
    sessionId,
    quote: quote?.content ?? '',
    onTurnEnd: () => void history.refresh(),
    onMemory: memory.replace,
  })
  const { rate, send } = chat
  const starsToggle = stars.toggle

  //: 切会话时"正在退场的那一屏"必须**在切换的那一刻**拍下来。
  //:
  //: useChat 有一个 effect 专门在 sessionId 变化时把 messages 重置成开场白，
  //: 而 effect 在同一轮里就跑完了 —— 等它跑完再读 chat.messages，拿到的是新会话
  //: 的内容，"淡出的那一层"和"淡入的那一层"于是长得一模一样，转场退化成原地闪一下。
  //:
  //: 时机只有这里对：本次渲染时 sessionId 已经变了，但 useChat 的 effect 还没执行，
  //: chat.messages 仍是**上一个会话**的消息。趁着这一帧把它存下来。
  //: 释放条件放在 `current !== sessionId` 上，重复渲染（StrictMode 会跑两遍）不会覆盖。
  const leavingMessages = useRef<ChatMessage[]>(chat.messages)
  const frozenFor = useRef(sessionId)
  if (frozenFor.current !== sessionId) {
    leavingMessages.current = chat.messages
    frozenFor.current = sessionId
  }

  //: 主题切换走"圆形揭示"：从指针位置扩散开，而不是整屏硬换一次色。
  //: 三处入口（页头按钮、命令面板的三条）都走这两个包装过的版本，口径一致。
  const applyTheme = useCallback(
    (next: 'system' | 'light' | 'dark') => withThemeWipe(() => setThemeChoice(next)),
    [setThemeChoice],
  )
  const cycleThemeWipe = useCallback(
    () => withThemeWipe(() => cycleTheme()),
    [cycleTheme],
  )

  // 只用来点亮页头的状态灯；拿不到就当离线，不打扰用户
  useEffect(() => {
    const controller = new AbortController()
    const poll = async () => setOnline((await checkHealth(controller.signal)).ok)
    void poll()
    const timer = window.setInterval(poll, 30_000)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [])

  //: 最近一次指针位置。主题切换的圆形揭示要拿它当圆心 ——
  //: 从用户刚刚点的地方扩散开来，而不是从屏幕中间。
  useEffect(() => trackPointer(), [])

  // 抽屉只在窄屏存在；转回宽屏时复位，免得下次变窄又莫名其妙弹开
  useEffect(() => {
    if (!isNarrow) setDrawer(null)
  }, [isNarrow])

  useEffect(
    () => () => {
      if (flashTimer.current) window.clearTimeout(flashTimer.current)
    },
    [],
  )

  // ---------------- 会话内查找 ----------------

  // 命中的是**消息**而不是字符位置：用户说的是"我记得它说过那么一句"，
  // 他需要定位到那一条，而不是定位到某段文本里的第 37 个字。
  const matches = useMemo(() => {
    const q = findQuery.trim().toLowerCase()
    if (!q) return []
    return chat.messages.filter((m) => m.content.toLowerCase().includes(q)).map((m) => m.id)
  }, [chat.messages, findQuery])

  const jumpTo = useCallback((messageId: string) => {
    if (!messageId) return
    document.getElementById(`msg-${messageId}`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    })
    setFlashId(messageId)
    if (flashTimer.current) window.clearTimeout(flashTimer.current)
    flashTimer.current = window.setTimeout(() => setFlashId(''), FLASH_MS)
  }, [])

  const stepMatch = useCallback(
    (delta: number) => {
      if (matches.length === 0) return
      const next = (findIndex + delta + matches.length) % matches.length
      setFindIndex(next)
      jumpTo(matches[next])
    },
    [findIndex, matches, jumpTo],
  )

  // 换关键词就回到第一条并跳过去 —— 不跳的话，用户看到"1 / 7"却不知道那条在哪
  useEffect(() => {
    setFindIndex(0)
    if (matches.length > 0) jumpTo(matches[0])
    // jumpTo 是稳定引用；matches 每次输入都会变，这是刻意的
  }, [findQuery]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggleFind = useCallback(() => {
    setFindOpen((open) => {
      if (open) setFindQuery('')
      return !open
    })
  }, [])

  // Ctrl/⌘+K 命令面板、Ctrl/⌘+F 查找、Ctrl/⌘+B 收起历史栏
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      const key = e.key.toLowerCase()
      if (key === 'k') {
        e.preventDefault()
        setPaletteOpen(true)
      } else if (key === 'f') {
        e.preventDefault()
        setFindOpen(true)
      } else if (key === 'b') {
        e.preventDefault()
        setRailOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // ---------------- 动作 ----------------

  const newSession = useCallback(() => {
    startNewSession()
    setDrawer(null)
    setTab('cards')
    setFindQuery('')
    setQuote(null)
    inputRef.current?.focus()
  }, [startNewSession])

  const pickSession = useCallback(
    (id: string) => {
      setSessionId(id)
      setDrawer(null)
      setFindQuery('')
      setQuote(null)
    },
    [setSessionId],
  )

  const removeSession = useCallback(
    async (id: string) => {
      const ok = await history.remove(id)
      // 删掉的正好是当前会话时，立刻开一个新的 —— 否则界面还停在一条
      // 已经不存在的会话上，下一次发言会把它又凭空建回来
      if (ok && id === sessionId) startNewSession()
      toast(ok ? '已删除这次对话' : '删除没成功，稍后再试', ok ? 'ok' : 'error')
    },
    [history, sessionId, startNewSession],
  )

  const onStar = useCallback(
    (message: ChatMessage) => starsToggle(message.turnId),
    [starsToggle],
  )

  const onRate = useCallback(
    (message: ChatMessage, rating: Rating) => {
      rate(message, rating)
      // 点踩之后让用户知道"收到"。不说这条反馈去哪了、会不会有人处理 ——
      // 那些是内部流程，对用户而言这一段就是过程。
      if (message.rating !== 'down' && rating === 'down') {
        toast('已记录，谢谢你的反馈', 'ok')
      }
    },
    [rate],
  )

  const onQuote = useCallback(
    (message: ChatMessage) => setQuote((prev) => (prev?.id === message.id ? null : message)),
    [],
  )

  /** 发送：把引用条一起清掉，并让它只作用于这一轮。 */
  const sendWithQuote = useCallback(
    (text: string) => {
      void send(text)
      setQuote(null)
    },
    [send],
  )

  const starred = useMemo(() => {
    const byTurn = new Map(chat.messages.map((m) => [m.turnId, m]))
    return stars.ids.map((id) => byTurn.get(id)).filter((m): m is ChatMessage => Boolean(m))
  }, [chat.messages, stars.ids])

  const canExport = chat.messages.some((m) => m.role === 'user' && m.kind === 'normal')

  const exportSession = useCallback(() => {
    const title = history.sessions.find((s) => s.sessionId === sessionId)?.title ?? ''
    const iso = new Date().toISOString()
    // 把工作区里的订单/物流也带上：用户导出这份记录时，多半连"当时查到的那单"
    // 一起要（贴工单、发给朋友帮看），只导气泡会丢掉一半信息。
    // 用 Map 按 id 去重 —— 工作区里同名卡片只留最新那张，导出也该如此。
    const notes = [...new Map(chat.cards.map((c) => [c.id, c])).values()]
      .map(cardToText)
      .join('\n\n')
    const name = exportFileName(title, iso)
    downloadText(name, sessionToMarkdown(chat.messages, { title, exportedAt: iso, notes }))
    toast(`已导出 ${name}`, 'ok')
  }, [chat.cards, chat.messages, history.sessions, sessionId])

  const addMemory = useCallback(
    async (text: string) => {
      const ok = await memory.add(text)
      toast(ok ? '记住了，以后会记得' : '没记住，稍后再试', ok ? 'ok' : 'error')
    },
    [memory],
  )

  const removeMemory = useCallback(
    async (item: MemoryItem) => {
      const ok = await memory.remove(item)
      toast(ok ? '已忘记这条' : '删除没成功，稍后再试', ok ? 'ok' : 'error')
    },
    [memory],
  )

  // ---------------- 布局 ----------------

  const railClass = isNarrow
    ? ` rail-slot--drawer${drawer === 'rail' ? ' rail-slot--open' : ''}`
    : railOpen
      ? ''
      : ' rail-slot--hidden'

  const wsClass = isNarrow
    ? ` pane-workspace--drawer${drawer === 'workspace' ? ' pane-workspace--open' : ''}`
    : wsOpen
      ? ''
      : ' pane-workspace--hidden'

  // 宽屏收起工作区是为了把整屏让给对话（笔记本上很常用），
  // 窄屏则是开抽屉 —— 两种模式共用一个按钮，按断点分派。
  const workspaceOpen = isNarrow ? drawer === 'workspace' : wsOpen
  const toggleWorkspace = () => {
    if (isNarrow) setDrawer((d) => (d === 'workspace' ? null : 'workspace'))
    else setWsOpen((v) => !v)
  }

  //: 图谱计数不进徽章：徽章回答的是"工作区有几件**与这次对话有关**的东西"，
  //: 图谱是整个知识库的常驻地图，把它算进来，这个数字从会话开始就不再是 0，
  //: 徽章也就失去了"有东西可看"的信号意义
  const wsCount = chat.cards.length + memory.items.length + starred.length

  // ---------------- 命令面板的动作清单 ----------------
  // 在这里组装，而不是写死在 CommandPalette 里：那个组件只负责"筛 + 选 + 键盘"，
  // 新增一个动作不用改它的代码。
  const actions = useMemo<PaletteAction[]>(() => {
    const list: PaletteAction[] = [
      {
        id: 'focus',
        label: '聚焦到输入框',
        hint: '说点什么',
        group: '动作',
        icon: 'chat',
        run: () => inputRef.current?.focus(),
      },
      { id: 'new', label: '开一次新对话', group: '动作', icon: 'plus', run: newSession },
      { id: 'find', label: '在本次对话里查找', hint: 'Ctrl+F', group: '动作', icon: 'search', run: () => setFindOpen(true) },
      {
        id: 'export',
        label: '导出这次对话（Markdown）',
        group: '动作',
        icon: 'download',
        run: exportSession,
      },
      {
        id: 'railtoggle',
        label: railOpen ? '收起历史栏' : '展开历史栏',
        hint: 'Ctrl+B',
        group: '动作',
        icon: 'history',
        run: () => setRailOpen((v) => !v),
      },
      {
        id: 'wstoggle',
        label: workspaceOpen ? '收起工作区' : '展开工作区',
        group: '动作',
        icon: 'panel',
        run: toggleWorkspace,
      },
      {
        id: 'theme',
        label: `切换主题（当前：${themeChoice === 'system' ? '跟随系统' : themeChoice === 'light' ? '浅色' : '深色'}）`,
        group: '外观',
        icon: 'theme',
        run: cycleThemeWipe,
      },
      {
        id: 'theme-light',
        label: '浅色主题',
        group: '外观',
        icon: 'theme',
        run: () => applyTheme('light'),
      },
      {
        id: 'theme-dark',
        label: '深色主题',
        group: '外观',
        icon: 'theme',
        run: () => applyTheme('dark'),
      },
      {
        id: 'tab-memory',
        label: '看系统记住了我什么',
        hint: `${memory.items.length} 条`,
        group: '跳到',
        icon: 'memory',
        run: () => {
          setTab('memory')
          if (!isNarrow && !wsOpen) setWsOpen(true)
        },
      },
      {
        id: 'tab-stars',
        label: '看我收藏的消息',
        hint: `${starred.length} 条`,
        group: '跳到',
        icon: 'star',
        run: () => {
          setTab('stars')
          if (!isNarrow && !wsOpen) setWsOpen(true)
        },
      },
      {
        id: 'tab-cards',
        label: '看这次查到的订单与依据',
        hint: `${chat.cards.length} 张`,
        group: '跳到',
        icon: 'panel',
        run: () => {
          setTab('cards')
          if (!isNarrow && !wsOpen) setWsOpen(true)
        },
      },
      {
        id: 'tab-graph',
        label: '看知识库图谱',
        hint: `${graph.graph.count} 条知识`,
        group: '跳到',
        icon: 'history',
        run: () => {
          setTab('graph')
          if (!isNarrow && !wsOpen) setWsOpen(true)
        },
      },
    ]

    // 最近的历史对话也进面板：切会话是"用得多、但每次都要去左边找"的动作，
    // 值得有一条直达路径
    for (const s of history.sessions.slice(0, 8)) {
      list.push({
        id: `session-${s.sessionId}`,
        label: s.title,
        hint: s.sessionId === sessionId ? '当前' : '历史对话',
        group: '最近对话',
        icon: 'history',
        run: () => pickSession(s.sessionId),
      })
    }
    return list
  }, [
    chat.cards.length,
    applyTheme,
    cycleThemeWipe,
    exportSession,
    graph.graph.count,
    history.sessions,
    isNarrow,
    memory.items.length,
    newSession,
    pickSession,
    railOpen,
    sessionId,
    starred.length,
    themeChoice,
    toggleWorkspace,
    workspaceOpen,
    wsOpen,
  ])

  return (
    <>
      <div className="app">
        <div className={`rail-slot${railClass}`}>
        <HistoryRail
          sessions={history.sessions}
          activeId={sessionId}
          loading={history.loading}
          onSelect={pickSession}
          onNew={newSession}
          onRename={(id, title) => void history.rename(id, title)}
          onDelete={(id) => void removeSession(id)}
          onClose={isNarrow ? () => setDrawer(null) : undefined}
        />
      </div>

      <div
        className={`panes${resizing ? ' panes--dragging' : ''}`}
        style={{ '--chat-grow': ratio, '--ws-grow': 1 - ratio } as CSSProperties}
      >
        <div className="pane-chat">
          <Header
            email={email}
            onLogout={onLogout}
            online={online}
            railOpen={isNarrow ? drawer === 'rail' : railOpen}
            onToggleRail={() =>
              isNarrow ? setDrawer((d) => (d === 'rail' ? null : 'rail')) : setRailOpen((v) => !v)
            }
            findOpen={findOpen}
            onToggleFind={toggleFind}
            onPalette={() => setPaletteOpen(true)}
            onExport={exportSession}
            canExport={canExport}
            onNewSession={newSession}
            workspaceOpen={workspaceOpen}
            workspaceCount={wsCount}
            onToggleWorkspace={toggleWorkspace}
            themeChoice={themeChoice}
            onCycleTheme={cycleThemeWipe}
          />

          {findOpen && (
            <FindBar
              query={findQuery}
              count={matches.length}
              index={matches.length === 0 ? 0 : findIndex + 1}
              onQuery={setFindQuery}
              onStep={stepMatch}
              onClose={toggleFind}
            />
          )}

          <div className="messages-wrap swap-root">
            {conversation.leaving !== null && (
              <div className="swap-out" aria-hidden="true">
                <MessageList
                  /* 退场的那一屏用**冻住的那份**，不是 chat.messages ——
                     后者此刻已经是新会话的内容了（见上面 leavingMessages 的说明） */
                  messages={leavingMessages.current}
                  canRetry={false}
                  onRetry={chat.retryLast}
                  query=""
                  flashId=""
                  quotedId=""
                  starredIds={stars.ids}
                  onRate={onRate}
                  onStar={onStar}
                  onQuote={onQuote}
                  streaming={false}
                />
              </div>
            )}
            <div className="swap-in swap-fill" key={conversation.current}>
              <MessageList
                messages={chat.messages}
                canRetry={chat.canRetry}
                onRetry={chat.retryLast}
                query={findQuery}
                flashId={flashId}
                quotedId={quote?.id ?? ''}
                starredIds={stars.ids}
                onRate={onRate}
                onStar={onStar}
                onQuote={onQuote}
                streaming={chat.isStreaming}
                onSuggest={sendWithQuote}
              />
            </div>
          </div>

          <Composer
            isStreaming={chat.isStreaming}
            onSend={sendWithQuote}
            onStop={chat.stop}
            inputRef={inputRef}
            quote={quote}
            onClearQuote={() => setQuote(null)}
          />
        </div>

        {/* 分隔条只在工作区可见、且不是窄屏时才有意义 ——
            窄屏下两栏是抽屉，中间根本没有一条可以拖的缝 */}
        {!isNarrow && wsOpen && (
          <SplitHandle
            ratio={ratio}
            onRatio={setRatio}
            onNudge={nudge}
            onReset={resetRatio}
            onDraggingChange={setResizing}
          />
        )}

        <div className={`pane-workspace${wsClass}`}>
          <Workspace
            tab={tab}
            onTab={setTab}
            cards={chat.cards}
            onClearCards={chat.clearCards}
            memories={memory.items}
            memoryLoading={memory.loading}
            memoryBusy={memory.busy}
            memoryError={memory.error}
            onAddMemory={(text) => void addMemory(text)}
            onToggleMemoryPin={(item) => void memory.togglePin(item)}
            onDeleteMemory={(item) => void removeMemory(item)}
            starred={starred}
            onJumpTo={jumpTo}
            onUnstar={(m) => starsToggle(m.turnId)}
            graph={graph.graph}
            graphLoading={graph.loading}
            graphError={graph.error}
            onRefreshGraph={() => void graph.refresh()}
            onClose={isNarrow ? () => setDrawer(null) : undefined}
          />
        </div>
      </div>

      {/* 抽屉打开时的遮罩：点空白处收起。
          退场要跟抽屉的滑出（220ms）一起走完，所以用 useDelayedUnmount 把它多留一会儿；
          外层的 isNarrow 判断是给"窄屏转宽屏"准备的 —— 那种情况下抽屉是被 effect
          直接清掉的，遮罩不该还留 220ms 在桌面端糊一层黑。 */}
      {isNarrow && scrim.mounted && (
        <button
          type="button"
          className={`drawer-scrim${scrim.closing ? ' is-closing' : ''}`}
          aria-label="收起面板"
          onClick={() => setDrawer(null)}
        />
      )}

      {palette.mounted && (
        <CommandPalette
          actions={actions}
          themeChoice={themeChoice}
          onClose={() => setPaletteOpen(false)}
          closing={palette.closing}
        />
      )}

      <Toaster />
      </div>
    </>
  )
}
