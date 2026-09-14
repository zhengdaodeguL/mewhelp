import type { ThemeChoice } from '../hooks/useTheme'
import {
  IconDownload,
  IconHistory,
  IconMonitor,
  IconMoon,
  IconPanel,
  IconPlus,
  IconSearch,
  IconSun,
} from './icons'

const THEME_LABEL: Record<ThemeChoice, string> = {
  system: '跟随系统',
  light: '浅色',
  dark: '深色',
}

const THEME_ICON: Record<ThemeChoice, typeof IconSun> = {
  system: IconMonitor,
  light: IconSun,
  dark: IconMoon,
}

interface Props {
  online: boolean
  /** 历史栏开关（窄屏下它就是抽屉开关） */
  railOpen: boolean
  onToggleRail: () => void
  /** 会话内查找 */
  findOpen: boolean
  onToggleFind: () => void
  /** 导出本次对话 */
  onExport: () => void
  canExport: boolean
  onNewSession: () => void
  /** 工作区开关：宽屏收起它把整屏让给对话，窄屏开抽屉 */
  workspaceOpen: boolean
  workspaceCount: number
  onToggleWorkspace: () => void
  themeChoice: ThemeChoice
  onCycleTheme: () => void
}

export function Header({
  online,
  railOpen,
  onToggleRail,
  findOpen,
  onToggleFind,
  onExport,
  canExport,
  onNewSession,
  workspaceOpen,
  workspaceCount,
  onToggleWorkspace,
  themeChoice,
  onCycleTheme,
}: Props) {
  const ThemeIcon = THEME_ICON[themeChoice]

  return (
    <header className="header">
      <div className="header-left">
        <button
          type="button"
          className={`icon-btn${railOpen ? ' icon-btn--on' : ''}`}
          title={railOpen ? '收起历史对话' : '展开历史对话'}
          aria-label="历史对话"
          aria-expanded={railOpen}
          onClick={onToggleRail}
        >
          <IconHistory />
        </button>
        <div className="brand-mark" aria-hidden="true">
          M
        </div>
        <div className="brand-text">
          <h1 className="brand-title">MewHelp 客服</h1>
          <p className="brand-status">
            <span className={`status-dot ${online ? 'status-dot--online' : 'status-dot--offline'}`} />
            {online ? '在线' : '连接中…'}
          </p>
        </div>
      </div>

      <div className="header-actions">
        <button
          type="button"
          className={`icon-btn${findOpen ? ' icon-btn--on' : ''}`}
          title="在对话里查找（Ctrl+F）"
          aria-label="在对话里查找"
          aria-expanded={findOpen}
          onClick={onToggleFind}
        >
          <IconSearch />
        </button>

        <button
          type="button"
          className="icon-btn"
          title="导出这次对话（Markdown）"
          aria-label="导出这次对话"
          disabled={!canExport}
          onClick={onExport}
        >
          <IconDownload />
        </button>

        <button
          type="button"
          className="icon-btn"
          title={`主题：${THEME_LABEL[themeChoice]}（点击切换）`}
          aria-label={`主题：${THEME_LABEL[themeChoice]}，点击切换`}
          onClick={onCycleTheme}
        >
          <ThemeIcon />
        </button>

        <button
          type="button"
          className={`icon-btn${workspaceOpen ? ' icon-btn--on' : ''}`}
          title={workspaceOpen ? '收起工作区' : '展开工作区'}
          aria-label="工作区"
          aria-expanded={workspaceOpen}
          onClick={onToggleWorkspace}
        >
          <IconPanel />
          {workspaceCount > 0 && <span className="badge">{workspaceCount}</span>}
        </button>

        <button
          type="button"
          className="icon-btn"
          title="新对话"
          aria-label="新对话"
          onClick={onNewSession}
        >
          <IconPlus />
        </button>
      </div>
    </header>
  )
}
