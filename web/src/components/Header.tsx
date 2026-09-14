import type { ThemeChoice } from '../hooks/useTheme'
import { IconMonitor, IconMoon, IconPanel, IconPlus, IconSun } from './icons'

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
  onReset: () => void
  /** 窄屏下工作区是抽屉，靠这个按钮开合 */
  showWorkspaceToggle: boolean
  workspaceOpen: boolean
  workspaceCount: number
  onToggleWorkspace: () => void
  themeChoice: ThemeChoice
  onCycleTheme: () => void
}

export function Header({
  online,
  onReset,
  showWorkspaceToggle,
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
          className="icon-btn"
          title={`主题：${THEME_LABEL[themeChoice]}（点击切换）`}
          aria-label={`主题：${THEME_LABEL[themeChoice]}，点击切换`}
          onClick={onCycleTheme}
        >
          <ThemeIcon />
        </button>

        {showWorkspaceToggle && (
          <button
            type="button"
            className="icon-btn"
            title="工作区"
            aria-label="工作区"
            aria-expanded={workspaceOpen}
            onClick={onToggleWorkspace}
          >
            <IconPanel />
            {workspaceCount > 0 && <span className="badge">{workspaceCount}</span>}
          </button>
        )}

        <button type="button" className="icon-btn" title="新会话" aria-label="新会话" onClick={onReset}>
          <IconPlus />
        </button>
      </div>
    </header>
  )
}
