interface Props {
  online: boolean
  onReset: () => void
  /** 窄屏下工作区是抽屉，靠这个按钮开合 */
  showWorkspaceToggle: boolean
  workspaceOpen: boolean
  workspaceCount: number
  onToggleWorkspace: () => void
}

export function Header({
  online,
  onReset,
  showWorkspaceToggle,
  workspaceOpen,
  workspaceCount,
  onToggleWorkspace,
}: Props) {
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
        {showWorkspaceToggle && (
          <button
            type="button"
            className="ghost-btn"
            aria-expanded={workspaceOpen}
            onClick={onToggleWorkspace}
          >
            工作区
            {workspaceCount > 0 && <span className="badge">{workspaceCount}</span>}
          </button>
        )}
        <button type="button" className="ghost-btn" onClick={onReset}>
          新会话
        </button>
      </div>
    </header>
  )
}
