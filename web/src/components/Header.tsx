interface Props {
  online: boolean
  onReset: () => void
}

export function Header({ online, onReset }: Props) {
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
      <button type="button" className="ghost-btn" onClick={onReset}>
        新会话
      </button>
    </header>
  )
}
