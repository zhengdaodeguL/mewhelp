import { useState, type FormEvent } from 'react'
import { authenticate, type AuthUser } from '../api/client'

interface Props {
  onAuthenticated: (user: AuthUser) => void
}

export function AuthGate({ onAuthenticated }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError('')
    const result = await authenticate(mode, email, password)
    setBusy(false)
    if (result.user) onAuthenticated(result.user)
    else setError(result.error)
  }

  return (
    <main className="auth-screen">
      <section className="auth-panel" aria-labelledby="auth-title">
        <div className="auth-brand">
          <span className="auth-brand-mark" aria-hidden="true">M</span>
          <span>MewHelp</span>
        </div>
        <div className="auth-heading">
          <p className="auth-eyebrow">在线客服</p>
          <h1 id="auth-title">{mode === 'login' ? '欢迎回来' : '创建账号'}</h1>
          <p className="auth-subtitle">{mode === 'login' ? '登录后继续你的对话' : '用邮箱注册，开始咨询'}</p>
        </div>
        <form className="auth-form" onSubmit={(event) => void submit(event)}>
          <label className="auth-field">
            <span>邮箱</span>
            <input
              type="email"
              name="email"
              autoComplete="email"
              autoFocus
              required
              maxLength={254}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="name@example.com"
            />
          </label>
          <label className="auth-field">
            <span>密码</span>
            <input
              type="password"
              name="password"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              minLength={10}
              maxLength={128}
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="至少 10 个字符"
            />
          </label>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy ? '请稍候…' : mode === 'login' ? '登录' : '注册'}
          </button>
        </form>
        <p className="auth-switch">
          {mode === 'login' ? '还没有账号？' : '已经有账号？'}{' '}
          <button
            type="button"
            onClick={() => {
              setError('')
              setMode((current) => current === 'login' ? 'register' : 'login')
            }}
          >
            {mode === 'login' ? '注册' : '登录'}
          </button>
        </p>
      </section>
      <footer className="auth-footer">MewHelp · 客户服务</footer>
    </main>
  )
}
