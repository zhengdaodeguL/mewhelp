/**
 * 导出对话。
 *
 * 纯函数（`sessionToMarkdown`）和副作用（`downloadText`）分开：
 * 前者能被 node 直接跑着测，后者只是个 Blob 下载壳子。
 *
 * 导出格式选 Markdown 而不是 JSON：用户拿到这份东西之后，下一步多半是
 * 贴进工单、微信或者笔记里 —— JSON 在这三处都是不可读的噪声。
 */

import type { ChatMessage } from './types'

export interface ExportOptions {
  title?: string
  /** 导出时间（ISO）。由调用方传入而不是内部取 now() —— 让输出可复现、可测。 */
  exportedAt?: string
  /** 附在末尾的补充信息（比如工作区里那些订单/物流卡片） */
  notes?: string
}

function stamp(iso?: string): string {
  if (!iso) return ''
  return iso.slice(0, 16).replace('T', ' ')
}

/**
 * 把一轮对话渲染成 Markdown。
 *
 * 只导出**正常消息**：`notice` 是界面上的状态提示条（"已转人工""回答中断了"），
 * 它不是客服说过的话，混进对话记录会让人以为客服真这么说了。
 */
export function sessionToMarkdown(messages: ChatMessage[], options: ExportOptions = {}): string {
  const lines: string[] = ['# MewHelp 客服对话记录', '']
  if (options.title) lines.push(`- 会话：${options.title}`)
  if (options.exportedAt) lines.push(`- 导出时间：${stamp(options.exportedAt)}`)
  lines.push('', '---', '')

  let wrote = false
  for (const msg of messages) {
    if (msg.kind !== 'normal') continue
    const text = (msg.content || '').trim()
    if (!text) continue
    lines.push(`**${msg.role === 'user' ? '我' : '客服'}：**`, '', text, '')
    wrote = true
  }
  if (!wrote) lines.push('（本次会话还没有内容）', '')

  const notes = (options.notes || '').trim()
  if (notes) {
    lines.push('---', '', '## 本次涉及的信息', '', notes, '')
  }
  return lines.join('\n')
}

/** 文件名要能一眼认出是哪次对话，又不至于长到放不下。 */
export function exportFileName(title: string, iso?: string): string {
  const day = (iso || '').slice(0, 10) || 'session'
  const safe = (title || '对话')
    .replace(/[\\/:*?"<>|\s]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 20)
  return `mewhelp-${day}-${safe || '对话'}.md`
}

/** 触发一次浏览器下载。不是安全上下文也能用（走 Blob URL，不依赖剪贴板权限）。 */
export function downloadText(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // 立刻回收会让部分浏览器来不及开始下载，所以延后一拍
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
