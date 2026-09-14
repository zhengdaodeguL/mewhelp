/**
 * 展示用的格式化。
 *
 * 单独成文件、并且把"现在"作为参数传进来（默认 `Date.now()`），
 * 是为了让输出可测：时间相关的函数一旦内部偷偷取当前时间，
 * 单测就只能写成"大约是多少"这种没有断言价值的形状。
 */

/** 后端下发的时间是带时区标记的 UTC（`...Z` / `+00:00`），`Date` 能直接解析。 */
function parse(iso: string): Date | null {
  if (!iso) return null
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d
}

/**
 * 相对时间：列表里那一行灰色小字。
 *
 * 超过一周就不再说"几天前"，直接给日期 —— "37 天前"这种说法要求用户自己
 * 心算，而历史列表里的时间是为了**定位**，不是为了精确。
 */
export function relativeTime(iso: string, now: number = Date.now()): string {
  const d = parse(iso)
  if (!d) return ''
  const diff = now - d.getTime()

  // 客户端时钟比服务端慢一点是常事，别显示"负 3 分钟前"
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  if (diff < 172_800_000) return '昨天'
  if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)} 天前`

  const sameYear = d.getFullYear() === new Date(now).getFullYear()
  const md = `${d.getMonth() + 1}月${d.getDate()}日`
  return sameYear ? md : `${d.getFullYear()}年${md}`
}

/** 消息条数 → 轮数（一问一答算一轮）。列表里说"轮"比说"条"更符合直觉。 */
export function turnsToRounds(turns: number): string {
  const rounds = Math.max(1, Math.ceil((turns || 0) / 2))
  return `${rounds} 轮`
}
