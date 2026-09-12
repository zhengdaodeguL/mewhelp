/**
 * SSE 解析 —— 纯函数，不碰 DOM、不碰 fetch。
 *
 * 单独放一个文件有两个理由：
 *   1. `client.ts` 只管"怎么发请求"，解析规则只有这一份，不会两处写法漂移；
 *   2. 它是**跨语言契约**的落点：后端推的字节流必须能被这里逐帧还原。
 *      tests/test_api.py 会用 node 直接跑这个文件，拿后端真实响应体
 *      （按 7 字节切碎，模拟 TCP 分包 + 多字节中文被截断）喂进来核对。
 *
 * 因此：这里只放纯逻辑，别引入任何浏览器 API，否则 node 侧跑不起来。
 */

export interface SseEvent {
  event: string
  /** data 是 JSON 时解析成对象，否则保留原始字符串 */
  data: unknown
}

/**
 * 统一换行。SSE 规范允许 \r\n、\r、\n 三种换行，先归一后面才好切帧。
 *
 * 结尾那个**孤立**的 \r 必须留着：它很可能只是 \r\n 的前半截，被 TCP 切在
 * 了两个 chunk 之间。此刻就把它当换行的话，下一块开头的 \n 会凑出一个假的
 * 空行 —— 于是「一个事件」被腰斩成两半：前半截（只有 event:）被丢掉，
 * 后半截（只有 data:）变成一条没有名字的 message 事件，token 就这么丢了。
 * 留着它，等下一块到了再一起归一。
 */
export function normalizeSse(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r(?!$)/g, '\n')
}

/** 解析单个帧（已被空行切开的文本块）。 */
export function parseEvent(frame: string): { event: string; data: string } | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const rawLine of frame.split('\n')) {
    if (!rawLine || rawLine.startsWith(':')) continue // 空行与注释行
    if (rawLine.startsWith('event:')) {
      event = rawLine.slice(6).trim()
    } else if (rawLine.startsWith('data:')) {
      // 规范：冒号后若有一个空格要去掉，其余原样保留
      dataLines.push(rawLine.slice(5).replace(/^ /, ''))
    }
  }

  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

/**
 * 从缓冲区里取出所有**完整**事件，返回剩余的不完整尾巴。
 *
 * 尾巴必须留着 —— 一个事件被拆到两个 TCP 包里是常态，
 * 丢掉尾巴的表现是"前端永远少最后一条消息"，且很难复现。
 */
export function drainEvents(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = []
  const frames = buffer.split('\n\n')
  const rest = frames.pop() ?? ''

  for (const frame of frames) {
    const parsed = parseEvent(frame)
    if (!parsed) continue
    let data: unknown = parsed.data
    try {
      data = JSON.parse(parsed.data)
    } catch {
      /* 不是 JSON 就按原样透传 */
    }
    events.push({ event: parsed.event, data })
  }

  return { events, rest }
}
