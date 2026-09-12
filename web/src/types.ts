/**
 * 前端只保留「用户能看见的东西」所需的类型。
 *
 * 后端 SSE 会推 meta / evidence / tool / done 等事件，其中意图、节点路径、
 * 证据块、工具名、置信度都是**开发者视角的内部信息**，不应出现在用户界面上。
 * 这里刻意不给它们建模 —— 类型层面拿不到，界面就不会不小心渲染出去。
 */

export type Role = 'user' | 'assistant'

/** 系统提示的气泡：转人工、出错等，视觉上与普通对话区分开。 */
export type MessageKind = 'normal' | 'notice'

export interface ChatMessage {
  id: string
  role: Role
  kind: MessageKind
  content: string
  /** 正在逐字接收中（用于显示光标动画） */
  streaming?: boolean
}

/** 一次问答结束时，用户**需要知道**的结果。 */
export interface TurnOutcome {
  /** 已转人工 —— 这是真实业务结果，必须告诉用户 */
  handoff: boolean
}
