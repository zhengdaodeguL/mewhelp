/**
 * 前端只保留「用户能看见的东西」所需的类型。
 *
 * 后端 SSE 会推 meta / evidence / memory / tool / done 等事件，其中意图、
 * 节点路径、证据块、工具名、置信度都是**开发者视角的内部信息**，不应出现在
 * 用户界面上。这里刻意不给它们建模 —— 类型层面拿不到，界面就不会不小心渲染出去。
 *
 * 唯一的例外是 `turnId`（后端字段名是 `request_id`）：它是不透明的句柄，
 * 界面拿它当"反馈"和"收藏"的键，但从不显示出来。刻意在这里改个名字，
 * 是为了让"这个值不是给人看的"在类型层面就能看出来 —— 一个叫 request_id
 * 的字段迟早会有人顺手渲染到界面上。
 */

export type Role = 'user' | 'assistant'

/** 系统提示的气泡：转人工、出错等，视觉上与普通对话区分开。 */
export type MessageKind = 'normal' | 'notice'

/** 用户对一条答复的评价。没评过是 undefined。 */
export type Rating = 'up' | 'down'

export interface ChatMessage {
  id: string
  role: Role
  kind: MessageKind
  content: string
  /** 正在逐字接收中（用于显示光标动画） */
  streaming?: boolean
  /** 本轮问答的不透明句柄：反馈与收藏的键。没有它就不能评、不能收藏。 */
  turnId?: string
  /** 我点过的赞/踩 */
  rating?: Rating
}

/** 一次问答结束时，用户**需要知道**的结果。 */
export interface TurnOutcome {
  /** 已转人工 —— 这是真实业务结果，必须告诉用户 */
  handoff: boolean
}

/**
 * 一条长期记忆。
 *
 * `kindLabel` 由服务端算好一起下发，前端不另写一份"order → 关注的订单"
 * 的映射表 —— 那张表是产品文案，两处各写一份迟早会出现
 * "日志写'关注的订单'、界面显示'订单'"的不一致，而且不会有任何测试失败。
 */
export interface MemoryItem {
  id: number
  kind: string
  kindLabel: string
  text: string
  /** auto = 系统从对话里抽的；manual = 用户自己在面板里写的 */
  source: 'auto' | 'manual' | string
  pinned: boolean
  createdAt: string
  updatedAt: string
}

/** 历史列表里的一行。 */
export interface SessionBrief {
  sessionId: string
  title: string
  /** 消息条数（一问一答算两条） */
  turns: number
  preview: string
  updatedAt: string
}

/** 恢复一次历史对话时拿到的消息。 */
export interface HistoryMessage {
  role: Role
  content: string
  handoff: boolean
  requestId: string
}
