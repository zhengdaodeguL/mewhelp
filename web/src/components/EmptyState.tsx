import { IconBox, IconDoc, IconSpark, IconTruck } from './icons'

/**
 * 首屏空态。
 *
 * 上一版这里是**一条开场白气泡**："你好，我是 MewHelp 客服助手 👋 …"。
 * 它有两个问题：一是它长得和真答复一模一样，用户会以为已经聊过一轮了；
 * 二是它占着"最新一条"的位置，等真消息进来就被顶上去，看起来像一句话被删了。
 *
 * 现在换成一块**不属于对话**的区域：图标块 + 一句话 + 建议问题。
 * 它只在一条用户消息都没有的时候出现，发出去第一句就整块消失 ——
 * 空态和对话是两种状态，不该共用同一种元素。
 *
 * 建议问题给的是**真实能力范围内**的入口。用户第一次进来往往不知道
 * 能问什么，给四个能直接点的例子，比一句"有什么可以帮你"有用得多。
 */
const SUGGESTIONS: ReadonlyArray<{ text: string; Icon: typeof IconBox }> = [
  { text: '我的订单到哪了？', Icon: IconTruck },
  { text: '生鲜类可以退款吗？', Icon: IconDoc },
  { text: '尺码怎么选？', Icon: IconBox },
  { text: '怎么开发票？', Icon: IconDoc },
  { text: '转人工', Icon: IconSpark },
]

interface Props {
  disabled: boolean
  onPick: (question: string) => void
}

export function EmptyState({ disabled, onPick }: Props) {
  return (
    <div className="empty-state">
      <div className="empty-mark" aria-hidden="true">
        <IconSpark width={20} height={20} />
      </div>
      <p className="empty-kicker">新的对话</p>
      <h2 className="empty-title">从一件事开始</h2>
      <p className="empty-sub">查订单、看物流、问退换货，或直接描述你要处理的问题。</p>
      <div className="empty-chips">
        {SUGGESTIONS.map(({ text, Icon }, i) => (
          <button
            key={text}
            type="button"
            className="empty-chip"
            // 依次浮现：一起出现像一排按钮，依次出现才像"给你的几个建议"。
            // 错开的量（55ms）比卡片入场（45ms）略大，是因为这里的元素更宽、
            // 排成两行，节奏太密会读成一次闪烁。
            style={{ '--i': i } as React.CSSProperties}
            disabled={disabled}
            onClick={() => onPick(text)}
          >
            <Icon width={14} height={14} />
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}
