import { IconBox, IconDoc, IconSpark, IconTruck } from './icons'

/**
 * 快捷问题。
 *
 * 用户第一次进来往往不知道该问什么，给几个**真实能力范围内**的入口，
 * 比一个光秃秃的输入框有用得多。每个都配个图标，是为了让这一排在
 * 一堆气泡中间能被一眼扫到 —— 纯文字条很容易被当成装饰忽略掉。
 */
const QUICK_QUESTIONS: ReadonlyArray<{ text: string; Icon: typeof IconBox }> = [
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

export function QuickReplies({ disabled, onPick }: Props) {
  return (
    <div className="quick-replies">
      <span className="quick-label">
        <IconSpark width={13} height={13} />
        试试
      </span>
      {QUICK_QUESTIONS.map(({ text, Icon }) => (
        <button
          key={text}
          type="button"
          className="chip"
          disabled={disabled}
          onClick={() => onPick(text)}
        >
          <Icon width={13} height={13} />
          {text}
        </button>
      ))}
    </div>
  )
}
