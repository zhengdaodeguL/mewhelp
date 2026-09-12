/**
 * 快捷问题。用户第一次进来往往不知道该问什么，给几个真实能力范围内的入口，
 * 比一个光秃秃的输入框有用得多。
 */
const QUICK_QUESTIONS = [
  '我的订单到哪了？',
  '生鲜类可以退款吗？',
  '尺码怎么选？',
  '怎么开发票？',
  '转人工',
]

interface Props {
  disabled: boolean
  onPick: (question: string) => void
}

export function QuickReplies({ disabled, onPick }: Props) {
  return (
    <div className="quick-replies">
      {QUICK_QUESTIONS.map((q) => (
        <button
          key={q}
          type="button"
          className="chip"
          disabled={disabled}
          onClick={() => onPick(q)}
        >
          {q}
        </button>
      ))}
    </div>
  )
}
