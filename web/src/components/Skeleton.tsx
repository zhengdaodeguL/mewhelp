/**
 * 骨架屏。
 *
 * 加载态有两种做法：转圈，或者画出"将要出现的东西的轮廓"。这里选后者，
 * 因为**位置不会跳** —— 数据回来时，用户的眼睛已经在正确的位置上了；
 * 而转圈之后内容一口气铺开，视线要重新找一遍。
 *
 * 形状由调用方决定（几行、多宽），这个文件只负责"一个在发光的方块"。
 */

interface LineProps {
  /** 宽度的预设：标题 / 正文 / 短行 / 元信息 */
  variant?: 'title' | 'body' | 'short' | 'meta'
}

export function SkeletonLine({ variant = 'body' }: LineProps) {
  const cls =
    variant === 'title'
      ? ' skeleton-line--title'
      : variant === 'short'
        ? ' skeleton-line--short'
        : variant === 'meta'
          ? ' skeleton-line--meta'
          : ''
  return <div className={`skeleton skeleton-line${cls}`} />
}

/** 历史栏里的一条会话。 */
export function SessionSkeleton() {
  return (
    <div className="skeleton-item">
      <SkeletonLine variant="title" />
      <SkeletonLine variant="meta" />
    </div>
  )
}

/** 记忆面板里的一张卡片。 */
export function MemorySkeleton() {
  return (
    <div className="skeleton-card">
      <SkeletonLine variant="body" />
      <SkeletonLine variant="meta" />
    </div>
  )
}
