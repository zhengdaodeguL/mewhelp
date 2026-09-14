import type { SVGProps } from 'react'

/**
 * 内联 SVG 图标集。
 *
 * 刻意不引图标库：一是这套界面只用得到十来个图标，二是产物里少一份依赖，
 * 断网构建也不受影响。统一 16px、stroke 线稿风格，跟卡片文字同色系。
 */

type IconProps = SVGProps<SVGSVGElement>

function Base({ children, ...rest }: IconProps) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

/** 订单：包裹盒 */
export function IconBox(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M21 8 12 3 3 8v8l9 5 9-5V8Z" />
      <path d="m3 8 9 5 9-5" />
      <path d="M12 13v8" />
    </Base>
  )
}

/** 物流：货车 */
export function IconTruck(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M3 6h11v9H3z" />
      <path d="M14 9h4l3 3v3h-7z" />
      <circle cx="7" cy="18" r="1.8" />
      <circle cx="17" cy="18" r="1.8" />
    </Base>
  )
}

/** 依据：文档 */
export function IconDoc(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h4" />
    </Base>
  )
}

/** 处理结果：对勾 */
export function IconCheck(props: IconProps) {
  return (
    <Base {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="m8.5 12.2 2.4 2.4 4.6-5" />
    </Base>
  )
}

export function IconCopy(props: IconProps) {
  return (
    <Base {...props}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3" />
    </Base>
  )
}

export function IconRefresh(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M20 11a8 8 0 1 0-2.3 6.3" />
      <path d="M20 5v6h-6" />
    </Base>
  )
}

export function IconSun(props: IconProps) {
  return (
    <Base {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" />
    </Base>
  )
}

export function IconMoon(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" />
    </Base>
  )
}

export function IconMonitor(props: IconProps) {
  return (
    <Base {...props}>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8 20h8M12 16v4" />
    </Base>
  )
}

export function IconPanel(props: IconProps) {
  return (
    <Base {...props}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M14 4v16" />
    </Base>
  )
}

export function IconArrowDown(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M12 5v14" />
      <path d="m6 13 6 6 6-6" />
    </Base>
  )
}

export function IconSpark(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M12 3v6M12 15v6M3 12h6M15 12h6" />
      <path d="m6.5 6.5 3 3M14.5 14.5l3 3M17.5 6.5l-3 3M9.5 14.5l-3 3" />
    </Base>
  )
}

export function IconPlus(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M12 5v14M5 12h14" />
    </Base>
  )
}

export function IconClose(props: IconProps) {
  return (
    <Base {...props}>
      <path d="m6 6 12 12M18 6 6 18" />
    </Base>
  )
}

export function IconStop(props: IconProps) {
  return (
    <Base {...props}>
      <rect x="7" y="7" width="10" height="10" rx="2" />
    </Base>
  )
}

export function IconSend(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M4 11.5 20 4l-7.5 16-2-6.5Z" />
      <path d="m10.5 13.5 4-4" />
    </Base>
  )
}

/** 历史：时钟 + 回溯箭头 */
export function IconHistory(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
      <path d="M12 8v4.5l3 1.7" />
    </Base>
  )
}

export function IconSearch(props: IconProps) {
  return (
    <Base {...props}>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </Base>
  )
}

/** 导出：向下的箭头落进托盘 */
export function IconDownload(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M12 3v11" />
      <path d="m7.5 10 4.5 4.5 4.5-4.5" />
      <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </Base>
  )
}

export function IconStar({ filled = false, ...rest }: IconProps & { filled?: boolean }) {
  return (
    <Base {...rest} fill={filled ? 'currentColor' : 'none'}>
      <path d="m12 3.6 2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8L3.5 9.8l5.9-.9Z" />
    </Base>
  )
}

export function IconThumbUp(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M7 21V10l4-7a2 2 0 0 1 2 2v4h5.5a2 2 0 0 1 2 2.4l-1.3 6A2 2 0 0 1 17.2 21Z" />
      <path d="M7 10H4v11h3" />
    </Base>
  )
}

export function IconThumbDown(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M17 3v11l-4 7a2 2 0 0 1-2-2v-4H5.5a2 2 0 0 1-2-2.4l1.3-6A2 2 0 0 1 6.8 3Z" />
      <path d="M17 14h3V3h-3" />
    </Base>
  )
}

export function IconTrash(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M4 7h16" />
      <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
      <path d="M6 7v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7" />
      <path d="M10 11v6M14 11v6" />
    </Base>
  )
}

export function IconPencil(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17v3Z" />
      <path d="m14.5 6.5 3 3" />
    </Base>
  )
}

/** 记忆：书签 */
export function IconPin({ filled = false, ...rest }: IconProps & { filled?: boolean }) {
  return (
    <Base {...rest} fill={filled ? 'currentColor' : 'none'}>
      <path d="M6 4h12a1 1 0 0 1 1 1v15l-7-4.2L5 20V5a1 1 0 0 1 1-1Z" />
    </Base>
  )
}

/** 长期记忆：大脑/卡片式档案 */
export function IconMemory(props: IconProps) {
  return (
    <Base {...props}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M8 9h8M8 13h5" />
    </Base>
  )
}

export function IconArrowUp(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M12 19V5" />
      <path d="m6 11 6-6 6 6" />
    </Base>
  )
}

export function IconArrowLeft(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M19 12H5" />
      <path d="m11 6-6 6 6 6" />
    </Base>
  )
}
