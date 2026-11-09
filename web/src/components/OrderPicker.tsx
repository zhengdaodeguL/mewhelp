import { useEffect, useState } from 'react'
import { fetchOrders, type OrderBrief } from '../api/client'
import { statusTone } from '../workspace'
import { IconBox } from './icons'

/**
 * 订单快捷选择。
 *
 * 存在的理由很直接：**订单号是用户最记不住的东西**。让用户为了查个物流
 * 去翻订单号、再手打一遍 ORD-1007，是把系统的记忆成本转嫁给用户。
 * 这里点一下就填进输入框。
 *
 * 刻意**只填不发送**：点了哪一单之后用户往往还想补一句（"这单能退吗"），
 * 直接发出去等于替他做了决定。
 */

interface Props {
  disabled: boolean
  onPick: (order: OrderBrief) => void
}

export function OrderPicker({ disabled, onPick }: Props) {
  const [orders, setOrders] = useState<OrderBrief[]>([])

  useEffect(() => {
    const controller = new AbortController()
    void fetchOrders(controller.signal).then((r) => {
      setOrders(r.orders)
    })
    return () => controller.abort()
  }, [])

  // 拿不到订单就整条收起 —— 一个空的选择器只会占地方
  if (orders.length === 0) return null

  return (
    <div className="order-picker">
      <span className="order-picker-label">
        <IconBox width={13} height={13} />
        我的订单
      </span>
      <div className="order-picker-items">
        {orders.map((o) => (
          <button
            key={o.order_id}
            type="button"
            className="order-chip"
            disabled={disabled}
            title={`${o.title} · ${o.item_count} 件 · ${o.status_label}`}
            onClick={() => onPick(o)}
          >
            <span className="order-chip-id">{o.order_id}</span>
            <span className="order-chip-title">{o.title}</span>
            <span className={`pill pill--${statusTone(o.status_label)}`}>{o.status_label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
