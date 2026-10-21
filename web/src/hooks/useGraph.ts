import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchKnowledgeGraph, type KnowledgeGraph } from '../api/client'

const EMPTY: KnowledgeGraph = { count: 0, docs: [] }

/**
 * 知识图谱的数据源。
 *
 * 拉取时机是**面板第一次被打开**，而不是页面加载时：图谱是四个标签里
 * 唯一"打开才知道有"的内容，为一棵多数会话里根本不会看的树抢启动
 * 带宽，不划算。拉过一次之后除非用户手动刷新，不再重拉 —— 知识库
 * 变更频率是"运营改文档"那个量级，不是"聊一句变一次"。
 *
 * 挂载后如果数据是空的（后端没配知识库），也照常展示空状态：
 * 读接口的纪律和别处一致 —— 拿不到就返回空，不抛。
 */
export function useGraph(open: boolean) {
  const [graph, setGraph] = useState<KnowledgeGraph>(EMPTY)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fetched = useRef(false)
  const seq = useRef(0)

  const load = useCallback(async () => {
    const mine = ++seq.current
    setLoading(true)
    setError('')
    const next = await fetchKnowledgeGraph()
    // 竞态防护：用户点了刷新又立刻切走标签，晚到的那次响应不该覆盖
    if (mine === seq.current) {
      setGraph({ count: next.count, docs: next.docs })
      setLoading(false)
      // 空结果**不是错误**。没配知识库的部署、或者索引还没建完，都是
      // 正常状态；把它写成红色报错等于告诉用户"坏了，再试一次"，
      // 而重试永远不会改变结果。面板自己会渲染空状态文案，那已经够了。
      // error 只留给"请求本身失败"（网络断、5xx）—— 那种情况下重试有意义。
      if (next.failed) setError('图谱没能加载出来，点右上角刷新再试一次')
    }
  }, [])

  useEffect(() => {
    if (!open || fetched.current) return
    fetched.current = true
    void load()
  }, [open, load])

  return { graph, loading, error, refresh: load }
}
