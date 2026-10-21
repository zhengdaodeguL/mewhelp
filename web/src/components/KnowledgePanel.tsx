import { useMemo, useState } from 'react'
import type { GraphSection, KnowledgeGraph } from '../api/client'
import { IconClose, IconDoc, IconRefresh } from './icons'

/**
 * 知识节点图谱 —— 把整个知识库折叠成一棵「文档 → 章节 → 小节」的树。
 *
 * 为什么是树而不是力导向图：这批知识是**规则书**，层级是它本来的形状
 * （《退款政策》→ 生鲜类 → 24 小时条件）。力导向图会把这个形状揉成一团
 * 靠弹簧力道摆布的点线，用户每次打开布局还都不一样 —— 那是给"关系本身
 * 是内容"的数据（引文网络、社交图）用的。这里关系只有一条：包含。
 * 树把它原样交出来，还免费送了"知道自己在哪、怎么回去"。
 *
 * 树是后端从切分面包屑折叠好的，前端只负责展开/收起和筛选 ——
 * 结构的口径只有一份，界面和检索就不会对"生鲜类属于退款政策"各说各话。
 *
 * 打开即展开：默认全部收起的话，用户面对的是六个文件名，还要再点七次
 * 才能看到第一条规则 —— "折叠"的代价不该加在第一次来的用户身上。
 */

interface Props {
  graph: KnowledgeGraph
  loading: boolean
  error: string
  onRefresh: () => void
}

/** 一条小节/文档下的知识片段。 */
function Leaf({ label, text, depth }: { label: string; text: string; depth: number }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="graph-leaf" style={{ '--depth': depth } as React.CSSProperties}>
      <button
        type="button"
        className="graph-leaf-label"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {label}
      </button>
      {open && text && <p className="graph-leaf-text">{text}</p>}
    </li>
  )
}

/** 这一枝下一共有多少条片段（含所有子章节）。 */
function countLeafs(section: GraphSection): number {
  return section.leafs.length + section.sections.reduce((sum, s) => sum + countLeafs(s), 0)
}

function SectionNode({ section, depth }: { section: GraphSection; depth: number }) {
  const [open, setOpen] = useState(true)
  const hasContent = section.sections.length > 0 || section.leafs.length > 0
  if (!hasContent) return null
  // 计数给的是**整枝**的条数，不是直接挂在这一层的条数。文档节点底下
  // 通常只挂一条正文（H1 那段），直接计数的话「退款政策」显示 1，
  // 而它下面明明有十几条规则 —— 这个数字看起来像"文档只有一条知识"。
  const total = countLeafs(section)
  return (
    <li className="graph-node" style={{ '--depth': depth } as React.CSSProperties}>
      <button
        type="button"
        className="graph-node-label"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="graph-caret" aria-hidden="true">
          {open ? '−' : '+'}
        </span>
        <span className="graph-node-title">{section.title}</span>
        <span className="graph-node-count" title={`这一枝共 ${total} 条知识片段`}>
          {total}
        </span>
      </button>
      {open && (
        <ul className="graph-children" role="group">
          {/* 片段排在子章节前面：这是 Markdown 的原始顺序 —— 标题下先有
              正文，再有子标题。反过来读的话，文档开头那段导语会掉到
              整棵子树的末尾，用户按"从上往下读"就会先读细节再读总述。 */}
          {section.leafs.map((leaf) => (
            <Leaf key={leaf.id} label={leaf.label} text={leaf.text} depth={depth + 1} />
          ))}
          {section.sections.map((s) => (
            <SectionNode key={s.id} section={s} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  )
}

export function KnowledgePanel({ graph, loading, error, onRefresh }: Props) {
  const [query, setQuery] = useState('')

  // 筛选只做"标题或片段文字里包含这个词"的直接匹配。图谱是用来浏览的，
  // 找一条具体规则有左侧的对话框 —— 那里背后是双路召回 + 精排，
  // 这里再造一遍检索就是第二套口径了。
  //
  // 命中之后**保留章节标题**：一条规则脱离「生鲜类」这个标题就读不懂了
  // （"24 小时内提交"是哪种商品？），所以筛的是片段，留下的是它所在的
  // 整条祖先链 —— 只把没命中的片段剪掉。
  const filtered = useMemo(() => {
    const q = query.trim()
    if (!q) return graph
    const hit = (s: GraphSection, inside: boolean): GraphSection | null => {
      // 章节标题命中时，它底下的片段全部保留 ——「生鲜类」这个词命中的
      // 是章节本身，而章节下的速查表（"速查表：商品状态、申请时间窗…"）
      // 字面上并不含"生鲜"。只看片段的话，搜章节名反而把该章节的
      // 表格筛没了。`inside` 把"祖先已命中"一路传下去。
      const self = inside || s.title.includes(q)
      const leafs = self ? s.leafs : s.leafs.filter((l) => l.label.includes(q) || l.text.includes(q))
      const sections = s.sections
        .map((s2) => hit(s2, self))
        .filter((s2): s2 is GraphSection => s2 !== null)
      if (!self && leafs.length === 0 && sections.length === 0) return null
      return { ...s, leafs, sections }
    }
    return {
      count: graph.count,
      docs: graph.docs
        .map((d) => hit(d, false))
        .filter((d): d is KnowledgeGraph['docs'][number] => d !== null),
    }
  }, [graph, query])

  return (
    <div className="graph">
      <div className="graph-bar">
        <input
          className="graph-input"
          value={query}
          placeholder="在图谱里筛一个词，比如「生鲜」"
          aria-label="筛选知识图谱"
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            type="button"
            className="icon-btn"
            aria-label="清空筛选"
            onClick={() => setQuery('')}
          >
            <IconClose width={12} height={12} />
          </button>
        )}
        <button
          type="button"
          className="icon-btn"
          title="刷新图谱"
          aria-label="刷新知识图谱"
          onClick={onRefresh}
        >
          <IconRefresh width={13} height={13} />
        </button>
      </div>

      {error && <p className="memory-error">{error}</p>}

      {loading && graph.count === 0 ? (
        <div className="panel-empty">
          <p>正在铺开知识图谱…</p>
        </div>
      ) : graph.count === 0 ? (
        <div className="panel-empty">
          <p>
            知识图谱暂时没有内容。
            <br />
            知识库文档加载进来之后，会按章节结构陈列在这里。
          </p>
        </div>
      ) : filtered.docs.length === 0 ? (
        <div className="panel-empty">
          <p>图谱里没有提到「{query.trim()}」。</p>
        </div>
      ) : (
        <ul className="graph-roots">
          {filtered.docs.map((doc) => (
            <SectionNode key={doc.id} section={doc} depth={0} />
          ))}
        </ul>
      )}

      {graph.count > 0 && (
        <p className="graph-meta">
          <IconDoc width={12} height={12} />
          {query.trim()
            ? `筛出 ${filtered.docs.reduce((sum, d) => sum + countLeafs(d), 0)} 条 · ${filtered.docs.length} 篇文档`
            : `${graph.count} 条知识片段 · ${graph.docs.length} 篇文档`}
        </p>
      )}
    </div>
  )
}
