import { Fragment, type ReactNode } from 'react'

/**
 * 极简 Markdown 渲染。
 *
 * 只处理对话里**真实会出现**的三种结构：空行分段、管道表格、`**加粗**`。
 * 刻意不引 markdown 库 —— 一个客服答复用不到完整的 Markdown 语法，
 * 而少一份依赖就少一处构建与断网风险。
 *
 * 安全上走的是"拆成 React 元素"而不是 `dangerouslySetInnerHTML`：
 * 只产生文本节点、<strong>、<td> 和 <mark>，答复里就算混进 HTML 也会被当字面量显示。
 *
 * 另外负责**查找高亮**：`query` 非空时把命中的片段包成 <mark>。
 * 高亮做在渲染这一层而不是"改文本再渲染"，是因为改文本会破坏 Markdown 结构
 * （搜 `*` 就能把加粗记号拆坏），而包成元素不会。
 */

/** 把 `query` 在 `text` 里出现的每一处包成 <mark>。 */
export function highlight(text: string, query: string): ReactNode {
  const needle = (query || '').trim().toLowerCase()
  // 空查询直接返回原文；不挡这一下，下面的循环会在 needle 为空时原地打转
  if (!needle) return text

  const haystack = text.toLowerCase()
  const parts: ReactNode[] = []
  let cursor = 0
  for (;;) {
    const at = haystack.indexOf(needle, cursor)
    if (at === -1) break
    if (at > cursor) parts.push(text.slice(cursor, at))
    parts.push(
      <mark key={`${at}`} className="hit">
        {text.slice(at, at + needle.length)}
      </mark>,
    )
    cursor = at + needle.length
  }
  if (parts.length === 0) return text
  parts.push(text.slice(cursor))
  return parts
}

/** 把 `**加粗**` 拆成元素（顺带做查找高亮）。 */
function renderInline(text: string, query: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**') && part.length > 4 ? (
      <strong key={i}>{highlight(part.slice(2, -2), query)}</strong>
    ) : (
      <Fragment key={i}>{highlight(part, query)}</Fragment>
    ),
  )
}

/** 管道表格。后端已经把 `|---|---|` 分隔行去掉了，这里只切单元格。 */
function renderTable(lines: string[], query: string): ReactNode {
  const rows = lines
    .filter((l) => l.includes('|'))
    .map((l) =>
      l
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((c) => c.trim()),
    )
  if (rows.length === 0) return null

  const [head, ...body] = rows
  return (
    <div className="rt-table-wrap">
      <table className="rt-table">
        <thead>
          <tr>
            {head.map((cell, i) => (
              <th key={i}>{renderInline(cell, query)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{renderInline(cell, query)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function renderBlock(block: string, key: number, query: string): ReactNode {
  const lines = block.split('\n')
  // 与后端 _looks_like_table 同一判据：按"含 | 的行数"而不是行首
  const pipeLines = lines.filter((l) => l.includes('|'))
  if (pipeLines.length >= 2) {
    return <Fragment key={key}>{renderTable(lines, query)}</Fragment>
  }
  return (
    <p className="rt-p" key={key}>
      {renderInline(block, query)}
    </p>
  )
}

export function RichText({ text, query = '' }: { text: string; query?: string }) {
  const blocks = text.split(/\n{2,}/).filter((b) => b.trim())
  return <>{blocks.map((b, i) => renderBlock(b.trim(), i, query))}</>
}
