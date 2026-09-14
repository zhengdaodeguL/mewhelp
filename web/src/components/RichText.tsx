import { Fragment, type ReactNode } from 'react'

/**
 * 极简 Markdown 渲染。
 *
 * 只处理对话里**真实会出现**的三种结构：空行分段、管道表格、`**加粗**`。
 * 刻意不引 markdown 库 —— 一个客服答复用不到完整的 Markdown 语法，
 * 而少一份依赖就少一处构建与断网风险。
 *
 * 安全上走的是"拆成 React 元素"而不是 `dangerouslySetInnerHTML`：
 * 只产生文本节点和 <strong>/<td>，答复里就算混进 HTML 也只会被当字面量显示。
 */

/** 把 `**加粗**` 拆成元素。 */
function renderInline(text: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**') && part.length > 4 ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  )
}

/** 管道表格。后端已经把 `|---|---|` 分隔行去掉了，这里只切单元格。 */
function renderTable(lines: string[]): ReactNode {
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
              <th key={i}>{renderInline(cell)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{renderInline(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function renderBlock(block: string, key: number): ReactNode {
  const lines = block.split('\n')
  // 与后端 _looks_like_table 同一判据：按"含 | 的行数"而不是行首
  const pipeLines = lines.filter((l) => l.includes('|'))
  if (pipeLines.length >= 2) return <Fragment key={key}>{renderTable(lines)}</Fragment>
  return (
    <p className="rt-p" key={key}>
      {renderInline(block)}
    </p>
  )
}

export function RichText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/).filter((b) => b.trim())
  return <>{blocks.map((b, i) => renderBlock(b.trim(), i))}</>
}
