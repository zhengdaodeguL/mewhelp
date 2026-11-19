import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'
import { cardsFromTools, mergeCards, startWorkspaceTurn } from '../src/workspace.ts'

// 使用实际组件生成 HTML；临时产物仍从本项目解析 React，无需新增测试依赖。
const root = fileURLToPath(new URL('../', import.meta.url))
await mkdir(path.join(root, 'node_modules/.tmp'), { recursive: true })
const temp = await mkdtemp(path.join(root, 'node_modules/.tmp/issue4-'))
const source = await readFile(path.join(root, 'src/components/RichText.tsx'), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext },
}).outputText
await writeFile(path.join(temp, 'RichText.mjs'), compiled)
const { RichText } = await import(pathToFileURL(path.join(temp, 'RichText.mjs')).href)
after(() => rm(temp, { recursive: true, force: true }))

for (const separator of ['| --- | --- |', '| :--- | ---: |', '--- | ---', '| - | - |']) {
  test(`表格分隔行不进入数据：${separator}`, () => {
    const text = `| 品类/口味 | 规格 |\n${separator}\n| 牛肉馅酥脆饼干 | 500g |\n| 桃酥饼干 | 500g（散装称重） |`
    const html = renderToStaticMarkup(createElement(RichText, { text }))
    assert.equal((html.match(/<th /g) ?? html.match(/<th>/g) ?? []).length, 2)
    assert.equal((html.match(/<tr>/g) ?? []).length, 3)
    assert.ok(!html.includes('<td>---</td>'))
    assert.ok(html.includes('桃酥饼干'))
  })
}

test('普通数据中的破折号仍保留', () => {
  const html = renderToStaticMarkup(createElement(RichText, {
    text: '| 商品 | 备注 |\n| --- | --- |\n| 饼干 | --- |',
  }))
  assert.ok(html.includes('<td>---</td>'))
})

test('新一轮的依据替换上一轮商品资料并可从历史恢复', async () => {
  const workspace = await import('../src/workspace.ts')
  const rows = [
    { role: 'assistant', evidence: [{ title: '羊角蜜', snippet: '冷藏。' }], tools: [] },
    { role: 'assistant', evidence: [{ title: '苹果', snippet: '冷藏。' }], tools: [] },
  ]
  assert.deepEqual(workspace.cardsFromHistory(rows).map((card) => card.title), ['苹果'])
  rows.push({ role: 'assistant', evidence: [], tools: [] })
  assert.deepEqual(workspace.cardsFromHistory(rows), [])
})

test('切换会话和刷新恢复各自资料，旧历史也可加载', async () => {
  const { cardsFromHistory } = await import('../src/workspace.ts')
  const rows = [{ role: 'assistant', evidence: [{ title: '苹果', snippet: '冷藏。' }], tools: [] }]
  const first = cardsFromHistory(rows)
  assert.deepEqual(cardsFromHistory([{ role: 'assistant', content: '旧答复' }]), [])
  assert.deepEqual(cardsFromHistory(JSON.parse(JSON.stringify(rows))), first)
  const order = cardsFromTools({ ok: true, kind: 'order', result: {
    order_id: 'ORD-1003', status_label: '已发货', items: [],
  } })
  assert.deepEqual(startWorkspaceTurn([...first, ...order]), order)
})

test('旧订单卡遇到新的状态快照会更新', () => {
  const event = (statusLabel) => ({ ok: true, kind: 'order', result: {
    order_id: 'ORD-1010', status_label: statusLabel, items: [],
  } })
  const cards = mergeCards(cardsFromTools(event('已签收')), cardsFromTools(event('已退款')))
  assert.equal(cards.length, 1)
  assert.equal(cards[0].statusLabel, '已退款')
})
