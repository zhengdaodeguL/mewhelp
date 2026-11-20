import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'

// 编译实际组件与纯状态逻辑，不增添浏览器测试依赖。
const root = fileURLToPath(new URL('../', import.meta.url))
await mkdir(path.join(root, 'node_modules/.tmp'), { recursive: true })
const temp = await mkdtemp(path.join(root, 'node_modules/.tmp/issue15-'))
const compiled = new Set()
async function compile(relative) {
  const target = path.join(temp, relative.replace(/\.tsx?$/, '.mjs'))
  if (compiled.has(relative)) return target
  compiled.add(relative)
  const source = await readFile(path.join(root, 'src', relative), 'utf8')
  let code = ts.transpileModule(source, {
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText
  for (const match of code.matchAll(/from ['"](\.[^'"]+)['"]/g)) {
    const base = path.join(path.dirname(relative), match[1])
    let dependency
    for (const ext of ['.ts', '.tsx']) {
      try { await readFile(path.join(root, 'src', base + ext)); dependency = base + ext; break } catch { /* try next */ }
    }
    assert.ok(dependency, `missing import: ${base}`)
    await compile(dependency)
    code = code.replaceAll(`'${match[1]}'`, `'${match[1]}.mjs'`).replaceAll(`"${match[1]}"`, `"${match[1]}.mjs"`)
  }
  await mkdir(path.dirname(target), { recursive: true })
  await writeFile(target, code)
  return target
}

test('会话标题和常显的更多菜单分属两个点击区域，重命名/删除藏在关闭的菜单内', async () => {
  const { HistoryRail } = await import(pathToFileURL(await compile('components/HistoryRail.tsx')).href)
  const html = renderToStaticMarkup(createElement(HistoryRail, {
    sessions: [{ sessionId: 's', title: '测试会话', turns: 2, preview: '部分回答', updatedAt: new Date().toISOString() }],
    activeId: 's', loading: false, onSelect() {}, onNew() {}, onRename() {}, onDelete() {},
  }))
  assert.match(html, /<button[^>]*class="session-select"/)
  assert.match(html, /<\/button>\s*<details class="session-menu"/)
  assert.match(html, /<summary[^>]*aria-label=".*更多操作/)
  assert.ok(!/<details[^>]*\bopen(?:=|>)/.test(html), '操作不能悬停即出现')
  const menu = html.slice(html.indexOf('<details'), html.indexOf('</details>'))
  assert.match(menu, /重命名/)
  assert.match(menu, /删除/)
  const css = await readFile(path.join(root, 'src/styles.css'), 'utf8')
  assert.ok(!/\.session:hover\s+\.session-actions/.test(css))
})

for (const content of ['', '已经收到的部分回答']) {
  test(`停止生成保留问题与回答并标记未完成：${content || '首字之前'}`, async () => {
    const { stopMessages } = await import(pathToFileURL(await compile('chat-state.ts')).href)
    const messages = [
      { id: 'u', role: 'user', kind: 'normal', content: '饼干有哪些规格？' },
      { id: 'a', role: 'assistant', kind: 'normal', content, streaming: true },
    ]
    const stopped = stopMessages(messages)
    assert.equal(stopped.length, 2)
    assert.deepEqual(stopped[0], messages[0])
    assert.equal(stopped[1].content, content)
    assert.equal(stopped[1].streaming, false)
    assert.equal(stopped[1].status, 'stopped')
    assert.deepEqual(stopMessages(stopped), stopped, '重复停止必须幂等')
  })
}

test('首字前主动 Abort 不会被网络错误覆盖', async () => {
  const { streamChat } = await import(pathToFileURL(await compile('api/client.ts')).href)
  const original = globalThis.fetch
  const controller = new AbortController()
  controller.abort()
  globalThis.fetch = async () => { throw new DOMException('stopped', 'AbortError') }
  const errors = []
  try {
    await streamChat('饼干规格', { sessionId: 's', signal: controller.signal }, {
      onError: (message) => errors.push(message), onToken() {}, onCards() {}, onMemory() {}, onTurnId() {}, onDone() {},
    })
    assert.deepEqual(errors, [])
  } finally { globalThis.fetch = original }
})

test('停止确认使用原会话和问答 ID，并带上已收到的正文', async () => {
  const { stopChat } = await import(pathToFileURL(await compile('api/client.ts')).href)
  const original = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body), credentials: options.credentials })
    return Response.json({ ok: true, status: 'stopped' })
  }
  try {
    assert.equal(await stopChat({ sessionId: 's', requestId: 'r', message: '饼干规格', answer: '第一条' }), true)
    assert.deepEqual(calls, [{
      url: '/api/chat/stop', credentials: 'same-origin',
      body: { session_id: 's', request_id: 'r', message: '饼干规格', answer: '第一条' },
    }])
  } finally { globalThis.fetch = original }
})

test('done 之后继续同步核验结果，answered 失败的更正与转接不会丢失', async () => {
  const { waitForTurn } = await import(pathToFileURL(await compile('api/client.ts')).href)
  const original = globalThis.fetch
  let calls = 0
  globalThis.fetch = async () => Response.json({
    ok: true, title: '规格', messages: [{
      role: 'assistant', request_id: 'r', content: calls++ === 0 ? '原答复' : '答复未通过核验，请人工核实。',
      status: calls === 1 ? 'verifying' : 'complete', handoff: calls > 1,
    }],
  })
  const snapshots = []
  try {
    assert.equal(await waitForTurn('s', 'r', new AbortController().signal, (_, reply) => snapshots.push(reply)), true)
    assert.deepEqual(snapshots.map((m) => m.status), ['verifying', 'complete'])
    assert.equal(snapshots[1].handoff, true)
    assert.match(snapshots[1].content, /未通过核验/)
    assert.ok(snapshots.every((m) => m.requestId === 'r'))
  } finally { globalThis.fetch = original }
})
