import assert from 'node:assert/strict';
import { test, mock, afterEach } from 'node:test';
import worker from './cloudflare/worker.js';

afterEach(() => mock.restoreAll());

test('forwards POST body, query and headers only to the fixed Instabox origin', async () => {
  mock.method(globalThis, 'fetch', async (request, options) => {
    assert.equal(request.url, 'https://8000-agq35mghm2gi5lgmzym264yvue.instabox.tech/api/chat?stream=1');
    assert.equal(request.method, 'POST');
    assert.equal(await request.text(), '{"message":"hello"}');
    assert.equal(request.headers.get('host'), null);
    assert.equal(request.headers.get('x-forwarded-host'), 'mewhelp.kdns.fr');
    assert.equal(request.headers.get('content-type'), 'application/json');
    assert.equal(options.redirect, 'manual');
    return new Response('ok');
  });
  const response = await worker.fetch(new Request('https://mewhelp.kdns.fr/api/chat?stream=1', {
    method: 'POST',
    headers: { 'content-type': 'application/json', host: 'untrusted.example' },
    body: '{"message":"hello"}',
  }));
  assert.equal(await response.text(), 'ok');
});

test('a double-slash path cannot turn the gateway into an open proxy', async () => {
  mock.method(globalThis, 'fetch', async (request) => {
    assert.equal(new URL(request.url).origin, 'https://8000-agq35mghm2gi5lgmzym264yvue.instabox.tech');
    assert.equal(new URL(request.url).pathname, '//other.example/private');
    return new Response('ok');
  });
  await worker.fetch(new Request('https://mewhelp.kdns.fr//other.example/private'));
});

test('returns SSE before the upstream stream closes', async () => {
  const frame = new TextEncoder().encode('event: token\ndata: hello\n\n');
  const body = new ReadableStream({ start(controller) { controller.enqueue(frame); } });
  mock.method(globalThis, 'fetch', async () => new Response(body, {
    headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' },
  }));
  const response = await worker.fetch(new Request('https://mewhelp.kdns.fr/api/chat'));
  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  assert.equal(response.headers.get('cache-control'), 'no-cache');
  const reader = response.body.getReader();
  const first = await reader.read();
  assert.equal(first.done, false);
  assert.deepEqual(first.value, frame);
  await reader.cancel();
});

test('keeps same-origin redirects on the public domain', async () => {
  mock.method(globalThis, 'fetch', async () => new Response(null, {
    status: 307,
    headers: { location: 'https://8000-agq35mghm2gi5lgmzym264yvue.instabox.tech/api/orders/?page=2' },
  }));
  const response = await worker.fetch(new Request('https://mewhelp.kdns.fr/api/orders'));
  assert.equal(response.status, 307);
  assert.equal(response.headers.get('location'), 'https://mewhelp.kdns.fr/api/orders/?page=2');
});

test('returns a non-cacheable response if the origin cannot be reached', async () => {
  mock.method(globalThis, 'fetch', async () => { throw new Error('network failure'); });
  const response = await worker.fetch(new Request('https://mewhelp.kdns.fr/'));
  assert.equal(response.status, 503);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});
