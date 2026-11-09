// The application and its data stay on the Instabox machine.
const ORIGIN = 'https://8000-agq35mghm2gi5lgmzym264yvue.instabox.tech';

export default {
  async fetch(request) {
    const incoming = new URL(request.url);
    const target = new URL(ORIGIN);
    // Assign the path separately: a path starting with // must not change the host.
    target.pathname = incoming.pathname;
    target.search = incoming.search;

    const forwarded = new Request(target, request);
    forwarded.headers.delete('host');
    forwarded.headers.set('x-forwarded-host', incoming.host);
    forwarded.headers.set('x-forwarded-proto', incoming.protocol.slice(0, -1));

    let upstream;
    try {
      upstream = await fetch(forwarded, { redirect: 'manual' });
    } catch {
      return new Response('MewHelp is temporarily unavailable. Please try again shortly.', {
        status: 503,
        headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' },
      });
    }

    const headers = new Headers(upstream.headers);
    const location = headers.get('location');
    if (location) {
      const redirect = new URL(location, target);
      if (redirect.origin === ORIGIN) {
        redirect.protocol = incoming.protocol;
        redirect.host = incoming.host;
        headers.set('location', redirect.href);
      }
    }

    // Pass the readable stream through without buffering SSE responses.
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  },
};
