// Talk-mode diagnostics uploader. The `[live]` events are batch-uploaded to
// /api/log so they're reviewable centrally in the Vercel function logs
// (greppable as "[talklog] <session>"). Upload + console mirror only — no
// on-screen panel and no localStorage (a per-user local copy isn't visible to
// us). Best-effort: upload errors are swallowed, and we never dlog on failure
// (that would loop).

export type LogEntry = { t: number; msg: string };

// Random id per page load so uploaded batches can be grouped by session in the
// server logs.
function makeSession(): string {
  try {
    const rnd = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2);
    return rnd.slice(0, 8);
  } catch {
    return Math.random().toString(36).slice(2, 10);
  }
}
const SESSION = makeSession();

// Upload buffer, flushed on a short debounce or when it fills, plus a beacon on
// pagehide so a closed tab still delivers its tail.
let pending: LogEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;

function upload(batch: LogEntry[], beacon: boolean): void {
  if (!batch.length) return;
  const body = JSON.stringify({
    session: SESSION,
    ua: typeof navigator !== 'undefined' ? navigator.userAgent : '',
    entries: batch,
  });
  try {
    if (beacon && typeof navigator !== 'undefined' && navigator.sendBeacon) {
      navigator.sendBeacon('/api/log', new Blob([body], { type: 'application/json' }));
      return;
    }
    void fetch('/api/log', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      keepalive: true,
    }).catch(() => {});
  } catch {
    // storage/network unavailable — drop this batch
  }
}

function flush(): void {
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
  if (!pending.length) return;
  const batch = pending;
  pending = [];
  upload(batch, false);
}

if (typeof window !== 'undefined') {
  // Deliver the tail when the user backgrounds/closes the tab.
  window.addEventListener('pagehide', () => {
    if (pending.length) {
      const batch = pending;
      pending = [];
      upload(batch, true);
    }
  });
}

// Append one diagnostic line. Objects should be pre-stringified by the caller.
export function dlog(msg: string): void {
  const entry = { t: Date.now(), msg };
  // Keep the familiar prefix for a connected desktop inspector.
  console.info('[live]', msg);
  // Ship it to /api/log so it's reviewable in the Vercel function logs.
  pending.push(entry);
  if (pending.length >= 25) {
    flush();
  } else if (!flushTimer) {
    flushTimer = setTimeout(flush, 3000);
  }
}
