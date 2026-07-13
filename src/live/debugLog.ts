// On-device debug log for talk mode. iPhone Safari can't easily be attached to
// a desktop Web Inspector in the field, so `[live]` diagnostics are (1) kept in
// a small on-screen ring buffer (<DebugLogPanel>) and (2) BATCH-UPLOADED to
// /api/log so they can be reviewed centrally in the Vercel function logs — a
// per-user localStorage copy alone isn't visible to us. Every entry also
// mirrors to console.info for a connected inspector.

export type LogEntry = { t: number; msg: string };

const MAX = 300;
const KEY = 'talkDebugLog';

// Random id per page load so uploaded batches can be grouped by session in the
// server logs. Prefixed with a short time tag for rough ordering across users.
function makeSession(): string {
  try {
    const rnd = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2);
    return rnd.slice(0, 8);
  } catch {
    return Math.random().toString(36).slice(2, 10);
  }
}
const SESSION = makeSession();
// Exposed so the on-screen panel can show which session id to grep for in the
// server logs (filter `vercel logs` on "[talklog] <id>").
export const logSession = SESSION;

// Upload buffer, flushed on a short debounce or when it fills, plus a beacon on
// pagehide so a closed tab still delivers its tail. Best-effort: upload errors
// are swallowed (never dlog on failure — that would loop).
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

function queueUpload(entry: LogEntry): void {
  pending.push(entry);
  if (pending.length >= 25) {
    flush();
  } else if (!flushTimer) {
    flushTimer = setTimeout(flush, 3000);
  }
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

function load(): LogEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as LogEntry[]) : [];
    return Array.isArray(parsed) ? parsed.slice(-MAX) : [];
  } catch {
    return [];
  }
}

// Reassigned (not mutated) on every append so useSyncExternalStore sees a fresh
// reference and re-renders. Frequency is low (connection/turn events, not audio
// chunks), so copying the array each time is cheap.
let entries: LogEntry[] = load();
const listeners = new Set<() => void>();

function persist(): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    // quota/full or storage disabled — the in-memory buffer still works
  }
}

// Append one diagnostic line. Objects should be pre-stringified by the caller.
export function dlog(msg: string): void {
  const entry = { t: Date.now(), msg };
  entries = [...entries, entry].slice(-MAX);
  persist();
  // Keep the familiar prefix for a connected desktop inspector.
  console.info('[live]', msg);
  // Ship it to /api/log so it's reviewable in the Vercel function logs.
  queueUpload(entry);
  listeners.forEach((l) => l());
}

export function clearLog(): void {
  entries = [];
  persist();
  listeners.forEach((l) => l());
}

export function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function getLog(): LogEntry[] {
  return entries;
}

// "HH:MM:SS.mmm  msg" per line — the format copied to the clipboard.
export function formatLog(list: LogEntry[] = entries): string {
  return list
    .map((e) => {
      const d = new Date(e.t);
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      const ms = String(d.getMilliseconds()).padStart(3, '0');
      return `${hh}:${mm}:${ss}.${ms}  ${e.msg}`;
    })
    .join('\n');
}
