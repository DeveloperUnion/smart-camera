// On-device debug log for talk mode. iPhone Safari can't easily be attached to
// a desktop Web Inspector in the field, so `[live]` diagnostics are also kept in
// a small ring buffer that survives reloads (localStorage) and is viewable /
// copyable on-screen via <DebugLogPanel>. Every entry still mirrors to
// console.info so a connected inspector shows it too.

export type LogEntry = { t: number; msg: string };

const MAX = 300;
const KEY = 'talkDebugLog';

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
  entries = [...entries, { t: Date.now(), msg }].slice(-MAX);
  persist();
  // Keep the familiar prefix for a connected desktop inspector.
  console.info('[live]', msg);
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
