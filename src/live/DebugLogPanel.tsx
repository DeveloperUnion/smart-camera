import { useState, useSyncExternalStore } from 'react';
import { subscribe, getLog, clearLog, formatLog } from './debugLog';

// Floating on-device log viewer. A small button toggles a panel listing the
// persisted `[live]` diagnostics (newest at the bottom) with Copy / Clear, so
// the log can be reviewed and shared straight from the phone — no desktop Web
// Inspector needed.
export function DebugLogPanel() {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const entries = useSyncExternalStore(subscribe, getLog);

  const copy = async () => {
    const text = formatLog(entries);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard blocked (insecure context / permissions): fall back to a
      // selectable prompt the user can long-press to copy.
      window.prompt('コピーできない場合は手動で選択してください', text);
    }
  };

  return (
    <>
      <button
        className="debug-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-label="デバッグログ"
      >
        🐞{entries.length ? ` ${entries.length}` : ''}
      </button>
      {open && (
        <div className="debug-panel" role="dialog" aria-label="デバッグログ">
          <div className="debug-panel-head">
            <span>ログ {entries.length}件</span>
            <div className="debug-panel-actions">
              <button onClick={copy}>{copied ? 'コピー済' : 'コピー'}</button>
              <button onClick={clearLog}>クリア</button>
              <button onClick={() => setOpen(false)}>閉じる</button>
            </div>
          </div>
          <div className="debug-panel-body">
            {entries.length === 0 ? (
              <div className="debug-empty">まだログがありません</div>
            ) : (
              entries.map((e, i) => {
                const d = new Date(e.t);
                const ts =
                  `${String(d.getHours()).padStart(2, '0')}:` +
                  `${String(d.getMinutes()).padStart(2, '0')}:` +
                  `${String(d.getSeconds()).padStart(2, '0')}.` +
                  `${String(d.getMilliseconds()).padStart(3, '0')}`;
                return (
                  <div key={i} className="debug-line">
                    <span className="debug-ts">{ts}</span>
                    <span className="debug-msg">{e.msg}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </>
  );
}
