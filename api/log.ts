import type { VercelRequest, VercelResponse } from '@vercel/node';

export const config = { maxDuration: 10 };

// Sink for on-device talk-mode diagnostics. The client batches its `[live]`
// events here so they can be reviewed centrally in the Vercel function logs
// (no localStorage-only, no DB): every line is printed with a "[talklog]"
// prefix + a per-page-load session id, so `vercel logs` (or the dashboard's
// Observability view) can be filtered/grepped per user session.
//
// Fire-and-forget from the client's side: we validate loosely and never fail
// hard, since dropping a diagnostic must never affect the user's session.
type Incoming = {
  session?: string;
  ua?: string;
  entries?: Array<{ t?: number; msg?: string }>;
};

export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }

  const { session, ua, entries } = (req.body ?? {}) as Incoming;
  if (!Array.isArray(entries) || entries.length === 0) {
    res.status(204).end();
    return;
  }

  const sid = String(session ?? 'nosession').slice(0, 64);
  // Cap per request so a malformed/huge payload can't flood the logs.
  const capped = entries.slice(0, 300);

  // One context line per batch (session -> device), then one line per event.
  console.log(
    `[talklog] ${sid} batch n=${capped.length} ua="${String(ua ?? '').slice(0, 180)}"`,
  );
  for (const e of capped) {
    const ts =
      typeof e?.t === 'number' ? new Date(e.t).toISOString() : '';
    const msg = String(e?.msg ?? '').slice(0, 2000);
    console.log(`[talklog] ${sid} ${ts} ${msg}`);
  }

  res.status(204).end();
}
