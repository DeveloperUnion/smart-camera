import { useState } from 'react';
import type { CartEntry, RefinedItem } from '../types';
import { Button } from '../ui/Button';

function displayLabel(e: CartEntry): string {
  return e.refined?.name ?? e.yolo_label;
}

// Ordered sublines for a cart row, shown under the name in muted text. Refined
// structured fields first (matching the previous behavior), then any voice-mode
// note/position annotations.
function sublines(e: CartEntry): string[] {
  const lines: string[] = [];
  const r: RefinedItem | undefined = e.refined;
  if (r) {
    if (r.modelNumber) lines.push(r.modelNumber);
    if (r.description) lines.push(r.description);
    const attrs: string[] = [];
    if (r.manufacturer) attrs.push(r.manufacturer);
    if (r.yearOfManufacture) attrs.push(`${r.yearOfManufacture}年`);
    if (r.capacity) attrs.push(r.capacity);
    if (attrs.length) lines.push(attrs.join(' · '));
  }
  if (e.position) lines.push(e.position);
  if (e.note) lines.push(e.note);
  return lines;
}

type Props = {
  cart: Map<number, CartEntry>;
  refineError: string | null;
  onRemove: (ids: number[]) => void;
  onReset: () => void;
};

export function CartView({ cart, refineError, onRemove, onReset }: Props) {
  // Snapshot being viewed full-screen (base64 JPEG), or null when closed.
  const [lightbox, setLightbox] = useState<string | null>(null);

  // Group by displayed label so distinct instances of the same product collapse
  // into one row with a × count (refined names group together, and so do
  // un-refined YOLO labels).
  const groups = new Map<string, CartEntry[]>();
  for (const entry of cart.values()) {
    const key = displayLabel(entry);
    const existing = groups.get(key);
    if (existing) existing.push(entry);
    else groups.set(key, [entry]);
  }
  const cartGroups = Array.from(groups.entries());

  return (
    <div className="screen stopped">
      <h1>カゴの中身</h1>
      {refineError && (
        <div className="status err">
          詳細取得に失敗しました ({refineError})。YOLO の暫定ラベルで表示しています。
        </div>
      )}
      {cartGroups.length === 0 ? (
        <p className="lead">何も追加されていません。</p>
      ) : (
        <ul className="cart">
          {cartGroups.map(([label, entries]) => {
            const ids = entries.map((e) => e.instance_id);
            const lines = sublines(entries[0]);
            const thumbs = entries.filter((e) => e.snapshot_b64);
            return (
              <li key={label}>
                <span className="label">
                  {label}
                  {lines.map((line, i) => (
                    <span
                      key={i}
                      style={{
                        display: 'block',
                        fontSize: 12,
                        color: '#888',
                      }}
                    >
                      {line}
                    </span>
                  ))}
                  {thumbs.length > 0 && (
                    <span className="cart-thumbs">
                      {thumbs.map((e) => (
                        <img
                          key={e.instance_id}
                          className="cart-thumb"
                          loading="lazy"
                          decoding="async"
                          src={`data:image/jpeg;base64,${e.snapshot_b64}`}
                          alt={label}
                          onClick={() => setLightbox(e.snapshot_b64!)}
                        />
                      ))}
                    </span>
                  )}
                </span>
                <span className="count">× {ids.length}</span>
                <button
                  className="remove"
                  onClick={() => onRemove(ids)}
                  aria-label="削除"
                >
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      )}
      <Button onClick={onReset}>最初から</Button>
      {lightbox && (
        <div
          className="lightbox"
          role="dialog"
          aria-label="スナップショット拡大表示"
          onClick={() => setLightbox(null)}
        >
          <img src={`data:image/jpeg;base64,${lightbox}`} alt="" />
        </div>
      )}
    </div>
  );
}
