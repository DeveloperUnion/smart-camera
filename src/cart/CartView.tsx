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
  // Group by displayed label so distinct instances of the same product collapse
  // into one row with a × count (refined names group together, and so do
  // un-refined YOLO labels).
  const groups = new Map<string, { ids: number[]; entry: CartEntry }>();
  for (const entry of cart.values()) {
    const key = displayLabel(entry);
    const existing = groups.get(key);
    if (existing) existing.ids.push(entry.instance_id);
    else groups.set(key, { ids: [entry.instance_id], entry });
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
          {cartGroups.map(([label, { ids, entry }]) => {
            const lines = sublines(entry);
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
    </div>
  );
}
