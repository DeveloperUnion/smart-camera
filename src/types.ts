// Local-mode raw detection from YOLO for the current frame. No tracking yet —
// see LocalTracker for the IoU-based matching that adds a sticky instance_id.
export type LiveBox = {
  // Pixel coords in the source video frame (xyxy).
  bbox: [number, number, number, number];
  label: string;
  classId: number;
  score: number;
};

// LiveBox after passing through LocalTracker — has a stable instance_id that
// persists across frames so the overlay and cart can recognize the same object.
export type TrackedBox = LiveBox & { instance_id: number };

// Per-item structured attributes returned by /api/refine-items after the user
// finishes selecting. Only specific_name is guaranteed; the rest are best-effort.
export type RefinedItem = {
  specific_name: string;
  brand?: string;
  category?: string;
  color?: string;
  size_estimate?: string;
};
