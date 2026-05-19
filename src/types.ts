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
// finishes selecting. Mirrors the schema used by ImageModel (the dustalk
// waste-disposal pipeline) so SmartCamera-collected items can flow into the
// same downstream consumers without remapping fields.
//
// Only `name` is guaranteed; everything else may be an empty string when
// Gemini couldn't read it from the snapshot. modelNumber is read for any
// category; manufacturer/yearOfManufacture/capacity are only filled for the
// "free pickup" categories (white goods + 2-wheelers) and stay empty
// otherwise. The frontend renders whatever non-empty fields come back.
export type RefinedItem = {
  name: string;
  description?: string;
  modelNumber?: string;
  manufacturer?: string;
  yearOfManufacture?: string;
  capacity?: string;
};
