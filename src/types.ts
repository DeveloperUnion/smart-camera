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

// One selected object in the cart, keyed by LocalTracker instance_id so tapping
// the same physical object twice is deduped while distinct instances of the
// same label aggregate as a count after refining.
//
// `source` records how it entered the cart: 'tap' (box tap on the live
// overlay) or 'voice' (Gemini Live talk mode). `note`/`position` are optional
// free-text annotations the voice mode can attach. `snapshot_b64` is optional
// because either path may fail to capture a frame (camera not ready, no frame
// sent to Live yet).
export type CartEntry = {
  instance_id: number;
  // Coarse label: the detector's generic label on tap, the spoken name on
  // voice. Shown until refined, and sent to refine-items as a hint.
  yolo_label: string;
  // Base64 JPEG (no data: prefix): a margin-padded crop of the object (or the
  // full frame when no box was available). KEPT after refine-items returns —
  // the cart screen shows it as the entry's confirmation thumbnail. Voice
  // entries start as a full frame and are re-cropped in App once refine-items
  // returns the object's box_2d (bbox is located during the "解析" step, not by
  // the Live model).
  snapshot_b64?: string;
  // Normalized 0-1 xyxy of the object WITHIN the snapshot image (inner box).
  // [0, 0, 1, 1] when the snapshot is an uncropped full frame.
  snapshot_bbox: [number, number, number, number];
  refined?: RefinedItem;
  source: 'tap' | 'voice';
  note?: string;
  position?: string;
};
