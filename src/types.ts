// Cloud-mode detection: an instance tracked across frames with multiple
// timestamped appearances.
export type Appearance = {
  time_s: number;
  // Normalized 0-1 in video frame coordinate space, xyxy.
  bbox: [number, number, number, number];
};

export type Detection = {
  instance_id: number;
  label: string;
  appearances: Appearance[];
};

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
