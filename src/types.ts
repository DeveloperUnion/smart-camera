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

// Local-mode live detection: a single bbox observed in the current frame.
// No tracking — each tap mints a fresh cart instance.
export type LiveBox = {
  // Pixel coords in the source video frame (xyxy).
  bbox: [number, number, number, number];
  label: string;
  classId: number;
  score: number;
};
