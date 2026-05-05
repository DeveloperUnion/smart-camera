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
