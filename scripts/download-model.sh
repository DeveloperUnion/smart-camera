#!/usr/bin/env bash
set -euo pipefail

# YOLOv10n (fp16, ~4.7MB) from public HF mirror.
# COCO 80 classes, NMS-free output [1, 300, 6] = [x1, y1, x2, y2, score, class].

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/public/models/yolov10n_fp16.onnx"
URL="https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model_fp16.onnx"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  echo "Model already present: $DEST"
  exit 0
fi

echo "Downloading $URL"
curl -fsSL -o "$DEST" "$URL"
echo "Saved to $DEST ($(wc -c < "$DEST") bytes)"
