# SmartCamera

スマホブラウザで動く、リアルタイム物体検知 + カゴ追加アプリ。

カメラを起動 → 写った物体に枠が出る → タップでカゴに追加 → 停止後にカゴから削除可能。
推論はオンデバイス (WebGPU / WASM) で完結し、映像は外部に送信されません。

仕様の詳細は [SPEC.md](./SPEC.md) を参照。

## セットアップ

```bash
npm install
npm run dev
```

モデルファイル (`public/models/yolov10n.onnx`, ~2.6MB) はリポジトリにコミット済みなので追加 DL 不要。
失われた場合は `./scripts/download-model.sh` で再取得できます。

`localhost` は `getUserMedia` が許可されますが、スマホ実機で確認するときは:
- `npm run dev -- --host` でネットワーク公開
- HTTPS が必要なら `vite-plugin-mkcert` を入れるか、Vercel preview URL で確認

## ビルド & デプロイ (Vercel)

```bash
npm run build      # → dist/
```

Vercel に Git push すれば自動検出されます (Framework: Vite)。

## ファイル構成

```
src/
├── App.tsx           状態遷移 (idle → running → stopped) + 描画 + タップ判定
├── App.css           スタイル
├── coco-labels.ts    COCO 80 → 日本語マッピング
├── types.ts          Detection 型
├── useCamera.ts      getUserMedia フック
├── useDetector.ts    onnxruntime-web 推論ループ (~10fps)
├── yolo.ts           前処理 (letterbox + 1/255) / 推論 / 後処理
├── index.css
└── main.tsx
public/
└── models/yolov10n.onnx   YOLOv10n uint8 量子化 ONNX (~2.6MB, コミット済み)
```

## 技術メモ

- 推論ランタイム: `onnxruntime-web` (WebGPU 優先、WASM SIMD フォールバック)
- WASM ファイルは jsDelivr CDN (`@1.24.3`) から実行時に取得
- モデル: YOLOv10n は NMS-free 出力 `[1, 300, 6]` = `[x1, y1, x2, y2, score, class]`
- 信頼度閾値 0.5、推論間隔 100ms (約 10fps)
- 入力: 640×640 にレターボックス + 1/255 で正規化のみ
