# SmartCamera

スマホブラウザで動く、動画ベースの物体検知 + カゴ追加アプリ。

最大 10 秒の動画を撮影 → Gemini 3 Flash が全フレームを解析 → 動画再生に同期して bbox が表示される → タップでカゴに追加。

仕様の詳細は [SPEC.md](./SPEC.md) を参照。

## セットアップ

```bash
npm install
```

### 環境変数

`GEMINI_API_KEY` が必須。[Google AI Studio](https://aistudio.google.com/apikey) で発行。

ローカル開発時はリポジトリ直下に `.env.local` を作成:

```
GEMINI_API_KEY=AIza...
```

Vercel デプロイでは Project Settings → Environment Variables に同名で登録。

### ローカル実行

API ルート (`/api/detect-video`) を含めて動かすには Vercel CLI を使う:

```bash
npx vercel dev
```

UI だけで API 不要なら `npm run dev` でも起動できるが、撮影後の解析でエラーになる。

スマホ実機で動作確認するときは:
- `npx vercel dev --listen 0.0.0.0:3000` でネットワーク公開
- HTTPS が必要なら Vercel preview URL に push してそちらで確認するのが早い (`getUserMedia` は HTTPS 必須)

## ビルド & デプロイ

```bash
npm run build      # → dist/
```

Vercel に Git push すれば自動デプロイ (Framework: Vite、Functions: `api/`)。

## ファイル構成

```
api/
└── detect-video.ts        Vercel Function (Gemini 3 Flash 呼び出し)
src/
├── App.tsx                idle → recording → analyzing → review → cart の状態遷移
├── App.css                スタイル
├── types.ts               Detection / Appearance 型
├── useCamera.ts           getUserMedia + MediaStream 管理
├── useRecorder.ts         MediaRecorder ラッパー (10s auto-stop)
├── useVideoDetector.ts    /api/detect-video を叩いて結果を返すフック
├── playbackOverlay.ts     currentTime に対する bbox 補間 (純関数)
├── index.css
└── main.tsx
```

## 技術メモ

- 推論はクラウド (Gemini 3 Flash) に丸投げするため、クライアントは onnxruntime や WebGPU 不要
- 録画形式は `MediaRecorder.isTypeSupported` で iOS は mp4/H.264、Android は webm/VP9 を選択
- 動画 Blob は base64 化して inline で Gemini に送信 (10s × 800kbps ≒ 1MB、Gemini inline 上限 20MB 内)
- Gemini 側の動画サンプリングは既定 1 fps だと取りこぼしが多いので、`videoMetadata.fps = 4` で密に見させる
- 同一物体の追跡は Gemini プロンプトで `instance_id` (整数) を明示要求し、クライアントはそれを信用してカゴ重複排除に使う
- bbox は時刻 `time_s` 付きで返ってくるため、動画再生時刻に応じて線形補間して overlay 表示
