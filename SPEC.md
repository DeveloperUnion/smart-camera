# SmartCamera 仕様書

スマホブラウザで動く、動画ベースの物体検知 + カゴ追加アプリ。

最大 15 秒の動画を撮影 → クラウドの VLM (Gemini 3 Flash) で全フレーム横断の物体検知 + instance tracking → 動画再生に同期して bbox を重畳表示し、ユーザーがタップでカゴに追加する。

## ユーザーフロー

1. ページを開く → 「📹 撮影開始」をタップ
2. 背面カメラのライブプレビューが表示され、自動で録画開始
3. 上部に **REC ●** と残り秒数、下部に進捗バー (最大 15s で自動停止、手動でも停止可)
4. 録画停止 → 画面に「解析中…」スピナー (5–15s 程度)
5. 解析完了 → 動画が自動再生開始、検知された物体に **グレー点線のバウンディングボックス + 日本語ラベル** が時間と同期して描画される
6. ユーザーが bbox をタップ → ボックスが **青の実線** に変化、カゴに +1 (同じ物体は何度タップしても重複しない)
7. ボックスの外側をタップで再生/一時停止トグル
8. 「終了」ボタンでカゴ画面表示
9. カゴ画面: `コーラ缶 × 3` 形式 (同一ラベルの instance を集計)、各行 X で削除可
10. 「最初から」でリセット → カゴクリア + idle 画面

## 技術スタック

| 領域 | 採用 |
|---|---|
| フロント | React 19 + Vite + TypeScript |
| 推論 | クラウド (Gemini 3 Flash via `@google/genai`) |
| サーバ | Vercel Serverless Function (`/api/detect-video`) |
| カメラ | `getUserMedia` (`facingMode: 'environment'`) |
| 録画 | `MediaRecorder` (iOS: mp4/H.264、Android: webm/VP9) |
| 描画 | `<video>` + `<canvas>` オーバーレイ (Canvas 2D) |
| ラベル | Gemini が日本語名詞句で直接出力 (静的マッピング不要) |
| 物体追跡 | Gemini プロンプトで instance_id を明示要求 |
| 状態管理 | React state (Zustand など導入しない) |
| デプロイ | Vercel (静的 SPA + Serverless API) |

## 動作環境

- iPhone 14.3 以降 (iOS Safari、`MediaRecorder` 対応)
- Android Chrome (近代版)
- HTTPS 必須 (`getUserMedia` の制約)
- ネットワーク必須 (Gemini API 呼び出し)

## パラメータ初期値

| 項目 | 値 |
|---|---|
| 最大録画時間 | 15 秒 |
| 録画解像度 | 640×480 (`getUserMedia` ideal) |
| 録画ビットレート | 800 kbps |
| Gemini モデル | `gemini-3-flash` |
| Gemini thinkingBudget | 0 (検知タスクは推論不要、レイテンシ優先) |
| Bbox active window | 動画 currentTime ±0.6s 内の appearance を表示対象 |

## API 契約

### `POST /api/detect-video`

```jsonc
// Request
{
  "video": "<base64-encoded video bytes>",
  "mimeType": "video/mp4" | "video/webm"
}

// Response (success)
{
  "detections": [
    {
      "instance_id": 1,
      "label": "コーラ缶",
      "appearances": [
        { "time_s": 0.5, "bbox": [0.10, 0.20, 0.30, 0.50] },
        { "time_s": 1.2, "bbox": [0.12, 0.21, 0.31, 0.51] }
      ]
    }
  ]
}
```

bbox は 0–1 正規化 (xyxy)。`time_s` は動画先頭からの秒数。

## 描画スタイル

- 未追加ボックス: `strokeStyle = '#9CA3AF'`、`setLineDash([6, 4])`、線幅 2px
- 追加済ボックス: `strokeStyle = '#3B82F6'`、実線、線幅 3px
- タップフラッシュ: 青実線 4px が 700ms フェードアウト
- ラベル: ボックス上端に半透明黒背景 + 白文字

## 非機能要件

- 録画動画は **解析完了後にサーバ側で破棄** (Vercel Function は永続化しない、Gemini も学習に使われない契約に従う)
- クライアント側の動画 Blob は再生終了 + 「終了」タップで `URL.revokeObjectURL` で破棄
- バックグラウンド化・タブ閉じでカメラ停止 (リーク防止)
- API キー (`GEMINI_API_KEY`) は Vercel 環境変数で管理、クライアントには露出しない

## 既知の割り切り

- 録画後の解析中はキャンセル不可 (Gemini 呼び出しが完了するまで待つ)
- 同じ物体の判定は Gemini の VLM 推論に依存。完璧ではなく、フレーム間で見失った後に再登場すると別 instance になりがち
- 録画は最大 15 秒 (それ以上は upload サイズと解析時間が膨らむため固定上限)
- オフライン動作不可 (Gemini API 必須)
- 1 セッション 1 動画。連続スキャンしたい場合は「最初から」で戻る

## コスト目安

15 秒動画 1 回スキャン: 約 $0.003–0.005 (Gemini 3 Flash の入出力トークン換算)
