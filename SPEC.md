# SmartCamera 仕様書

スマホブラウザで動く、物体検知 + カゴ追加アプリ。idle 画面のトグルで 2 つのモードを切り替え可能:

- **クラウド (動画解析)**: 最大 10 秒の動画を撮影 → Gemini 3 Flash で全フレーム横断の物体検知 + instance tracking → 動画再生に同期して bbox を重畳表示 → タップでカゴ追加。任意物体・日本語ラベル直接出力。
- **ローカル (リアルタイム)**: YOLOv8n-OIV7 (Open Images V7 事前学習版、ONNX, dynamic int8 量子化, ~3.7MB) を on-device で実行、ライブカメラに bbox を毎フレーム (~3fps) 重畳 → タップでカゴ追加。**601 クラス**(家電・道具・文房具・容器・食品・動植物等)、Gemini 不要、ネットワーク不要。iOS Safari ではメモリ上限により落ちる場合があるため検証必要。

モード選択は永続化されない (リロードでクラウドに戻る)。

## ユーザーフロー

1. ページを開く → 「📹 撮影開始」をタップ
2. 背面カメラのライブプレビューが表示され、自動で録画開始
3. 上部に **REC ●** と残り秒数、下部に進捗バー (最大 10s で自動停止、手動でも停止可)
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
| 最大録画時間 | 10 秒 |
| 録画解像度 | 640×480 (`getUserMedia` ideal) |
| 録画ビットレート | 800 kbps |
| Gemini モデル | `gemini-3-flash-preview` |
| Gemini thinking | デフォルト (auto) — Agentic Vision の精度に必要 |
| 動画サンプリング | 4 fps (`videoMetadata.fps`) |
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
- 録画は最大 10 秒 (それ以上は upload サイズと解析時間が膨らむため固定上限)
- カメラを速く動かすとモーションブラーで検出精度が落ちる。撮影中はゆっくり動かすよう UI で案内する
- オフライン動作不可 (Gemini API 必須)
- 1 セッション 1 動画。連続スキャンしたい場合は「最初から」で戻る

## コスト目安

10 秒動画 1 回スキャン (4 fps サンプリング): 約 $0.010–0.015 (Gemini 3 Flash の入出力トークン換算)

## ローカル検出モデル(OIV7-601)

ローカルモードは **Ultralytics 公式の `yolov8n-oiv7.pt`** をベースに ONNX export + INT8 動的量子化したものを使用。601 クラスで COCO 80 を大きく上回るカバレッジを得つつ、量子化後 3.7 MB と現実的なサイズに収まる。

候補比較(採用時の検討):

| データセット | クラス数 | 出力テンソル(FP32) | 公式重み入手 | 採否 |
|---|---|---|---|---|
| COCO(旧) | 80 | ~2.8 MB | ✅ `yolo11n.pt` | 旧構成 |
| **Open Images V7(採用)** | **601** | **~20 MB** | ✅ **`yolov8n-oiv7.pt`** | ✅ |
| Objects365 | 365 | ~12 MB | △ 部分的 | iOS で落ちる場合の代替候補 |
| LVIS | 1203 | ~40 MB | △ community release | ロングテイルで実用精度落ち + iOS メモリ危険のため不採用 |

注: 同じ公式リリースに `yolo11n-oiv7.pt` は存在しないため v8n ベースを採用。出力フォーマット(`[1, 4+NUM_CLASSES, 8400]` anchor-free 形式)は v11 と同一なので推論コード(`src/yolo11.ts`)はそのまま動作する。ファイル名 `yolo11.ts` は歴史的経緯で残置(改名は将来のリファクタで)。

### モデル成果物(リポジトリ外の export 手順、再生成する場合)

```python
from ultralytics import YOLO
m = YOLO('yolov8n-oiv7.pt')  # 公式重みを自動ダウンロード(6.9 MB)
m.export(format='onnx', imgsz=640, opset=17, dynamic=False, simplify=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    'yolov8n-oiv7.onnx',
    'yolov8n_oiv7_uint8.onnx',
    weight_type=QuantType.QUInt8,
    per_channel=True,
)
```

ノートPC で数分、GPU 不要、fine-tuning 不要。

成果物:
- `public/models/yolov8n_oiv7_uint8.onnx` — 量子化済みモデル(~3.7 MB)
- `src/oiv7-labels.ts` — 601 クラスの日本語ラベル配列 + `labelOf(classId)` ヘルパー

### iOS 互換性リスクと対処

OIV7 の出力テンソル ~20MB(FP32 で 4×8400 + 601×8400)は現状の COCO 構成 ~2.8MB から +17MB の上乗せ。iOS Safari の memory-pressure 閾値内に収まる公算は高いが実機検証必須。

**iOS 実機で落ちる場合の対処順序:**
1. `postprocess` の `SCORE_THRESHOLD` を 0.3 → 0.35 に引き上げ、`bestScore < threshold` の anchor を bbox 復号前にスキップ
2. **Objects365 (365クラス, ~12MB) に切替** — community release を要探索だが出力テンソル半減
3. それでもダメなら入力解像度 640²→416² で再 export(活性化テンソル ~40% 削減)

### 検証要件

- iPhone 14 以降の実機で 5 分連続ライブ実行、タブキル無し
- `performance.measureUserAgentSpecificMemory()` でピーク working set < 250 MB
- COCO 同等クラス(人、車、椅子、ノートPC等)の検出が旧構成と同等以上
- OIV7 で広がった「細かい物」サンプル動作確認: ハサミ・包丁・リモコン・電卓・各種容器・キーボード・USBケーブル等

### ライセンス注意

Ultralytics YOLO は **AGPL-3.0**。社外公開する場合は商用ライセンス、または Apache-2.0 の代替(YOLOE-S 等)への切替検討が必要。OIV7 自体のライセンスは Apache-2.0 + CC-BY-4.0(再配布可)。
