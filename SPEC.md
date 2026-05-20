# SmartCamera 仕様書

スマホブラウザで動く、物体検知 + カゴ追加アプリ。**ローカル YOLO + Gemini 詳細化のハイブリッド**:

1. ライブカメラに対し on-device で **FastSAM-s** (class-agnostic objectness, 単一クラス "物体") を毎フレーム (~3fps) 実行、bbox を重畳表示
2. ユーザーがタップ → bbox 内のフレームをスナップショット (フルフレーム JPEG + bbox メタ) として保持。カートには即座に "物体" として追加
3. 「停止」タップ → スナップショット群を 1 リクエストで Gemini に送り、各物体の構造化属性 (`name`, `description`, `modelNumber`, `manufacturer`, `yearOfManufacture`, `capacity`) を取得
4. カート画面で Gemini の `name` (カテゴリ表記) と補助属性を表示

オフラインや API 失敗時は YOLO ラベルで確定する (フォールバック)。

## ユーザーフロー

1. ページを開く → 「カメラ開始」をタップ
2. 背面カメラのライブプレビュー + bbox 重畳 (グレー点線) が出る
3. 物体をタップ → bbox が **青の実線** に変化、下部カート chip に YOLO ラベルで追加
4. 同じ物体を 2 度タップしても重複しない (LocalTracker の `instance_id` でデデュプ)
5. 最大 30 個まで選択可。上限到達時は注意表示
6. 「停止」 → スピナー「解析中…」 (Gemini 5–15 秒)
7. カート画面: `具体名 × N` 表示。2行目に補助属性 (ブランド / 色 / サイズ等)
8. 「最初から」でリセット → idle 画面

ネットワーク失敗時はカート画面に到達し、各行は YOLO ラベル (例: 「缶 × 2」) のまま、上部にエラーバナー表示。

## 技術スタック

| 領域 | 採用 |
|---|---|
| フロント | React 19 + Vite + TypeScript |
| ローカル推論 | FastSAM-s INT8 ONNX (class-agnostic, 単一クラス), `onnxruntime-web` WASM |
| 詳細化 | Gemini 3 Flash (`@google/genai`, multimodal) |
| サーバ | Vercel Serverless Function (`/api/refine-items`) |
| カメラ | `getUserMedia` (`facingMode: 'environment'`) |
| 描画 | `<video>` + `<canvas>` オーバーレイ (Canvas 2D) |
| 物体追跡 | IoU ベース sticky `instance_id` (`LocalTracker`) |
| 状態管理 | React state |
| デプロイ | Vercel (静的 SPA + Serverless API) |

## 動作環境

- iPhone 14 / iOS Safari 以降 (実機検証下限)
- Android Chrome (近代版)
- HTTPS 必須 (`getUserMedia` の制約)
- ネットワーク必須 (Gemini 詳細化)。オフライン時は YOLO ラベルで確定

## パラメータ初期値

| 項目 | 値 |
|---|---|
| ローカル推論 fps | デスクトップ ~3、iOS WebKit ~1.5 (`?fps=N` で上書き) |
| YOLO 入力解像度 | 512² (letterbox + 114-gray padding) |
| YOLO スコア閾値 | 0.2 (`?conf=N` で上書き) |
| NMS IoU | 0.5、class-agnostic |
| スナップショット | 長辺 720px JPEG quality 0.8 |
| 選択上限 | 30 個 |
| Gemini モデル | `gemini-3-flash-preview` |
| Gemini thinking | デフォルト (auto) |
| Vercel Function `maxDuration` | 60 秒 |

## API 契約

### `POST /api/refine-items`

```jsonc
// Request
{
  "items": [
    {
      "id": 7,                              // クライアント側 instance_id
      "yolo_label": "缶",                   // YOLO 由来のヒント
      "image_b64": "<base64 JPEG>",         // フルフレーム
      "image_mime": "image/jpeg",
      "bbox": [0.31, 0.42, 0.58, 0.69]     // 0-1 正規化 xyxy
    }
  ]
}

// Response (success)
{
  "items": [
    {
      "id": 7,
      "refined": {
        "specific_name": "コカコーラ缶 350ml",
        "brand": "コカコーラ",
        "category": "飲料",
        "color": "赤",
        "size_estimate": "350ml 缶"
      }
    }
  ]
}
```

`specific_name` 以外の属性は最大限の best-effort、欠落する場合あり。Gemini が判定できなかったアイテムは items から省略される (フロントは YOLO ラベルで継続表示)。

失敗時はステータス 500 + `{ "error": "..." }`、フロントは YOLO ラベルで確定 + 上部にエラーバナー。

## 描画スタイル

- 未追加ボックス: `strokeStyle = '#9CA3AF'`、`setLineDash([6, 4])`、線幅 2px
- 追加済ボックス: `strokeStyle = '#3B82F6'`、実線、線幅 3px
- タップフラッシュ: 青実線 4px が 700ms フェードアウト
- ラベル: ボックス上端に半透明黒背景 + 白文字

## 非機能要件

- スナップショット JPEG は Gemini 応答 (成功・失敗いずれも) 後にカートエントリから削除 (メモリ解放)
- API キー (`GEMINI_API_KEY`) は Vercel 環境変数で管理、クライアントには露出しない
- Gemini は学習に使われない契約 (Google Cloud の AI 利用規約による)
- バックグラウンド化時に YOLO 推論を一時停止 (`useLocalDetector` の `document.hidden` ガード)

## 既知の割り切り

- 詳細化 (`/api/refine-items`) はキャンセル不可 (フロントから fetch を中断する手段は実装していない)
- LocalTracker は IoU ベースで 1.5 秒の GC、見失って再登場した物体は別 `instance_id` になる (= カートに重複追加され得る)
- カメラを速く動かすと YOLO の検出が安定しないので、撮影中はゆっくり動かす案内を継続
- Vercel リクエストボディは 4.5MB 上限。30 個 × ~80KB ≈ 2.4MB を想定 (720px / quality 0.8)、十分余裕
- 1 セッション 1 確定。連続スキャンは「最初から」で戻る

## コスト目安

選択 5 個程度のセッション 1 回で約 $0.005–0.015 (Gemini 3 Flash の入出力トークン換算、画像 5 枚 720px + 短いプロンプト + 構造化 JSON 出力)。30 個満載で $0.02–0.04 程度。

## ローカル検出モデル (FastSAM-s, class-agnostic)

ローカル推論は **FastSAM-s** (Ultralytics 配布の `FastSAM-s.pt`) を **640² 入力**で ONNX export + INT8 動的量子化 (per-tensor) したものを使用。

### なぜ FastSAM か

FastSAM-s は YOLOv8s-seg アーキテクチャを **SA-1B (Segment Anything Model の 1.1 億マスクデータセット)** で再訓練したモデルで、教師タスクが「画像内の全物体に箱とマスクを引く」そのもの。

**class-agnostic** (単一クラス "object"): 何クラスに属するかは予測しない。「ここに物体がある」だけを出力する。クラス頭がないので per-anchor argmax ループも不要 (postprocess が単純化)。

これは SmartCamera の方針 — **「ラベルは Gemini に丸投げ、YOLO は箱だけ出す」** — と完全に噛み合う。COCO 80 クラス / OIV7 601 クラスのような訓練範囲制約が消え、写った物体には基本的に箱が立つ。

### 候補比較 (検討時)

| モデル | クラス数 | INT8 サイズ | 出力テンソル | 設計 | 採否 |
|---|---|---|---|---|---|
| **FastSAM-s (採用)** | **1 (class-agnostic)** | **~12 MB** | **~1.2 MB @640²** | YOLOv8s-seg を SA-1B で訓練 | ✅ |
| YOLOv11n-COCO | 80 | ~3 MB | ~2.8 MB | COCO 訓練、密な教師信号 | 旧構成、COCO 範囲外で recall 不足 |
| YOLOv8n-OIV7 | 601 | ~3.6 MB | ~13 MB @512² | OIV7 訓練、階層クラス | 旧旧構成、per-class 信号薄でスコア低 |
| MobileSAM / EdgeSAM | 1 | ~10 MB | varies | ViT-tiny に SAM 蒸留、prompt-based | "全物体モード" は grid-sample で重い |
| FastSAM-x | 1 | ~40 MB | larger | FastSAM 大型版 | iOS で動作圏外 |
| YOLOE-S | open-vocab | ~10 MB | varies | 2024 新顔、prompt-free 可 | ONNX export 未成熟、将来候補 |

### モデル出力形状

```
output0: [1, 37, 8400]
  channels[0:4]   = cx, cy, w, h (640px space)
  channels[4]     = objectness (sigmoid'd)
  channels[5:37]  = 32 マスク係数 ← 読まない
output1: [1, 32, 160, 160]    # proto masks ← 読まない
```

`output1` (~3.3MB FP32) と `output0` のマスク係数部分は ORT が allocate するが我々は dereference しない。実質的な活性化メモリ使用量は YOLO COCO とほぼ同等。

### SCORE_THRESHOLD = 0.1

class-agnostic objectness は COCO YOLO の per-class score より絶対値が低めに出るので閾値も下げる。false positive は「タップされないだけ」なのでコストはなく、recall を優先。`?conf=N` で上書き可。

### NMS は class-agnostic

クラスは元々 1 種類だが、近接物体に対する重複箱を IoU > 0.5 で抑制 (`src/yolo11.ts:postprocess`)。

### モデル成果物 (リポジトリ外の export 手順、再生成する場合)

```python
from ultralytics import YOLO
m = YOLO('FastSAM-s.pt')  # 公式重みを自動ダウンロード (~22.7 MB)
m.export(format='onnx', imgsz=640, opset=17, dynamic=False, simplify=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    'FastSAM-s.onnx',
    'fastsam_s_640_uint8.onnx',
    weight_type=QuantType.QUInt8,
    per_channel=False,  # per-tensor は onnxruntime-web WASM との相性が良い
)
```

ノートPC で 1 分。GPU 不要、fine-tuning 不要。

成果物:
- `public/models/fastsam_s_640_uint8.onnx` — 量子化済みモデル (~11.8 MB)
- ラベル辞書ファイルは不要 (全 box が "物体" 固定、`src/yolo11.ts:postprocess` でハードコード)

### 検証要件

- iPhone 14 以降の実機で 5 分連続ライブ実行、タブキル無し (12MB モデルが iOS WebKit のメモリプレッシャー境界に近い)
- 推論レイテンシ中央値 < 1000ms (1fps 維持)
- COCO 範囲外の物体 (USB ケーブル、ハサミ、リップ、書類、容器の蓋等) に箱が立つこと
- LocalTracker の IoU マッチングが class label "物体" 一律でも instance_id を正しく区別できること (同種物体が並んでいても別々に追跡)

### iOS で落ちる場合のフォールバック

1. **入力 512² に再 export** (出力 anchors 5376 に減、活性化半減)
2. **`SCORE_THRESHOLD` 0.1 → 0.2** に引き上げ (raw candidate 数削減で postprocess 軽量化)
3. **YOLOv11n-COCO に rollback** (commit 履歴で復旧可)

### ライセンス注意

FastSAM および Ultralytics YOLO は **AGPL-3.0**。SA-1B 自体は Apache-2.0 + CC-BY-4.0 (研究/商用とも可、再配布可)。社外公開する場合は商用ライセンス、または Apache-2.0 互換の代替 (YOLOE 等) への切替検討が必要。
