# ローカル検出モデル試行ログ

スマホブラウザ (iOS Safari / Android Chrome) WASM 単スレ環境で動かすローカル検出モデルの試行履歴。Gemini が後段でラベルを書き換えるので **絶対 mAP より bbox の recall + 数のバランス** を重視。

## 評価軸

各試行で以下を記録:

- **構成**: モデル名、入力解像度、量子化、ファイルサイズ
- **動作**: iOS Safari ロード可否、推論レイテンシ、平均 fps
- **検出品質**: 1 フレームあたりの kept box 数 (`?debug=1` の `kept`)、recall 体感、誤検出傾向
- **判断**: ✅ 採用 / ❌ 不採用 / ⚠️ 保留、理由

---

## 試行 #1 — YOLOv8n-OIV7 @ 512² (601 クラス)

- **期間**: 2026-05-17 〜 05-19
- **commit**: `731a32e` 〜 `a6114e6`
- **モデル**: `yolov8n-oiv7.pt` (Ultralytics 公式、Open Images V7 pretrain)
- **構成**: 入力 512²、INT8 動的量子化 per-tensor、~3.6MB、class-agnostic NMS、`SCORE_THRESHOLD=0.2`
- **狙い**: COCO 80 では語彙不足。OIV7 601 クラスでハサミ・USB ケーブル等の細かい物まで検出したい

### 結果

- **iOS ロード**: 当初 640² で "Load failed" (出力テンソル 20MB FP32 で memory-pressure 死)。512² に下げて回避
- **検出**: ❌ そもそも箱がほぼ出ない。`maxScore` が常時 0.1-0.2 付近で threshold 突破できず
- **原因**: 601 クラスに教師信号が薄まり、per-class sigmoid score が COCO 80 の半分以下になる

### 判断: ❌ 不採用

クラス数を増やしすぎると nano-tier モデルでは per-class confidence が崩れる。recall を上げるには threshold を 0.05 まで下げないといけないが、それだと誤検出が多すぎる。**「広い語彙 ⇆ 高い recall」のトレードオフは nano では選べない**。

---

## 試行 #2 — YOLOv11n-COCO @ 640² (80 クラス)

- **期間**: 2026-05-19 〜 (現在)
- **commit**: `c6418b2` (初回投入)、`7f0deb3` (FastSAM から復帰)
- **モデル**: `yolo11n.pt` (Ultralytics 公式、COCO pretrain)
- **構成**: 入力 640²、INT8 動的量子化 per-tensor、~2.9MB、class-agnostic NMS、`SCORE_THRESHOLD=0.15`
- **狙い**: 教師信号の密度を最大化。語彙は狭いが箱を確実に立てる

### 結果

- **iOS ロード**: ✅ 安定。出力テンソル ~2.8MB FP32 で余裕
- **検出**: ◯ COCO 主要クラス (人、車、椅子、ボトル、ノートPC、本、リモコン等) で箱が立つ
- **限界**: COCO 80 訓練範囲外 (USB ケーブル、書類、文房具、容器の蓋、リップ等) は **そもそも箱が立たない**

### 判断: ⚠️ ベースライン (現在のデフォルト)

語彙制約は本質的な限界。「COCO に居る物体しかタップできない」が許容できるなら ◯。Apache 2.0 ではなく AGPL-3.0 なのも将来課題。

---

## 試行 #3 — FastSAM-s @ 640² (class-agnostic)

- **期間**: 2026-05-20
- **commit**: `0e1fe96` (投入)、`b73d2ca` (4 段フィルタ追加)、`7f0deb3` (revert)
- **モデル**: `FastSAM-s.pt` (Ultralytics、YOLOv8s-seg を SA-1B 訓練)
- **構成**: 入力 640²、INT8 動的量子化、~11.8MB、単一クラス "object"、`SCORE_THRESHOLD=0.1` → 後に 0.25 + 面積フィルタ + containment フィルタ + top-15 cap
- **狙い**: 訓練範囲制約を捨てる。「画像内のあらゆる物体に箱」が SA-1B の教師タスク

### 結果

- **iOS ロード**: ⚠️ 12MB で動作したが境界線。詳細レイテンシ未計測
- **検出**: ❌ **箱が出すぎ**。SA-1B が「物体 + 部分 + 領域」を別々の正解として学習しているため、1 物体に複数箱が乱立
- **対策試行**: score threshold 0.25 化、NMS IoU 0.3 化、面積フィルタ 0.5%-70%、containment 80%、top-15 cap — それでも視認上多すぎ
- **根本原因**: SA-1B の "everything mask" 設計と「物体単位でタップ」の UX が噛み合わない

### 判断: ❌ 不採用

class-agnostic はアーキテクチャ的に正しいが、SA-1B 訓練のモデルは粒度が細かすぎる。**「object-level の単位で 1 物体 1 箱」が SA-1B からは復元しにくい**。

---

## 試行 #4 — DEIMv2-Pico @ 640² (実機テスト中)

- **期間**: 2026-05-21 〜
- **commit**: TBD (このコミット)
- **モデル**: DEIMv2-Pico (Intellindust AI Lab、CVPR 2025、Apache 2.0)
- **構成**:
  - backbone: **HGNetv2-Pico** (DINOv3 ではなく軽量 CNN backbone) ← 訂正
  - 入力 640²、INT8 動的量子化 per-tensor、**~2.3MB** (実測)
  - 200 queries (deploy postprocessor が上位 300 を返す)
  - SCORE_THRESHOLD = 0.25 (`?conf=N` で上書き可)
  - DETR set-prediction、**NMS 不要**

### Export 経緯

1. DEIMv2 リポジトリ `Intellindust-AI-Lab/DEIMv2` を clone
2. HuggingFace `Intellindust/DEIMv2_HGNetv2_PICO_COCO` から `model.safetensors` (6.3MB) を取得
3. safetensors → state_dict → 公式 `tools/deployment/export_onnx.py` に投入
4. **`load_state_dict` で missing keys 14 個** (`decoder.up`, `decoder.reg_scale`, `dec_bbox_head.1.*`, `dec_bbox_head.2.*`) → `strict=False` パッチで回避
   - `up` と `reg_scale` はモデルの `__init__` で `nn.Parameter` のデフォルト値 (0.5, 4.0) が設定済
   - `dec_bbox_head.1` と `.2` は `share_bbox_head=True` で `.0` と同一 Python オブジェクト = ロードは 1 回で足りる
5. opset 17 で export → FP32 ~6.4MB
6. `onnxruntime.quantization.quantize_dynamic` で UInt8 per-tensor → **2.3MB**

### 推論インターフェース

入力 (2 つ):
- `images`: float32 [1, 3, 640, 640]
- `orig_target_sizes`: int64 [1, 2] = [[640, 640]]

出力 (3 つ、postprocess 込み):
- `labels`: int64 [1, 300]、class ID
- `boxes`: float32 [1, 300, 4]、xyxy in input pixel space
- `scores`: float32 [1, 300]、score 降順

**JS 側 postprocess は score threshold だけ。** per-anchor argmax ループも NMS も letterbox 復元のみ。

### 開発機 (M4 Mac CPU) でのレイテンシ

ランダム入力で 22ms。iOS Safari WASM 単スレでは経験上 5-10 倍に落ちるので **100-200ms / フレーム** の見込み = **5-10fps**。これは現 YOLOv11n より速い可能性が高い (YOLOv11n は M4 CPU で ~30ms、iOS で ~600ms)。

### 結果

- **iOS ロード**: ✅ 成功 (2.3MB は余裕)
- **bbox 配置**: ◯ 物体に概ね正しく箱が立つ。FastSAM のような乱立はなく、YOLOv11n-COCO 並みのクリーンさ
- **classification**: ❌ 著しく劣化。例: **ポスターが「歯ブラシ」(COCO class 79) として分類**
- **原因推定**: 1.5M params + INT8 per-tensor 量子化で分類ヘッドが崩れる。DETR は attention 層が量子化感度高く、Pico 規模では誤分類が出やすい
- **mitigation**: COCO ラベルを UI に出さず **全 box を `"物体"` 固定表示** に変更 (commit TBD)。box 自体は使えるので Gemini 詳細化に流す前提で UX 損失ゼロ

### 判断: ⚠️ ラベル捨てて採用、ただし classification 改善余地あり

- box 配置は良好で、UI でラベル隠せば運用可能
- ただし classification の質を取り戻したいなら **DEIMv2-N (3.6M, AP 43.0)** に格上げが候補
- 一旦この構成 (Pico + ラベル隠し) でユーザー確認 → ダメなら次の試行へ

**追記 (2026-05-21)**: ユーザー要望で box 上のテキストラベルを完全に非表示化、live mode のカート chip も「選択中 ×N」に集約 (commit `19f9c39`)。

---

## 試行 #5 — DEIMv2-N @ 640²

- **期間**: 2026-05-21 〜
- **commit**: TBD (このコミット)
- **モデル**: DEIMv2-N (Intellindust AI Lab、CVPR 2025、Apache 2.0)
- **構成**:
  - backbone: HGNetv2-N
  - 入力 640²、INT8 動的量子化 per-tensor、**~4.4MB** (実測)
  - Pico と違って `share_bbox_head=False` + `gateway=True` で各層独立 = 表現力↑
  - I/O は Pico と同一: `images` + `orig_target_sizes` 入、`labels/boxes/scores` 出
  - SCORE_THRESHOLD = 0.25 維持
- **狙い**: Pico (AP 38.5) で box の質が惜しいときの格上げ。N は **AP 43.0** で +4.5、param 数 1.5M → 3.6M、INT8 サイズ 2.3MB → 4.4MB
- **コード変更**: 推論コード無変更、`modelUrl` の 1 文字差し替えのみ

### Export 経緯

Pico と同パイプライン:
1. HF `Intellindust/DEIMv2_HGNetv2_N_COCO` から safetensors (14MB) 取得
2. state_dict 化 → `tools/deployment/export_onnx.py` の strict=False パッチ版に投入
   - N は `share_bbox_head=False` なので tied weights 問題は無し、念のため strict=False で
3. FP32 ONNX 14.8MB → INT8 per-tensor 量子化 → **4.4MB**

開発機 (M4 Mac CPU) でランダム入力推論 ~27ms (Pico 22ms より 23% 遅い)。iOS WASM 単スレで 150-300ms / フレーム = 3-7fps 見込み。

### 結果

- **iOS ロード**: ✅ 成功 (4.4MB は余裕)
- **bbox 配置**: ✅ Pico より明確に改善。物体への当たりが良くなり、見落としも減
- **box 数**: 安定。1 物体 1 箱の傾向を維持 (FastSAM のような乱立なし)
- ラベルは UI で非表示 (`19f9c39` で設定済)、Gemini が後段で詳細化

### 判断: ✅ 採用 (current default)

box 品質を理由にした格上げ目的は達成。これ以上 (DEIMv2-S = 9.7M / ~10MB / AP 50.9) は box 品質伸びはあるがファイルサイズが iOS で OIV7-512² や FastSAM と同じ境界線に戻ってしまうので、**ここで一旦確定**。

更新が必要になる条件:
- 特定シーンで box の取りこぼしが目立つ → DEIMv2-S 検討
- iOS で 1fps を切るほど遅い → DEIMv2-Pico に降格 or DEIMv2-Atto (0.49M) 試行
- ライセンスや精度で要件変わる → YOLO26-N / RF-DETR 等再検討

---

## 候補リスト (試行待ち)

優先度順:

1. **DEIMv2-Pico** ← 試行 #4 で実施中
2. **YOLO26-N** (Ultralytics 2025/10) — 既存 YOLOv11n の素直な後継、CPU 43% 速い、NMS-free、AGPL-3.0
3. **DEIMv2-S** (9.71M / ~10MB) — Pico でリコール不足なら精度上げ
4. **YOLOv10n** (2024) — Apache 2.0、NMS-free、ライセンス避難先
5. **DEIMv2-Atto** (0.49M / ~1MB) — Pico で速度足りなければ降格

検討から外れたもの:

- **RF-DETR-Nano**: 名前は Nano だが 30.5M params → iOS WASM 圏外
- **YOLOv12n**: AGPL のまま、YOLO26 の方が新しく速い
- **YOLO-World / OWLv2 / Grounding DINO**: open-vocab 系は CLIP encoder 等で重すぎ
- **MobileSAM / EdgeSAM / TinySAM**: prompt-based なので "everything mode" が grid-sample で重い
- **LVIS-1203 系**: nano-tier で per-class 信号薄、OIV7 と同じ失敗パターン
