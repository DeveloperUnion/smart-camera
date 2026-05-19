// Japanese display labels for the 80 COCO classes, in the canonical
// Ultralytics class order (= the order baked into yolo11n.pt's `model.names`).
// We sourced the English order from coco_names.json at export time and
// hand-mapped each to a short, natural Japanese noun phrase suitable for
// showing inside a small bbox label on a phone screen.
//
// These labels are intentionally generic — the user will tap whatever box
// looks promising and /api/refine-items asks Gemini for the specific name
// (e.g. "コカコーラ 350ml 缶" vs the YOLO-coarse "ボトル"). So accuracy of
// the displayed label matters less than the bbox being there at all.
export const COCO_LABELS_JP: readonly string[] = [
  '人',           // person
  '自転車',       // bicycle
  '車',           // car
  'バイク',       // motorcycle
  '飛行機',       // airplane
  'バス',         // bus
  '電車',         // train
  'トラック',     // truck
  'ボート',       // boat
  '信号機',       // traffic light
  '消火栓',       // fire hydrant
  '一時停止標識', // stop sign
  'パーキング',   // parking meter
  'ベンチ',       // bench
  '鳥',           // bird
  '猫',           // cat
  '犬',           // dog
  '馬',           // horse
  '羊',           // sheep
  '牛',           // cow
  '象',           // elephant
  '熊',           // bear
  'シマウマ',     // zebra
  'キリン',       // giraffe
  'リュック',     // backpack
  '傘',           // umbrella
  'ハンドバッグ', // handbag
  'ネクタイ',     // tie
  'スーツケース', // suitcase
  'フリスビー',   // frisbee
  'スキー',       // skis
  'スノーボード', // snowboard
  'ボール',       // sports ball
  '凧',           // kite
  'バット',       // baseball bat
  'グローブ',     // baseball glove
  'スケボー',     // skateboard
  'サーフボード', // surfboard
  'テニスラケット', // tennis racket
  'ボトル',       // bottle
  'ワイングラス', // wine glass
  'コップ',       // cup
  'フォーク',     // fork
  'ナイフ',       // knife
  'スプーン',     // spoon
  'ボウル',       // bowl
  'バナナ',       // banana
  'りんご',       // apple
  'サンドイッチ', // sandwich
  'オレンジ',     // orange
  'ブロッコリー', // broccoli
  'にんじん',     // carrot
  'ホットドッグ', // hot dog
  'ピザ',         // pizza
  'ドーナツ',     // donut
  'ケーキ',       // cake
  '椅子',         // chair
  'ソファ',       // couch
  '鉢植え',       // potted plant
  'ベッド',       // bed
  'ダイニング',   // dining table
  'トイレ',       // toilet
  'テレビ',       // tv
  'ノートPC',     // laptop
  'マウス',       // mouse
  'リモコン',     // remote
  'キーボード',   // keyboard
  'スマホ',       // cell phone
  '電子レンジ',   // microwave
  'オーブン',     // oven
  'トースター',   // toaster
  'シンク',       // sink
  '冷蔵庫',       // refrigerator
  '本',           // book
  '時計',         // clock
  '花瓶',         // vase
  'はさみ',       // scissors
  'ぬいぐるみ',   // teddy bear
  'ドライヤー',   // hair drier
  '歯ブラシ',     // toothbrush
];

export function labelOf(classId: number): string {
  return COCO_LABELS_JP[classId] ?? `class_${classId}`;
}
