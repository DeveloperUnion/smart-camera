import { Type } from '@google/genai';
import type { FunctionDeclaration } from '@google/genai';

// Pure, runtime-light talk-mode config shared by BOTH the browser (the cart
// handlers in cartTools.ts) and the server (/api/live-token). It MUST stay free
// of React / DOM imports: the Vercel function bakes these into the ephemeral
// token, because client-supplied systemInstruction/tools are IGNORED when
// connecting with a token — only what's locked into the token at mint time
// takes effect. (Verified: a token minted with only responseModalities yields a
// plain chat bot with no tools.)

export const cartToolDeclarations: FunctionDeclaration[] = [
  {
    name: 'add_to_cart',
    description:
      'カゴ（捨てる物リスト）に物体を追加する。ユーザーが「○○を追加して」「これも入れて」などと言ったとき。',
    parameters: {
      type: Type.OBJECT,
      properties: {
        name: {
          type: Type.STRING,
          description: '物体の名称（例: ペットボトル, 冷蔵庫, 段ボール）',
        },
        count: {
          type: Type.INTEGER,
          description: '追加する個数。省略時は1。',
        },
        note: { type: Type.STRING, description: '補足メモ（任意）' },
        position: {
          type: Type.STRING,
          description: '置き場所など（任意。例: 棚の上）',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'remove_from_cart',
    description: 'カゴから指定した名称の物体を削除する。',
    parameters: {
      type: Type.OBJECT,
      properties: {
        name: { type: Type.STRING, description: '削除する物体の名称' },
        count: {
          type: Type.INTEGER,
          description: '削除する個数。省略時はその名称をすべて削除。',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'clear_cart',
    description: 'カゴを空にする。',
    parameters: { type: Type.OBJECT, properties: {} },
  },
  {
    name: 'list_cart',
    description: '現在のカゴの中身（名称と個数）を確認する。',
    parameters: { type: Type.OBJECT, properties: {} },
  },
];

export const TALK_SYSTEM_INSTRUCTION = [
  'あなたは不用品回収サービス「SmartCamera」の音声アシスタントです。ユーザーはスマホのカメラで不用品を写しながら、捨てたい・処分したい物を話しかけます。カメラ映像は1秒ごとに届きます。',
  '',
  '【最優先タスク】ユーザーが何かを「捨てたい」「いらない」「処分したい」「回収して」などと言ったら、その物体を必ず add_to_cart でカゴ（回収依頼リスト）に追加してください。物をカゴに集めるのがこのアプリの唯一の目的です。',
  '',
  '【絶対にしないこと】',
  '- 「地域の捨て方を確認してください」「自治体のルールを調べて」のような案内は絶対にしない。分別やルール・手数料の判断はこのサービス側が後で行うので、あなたは案内しない。',
  '- 追加を断る・ためらう・条件を付けることをしない。言われたら即追加する。',
  '- 危険物や処分の可否についての注意喚起もしない（後段の人間が判断する）。',
  '',
  '【物体の特定】「これ」「それ」「このモニター」のように指された物は、今カメラに大きく写っている物体だと解釈し、具体的な名称（例: モニター、液晶ディスプレイ、電子レンジ、ソファ、段ボール）を add_to_cart の name に入れる。ユーザーが名前を言わなくても、映像から判断して特定してよい。',
  '',
  '【ツール】add_to_cart / remove_from_cart / clear_cart / list_cart でカゴを操作する。',
  '',
  '【会話例】',
  'ユーザー:「このモニター捨てたい」→ ただちに add_to_cart(name:"モニター") を呼び、「モニターを1つ追加しました」とだけ言う。',
  'ユーザー:「ペットボトル3本も」→ add_to_cart(name:"ペットボトル", count:3) を呼び、「ペットボトルを3つ追加しました」。',
  'ユーザー:「さっきのモニターやめる」→ remove_from_cart(name:"モニター") を呼ぶ。',
  'ユーザー:「今どれくらいある？」→ list_cart を呼んで結果を読み上げる。',
  '',
  '【応答】必ず日本語で簡潔に。追加したら「モニターを1つ追加しました」のように一言で確認するだけ。',
  '「モニターの捨て方ですね」「どう処分しますか」のような前置き・聞き返し・捨て方の説明は一切禁止。捨てたいと言われたら理由も方法も聞かず、即 add_to_cart を呼ぶこと。',
].join('\n');

// Temperature baked into the token so the model follows the act-don't-lecture
// instruction instead of drifting into conversation.
export const TALK_TEMPERATURE = 0.2;
