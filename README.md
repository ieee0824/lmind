# lmind

複数のローカルLLMを脳の部位に見立て、常時思考ループを回すことで「思考」を実現する実験的プロジェクト。

## 仮説

- LLMの単発出力は思考ではない
- LLMの出力をループさせた上で、間に発生する中間情報が思考である

## アーキテクチャ

```
┌──────────┐  ┌──────────┐  ┌────────────┐
│ frontal  │←→│ temporal  │←→│hippocampus │
│ 前頭葉   │  │ 側頭葉    │  │ 海馬       │
│ gemma3:4b│  │ gemma3:1b │  │ gemma3:1b  │
│ 推論/判断│  │ 連想/認識 │  │ 記憶/想起  │
└────┬─────┘  └─────┬─────┘  └─────┬──────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
            ┌───────▼───────┐
            │  思考バス      │
            │ (Go channels) │
            └───────┬───────┘
                    │
            ┌───────▼───────┐
            │   Chat I/F    │
            └───────────────┘
```

各脳部位はgoroutineで独立に動作し、思考バス経由で中間情報（＝思考）を交換し続ける。

## 依存

- [Ollama](https://ollama.ai) — ローカルLLM実行環境
- [memAI-go](https://github.com/ieee0824/memAI-go) — 脳科学インスパイアの記憶システム（STM/LTM/感情検出）

## セットアップ

```bash
ollama pull gemma3:4b
ollama pull gemma3:1b
go build -o lmind .
./lmind
```

## 使い方

起動すると各脳部位が自律的に思考を開始する。CLIから対話可能。

```
lmind chat (type 'quit' to exit, 'thoughts' to see recent thinking)
> こんにちは
> thoughts   # 内部の思考状態を表示
> quit       # 終了
```
