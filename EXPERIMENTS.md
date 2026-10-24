# 記事ごとの実行コマンド

リポジトリのルートで実行します。Pythonは仮想環境を使い、PyTorchとtorchvisionを[公式のインストール手順](https://pytorch.org/get-started/locally/)で先に入れてください。記事のCUDA実験はWSL 2上で実行しています。

```bash
python -m pip install -r requirements-experiments.txt
```

初回のFashionMNISTは約30 MBを取得します。結果は `ai_lab/results/`、図は `figures/`。GPUを使う実験は `python -c 'import torch; print(torch.cuda.is_available())'` が `True` の環境で実行してください。モデル取得に必要な容量はOllamaのモデルごとに異なり、4B・8B・12Bを揃えると合計15 GB以上になります。

## 数字画像の分類

scikit-learn同梱の小さな画像を使います。最初の2本はCPUだけで動きます。

```bash
python ai_lab/experiment_digits_classic.py --output ai_lab/results/digits_classic.json --figure figures/digits-classic.png --mistakes-figure figures/digits-mistakes.png
python ai_lab/experiment_digits_numpy_nn.py --output ai_lab/results/digits_numpy_nn.json --figure figures/digits-numpy.png
python ai_lab/experiment_digits_pytorch_nn.py --output ai_lab/results/digits_pytorch_nn.json --figure figures/digits-pytorch.png
python -m pytest ai_lab/test_numpy_nn.py ai_lab/test_pytorch_numpy_equivalence.py -q
```

## GPUの計算とFashionMNISTの学習

```bash
python ai_lab/benchmark_matmul.py --output ai_lab/results/matmul.json
python ai_lab/experiment_fashion_precision.py --data-dir ai_lab/data --output ai_lab/results/fashion_precision.json --figure figures/fashion-precision.png
python ai_lab/experiment_fashion_batch_oom.py --data-dir ai_lab/data --output ai_lab/results/fashion_batch.json --figure figures/fashion-batch.png
python ai_lab/experiment_dataloader_bottleneck.py --data-dir ai_lab/data --output ai_lab/results/dataloader.json --figure figures/dataloader.png
```

batch比較は既定では学習速度だけを測ります。巨大な合成入力でメモリ不足まで調べる場合は `--include-oom-probes` を付けますが、WSLではホスト側メモリまで圧迫し、PCの反応が悪くなることがあります。他のGPU作業を止めてから行ってください。図のOOM欄は、この指定を省くと空になります。

CNNのCPU/GPU比較とHOG比較は [README_EN.md](README_EN.md)、詳細な待ち時間の記録は [README.md](README.md#dataloaderの待ち時間を見る) に手順があります。

## ONNXと重みの量子化

```bash
python ai_lab/experiment_pytorch_onnx.py
python ai_lab/plot_pytorch_onnx.py
python ai_lab/experiment_weight_quant_sim.py
python ai_lab/plot_weight_quant_sim.py
```

ONNXの実験はCUDAでCNNを3エポック学習し直し、`ai_lab/models/` の重みとONNXを更新します。量子化シミュレーションだけなら、同梱の `fashion_cnn_state_dict.pt` をそのまま使えます。こちらは重みを丸めてFP32へ戻す実験であり、整数演算での高速化は測りません。

## Ollamaのモデル比較

Ollamaを起動し、実行するスクリプトのモデルを取得します。APIは同じLinux環境の `http://localhost:11434` を使います。

```bash
ollama pull gemma3:4b
ollama pull gemma3:4b-it-q4_K_M
ollama pull gemma3:4b-it-q8_0
ollama pull qwen3:8b
ollama pull gemma3:12b
python ai_lab/experiment_ollama_model_sizes.py
python ai_lab/experiment_gemma_q4_q8.py
python ai_lab/experiment_ollama_context_lengths.py
python ai_lab/experiment_truthfulqa_local.py
```

TruthfulQAは固定コミットのCSVを取得し、複数モデルへ繰り返し問い合わせます。長めの実験なので、まずモデルが応答することを `ollama run` で確かめてください。タグが同じでも中身が更新されることがあるため、結果JSONのモデル情報も保存します。

## 記事を検索して回答する

```bash
ollama pull qwen3:8b
ollama pull embeddinggemma
python ai_lab/experiment_local_rag.py
python ai_lab/plot_local_rag.py
python ai_lab/experiment_embedding_search.py
python ai_lab/plot_embedding_search.py
```

`corpus/` に13本の記事とSHA-256付きの `manifest.json` を置いています。配布版は文章とパスを修正した198チャンクの文書群です。記事の最初の測定時とは文章が異なるため、検索順位と回答数は一致しません。質問・期待パターン・返答・取得した本文をJSONで見比べてください。パターン一致だけで回答の正確さ全体を判断することはできません。

## ローカル画像生成

```bash
python -m pip install diffusers==0.40.0 transformers==5.15.1 accelerate==1.14.0 safetensors==0.8.0
python ai_lab/experiment_local_image_generation.py
```

CUDAが必要です。初回は `stabilityai/sd-turbo` をダウンロードします。Hugging Faceのモデルカードで利用条件を確認してください。生成画像は `ai_lab/generated_images/`、比較図は `figures/` に保存します。

## LLM生成コードのテスト

`experiment_llm_codegen_pytest.py` はモデルの返したPythonを実行します。ASTの検査とタイムアウトはサンドボックスではありません。普段使いのPCや、秘密情報・個人ファイルを読める環境では実行しないでください。

使い捨てのVMなどで、ホストのフォルダーや認証情報を渡さず、Ollamaもその環境内に用意してから実行します。

```bash
python ai_lab/experiment_llm_codegen_pytest.py --execute-in-disposable-environment
```

フラグがない場合は実行を拒否します。生成されたコードは実行ごとに別フォルダーへ保存するので、再実行しても前のコードは消えません。
