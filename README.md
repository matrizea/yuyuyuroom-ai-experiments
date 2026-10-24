# ぬるま湯ハードウェアの実験コード

[English reproduction guide](README_EN.md)

[ぬるま湯ハードウェア](https://yuyuyuroom.site/)の記事で使ったコードを置いています。PyTorchの学習、ONNXの量子化、GPUメモリの確認など。

## コードを手元に置く

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git ~/ai-experiments
cd ~/ai-experiments
```

Gitを使わない場合は、このページの「Code」→「Download ZIP」からダウンロードできます。以下のコマンドは、取得したフォルダー内で実行してください。

PyTorchとtorchvisionは先に用意します。インストール方法は[PyTorch公式](https://pytorch.org/get-started/locally/)を参照してください。記事で使ったのはWindows 11のWSL 2、Python 3.14.4、PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130、RTX 5070 Tiです。

FashionMNISTを使う実験では、初回に画像をダウンロードします。GPUメモリの実験には画像もモデルも要りません。

## 実験を選ぶ

| 調べたいこと | 実行手順 |
|---|---|
| GPUメモリが解放されない | [empty_cache・del・detach](CUDA_MEMORY.md) |
| CNNをCPUとGPUで学習する | [CNNの学習手順（英語）](README_EN.md#fashionmnist-cnn-training) |
| ONNXモデルをINT8にする | [INT8量子化](#onnx-runtimeのint8量子化) |
| 学習中にデータ待ちがあるか | [DataLoaderの待ち時間](#dataloaderの待ち時間を見る) |
| HOG＋SVMとCNNを比べる | [比較実験の手順（英語）](README_EN.md#hog--linear-svm-versus-a-cnn) |
| NumPyの学習・混合精度・RAG・ローカルLLMなど | [記事ごとの実行コマンド](EXPERIMENTS.md) |
| NV-Tesseractで予測と異常検知を動かす | [SDKの取得と実行](nv_tesseract_lab/README.md) |
| Depth Anything V1・V2・V3の出力と速度を比べる | [深度推定の比較](depth_anything/README.md) |

## ONNX RuntimeのINT8量子化

対応記事: [ONNXのCNNを実際にINT8量子化したら速くなる？](https://yuyuyuroom.site/?p=471)

同梱のCNNを量子化し、精度、ファイルサイズ、CPUでの推論速度を比べます。GPUは使いません。ONNX、ONNX Runtime、NumPyは記事と同じ版を指定しています。

```bash
python -m pip install onnx==1.22.0 onnxruntime==1.29.0 numpy==2.5.2 matplotlib
python ai_lab/experiment_onnx_int8_static.py --preprocess
python ai_lab/plot_onnx_int8_static.py
python -m unittest discover -s ai_lab -p test_onnx_int8_static.py -v
```

結果は `ai_lab/results/onnx_int8_static/preprocessed/` の `result.json` と `timings_raw.csv`、図は `figures/onnx-int8-static-cpu-comparison.png` に出ます。テストはこの結果ファイルを使うため、実験の後に実行します。

元のモデルは `ai_lab/models/fashion_cnn_dynamic.onnx`。ファイルを照合する場合のSHA-256は `1147ce7c98c4e2e4bfb1eadaaf04cb4b899590d1693a6cbede63a32447be2074` です。

## DataLoaderの待ち時間を見る

対応記事: [PyTorch学習が遅いのはDataLoaderかGPUか？](https://yuyuyuroom.site/?p=476)

DataLoaderのworker数を変え、データの準備とGPUの計算を時系列で見ます。こちらはCUDAが必要です。最初のコマンドが `True` を返す環境で実行してください。記録ファイルが大きくなるので、300 MB以上の空きを用意します。

```bash
python -c 'import torch; print(torch.cuda.is_available())'
python -m pip install matplotlib numpy
python ai_lab/experiment_training_profiler.py
python ai_lab/plot_training_profiler.py
python -m unittest discover -s ai_lab -p test_training_profiler.py -v
```

`ai_lab/results/training_profiler/` に集計JSON、600ステップのCSV、2条件の実行記録（trace）と演算子表が作られます。図は `figures/pytorch-profiler-dataloader-wait.png` です。

profiler自身も処理を増やすため、通常の学習速度は記録を取らない別実行で測っています。trace上の時間と速度測定の値は分けて読んでください。

## データと生成ファイル

FashionMNISTの画像、量子化したモデル、trace、図は実行時に作られます。結果は主に `ai_lab/results/`、図は `figures/` に保存します。GPUメモリ実験の記事で使った測定値は `evidence/` にも置いています。

モデルやデータを別の用途に使う場合は、FashionMNISTと各ライブラリのライセンスを確認してください。
