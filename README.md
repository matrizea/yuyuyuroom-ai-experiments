# yuyuyuroom.site AI experiments

[English reproduction guide](README_EN.md)

「ぬるま湯ハードウェア」に掲載したAI実験の再現コードです。WordPressへZIPを直接置かず、このリポジトリから取得できるようにしています。実験結果は使用するPC、ソフトウェアの版、GPUの状態で変わります。

## 取得

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git ~/ai-experiments
cd ~/ai-experiments
```

Gitが未導入なら、GitHubの「Code」→「Download ZIP」から取得しても構いません。展開後のフォルダーを `~/ai-experiments` として扱います。データセットは同梱せず、初回実行時に `torchvision` が取得します。

## ONNX Runtime の静的 INT8 量子化

対応記事: [ONNXのCNNを実際にINT8量子化したら速くなる？](https://yuyuyuroom.site/?p=471)

Python、PyTorch、torchvisionを先に用意してください。PyTorchの組み合わせは[公式インストール案内](https://pytorch.org/get-started/locally/)で選びます。実験時は Python 3.14.4、PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130、ONNX 1.22.0、ONNX Runtime 1.29.0、NumPy 2.5.2でした。推論にはCPUを使い、GPUは不要です。

```bash
python -m pip install onnx==1.22.0 onnxruntime==1.29.0 numpy==2.5.2 matplotlib
python ai_lab/experiment_onnx_int8_static.py --preprocess
python ai_lab/plot_onnx_int8_static.py
python -m unittest discover -s ai_lab -p test_onnx_int8_static.py -v
```

元のFP32モデルは `ai_lab/models/fashion_cnn_dynamic.onnx` です。SHA-256 は `1147ce7c98c4e2e4bfb1eadaaf04cb4b899590d1693a6cbede63a32447be2074`。実行後の未丸め数値は `ai_lab/results/onnx_int8_static/preprocessed/result.json`、各推論呼び出しは同ディレクトリの `timings_raw.csv`、図は `figures/onnx-int8-static-cpu-comparison.png` に出ます。

## PyTorch profiler による DataLoader 待ちの確認

対応記事: [PyTorch学習が遅いのはDataLoaderかGPUか？](https://yuyuyuroom.site/?p=476)

CUDAが動くPyTorchとtorchvisionが必要です。実験時はWindows 11のWSL 2、RTX 5070 Ti 16 GB、PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130でした。詳細traceは100 MBを超えることがあるため、300 MB以上の空きを確保してください。

```bash
python -c 'import torch; print(torch.cuda.is_available())'
python -m pip install matplotlib numpy
python ai_lab/experiment_training_profiler.py
python ai_lab/plot_training_profiler.py
python -m unittest discover -s ai_lab -p test_training_profiler.py -v
```

最初の確認が `True` でない場合、CUDA環境を先に直してください。`ai_lab/results/training_profiler/` に集計JSON、600 stepのCSV、2つのtraceと演算子表が作られます。図は `figures/pytorch-profiler-dataloader-wait.png` に出ます。速度測定はprofilerなしの別実行であり、profiler traceの時間と混ぜません。

## 配布範囲

コード、元のONNXモデル、テストだけをGitで管理します。FashionMNISTの画像、量子化後の生成モデル、測定時のtraceと結果、生成図はスクリプトを実行して得ます。サイトの記事には実測の図と条件・結果を載せています。モデルはこの実験の再現用であり、製品品質を保証するものではありません。再利用時はFashionMNIST・PyTorch・ONNX Runtime等の利用条件も確認してください。
