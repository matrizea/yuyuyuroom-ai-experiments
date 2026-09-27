# Reproducing the yuyuyuroom.site AI experiments

[日本語](README.md) · [English articles](https://yuyuyuroom.site/category/en/)

This repository contains small, runnable experiments used in the site's articles. Timings depend on your PC, software versions and background load. The articles retain the original measurements; running a script again is not expected to reproduce the exact elapsed time.

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git ~/ai-experiments
cd ~/ai-experiments
```

Use a Python environment with PyTorch and compatible torchvision. Select the installation from [PyTorch's official instructions](https://pytorch.org/get-started/locally/). The measured environment used Python 3.14.4, PyTorch 2.13.0+cu130 and torchvision 0.28.0+cu130 on WSL 2. Datasets are not included: torchvision downloads FashionMNIST on first use (about 31 MB compressed).

## Verify CUDA in the selected Python environment

```bash
python ai_lab/verify_cuda.py
```

Success means `passed: true` and a 32×32 matrix whose entries are 32. This is a small execution check, not a benchmark or a general numerical validation. If CUDA is unavailable the script exits nonzero with environment information.

## FashionMNIST CNN training — CPU versus CUDA

```bash
python -m pip install numpy matplotlib
python -m unittest discover -s ai_lab -p test_fashion_cnn.py -v
python ai_lab/experiment_fashion_cnn.py \
  --data-dir ai_lab/data \
  --output ai_lab/results/fashion_cnn_cpu_gpu.json \
  --figure figures/fashion-cnn-cpu-gpu.png
```

Allow about 100 MB for data and outputs, plus the Python environment. The script runs three CPU epochs followed by three CUDA epochs if CUDA is available. If it is not, the script produces CPU-only results; inspect `results.cuda` before interpreting a comparison. The original plotting code uses Japanese labels and optionally a Windows Japanese font. Numbers are also available in JSON.

The epoch timer includes loading, transfers, training and per-batch metric reads; test evaluation is outside it. GPU peak allocation is read after evaluation and is not training-only memory. Timings and CUDA accuracy can vary on reruns. The model test verifies shape, parameter count and finite gradients without downloading data.

## ONNX Runtime static INT8 quantization — CPU



No GPU is needed. The FP32 model is included at `ai_lab/models/fashion_cnn_dynamic.onnx` (SHA-256 `1147ce7c98c4e2e4bfb1eadaaf04cb4b899590d1693a6cbede63a32447be2074`).

```bash
python -m pip install onnx==1.22.0 onnxruntime==1.29.0 numpy==2.5.2 matplotlib
python ai_lab/experiment_onnx_int8_static.py --preprocess
python ai_lab/plot_onnx_int8_static.py
python -m unittest discover -s ai_lab -p test_onnx_int8_static.py -v
```

Results: `ai_lab/results/onnx_int8_static/preprocessed/result.json` and `timings_raw.csv`. Chart: `figures/onnx-int8-static-cpu-comparison.png`. The five artifact checks require the experiment to have run first. Allow roughly 100 MB for dataset and outputs, separate from packages.

## DataLoader waiting time — CUDA


```bash
python -c 'import torch; print(torch.cuda.is_available())'
python -m pip install matplotlib numpy
python ai_lab/experiment_training_profiler.py
python ai_lab/plot_training_profiler.py
python -m unittest discover -s ai_lab -p test_training_profiler.py -v
```

The CUDA check must print `True`. Results and two traces are under `ai_lab/results/training_profiler/`; the chart is `figures/pytorch-profiler-dataloader-wait.png`. Allow at least 300 MB beyond the Python environment. The three artifact checks require the result files. Throughput is measured separately with profiling disabled; do not use trace durations as ordinary training speed.

## HOG + linear SVM versus a CNN — CPU and CUDA

The SVM runs on the CPU and the CNN requires CUDA. The script preserves the original experiment's model, split, HOG settings, seed, warm-up/reset procedure and three-epoch training budget. Packaging changes make the model class self-contained, add a CUDA check and dataset download, and use English plot labels.

```bash
python -m pip install numpy matplotlib scikit-learn scikit-image
python -m unittest discover -s ai_lab -p test_hog_svm_vs_cnn.py -v
python ai_lab/experiment_hog_svm_vs_cnn.py \
  --data-dir ai_lab/data \
  --output ai_lab/results/hog_svm_vs_cnn.json \
  --figure figures/hog-svm-vs-cnn.png
```

The three unit tests are CPU-only. The full comparison downloads FashionMNIST if needed, extracts 1,296 HOG features per image, fits LinearSVC and trains the CNN. The training-feature array alone uses about 297 MiB; allow additional memory for intermediate arrays and fitting.

JSON contains separate extraction, fitting and prediction times, both confusion matrices, class accuracies and correctness overlap counts. Seed 42 fixes the requested initial state, but deterministic CUDA algorithms are not forced: a packaging verification run gave CNN accuracy 86.89%, compared with 87.00% in the original article. SVM accuracy was 88.85% in both. Do not replace individual numbers in the original comparison with values from a different run.

## Scope of the files

Large traces, datasets, generated models, figures and result directories are excluded from Git. Generate them locally with the commands above. The bundled model is for reproducing the experiment, not a production-quality classifier. Check the licenses of the datasets and libraries for your intended use.
