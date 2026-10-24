# yuyuyuroom.site experiment code

[日本語](README.md) · [English articles](https://yuyuyuroom.site/category/en/)

Code used in the [yuyuyuroom.site](https://yuyuyuroom.site/) articles: PyTorch training, ONNX quantization, GPU memory checks, and a HOG/SVM comparison.

## Setup

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git ~/ai-experiments
cd ~/ai-experiments
```

Install PyTorch and compatible torchvision using [PyTorch's instructions](https://pytorch.org/get-started/locally/), then run the commands below from the repository directory. The articles used Python 3.14.4, PyTorch 2.13.0+cu130 and torchvision 0.28.0+cu130 on WSL 2. FashionMNIST is downloaded on first use (about 31 MB compressed).

For the `empty_cache()`, `del` and `detach()` experiment, see [CUDA_MEMORY.md](CUDA_MEMORY.md). It needs no dataset or model download.

## Check CUDA

```bash
python ai_lab/verify_cuda.py
```

Look for `passed: true`: the script multiplies two 32×32 matrices of ones on the GPU and checks that every result is 32. If CUDA is unavailable, it prints environment details and exits with an error. This check does not measure speed.

## FashionMNIST CNN training

```bash
python -m pip install numpy matplotlib
python -m unittest discover -s ai_lab -p test_fashion_cnn.py -v
python ai_lab/experiment_fashion_cnn.py \
  --data-dir ai_lab/data \
  --output ai_lab/results/fashion_cnn_cpu_gpu.json \
  --figure figures/fashion-cnn-cpu-gpu.png
```

The script trains for three epochs on the CPU, then three on CUDA. Without CUDA it runs only the CPU part; check `results.cuda` in the JSON. Allow about 100 MB for data and outputs, in addition to the Python environment. The chart has Japanese labels; the JSON contains the same numbers.

Epoch time includes data loading, transfers, training and per-batch metric reads, but not test evaluation. Peak GPU memory covers both training and evaluation. The unit test checks the model's shape, parameter count and gradients without downloading data.

## ONNX Runtime INT8 quantization

Quantize the bundled CNN and compare accuracy, file size and CPU inference speed. No GPU is needed. The FP32 model is `ai_lab/models/fashion_cnn_dynamic.onnx` (SHA-256 `1147ce7c98c4e2e4bfb1eadaaf04cb4b899590d1693a6cbede63a32447be2074`).

```bash
python -m pip install onnx==1.22.0 onnxruntime==1.29.0 numpy==2.5.2 matplotlib
python ai_lab/experiment_onnx_int8_static.py --preprocess
python ai_lab/plot_onnx_int8_static.py
python -m unittest discover -s ai_lab -p test_onnx_int8_static.py -v
```

Results are in `ai_lab/results/onnx_int8_static/preprocessed/result.json` and `timings_raw.csv`; the chart is `figures/onnx-int8-static-cpu-comparison.png`. Run the experiment before the tests, which read these files. Allow roughly 100 MB for data and outputs, in addition to packages.

## DataLoader waiting time

Compare worker counts and inspect where data preparation leaves the GPU waiting. This experiment requires CUDA and at least 300 MB of disk space for outputs.
```bash
python -c 'import torch; print(torch.cuda.is_available())'
python -m pip install matplotlib numpy
python ai_lab/experiment_training_profiler.py
python ai_lab/plot_training_profiler.py
python -m unittest discover -s ai_lab -p test_training_profiler.py -v
```

The first command must print `True`. Results and two traces go in `ai_lab/results/training_profiler/`; the chart is `figures/pytorch-profiler-dataloader-wait.png`. Run the experiment before the tests. Since profiling adds overhead, training speed is measured in a separate run with profiling disabled.

## HOG + linear SVM versus a CNN

The SVM runs on the CPU; the CNN requires CUDA and trains for three epochs. The chart uses English labels.

```bash
python -m pip install numpy matplotlib scikit-learn scikit-image
python -m unittest discover -s ai_lab -p test_hog_svm_vs_cnn.py -v
python ai_lab/experiment_hog_svm_vs_cnn.py \
  --data-dir ai_lab/data \
  --output ai_lab/results/hog_svm_vs_cnn.json \
  --figure figures/hog-svm-vs-cnn.png
```

The three unit tests need only a CPU. The full experiment downloads FashionMNIST if needed, extracts 1,296 HOG features per image, fits LinearSVC and trains the CNN. The training-feature array alone takes about 297 MiB of RAM; fitting and intermediate arrays need more.

The JSON includes feature-extraction, fitting and prediction times, confusion matrices, per-class accuracy and which images each method got right. Seed 42 is set, but CUDA operations are not forced to be deterministic. One rerun gave CNN accuracy of 86.89% versus 87.00% in the article; SVM accuracy was 88.85% in both. Keep results from each run together when comparing them.

## Data and outputs

The scripts create datasets, quantized models, traces and figures locally. Results usually go in `ai_lab/results/` and charts in `figures/`. The GPU-memory measurements used in the article are also included under `evidence/`.

Check the dataset and library licenses before reusing models or data elsewhere.
