# PyTorch CUDA memory lifetime

Small reference-lifetime experiment. No dataset/model downloads, no training,
no deliberate out-of-memory allocation. Tested in WSL2 with Python 3.14.4,
PyTorch 2.13.0+cu130, RTX 5070 Ti and the native caching allocator.

Use an existing CUDA-enabled PyTorch environment. Check it first:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python ai_lab/experiment_cuda_memory_lifetime.py --output ai_lab/results/cuda_memory.json
```

The experiment needs at least 1 GiB free device memory before starting. Its
largest explicit tensor is 256 MiB. It checks three repetitions with assertions
and stores raw integer byte measurements, environment, and the saved object's
device/gradient information in JSON. Expect a few seconds after Python starts;
allow one minute. It does not change allocator configuration, terminate other
processes, or install anything. CUDA unavailable or a non-native allocator
produces an explanatory error instead of pretending to measure CUDA.

Optional figure (requires matplotlib):

```bash
python ai_lab/plot_cuda_memory_lifetime.py ai_lab/results/cuda_memory.json figures/cuda-memory-lifetime.png
```

`loss_tensor` intentionally saves scalar losses without backward, keeping the
graph alive. `loss_item` keeps Python floats instead. `output_detach_gpu` keeps
the output data on CUDA despite removing gradient history; the CPU variants
copy the data to host RAM. End-of-iteration allocated memory is measured after
temporary names are deleted and CUDA synchronization completes. These are not
peak-memory or timing measurements. CPU copies cost host memory and transfer
time; this experiment does not benchmark that cost.

Reference evidence from the article run: `evidence/cuda_memory_lifetime.json`.
Exact cache reservation can differ with allocator/version and existing process
state. The assertions assume a fresh process and native allocator. In notebooks,
outputs, references, and exception tracebacks can keep objects alive.
