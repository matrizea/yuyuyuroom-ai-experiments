#!/usr/bin/env python3
"""Compare handcrafted HOG + linear SVM with a learned CNN on FashionMNIST."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from skimage.feature import hog
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


HOG_KWARGS = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def extract_hog(images: np.ndarray) -> np.ndarray:
    return np.asarray([hog(image, **HOG_KWARGS) for image in images], dtype=np.float32)


def train_cnn(data_dir: Path) -> tuple[np.ndarray, dict]:
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(data_dir, train=True, download=False, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    torch.manual_seed(42)
    model = SmallCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    dummy = torch.zeros(256, 1, 28, 28, device="cuda"); labels = torch.zeros(256, dtype=torch.long, device="cuda")
    optimizer.zero_grad(set_to_none=True); loss_fn(model(dummy), labels).backward(); optimizer.step(); torch.cuda.synchronize()
    # Restore identical pre-warm-up initial weights for actual convergence.
    torch.manual_seed(42); model = SmallCNN().cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = []
    for epoch in range(1, 4):
        model.train(); seen = 0; loss_sum = 0.0
        torch.cuda.synchronize(); started = time.perf_counter()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step(); seen += labels.numel(); loss_sum += float(loss.detach()) * labels.size(0)
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        epochs.append({"epoch": epoch, "seconds": seconds, "loss": loss_sum / seen, "images_per_second": seen / seconds})
    model.eval(); predictions = []
    torch.cuda.synchronize(); infer_started = time.perf_counter()
    with torch.no_grad():
        for images, _ in test_loader:
            predictions.append(model(images.cuda(non_blocking=True)).argmax(dim=1).cpu().numpy())
    torch.cuda.synchronize(); infer_seconds = time.perf_counter() - infer_started
    return np.concatenate(predictions), {"epochs": epochs, "total_train_seconds": sum(e["seconds"] for e in epochs), "test_inference_seconds": infer_seconds}


def per_class_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    return matrix, np.diag(matrix) / matrix.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This comparison requires a CUDA GPU. Check your PyTorch installation first.")
    torch.backends.cuda.matmul.fp32_precision = "ieee"; torch.backends.cudnn.fp32_precision = "ieee"

    raw_train = datasets.FashionMNIST(args.data_dir, train=True, download=True)
    raw_test = datasets.FashionMNIST(args.data_dir, train=False, download=True)
    x_train = raw_train.data.numpy(); y_train = raw_train.targets.numpy()
    x_test = raw_test.data.numpy(); y_test = raw_test.targets.numpy()

    started = time.perf_counter(); hog_train = extract_hog(x_train); train_extract_seconds = time.perf_counter() - started
    started = time.perf_counter(); hog_test = extract_hog(x_test); test_extract_seconds = time.perf_counter() - started
    svm = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=42)
    started = time.perf_counter(); svm.fit(hog_train, y_train); svm_fit_seconds = time.perf_counter() - started
    started = time.perf_counter(); svm_predictions = svm.predict(hog_test); svm_infer_seconds = time.perf_counter() - started

    cnn_predictions, cnn_timing = train_cnn(args.data_dir)
    svm_matrix, svm_class = per_class_accuracy(y_test, svm_predictions)
    cnn_matrix, cnn_class = per_class_accuracy(y_test, cnn_predictions)
    svm_ok = svm_predictions == y_test; cnn_ok = cnn_predictions == y_test
    agreement = {
        "both_correct": int((svm_ok & cnn_ok).sum()),
        "svm_only_correct": int((svm_ok & ~cnn_ok).sum()),
        "cnn_only_correct": int((~svm_ok & cnn_ok).sum()),
        "both_wrong": int((~svm_ok & ~cnn_ok).sum()),
        "prediction_agreement": float((svm_predictions == cnn_predictions).mean()),
    }
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "protocol": {"train_samples": 60000, "test_samples": 10000, "hog": HOG_KWARGS, "svm": "LinearSVC C=1 dual=auto", "cnn": "SmallCNN 105866 params, Adam 1e-3, 3 epochs, batch 256"},
        "hog_feature_dimension": int(hog_train.shape[1]),
        "svm": {
            "test_accuracy": float(svm_ok.mean()), "train_hog_seconds": train_extract_seconds,
            "test_hog_seconds": test_extract_seconds, "fit_seconds": svm_fit_seconds,
            "predict_seconds": svm_infer_seconds, "support": "linear coefficients, not kernel support vectors",
            "per_class_accuracy": svm_class.tolist(), "confusion_matrix": svm_matrix.tolist(),
        },
        "cnn": {"test_accuracy": float(cnn_ok.mean()), **cnn_timing, "per_class_accuracy": cnn_class.tolist(), "confusion_matrix": cnn_matrix.tolist()},
        "agreement": agreement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(12, 6.75), dpi=150, facecolor="#07111f")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1, 1.35])
    axes = [fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[:,2]), fig.add_subplot(grid[1,:2])]
    for ax in axes:
        ax.set_facecolor("#0b172a"); ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values(): spine.set_visible(False)
    sample_index = int(np.flatnonzero(y_test == 6)[0])
    _, hog_image = hog(x_test[sample_index], visualize=True, **HOG_KWARGS)
    axes[0].imshow(x_test[sample_index], cmap="gray"); axes[0].set_title("Original image: Shirt", color="white"); axes[0].axis("off")
    axes[1].imshow(hog_image, cmap="magma"); axes[1].set_title(f"HOG ({hog_train.shape[1]} dimensions)", color="white"); axes[1].axis("off")
    values = [svm_ok.mean()*100, cnn_ok.mean()*100]
    bars = axes[2].bar(["HOG +\nLinear SVM", "CNN\n3 epoch"], values, color=["#fbbf24", "#38bdf8"])
    axes[2].set_ylim(80, 95); axes[2].set_ylabel("test accuracy [%]", color="#dbeafe"); axes[2].set_title("Same 60k/10k split", color="white")
    axes[2].grid(axis="y", alpha=.18, color="white")
    for bar, value in zip(bars, values, strict=True): axes[2].text(bar.get_x()+bar.get_width()/2, value+.25, f"{value:.2f}%", ha="center", color="white")
    x = np.arange(10); axes[3].plot(x, svm_class*100, "o-", color="#fbbf24", label="HOG+SVM"); axes[3].plot(x, cnn_class*100, "s-", color="#38bdf8", label="CNN")
    axes[3].set_xticks(x, CLASS_NAMES, rotation=25, ha="right"); axes[3].set_ylim(40,100); axes[3].set_ylabel("class accuracy [%]", color="#dbeafe"); axes[3].set_title("Accuracy by clothing class", color="white"); axes[3].grid(alpha=.18, color="white"); axes[3].legend(facecolor="#0b172a", labelcolor="white")
    fig.suptitle("FashionMNIST: HOG + SVM versus CNN", color="white", fontsize=18, y=.97)
    fig.text(.5,.02,f"Both correct {agreement['both_correct']:,} / SVM only {agreement['svm_only_correct']:,} / CNN only {agreement['cnn_only_correct']:,} / Both wrong {agreement['both_wrong']:,}",ha="center",color="#cbd5e1",fontsize=10)
    fig.subplots_adjust(left=.06,right=.97,bottom=.16,top=.87,wspace=.35,hspace=.38); args.figure.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.figure,facecolor=fig.get_facecolor()); plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
