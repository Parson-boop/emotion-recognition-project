import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ps.append(out.argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def save_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("époque"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("époque"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_confusion(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("prédit"); ax.set_ylabel("réel"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_metrics(y_true, y_pred, classes, out_dir: Path, extra: dict):
    metrics = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, target_names=classes,
                                           output_dict=True, zero_division=0),
        **extra,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    return metrics
