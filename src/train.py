"""Entraîne un modèle et évalue sur le jeu de test.

Exemple :
    python -m src.train --model resnet18 --augment --epochs 25 --data-dir data/fer2013
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_loaders
from src.models import build_model
from src.utils import predict, save_curves, save_metrics, set_seed


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data/fer2013")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.model}_{'aug' if args.augment else 'noaug'}_seed{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl, classes = get_loaders(
        args.data_dir, args.img_size, args.batch_size, args.augment, seed=args.seed)
    model = build_model(args.model, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val, best_path = 0.0, out_dir / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, criterion, optimizer)
        vl, va = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        for k, v in zip(history, [tl, ta, vl, va]):
            history[k].append(v)
        if va > best_val:
            best_val = va; torch.save(model.state_dict(), best_path)
        print(f"[{run_name}] epoch {epoch:02d}/{args.epochs} "
              f"train_loss={tl:.3f} train_acc={ta:.3f} val_loss={vl:.3f} val_acc={va:.3f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = predict(model, test_dl, device)
    metrics = save_metrics(y_true, y_pred, classes, out_dir, {
        "run": run_name, "model": args.model, "augment": args.augment,
        "epochs": args.epochs, "best_val_accuracy": best_val,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_time_sec": round(time.time() - t0, 1)})
    save_curves(history, out_dir / "curves.png")
    print(f"TEST accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
