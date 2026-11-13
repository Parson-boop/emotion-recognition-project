"""Agrège tous les results/*/metrics.json dans results/summary.md."""
import json
from pathlib import Path

rows = [json.loads(f.read_text()) for f in sorted(Path("results").glob("*/metrics.json"))]

lines = ["| Run | Modèle | Augmentation | Params | Val acc | Test acc | Test macro-F1 | Temps (s) |",
         "|---|---|---|---|---|---|---|---|"]
for m in rows:
    lines.append(f"| {m['run']} | {m['model']} | {'oui' if m['augment'] else 'non'} | "
                 f"{m['n_params']:,} | {m['best_val_accuracy']:.3f} | {m['test_accuracy']:.3f} | "
                 f"{m['test_macro_f1']:.3f} | {m['train_time_sec']} |")

lines += ["", "F1 par classe (test) :", ""]
if rows:
    classes = [k for k in rows[0]["per_class"] if k not in ("accuracy", "macro avg", "weighted avg")]
    lines += ["| Run | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for m in rows:
        lines.append(f"| {m['run']} | " + " | ".join(f"{m['per_class'][c]['f1-score']:.2f}" for c in classes) + " |")

Path("results/summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
