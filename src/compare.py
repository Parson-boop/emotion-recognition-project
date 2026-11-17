"""Regroupe les results/*/metrics.json dans results/summary.md.

Le fichier contient : un tableau par run, la moyenne et l'écart-type sur les graines
pour chaque configuration, et le F1 moyen par classe.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

rows = [json.loads(f.read_text()) for f in sorted(Path("results").glob("*/metrics.json"))]
SKIP = ("accuracy", "macro avg", "weighted avg")
classes = [k for k in rows[0]["per_class"] if k not in SKIP] if rows else []


def mean_std(values):
    m = st.mean(values)
    s = st.stdev(values) if len(values) > 1 else 0.0
    return m, s


lines = ["## Résultats par run", "",
         "| Run | Modèle | Augmentation | Params | Val acc | Test acc | Test macro-F1 | Temps (s) |",
         "|---|---|---|---|---|---|---|---|"]
for m in rows:
    lines.append(f"| {m['run']} | {m['model']} | {'oui' if m['augment'] else 'non'} | "
                 f"{m['n_params']:,} | {m['best_val_accuracy']:.3f} | {m['test_accuracy']:.3f} | "
                 f"{m['test_macro_f1']:.3f} | {m['train_time_sec']} |")

groups = defaultdict(list)
for m in rows:
    groups[(m["model"], m["augment"])].append(m)

lines += ["", "## Moyenne ± écart-type sur les graines", "",
          "| Modèle | Augmentation | Nb de graines | Test acc | Test macro-F1 |", "|---|---|---|---|---|"]
for (model, aug), ms in sorted(groups.items()):
    a, sa = mean_std([x["test_accuracy"] for x in ms])
    f, sf = mean_std([x["test_macro_f1"] for x in ms])
    lines.append(f"| {model} | {'oui' if aug else 'non'} | {len(ms)} | {a:.3f} ± {sa:.3f} | {f:.3f} ± {sf:.3f} |")

lines += ["", "## F1 moyen par classe (test)", "",
          "| Modèle | Augmentation | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
for (model, aug), ms in sorted(groups.items()):
    cells = [f"{st.mean([x['per_class'][c]['f1-score'] for x in ms]):.2f}" for c in classes]
    lines.append(f"| {model} | {'oui' if aug else 'non'} | " + " | ".join(cells) + " |")

Path("results/summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
