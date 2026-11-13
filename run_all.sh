#!/usr/bin/env bash
# Reproduit les expériences : 2 modèles x (avec/sans augmentation) x 3 graines.
set -e
DATA=${1:-data/fer2013}
for SEED in 42 43 44; do
  for MODEL in cnn resnet18; do
    python -m src.train --model $MODEL --epochs 25 --seed $SEED --data-dir $DATA
    python -m src.train --model $MODEL --augment --epochs 25 --seed $SEED --data-dir $DATA
  done
done
python -m src.compare
