# Reconnaissance des émotions sur FER2013 : CNN simple ou ResNet18 pré-entraîné ?

Projet final du cours *Réseaux de neurones : vision par ordinateur*.

Auteur(s) : José Fanantenana Parson Randriamanantsoa 

## 1. Objectifs et méthode

On voulait savoir si un modèle pré-entraîné sur ImageNet (ResNet18) reconnaît mieux les émotions sur un visage qu'un petit CNN entraîné de zéro, et si l'augmentation de données change la réponse. On a aussi regardé quelles émotions posent problème.

**Données.** FER2013 (version Kaggle de msambare) : des images de visages en niveaux de gris, 48×48 pixels, réparties en 7 classes : angry, disgust, fear, happy, neutral, sad, surprise. Le dossier `train` est coupé en 90 % pour l'entraînement et 10 % pour la validation (avec une graine fixe). Le dossier `test` ne sert qu'à la fin, pour les résultats.

**Modèles.**
- `cnn` : un CNN de 4 blocs (convolutions, batch norm, pooling, dropout), environ 1,2 million de paramètres, entraîné de zéro.
- `resnet18` : ResNet18 pré-entraîné sur ImageNet, dont on remplace la dernière couche, puis on entraîne tout le réseau (environ 11 millions de paramètres).

**Ce qu'on fait varier.** Avec ou sans augmentation de données (flip horizontal, rotation de ±10°, recadrage aléatoire). Cela donne 4 configurations, chacune lancée avec 3 graines (42, 43, 44), donc 12 entraînements.

**Réglages communs.** 25 époques, AdamW (lr 1e-3, weight decay 1e-4), cosine annealing, batch de 128, images agrandies à 64×64 et mises en 3 canaux. On garde le checkpoint qui a la meilleure accuracy de validation, puis on l'évalue une seule fois sur le test. Mesures : accuracy, macro-F1 (moyenne du F1 des 7 classes, utile car les classes sont déséquilibrées), F1 par classe et matrice de confusion.

## 2. Installation et exécution

```bash
git clone https://github.com/Parson-boop/emotion-recognition-project.git
cd emotion-recognition-project
pip install -r requirements.txt
```

**Données.** Avec la CLI Kaggle (il faut un compte et le fichier `~/.kaggle/kaggle.json`) :

```bash
kaggle datasets download -d msambare/fer2013 -p data --unzip
```

On obtient `data/train/<classe>/` et `data/test/<classe>/`. Sur un notebook Kaggle, on ajoute plutôt le dataset `msambare/fer2013` en entrée ; le chemin est alors `/kaggle/input/datasets/msambare/fer2013`.

**Un entraînement :**

```bash
python -m src.train --model resnet18 --augment --epochs 25 --seed 42 --data-dir data
```

Options : `--model {cnn,resnet18}`, `--augment` (pour activer l'augmentation), `--epochs`, `--lr`, `--batch-size`, `--img-size`, `--seed`, `--data-dir`.

Un GPU est nécessaire en pratique. Sur un GPU de Kaggle, un entraînement a pris entre 10 et 14 minutes (un des runs a pris 30 minutes, sans doute à cause de la charge des serveurs).

## 3. Résultats

Moyenne ± écart-type sur les 3 graines, évalué sur le jeu de test :

| Modèle | Augmentation | Accuracy | Macro-F1 |
|---|---|---|---|
| CNN | non | 0,650 ± 0,007 | 0,584 ± 0,025 |
| CNN | oui | 0,630 ± 0,004 | 0,516 ± 0,005 |
| ResNet18 | non | 0,640 ± 0,005 | 0,625 ± 0,005 |
| ResNet18 | oui | **0,668 ± 0,003** | **0,656 ± 0,006** |

F1 moyen par classe :

| Modèle | Augmentation | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|
| CNN | non | 0,57 | 0,31 | 0,41 | 0,85 | 0,63 | 0,54 | 0,78 |
| CNN | oui | 0,56 | 0,01 | 0,32 | 0,85 | 0,61 | 0,50 | 0,76 |
| ResNet18 | non | 0,55 | 0,62 | 0,50 | 0,83 | 0,58 | 0,51 | 0,78 |
| ResNet18 | oui | 0,59 | 0,68 | 0,51 | 0,86 | 0,63 | 0,51 | 0,79 |

Les résultats de chaque run (métriques, courbes d'apprentissage, matrice de confusion) sont dans `results/<nom_du_run>/`, et le tableau complet est dans `results/summary.md`.

### Analyse

**Le meilleur modèle est ResNet18 avec augmentation** (0,668 d'accuracy, 0,656 de macro-F1). Il est devant les trois autres configurations à chaque graine.

**L'augmentation n'a pas le même effet selon le modèle.** Elle aide ResNet18 (environ +3 points d'accuracy et +3 de macro-F1). Elle pénalise le CNN : −2 points d'accuracy et surtout −7 points de macro-F1. Le sens est le même sur les 3 graines. On n'a pas testé pourquoi. Notre hypothèse : un petit CNN entraîné de zéro n'arrive pas à apprendre les classes rares quand les images sont déformées, mais on n'a rien fait pour le vérifier.

**Le pré-entraînement aide surtout sur les classes rares.** Sans augmentation, le CNN est même un peu meilleur en accuracy (0,650 contre 0,640), mais ResNet18 a un meilleur macro-F1 (0,625 contre 0,584). La différence vient surtout de *disgust* : le F1 du CNN est de 0,31 sans augmentation et presque nul avec (0,01), alors qu'il est de 0,62 à 0,68 pour ResNet18. Dans le CNN avec augmentation, *disgust* explique à lui seul environ les deux tiers de la baisse de macro-F1.

**Ce que montrent les matrices de confusion** (graine 42, pour `resnet18_aug` et `cnn_noaug`) :
- *happy* est l'émotion la mieux reconnue dans les deux modèles (0,87), et *surprise* est aussi bien reconnue.
- *fear* est assez mal reconnue et souvent confondue avec *sad*, surtout dans le CNN.
- *sad* et *neutral* sont confondues dans les deux modèles.
- Pour *disgust*, l'écart est grand : 0,59 de bonnes réponses pour ResNet18 contre 0,18 pour le CNN. Dans le CNN, *disgust* est surtout pris pour *angry* (0,60).

**Ce que montrent les courbes** (graine 42) : dans tous les cas, la loss d'entraînement baisse et l'accuracy monte, mais la validation stagne alors que l'entraînement continue de s'améliorer. Le surapprentissage est plus marqué pour ResNet18 avec augmentation (par exemple, vers la fin d'un run, 0,91 d'accuracy en entraînement contre 0,66 en validation, et une loss de validation qui remonte). Le CNN est un peu plus stable, mais il surapprend aussi. Comme on garde le meilleur checkpoint de validation, le test n'est pas faussé par cela.

### Limites

- 3 graines, c'est peu : les écarts d'environ 1 point (par exemple CNN contre ResNet18 sans augmentation en accuracy) sont à prendre avec prudence.
- *disgust* a très peu d'images, donc son F1 varie beaucoup d'une graine à l'autre (de 0,18 à 0,47 pour le CNN sans augmentation).
- Il n'y a eu aucun réglage des hyperparamètres : tous les modèles utilisent les mêmes valeurs. Une seule taille d'image (64×64) a été testée.
- Les annotations de FER2013 sont connues pour contenir des erreurs, ce qui limite l'accuracy possible.
- Les temps d'entraînement ne sont pas comparables précisément (la charge des serveurs a varié).

Pistes si on avait plus de temps : pondérer la loss par classe pour aider *disgust* et *fear*, tester des augmentations plus légères pour le CNN, essayer d'autres tailles d'image.

## 4. Reproduire les résultats

1. Installer les dépendances et récupérer les données (section 2).
2. Lancer toutes les expériences (12 entraînements, environ 2 h 30 sur un GPU Kaggle) :

```bash
./run_all.sh data
```

3. Générer les tableaux (par run, moyenne ± écart-type, F1 par classe) :

```bash
python -m src.compare
```

Les graines sont fixées (42, 43, 44), mais de petites différences restent possibles d'un GPU à l'autre.

## Structure du dépôt

```
src/data.py      chargement des données, découpage, augmentations
src/models.py    CNN et ResNet18
src/train.py     entraînement et évaluation
src/utils.py     métriques et figures
src/compare.py   tableaux récapitulatifs
run_all.sh       lance toutes les expériences
results/         résultats de chaque run
```

## Références

- Goodfellow et al., *Challenges in Representation Learning: A report on three machine learning contests*, Neural Networks, 2015 (le jeu de données FER2013 vient de ce challenge).
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Jeu de données : FER-2013 sur Kaggle (msambare/fer2013).
