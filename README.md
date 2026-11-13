# Reconnaissance des émotions faciales : CNN from scratch vs ResNet18 pré-entraîné (FER2013)

Projet final — *Réseaux de neurones : vision par ordinateur* (date limite : 19 novembre 2026).

Auteur(s) : **[Prénom Nom, Prénom Nom]**

## 1. Objectifs et méthode

**Question de recherche.** Sur FER2013 (images 48×48, 7 émotions), un modèle pré-entraîné sur ImageNet
apporte-t-il un gain par rapport à un CNN entraîné from scratch ? L'augmentation de données change-t-elle
cette conclusion ? Quelles émotions sont les plus difficiles à reconnaître, et pourquoi ?

**Données.** FER2013 (Kaggle) : 7 classes (angry, disgust, fear, happy, neutral, sad, surprise).
Le dossier `train` est séparé en 90 % entraînement / 10 % validation (graine fixe) ; le dossier `test`
n'est utilisé qu'à la fin.

**Modèles comparés.**
- `cnn` : CNN à 4 blocs (Conv-BN-ReLU ×2, MaxPool, Dropout), entraîné from scratch.
- `resnet18` : ResNet18 pré-entraîné ImageNet, dernière couche remplacée, fine-tuning complet.

**Facteur étudié.** Augmentation de données (flip horizontal, rotation ±10°, recadrage aléatoire) : oui / non.

**Protocole.** 4 configurations × 3 graines (42, 43, 44), 25 époques, AdamW (lr 1e-3, weight decay 1e-4),
cosine annealing, images redimensionnées à 64×64 et converties en 3 canaux. Le meilleur checkpoint
(accuracy de validation) est évalué sur le test. Métriques : accuracy, macro-F1, F1 par classe,
matrice de confusion.

## 2. Installation et exécution

```bash
git clone <URL_DU_DEPOT> && cd emotion-recognition-project
pip install -r requirements.txt

# Données (nécessite un compte Kaggle et le fichier ~/.kaggle/kaggle.json)
kaggle datasets download -d msambare/fer2013 -p data --unzip
# Attendu : data/train/<classe>/*.jpg et data/test/<classe>/*.jpg
# (si les dossiers sont dans data/fer2013/, passer --data-dir data/fer2013)

# Un entraînement
python -m src.train --model resnet18 --augment --epochs 25 --data-dir data

# Toutes les expériences + tableau récapitulatif (results/summary.md)
./run_all.sh data
```

GPU recommandé (Google Colab / Kaggle Notebooks : activer le GPU). Un run dure quelques minutes sur GPU.

## 3. Résultats

> **À compléter avec vos propres résultats** : coller le contenu de `results/summary.md`
> (moyenne ± écart-type sur les 3 graines) et ajouter les figures de `results/<run>/`.

| Modèle | Augmentation | Test acc (moy ± std) | Macro-F1 (moy ± std) |
|---|---|---|---|
| CNN | non | … | … |
| CNN | oui | … | … |
| ResNet18 | non | … | … |
| ResNet18 | oui | … | … |

Figures : `results/<run>/curves.png` (courbes loss/accuracy), `results/<run>/confusion_matrix.png`.

### Analyse (à rédiger d'après vos résultats)

À discuter, en vous appuyant sur les chiffres :
- Le pré-entraînement améliore-t-il l'accuracy et le macro-F1 ? À quel coût (paramètres, temps) ?
- Effet de l'augmentation : réduit-elle le surapprentissage (écart train/val dans `curves.png`) ?
- Classes difficiles (souvent *disgust*, très peu représentée, et *fear*/*sad* confondues) : que montre la matrice de confusion ?
- Limites : bruit d'annotation connu de FER2013, déséquilibre des classes, une seule résolution testée.
- Pistes d'amélioration : pondération des classes, autre résolution, autres architectures.

## 4. Reproduire les résultats

1. Installer les dépendances et télécharger FER2013 (section 2).
2. Lancer `./run_all.sh data` (graines 42, 43, 44 fixées ; de petites variations GPU restent possibles).
3. Les métriques, courbes et matrices sont écrites dans `results/`, et `python -m src.compare` génère `results/summary.md`.

## Structure du dépôt

```
src/data.py      chargement, split, augmentations
src/models.py    CNN et ResNet18
src/train.py     entraînement + évaluation
src/utils.py     métriques et figures
src/compare.py   tableau récapitulatif
run_all.sh       toutes les expériences
results/         sorties des runs
```

## Références

- Goodfellow et al., *Challenges in Representation Learning: FER2013* (Kaggle, 2013).
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Modi et al., reconnaissance des émotions faciales avec CNN (cité dans les consignes du cours).
