# Detection Anomalie polSAR (CVNN)

Ce projet implémente un pipeline complet pour la détection d'anomalies non-supervisée dans des images radar à synthèse d'ouverture polarimétrique (PolSAR). Il inclut des réseaux de neurones à valeurs complexes (CVNN), ainsi que des méthodes de références (baselines) classiques (Statistiques physiques) et de Machine Learning (SVDD).

## Prérequis

- Python 3.9+ (3.12 recommandé)
- PyTorch
- [Poetry](https://python-poetry.org/) (recommandé pour la gestion des dépendances) ou un environnement virtuel classique (`venv`).

## Installation

1. Cloner le dépôt et se rendre dans le dossier du projet.
2. Créer et activer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   # ou via poetry : poetry install
   ```

## Structure du projet

- `cvnn/` : Cœur de la bibliothèque pour les réseaux de neurones complexes, le chargement des images (ALOS/PolSF) et la gestion des dataloaders.
- `configs/` : Fichiers YAML de configuration (découpage azimutal, patchs, transformations, etc.).
- `src/methode_Classique/` : Pipeline d'extraction de caractéristiques physiques (Covariance globale, Corrélations, Intensités). Comprend l'étalonnage de la normalité (`H0.py`), l'injection d'anomalies physiques comme le CrossTalk(`H1.py`) et l'évaluation (`plot_metrics.py`).
- `src/methode_MachineLearning/` : Implémentation d'algorithmes de détection d'anomalies traditionnels comme le SVDD (Support Vector Data Description).

## Exécution du pipeline d'anomalies (Classique & ML)

Le workflow de validation sépare rigoureusement l'apprentissage de la normalité (Zone 1 saine) et l'évaluation sur des données de test (Zone 2.1 saine et Zone 2.2 avec injection de défauts physiques).

```bash
# 1. Étalonnage du comportement normal (Zone 1)
cd src/methode_Classique/
python3 H0.py

# 2. Extraction des tests et injection d'anomalies (Zone 2.1 et 2.2)
python3 H1.py

# 3. Évaluation des méthodes statistiques (Profondeur de Stahel-Donoho, Cohérence, Entropie)
python3 plot_metrics.py

# 4. Entraînement et évaluation avec SVDD (Machine Learning)
cd ../methode_MachineLearning/
python3 train_svdd.py
```